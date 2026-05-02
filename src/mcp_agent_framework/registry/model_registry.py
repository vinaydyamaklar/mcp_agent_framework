"""
Model Registry — register any LLM model once, call it anywhere by name.

Design principles:
  - Register at startup, query at runtime — O(1) lookup
  - Provider-agnostic — Anthropic, OpenAI, Gemini, Grok, Ollama all treated identically
  - auto_execute=False by default — tool calls are RETURNED to you, not silently executed
  - When auto_execute=True, you supply the tool_executor — the registry never touches MCP directly

Usage:

    # --- Register once at app startup ---
    registry = ModelRegistry()
    registry.register("fast",    AnthropicClient("claude-haiku-4-5"),   tags=["cheap", "fast"])
    registry.register("smart",   AnthropicClient("claude-sonnet-4-6"),  tags=["balanced"])
    registry.register("best",    AnthropicClient("claude-opus-4-6"),    tags=["powerful"])
    registry.register("gpt4o",   OpenAIClient("gpt-4o"),                tags=["openai"])
    registry.register("grok",    OpenAIClient("grok-3", base_url="https://api.x.ai/v1"), tags=["grok"])
    registry.register("flash",   GeminiClient("gemini-2.5-flash"),      tags=["cheap", "fast"])
    registry.register("local",   OpenAIClient("llama3", base_url="http://localhost:11434/v1"))

    # --- Single call, auto_execute=False (default) ---
    # Returns raw ModelResponse — YOU decide what to do with tool_calls
    response = await registry.complete("smart", messages, tools=tools)

    if response.has_tool_calls:
        for tc in response.tool_calls:
            result = await my_mcp.call_tool(tc.name, tc.arguments)
            messages.append(tc.to_tool_result_message(result))
        messages.append(response.to_message())  # append assistant turn first

    # --- Full loop, auto_execute=True ---
    # Runs complete tool-call loop — you supply the executor, registry runs the loop
    async def execute(name: str, args: dict) -> str:
        return await mcp.call_tool(name, args)

    response = await registry.complete(
        "smart", messages, tools=tools,
        auto_execute=True,
        tool_executor=execute,
    )
    print(response.text)   # final answer after all tool calls resolved

    # --- Query by tag ---
    cheap_models = registry.find_by_tag("cheap")   # ["fast", "flash"]
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.clients.schema_utils import schema_from
from mcp_agent_framework.types import MCPTool, Message, StopReason, ToolCall

logger = logging.getLogger(__name__)

# Callable the registry uses to execute a single tool call when auto_execute=True.
# Signature: (tool_name, arguments) -> result_string
ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]


# ── Errors ─────────────────────────────────────────────────────────────────────

class ModelNotFoundError(KeyError):
    """Raised when get() is called with an unregistered model name."""


class ModelRegistryError(RuntimeError):
    """Raised for invalid registry operations (e.g. missing tool_executor)."""


class MaxIterationsError(RuntimeError):
    """Raised when auto_execute loop hits max_iterations without finishing."""


# ── Response types ──────────────────────────────────────────────────────────────

@dataclass
class UsageInfo:
    input_tokens:  int | None = None
    output_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass
class ModelResponse:
    """
    Normalised response from any model, returned by registry.complete().

    When auto_execute=False (default):
        - tool_calls may be populated — the caller must handle them
        - content may be None if the model only returned tool calls

    When auto_execute=True:
        - tool_calls is always None (all calls were resolved internally)
        - content holds the final text answer
    """
    content:    str | None
    tool_calls: list[ToolCall] | None
    stop_reason: StopReason
    model_name:  str
    usage:       UsageInfo | None = None

    # ── Convenience ─────────────────────────────────────────────────────────

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def text(self) -> str:
        """Safe access to content — returns empty string if None."""
        return self.content or ""

    def to_message(self) -> Message:
        """
        Convert this response to a Message for appending to conversation history.

        Use this when auto_execute=False and you are manually managing tool calls:

            messages.append(response.to_message())   # append assistant turn
            # then append tool result messages
            # then call registry.complete() again
        """
        return Message(
            role="assistant",
            content=self.content,
            tool_calls=self.tool_calls,
        )


@dataclass
class StructuredModelResponse:
    """
    Response from registry.complete_structured().

    data       — the structured output as a plain dict (parse into your model via MyModel(**data))
    model_name — which registered model produced this response
    usage      — token counts (may be None for providers that don't report them)

    Usage:
        result = await registry.complete_structured("smart", messages, MyModel)
        obj = MyModel(**result.data)
        print(obj.title)
    """
    data:       dict[str, Any]
    model_name: str
    usage:      UsageInfo | None = None


@dataclass
class ToolCallMessage:
    """
    A single tool call from the model, with a helper to build the tool result message.
    Access via response.tool_calls — each element is a ToolCall (from types.py).
    Use ToolCall.to_tool_result_message(result) to build the reply.
    """


# Attach helper method to ToolCall at module level
# (avoids modifying types.py which is a pure data module)
def _to_tool_result_message(self: ToolCall, result: str) -> Message:
    """
    Build the tool result Message to append after executing this tool call.

    Usage:
        for tc in response.tool_calls:
            result = await mcp.call_tool(tc.name, tc.arguments)
            messages.append(tc.to_tool_result_message(result))
    """
    return Message(
        role="tool",
        content=result,
        tool_call_id=self.id,
        name=self.name,
    )

ToolCall.to_tool_result_message = _to_tool_result_message  # type: ignore[attr-defined]


# ── Registry entry ──────────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    name:        str
    client:      BaseLLMClient
    description: str        = ""
    tags:        list[str]  = field(default_factory=list)

    @property
    def provider(self) -> str:
        return self.client.provider_name().split("/")[0]


# ── Registry ────────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Central registry for all LLM models in your application.

    Thread-safe for reads. Registration should happen at startup before
    serving requests. Hot-swap via swap() is safe at runtime.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelEntry] = {}

    # ── Registration ────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        client: BaseLLMClient,
        description: str = "",
        tags: list[str] | None = None,
    ) -> "ModelRegistry":
        """
        Register a model. Returns self for fluent chaining.

        Args:
            name:        Unique name used to look up this model. e.g. "fast", "smart"
            client:      Any BaseLLMClient — Anthropic, OpenAI, Gemini, Grok, Ollama...
            description: Human-readable description. Shown in registry.list().
            tags:        Free-form tags for querying. e.g. ["cheap", "vision", "fast"]

        Raises:
            TypeError: if client is not a BaseLLMClient instance.
        """
        if not isinstance(client, BaseLLMClient):
            raise TypeError(f"client must be a BaseLLMClient, got {type(client)}")
        self._models[name] = ModelEntry(
            name=name,
            client=client,
            description=description,
            tags=tags or [],
        )
        logger.debug("[registry] registered model '%s' (%s)", name, client.provider_name())
        return self

    def swap(self, name: str, new_client: BaseLLMClient) -> None:
        """
        Hot-swap a registered model without restarting.
        Use for model upgrades, A/B testing, fallback switching.
        """
        if name not in self._models:
            raise ModelNotFoundError(f"Cannot swap '{name}' — not registered.")
        self._models[name].client = new_client
        logger.info("[registry] swapped model '%s' → %s", name, new_client.provider_name())

    # ── Lookup ──────────────────────────────────────────────────────────────

    def get(self, name: str) -> BaseLLMClient:
        """
        Get a registered client by name.

        Raises:
            ModelNotFoundError: if name is not registered.
        """
        if name not in self._models:
            available = list(self._models.keys())
            raise ModelNotFoundError(
                f"Model '{name}' not registered. Available: {available}"
            )
        return self._models[name].client

    def list(self) -> list[ModelEntry]:
        """Return all registered model entries."""
        return list(self._models.values())

    def find_by_tag(self, tag: str) -> list[str]:
        """Return names of all models that have this tag."""
        return [
            name for name, entry in self._models.items()
            if tag in entry.tags
        ]

    def find_by_provider(self, provider: str) -> list[str]:
        """
        Return names of all models from a provider.
        e.g. find_by_provider("anthropic") → ["fast", "smart", "best"]
        """
        return [
            name for name, entry in self._models.items()
            if provider.lower() in entry.provider.lower()
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._models

    def __repr__(self) -> str:
        entries = ", ".join(
            f"{e.name}={e.client.provider_name()}"
            for e in self._models.values()
        )
        return f"ModelRegistry({entries})"

    # ── Core call ───────────────────────────────────────────────────────────

    async def complete(
        self,
        model_name: str,
        messages: list[Message],
        tools: list[MCPTool] | None = None,
        system: str | None = None,
        auto_execute: bool = False,
        tool_executor: ToolExecutor | None = None,
        max_iterations: int = 10,
    ) -> ModelResponse:
        """
        Call a registered model.

        Args:
            model_name:     Name used when registering the model.
            messages:       Conversation history. Not mutated — a copy is used internally.
            tools:          MCP tools to expose to the model. Pass None for text-only calls.
            system:         System prompt. Handled correctly per-provider (Anthropic takes
                            it as a top-level param; OpenAI/Gemini as a system message).
            auto_execute:   False (default) — return raw ModelResponse, tool_calls included.
                            True — run tool-call loop until model stops, return final answer.
            tool_executor:  Required when auto_execute=True.
                            Async callable: (tool_name, arguments) → result_string.
                            The registry calls this for each tool call and feeds results back.
            max_iterations: Safety cap for the auto_execute loop.

        Returns:
            ModelResponse — always. Check .has_tool_calls when auto_execute=False.

        Raises:
            ModelNotFoundError:  model_name not registered.
            ModelRegistryError:  auto_execute=True but no tool_executor provided.
            MaxIterationsError:  auto_execute loop hit max_iterations.
        """
        client = self.get(model_name)

        if not auto_execute:
            return await self._single_call(client, model_name, messages, tools, system)

        # ── auto_execute=True ────────────────────────────────────────────
        if tool_executor is None:
            raise ModelRegistryError(
                "tool_executor is required when auto_execute=True. "
                "Provide an async callable: async def execute(name, args) -> str"
            )
        return await self._auto_execute_loop(
            client, model_name, messages, tools, system,
            tool_executor, max_iterations,
        )

    async def complete_structured(
        self,
        model_name: str,
        messages: list[Message],
        response_schema: type | dict[str, Any],
        system: str | None = None,
    ) -> StructuredModelResponse:
        """
        Call a registered model and guarantee a structured JSON response.

        Delegates to the provider client's native structured-output mechanism:
          - Anthropic: forced tool use with synthetic "structured_output" tool
          - OpenAI:    response_format with json_schema + strict=True
          - Gemini:    response_mime_type="application/json" + response_schema

        Args:
            model_name:      Name used when registering the model.
            messages:        Conversation history.
            response_schema: A Pydantic model class (v1 or v2) or a JSON Schema dict.
            system:          Optional system prompt.

        Returns:
            StructuredModelResponse — access .data for the plain dict.
            Parse into your Pydantic model: MyModel(**result.data)

        Raises:
            ModelNotFoundError:   model_name not registered.
            StructuredOutputError: the model failed to return valid structured output.

        Usage:
            result = await registry.complete_structured("smart", messages, ArticleSchema)
            article = ArticleSchema(**result.data)
        """
        client = self.get(model_name)
        logger.debug("[registry] complete_structured calling '%s'", model_name)

        data = await client.complete_structured(messages, response_schema, system=system)

        # UsageInfo is not available from complete_structured — structured calls
        # don't go through LLMResponse. Token tracking can be added per-provider later.
        return StructuredModelResponse(
            data=data,
            model_name=model_name,
        )

    # ── Internal ────────────────────────────────────────────────────────────

    async def _single_call(
        self,
        client: BaseLLMClient,
        model_name: str,
        messages: list[Message],
        tools: list[MCPTool] | None,
        system: str | None,
    ) -> ModelResponse:
        """One LLM call. Returns raw response including any tool_calls."""
        logger.debug("[registry] calling '%s' (auto_execute=False)", model_name)
        llm_resp = await client.complete(messages, tools=tools, system=system)
        return ModelResponse(
            content=llm_resp.content,
            tool_calls=llm_resp.tool_calls,
            stop_reason=llm_resp.stop_reason,
            model_name=model_name,
            usage=UsageInfo(llm_resp.input_tokens, llm_resp.output_tokens),
        )

    async def _auto_execute_loop(
        self,
        client: BaseLLMClient,
        model_name: str,
        messages: list[Message],
        tools: list[MCPTool] | None,
        system: str | None,
        tool_executor: ToolExecutor,
        max_iterations: int,
    ) -> ModelResponse:
        """
        Full tool-call loop. Runs until model stops requesting tools or
        max_iterations is reached.

        Tool calls within a single response are executed in PARALLEL
        (model returned them together — they are independent by definition).
        """
        working = list(messages)  # never mutate caller's list

        for iteration in range(max_iterations):
            logger.debug(
                "[registry] auto_execute iteration %d/%d for '%s'",
                iteration + 1, max_iterations, model_name,
            )
            llm_resp = await client.complete(working, tools=tools, system=system)

            # Always append the assistant turn first
            working.append(Message(
                role="assistant",
                content=llm_resp.content,
                tool_calls=llm_resp.tool_calls,
            ))

            # Model is done
            if llm_resp.stop_reason != StopReason.TOOL_USE or not llm_resp.tool_calls:
                logger.debug("[registry] '%s' finished after %d iteration(s)", model_name, iteration + 1)
                return ModelResponse(
                    content=llm_resp.content,
                    tool_calls=None,
                    stop_reason=StopReason.END_TURN,
                    model_name=model_name,
                    usage=UsageInfo(llm_resp.input_tokens, llm_resp.output_tokens),
                )

            # Execute all tool calls from this response in parallel
            tool_calls = llm_resp.tool_calls
            logger.debug(
                "[registry] executing %d tool call(s) in parallel: %s",
                len(tool_calls),
                [tc.name for tc in tool_calls],
            )

            raw_results = await asyncio.gather(
                *[self._safe_execute(tool_executor, tc) for tc in tool_calls],
            )

            # Append one tool-result message per call
            for tc, result in zip(tool_calls, raw_results):
                working.append(Message(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id,
                    name=tc.name,
                ))
                logger.debug("[registry] tool '%s' → %s", tc.name, result[:100])

        raise MaxIterationsError(
            f"Model '{model_name}' did not finish within {max_iterations} iterations. "
            "Increase max_iterations or check for a tool-calling loop."
        )

    @staticmethod
    async def _safe_execute(executor: ToolExecutor, tc: ToolCall) -> str:
        """Execute one tool call, catching exceptions so one failure doesn't abort the batch."""
        try:
            return await executor(tc.name, tc.arguments)
        except Exception as exc:
            error = f"Tool '{tc.name}' failed: {exc}"
            logger.error("[registry] %s", error)
            return error
