"""
OpenAI client — also covers Grok (xAI), Ollama, Together AI, and any
OpenAI-compatible endpoint via the base_url parameter.

Install: pip install "mcp-agent-framework[openai]"

Model IDs (April 2026):
    gpt-5.4        — latest, most capable
    gpt-5.4-mini   — fast, cost-efficient
    gpt-5.4-nano   — fastest, cheapest

Provider variants:
    OpenAI:   OpenAIClient("gpt-5.4")
    Grok:     OpenAIClient("grok-3",  api_key=XAI_KEY,       base_url="https://api.x.ai/v1")
    Ollama:   OpenAIClient("llama3",  api_key="ollama",       base_url="http://localhost:11434/v1")
    Together: OpenAIClient("mistral", api_key=TOGETHER_KEY,   base_url="https://api.together.xyz/v1")

Tool calling contract (OpenAI-specific rules):
  - Tools use "parameters" key  (NOT "input_schema" — that is Anthropic)
  - Stop reason is "tool_calls" (NOT "tool_use")
  - arguments in the response are a JSON STRING  → must json.loads() before use
  - Tool results go as role="tool" messages with tool_call_id
  - The assistant message that requested tools sets content=None (not omitted — None)
  - System prompt prepended as role="system" message (first in the list)

Structured output (complete_structured):
  - Uses response_format={"type": "json_schema", "json_schema": {"name": "response", "schema": ..., "strict": True}}
  - Only supported by models that implement the Structured Outputs feature (gpt-5.4, gpt-4o-mini, etc.)
  - Ollama / Grok / Together compatibility varies — falls back gracefully if unsupported
  - message.content is a JSON STRING → json.loads() to get the dict
"""
from __future__ import annotations

import json
import os
from typing import Any

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.clients.schema_utils import StructuredOutputError, schema_from
from mcp_agent_framework.types import LLMResponse, MCPTool, Message, StopReason, ToolCall


class OpenAIClient(BaseLLMClient):

    def __init__(
        self,
        model:    str = "gpt-5.4",
        api_key:  str | None = None,
        base_url: str | None = None,   # override for Grok, Ollama, Together, etc.
        **kwargs: Any,
    ) -> None:
        try:
            from openai import AsyncOpenAI
            self._AsyncOpenAI = AsyncOpenAI
        except ImportError:
            raise ImportError(
                'Install OpenAI support: pip install "mcp-agent-framework[openai]"'
            )

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client  = self._AsyncOpenAI(api_key=resolved_key, base_url=base_url)
        self._model   = model
        self._extra   = kwargs   # temperature, top_p, etc.

    def provider_name(self) -> str:
        return f"openai/{self._model}"

    # ── Public ──────────────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[Message],
        tools:    list[MCPTool] | None = None,
        system:   str | None = None,
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "model":    self._model,
            "messages": self._build_messages(messages, system),
            **self._extra,
        }
        if tools:
            params["tools"] = [self._encode_tool(t) for t in tools]

        raw    = await self._client.chat.completions.create(**params)
        return self._decode_response(raw)

    async def complete_structured(
        self,
        messages: list[Message],
        response_schema: type | dict[str, Any],
        system: str | None = None,
    ) -> dict[str, Any]:
        """
        Use OpenAI Structured Outputs (response_format=json_schema) to guarantee
        a valid JSON response matching response_schema.

        strict=True enables constrained decoding — the model CANNOT deviate from
        the schema. Only supported on models with Structured Outputs support.
        """
        schema = schema_from(response_schema)

        params: dict[str, Any] = {
            "model":    self._model,
            "messages": self._build_messages(messages, system),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name":   "response",
                    "schema": schema,
                    "strict": True,
                },
            },
            **self._extra,
        }

        raw = await self._client.chat.completions.create(**params)
        if not raw.choices:
            raise StructuredOutputError(
                f"OpenAI ({self._model}) returned no choices for structured output request."
            )
        content = raw.choices[0].message.content

        if not content:
            raise StructuredOutputError(
                f"OpenAI ({self._model}) returned empty content for structured output request. "
                f"finish_reason={raw.choices[0].finish_reason}"
            )

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(
                f"OpenAI ({self._model}) returned non-JSON content: {content[:200]}"
            ) from exc

    # ── Encode: Canonical → OpenAI ──────────────────────────────────────────

    def _build_messages(
        self, messages: list[Message], system: str | None
    ) -> list[dict[str, Any]]:
        """Prepend system prompt as a role=system message if provided."""
        result: list[dict[str, Any]] = []
        if system:
            result.append({"role": "system", "content": system})
        for m in messages:
            result.append(self._encode_message(m))
        return result

    def _encode_message(self, msg: Message) -> dict[str, Any]:
        if msg.role == "tool":
            # Tool result: role=tool with tool_call_id linking it to the request.
            return {
                "role":         "tool",
                "tool_call_id": msg.tool_call_id,   # REQUIRED — links result to call
                "content":      msg.content or "",
            }

        if msg.role == "assistant" and msg.tool_calls:
            # Assistant turn with tool calls.
            # content MUST be None (not omitted) when tool_calls are present.
            # arguments MUST be a JSON string (not a dict).
            return {
                "role":       "assistant",
                "content":    None,
                "tool_calls": [
                    {
                        "id":   tc.id,
                        "type": "function",
                        "function": {
                            "name":      tc.name,
                            "arguments": json.dumps(tc.arguments),  # string — NOT dict
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }

        return {"role": msg.role, "content": msg.content or ""}

    def _encode_tool(self, tool: MCPTool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name":        tool.name,
                "description": tool.description,
                "parameters":  tool.input_schema,   # OpenAI key — NOT "input_schema"
            },
        }

    # ── Decode: OpenAI → Canonical ───────────────────────────────────────────

    def _decode_response(self, raw: Any) -> LLMResponse:
        if not raw.choices:
            raise StructuredOutputError(
                f"OpenAI ({self._model}) returned no choices."
            )
        choice  = raw.choices[0]
        message = choice.message
        reason  = choice.finish_reason

        if reason == "tool_calls":
            stop_reason = StopReason.TOOL_USE
        elif reason == "length":
            stop_reason = StopReason.MAX_TOKENS
        else:
            stop_reason = StopReason.END_TURN

        tool_calls: list[ToolCall] | None = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = raw.usage
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
