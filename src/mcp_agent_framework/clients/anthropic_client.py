"""
Anthropic (Claude) client.

Install: pip install "mcp-agent-framework[anthropic]"

Model IDs (April 2026):
    claude-opus-4-6            — most powerful
    claude-sonnet-4-6          — balanced (default)
    claude-haiku-4-5-20251001  — fastest, cheapest

Tool calling contract (Anthropic-specific rules):
  - Tools use "input_schema" key  (NOT "parameters" — that is OpenAI)
  - Stop reason is "tool_use"     (NOT "tool_calls")
  - Tool results go as a USER turn with content type "tool_result"
  - The assistant turn that requested tools must contain the FULL content list
    (text blocks + tool_use blocks) — Anthropic will reject partial content
  - Tool result content is a string (or list of content blocks for images)

Structured output (complete_structured):
  - Uses forced tool use: tool_choice={"type": "tool", "name": "structured_output"}
  - A synthetic tool named "structured_output" is created with the caller's schema
    as its input_schema — Anthropic guarantees the tool is called exactly once
  - The tool call arguments (block.input) ARE the structured result — no JSON parsing needed
"""
from __future__ import annotations

import os
from typing import Any

from collections.abc import AsyncIterator

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.clients.schema_utils import StructuredOutputError, schema_from
from mcp_agent_framework.types import LLMResponse, MCPTool, Message, StopReason, StreamEvent, ToolCall


class AnthropicClient(BaseLLMClient):

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_tokens: int = 8096,
        enable_thinking: bool = False,
        thinking_budget: int = 8000,
        **kwargs: Any,
    ) -> None:
        try:
            import anthropic as _anthropic
            self._anthropic = _anthropic
        except ImportError:
            raise ImportError(
                'Install Anthropic support: pip install "mcp-agent-framework[anthropic]"'
            )

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Export it or pass api_key= explicitly."
            )
        self._client          = self._anthropic.AsyncAnthropic(api_key=resolved_key)
        self._model           = model
        self._max_tokens      = max_tokens
        self._enable_thinking = enable_thinking
        self._thinking_budget = thinking_budget
        self._extra           = kwargs  # temperature, top_p, etc.

    def provider_name(self) -> str:
        return f"anthropic/{self._model}"

    # ── Public ──────────────────────────────────────────────────────────────

    async def complete(
        self,
        messages:  list[Message],
        tools:     list[MCPTool] | None = None,
        system:    str | None = None,
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "model":      self._model,
            "max_tokens": self._max_tokens,
            "messages":   [self._encode_message(m) for m in messages],
            **self._extra,
        }
        if system:
            params["system"] = system            # Anthropic: system is a top-level param
        if tools:
            params["tools"] = [self._encode_tool(t) for t in tools]
        if self._enable_thinking:
            params["thinking"] = {"type": "enabled", "budget_tokens": self._thinking_budget}

        raw = await self._client.messages.create(**params)
        return self._decode_response(raw)

    async def complete_structured(
        self,
        messages: list[Message],
        response_schema: type | dict[str, Any],
        system: str | None = None,
    ) -> dict[str, Any]:
        """
        Force Anthropic to return a structured response by using a synthetic tool.

        Anthropic's structured output mechanism: define a tool whose input_schema
        matches your desired output shape, then force the model to call it exactly
        once via tool_choice={"type": "tool", "name": "structured_output"}.
        The tool's call arguments ARE the structured output — no JSON parsing needed.
        """
        schema = schema_from(response_schema)

        params: dict[str, Any] = {
            "model":      self._model,
            "max_tokens": self._max_tokens,
            "messages":   [self._encode_message(m) for m in messages],
            "tools": [
                {
                    "name":         "structured_output",
                    "description":  "Return a structured response matching the provided schema.",
                    "input_schema": schema,
                }
            ],
            # Force this specific tool to be called exactly once
            "tool_choice": {"type": "tool", "name": "structured_output"},
            **self._extra,
        }
        if system:
            params["system"] = system

        raw = await self._client.messages.create(**params)

        for block in raw.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "structured_output":
                return block.input  # already a dict — Anthropic deserialises for us

        raise StructuredOutputError(
            f"Anthropic ({self._model}) did not call the structured_output tool. "
            f"stop_reason={raw.stop_reason}"
        )

    # ── Encode: Canonical → Anthropic ───────────────────────────────────────

    def _encode_message(self, msg: Message) -> dict[str, Any]:
        if msg.role == "system":
            # Should be passed via `system=` param, but if it ends up here
            # treat it as a user message so Anthropic doesn't reject the call.
            return {"role": "user", "content": msg.content or ""}

        if msg.role == "tool":
            # Tool results: user turn with tool_result content block.
            # tool_call_id maps back to the tool_use block id in the prior assistant turn.
            return {
                "role": "user",  # Anthropic only has "user" and "assistant" nothing like "tool"
                "content": [
                    {
                        "type":        "tool_result",
                        "tool_use_id": msg.tool_call_id,   # REQUIRED by Anthropic
                        "content":     msg.content or "",
                    }
                ],
            }

        if msg.role == "assistant" and msg.tool_calls:
            # Assistant turn that contains tool calls.
            # Anthropic REQUIRES the full content list:
            #   optional TextBlock first, then one ToolUseBlock per call.
            content: list[dict[str, Any]] = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                content.append({
                    "type":  "tool_use",
                    "id":    tc.id,
                    "name":  tc.name,
                    "input": tc.arguments,   # dict, not JSON string
                })
            return {"role": "assistant", "content": content}

        # Plain user or assistant message
        return {"role": msg.role, "content": msg.content or ""}

    def _encode_tool(self, tool: MCPTool) -> dict[str, Any]:
        return {
            "name":         tool.name,
            "description":  tool.description,
            "input_schema": tool.input_schema,   # Anthropic key — NOT "parameters"
        }

    # ── Decode: Anthropic → Canonical ───────────────────────────────────────

    def _decode_response(self, raw: Any) -> LLMResponse:
        stop = raw.stop_reason
        if stop == "tool_use":
            stop_reason = StopReason.TOOL_USE
        elif stop == "max_tokens":
            stop_reason = StopReason.MAX_TOKENS
        else:
            stop_reason = StopReason.END_TURN

        text:       str | None        = None
        tool_calls: list[ToolCall] | None = None
        reasoning_parts: list[str] = []

        for block in raw.content:
            if block.type == "thinking":
                reasoning_parts.append(block.thinking)
            elif block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,   # already a dict
                ))

        reasoning = "".join(reasoning_parts) if reasoning_parts else None

        return LLMResponse(
            content=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            input_tokens=raw.usage.input_tokens if raw.usage else None,
            output_tokens=raw.usage.output_tokens if raw.usage else None,
            reasoning=reasoning,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[MCPTool] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        params: dict[str, Any] = {
            "model":      self._model,
            "max_tokens": self._max_tokens,
            "messages":   [self._encode_message(m) for m in messages],
            **self._extra,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [self._encode_tool(t) for t in tools]
        if self._enable_thinking:
            params["thinking"] = {"type": "enabled", "budget_tokens": self._thinking_budget}

        async with self._client.messages.stream(**params) as stream:
            async for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_delta":
                    delta = event.delta
                    dtype = getattr(delta, "type", None)
                    if dtype == "text_delta" and getattr(delta, "text", None):
                        yield StreamEvent(type="text", delta=delta.text)
                    elif dtype == "thinking_delta" and getattr(delta, "thinking", None):
                        yield StreamEvent(type="thinking", delta=delta.thinking)
                    # input_json_delta: skip — accumulated in final message

            final = await stream.get_final_message()
            for block in final.content:
                if getattr(block, "type", None) == "tool_use":
                    yield StreamEvent(
                        type="tool_call",
                        tool_name=block.name,
                        tool_call_id=block.id,
                        tool_args=block.input,
                    )

            stop = final.stop_reason
            if stop == "tool_use":
                sr = "tool_use"
            elif stop == "max_tokens":
                sr = "max_tokens"
            else:
                sr = "end_turn"
            yield StreamEvent(type="done", stop_reason=sr)
