"""
Google Gemini client (google-genai SDK).
Install: pip install "mcp-agent-framework[gemini]"

SDK: from google import genai  (NOT google-generativeai — that is the old SDK)

Tool calling contract (Gemini-specific, verified April 2026):
  - FunctionDeclaration uses parameters_json_schema — accepts plain JSON Schema dict directly.
    Do NOT use types.Schema / types.FunctionDeclaration(parameters=...) — that is the old API.
  - Stop reason does NOT distinguish tool use — detect by checking parts for function_call.
  - part.function_call.id exists in current API — use it directly (no fallback needed in prod,
    but we keep a graceful fallback for safety).
  - Tool results: role="user" Content with FunctionResponse(name, id, response={"result": ...})
    response MUST be a dict — plain strings are rejected.
  - System prompt: GenerateContentConfig(system_instruction=...) — NOT a message in contents.
  - Gemini role for model turns: "model"  (NOT "assistant")
  - Async: client.aio.models.generate_content(...)

Structured output (complete_structured):
  - Uses GenerateContentConfig(response_mime_type="application/json", response_schema=schema)
  - schema is a plain JSON Schema dict (same as parameters_json_schema — no conversion needed)
  - raw.text is the JSON string — json.loads() to get the dict
  - Do NOT pass tools alongside response_mime_type — Gemini rejects that combination
"""
from __future__ import annotations

import json
import os
from typing import Any

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.clients.schema_utils import StructuredOutputError, schema_from
from mcp_agent_framework.types import LLMResponse, MCPTool, Message, StopReason, ToolCall


class GeminiClient(BaseLLMClient):

    def __init__(
        self,
        model:   str = "gemini-2.5-flash",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model:   Gemini model ID.
                     Stable:   "gemini-2.5-flash", "gemini-2.5-pro"
                     Latest:   "gemini-3.1-pro"  (April 2026)
            api_key: GOOGLE_API_KEY — reads from env if not provided.
            **kwargs: Forwarded to GenerateContentConfig
                      (temperature, top_p, max_output_tokens, etc.)
        """
        try:
            from google import genai as _genai
            from google.genai import types as _types
            self._genai  = _genai
            self._types  = _types
        except ImportError:
            raise ImportError(
                'Install Gemini support: pip install "mcp-agent-framework[gemini]"'
            )

        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Export it or pass api_key= explicitly."
            )
        self._client = _genai.Client(api_key=resolved_key)
        self._model  = model
        self._extra  = kwargs

    def provider_name(self) -> str:
        return f"gemini/{self._model}"

    # ── Public ──────────────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[Message],
        tools:    list[MCPTool] | None = None,
        system:   str | None = None,
    ) -> LLMResponse:
        t        = self._types
        contents = [self._encode_message(m) for m in messages]

        config_kwargs: dict[str, Any] = {**self._extra}
        if system:
            config_kwargs["system_instruction"] = system   # top-level config, NOT a message
        if tools:
            config_kwargs["tools"] = [self._encode_tools(tools)]

        config = t.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        raw = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        return self._decode_response(raw)

    async def complete_structured(
        self,
        messages: list[Message],
        response_schema: type | dict[str, Any],
        system: str | None = None,
    ) -> dict[str, Any]:
        """
        Use Gemini's native JSON mode to guarantee a structured response.

        GenerateContentConfig(response_mime_type="application/json", response_schema=...)
        tells Gemini to constrain its output to valid JSON matching the schema.
        Do NOT combine with tools — Gemini rejects that combination.
        """
        t      = self._types
        schema = schema_from(response_schema)

        config_kwargs: dict[str, Any] = {
            **self._extra,
            "response_mime_type": "application/json",
            "response_schema":    schema,
        }
        if system:
            config_kwargs["system_instruction"] = system

        raw = await self._client.aio.models.generate_content(
            model=self._model,
            contents=[self._encode_message(m) for m in messages],
            config=t.GenerateContentConfig(**config_kwargs),
        )

        text = getattr(raw, "text", None)
        if not text:
            raise StructuredOutputError(
                f"Gemini ({self._model}) returned empty response for structured output request."
            )

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(
                f"Gemini ({self._model}) returned non-JSON content: {text[:200]}"
            ) from exc

    # ── Encode: Canonical → Gemini ───────────────────────────────────────────

    def _encode_message(self, msg: Message) -> Any:
        t = self._types

        if msg.role == "tool":
            # Tool result: user-role Content with FunctionResponse.
            # response MUST be a dict — plain strings cause API errors.
            return t.Content(
                role="user",
                parts=[t.Part(
                    function_response=t.FunctionResponse(
                        name=msg.name or "",
                        id=msg.tool_call_id,
                        response={"result": msg.content or ""},
                    )
                )],
            )

        gemini_role = "model" if msg.role == "assistant" else "user"
        parts: list[Any] = []

        if msg.role == "assistant" and msg.tool_calls:
            if msg.content:
                parts.append(t.Part(text=msg.content))
            for tc in msg.tool_calls:
                parts.append(t.Part(
                    function_call=t.FunctionCall(
                        name=tc.name,
                        args=tc.arguments,
                        id=tc.id,
                    )
                ))
        else:
            parts.append(t.Part(text=msg.content or ""))

        return t.Content(role=gemini_role, parts=parts)

    def _encode_tools(self, tools: list[MCPTool]) -> Any:
        """
        Build a single types.Tool from all MCPTool definitions.

        Uses parameters_json_schema (April 2026 API) — passes the JSON Schema
        dict directly. No types.Schema conversion needed.
        """
        t = self._types
        return t.Tool(
            function_declarations=[
                t.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.input_schema,  # plain JSON Schema dict
                )
                for tool in tools
            ]
        )

    # ── Decode: Gemini → Canonical ───────────────────────────────────────────

    def _decode_response(self, raw: Any) -> LLMResponse:
        candidate = raw.candidates[0] if raw.candidates else None
        if not candidate:
            return LLMResponse(
                content=None,
                tool_calls=None,
                stop_reason=StopReason.END_TURN,
            )

        finish = str(getattr(candidate, "finish_reason", "STOP"))
        stop_reason = (
            StopReason.MAX_TOKENS
            if "MAX_TOKENS" in finish or finish == "2"
            else StopReason.END_TURN
        )

        text:       str | None            = None
        tool_calls: list[ToolCall] | None = None

        for part in (candidate.content.parts if candidate.content else []):
            if hasattr(part, "text") and part.text:
                text = part.text

            elif getattr(part, "function_call", None) is not None:
                fc = part.function_call
                if tool_calls is None:
                    tool_calls  = []
                    stop_reason = StopReason.TOOL_USE

                # id is present in current API; fallback to name for safety
                call_id = getattr(fc, "id", None) or fc.name
                tool_calls.append(ToolCall(
                    id=call_id,
                    name=fc.name,
                    arguments=dict(fc.args) if fc.args else {},
                ))

        usage = getattr(raw, "usage_metadata", None)
        return LLMResponse(
            content=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )
