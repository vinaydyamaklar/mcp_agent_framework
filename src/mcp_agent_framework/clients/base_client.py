"""
Abstract base class for all LLM provider clients.

Every provider (Anthropic, OpenAI, Gemini, Grok, ...) must subclass this
and implement `complete()` and `complete_structured()`. The rest of the
framework only talks to this interface — it never imports a provider SDK directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from mcp_agent_framework.types import LLMResponse, MCPTool, Message, StreamEvent


class BaseLLMClient(ABC):
    """
    Provider-agnostic LLM interface.

    Subclasses handle:
    - Converting MCPTool list → provider tool format
    - Converting Message list → provider message format
    - Calling the provider API
    - Converting the provider response → LLMResponse
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[MCPTool] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """
        Send messages to the LLM and return a normalised response.

        Args:
            messages: Conversation history in canonical Message format.
            tools:    Available MCP tools. Pass None or [] for text-only calls.
            system:   System prompt. Some providers take it separately (Anthropic),
                      others prepend it as a "system" message (OpenAI). Each
                      client handles this difference internally.

        Returns:
            LLMResponse with normalised content, tool_calls, and stop_reason.
        """
        ...

    @abstractmethod
    async def complete_structured(
        self,
        messages: list[Message],
        response_schema: type | dict[str, Any],
        system: str | None = None,
    ) -> dict[str, Any]:
        """
        Call the LLM and guarantee a structured JSON response matching response_schema.

        Each provider uses its native structured-output mechanism:
          - Anthropic: forced tool use with synthetic "structured_output" tool
          - OpenAI:    response_format with json_schema + strict=True
          - Gemini:    response_mime_type="application/json" + response_schema

        Args:
            messages:        Conversation history in canonical Message format.
            response_schema: A Pydantic model class (v1 or v2) or a JSON Schema dict.
            system:          Optional system prompt.

        Returns:
            A plain dict that matches the schema. Parse into your Pydantic model if needed:
                result = await client.complete_structured(messages, MyModel)
                obj = MyModel(**result)

        Raises:
            StructuredOutputError: if the model fails to return valid structured output.
        """
        ...

    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name used in logs."""
        ...

    async def stream(
        self,
        messages: list[Message],
        tools: list[MCPTool] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream events from the LLM. Default implementation falls back to complete()
        and yields events as one chunk. Override in provider clients for real streaming.
        """
        response = await self.complete(messages, tools=tools, system=system)
        if response.reasoning:
            yield StreamEvent(type="thinking", delta=response.reasoning)
        if response.content:
            yield StreamEvent(type="text", delta=response.content)
        if response.tool_calls:
            for tc in response.tool_calls:
                yield StreamEvent(type="tool_call", tool_name=tc.name, tool_call_id=tc.id, tool_args=tc.arguments)
        yield StreamEvent(type="done", stop_reason=response.stop_reason.value)
