"""
Pattern: Single Agent Loop (ReAct)

The simplest agentic pattern:
  1. LLM receives messages + available tools
  2. LLM either responds with text (done) or requests tool calls
  3. Tools are executed via MCP
  4. Results are appended to conversation and we go back to step 1

Use this when: one LLM + one set of tools is enough for the task.
This is the building block all other patterns are built on.

                  ┌─────────────┐
   user message → │     LLM     │ → text response (done)
                  └──────┬──────┘
                         │ tool_calls
                         ▼
                  ┌─────────────┐
                  │  MCP Server │ (tools live here)
                  └──────┬──────┘
                         │ tool results
                         └──────────── back to LLM
"""
from __future__ import annotations

import logging
import time
from typing import AsyncIterator

from fastmcp import Client

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.observability.run_context import RunContext
from mcp_agent_framework.observability.tracer import TraceEventType
from mcp_agent_framework.patterns._tool_utils import call_tool, list_tools
from mcp_agent_framework.types import AgentConfig, Message, StopReason, ToolCall

logger = logging.getLogger(__name__)


class SingleAgentLoop:
    """
    One LLM + one MCP server, looping until the model stops calling tools.

    Usage:
        config = AgentConfig(
            mcp_server_config={"mcpServers": {"my_server": {"url": "http://localhost:8001/mcp"}}},
            system_prompt="You are a helpful assistant.",
        )
        agent = SingleAgentLoop(llm_client=AnthropicClient(), config=config)
        result = await agent.run("What files are in /tmp?")
        print(result)
    """

    def __init__(self, llm_client: BaseLLMClient, config: AgentConfig):
        self._llm    = llm_client
        self._config = config

    async def run(
        self,
        user_message: str,
        history: list[Message] | None = None,
        context: RunContext | None = None,
    ) -> str:
        """
        Run the agent loop on a single user message.

        Args:
            user_message: The user's input.
            history:      Optional prior conversation context.
            context:      Optional RunContext for tracing. If provided, every
                          LLM call and tool call is recorded automatically.

        Returns:
            The model's final text response.
        """
        t_pattern_start = time.monotonic()

        if context:
            await context.emit(TraceEventType.PATTERN_START, {
                "pattern_name": "SingleAgentLoop",
                "user_message": user_message,
                "model":        self._llm.provider_name(),
            })

        async with Client(self._config.mcp_server_config) as mcp:
            tools    = await list_tools(mcp)
            messages = list(history or [])  # copy — protects caller's list from mutation
            messages.append(Message(role="user", content=user_message))

            for iteration in range(self._config.max_iterations):
                logger.debug("[%s] iteration %d", self._llm.provider_name(), iteration + 1)

                # Emit LLM_START before each LLM call
                if context:
                    await context.emit(TraceEventType.LLM_START, {
                        "model":          self._llm.provider_name(),
                        "iteration":      iteration + 1,
                        "message_count":  len(messages),
                        "tool_count":     len(tools),
                    })

                t_llm = time.monotonic()
                response = await self._llm.complete(
                    messages=messages,
                    tools=tools,
                    system=self._config.system_prompt or None,
                )

                # Emit LLM_END with latency and stop reason
                if context:
                    await context.emit(TraceEventType.LLM_END, {
                        "model":        self._llm.provider_name(),
                        "stop_reason":  response.stop_reason,
                        "elapsed_ms":   round((time.monotonic() - t_llm) * 1000, 2),
                        "input_tokens": response.input_tokens,
                        "output_tokens":response.output_tokens,
                    })

                # Append the assistant's turn to history
                messages.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                if response.stop_reason != StopReason.TOOL_USE or not response.tool_calls:
                    # Model is done — return its text
                    final = response.content or ""
                    if context:
                        await context.emit(TraceEventType.PATTERN_END, {
                            "pattern_name": "SingleAgentLoop",
                            "iterations":   iteration + 1,
                            "result":       final[:200],
                            "elapsed_ms":   round((time.monotonic() - t_pattern_start) * 1000, 2),
                        })
                    return final

                # Execute every tool call the model requested
                for tool_call in response.tool_calls:
                    # call_tool handles TOOL_START / TOOL_END / TOOL_ERROR tracing
                    result = await call_tool(mcp, tool_call, context=context)
                    messages.append(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    ))
                    logger.debug("[tool] %s → %s", tool_call.name, result[:120])

            logger.warning("Hit max_iterations=%d, returning last content", self._config.max_iterations)
            # Scan backwards for the last assistant message so we don't return
            # a tool result as the final response.
            for msg in reversed(messages):
                if msg.role == "assistant" and msg.content:
                    if context:
                        await context.emit(TraceEventType.PATTERN_END, {
                            "pattern_name": "SingleAgentLoop",
                            "iterations":   self._config.max_iterations,
                            "result":       msg.content[:200],
                            "elapsed_ms":   round((time.monotonic() - t_pattern_start) * 1000, 2),
                            "hit_max_iterations": True,
                        })
                    return msg.content
            return ""

    async def run_stream(
        self,
        user_message: str,
        history: list[Message] | None = None,
        context: RunContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream events from the agent loop token by token.

        Yields:
            StreamEvent(type="thinking") — reasoning tokens
            StreamEvent(type="text")     — response tokens
            StreamEvent(type="tool_start") — before tool execution
            StreamEvent(type="tool_end")   — after tool execution
        Tool calls are executed between LLM calls; only text/thinking events stream live.
        """
        from mcp_agent_framework.types import StreamEvent

        has_mcp = (
            not isinstance(self._config.mcp_server_config, dict)
            or bool(self._config.mcp_server_config.get("mcpServers"))
        )

        async def _stream_loop(mcp):
            tools = await list_tools(mcp) if mcp else []
            messages = list(history or [])
            messages.append(Message(role="user", content=user_message))

            for iteration in range(self._config.max_iterations):
                text_parts: list[str] = []
                thinking_parts: list[str] = []
                tool_call_events: list[StreamEvent] = []
                stop_reason: str | None = None

                async for event in self._llm.stream(
                    messages,
                    tools=tools,
                    system=self._config.system_prompt or None,
                ):
                    if event.type == "thinking":
                        thinking_parts.append(event.delta)
                        yield event
                    elif event.type == "text":
                        text_parts.append(event.delta)
                        yield event
                    elif event.type == "tool_call":
                        tool_call_events.append(event)
                    elif event.type == "done":
                        stop_reason = event.stop_reason

                content = "".join(text_parts) or None
                tool_calls = (
                    [ToolCall(id=e.tool_call_id or "", name=e.tool_name or "", arguments=e.tool_args or {})
                     for e in tool_call_events]
                    if tool_call_events else None
                )
                messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))

                if stop_reason != "tool_use" or not tool_call_events:
                    return

                for tc_event in tool_call_events:
                    tool_call = ToolCall(
                        id=tc_event.tool_call_id or "",
                        name=tc_event.tool_name or "",
                        arguments=tc_event.tool_args or {},
                    )
                    yield StreamEvent(
                        type="tool_start",
                        tool_name=tc_event.tool_name,
                        tool_call_id=tc_event.tool_call_id,
                        tool_args=tc_event.tool_args,
                    )
                    result = await call_tool(mcp, tool_call, context=context)
                    yield StreamEvent(
                        type="tool_end",
                        tool_name=tc_event.tool_name,
                        tool_call_id=tc_event.tool_call_id,
                        tool_result=result,
                    )
                    messages.append(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    ))
                    logger.debug("[stream:tool] %s → %s", tool_call.name, result[:80])

        if has_mcp:
            async with Client(self._config.mcp_server_config) as mcp:
                async for event in _stream_loop(mcp):
                    yield event
        else:
            async for event in _stream_loop(None):
                yield event

