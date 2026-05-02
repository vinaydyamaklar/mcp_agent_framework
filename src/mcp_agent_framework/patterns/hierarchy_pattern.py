"""
Pattern: Hierarchical Multi-Agent

A parent agent can delegate entire sub-tasks to child agents. Each child
is itself a full SingleAgentLoop with its own LLM, tools, and system prompt.
The parent calls a child by name as if it were a tool.

Use this when: sub-problems are complex enough to need their own reasoning
loop, not just a single tool call. The parent doesn't need to know HOW a
child does its work — just what to ask it and what it returns.

                      ┌────────────────┐
    user message  →   │  Parent Agent  │  has regular MCP tools
                      │     (LLM)      │  + "call_child_agent" tools
                      └───────┬────────┘
                              │ delegate sub-task
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
         ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
         │ Child A     │ │ Child B     │ │ Child C     │
         │ (full loop) │ │ (full loop) │ │ (full loop) │
         │ own LLM     │ │ own MCP     │ │ own MCP     │
         └─────────────┘ └─────────────┘ └─────────────┘

Each child can itself be a HierarchicalAgentPattern (unlimited depth).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from fastmcp import Client

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.observability.run_context import RunContext
from mcp_agent_framework.observability.tracer import TraceEventType
from mcp_agent_framework.patterns._tool_utils import call_tool, list_tools
from mcp_agent_framework.patterns.single_agent_loop import SingleAgentLoop
from mcp_agent_framework.types import AgentConfig, MCPTool, Message, StopReason, ToolCall

logger = logging.getLogger(__name__)

_CHILD_AGENT_TOOL_PREFIX = "call_agent__"


@dataclass
class ChildAgentConfig:
    """
    Registers a child agent that the parent can delegate to.

    name:        Used as the tool name: call_agent__<name>
    description: Shown to the parent LLM so it knows when to call this child.
    agent:       A SingleAgentLoop (or HierarchicalAgentPattern) instance.
    """
    name: str
    description: str
    agent: SingleAgentLoop


class HierarchicalAgentPattern:

    def __init__(
        self,
        llm_client: BaseLLMClient,
        config: AgentConfig,
        children: list[ChildAgentConfig],
    ):
        self._llm      = llm_client
        self._config   = config
        self._children = {c.name: c for c in children}

    async def run(
        self,
        user_message: str,
        history: list[Message] | None = None,
        context: RunContext | None = None,
    ) -> str:
        t_start = time.monotonic()

        if context:
            await context.emit(TraceEventType.PATTERN_START, {
                "pattern_name": "HierarchicalAgentPattern",
                "user_message": user_message,
                "model":        self._llm.provider_name(),
                "children":     list(self._children.keys()),
            })

        child_tools = self._build_child_tools()
        messages = list(history or [])
        messages.append(Message(role="user", content=user_message))

        async def _run_loop(mcp: Client | None) -> str:
            all_tools = child_tools + (await list_tools(mcp) if mcp else [])

            for iteration in range(self._config.max_iterations):
                logger.debug("[hierarchy:parent] iteration %d", iteration + 1)

                if context:
                    await context.emit(TraceEventType.LLM_START, {
                        "model":         self._llm.provider_name(),
                        "iteration":     iteration + 1,
                        "message_count": len(messages),
                        "tool_count":    len(all_tools),
                    })

                t_llm = time.monotonic()
                response = await self._llm.complete(
                    messages=messages,
                    tools=all_tools,
                    system=self._config.system_prompt or None,
                )

                if context:
                    await context.emit(TraceEventType.LLM_END, {
                        "model":         self._llm.provider_name(),
                        "stop_reason":   response.stop_reason,
                        "elapsed_ms":    round((time.monotonic() - t_llm) * 1000, 2),
                        "input_tokens":  response.input_tokens,
                        "output_tokens": response.output_tokens,
                    })

                messages.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                if response.stop_reason != StopReason.TOOL_USE or not response.tool_calls:
                    final = response.content or ""
                    if context:
                        await context.emit(TraceEventType.PATTERN_END, {
                            "pattern_name": "HierarchicalAgentPattern",
                            "iterations":   iteration + 1,
                            "result":       final[:200],
                            "elapsed_ms":   round((time.monotonic() - t_start) * 1000, 2),
                        })
                    return final

                for tool_call in response.tool_calls:
                    result = await self._dispatch(tool_call, mcp, context=context)
                    messages.append(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    ))
                    logger.debug("[parent tool] %s → %s", tool_call.name, result[:120])

            logger.warning("Hit max_iterations=%d", self._config.max_iterations)
            for msg in reversed(messages):
                if msg.role == "assistant" and msg.content:
                    if context:
                        await context.emit(TraceEventType.PATTERN_END, {
                            "pattern_name":     "HierarchicalAgentPattern",
                            "iterations":       self._config.max_iterations,
                            "result":           msg.content[:200],
                            "elapsed_ms":       round((time.monotonic() - t_start) * 1000, 2),
                            "hit_max_iterations": True,
                        })
                    return msg.content
            return ""

        has_parent_mcp = (
                not isinstance(self._config.mcp_server_config, dict)
                or bool(self._config.mcp_server_config.get("mcpServers"))
        )

        if has_parent_mcp:
            async with Client(self._config.mcp_server_config) as mcp:
                return await _run_loop(mcp)
        else:
            return await _run_loop(None)

    async def run_stream(
        self,
        user_message: str,
        history: list[Message] | None = None,
        context: RunContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        from mcp_agent_framework.types import StreamEvent, ToolCall as ToolCallType

        child_tools = self._build_child_tools()

        has_parent_mcp = (
            not isinstance(self._config.mcp_server_config, dict)
            or bool(self._config.mcp_server_config.get("mcpServers"))
        )

        async def _stream_loop(mcp):
            all_tools = child_tools + (await list_tools(mcp) if mcp else [])
            messages = list(history or [])
            messages.append(Message(role="user", content=user_message))

            for iteration in range(self._config.max_iterations):
                text_parts: list[str] = []
                thinking_parts: list[str] = []
                tool_call_events: list[StreamEvent] = []
                stop_reason: str | None = None

                async for event in self._llm.stream(
                    messages,
                    tools=all_tools,
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
                tool_calls_list = (
                    [ToolCallType(id=e.tool_call_id or "", name=e.tool_name or "", arguments=e.tool_args or {})
                     for e in tool_call_events]
                    if tool_call_events else None
                )
                messages.append(Message(role="assistant", content=content, tool_calls=tool_calls_list))

                if stop_reason != "tool_use" or not tool_call_events:
                    return

                for tc_event in tool_call_events:
                    tool_call = ToolCallType(
                        id=tc_event.tool_call_id or "",
                        name=tc_event.tool_name or "",
                        arguments=tc_event.tool_args or {},
                    )
                    yield StreamEvent(type="tool_start", tool_name=tc_event.tool_name, tool_call_id=tc_event.tool_call_id, tool_args=tc_event.tool_args)
                    result = await self._dispatch(tool_call, mcp, context=context)
                    yield StreamEvent(type="tool_end", tool_name=tc_event.tool_name, tool_call_id=tc_event.tool_call_id, tool_result=result)
                    messages.append(Message(role="tool", content=result, tool_call_id=tool_call.id, name=tool_call.name))

        if has_parent_mcp:
            async with Client(self._config.mcp_server_config) as mcp:
                async for event in _stream_loop(mcp):
                    yield event
        else:
            async for event in _stream_loop(None):
                yield event

    def _build_child_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                name=f"{_CHILD_AGENT_TOOL_PREFIX}{name}",
                description=child.description,
                input_schema={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task or question to give to this agent.",
                        }
                    },
                    "required": ["task"],
                },
            )
            for name, child in self._children.items()
        ]

    async def _dispatch(
        self,
        tool_call: ToolCall,
        parent_mcp: Client | None,
        context: RunContext | None = None,
    ) -> str:
        if tool_call.name.startswith(_CHILD_AGENT_TOOL_PREFIX):
            child_name = tool_call.name[len(_CHILD_AGENT_TOOL_PREFIX):]
            child = self._children.get(child_name)
            if not child:
                return f"Error: no child agent named '{child_name}'"
            task = tool_call.arguments.get("task", "")
            logger.debug("[hierarchy] delegating to child '%s': %s", child_name, task[:80])
            child_context = context.child() if context else None
            return await child.agent.run(task, context=child_context)

        if parent_mcp:
            return await call_tool(parent_mcp, tool_call, context=context)

        return f"Error: no handler for tool '{tool_call.name}'"