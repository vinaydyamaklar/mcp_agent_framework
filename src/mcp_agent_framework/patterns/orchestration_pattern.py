"""
Pattern: Orchestration (Orchestrator → Workers)

One orchestrator LLM has access to tools from MULTIPLE MCP servers simultaneously.
Each MCP server is a "worker" — a specialised agent or service. The orchestrator
decides which worker's tools to call and in what order.

Use this when: you have distinct domains of capability (e.g. research tools vs
writing tools) and one LLM should coordinate across all of them.

                        ┌─────────────────┐
      user message  →   │  Orchestrator   │  (sees ALL tools from all workers)
                        │      LLM        │
                        └────────┬────────┘
                                 │ calls tools from any worker
               ┌─────────────────┼─────────────────┐
               ▼                 ▼                 ▼
       ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
       │   Worker A   │  │   Worker B   │  │   Worker C   │
       │  MCP Server  │  │  MCP Server  │  │  MCP Server  │
       │ (e.g. Nova)  │  │ (e.g. Brown) │  │  (e.g. DB)   │
       └──────────────┘  └──────────────┘  └──────────────┘

Real-world example from this course:
  Orchestrator LLM + Nova (research tools) + Brown (writing tools)
  → LLM runs research phase using Nova tools, then writing phase using Brown tools.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from fastmcp import Client

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.observability.run_context import RunContext
from mcp_agent_framework.observability.tracer import TraceEventType
from mcp_agent_framework.patterns._tool_utils import call_tool, list_tools
from mcp_agent_framework.types import AgentConfig, Message, StopReason, ToolCall

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """
    Configuration for one worker MCP server.

    name:          Human-readable label used in logs.
    server_config: MCP server config for this worker only, e.g.:
                   {"mcpServers": {"nova": {"url": "http://localhost:8001/mcp"}}}
                   Also accepts a FastMCP instance for in-process workers.
    """
    name: str
    server_config: Any


class OrchestratorWorkerPattern:
    """
    Orchestrator LLM with access to multiple worker MCP servers.

    All worker tools are exposed to the orchestrator at once. The LLM
    decides which tools to call; the pattern routes each call to the
    correct worker.

    Usage:
        workers = [
            WorkerConfig("nova",  {"mcpServers": {"nova":  {"url": "http://localhost:8001/mcp"}}}),
            WorkerConfig("brown", {"mcpServers": {"brown": {"url": "http://localhost:8002/mcp"}}}),
        ]
        config = AgentConfig(
            mcp_server_config={},   # not used directly; workers supply their own configs
            system_prompt="You are a research and writing agent...",
        )
        agent = OrchestratorWorkerPattern(
            llm_client=AnthropicClient(),
            config=config,
            workers=workers,
        )
        result = await agent.run("Write an article about MCP servers.")
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        config: AgentConfig,
        workers: list[WorkerConfig],
    ):
        self._llm     = llm_client
        self._config  = config
        self._workers = workers

    async def run(
        self,
        user_message: str,
        history: list[Message] | None = None,
        context: RunContext | None = None,
    ) -> str:
        """Run the orchestrator loop across all workers."""
        t_start = time.monotonic()

        if context:
            await context.emit(TraceEventType.PATTERN_START, {
                "pattern_name": "OrchestratorWorkerPattern",
                "user_message": user_message,
                "model":        self._llm.provider_name(),
                "workers":      [w.name for w in self._workers],
            })

        worker_clients: dict[str, Client] = {}
        tool_to_worker: dict[str, str] = {}
        all_tools: list = []

        stack = contextlib.AsyncExitStack()

        try:
            await stack.__aenter__()
            for worker in self._workers:
                client = await stack.enter_async_context(Client(worker.server_config))
                worker_clients[worker.name] = client
                tools = await list_tools(client)
                for t in tools:
                    tool_to_worker[t.name] = worker.name
                all_tools.extend(tools)
                logger.debug("[orchestrator] worker '%s' provides %d tools", worker.name, len(tools))

            messages = list(history or [])
            messages.append(Message(role="user", content=user_message))

            for iteration in range(self._config.max_iterations):
                logger.debug("[orchestrator] iteration %d", iteration + 1)

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
                            "pattern_name": "OrchestratorWorkerPattern",
                            "iterations":   iteration + 1,
                            "result":       final[:200],
                            "elapsed_ms":   round((time.monotonic() - t_start) * 1000, 2),
                        })
                    return final

                # Route each tool call to the correct worker and execute in parallel.
                async def _execute_one(tool_call):
                    worker_name = tool_to_worker.get(tool_call.name)
                    if not worker_name:
                        return tool_call, f"Error: no worker registered for tool '{tool_call.name}'"
                    mcp = worker_clients[worker_name]
                    result = await call_tool(mcp, tool_call, context=context)
                    logger.debug("[worker:%s] %s → %s", worker_name, tool_call.name, result[:120])
                    return tool_call, result

                tool_results = await asyncio.gather(
                    *[_execute_one(tc) for tc in response.tool_calls]
                )
                for tc, result in tool_results:
                    messages.append(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tc.id,
                        name=tc.name,
                    ))

            logger.warning("Hit max_iterations=%d", self._config.max_iterations)
            for msg in reversed(messages):
                if msg.role == "assistant" and msg.content:
                    if context:
                        await context.emit(TraceEventType.PATTERN_END, {
                            "pattern_name":       "OrchestratorWorkerPattern",
                            "iterations":         self._config.max_iterations,
                            "result":             msg.content[:200],
                            "elapsed_ms":         round((time.monotonic() - t_start) * 1000, 2),
                            "hit_max_iterations": True,
                        })
                    return msg.content
            return ""

        finally:
            await stack.__aexit__(None, None, None)

    async def run_stream(
        self,
        user_message: str,
        history: list[Message] | None = None,
        context: RunContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream orchestrator events. Tool calls are routed to workers mid-stream."""
        from mcp_agent_framework.types import StreamEvent, ToolCall

        worker_clients: dict[str, Client] = {}
        tool_to_worker: dict[str, str] = {}
        all_tools: list = []

        stack = contextlib.AsyncExitStack()
        try:
            await stack.__aenter__()
            for worker in self._workers:
                client = await stack.enter_async_context(Client(worker.server_config))
                worker_clients[worker.name] = client
                tools = await list_tools(client)
                for t in tools:
                    tool_to_worker[t.name] = worker.name
                all_tools.extend(tools)

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
                tool_calls = (
                    [ToolCall(id=e.tool_call_id or "", name=e.tool_name or "", arguments=e.tool_args or {})
                     for e in tool_call_events]
                    if tool_call_events else None
                )
                messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))

                if stop_reason != "tool_use" or not tool_call_events:
                    return

                for tc_event in tool_call_events:
                    worker_name = tool_to_worker.get(tc_event.tool_name or "")
                    if not worker_name:
                        result = f"Error: no worker for tool '{tc_event.tool_name}'"
                    else:
                        tool_call = ToolCall(
                            id=tc_event.tool_call_id or "",
                            name=tc_event.tool_name or "",
                            arguments=tc_event.tool_args or {},
                        )
                        yield StreamEvent(type="tool_start", tool_name=tc_event.tool_name, tool_call_id=tc_event.tool_call_id, tool_args=tc_event.tool_args)
                        mcp = worker_clients[worker_name]
                        result = await call_tool(mcp, tool_call, context=context)
                        yield StreamEvent(type="tool_end", tool_name=tc_event.tool_name, tool_call_id=tc_event.tool_call_id, tool_result=result)
                    messages.append(Message(role="tool", content=result, tool_call_id=tc_event.tool_call_id or "", name=tc_event.tool_name or ""))
        finally:
            await stack.__aexit__(None, None, None)
