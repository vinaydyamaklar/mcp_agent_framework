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
from dataclasses import dataclass
from typing import Any

from fastmcp import Client

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.patterns._tool_utils import call_tool, list_tools
from mcp_agent_framework.types import AgentConfig, Message, StopReason

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """
    Configuration for one worker MCP server.

    name:          Human-readable label used in logs.
    server_config: MCP server config for this worker only, e.g.:
                   {"mcpServers": {"nova": {"url": "http://localhost:8001/mcp"}}}
    """
    name: str
    server_config: dict[str, Any]


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

    async def run(self, user_message: str, history: list[Message] | None = None) -> str:
        """Run the orchestrator loop across all workers."""
        # Open connections to all workers simultaneously
        worker_clients: dict[str, Client] = {}
        tool_to_worker: dict[str, str] = {}   # tool_name → worker_name
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

                response = await self._llm.complete(
                    messages=messages,
                    tools=all_tools,
                    system=self._config.system_prompt or None,
                )

                messages.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                if response.stop_reason != StopReason.TOOL_USE or not response.tool_calls:
                    return response.content or ""

                # Route each tool call to the correct worker and execute in parallel.
                async def _execute_one(tool_call):
                    worker_name = tool_to_worker.get(tool_call.name)
                    if not worker_name:
                        return tool_call, f"Error: no worker registered for tool '{tool_call.name}'"
                    mcp = worker_clients[worker_name]
                    result = await call_tool(mcp, tool_call)
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
                    return msg.content
            return ""

        finally:
            await stack.__aexit__(None, None, None)

