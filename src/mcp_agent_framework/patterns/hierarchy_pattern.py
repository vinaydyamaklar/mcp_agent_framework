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
         │ own LLM     │ │ own LLM     │ │ own LLM     │
         │ own MCP     │ │ own MCP     │ │ own MCP     │
         └─────────────┘ └─────────────┘ └─────────────┘

Real-world example:
  Parent = coordinator agent
  Child A = ResearchAgent (Nova) — does deep research, returns research.md
  Child B = WritingAgent  (Brown) — takes research, returns article draft

Each child can itself be a HierarchicalAgentPattern (unlimited depth).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from fastmcp import Client

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.patterns._tool_utils import call_tool, list_tools
from mcp_agent_framework.patterns.single_agent_loop import SingleAgentLoop
from mcp_agent_framework.types import AgentConfig, MCPTool, Message, StopReason, ToolCall

logger = logging.getLogger(__name__)

# Synthetic tool name prefix used when registering child agents as tools
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
    agent: SingleAgentLoop  # can also be HierarchicalAgentPattern — same interface


class HierarchicalAgentPattern:
    """
    Parent agent that can invoke child agents as tools.

    Child agents appear to the parent as regular tools named
    "call_agent__<name>". When the parent calls one, this class
    runs that child's full agent loop and returns the result.

    Usage:
        research_agent = SingleAgentLoop(
            llm_client=GeminiClient(),
            config=AgentConfig(
                mcp_server_config={"mcpServers": {"nova": {"url": "http://localhost:8001/mcp"}}},
                system_prompt="You are a research agent. Given a topic, return a detailed research summary.",
            ),
        )
        writing_agent = SingleAgentLoop(
            llm_client=AnthropicClient(),
            config=AgentConfig(
                mcp_server_config={"mcpServers": {"brown": {"url": "http://localhost:8002/mcp"}}},
                system_prompt="You are a writing agent. Given research notes, write a polished article.",
            ),
        )

        children = [
            ChildAgentConfig("research", "Run deep research on a topic and return a summary.", research_agent),
            ChildAgentConfig("writing",  "Write an article given research notes.",             writing_agent),
        ]
        parent_config = AgentConfig(
            mcp_server_config={},   # parent may have its own tools too, or empty
            system_prompt="You are a coordinator. Use research agent then writing agent.",
        )
        hierarchy = HierarchicalAgentPattern(
            llm_client=AnthropicClient(model="claude-opus-4-6"),
            config=parent_config,
            children=children,
        )
        result = await hierarchy.run("Write an article about RAG systems.")
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        config: AgentConfig,
        children: list[ChildAgentConfig],
    ):
        self._llm      = llm_client
        self._config   = config
        self._children = {c.name: c for c in children}

    async def run(self, user_message: str, history: list[Message] | None = None) -> str:
        """Run the parent agent. It may delegate to children as needed."""
        # Build child-agent tools (synthetic — not from any MCP server)
        child_tools = self._build_child_tools()

        # Optionally also connect to a parent MCP server for additional tools
        parent_mcp_tools: list[MCPTool] = []
        parent_mcp: Client | None = None

        has_parent_mcp = bool(
            isinstance(self._config.mcp_server_config, dict)
            and self._config.mcp_server_config.get("mcpServers")
        )

        messages = list(history or [])
        messages.append(Message(role="user", content=user_message))

        async def _run_loop(mcp: Client | None) -> str:
            all_tools = child_tools + (
                await list_tools(mcp) if mcp else []
            )

            for iteration in range(self._config.max_iterations):
                logger.debug("[hierarchy:parent] iteration %d", iteration + 1)

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

                for tool_call in response.tool_calls:
                    result = await self._dispatch(tool_call, mcp)
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
                    return msg.content
            return ""

        if has_parent_mcp:
            async with Client(self._config.mcp_server_config) as mcp:
                return await _run_loop(mcp)
        else:
            return await _run_loop(None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_child_tools(self) -> list[MCPTool]:
        """Expose each child agent as a tool to the parent LLM."""
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

    async def _dispatch(self, tool_call: ToolCall, parent_mcp: Client | None) -> str:
        """Route tool call to a child agent or parent MCP tool."""
        if tool_call.name.startswith(_CHILD_AGENT_TOOL_PREFIX):
            child_name = tool_call.name[len(_CHILD_AGENT_TOOL_PREFIX):]
            child = self._children.get(child_name)
            if not child:
                return f"Error: no child agent named '{child_name}'"
            task = tool_call.arguments.get("task", "")
            logger.debug("[hierarchy] delegating to child '%s': %s", child_name, task[:80])
            return await child.agent.run(task)

        # Fall through to parent MCP tools
        if parent_mcp:
            return await call_tool(parent_mcp, tool_call)

        return f"Error: no handler for tool '{tool_call.name}'"

