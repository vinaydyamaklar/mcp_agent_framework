"""
Pattern: Human-in-the-Loop

The agent pauses before executing certain tool calls and asks a human
for approval. The human can approve, reject, or modify the action.

Use this when: the agent may take irreversible or sensitive actions
(deleting files, sending emails, making API calls with side effects)
and you want a human checkpoint before those actions execute.

                    ┌─────────────┐
  user message  →   │     LLM     │
                    └──────┬──────┘
                           │ wants to call a tool
                           ▼
                    ┌─────────────┐
                    │  approval?  │  ← your callback decides yes/no
                    └──────┬──────┘
                    approved│      rejected
                           ▼            ▼
                    ┌─────────────┐  ┌──────────────────────┐
                    │  MCP Tool   │  │  skip + tell the LLM  │
                    └─────────────┘  └──────────────────────┘

The approval callback receives the tool name and arguments and returns:
  - True / approved string  → execute the tool
  - False / rejection reason → skip the tool, tell the LLM why
  - modified arguments dict  → execute with different arguments
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from fastmcp import Client

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.patterns._tool_utils import call_tool, list_tools
from mcp_agent_framework.types import AgentConfig, Message, StopReason, ToolCall

logger = logging.getLogger(__name__)

# Type for the approval callback
# Returns: True (approve) | False (reject) | str (rejection reason) | dict (modified args)
ApprovalCallback = Callable[[str, dict[str, Any]], Awaitable[bool | str | dict[str, Any]]]


def _default_cli_approval(tool_name: str, arguments: dict[str, Any]) -> bool | str:
    """
    Built-in CLI approval — prints the proposed action and waits for y/n.
    Use this for local scripts. Replace with a custom callback for web apps.
    """
    print(f"\n[HUMAN APPROVAL REQUIRED]")
    print(f"  Tool:      {tool_name}")
    print(f"  Arguments: {arguments}")
    answer = input("  Approve? [y/n]: ").strip().lower()
    if answer == "y":
        return True
    reason = input("  Rejection reason (optional): ").strip()
    return reason or "User rejected this action."


class HumanInLoopPattern:
    """
    SingleAgentLoop with a human approval gate before tool execution.

    By default uses a CLI prompt. Pass a custom `approval_callback` for
    async web UIs, Slack bots, or any other interface.

    Usage — CLI (local scripts):
        agent = HumanInLoopPattern(
            llm_client=AnthropicClient(),
            config=AgentConfig(
                mcp_server_config={"mcpServers": {"files": {"command": "python", "args": ["server.py"]}}},
                system_prompt="You are a file management assistant.",
            ),
            # uses built-in CLI approval by default
            requires_approval={"delete_file", "overwrite_file"},  # only these tools need approval
        )
        result = await agent.run("Clean up all .tmp files in /workspace")

    Usage — custom async callback:
        async def my_approval(tool_name: str, args: dict) -> bool | str:
            # post to Slack, wait for reaction, return True/False
            return await slack_approval_flow(tool_name, args)

        agent = HumanInLoopPattern(
            llm_client=...,
            config=...,
            approval_callback=my_approval,
        )
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        config: AgentConfig,
        approval_callback: ApprovalCallback | None = None,
        requires_approval: set[str] | None = None,  # None = ALL tools require approval
    ):
        self._llm      = llm_client
        self._config   = config
        self._callback = approval_callback or self._wrap_sync(_default_cli_approval)
        # If requires_approval is None, every tool needs approval.
        # Pass an empty set to skip approval entirely (equivalent to SingleAgentLoop).
        self._requires_approval = requires_approval  # None means "all tools"

    async def run(self, user_message: str, history: list[Message] | None = None) -> str:
        """Run the agent loop with human approval gates."""
        async with Client(self._config.mcp_server_config) as mcp:
            tools    = await list_tools(mcp)
            messages = list(history or [])
            messages.append(Message(role="user", content=user_message))

            for iteration in range(self._config.max_iterations):
                logger.debug("[human-in-loop] iteration %d", iteration + 1)

                response = await self._llm.complete(
                    messages=messages,
                    tools=tools,
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
                    result = await self._execute_with_approval(mcp, tool_call)
                    messages.append(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    ))

            logger.warning("Hit max_iterations=%d", self._config.max_iterations)
            for msg in reversed(messages):
                if msg.role == "assistant" and msg.content:
                    return msg.content
            return ""

    # ------------------------------------------------------------------
    # Approval logic
    # ------------------------------------------------------------------

    async def _execute_with_approval(self, mcp: Client, tool_call: ToolCall) -> str:
        needs_approval = (
            self._requires_approval is None            # all tools need approval
            or tool_call.name in self._requires_approval   # this specific tool needs it
        )

        if not needs_approval:
            return await call_tool(mcp, tool_call)

        decision = await self._callback(tool_call.name, tool_call.arguments)

        if decision is True:
            logger.info("[approval] APPROVED: %s", tool_call.name)
            return await call_tool(mcp, tool_call)

        if isinstance(decision, dict):
            # Human modified the arguments — use the modified version
            logger.info("[approval] APPROVED with modified args: %s", tool_call.name)
            modified = ToolCall(id=tool_call.id, name=tool_call.name, arguments=decision)
            return await call_tool(mcp, modified)

        # Rejected — tell the LLM clearly so it can try a different approach
        reason = decision if isinstance(decision, str) else "Action rejected by user."
        logger.info("[approval] REJECTED: %s — %s", tool_call.name, reason)
        return f"Action '{tool_call.name}' was rejected by the user. Reason: {reason}"

    @staticmethod
    def _wrap_sync(fn: Callable) -> ApprovalCallback:
        """Wrap a synchronous approval function so it works in async context."""
        async def wrapper(tool_name: str, arguments: dict[str, Any]) -> bool | str:
            return fn(tool_name, arguments)
        return wrapper
