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
import time
from typing import Any, AsyncIterator, Awaitable, Callable

from fastmcp import Client

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.observability.run_context import RunContext
from mcp_agent_framework.observability.tracer import TraceEventType
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

    async def run(
        self,
        user_message: str,
        history: list[Message] | None = None,
        context: RunContext | None = None,
    ) -> str:
        """Run the agent loop with human approval gates."""
        t_start = time.monotonic()

        if context:
            await context.emit(TraceEventType.PATTERN_START, {
                "pattern_name": "HumanInLoopPattern",
                "user_message": user_message,
                "model":        self._llm.provider_name(),
                "requires_approval": (
                    list(self._requires_approval)
                    if self._requires_approval is not None
                    else "all"
                ),
            })

        has_mcp = (
            not isinstance(self._config.mcp_server_config, dict)
            or bool(self._config.mcp_server_config.get("mcpServers"))
        )

        async def _run(mcp: Client | None) -> str:
            tools    = await list_tools(mcp) if mcp else []
            messages = list(history or [])
            messages.append(Message(role="user", content=user_message))

            for iteration in range(self._config.max_iterations):
                logger.debug("[human-in-loop] iteration %d", iteration + 1)

                if context:
                    await context.emit(TraceEventType.LLM_START, {
                        "model":         self._llm.provider_name(),
                        "iteration":     iteration + 1,
                        "message_count": len(messages),
                        "tool_count":    len(tools),
                    })

                t_llm = time.monotonic()
                response = await self._llm.complete(
                    messages=messages,
                    tools=tools,
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
                            "pattern_name": "HumanInLoopPattern",
                            "iterations":   iteration + 1,
                            "result":       final[:200],
                            "elapsed_ms":   round((time.monotonic() - t_start) * 1000, 2),
                        })
                    return final

                for tool_call in response.tool_calls:
                    result = await self._execute_with_approval(mcp, tool_call, context=context)
                    messages.append(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    ))

            logger.warning("Hit max_iterations=%d", self._config.max_iterations)
            for msg in reversed(messages):
                if msg.role == "assistant" and msg.content:
                    if context:
                        await context.emit(TraceEventType.PATTERN_END, {
                            "pattern_name":       "HumanInLoopPattern",
                            "iterations":         self._config.max_iterations,
                            "result":             msg.content[:200],
                            "elapsed_ms":         round((time.monotonic() - t_start) * 1000, 2),
                            "hit_max_iterations": True,
                        })
                    return msg.content
            return ""

        if has_mcp:
            async with Client(self._config.mcp_server_config) as mcp:
                return await _run(mcp)
        else:
            return await _run(None)

    async def run_stream(
        self,
        user_message: str,
        history: list[Message] | None = None,
        context: RunContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        from mcp_agent_framework.types import StreamEvent, ToolCall

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
                tool_calls_list = (
                    [ToolCall(id=e.tool_call_id or "", name=e.tool_name or "", arguments=e.tool_args or {})
                     for e in tool_call_events]
                    if tool_call_events else None
                )
                messages.append(Message(role="assistant", content=content, tool_calls=tool_calls_list))

                if stop_reason != "tool_use" or not tool_call_events:
                    return

                for tc_event in tool_call_events:
                    tool_call = ToolCall(
                        id=tc_event.tool_call_id or "",
                        name=tc_event.tool_name or "",
                        arguments=tc_event.tool_args or {},
                    )
                    yield StreamEvent(type="tool_start", tool_name=tc_event.tool_name, tool_call_id=tc_event.tool_call_id, tool_args=tc_event.tool_args)
                    result = await self._execute_with_approval(mcp, tool_call, context=context)
                    yield StreamEvent(type="tool_end", tool_name=tc_event.tool_name, tool_call_id=tc_event.tool_call_id, tool_result=result)
                    messages.append(Message(role="tool", content=result, tool_call_id=tool_call.id, name=tool_call.name))

        if has_mcp:
            async with Client(self._config.mcp_server_config) as mcp:
                async for event in _stream_loop(mcp):
                    yield event
        else:
            async for event in _stream_loop(None):
                yield event

    # ------------------------------------------------------------------
    # Approval logic
    # ------------------------------------------------------------------

    async def _execute_with_approval(
        self,
        mcp: Client | None,
        tool_call: ToolCall,
        context: RunContext | None = None,
    ) -> str:
        needs_approval = (
            self._requires_approval is None            # all tools need approval
            or tool_call.name in self._requires_approval   # this specific tool needs it
        )

        if not needs_approval:
            return await call_tool(mcp, tool_call, context=context)

        decision = await self._callback(tool_call.name, tool_call.arguments)

        if decision is True:
            logger.info("[approval] APPROVED: %s", tool_call.name)
            return await call_tool(mcp, tool_call, context=context)

        if isinstance(decision, dict):
            # Human modified the arguments — use the modified version
            logger.info("[approval] APPROVED with modified args: %s", tool_call.name)
            modified = ToolCall(id=tool_call.id, name=tool_call.name, arguments=decision)
            return await call_tool(mcp, modified, context=context)

        # Rejected — tell the LLM clearly so it can try a different approach
        reason = decision if isinstance(decision, str) else "Action rejected by user."
        logger.info("[approval] REJECTED: %s, %s", tool_call.name, reason)
        return f"Action '{tool_call.name}' was rejected by the user. Reason: {reason}"

    @staticmethod
    def _wrap_sync(fn: Callable) -> ApprovalCallback:
        """Wrap a synchronous approval function so it works in async context."""
        async def wrapper(tool_name: str, arguments: dict[str, Any]) -> bool | str:
            return fn(tool_name, arguments)
        return wrapper
