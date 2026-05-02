"""
Shared tool utilities for all agent patterns.

Import these instead of copy-pasting _list_tools/_call_tool into every pattern.
"""
from __future__ import annotations

import logging
import time

from mcp_agent_framework.types import MCPTool, ToolCall

logger = logging.getLogger(__name__)


async def list_tools(mcp_client) -> list[MCPTool]:
    """List tools from an open fastmcp Client, converting to MCPTool canonical type."""
    raw = await mcp_client.list_tools()
    return [
        MCPTool(
            name=t.name,
            description=t.description or "",
            input_schema=t.inputSchema if isinstance(t.inputSchema, dict) else t.inputSchema.model_dump(),
        )
        for t in raw
    ]


async def call_tool(mcp_client, tool_call: ToolCall, context=None) -> str:
    """
    Execute one tool call and return the result as a string.

    If a RunContext is provided, emits TOOL_START, TOOL_END, and TOOL_ERROR
    trace events automatically — no extra wiring needed in the caller.
    """
    from mcp_agent_framework.observability.tracer import TraceEventType

    # Emit TOOL_START before execution
    if context:
        await context.emit(TraceEventType.TOOL_START, {
            "tool_name":  tool_call.name,
            "arguments":  tool_call.arguments,
        })

    t0 = time.monotonic()
    try:
        result = await mcp_client.call_tool(tool_call.name, tool_call.arguments)

        # MCP returns a list of content blocks; join them into a string
        parts: list[str] = []
        for block in (result if isinstance(result, list) else [result]):
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        text_result = "\n".join(parts)

        # Emit TOOL_END with latency
        if context:
            await context.emit(TraceEventType.TOOL_END, {
                "tool_name":  tool_call.name,
                "result":     text_result[:200],
                "elapsed_ms": round((time.monotonic() - t0) * 1000, 2),
            })

        return text_result

    except Exception as exc:
        error_msg = f"Tool '{tool_call.name}' failed: {exc}"
        logger.error(error_msg)

        # Emit TOOL_ERROR
        if context:
            await context.emit(TraceEventType.TOOL_ERROR, {
                "tool_name": tool_call.name,
                "error":     str(exc),
            })

        return error_msg
