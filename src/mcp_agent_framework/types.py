"""
Shared types used across all clients and patterns.

All LLM providers speak different formats internally. This module defines
a single canonical format. Each client adapter converts to/from it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StopReason(str, Enum):
    """Why did LLM stop the reasoning loop"""
    END_TURN   = "end_turn"    # model finished naturally
    TOOL_USE   = "tool_use"    # model wants to call a tool
    MAX_TOKENS = "max_tokens"  # hit token limit mid-response


@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""
    id: str               # unique ID for this call (used to match results back)
    name: str             # tool name as registered on the MCP server
    arguments: dict[str, Any]  # parsed JSON arguments


class Role(str, Enum):
    """Who has sent the message"""
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


@dataclass
class Message:
    """
    A single turn in the conversation.

    role values:
      "system"    - instructions given to the model before the conversation
      "user"      - human turn
      "assistant" - model turn (may include tool_calls)
      "tool"      - result of a tool call sent back to the model
    """
    role: Role
    content: str | None = None

    # present on role="assistant" when the model calls tools
    tool_calls: list[ToolCall] | None = None

    # present on role="tool" - ties result back to the original call
    tool_call_id: str | None = None

    # present on role="tool" - the tool name (required by some providers)
    name: str | None = None


@dataclass
class MCPTool:
    """
    A tool as returned by an MCP server's list_tools() call.
    This is the MCP-native format before conversion to any LLM format.
    """
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema object


@dataclass
class LLMResponse:
    """Normalised response from any LLM provider."""
    content: str | None           # text content (None when stop_reason is TOOL_USE)
    tool_calls: list[ToolCall] | None  # populated when stop_reason is TOOL_USE
    stop_reason: StopReason

    # raw usage stats - populated when available, None otherwise
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class AgentConfig:
    """
    Configuration passed when constructing any agent pattern.

    mcp_server_config:
        A FastMCP app instance (in-process, no network) or a dict in MCP
        multi-server format, e.g.:
        {"mcpServers": {"my_server": {"command": "python", "args": ["server.py"]}}}
        or for HTTP:
        {"mcpServers": {"my_server": {"url": "http://localhost:8001/mcp"}}}

    system_prompt:
        Instructions given to the LLM as the "system" role before any user messages.

    max_iterations:
        Safety cap on the agent loop. Prevents infinite tool-call cycles.
    """
    mcp_server_config: Any  # FastMCP instance or MCP server config dict
    system_prompt: str = ""
    max_iterations: int = 50
    extra: dict[str, Any] = field(default_factory=dict)  # pattern-specific options
