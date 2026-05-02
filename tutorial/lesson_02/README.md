# Lesson 2 — Types: The Shared Language

**Unit 1: Foundations**

---

## What you will learn

- Why a shared type system matters in a multi-provider framework
- Every type used across the entire framework, what it holds, and why
- How the message list is structured and why that structure is universal
- How to read a conversation history by hand

---

## The concept

Every piece of the framework — clients, patterns, memory, evaluation — needs to pass data between each other. Without a shared language, every client would invent its own format, every pattern would need to translate between formats, and swapping providers would require rewriting everything.

`types.py` is that shared language. It is small (< 100 lines) but everything in the framework speaks it.

### `StopReason` — why did the LLM stop?

```python
class StopReason(str, Enum):
    END_TURN   = "end_turn"    # model finished and gave an answer
    TOOL_USE   = "tool_use"    # model wants to call a tool
    MAX_TOKENS = "max_tokens"  # hit token limit mid-response
```

Every provider has a different internal name for these states. Anthropic calls them `"end_turn"` and `"tool_use"`. OpenAI calls them `"stop"` and `"tool_calls"`. Gemini has its own names. The client layer maps all of them to this one enum. Every pattern only checks `StopReason` — never a provider-specific string.

### `ToolCall` — what the LLM requests

```python
@dataclass
class ToolCall:
    id:        str              # unique call ID — used to match results back
    name:      str              # tool name as registered on the MCP server
    arguments: dict[str, Any]  # parsed JSON arguments
```

When an LLM decides to use a tool, it doesn't execute the tool itself — it sends back a `ToolCall` describing what it wants. Your code executes it and sends the result back. The `id` is critical: it links the result message back to the original request. Without it, the LLM doesn't know which result belongs to which call (especially when multiple tools are called in one turn).

### `Message` — a single turn in the conversation

```python
@dataclass
class Message:
    role:         str                    # "user" | "assistant" | "tool" | "system"
    content:      str | None = None      # the text content
    tool_calls:   list[ToolCall] | None  # set when role="assistant" + model used tools
    tool_call_id: str | None            # set when role="tool" — links to the ToolCall
    name:         str | None            # set when role="tool" — the tool name
```

The four roles and when they appear:

| Role | When used | Contains |
|------|-----------|----------|
| `"system"` | First message, instructions | `content` only |
| `"user"` | Human input | `content` only |
| `"assistant"` | Model response | `content` or `tool_calls` (or both) |
| `"tool"` | Tool execution result | `content` + `tool_call_id` + `name` |

A real conversation looks like this:

```
Message(role="system",    content="You are a helpful assistant.")
Message(role="user",      content="What files are in /tmp?")
Message(role="assistant", content=None,            tool_calls=[ToolCall(id="tc_1", name="list_files", arguments={"path": "/tmp"})])
Message(role="tool",      content="file1.txt\nfile2.log", tool_call_id="tc_1", name="list_files")
Message(role="assistant", content="The files in /tmp are: file1.txt, file2.log")
```

That list of 5 messages is the complete agent run. The loop built it one message at a time.

### `MCPTool` — a tool as advertised by the MCP server

```python
@dataclass
class MCPTool:
    name:         str
    description:  str
    input_schema: dict[str, Any]   # JSON Schema
```

Before the agent can use tools, it must know what tools exist. `mcp.list_tools()` returns these. The `input_schema` is a JSON Schema object describing what arguments the tool accepts — the LLM reads this to know what to put in `ToolCall.arguments`.

### `LLMResponse` — what comes back from any client

```python
@dataclass
class LLMResponse:
    content:       str | None             # text (None when stop_reason is TOOL_USE)
    tool_calls:    list[ToolCall] | None  # tool requests (None when stop_reason is END_TURN)
    stop_reason:   StopReason
    input_tokens:  int | None             # cost tracking
    output_tokens: int | None
    reasoning:     str | None             # model's chain-of-thought (Claude extended thinking)
```

The pattern is always: check `stop_reason` first, then read either `content` or `tool_calls`. `reasoning` is populated only when the client has `enable_thinking=True` — it is `None` otherwise.

### `StreamEvent` — one event from a streaming run

```python
@dataclass
class StreamEvent:
    type:         str                    # "thinking" | "text" | "tool_start" | "tool_end" | "done"
    delta:        str = ""              # incremental text or thinking token
    tool_name:    str | None = None
    tool_call_id: str | None = None
    tool_args:    dict | None = None    # on tool_start — the arguments
    tool_result:  str | None = None     # on tool_end — the result string
    stop_reason:  str | None = None     # on done — "end_turn" | "tool_use" | "max_tokens"
```

`StreamEvent` is what `run_stream()` yields on every pattern. Instead of waiting for the full answer, the caller receives one event per token — or per tool execution step. `"thinking"` and `"text"` events carry incremental `delta` strings. `"tool_start"` / `"tool_end"` bracket each MCP tool call.

### `AgentConfig` — how you configure any pattern

```python
@dataclass
class AgentConfig:
    mcp_server_config: Any        # FastMCP instance or MCP server config dict
    system_prompt:     str = ""
    max_iterations:    int = 50
    extra:             dict[str, Any] = field(default_factory=dict)
```

`mcp_server_config` accepts either:
- A `FastMCP` app instance (in-process, no network hop, best for tests and single-process apps)
- A config dict with connection info (subprocess or HTTP, for multi-process setups)

`extra` is a catch-all for pattern-specific settings that don't belong in the shared config.

---

## Why this design matters

### Swapping providers in one line

```python
# This:
agent = SingleAgentLoop(llm_client=AnthropicClient("claude-sonnet-4-6"), config=config)
# Becomes this:
agent = SingleAgentLoop(llm_client=OpenAIClient("gpt-4o"), config=config)
```

Both clients produce `LLMResponse`. The `SingleAgentLoop` only sees `LLMResponse`. No other changes needed.

### The message list is the protocol

There is no special "agent state" object. The agent's entire state at any point is the `list[Message]`. You can:
- Pass it between patterns
- Save it to a database and resume later
- Print it to debug exactly what happened
- Replay it to reproduce a bug

---

## Read this file

```
src/mcp_agent_framework/types.py
```

Read it completely. It is short. Notice:
- Every field has a comment explaining its purpose
- Optional fields use `| None` with a default of `None` (Python 3.10+ union syntax)
- `AgentConfig.extra` is a deliberate escape hatch — it avoids making `AgentConfig` a kitchen sink

---

## Run this

No example file for this lesson. Instead, open a Python REPL and build a conversation by hand:

```python
from mcp_agent_framework.types import Message, ToolCall, StopReason, LLMResponse

# Build a fake conversation manually
messages = [
    Message(role="system", content="You are a calculator assistant."),
    Message(role="user", content="What is 12 times 8?"),
    Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="tc_001", name="multiply", arguments={"a": 12, "b": 8})]
    ),
    Message(role="tool", content="96", tool_call_id="tc_001", name="multiply"),
    Message(role="assistant", content="12 times 8 is 96."),
]

# Walk through it like the loop does
for msg in messages:
    print(f"[{msg.role}]", msg.content or f"→ calls {[tc.name for tc in (msg.tool_calls or [])]}")
```

---

## Build this

Write a function `pretty_print_conversation(messages: list[Message]) -> None` that prints the conversation in a readable format:

```
[SYSTEM] You are a calculator assistant.
[USER]   What is 12 times 8?
[ASST]   → calls: multiply(a=12, b=8) [id=tc_001]
[TOOL]   multiply → 96
[ASST]   12 times 8 is 96.
```

This function will be your most-used debug tool for the rest of the curriculum. Build it well.

---

## Key terms

| Term | Meaning |
|------|---------|
| `StopReason` | Why the LLM stopped: done, needs tools, or hit token limit |
| `ToolCall` | The LLM's request to execute a tool — name + args + unique ID |
| `Message` | One turn in the conversation — user, assistant, tool result, or system |
| `LLMResponse` | Normalised output from any provider; includes `reasoning` when thinking is enabled |
| `StreamEvent` | One event from a streaming run — thinking token, text token, tool step, or done |
| `AgentConfig` | Config passed to any pattern |
| `MCPTool` | A tool's description as advertised by the MCP server |

---

## Connects to

- **Lesson 3** — Clients transform provider-specific responses into `LLMResponse`
- **Lesson 4** — MCP returns `MCPTool` objects that get passed to clients
- **Lesson 5** — The Single Agent Loop builds a `list[Message]` one by one
- **Every other lesson** — all patterns speak this type system

---

*Lesson 2 of 21 — Applied AI Engineering*
