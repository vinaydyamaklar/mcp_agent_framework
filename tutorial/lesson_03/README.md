# Lesson 3 — Clients: Talking to LLMs

**Unit 1: Foundations**

---

## What you will learn

- Why every LLM provider needs its own adapter
- What `BaseLLMClient` enforces and why
- How `AnthropicClient`, `OpenAIClient`, and `GeminiClient` translate between provider formats and the shared type system
- How structured output works differently on each provider — and how the framework hides that

---

## The concept

Here is the problem. You write an agent. It calls Anthropic's API. Six months later you want to try OpenAI's GPT-5. You open your code and discover that everything is tangled with Anthropic-specific objects: `anthropic.types.ContentBlock`, `anthropic.types.ToolUseBlock`, `response.stop_reason == "end_turn"`. Migrating means touching every file.

The client layer solves this with the **port-and-adapter pattern**:

```
Your code (patterns, registry, examples)
        │
        │  speaks only BaseLLMClient
        ▼
┌──────────────────────────────────────┐
│           BaseLLMClient              │  ← the port (interface)
└──────────────────────────────────────┘
        │                │               │
        ▼                ▼               ▼
AnthropicClient   OpenAIClient    GeminiClient   ← the adapters
```

Every adapter converts *into* and *out of* the shared `LLMResponse` / `Message` types from Lesson 2. Your patterns never import `anthropic`, `openai`, or `google-genai` directly.

---

## `BaseLLMClient` — the contract

```python
class BaseLLMClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools:    list[MCPTool] | None = None,
        system:   str | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    async def complete_structured(
        self,
        messages:       list[Message],
        response_model: type[BaseModel],
    ) -> dict[str, Any]: ...

    @abstractmethod
    def provider_name(self) -> str: ...

    async def stream(
        self,
        messages: list[Message],
        tools:    list[MCPTool] | None = None,
        system:   str | None = None,
    ) -> AsyncIterator[StreamEvent]: ...
```

Four methods. The first three are abstract — every client must implement them. `stream()` has a default implementation that falls back to `complete()` and yields the full response as a single event, so it works on any client without override. Override it in provider clients for real token-by-token streaming.

---

## Inside each client

### `AnthropicClient`

**Extended thinking** — Claude supports a reasoning mode where the model "thinks out loud" before answering. Enable it at construction time:

```python
client = AnthropicClient(
    enable_thinking=True,   # turn on reasoning blocks
    thinking_budget=8000,   # max tokens the model can spend thinking
)

# Non-streaming: reasoning lands in LLMResponse.reasoning
response = await client.complete(messages)
print(response.reasoning)  # "Let me consider... first I should..."
print(response.content)    # the actual answer

# Streaming: reasoning arrives as StreamEvent(type="thinking") events
async for event in client.stream(messages):
    if event.type == "thinking":
        print(event.delta, end="")  # live reasoning tokens
    elif event.type == "text":
        print(event.delta, end="")  # live response tokens
```

`enable_thinking=False` is the default. When disabled, `response.reasoning` is always `None`.

**Message encoding** — Anthropic's format is unique: each message has a `content` field that is a *list* of content blocks, not a plain string. A tool result is a content block with `type="tool_result"`.

```python
# What the framework sends:
Message(role="tool", content="96", tool_call_id="tc_001", name="multiply")

# What AnthropicClient converts it to:
{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tc_001", "content": "96"}]}
```

Note that Anthropic puts tool results in `role="user"` messages. That is an Anthropic quirk hidden by the adapter.

**Tool encoding** — tools become `ToolParam` objects with `input_schema`.

**Structured output** — uses `tool_choice={"type": "tool", "name": "structured_output"}`. Forces the model to respond by "calling" a fake tool whose schema is the Pydantic model. The result comes back as a `ToolUseBlock` with JSON arguments.

### `OpenAIClient`

**Message encoding** — cleaner. Tool results are `role="tool"` with `tool_call_id`. Matches the framework's `Message` format closely.

**Tool encoding** — `function` type tools with `parameters` (JSON Schema).

**Structured output** — `response_format={"type": "json_schema", "json_schema": {...}}`. The model returns raw JSON that you parse into the Pydantic model.

**Compatibility** — because OpenAI's API is widely used as a standard, this client also works with:
- **Grok** — `base_url="https://api.x.ai/v1"` + X.AI API key
- **Ollama** — `base_url="http://localhost:11434/v1"` + any key string
- **Together AI, Anyscale, Fireworks** — any OpenAI-compatible endpoint

### `GeminiClient`

**Message encoding** — Gemini uses `"user"` and `"model"` roles (not `"assistant"`). Tool calls are `Part` objects. The client maps `"assistant"` → `"model"` on encode and back on decode.

**Tool encoding** — `FunctionDeclaration` with `parameters_json_schema`. Important April 2026 detail: pass the JSON Schema dict directly — the old `types.Schema` conversion is gone.

**Structured output** — `response_schema=FunctionDeclaration(...)`. The model fills a schema.

---

## The encoding/decoding pattern

Every client has the same internal structure:

```python
async def complete(self, messages, tools, system) -> LLMResponse:
    # 1. Encode: framework types → provider format
    provider_messages = [self._encode_message(m) for m in messages]
    provider_tools    = [self._encode_tool(t) for t in (tools or [])]

    # 2. Call the provider API
    raw_response = await self._provider_client.create(
        model=self._model,
        messages=provider_messages,
        tools=provider_tools,
        system=system,
    )

    # 3. Decode: provider format → framework types
    return self._decode_response(raw_response)
```

When debugging a client issue, these three steps tell you exactly where to look: encoding, the API call, or decoding.

---

## `schema_utils.py` — shared schema helper

```
src/mcp_agent_framework/clients/schema_utils.py
```

`schema_from(MyPydanticModel)` generates a JSON Schema dict from a Pydantic model. All three clients use this to convert your `BaseModel` subclass into the format their structured output mechanism requires. You rarely call this directly — `complete_structured()` calls it for you.

---

## Error handling

**`StructuredOutputError`** — raised when a provider fails to return valid structured output. Can happen when:
- The schema is too complex
- The model returns malformed JSON
- The provider's structured output feature is rate-limited

All three clients guard against missing `choices[0]` / empty responses and raise this error with a clear message rather than a cryptic `IndexError`.

---

## Read these files

```
src/mcp_agent_framework/clients/base_client.py          ← the interface
src/mcp_agent_framework/clients/anthropic_client.py     ← study _encode_message() and _decode_response()
src/mcp_agent_framework/clients/openai_client.py        ← compare with Anthropic's encoding
src/mcp_agent_framework/clients/schema_utils.py         ← schema_from()
```

Focus on `_encode_message()` in `anthropic_client.py`. Trace through what happens to a `role="tool"` message — that is the trickiest case.

---

## Run this

```bash
python examples/02_structured_output.py
```

This example uses `complete_structured()` with a Pydantic model. Watch the output — the model returns structured data, not free text.

---

## Build this

Write a `compare_providers.py` that asks the same question to both Anthropic and OpenAI, measures wall-clock time, and prints both answers side-by-side:

```python
import asyncio, time
from mcp_agent_framework import AnthropicClient, OpenAIClient
from mcp_agent_framework.types import Message

async def ask(client, question: str) -> tuple[str, float]:
    start = time.monotonic()
    resp = await client.complete([Message(role="user", content=question)])
    elapsed = time.monotonic() - start
    return resp.content, elapsed

async def main():
    question = "Explain cosine similarity in two sentences."
    anthropic = AnthropicClient("claude-haiku-4-5-20251001")
    openai    = OpenAIClient("gpt-4o-mini")

    (ans_a, t_a), (ans_o, t_o) = await asyncio.gather(
        ask(anthropic, question),
        ask(openai, question),
    )
    print(f"Anthropic ({t_a:.2f}s): {ans_a}")
    print(f"OpenAI   ({t_o:.2f}s): {ans_o}")

asyncio.run(main())
```

Notice: your calling code only uses `BaseLLMClient`-compatible methods. Swapping providers is just the constructor argument.

---

## Key terms

| Term | Meaning |
|------|---------|
| Port-and-adapter | Design pattern: define an interface (port), implement per-provider (adapter) |
| `BaseLLMClient` | The interface every client must implement |
| `_encode_message` | Converts `Message` → provider-specific format |
| `_decode_response` | Converts provider response → `LLMResponse` |
| `complete_structured` | Forces a Pydantic model response from any provider |
| `stream()` | Yields `StreamEvent` objects token by token; falls back to `complete()` if not overridden |
| `enable_thinking` | `AnthropicClient` flag that enables Claude's extended reasoning mode |
| `thinking_budget` | Max tokens Claude may spend on reasoning before producing the answer |
| `LLMResponse.reasoning` | The model's chain-of-thought; populated only when `enable_thinking=True` |
| `StructuredOutputError` | Raised when the provider fails to return valid structured data |

---

## Connects to

- **Lesson 4** — MCP gives you the `MCPTool` list you pass to `client.complete()`
- **Lesson 5** — `SingleAgentLoop` calls `client.complete()` in a loop
- **Lesson 6** — Tool calling: the flow from `LLMResponse.tool_calls` back to tool results
- **Lesson 11** — `LLMEvaluator` calls `client.complete()` to score output

---

*Lesson 3 of 21 — Applied AI Engineering*
