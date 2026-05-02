# Lesson 5 — The Single Agent Loop

**Unit 2: Core Patterns**

---

## What you will learn

- How `SingleAgentLoop` assembles everything from Lessons 2–4 into a working agent
- The exact sequence of operations in every `run()` call
- What `system_prompt` does and how to write one well
- The `stream()` method and its limitations
- When `SingleAgentLoop` is enough and when you need something more powerful

---

## The concept

`SingleAgentLoop` is the foundation of the entire framework. Every other pattern — OrchestratorWorker, Hierarchy, PlannerExecutor — is either built on top of it or replicates its core loop with additions.

Understanding `SingleAgentLoop` completely means understanding 80% of agent engineering. The rest is coordination.

### The sequence of a `run()` call

```
run(user_message)
    │
    ├─ 1. Open MCP connection (Client context manager)
    ├─ 2. list_tools() → get the tool inventory from the server
    ├─ 3. Build initial messages: [history..., Message(role="user", content=user_message)]
    │
    └─ loop (up to max_iterations):
           │
           ├─ 4. client.complete(messages, tools, system) → LLMResponse
           ├─ 5. Append assistant message to history
           │
           ├─ if stop_reason != TOOL_USE:
           │       └─ 6. return response.content  ← DONE
           │
           └─ for each tool_call in response.tool_calls:
                  ├─ 7. call_tool(mcp, tool_call) → result string
                  └─ 8. Append tool result message to history
                         (loop continues)
```

### The MCP connection is scoped to one `run()`

```python
async with Client(self._config.mcp_server_config) as mcp:
    tools = await list_tools(mcp)
    ...
```

The `async with` opens a fresh connection at the start and closes it when `run()` returns. This means:
- Each `run()` is independent — no state leaks between calls
- You can call `run()` concurrently on the same `SingleAgentLoop` instance (different connections)
- The tool list is fetched fresh each `run()` — if the server adds a tool between calls, the next call sees it

### `history` — continuing conversations

```python
result1 = await agent.run("What is BM25?")
history = [
    Message(role="user", content="What is BM25?"),
    Message(role="assistant", content=result1),
]
result2 = await agent.run("How does it compare to vector search?", history=history)
```

Pass `history` to maintain conversation context across multiple `run()` calls. The agent will see the prior exchange and can refer back to it.

Without `history`, every `run()` starts fresh — the agent has no memory of previous calls. For short single-turn tasks, this is exactly what you want. For multi-turn conversations, pass the history.

### `system_prompt` — the most important config

The system prompt is your single biggest lever on agent behaviour. It:
1. Tells the agent who it is and what it is trying to accomplish
2. Constrains the scope (what it should and shouldn't do)
3. Sets the output format expectations
4. Establishes the reasoning style

**Bad system prompt:**
```
You are a helpful assistant.
```

**Good system prompt:**
```
You are a financial data analyst assistant. You have access to a database of 
quarterly earnings reports via the tools available to you.

When answering questions:
1. Always search for relevant data first — do not guess numbers
2. If you cannot find specific data, say so clearly
3. Round currency figures to 2 decimal places
4. Always cite the source quarter/year for any data you reference

Do not speculate about future performance or give investment advice.
```

The good prompt gives the agent clear identity, procedure, formatting rules, and constraints.

### `max_iterations` — right-sizing the cap

| Task type | Typical iterations | Suggested cap |
|-----------|-------------------|---------------|
| Single tool lookup | 2–3 | 5–10 |
| Multi-step research | 5–10 | 15–20 |
| Complex analysis | 10–20 | 30 |
| Agentic RAG | 10–20 | 25 |
| Open-ended | varies | 30–50 |

Setting this too low cuts off agents mid-task. Too high means a stuck agent wastes money. Start at 20 and tune down once you understand your task's typical iteration count.

### `run_stream()` — live token streaming

Every pattern has a `run_stream()` alongside `run()`. Same inputs, different output — instead of waiting for the full answer, you receive one `StreamEvent` per token:

```python
async for event in agent.run_stream("Write a blog post about MCP"):
    match event.type:
        case "thinking":    print(event.delta, end="", flush=True)  # reasoning tokens
        case "text":        print(event.delta, end="", flush=True)  # response tokens
        case "tool_start":  print(f"\n[calling {event.tool_name}...]")
        case "tool_end":    print(f" done\n")
```

The loop still executes tool calls synchronously — streaming is paused while a tool runs, then resumes for the next LLM turn. The `tool_start` / `tool_end` events let you show a spinner or progress indicator in your UI.

**All four combinations:**

| `enable_thinking` | method | what arrives |
|---|---|---|
| `False` (default) | `run()` | `str` at the end |
| `False` (default) | `run_stream()` | `text` events, token by token |
| `True` | `client.complete()` | `LLMResponse` with both `content` and `reasoning` |
| `True` | `run_stream()` | `thinking` events then `text` events, both live |

Enable thinking on the client, not on the pattern:

```python
client = AnthropicClient(enable_thinking=True, thinking_budget=8000)
agent  = SingleAgentLoop(llm_client=client, config=config)

async for event in agent.run_stream("Explain why BM25 beats TF-IDF for short queries"):
    if event.type == "thinking":
        print(f"💭 {event.delta}", end="")
    elif event.type == "text":
        print(event.delta, end="")
```

---

## When is `SingleAgentLoop` enough?

**Use `SingleAgentLoop` when:**
- One LLM, one set of tools, one task
- The task fits in a reasonable number of iterations
- You don't need human approval gates
- You don't need multiple specialised workers
- You don't need quality evaluation loops

**Upgrade to a pattern when:**
- You need multiple specialised tool sets → `OrchestratorWorkerPattern`
- Sub-tasks are complex enough to need their own loops → `HierarchicalAgentPattern`
- Quality matters enough to warrant rewrites → `EvaluatorOptimizerPattern`
- The task has clear phases → `PlannerExecutorPattern`
- Independent tasks can run simultaneously → `ParallelPattern`

For most real-world tasks, a well-prompted `SingleAgentLoop` with good tools gets you 80% of the way there. Add complexity only when you hit a clear limit.

---

## Read this file

```
src/mcp_agent_framework/patterns/single_agent_loop.py
```

Read the entire file. It is under 120 lines. Notice:
- The `async with Client(...)` scoping
- The `list_tools()` call before the loop
- The backwards scan for last assistant message at `max_iterations`
- The `stream()` method's simplicity

---

## Run this

```bash
python examples/01_hello_agent.py
```

Then modify the system prompt to something specific — make it a "Python code reviewer" that only talks about Python code quality. Ask it a non-Python question and observe how the system prompt shapes the response.

---

## Build this

Build a `PersonalAssistant` using `SingleAgentLoop` with these tools:

```python
app = FastMCP("assistant")

@app.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"   # fake data for now

@app.tool
async def set_reminder(task: str, time: str) -> str:
    """Set a reminder. Returns confirmation."""
    return f"Reminder set: '{task}' at {time}"

@app.tool
async def search_notes(query: str) -> str:
    """Search your personal notes."""
    notes = {"meeting": "10am standup", "birthday": "mum's birthday is March 5"}
    matches = [v for k, v in notes.items() if query.lower() in k]
    return "\n".join(matches) if matches else "No notes found."
```

Ask it: *"What's the weather in London, remind me to call the office at 3pm, and check if I have any notes about meetings."*

Observe how many iterations it takes and which tools it calls in what order.

---

## Key terms

| Term | Meaning |
|------|---------|
| `run()` | Execute the full agent loop, block until done, return `str` |
| `run_stream()` | Execute the full agent loop, yield `StreamEvent` objects live |
| `history` | Prior conversation context to continue from |
| `system_prompt` | Instructions given to the LLM before any user messages |
| `max_iterations` | Safety cap — prevents infinite tool-call loops |
| `StreamEvent` | One event from a streaming run: text token, thinking token, or tool step |

---

## Connects to

- **Lesson 6** — deep dive into exactly what happens during a tool call
- **Lesson 7** — add memory tools so the agent remembers across `run()` calls
- **Lesson 8** — `OrchestratorWorkerPattern` uses multiple `SingleAgentLoop` instances as workers
- **Lesson 12** — `EvaluatorOptimizerPattern` runs `SingleAgentLoop` in a quality loop
- **Lesson 20** — `SkillAwareAgent` wraps `SingleAgentLoop` with a skill registry
- **`examples/13_streaming.py`** — all four streaming/thinking combinations in one runnable file

---

*Lesson 5 of 21 — Applied AI Engineering*
