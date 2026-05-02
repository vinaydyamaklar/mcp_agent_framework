# Lesson 16 — Observability

**Unit 5: Scale**

---

## What you will learn

- Why "it worked on my machine" is not a production debugging strategy
- `RunContext`, `BaseTracer`, `LoggingTracer` — the observability stack
- `TraceEvent` and the event types that get recorded
- How `parent_run_id` gives you the full call tree
- How to build a custom tracer for any backend

---

## The concept

Your agent runs in production. It takes 12 seconds to answer a question instead of the usual 3. Or it calls the wrong tool. Or it loops 47 times before stopping. Or it works fine 95% of the time and fails mysteriously 5% of the time.

Without observability, you are flying blind. You can't debug what you can't see.

The framework's observability layer captures every significant event in an agent run and routes them to a tracer. The default tracer logs to Python's standard `logging` module. You can replace it with anything.

---

## `RunContext` — the observability container

```python
from mcp_agent_framework import RunContext, LoggingTracer

context = RunContext(
    run_id="run_20260416_143022_abc",      # unique ID for this run
    parent_run_id=None,                     # None for root runs
    tracer=LoggingTracer(),
)

# Pass context to any pattern
result = await agent.run("What is BM25?", context=context)
```

`run_id` is auto-generated if not provided. `parent_run_id` links child runs (worker agents, sub-agents) back to the parent — this gives you a call tree, not just a flat log.

---

## `TraceEvent` — what gets recorded

```python
@dataclass
class TraceEvent:
    event_type: TraceEventType
    run_id:     str
    parent_run_id: str | None
    timestamp:  datetime
    payload:    dict[str, Any]  # event-specific data
```

**`TraceEventType` values:**

| Event | When | Payload |
|-------|------|---------|
| `LLM_START` | Before each `client.complete()` | model name, message count, tool count |
| `LLM_END` | After each `client.complete()` | stop_reason, token counts, latency |
| `TOOL_START` | Before each `call_tool()` | tool name, arguments |
| `TOOL_END` | After each `call_tool()` | result (first 200 chars), latency |
| `AGENT_START` | Start of a `run()` | config, system prompt |
| `AGENT_END` | End of a `run()` | final output (first 200 chars), total iterations |
| `ERROR` | On any exception | exception type, message, traceback |

These events tell you:
- Which model was called, how many times, how many tokens
- Which tools were called, with what arguments, what they returned
- How many iterations the loop ran
- Where failures occurred

---

## `BaseTracer` — implement your own

```python
from mcp_agent_framework import BaseTracer, TraceEvent

class BaseTracer(ABC):
    @abstractmethod
    async def record_event(self, event: TraceEvent) -> None: ...
```

One method. Implement it to route events anywhere.

**Built-in:** `LoggingTracer` — logs each event to Python's `logging` module at `DEBUG` level.

**Custom examples:**

```python
# Write JSON lines to a file
class JsonFileTracer(BaseTracer):
    def __init__(self, filepath: str):
        self._file = open(filepath, "a")

    async def record_event(self, event: TraceEvent) -> None:
        line = json.dumps({
            "type":    event.event_type.value,
            "run_id":  event.run_id,
            "parent":  event.parent_run_id,
            "ts":      event.timestamp.isoformat(),
            **event.payload,
        })
        self._file.write(line + "\n")
        self._file.flush()
```

```python
# Send to OpenTelemetry
class OtelTracer(BaseTracer):
    def __init__(self, tracer):
        self._tracer = tracer  # opentelemetry.trace.Tracer

    async def record_event(self, event: TraceEvent) -> None:
        with self._tracer.start_as_current_span(event.event_type.value) as span:
            span.set_attribute("run_id", event.run_id)
            for k, v in event.payload.items():
                span.set_attribute(k, str(v))
```

```python
# Send to your database
class DatabaseTracer(BaseTracer):
    async def record_event(self, event: TraceEvent) -> None:
        await db.insert("agent_events", {
            "run_id": event.run_id,
            "type":   event.event_type.value,
            "ts":     event.timestamp,
            "data":   json.dumps(event.payload),
        })
```

---

## The parent_run_id call tree

When a hierarchical agent (Lesson 9) runs, each level gets its own `run_id`, with `parent_run_id` linking them:

```
root_run (run_id="abc", parent=None)
  ├── research_worker (run_id="def", parent="abc")
  │     ├── LLM_START (run_id="def", parent="abc")
  │     ├── TOOL_START: search_knowledge (run_id="def")
  │     └── LLM_END (run_id="def")
  └── writer_worker (run_id="ghi", parent="abc")
        ├── LLM_START (run_id="ghi", parent="abc")
        └── LLM_END (run_id="ghi")
```

With a database tracer, you can reconstruct this tree:
```sql
SELECT * FROM agent_events WHERE run_id = 'abc' OR parent_run_id = 'abc'
ORDER BY ts;
```

---

## Debugging with observability

**Diagnosing a slow run:**
```
LLM_START:  12:00:00.000
LLM_END:    12:00:00.850  ← 850ms, normal
TOOL_START: 12:00:00.851  (search_database)
TOOL_END:   12:00:07.200  ← 6.3 seconds — THIS IS YOUR BOTTLENECK
LLM_START:  12:00:07.201
LLM_END:    12:00:08.100
```

You immediately see that `search_database` is the bottleneck. Without observability, you'd be guessing.

**Diagnosing wrong tool selection:**
```
TOOL_START: {name: "get_weather", arguments: {city: "London"}}
TOOL_END:   "Sunny, 22°C"
TOOL_START: {name: "get_weather", arguments: {city: "London"}}  ← called again?
TOOL_END:   "Sunny, 22°C"
```

The agent called the same tool twice. Now you know to look at the system prompt or tool description — probably the result isn't being interpreted correctly.

---

## Read these files

```
src/mcp_agent_framework/observability/tracer.py       ← BaseTracer, LoggingTracer, TraceEvent
src/mcp_agent_framework/observability/run_context.py  ← RunContext
```

---

## Run this

Enable debug logging in any example:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

You will see `LoggingTracer` output for every event. Find the `TOOL_START` events and match them to the tool calls you see in the final output.

---

## Build this

Build a `JsonFileTracer` that writes one JSON line per event to `agent_trace.jsonl`. Run `01_hello_agent.py` with this tracer. Then write a `print_timeline(filepath)` function that reads the file and prints:

```
12:00:00.000  AGENT_START  run_id=abc
12:00:00.001  LLM_START    model=claude-haiku messages=2 tools=3
12:00:00.843  LLM_END      stop=tool_use tokens=in:145 out:32 latency=842ms
12:00:00.844  TOOL_START   name=search_knowledge args={query: "vector search"}
12:00:00.851  TOOL_END     result="Vector databases store..." latency=7ms
12:00:00.852  LLM_START    model=claude-haiku messages=4 tools=3
12:00:01.701  LLM_END      stop=end_turn tokens=in:312 out:87 latency=849ms
12:00:01.702  AGENT_END    iterations=2 output="Vector search uses..."
```

This becomes your most valuable debug tool for everything that follows.

---

## Key terms

| Term | Meaning |
|------|---------|
| `RunContext` | Container for run_id, parent_run_id, and tracer |
| `TraceEvent` | A single captured event: type, IDs, timestamp, payload |
| `TraceEventType` | Enum: LLM_START/END, TOOL_START/END, AGENT_START/END, ERROR |
| `BaseTracer` | Abstract base — implement `record_event()` to build custom tracers |
| `LoggingTracer` | Default tracer — logs to Python logging at DEBUG level |
| `parent_run_id` | Links child runs to parent — enables call tree reconstruction |

---

## Connects to

- **Lesson 15** — resilience: circuit breaker state changes are trace events
- **Lesson 9** — hierarchy: each level gets its own run_id for call-tree tracing
- **Lesson 19** — LangGraph has its own checkpoint history — compare with trace history

---

*Lesson 16 of 21 — Applied AI Engineering*
