# Lesson 19 — LangGraph Integration

**Unit 7: Production Infrastructure**

---

## What you will learn

- What LangGraph adds that this framework doesn't have
- Checkpointing: surviving process restarts mid-run
- `interrupt()`: async human approval that survives days and restarts
- Time travel: branching from any past checkpoint
- Streaming: real-time token delivery
- How this framework and LangGraph combine into a complete production stack

---

## The concept

This framework handles **what the agent does** — the ReAct loop, tool calling, patterns, memory, evaluation.

LangGraph handles **when and how the agent runs** — persisting state, pausing for humans, branching to explore alternatives, streaming output.

They are not competing. They compose:

```
┌─────────────────────────────────────────────────────────────┐
│                        LangGraph                             │
│  (state machine, checkpointing, interrupt, time travel)      │
│                                                             │
│    node_1                node_2                node_3        │
│  ┌────────┐           ┌────────┐           ┌────────┐       │
│  │SingleAgent│        │interrupt│           │SingleAgent│    │
│  │  Loop   │  ──────▶ │  (HITL) │  ──────▶ │  Loop   │     │
│  └────────┘           └────────┘           └────────┘       │
└─────────────────────────────────────────────────────────────┘
```

Each LangGraph node wraps a framework pattern. LangGraph provides the infrastructure between nodes.

---

## Feature 1 — Checkpointing

Without checkpointing: if the process crashes at step 7 of a 20-step task, you restart from step 1.

With checkpointing: the state after every node is saved to a database. Restart, and you resume from step 7.

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    result:   str

async def run_agent_node(state: AgentState) -> dict:
    """Wrap SingleAgentLoop as a LangGraph node."""
    agent = SingleAgentLoop(llm_client=..., config=config)
    result = await agent.run(state["messages"][-1].content)
    return {"result": result}

builder = StateGraph(AgentState)
builder.add_node("agent", run_agent_node)
builder.set_entry_point("agent")
builder.set_finish_point("agent")

checkpointer = MemorySaver()   # in-memory for dev
# checkpointer = AsyncSqliteSaver.from_conn_string("agent.db")  # persistent

graph = builder.compile(checkpointer=checkpointer)

# Run — state saved after each node
config = {"configurable": {"thread_id": "conversation-42"}}
result = await graph.ainvoke({"messages": [HumanMessage(content="Hello")]}, config)

# Run again with same thread_id — continues from where it left off
result = await graph.ainvoke({"messages": [HumanMessage(content="Follow up")]}, config)
```

**`thread_id`** is the conversation identifier. Same `thread_id` = same conversation. Different `thread_id` = new conversation.

**`MemorySaver`** is in-memory — lost on restart. Use `AsyncSqliteSaver` for development persistence, Postgres for production.

---

## Feature 2 — `interrupt()`: async human approval

```python
from langgraph.types import interrupt, Command

async def review_node(state: AgentState) -> dict:
    """Pause here and wait for human approval."""
    plan = state["plan"]

    # This saves the graph state to the checkpointer and returns immediately.
    # The process does NOT block. The human can approve hours later.
    human_decision = interrupt({
        "message": "Please review this plan before execution",
        "plan":    plan,
    })

    # This line only runs after Command(resume=...) is sent
    if human_decision.get("approved"):
        return {"approved": True}
    else:
        return {"approved": False, "reason": human_decision.get("reason", "")}
```

**The resume pattern:**
```python
# Initial invocation — triggers the interrupt
try:
    result = await graph.ainvoke({"task": "..."}, config)
except:
    pass  # interrupt raised

# Human reviews the plan (via Slack, web UI, email, etc.)
# Your UI calls:
result = await graph.ainvoke(
    Command(resume={"approved": True, "reviewer": "alice"}),
    config,  # same thread_id — loads saved state
)
```

The gap between interrupt and resume can be seconds, hours, or days. The process can restart. The state is safe in the checkpointer.

---

## Feature 3 — Time Travel

Every checkpoint is immutable history. You can go back.

```python
# List all checkpoints for a thread
history = [s async for s in graph.aget_state_history(config)]

for checkpoint in history:
    print(checkpoint.config["configurable"]["checkpoint_id"])
    print(checkpoint.next)   # which nodes would run next
    print(len(checkpoint.values.get("messages", [])))  # message count

# Branch from a specific checkpoint
old_checkpoint = history[2]  # step 3 of the original run

# Create a new thread branching from that checkpoint
new_config = {
    "configurable": {
        "thread_id": "experiment-branch-1",  # new thread
        "checkpoint_id": old_checkpoint.config["configurable"]["checkpoint_id"],
    }
}

# Run from that branch point — original thread is untouched
result = await graph.ainvoke(
    Command(resume={"use_different_model": True}),
    new_config,
)
```

**Use cases:**
- "What would have happened if I'd used GPT-4o at step 3 instead of Claude Haiku?"
- Debug production incidents: replay from the moment before the failure
- A/B test different strategies starting from the same conversation state

---

## Feature 4 — Streaming

```python
from langgraph.types import StreamWriter

async def streaming_node(state: AgentState, writer: StreamWriter) -> dict:
    """Stream tokens as they arrive."""
    client = anthropic.AsyncAnthropic()

    async with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": state["task"]}],
    ) as stream:
        async for text in stream.text_stream:
            writer({"token": text})   # pushed to the caller immediately

    final_text = await stream.get_final_message()
    return {"result": final_text.content[0].text}

# On the calling side:
async for chunk in graph.astream(
    {"task": "Write a summary of..."},
    config,
    stream_mode="custom",   # receive custom writer() payloads
):
    print(chunk["token"], end="", flush=True)  # real-time token printing
```

**`stream_mode="values"`** — receive the full state after each node completes.

**`stream_mode="custom"`** — receive exactly what you `writer()` push from inside nodes.

---

## The combined architecture

```
┌─ LangGraph Graph ──────────────────────────────────────────────────────┐
│                                                                         │
│  research_node                  review_node            finalise_node   │
│  ┌──────────────────┐          ┌──────────┐           ┌─────────────┐  │
│  │ SingleAgentLoop  │ ──────▶  │interrupt │ ──────▶   │EvaluatorOpt │  │
│  │ + AgenticRAG     │          │ (HITL)   │           │ + Streaming │  │
│  └──────────────────┘          └──────────┘           └─────────────┘  │
│                                                                         │
│  Checkpointer: AsyncSqliteSaver (survives restarts)                     │
└─────────────────────────────────────────────────────────────────────────┘
```

This is the production-grade agent:
- `research_node`: Agentic RAG (Lesson 18) + SingleAgentLoop (Lesson 5)
- `review_node`: interrupt() — async human approval (this lesson)
- `finalise_node`: EvaluatorOptimizer (Lesson 12) + streaming tokens to the user
- `AsyncSqliteSaver`: state survives process restarts; can resume after days

---

## Read these files

```
examples/langgraph+mcp_agent_framework/01_checkpointing.py
examples/langgraph+mcp_agent_framework/02_interrupts_hitl.py
examples/langgraph+mcp_agent_framework/03_time_travel.py
examples/langgraph+mcp_agent_framework/04_streaming.py
examples/langgraph+mcp_agent_framework/05_all_combined.py
```

Start with `01_checkpointing.py` and work through to `05_all_combined.py`. The combined example is the target architecture.

---

## Run these

```bash
pip install "langgraph[sqlite]"
python examples/langgraph+mcp_agent_framework/01_checkpointing.py
python examples/langgraph+mcp_agent_framework/02_interrupts_hitl.py
python examples/langgraph+mcp_agent_framework/05_all_combined.py
```

---

## Build this

Build a "multi-turn research assistant" using LangGraph + this framework:

**3 nodes:**
1. `gather_node` — `SingleAgentLoop` asks the user clarifying questions and structures the research task
2. `research_node` — Agentic RAG (Lesson 18) does the actual research, streams progress via `StreamWriter`
3. `report_node` — `EvaluatorOptimizerPattern` (Lesson 12) writes and refines the final report

**Requirements:**
- Checkpoint to SQLite — survives restart between the three nodes
- `interrupt()` after `research_node` — human can add context or approve before report is written
- Stream the final report token-by-token

Run it. Kill the process mid-research. Restart. Observe that it resumes from the research checkpoint.

---

## Key terms

| Term | Meaning |
|------|---------|
| `StateGraph` | LangGraph's graph definition |
| `AgentState` | TypedDict that flows between nodes |
| `checkpointer` | Where state is saved (MemorySaver, AsyncSqliteSaver, Postgres) |
| `thread_id` | Identifies one conversation — same ID continues the same thread |
| `interrupt()` | Saves state, pauses execution, returns to caller |
| `Command(resume=...)` | Loads state and resumes after an interrupt |
| `StreamWriter` | Pushes data from inside a node to the caller in real time |
| `stream_mode` | `"values"` (full state per node) or `"custom"` (writer payloads) |
| Time travel | Branching from any past checkpoint — original untouched |

---

## Connects to

- **Lesson 10** — Human-in-the-Loop: `interrupt()` is the production upgrade of `HumanInLoopPattern`
- **Lesson 15** — Resilience: circuit breakers and retry work within LangGraph nodes
- **Lesson 20** — Skills as LangGraph subgraphs: a `Skill` handler can be a LangGraph graph

---

*Lesson 19 of 21 — Applied AI Engineering*
