# Lesson 7 — Memory

**Unit 2: Core Patterns**

---

## What you will learn

- Why the message list alone isn't enough for long-running agents
- The three memory types and when each one applies
- How cosine similarity works (and why you don't need to understand the math to use it)
- How to wire memory into an MCP server so the agent can call it as a tool
- The deque implementation detail that matters for production

---

## The concept

By default, `SingleAgentLoop` has no memory between `run()` calls. Each call starts fresh. For a customer support agent that handles thousands of conversations, this is fine — each conversation is independent.

But what if you are building:
- A coding assistant that learns your preferences over time
- A research agent that accumulates knowledge across sessions
- A task agent that needs to remember what it did last week

The framework provides three memory types. Each solves a different problem.

---

## The three memory types

### `SemanticMemory` — remember by *meaning*

```python
memory = SemanticMemory()
await memory.add("Python's GIL prevents true multi-threading for CPU-bound tasks")
await memory.add("asyncio uses cooperative multitasking — tasks yield voluntarily")
await memory.add("multiprocessing bypasses the GIL by using separate processes")

results = await memory.search("how to run Python code in parallel?", top_k=2)
# Returns the GIL entry and multiprocessing entry — not the asyncio one (less relevant)
```

**How it works:** When you `add()` text, it gets an embedding — a vector of ~1536 numbers where each number encodes some aspect of meaning. Similar meanings produce similar vectors. When you `search()`, your query gets embedded and the system finds the entries with the smallest angle between their vector and the query vector.

```
cosine_similarity = dot(query_vec, entry_vec) / (|query_vec| × |entry_vec|)
```

Score of 1.0 = identical meaning. Score of 0.0 = unrelated. Score approaching -1.0 = opposite meaning. In practice, "good match" is anything above 0.7–0.8.

**The embedding function is pluggable.** By default, `SemanticMemory` uses a simple TF-IDF-like approach for demonstrations. In production, you would plug in OpenAI's `text-embedding-3-small` or Anthropic's embeddings for higher quality retrieval.

**Use when:** You need "find relevant facts" — documents, knowledge base entries, past conversation summaries.

### `EpisodicMemory` — remember *what happened*

```python
memory = EpisodicMemory(max_entries=1000)
await memory.add("User asked about the refund policy for order #1234")
await memory.add("Agent retrieved policy document successfully")
await memory.add("User seemed satisfied with the response")

recent = await memory.get_recent(n=3)    # last 3 events, newest first
relevant = await memory.search("refund", top_k=5)  # events mentioning refunds
```

**How it works:** A time-ordered log with recency and keyword scoring. Newer events score higher. Events containing your search keywords score higher. The final score is a weighted combination.

**The `deque` implementation detail:** Episodes are stored in a `collections.deque(maxlen=max_entries)`. When the deque is full, the oldest entry drops off automatically — O(1) operation. If you used a plain `list`, eviction would be `list.pop(0)` — O(n) because every element shifts left. At 10,000 entries this is a noticeable performance difference.

```python
# Wrong (O(n) eviction):
self._entries: list[MemoryEntry]
self._entries.pop(0)

# Correct (O(1) eviction):
self._entries: deque[MemoryEntry]
self._entries.popleft()
```

**Note:** `deque` doesn't support slicing (`self._entries[-n:]` fails). Convert to list first: `list(self._entries)[-n:]`.

**Use when:** You need "what happened recently" — agent logs, conversation history, audit trails.

### `ProceduralMemory` — remember *how to do things*

```python
memory = ProceduralMemory()
await memory.store(
    task="summarise a long document",
    procedure="1. Split into 500-word chunks. 2. Summarise each chunk. 3. Merge summaries into final."
)
await memory.store(
    task="answer a customer question about billing",
    procedure="1. Search billing FAQ. 2. If not found, search order history. 3. Escalate if unresolved."
)

result = await memory.retrieve("how to condense a lengthy article")
# Returns the "summarise a long document" procedure — matched by Jaccard similarity
```

**How it works:** Jaccard similarity on word sets.
```
Jaccard("condense lengthy article", "summarise long document") = |intersection| / |union|
= |{article}| / |{condense, lengthy, article, summarise, long, document}| ≈ 0.17
```
Low, but combined with the semantic overlap ("condense" ≈ "summarise") it finds the right match.

**Use when:** You need "how to handle this type of task" — playbooks, SOPs, workflow templates.

---

## Wiring memory to an MCP server

Memory is only useful if the agent can access it. You expose it as MCP tools:

```python
from fastmcp import FastMCP
from mcp_agent_framework.memory import SemanticMemory, EpisodicMemory

app    = FastMCP("memory_server")
sem    = SemanticMemory()
episodic = EpisodicMemory(max_entries=500)

@app.tool
async def remember(text: str) -> str:
    """Store a piece of information for later retrieval."""
    await sem.add(text)
    return "Stored."

@app.tool
async def recall(query: str) -> str:
    """Search stored knowledge for information relevant to the query."""
    results = await sem.search(query, top_k=3)
    if not results:
        return "Nothing relevant found."
    return "\n\n".join(r.content for r in results)

@app.tool
async def log_event(event: str) -> str:
    """Log an event to the agent's episode history."""
    await episodic.add(event)
    return "Logged."

@app.tool
async def recent_events(n: int = 5) -> str:
    """Retrieve the N most recent logged events."""
    events = await episodic.get_recent(n=n)
    return "\n".join(f"[{e.created_at}] {e.content}" for e in events)
```

The agent calls these tools exactly like any other tool. From the agent's perspective, memory is just another tool call.

---

## The `MemoryEntry` type

Every stored item becomes a `MemoryEntry`:

```python
@dataclass
class MemoryEntry:
    id:         str
    content:    str
    metadata:   dict[str, Any]   # arbitrary extra data (source, timestamp, tags)
    created_at: datetime
    embedding:  list[float] | None
```

`metadata` is your hook for rich retrieval. Store the source document, the user session ID, confidence scores, tags — anything your application needs:

```python
await memory.add(
    "The company was founded in 2018",
    metadata={"source": "about_page", "confidence": "high", "section": "company_history"}
)
```

Then filter retrieved results by metadata in your tool logic.

---

## Read these files

```
src/mcp_agent_framework/memory/base.py          ← AbstractMemoryStore interface
src/mcp_agent_framework/memory/semantic.py      ← add(), search(), cosine similarity
src/mcp_agent_framework/memory/episodic.py      ← deque, recency scoring, get_recent()
src/mcp_agent_framework/memory/procedural.py    ← Jaccard similarity, store(), retrieve()
```

In `episodic.py`, find the `deque` import and the `popleft()` call. In `semantic.py`, find the cosine similarity calculation and trace it through a search.

---

## Run this

```bash
python examples/08_memory_agent.py
```

In one run, tell the agent several facts about yourself. In a second run (with history), ask a question that requires combining two of those facts.

---

## Build this

Build a "learning agent" that accumulates knowledge across multiple sessions:

```python
# Session 1
agent.run("My name is Alex. I prefer Python over JavaScript. I work on data pipelines.")

# Session 2 (fresh run, but memory persists)
agent.run("What do I prefer for web development?")
# Should answer "JavaScript" — wait, no. It should say Python based on what it stored.
```

The trick: at the end of Session 1, the agent should call `remember()` to store key facts. At the start of Session 2, the agent should call `recall()` before answering. Write the system prompt that makes this happen reliably.

---

## Key terms

| Term | Meaning |
|------|---------|
| `SemanticMemory` | Vector store — find by meaning via cosine similarity |
| `EpisodicMemory` | Time-ordered log — find by recency and keywords |
| `ProceduralMemory` | Task → procedure map — find by Jaccard similarity |
| Cosine similarity | Angle between two vectors; 1.0 = identical meaning |
| `deque` | Double-ended queue with O(1) append and popleft — used for fixed-size episode log |
| Embedding | A vector representation of text meaning |
| `MemoryEntry` | The stored unit: content + metadata + embedding + timestamp |

---

## Connects to

- **Lesson 5** — memory tools are just tools; they fit into `SingleAgentLoop` with no changes
- **Lesson 17** — RAG is SemanticMemory at scale: chunking documents and searching by meaning
- **Lesson 18** — Agentic RAG adds BM25 and self-evaluation on top of SemanticMemory
- **Lesson 20** — a `research_topic` skill can use SemanticMemory as its knowledge store

---

*Lesson 7 of 20 — Applied AI Engineering*
