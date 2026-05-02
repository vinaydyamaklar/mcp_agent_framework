# Lesson 8 — Orchestrator Pattern

**Unit 3: Multi-Agent Coordination**

---

## What you will learn

- The one-brain-many-hands architecture
- How `OrchestratorWorkerPattern` differs from `SingleAgentLoop`
- How workers are exposed as tools and why this is elegant
- Parallel vs sequential tool execution and when each matters
- The real-world tasks this pattern is built for

---

## The concept

`SingleAgentLoop` has one LLM and one set of tools. Everything lives in one namespace. Works great for tasks with a unified toolset. Falls apart when your task requires specialised capabilities that shouldn't be mixed together.

Imagine a task: *"Research our top 5 competitors, pull their pricing from our database, and write a competitive analysis report."*

This requires:
- Web search tools (research capability)
- Database query tools (pricing capability)
- A writing instruction (no tools needed — pure LLM)

If you dump all tools into one `SingleAgentLoop`, the agent might get confused, call the wrong tools, or use database tools when it should use web search. More importantly, each capability might be maintained by a different team and live on a different server.

**The Orchestrator pattern solves this with separation of concerns:**

```
User task
    │
    ▼
Orchestrator LLM
(sees workers as tools, coordinates)
    │          │          │
    ▼          ▼          ▼
research    database    writer
 worker      worker     worker
(web tools) (SQL tools) (no tools)
```

The orchestrator's "tools" are the workers themselves. It decides which worker to call, passes the sub-task, receives the result as a string, and coordinates until the final answer is ready.

---

## How workers become tools

Each worker is a `WorkerConfig`:

```python
from mcp_agent_framework import OrchestratorWorkerPattern, WorkerConfig

pattern = OrchestratorWorkerPattern(
    orchestrator_client=AnthropicClient("claude-sonnet-4-6"),
    workers=[
        WorkerConfig(
            name="research_worker",
            description="Researches topics using web search and the knowledge base. "
                        "Pass a specific research question. Returns a detailed summary.",
            mcp_server_config=research_app,
            system_prompt="You are a research specialist. Be thorough and cite sources.",
        ),
        WorkerConfig(
            name="database_worker",
            description="Queries the product and pricing database. "
                        "Pass a specific data question. Returns structured data.",
            mcp_server_config=database_app,
            system_prompt="You are a data analyst. Return clean, formatted data.",
        ),
        WorkerConfig(
            name="writer_worker",
            description="Writes polished prose from bullet points or research notes. "
                        "Pass the content and desired output format. Returns finished text.",
            mcp_server_config=FastMCP("empty"),  # no tools — pure LLM writing
            system_prompt="You are a professional technical writer.",
        ),
    ],
)
```

At runtime, the orchestrator sees these as tools:

```json
[
  {"name": "research_worker", "description": "Researches topics..."},
  {"name": "database_worker", "description": "Queries the database..."},
  {"name": "writer_worker",   "description": "Writes polished prose..."}
]
```

When the orchestrator calls `research_worker(task="Find competitor pricing for product X")`, the framework spins up a `SingleAgentLoop` with the research tools, runs it with that task, and returns the result string back to the orchestrator as a tool result.

**Workers are SingleAgentLoops under the hood.** The orchestrator just sees the result string — it doesn't know or care how the worker produced it.

---

## Parallel tool execution

When the orchestrator calls multiple workers in one response, they run in parallel:

```python
# OrchestratorWorkerPattern internals:
results = await asyncio.gather(*[_execute_one(tc) for tc in response.tool_calls])
```

This means if the orchestrator decides to call `research_worker` and `database_worker` in the same turn, both workers run simultaneously. Total time ≈ the slowest worker, not the sum of all workers.

The orchestrator learns to do this when it notices the workers are independent. Help it by making independence clear in the worker descriptions: *"Queries the database independently — does not need output from other workers to run."*

---

## The orchestrator's system prompt matters

The orchestrator needs to know the strategy, not just the goal:

```python
system_prompt="""
You are coordinating a competitive analysis. You have three specialist workers.

Strategy:
1. Use research_worker AND database_worker in parallel to gather information
2. Only call writer_worker AFTER you have the research and pricing data
3. Pass all gathered data to writer_worker in a single, structured call
4. Do not attempt to synthesise yourself — let the writer_worker do that

The final output should be a formatted report from writer_worker.
"""
```

Without explicit strategy guidance, the orchestrator may serialise everything unnecessarily or try to synthesise itself instead of delegating to the writer.

---

## OrchestratorWorker vs SingleAgentLoop

| | `SingleAgentLoop` | `OrchestratorWorkerPattern` |
|--|---|---|
| Number of LLMs | 1 | 1 orchestrator + N workers |
| Tool namespaces | 1 | N (one per worker) |
| Tool isolation | No | Yes — each worker has its own tools |
| Parallelism | Sequential | Parallel workers when independent |
| Cost | Lower | Higher (multiple LLM calls) |
| Complexity | Low | Medium |

---

## Read this file

```
src/mcp_agent_framework/patterns/orchestration_pattern.py
```

Focus on:
- How `WorkerConfig` is converted into a tool schema
- The `asyncio.gather` for parallel execution
- How each worker result is packaged as a tool result message

---

## Run this

No dedicated example file. Build one in the exercise below.

---

## Build this

Build a "company research platform" with three workers:

```python
# Worker 1: news researcher (returns fake news)
news_app = FastMCP("news")
@news_app.tool
async def search_news(company: str) -> str:
    """Search recent news about a company."""
    return f"[News] {company} launched new product line. Revenue up 12% YoY."

# Worker 2: financial data worker (returns fake financials)
finance_app = FastMCP("finance")
@finance_app.tool
async def get_financials(company: str) -> str:
    """Get financial summary for a company."""
    return f"[Financials] {company}: Revenue $2.4B, Margin 18%, P/E 24x"

# Worker 3: writer (no tools)
writer_app = FastMCP("writer")

pattern = OrchestratorWorkerPattern(
    orchestrator_client=AnthropicClient("claude-sonnet-4-6"),
    workers=[
        WorkerConfig(name="news_worker",     description="...", mcp_server_config=news_app),
        WorkerConfig(name="finance_worker",  description="...", mcp_server_config=finance_app),
        WorkerConfig(name="writer_worker",   description="...", mcp_server_config=writer_app),
    ],
)

result = await pattern.run("Write a one-page investment brief on Anthropic.")
```

Add timing to see if news and finance workers run in parallel. Check the orchestrator's tool call sequence — does it call them together or sequentially?

---

## Key terms

| Term | Meaning |
|------|---------|
| Orchestrator | The coordinating LLM — sees workers as tools |
| Worker | A `SingleAgentLoop` with a specialised toolset, exposed as a tool |
| `WorkerConfig` | Name + description + MCP server + system prompt for one worker |
| Parallel execution | `asyncio.gather` — independent workers run simultaneously |

---

## Connects to

- **Lesson 9** — Hierarchy takes this further: workers can themselves have sub-workers
- **Lesson 14** — Parallel pattern is specialised fan-out: same task → N independent agents
- **Lesson 20** — SkillAwareAgent is an orchestrator where workers are Skills

---

*Lesson 8 of 21 — Applied AI Engineering*
