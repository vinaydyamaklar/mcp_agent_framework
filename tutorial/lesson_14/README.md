# Lesson 14 — Parallel Pattern

**Unit 5: Scale**

---

## What you will learn

- Fan-out / gather: the architecture of parallelism
- `ParallelPattern`, `ParallelTask`, `ParallelResult`
- The template validation that guards against silent bugs
- When parallelism helps and when it hurts
- How to measure the speedup

---

## The concept

Some tasks have independent sub-tasks that can run simultaneously. Running them sequentially wastes time proportional to N. Running them in parallel costs only the time of the slowest one.

```
Sequential: task_A (3s) → task_B (3s) → task_C (3s) = 9 seconds
Parallel:   task_A (3s)
            task_B (3s)  ← all at once
            task_C (3s)
            = 3 seconds + gather overhead ≈ 3.1 seconds
```

`ParallelPattern` implements this with `asyncio.gather`:

```
user_message
     │
     ├──────────────────────────────┐
     ▼                              ▼
task_A (SingleAgentLoop)     task_B (SingleAgentLoop)
   runs its own loop            runs its own loop
     │                              │
     └──────────────┬───────────────┘
                    ▼
            synthesiser LLM
            (combines all results into final answer)
```

---

## Usage

```python
from mcp_agent_framework import ParallelPattern, ParallelTask, AnthropicClient, AgentConfig
from fastmcp import FastMCP

# Each task has its own agent (its own tools and system prompt)
def make_agent(system_prompt):
    return SingleAgentLoop(
        llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
        config=AgentConfig(mcp_server_config=research_app, system_prompt=system_prompt),
    )

tasks = [
    ParallelTask(
        name="python_research",
        prompt="Research Python's strengths for backend web development in 2026.",
        agent=make_agent("You are a Python expert."),
    ),
    ParallelTask(
        name="go_research",
        prompt="Research Go's strengths for backend web development in 2026.",
        agent=make_agent("You are a Go expert."),
    ),
    ParallelTask(
        name="rust_research",
        prompt="Research Rust's strengths for backend web development in 2026.",
        agent=make_agent("You are a Rust expert."),
    ),
]

pattern = ParallelPattern(
    tasks=tasks,
    synthesiser_client=AnthropicClient("claude-sonnet-4-6"),
    synthesis_template="""
    The user asked: {user_message}

    Research results:
    {task_results}

    Synthesise a balanced comparison and recommendation.
    """,
)

result = await pattern.run("Which language should I use for my new backend service?")
```

---

## The template validation

The `synthesis_template` must contain both `{user_message}` and `{task_results}`. The `ParallelPattern.__init__` validates this at construction time:

```python
if "{task_results}" not in synthesis_template:
    raise ValueError("synthesis_template must contain {task_results}")
if "{user_message}" not in synthesis_template:
    raise ValueError("synthesis_template must contain {user_message}")
```

Why validate at init? Because failing at construction (immediately, when you build the pattern) is far better than failing at runtime (when a user sends a message, in production, at 3am). This is a general principle: **fail fast at the earliest possible point.**

---

## `ParallelResult`

Each completed task produces a `ParallelResult`:

```python
@dataclass
class ParallelResult:
    name:    str
    output:  str | None      # the task's result string
    error:   str | None      # set if the task raised an exception
    elapsed: float           # wall-clock time in seconds
```

Before synthesis, task results are formatted like this:

```
## python_research (2.3s)
Python's strengths include excellent ecosystem...

## go_research (1.9s)
Go excels at concurrent programming...

## rust_research (3.1s)
Rust provides memory safety guarantees...
```

Tasks that failed show their error message instead. The synthesiser still runs — it can note that some research was unavailable.

---

## When parallelism helps

**Three conditions must all be true:**

1. **Tasks are independent** — task B doesn't need task A's output to run
2. **Tasks are slow** — each takes > ~1 second (if tasks are instant, the overhead dominates)
3. **You have 3+ tasks** — with 2 tasks, sequential is often simpler and fast enough

**When parallelism hurts:**
- Tasks depend on each other (use PlannerExecutor instead)
- You are rate-limited — all parallel calls hit the same quota simultaneously, causing throttling
- Tasks are fast — `asyncio.gather` overhead (~10–50ms) is noticeable on sub-100ms tasks

---

## Measuring the speedup

```python
import time

start = time.monotonic()
result = await pattern.run("Compare Python, Go, and Rust for backend development")
elapsed = time.monotonic() - start

# Each ParallelResult has its own elapsed time
# Total elapsed ≈ max(task.elapsed for task in results) + synthesis_time
```

Compare against sequential execution:

```python
# Sequential version for benchmarking:
results = []
start = time.monotonic()
for task in tasks:
    r = await task.agent.run(task.prompt)
    results.append(r)
sequential_elapsed = time.monotonic() - start

# Speedup = sequential_elapsed / parallel_elapsed
```

For 3 tasks of ~3 seconds each: sequential ≈ 9s, parallel ≈ 3.2s — roughly 3× faster.

---

## Read this file

```
src/mcp_agent_framework/patterns/parallel_pattern.py
```

Focus on:
- The template validation in `__init__`
- The `asyncio.gather` call that runs all tasks simultaneously
- How failed tasks are handled (error in `ParallelResult` instead of crashing)
- How the synthesiser formats all results

---

## Run this

```bash
python examples/07_parallel_agents.py
```

Add timing:
```python
import time
start = time.monotonic()
result = await pattern.run(...)
print(f"Total time: {time.monotonic() - start:.2f}s")
```

---

## Build this

Build a parallel "fact checker" for 4 claims:

```python
claims = [
    "Python was created by Guido van Rossum in 1991",
    "The GIL was removed in Python 3.13",
    "asyncio uses preemptive multitasking",
    "FastMCP is built on the MCP protocol standard",
]

# Create one task per claim
tasks = [
    ParallelTask(
        name=f"claim_{i}",
        prompt=f"Check this claim against your knowledge and rate it TRUE, FALSE, or UNCERTAIN with a brief reason: '{claim}'",
        agent=SingleAgentLoop(
            llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
            config=AgentConfig(mcp_server_config=FastMCP("empty")),
        ),
    )
    for i, claim in enumerate(claims)
]

pattern = ParallelPattern(
    tasks=tasks,
    synthesiser_client=AnthropicClient("claude-haiku-4-5-20251001"),
    synthesis_template="Claims to check:\n{user_message}\n\nResults:\n{task_results}\n\nSummary:",
)
```

Measure time. Compare to running the claims sequentially. Which claims does it get right/wrong?

---

## Key terms

| Term | Meaning |
|------|---------|
| Fan-out | Sending the same (or related) input to multiple agents simultaneously |
| Gather | Collecting all parallel results before synthesis |
| `ParallelTask` | Name + prompt + agent for one parallel sub-task |
| `ParallelResult` | Output, error, and elapsed time for one completed task |
| Synthesiser | The LLM that combines all parallel results into a final answer |
| Template validation | Fail-fast check that `{user_message}` and `{task_results}` are in the template |

---

## Connects to

- **Lesson 8** — OrchestratorWorker also uses parallel execution when the orchestrator calls multiple workers in one turn
- **Lesson 20** — `SkillRegistry.invoke_many()` uses the same `asyncio.gather` pattern for parallel skill execution

---

*Lesson 14 of 21 — Applied AI Engineering*
