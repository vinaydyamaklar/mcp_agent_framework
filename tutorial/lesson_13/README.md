# Lesson 13 — PlannerExecutor

**Unit 4: Quality and Improvement**

---

## What you will learn

- Why some tasks need explicit planning before execution
- The `ExecutionPlan` and `ExecutionStep` types
- The two-model cost optimisation pattern
- Dynamic replan when a step fails
- The step renumbering bug and how it's fixed

---

## The concept

`SingleAgentLoop` is reactive — it figures out what to do next at each iteration, based only on the current state. For many tasks this works fine. But for complex multi-step tasks, this leads to:

1. **Wasted computation** — the agent re-decides what to do at each step instead of following a plan
2. **No visibility** — you don't know what the agent intends to do until it does it
3. **Poor recovery** — when something fails, the agent improvises rather than replanning

`PlannerExecutorPattern` separates thinking from doing:

```
Phase 1 — PLAN
    Smart model receives the task
    Produces a structured ExecutionPlan
    Plan is printed / logged before execution starts

Phase 2 — EXECUTE
    Cheap model executes each step in order
    Each step's output is passed to the next step
    If a step fails → replan from that point
```

---

## The `ExecutionPlan` type

```python
@dataclass
class ExecutionPlan(BaseModel):
    steps:     list[ExecutionStep]
    rationale: str   # why this plan structure was chosen

@dataclass
class ExecutionStep(BaseModel):
    step_number:    int
    description:    str
    requires_tools: bool
    dependencies:   list[int]   # step numbers this step depends on
```

The plan is generated using `complete_structured()` (Lesson 3) — the planner model is forced to return a valid `ExecutionPlan` object. If the model's response can't be parsed into the schema, a `StructuredOutputError` is raised.

The `dependencies` field lets the executor know which steps can run in parallel (no dependencies on each other) vs which must wait for others. In the current implementation, steps run sequentially — parallel execution is a future enhancement.

---

## The two-model cost optimisation

```python
pattern = PlannerExecutorPattern(
    planner_client=AnthropicClient("claude-sonnet-4-6"),       # smart — runs once
    executor_client=AnthropicClient("claude-haiku-4-5-20251001"), # cheap — runs per step
    config=AgentConfig(mcp_server_config=app),
    max_steps=20,
)
```

The planner runs **once** to create the plan. It needs to be a capable model — the plan quality determines everything downstream.

The executor runs **once per step**. For a 10-step plan, that's 10 executor calls. Using a cheap model here reduces cost by 5–10× compared to using the smart model for everything.

**Rule of thumb:** spend on the plan, save on execution.

---

## Dynamic replan

When a step fails (tool error, bad result, the executor's LLM decides the step can't be completed as described), the executor can request a replan:

```python
# Executor signals replanning needed:
if step_result.startswith("REPLAN:"):
    remaining_context = step_result[len("REPLAN:"):]
    new_plan = await self._planner.complete_structured(
        messages=[...context..., Message(role="user",
            content=f"Step {step.step_number} failed: {remaining_context}. Replan from here.")],
        response_model=ExecutionPlan,
    )
    # Splice new steps in place of remaining steps
    plan_obj.steps[current_index:] = new_plan.steps
```

After splicing, **step numbers must be renumbered**:

```python
for j, s in enumerate(plan_obj.steps):
    s.step_number = j + 1
```

Without renumbering, `dependencies` references break. Step 7 (from the new plan, internally numbered 1) is now step 4 in the merged plan. Any other step that depended on "step 7" in the original plan would be wrong.

---

## The plan as a transparency tool

Before execution starts, the plan is available:

```python
result = await pattern.run("Research and summarise the top 3 AI frameworks for 2026")
# Access the plan (implementation-dependent — see source for how to hook in)
```

This is the real value of PlannerExecutor: **you can see the plan before committing to execution.** In a Human-in-the-Loop setup (Lesson 10), you can pause after planning and ask a human to review and approve the plan before the executor runs.

---

## When to use PlannerExecutor

**Use when:**
- The task has 5+ steps with clear sequential dependencies
- You need visibility into what the agent intends to do
- Recovery from step failures matters (replan vs. just failing)
- The cost difference between smart/cheap models is significant
- You want to add a human review gate between plan and execution

**Don't use when:**
- The task is exploratory (you don't know the steps upfront)
- The task is simple (a plan adds overhead without benefit)
- Steps are highly dynamic (the plan becomes stale immediately)

---

## Read this file

```
src/mcp_agent_framework/patterns/planner_executor_pattern.py
```

Focus on:
- How `complete_structured()` is used to get a typed plan
- The step renumbering after replan
- The executor loop: how each step's output feeds into the next

---

## Run this

```bash
python examples/06_planner_executor.py
```

Watch the plan printed before execution starts. Count the steps. Observe how the executor uses the description and previous step outputs.

---

## Build this

Build a "data pipeline planner" for this task: *"Download sales data for Q1 2026, clean it (remove nulls and duplicates), calculate monthly totals, and produce a summary report."*

Use fake tools:

```python
app = FastMCP("data_pipeline")

@app.tool
async def download_data(period: str) -> str:
    """Download sales data for a given period."""
    return "sales_q1_2026.csv: 1000 rows, 5 columns (date, product, qty, price, region)"

@app.tool
async def clean_data(filename: str) -> str:
    """Remove null values and duplicate rows."""
    return "Cleaned: removed 12 nulls, 3 duplicates. 985 rows remaining."

@app.tool
async def calculate_totals(filename: str, groupby: str) -> str:
    """Calculate totals grouped by a column."""
    return "Totals by month: Jan: $142k, Feb: $159k, Mar: $171k"

@app.tool
async def write_report(data: str, format: str = "markdown") -> str:
    """Write a formatted report from the provided data."""
    return f"# Q1 2026 Sales Report\n{data}"
```

Print the generated plan. Then try breaking one step (make `calculate_totals` return `"REPLAN: data format error"`). Does the replanning kick in?

---

## Key terms

| Term | Meaning |
|------|---------|
| `ExecutionPlan` | Structured plan: list of steps + rationale |
| `ExecutionStep` | One step: number + description + requires_tools + dependencies |
| Planner | Smart model that generates the plan (runs once) |
| Executor | Cheap model that executes each step (runs per step) |
| Replan | Dynamic plan update when a step fails |
| Step renumbering | Fixing step numbers after splicing new steps into the plan |

---

## Connects to

- **Lesson 10** — add a Human-in-the-Loop gate between plan and execution
- **Lesson 12** — apply EvaluatorOptimizer to the plan itself before execution
- **Lesson 11** — evaluate each step's output quality before proceeding to the next

---

*Lesson 13 of 20 — Applied AI Engineering*
