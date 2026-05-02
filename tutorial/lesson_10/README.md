# Lesson 10 — Human-in-the-Loop

**Unit 3: Multi-Agent Coordination**

---

## What you will learn

- Why autonomous agents need human oversight gates
- The two HITL approaches: in-framework (`HumanInLoopPattern`) vs LangGraph `interrupt()`
- The critical difference: blocking vs async (process-surviving) pauses
- When each approach is appropriate
- How `requires_approval` works at the tool level

---

## The concept

Autonomous agents are powerful. They're also capable of making expensive, irreversible, or catastrophic mistakes. Before an agent sends an email to 10,000 customers, deletes a database table, or deploys to production — a human should confirm.

There are two fundamentally different ways to implement this confirmation:

### Approach 1 — `HumanInLoopPattern` (in-framework, blocking)

```python
from mcp_agent_framework import HumanInLoopPattern

pattern = HumanInLoopPattern(
    llm_client=AnthropicClient("claude-sonnet-4-6"),
    config=AgentConfig(
        mcp_server_config=app,
        system_prompt="...",
        extra={"requires_approval": ["deploy_to_production", "send_bulk_email", "delete_records"]},
    ),
    approval_callback=None,  # defaults to input() — blocks waiting for terminal input
)
```

When the agent calls a tool in the `requires_approval` list, the framework pauses and calls `input()`:

```
Agent wants to call: deploy_to_production
Arguments: {"environment": "production", "version": "v2.1.4"}
Approve? (y/n):
```

**The critical limitation: `input()` blocks the event loop.** This works for:
- CLI tools
- Local development and demos
- Scripts where you sit at the terminal

It **does not work** for:
- Web applications (the HTTP request would time out)
- Multi-tenant systems (you can't `input()` across users)
- Async workflows where approval happens via Slack/email (process must stay alive)
- Situations where review takes hours or days

### Approach 2 — LangGraph `interrupt()` (async, process-surviving)

```python
# Inside a LangGraph node:
from langgraph.types import interrupt, Command

async def deploy_node(state: AgentState) -> dict:
    # Agent has prepared a deployment plan
    plan = state["deployment_plan"]

    # Pause here — save state to database, return to caller
    human_decision = interrupt({
        "message": "Please review this deployment plan",
        "plan": plan,
        "risk_level": "HIGH",
    })

    # This line only runs AFTER the human sends Command(resume=...)
    if human_decision["approved"]:
        result = await deploy(plan)
        return {"result": result}
    else:
        return {"result": f"Deployment cancelled: {human_decision['reason']}"}
```

**What `interrupt()` does:**
1. Saves the entire graph state to a checkpointer (SQLite, Postgres, Redis)
2. Returns from `graph.ainvoke()` immediately — the process doesn't block
3. The caller receives an `Interrupt` exception with the payload you passed
4. Hours, days, or restarts later — the human sends `Command(resume={"approved": True})`
5. LangGraph loads the state from the checkpointer and resumes from the exact line after `interrupt()`

This is the production-grade approach. The human approval workflow can be:
- A Slack bot that posts the approval request and waits for a button click
- An email with approve/reject links
- A web dashboard where reviewers queue up pending approvals
- An automated approval system that checks policy rules

---

## The resume pattern

```python
# Initial run — triggers the interrupt
config = {"configurable": {"thread_id": "deploy-job-42"}}
try:
    result = await graph.ainvoke({"task": "deploy v2.1.4"}, config)
except Exception:
    pass  # interrupt raised — approval is pending

# Later — human approves via your UI
# Your UI calls:
result = await graph.ainvoke(
    Command(resume={"approved": True, "reviewer": "alice@company.com"}),
    config,  # same thread_id — loads saved state
)
```

The `thread_id` is the key. It identifies which specific run to resume. Store it in your database when the interrupt fires. Retrieve it when the human approves.

---

## Custom approval callbacks

You can replace `input()` with any async callable:

```python
async def slack_approval(tool_name: str, arguments: dict) -> bool:
    """Post to Slack and wait for a button click."""
    message = await slack.post_message(
        channel="#agent-approvals",
        text=f"Agent wants to call `{tool_name}`\nArgs: {arguments}",
        blocks=[approve_button, reject_button],
    )
    # Store the pending approval in database
    approval_id = await db.create_pending_approval(tool_name, arguments)
    # ... (your Slack webhook handler calls db.resolve_approval(approval_id, approved))
    # Wait for resolution (polling or event)
    return await db.wait_for_approval(approval_id, timeout=3600)

pattern = HumanInLoopPattern(
    llm_client=...,
    config=...,
    approval_callback=slack_approval,
)
```

This extends `HumanInLoopPattern` to production use cases while keeping the process alive. However, if the process restarts while waiting, the approval is lost — which is why LangGraph `interrupt()` (with a persistent checkpointer) is the correct production solution.

---

## Choosing between the two approaches

| Scenario | Use |
|----------|-----|
| CLI tool, local dev, demo | `HumanInLoopPattern` with `input()` |
| Process stays alive, same session | `HumanInLoopPattern` with custom callback |
| Approval can take hours/days | LangGraph `interrupt()` |
| Web app, multi-user, production | LangGraph `interrupt()` |
| Process may restart between request and approval | LangGraph `interrupt()` |

---

## Read these files

```
src/mcp_agent_framework/patterns/human_in_loop_pattern.py
examples/langgraph+mcp_agent_framework/02_interrupts_hitl.py
```

In `human_in_loop_pattern.py`, find where `requires_approval` is checked and how the `approval_callback` is called.

In `02_interrupts_hitl.py`, trace the full interrupt → resume cycle.

---

## Run these

```bash
python examples/04_human_in_loop.py
```

Then (after installing langgraph):
```bash
pip install "langgraph[sqlite]"
python examples/langgraph+mcp_agent_framework/02_interrupts_hitl.py
```

In the first example, you'll be prompted to approve in the terminal. In the second, the graph pauses and then resumes programmatically (simulating a human decision).

---

## Build this

Build a "deployment pipeline agent" with two phases:

1. **Automated phase** — agent runs `run_tests()`, `build_image()`, `deploy_to_staging()` without approval
2. **Human gate** — agent pauses before `deploy_to_production()` and asks for approval

Use `HumanInLoopPattern` with `requires_approval=["deploy_to_production"]`. Then upgrade it to LangGraph `interrupt()` — compare the code and understand why the LangGraph version is more robust.

```python
app = FastMCP("deployment")

@app.tool
async def run_tests() -> str:
    """Run the test suite."""
    return "Tests passed: 142/142"

@app.tool
async def build_image() -> str:
    """Build the Docker image."""
    return "Image built: myapp:v2.1.4"

@app.tool
async def deploy_to_staging() -> str:
    """Deploy to staging environment."""
    return "Deployed to staging.myapp.com"

@app.tool
async def deploy_to_production() -> str:
    """Deploy to production. IRREVERSIBLE."""
    return "Deployed to myapp.com"
```

---

## Key terms

| Term | Meaning |
|------|---------|
| `requires_approval` | List of tool names that trigger human confirmation |
| `approval_callback` | Async function called instead of `input()` |
| `interrupt()` | LangGraph function: saves state, pauses execution, returns to caller |
| `Command(resume=...)` | LangGraph type: load saved state and continue after interrupt |
| Thread ID | Identifier for a specific graph run — used to resume the right state |
| Checkpointer | Persistent state store for LangGraph (MemorySaver, AsyncSqliteSaver) |

---

## Connects to

- **Lesson 19** — LangGraph integration: full coverage of `interrupt()`, checkpointing, and the `Command` API
- **Lesson 13** — PlannerExecutor can pause for human review after planning before executing

---

*Lesson 10 of 20 — Applied AI Engineering*
