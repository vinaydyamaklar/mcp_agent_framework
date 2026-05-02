# Lesson 9 — Hierarchy Pattern

**Unit 3: Multi-Agent Coordination**

---

## What you will learn

- How agents can delegate to sub-agents (recursive depth)
- The difference between OrchestratorWorker and Hierarchy
- When sub-tasks are complex enough to need their own reasoning loops
- The depth trap — why more levels isn't always better
- How the hierarchy detects FastMCP vs dict config for each child

---

## The concept

Orchestrator-Worker (Lesson 8) has one fundamental limit: each worker is a `SingleAgentLoop`. If a sub-task is complex enough to need its own multi-step planning with multiple tool calls and its own reasoning loop — a `SingleAgentLoop` is often enough. But sometimes the sub-task is genuinely as complex as the top-level task.

**Hierarchy** gives you recursive delegation:

```
Root Agent                   ← strategic reasoning, high-level coordination
├── Research Team Agent      ← research strategy, delegates to specialist agents
│   ├── Web Search Agent     ← runs its own ReAct loop with web tools
│   └── Database Agent       ← runs its own ReAct loop with SQL tools
├── Analysis Agent           ← data analysis coordination
│   ├── Statistics Agent     ← statistical computations
│   └── Visualisation Agent  ← chart generation
└── Writing Agent            ← report writing coordination
    ├── Drafter Agent        ← produces draft
    └── Editor Agent         ← improves draft
```

Every node in the tree is a full agent. Each has its own LLM client, its own set of tools, and its own system prompt. Each can run its own multi-iteration ReAct loop. The output of each node is a string that its parent receives as a tool result.

---

## Hierarchy vs OrchestratorWorker

| | OrchestratorWorker | Hierarchy |
|--|---|---|
| Worker depth | 1 level | N levels (recursive) |
| Worker complexity | `SingleAgentLoop` | Full `HierarchicalAgentPattern` |
| Use when | Workers are straightforward | Sub-tasks need their own sub-delegation |
| LLM calls | O(orchestrator + workers) | O(exponential with depth) |
| Latency | Moderate | High |

The right question to ask: *"Does this sub-task need to delegate further?"* If yes, you need Hierarchy. If the sub-task can be completed with a simple ReAct loop, Orchestrator-Worker is sufficient and cheaper.

---

## The depth trap

Hierarchy is seductive. More levels = more power, right?

Wrong. Depth multiplies cost and failure surface:

- **2-level hierarchy**: 1 root + 3 children = ~4 LLM calls minimum
- **3-level hierarchy**: 1 root + 3 children + 9 grandchildren = ~13 LLM calls minimum
- **4-level hierarchy**: ~40 LLM calls minimum

Each level also adds:
- A new point of failure
- A new system prompt to maintain
- A new opportunity for miscommunication between levels
- Latency (each level must wait for its children)

**The practical rule:** Before adding a level, ask if you can solve the sub-task with a well-prompted `SingleAgentLoop`. Most of the time, you can. Hierarchy is for genuinely complex sub-tasks that themselves require multi-agent coordination.

In practice, 2 levels is the common case. 3 levels is unusual. 4+ levels is almost always a design problem.

---

## How child agents are configured

```python
from mcp_agent_framework import HierarchicalAgentPattern, ChildAgentConfig

pattern = HierarchicalAgentPattern(
    root_client=AnthropicClient("claude-sonnet-4-6"),
    root_config=AgentConfig(
        mcp_server_config=coordination_tools,
        system_prompt="You are the root coordinator...",
    ),
    children=[
        ChildAgentConfig(
            name="research_team",
            description="Coordinates research sub-agents to gather comprehensive information.",
            client=AnthropicClient("claude-haiku-4-5-20251001"),
            config=AgentConfig(
                mcp_server_config=research_tools,
                system_prompt="You are the research team lead...",
            ),
        ),
        ChildAgentConfig(
            name="writing_team",
            description="Coordinates drafting and editing to produce polished output.",
            client=AnthropicClient("claude-haiku-4-5-20251001"),
            config=AgentConfig(
                mcp_server_config=writing_tools,
                system_prompt="You are the writing team lead...",
            ),
        ),
    ],
)
```

Each `ChildAgentConfig` has its own `client` (so you can use cheaper models for children) and its own `config` (its own MCP server and system prompt). At runtime, the root agent sees each child as a tool.

---

## The FastMCP vs dict detection

Inside `hierarchy_pattern.py`, there is a guard for when the child's `mcp_server_config` is a dict vs a FastMCP instance:

```python
if isinstance(self._config.mcp_server_config, dict) and self._config.mcp_server_config.get("mcpServers"):
    # dict config: stdio or HTTP transport
    ...
else:
    # FastMCP instance: in-process
    ...
```

This matters because a `FastMCP` instance has no `"mcpServers"` key — calling `.get("mcpServers")` on it would raise `AttributeError`. The `isinstance` check prevents that.

---

## Communication between levels

Every level communicates via natural language strings. The parent sends a task description string. The child runs its loop and returns a text answer. The parent receives that as a tool result and continues.

This is both the strength and the weakness:
- **Strength:** Flexible. Any LLM at any level can understand any other level's output.
- **Weakness:** No structured handoffs. Parsing errors are possible. The child can return a poor-quality string and the parent has no way to validate it beyond re-reading it.

For structured handoffs, combine Hierarchy with Evaluation (Lesson 11): the parent can validate the child's output using `LLMEvaluator` before continuing.

---

## Read this file

```
src/mcp_agent_framework/patterns/hierarchy_pattern.py
```

Focus on:
- How `ChildAgentConfig` becomes a tool schema for the root agent
- The `isinstance(config, dict)` guard
- How the child's `run()` result is returned as the tool result

---

## Build this

Build a 2-level hierarchy for a "software code review" task:

**Level 1 (root):** receives the code, delegates to three specialist reviewers, synthesises their feedback into a final review.

**Level 2 (specialists):**
- `security_reviewer` — checks for injection attacks, hardcoded secrets, unsafe operations
- `performance_reviewer` — checks for N+1 queries, unnecessary loops, memory leaks
- `style_reviewer` — checks for naming conventions, function length, docstrings

```python
code_to_review = """
def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    results = db.execute(query)
    data = []
    for row in results:
        user = {'id': row[0], 'name': row[1], 'email': row[2]}
        data.append(user)
    return data
"""
```

The root agent should call all three reviewers, collect their findings, and produce a unified "Code Review Report" with prioritised issues.

---

## Key terms

| Term | Meaning |
|------|---------|
| Hierarchy | Recursive agent delegation — agents calling agents |
| `ChildAgentConfig` | Configuration for one child: client + config + description |
| Depth trap | More levels = exponential cost and failure surface |
| Natural language handoff | Children return string results to parents |

---

## Connects to

- **Lesson 8** — OrchestratorWorker is a flat 2-level hierarchy where children are simple loops
- **Lesson 11** — use evaluation to validate child outputs before the parent continues
- **Lesson 15** — wrap child agent calls with retry logic for production reliability

---

*Lesson 9 of 20 — Applied AI Engineering*
