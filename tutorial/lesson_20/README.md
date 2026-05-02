# Lesson 20 — Skills: Composable Agentic Capabilities

**Unit 7: Production Infrastructure**

---

## What you will learn

- What Skills are and why they are the final abstraction layer
- `Skill`, `SkillRegistry`, `SkillAwareAgent` — the three components
- How skills compose with each other
- The industry context: Claude Code skills, OpenAI GPT Actions, LangGraph subgraphs
- The capstone project: a full production agent system using every lesson

---

## The concept

You have reached the last lesson. At this point you know:

- **Types** (L2) — the shared language
- **Clients** (L3) — multi-provider LLM calls
- **MCP** (L4) — tool definitions and execution
- **SingleAgentLoop** (L5) — the ReAct foundation
- **Tool calling** (L6) — the full lifecycle
- **Memory** (L7) — what agents remember
- **Orchestrator** (L8) — one brain, many workers
- **Hierarchy** (L9) — recursive agent delegation
- **HITL** (L10) — human control gates
- **Evaluation** (L11) — quality measurement
- **EvaluatorOptimizer** (L12) — self-improvement loops
- **PlannerExecutor** (L13) — structured execution
- **Parallel** (L14) — fan-out/gather
- **Resilience** (L15) — fault tolerance
- **Observability** (L16) — production debugging
- **RAG** (L17) — knowledge retrieval
- **Agentic RAG** (L18) — intelligent retrieval
- **LangGraph** (L19) — production infrastructure

Skills are the packaging layer. They let you wrap any of the above into a named, discoverable, reusable capability.

---

## The architecture layer

```
Raw API calls
    ↓
Clients (L3)             ← talk to any LLM
    ↓
Tools / MCP (L4)         ← single functions
    ↓
Skills (L20)             ← named capabilities (compose tools + patterns)
    ↓
Patterns (L5–L14)        ← coordination strategies
    ↓
Application
```

A Tool is: `def search_database(query: str) -> str`

A Skill is: `"research_topic" — runs SingleAgentLoop with AgenticRAG over the knowledge base`

A Pattern is: `EvaluatorOptimizer — runs generate → evaluate → rewrite`

Skills compose these. The `research_topic` Skill uses `SingleAgentLoop` (a pattern) which calls `search_semantic` and `search_keyword` (MCP tools).

---

## `Skill` — the unit

```python
from mcp_agent_framework.skills import Skill

async def _research_handler(inputs: dict) -> str:
    topic = inputs["topic"]
    agent = SingleAgentLoop(
        llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
        config=AgentConfig(
            mcp_server_config=agentic_rag_app,
            system_prompt="Research thoroughly using all available search tools.",
        ),
    )
    return await agent.run(f"Research: {topic}")

research_skill = Skill(
    name="research_topic",
    description=(
        "Deep research on any topic using semantic and keyword search. "
        "Returns a comprehensive explanation with sources. "
        "Use this when the user asks to learn about, understand, or investigate a topic."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "What to research"},
        },
        "required": ["topic"],
    },
    handler=_research_handler,
    tags=["research", "read-only"],
)
```

The `name` must be a valid Python identifier (no spaces).
The `description` is what the LLM reads when deciding which skill to invoke.
The `input_schema` tells the LLM what arguments to pass.
The `handler` is the async function that does the actual work.

---

## `SkillRegistry` — the catalog

```python
from mcp_agent_framework.skills import SkillRegistry

registry = SkillRegistry()
registry.register(research_skill)
registry.register(summarise_skill)
registry.register(compare_skill)
registry.register(write_report_skill)
registry.register(fact_check_skill)

# Discover
for skill in registry.list_skills():
    print(f"{skill.name}: {skill.description[:60]}")

# Filter by tag
read_only = registry.list_skills(tag="read-only")

# Invoke directly
result = await registry.invoke("research_topic", {"topic": "vector databases"})

# Invoke in parallel
results = await registry.invoke_many([
    ("research_topic", {"topic": "vector search"}),
    ("research_topic", {"topic": "BM25 keyword search"}),
])
```

---

## `SkillAwareAgent` — the discovery interface

```python
from mcp_agent_framework.skills import SkillAwareAgent

agent = SkillAwareAgent(
    llm_client=AnthropicClient("claude-sonnet-4-6"),
    registry=registry,
    system_prompt=(
        "You are a research and writing assistant. "
        "Use list_skills() to see what capabilities you have. "
        "Compose skills as needed to fulfil the user's request."
    ),
    max_iterations=15,
)
```

The `SkillAwareAgent` auto-wires two MCP tools:

**`list_skills(tag="")`** — the LLM calls this to discover available skills. Returns names, descriptions, and input schemas.

**`invoke_skill(name, inputs_json)`** — the LLM calls this to run a skill. The framework finds the skill in the registry and calls its handler.

The LLM decides which skills to call, in what order, and how to combine results — exactly like it decides which tools to call in `SingleAgentLoop`.

---

## Skill composition

Skills can call other skills. This is how you build complex capabilities from simple ones:

```python
async def _compare_handler(inputs: dict) -> str:
    topic_a = inputs["topic_a"]
    topic_b = inputs["topic_b"]

    # Research both in parallel — skill composition using invoke_many
    research_a, research_b = await registry.invoke_many([
        ("research_topic", {"topic": topic_a}),
        ("research_topic", {"topic": topic_b}),
    ])

    # Synthesise the comparison
    return await registry.invoke("write_report", {
        "content": f"Topic A:\n{research_a}\n\nTopic B:\n{research_b}",
        "format": "comparison",
    })

compare_skill = Skill(
    name="compare_topics",
    description="Research two topics and produce a structured side-by-side comparison.",
    input_schema={"type": "object", "properties": {
        "topic_a": {"type": "string"}, "topic_b": {"type": "string"}
    }, "required": ["topic_a", "topic_b"]},
    handler=_compare_handler,
)
```

---

## Industry context

**Claude Code** has `/skills` — reusable operations for coding tasks. Each skill is a named, invokable capability.

**OpenAI GPT Actions** — named capabilities exposed via API that GPT can discover and call.

**LangGraph subgraphs** — a compiled LangGraph graph can itself be a `Skill` handler:

```python
# A Skill backed by a full LangGraph graph (with checkpointing!)
async def _complex_research(inputs: dict) -> str:
    graph = build_research_graph()  # returns a compiled LangGraph graph
    result = await graph.ainvoke(
        {"task": inputs["topic"]},
        {"configurable": {"thread_id": f"skill-{uuid.uuid4()}"}}
    )
    return result["final_report"]

research_skill = Skill(
    name="deep_research",
    handler=_complex_research,
    ...
)
```

The caller just invokes `"deep_research"` — they don't know or care that it runs a LangGraph graph with checkpointing and streaming internally.

---

## The capstone project

Build a "Research & Publish Platform" using every lesson.

**5 Skills:**

```python
skills = [
    Skill("research_topic",  handler=agentic_rag_loop, ...),   # L18 + L5
    Skill("evaluate_quality", handler=rubric_evaluator, ...),  # L11
    Skill("write_report",    handler=evaluator_optimizer, ...), # L12
    Skill("compare_topics",  handler=parallel_research, ...),  # L14 + L8
    Skill("fact_check",      handler=hitl_verification, ...),  # L10
]
```

**The platform:**

```python
registry = SkillRegistry()
for skill in skills:
    registry.register(skill)

agent = SkillAwareAgent(
    llm_client=AnthropicClient("claude-sonnet-4-6"),
    registry=registry,
    system_prompt="""
    You are a research and publishing assistant. You have access to powerful skills.
    For any research task:
    1. Use research_topic to gather information
    2. Use evaluate_quality to check what you have
    3. Use write_report to produce the final document
    4. Use fact_check for claims that need human verification
    5. Use compare_topics when the user wants a comparison
    """,
)
```

**The task:**

```python
task = """
Research quantum computing and classical computing approaches to optimisation problems.
Compare them side by side.
Write a report aimed at a senior engineering audience.
Evaluate the quality of the report — if it scores below 0.85, tell me what to improve.
Flag any specific numerical claims for fact checking.
"""
result = await agent.run(task)
```

**What this demonstrates:**
- `research_topic` → Agentic RAG (L18) over a knowledge base
- `compare_topics` → `invoke_many` in parallel (L14)
- `write_report` → EvaluatorOptimizer (L12) quality loop
- `evaluate_quality` → RubricEvaluator (L11)
- `fact_check` → HumanInLoopPattern (L10) approval gate
- All coordinated by the LLM via `list_skills()` + `invoke_skill()` (L20)
- Memory via SemanticMemory (L7) in the RAG pipeline
- Resilience (L15) on all tool calls
- Observability (L16) via LoggingTracer on the SkillAwareAgent

---

## Read these files

```
src/mcp_agent_framework/skills/skill.py
src/mcp_agent_framework/skills/skill_aware_agent.py
examples/12_skills.py
```

In `skill.py`, read `Skill.__post_init__` validation and `SkillRegistry.invoke_many`. In `skill_aware_agent.py`, trace how `_build_mcp_server()` creates the `list_skills` and `invoke_skill` tools from the registry.

---

## Run this

```bash
python examples/12_skills.py
```

Watch the four demos:
1. Direct registry invocation
2. Parallel invocation
3. LLM-driven SkillAwareAgent
4. Tag filtering

---

## Key terms

| Term | Meaning |
|------|---------|
| `Skill` | Named, typed, invokable capability with an async handler |
| `SkillRegistry` | Central catalog: register / list / invoke by name |
| `SkillAwareAgent` | Agent that auto-wires registry as `list_skills` + `invoke_skill` tools |
| `invoke_many` | Parallel skill execution via `asyncio.gather` |
| Skill composition | A skill's handler calls other skills |
| Tags | Labels for grouping and filtering skills |

---

## You are done

You have completed the full curriculum. You understand:
- Every layer of the framework, from types to skills
- How to build agents that remember, plan, evaluate, and improve their own output
- How to coordinate multiple agents safely with human oversight
- How to make agents know things via RAG and Agentic RAG
- How to make agents reliable with resilience and observable with tracing
- How to deploy production-grade agents with LangGraph
- How to package it all as reusable named Skills

Build something real.

---

*Lesson 20 of 21 — Applied AI Engineering*
