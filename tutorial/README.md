# Applied AI Engineering — Tutorial

**21 lessons. One framework. From zero to production.**

This tutorial uses the `mcp_agent_framework` as the textbook. Every lesson has a concept explanation, source files to read, an example to run, and an exercise to build.

---

## Curriculum

| # | Lesson | What you'll understand after |
|---|--------|------------------------------|
| [1](lesson_01/README.md) | Why agents exist | The problem LLMs alone cannot solve |
| [2](lesson_02/README.md) | Types — the shared language | How every piece of the framework talks to each other |
| [3](lesson_03/README.md) | Clients — talking to LLMs | How Anthropic, OpenAI, Gemini actually work under the hood |
| [4](lesson_04/README.md) | MCP — tools for your agent | What MCP is and why it changed everything |
| [5](lesson_05/README.md) | The Single Agent Loop | The ReAct loop — foundation of every agent |
| [6](lesson_06/README.md) | Tool calling deep dive | How an LLM decides to use a tool and what happens next |
| [7](lesson_07/README.md) | Memory | SemanticMemory vs EpisodicMemory — what agents remember and why |
| [8](lesson_08/README.md) | Orchestrator pattern | One brain, many workers |
| [9](lesson_09/README.md) | Hierarchy pattern | Agents delegating to agents |
| [10](lesson_10/README.md) | Human-in-the-Loop | When humans stay in control |
| [11](lesson_11/README.md) | Evaluation | How to measure if your agent is actually good |
| [12](lesson_12/README.md) | EvaluatorOptimizer | Agents that critique and improve their own output |
| [13](lesson_13/README.md) | PlannerExecutor | Thinking before doing |
| [14](lesson_14/README.md) | Parallel pattern | Doing many things at once |
| [15](lesson_15/README.md) | Resilience | Retry, circuit breakers — surviving the real world |
| [16](lesson_16/README.md) | Observability | Seeing inside a running agent |
| [17](lesson_17/README.md) | RAG | Making agents know things |
| [18](lesson_18/README.md) | Agentic RAG | Agents that decide how to retrieve |
| [19](lesson_19/README.md) | LangGraph integration | Production-grade agents |
| [20](lesson_20/README.md) | Skills | Composable named capabilities — the final layer |
| [21](lesson_21/README.md) | Multi-modal pipeline | Combining LLMs, image models, and local tools in one agent |

---

## How to use this tutorial

1. Work through lessons in order — each one builds on the previous
2. For each lesson: read the concept → read the source files → run the example → build the exercise
3. Don't skip the exercises. Reading is not the same as building.

**Estimated time:** 40–60 hours of focused study and hands-on work.

---

## Setup

```bash
cd /path/to/mcp_agent_framework
pip install -r requirements.txt
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
# optionally:
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
```

---

## Units

| Unit | Lessons | Theme |
|------|---------|-------|
| 1 — Foundations | 1–4 | Types, clients, MCP, why agents exist |
| 2 — Core Patterns | 5–6 | The loop, tool calling |
| 3 — Multi-Agent | 7–10 | Memory, orchestration, hierarchy, human oversight |
| 4 — Quality | 11–13 | Evaluation, self-improvement, planning |
| 5 — Scale | 14–16 | Parallelism, resilience, observability |
| 6 — Knowledge | 17–18 | RAG, Agentic RAG |
| 7 — Production | 19–21 | LangGraph, Skills, Multi-modal pipelines |

---

*mcp_agent_framework — April 2026*
