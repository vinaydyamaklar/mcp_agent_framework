# LangGraph + MCP Agent Framework

Examples showing how to combine LangGraph's production infrastructure with
this framework's MCP tooling and agent patterns.

## Install

```bash
pip install "langgraph[sqlite]"
pip install -r ../../requirements.txt
pip install -e ../..
export ANTHROPIC_API_KEY=sk-ant-...
```

## Examples

| File | Capability | What it shows |
|---|---|---|
| `01_checkpointing.py` | Checkpointing | Wrap `SingleAgentLoop` in a LangGraph node; state persists across invocations via `MemorySaver` / `AsyncSqliteSaver` |
| `02_interrupts_hitl.py` | Interrupts + Human-in-the-loop | `interrupt()` pauses the graph mid-run; human reviews asynchronously; `Command(resume=...)` continues from the exact pause point |
| `03_time_travel.py` | Time travel + Branching | `get_state_history()` lists all checkpoints; `update_state()` forks a new branch from any prior checkpoint without touching the original |
| `04_streaming.py` | Streaming token-by-token | `StreamWriter` pushes LLM tokens through `graph.astream(stream_mode="custom")` to the caller as they are generated |
| `05_all_combined.py` | All four combined | A research assistant with checkpointing, interrupts, streaming, and time travel in one graph |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  LangGraph (outer shell)                                  │
│  - StateGraph: nodes, edges, conditional routing          │
│  - Checkpointer: SqliteSaver / PostgresSaver              │
│  - interrupt() / Command(resume=...): async human review  │
│  - astream(): token-by-token streaming transport          │
│                                                           │
│   ┌────────────────────────────────────────────────┐      │
│   │  MCP Agent Framework (inner engine)             │      │
│   │  - AnthropicClient / OpenAIClient / GeminiClient│      │
│   │  - SingleAgentLoop, PlannerExecutorPattern, ...  │      │
│   │  - FastMCP: tool definitions and MCP connections │      │
│   └────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────┘
```

LangGraph and this framework do not overlap — LangGraph handles when and how
the agent runs; this framework handles what the agent does.
