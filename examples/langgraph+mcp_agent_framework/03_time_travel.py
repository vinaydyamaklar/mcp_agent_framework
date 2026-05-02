"""
=============================================================================
LangGraph + MCP Agent Framework — 03: Time Travel & Branching
=============================================================================

WHAT THIS EXAMPLE TEACHES
--------------------------
With checkpointing, every graph invocation saves a snapshot of the full state
after each node. LangGraph lets you:

  TIME TRAVEL  — list the full history of checkpoints for a thread, inspect
                 the state at any prior point in time
  BRANCHING    — rewind to any checkpoint and run a different path from there,
                 creating a new "branch" without affecting the original thread
  REPLAY       — re-execute from a past checkpoint exactly (useful for
                 debugging: reproduce a bad agent decision and trace it)

WHY THIS IS USEFUL
------------------
  • Agent made a bad decision at step 3? Rewind to step 2, inject a corrected
    system prompt, and re-run. Compare the two branches.
  • A/B test: same conversation, different models. Fork at message 5, run
    branch A with claude-sonnet, branch B with claude-haiku, compare quality.
  • Post-mortem debugging: replay a failed production run step-by-step to
    see exactly what state each node saw.

HOW BRANCHING WORKS
--------------------
  Original thread:   C0 → C1 → C2 → C3   (C = checkpoint)
  Branch from C1:    C0 → C1 → C2b → C3b  (new thread_id, starts from C1's state)

  graph.update_state() injects new state at the chosen checkpoint.
  graph.ainvoke() with the new config runs forward from there.

RUNNING
-------
    pip install langgraph
    pip install -r ../../requirements.txt && pip install -e ../..
    export ANTHROPIC_API_KEY=sk-ant-...
    python 03_time_travel.py
=============================================================================
"""

from __future__ import annotations

import asyncio
from typing import Annotated, TypedDict

from fastmcp import FastMCP
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from mcp_agent_framework import AgentConfig, AnthropicClient
from mcp_agent_framework.patterns import SingleAgentLoop
from mcp_agent_framework.types import Message

# =============================================================================
# MCP server
# =============================================================================

app = FastMCP("time_travel_demo")

KNOWLEDGE = {
    "chunking":   "Fixed-size chunking splits by character count. Recursive chunking respects paragraph and sentence boundaries.",
    "retrieval":  "Semantic retrieval uses cosine similarity. Keyword retrieval uses BM25 scoring. Hybrid combines both.",
    "evaluation": "LLMEvaluator asks an LLM to score content 0–10. RubricEvaluator scores against named criteria with weights.",
}

@app.tool
async def search(topic: str) -> str:
    """Search the knowledge base."""
    for key, val in KNOWLEDGE.items():
        if key in topic.lower():
            return val
    return f"No entry found for '{topic}'. Available: {list(KNOWLEDGE)}"


# =============================================================================
# State
# =============================================================================

class State(TypedDict):
    messages:  Annotated[list, add_messages]
    response:  str
    model_used: str


# =============================================================================
# Node
# =============================================================================

async def agent_node(state: State, model: str = "claude-haiku-4-5-20251001") -> dict:
    """Run the agent on the latest user message."""
    msgs    = state["messages"]
    latest  = msgs[-1]
    content = latest["content"] if isinstance(latest, dict) else latest.content
    history = [
        Message(role=m["role"], content=m["content"])
        for m in msgs[:-1]
        if isinstance(m, dict)
    ]

    config = AgentConfig(
        mcp_server_config=app,
        system_prompt="You are a RAG systems expert. Use the search tool to answer questions.",
        max_iterations=4,
    )
    resp = await SingleAgentLoop(
        llm_client=AnthropicClient(model),
        config=config,
    ).run(content, history=history)

    return {
        "messages":   [{"role": "assistant", "content": resp}],
        "response":   resp,
        "model_used": model,
    }


# =============================================================================
# Build graph
# =============================================================================

def build_graph(checkpointer):
    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# Time travel helpers
# =============================================================================

def print_history(graph, config: dict) -> list:
    """Print all checkpoints for a thread and return the list."""
    history = list(graph.get_state_history(config))
    print(f"\n  {'#':<4} {'Checkpoint ID':<20} {'Next nodes':<20} Messages")
    print(f"  {'-'*4} {'-'*20} {'-'*20} {'-'*30}")
    for i, snapshot in enumerate(history):
        checkpoint_id = snapshot.config["configurable"].get("checkpoint_id", "—")[:18]
        next_nodes    = str(snapshot.next) if snapshot.next else "(done)"
        msg_count     = len(snapshot.values.get("messages", []))
        print(f"  {i:<4} {checkpoint_id:<20} {next_nodes:<20} {msg_count} messages")
    return history


# =============================================================================
# Demo
# =============================================================================

async def main() -> None:
    checkpointer = MemorySaver()
    graph        = build_graph(checkpointer)

    # ── Original run: 3 turns of conversation ────────────────────────────────
    print("=" * 60)
    print("Original thread — 3 turns")
    print("=" * 60)
    config = {"configurable": {"thread_id": "time-travel-demo"}}

    questions = [
        "What is fixed-size chunking?",
        "How does semantic retrieval work?",
        "What is the difference between LLMEvaluator and RubricEvaluator?",
    ]
    for q in questions:
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": q}]},
            config,
        )
        print(f"\nQ: {q}")
        print(f"A: {result['response'][:120]}...")

    # ── Inspect the full checkpoint history ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Time travel — full checkpoint history")
    print("=" * 60)
    history = print_history(graph, config)

    # ── Branch from the second checkpoint ────────────────────────────────────
    # We'll rewind to after Turn 1 and run Turn 2 with a different model,
    # creating a new branch without touching the original thread.
    if len(history) >= 2:
        # history[0] is most recent; history[-1] is oldest
        # We want the state after turn 1 — that's history[-2] (after first agent run)
        branch_point   = history[-2]
        branch_checkpoint_id = branch_point.config["configurable"]["checkpoint_id"]

        print(f"\n" + "=" * 60)
        print(f"Branching from checkpoint after Turn 1")
        print(f"  checkpoint_id = {branch_checkpoint_id[:24]}...")
        print(f"  New branch uses claude-sonnet-4-6 instead of claude-haiku")
        print("=" * 60)

        # New thread_id = new branch — does not overwrite the original
        branch_config = {
            "configurable": {
                "thread_id":     "time-travel-demo-branch-A",
                "checkpoint_id": branch_checkpoint_id,
            }
        }

        # Copy the branched state into the new thread
        branch_state = branch_point.values
        graph.update_state(
            {"configurable": {"thread_id": "time-travel-demo-branch-A"}},
            branch_state,
        )

        # Now run Turn 2 from the branch point — with a different model
        branch_result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": questions[1]}]},
            {"configurable": {"thread_id": "time-travel-demo-branch-A"}},
        )
        print(f"\nBranch response (Turn 2 with sonnet):")
        print(f"{branch_result['response'][:200]}...")
        print(f"\nOriginal Turn 2 vs Branch Turn 2 now exist independently.")
        print("Original thread is unaffected — inspect it again:")
        print_history(graph, config)


if __name__ == "__main__":
    asyncio.run(main())
