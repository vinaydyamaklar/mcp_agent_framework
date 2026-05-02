"""
=============================================================================
LangGraph + MCP Agent Framework — 01: Checkpointing
=============================================================================

WHAT THIS EXAMPLE TEACHES
--------------------------
By default, every SingleAgentLoop.run() call is stateless — when it returns,
the conversation history is gone. If the process restarts mid-task, you lose
everything.

LangGraph's checkpointer saves the full graph state to a backend (memory,
SQLite, Postgres) after every node execution. Any run can be resumed by
passing the same thread_id, even after a process restart.

HOW IT WORKS
------------
  1. Wrap SingleAgentLoop inside a LangGraph node function
  2. The state TypedDict holds conversation history across invocations
  3. Compile the graph with a checkpointer
  4. Every ainvoke() with the same thread_id picks up where the last left off

CHECKPOINTER OPTIONS
--------------------
  MemorySaver       — in-process memory (demo only, lost on restart)
  AsyncSqliteSaver  — persists to a SQLite file (survives restarts)
  PostgresSaver     — production-grade, shareable across processes

RUNNING
-------
    pip install langgraph aiosqlite
    pip install -r ../../requirements.txt && pip install -e ../..
    export ANTHROPIC_API_KEY=sk-ant-...
    python 01_checkpointing.py
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
# In-process MCP server — simple knowledge tool
# =============================================================================

app = FastMCP("checkpointing_demo")

FACTS = {
    "python": "Python was created by Guido van Rossum, first released in 1991.",
    "mcp":    "MCP (Model Context Protocol) lets LLMs call external tools via a standard protocol.",
    "rag":    "RAG (Retrieval-Augmented Generation) grounds LLM answers in a document corpus.",
}

@app.tool
async def lookup_fact(topic: str) -> str:
    """Look up a fact about a given topic."""
    key = topic.lower().strip()
    return FACTS.get(key, f"No fact found for '{topic}'. Known topics: {list(FACTS)}")


# =============================================================================
# LangGraph state
#
# `messages` uses add_messages — a LangGraph reducer that appends new messages
# to the list rather than replacing it. This is how conversation history
# accumulates across checkpointed invocations.
# =============================================================================

class AgentState(TypedDict):
    # add_messages reducer: new messages are appended, not overwritten
    messages: Annotated[list, add_messages]
    last_response: str


# =============================================================================
# LangGraph node — wraps SingleAgentLoop
#
# Every field returned from a node function updates the state.
# LangGraph checkpoints the state AFTER this function returns.
# =============================================================================

async def run_agent(state: AgentState) -> dict:
    """Run the agent on the latest user message, preserving full history."""
    # Convert LangGraph messages back to framework Message objects
    history = [
        Message(role=m["role"], content=m["content"])
        for m in state["messages"][:-1]   # all except the latest user turn
        if isinstance(m, dict)
    ]
    # The latest message is the current user turn
    latest = state["messages"][-1]
    user_message = latest["content"] if isinstance(latest, dict) else latest.content

    config = AgentConfig(
        mcp_server_config=app,
        system_prompt=(
            "You are a helpful assistant with access to a fact lookup tool. "
            "Always use lookup_fact before answering topic questions."
        ),
        max_iterations=5,
    )

    response = await SingleAgentLoop(
        llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
        config=config,
    ).run(user_message, history=history)

    return {
        # Add the assistant's response to the persisted message history
        "messages": [{"role": "assistant", "content": response}],
        "last_response": response,
    }


# =============================================================================
# Build and compile the graph with a checkpointer
# =============================================================================

def build_graph(checkpointer):
    builder = StateGraph(AgentState)
    builder.add_node("agent", run_agent)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# Demo
# =============================================================================

async def main() -> None:
    # MemorySaver = in-memory (demo). Swap for AsyncSqliteSaver for persistence:
    #   from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    #   async with AsyncSqliteSaver.from_conn_string("agent_state.db") as checkpointer:
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer)

    # thread_id identifies this conversation. Reuse the same ID to resume.
    config = {"configurable": {"thread_id": "demo-session-1"}}

    print("=" * 60)
    print("Turn 1 — initial question")
    print("=" * 60)
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "What is Python?"}]},
        config,
    )
    print(f"Agent: {result['last_response']}")

    print("\n" + "=" * 60)
    print("Turn 2 — follow-up (agent has memory of Turn 1 via checkpoint)")
    print("=" * 60)
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "And what is MCP?"}]},
        config,
    )
    print(f"Agent: {result['last_response']}")

    # Inspect the saved state
    saved = graph.get_state(config)
    print(f"\n[Checkpoint] {len(saved.values['messages'])} messages persisted for thread '{config['configurable']['thread_id']}'")
    print("[Checkpoint] To resume after a restart: reuse the same thread_id with AsyncSqliteSaver")


if __name__ == "__main__":
    asyncio.run(main())
