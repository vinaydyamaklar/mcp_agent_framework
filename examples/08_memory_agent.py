"""
=============================================================================
Example 08 — Memory Agent (EpisodicMemory + SemanticMemory)
=============================================================================

WHAT THIS EXAMPLE TEACHES
--------------------------
Agents are stateless by default: each `agent.run()` call starts fresh with
no knowledge of prior conversations. This example shows how to give an agent
persistent memory using two complementary memory stores:

    EpisodicMemory  — an ordered event log with recency-biased search.
                      Good for: conversation history, recent observations.

    SemanticMemory  — cosine-similarity search over stored text.
                      Good for: finding facts by meaning, not just keywords.

The key insight: these stores live OUTSIDE the agent. They persist across
multiple `agent.run()` calls. The agent accesses them through MCP tools
(`remember` and `recall`), so from the agent's perspective they are just
tools — but from the developer's perspective they are long-lived Python
objects that accumulate knowledge over time.

THE TWO-TURN STRUCTURE
----------------------
Turn 1:  Tell the agent three facts about yourself. The agent calls
         `remember(fact)` for each fact, writing into both stores.

Turn 2:  Ask the agent what it remembers. The agent calls `recall(query)`,
         which reads back from both stores. You will see it correctly
         recall the name, preference, and project from Turn 1.

The `history` parameter on the second `agent.run()` call adds the Turn 1
exchange to the conversation context. This lets the agent understand that
Turn 2 is a follow-up, not a completely new conversation.

HOW EpisodicMemory WORKS
-------------------------
    await episodic.add(text, metadata={...})
        Appends a timestamped entry. If max_entries is set and the store is
        full, the oldest entry is evicted (sliding window).

    await episodic.get_recent(n)
        Returns the n most recent entries, newest first.
        Perfect for injecting a short-term context window into a prompt.

    await episodic.search(query, top_k=5)
        Scores each entry by (keyword_overlap + recency_bonus).
        Recent entries get a small bonus even if they don't match the query.

HOW SemanticMemory WORKS
------------------------
    await semantic.add(text, metadata={...})
        Stores the entry. With the default bag-of-words embedder, the actual
        embedding is deferred to search time so the shared vocabulary is
        always up to date.

    await semantic.search(query, top_k=3)
        Builds TF vectors for all entries + query over a shared vocabulary,
        then ranks by cosine similarity. Pass embed_fn=your_async_fn to use
        real embeddings (e.g. OpenAI text-embedding-3-small) in production.

WHY USE BOTH?
-------------
They complement each other:
    - EpisodicMemory preserves TIME ORDER and rewards recency.
    - SemanticMemory finds the MOST RELEVANT facts by meaning.

The `recall` tool queries both and merges the results, so the agent always
gets both "what just happened" and "what is most relevant".

RUNNING
-------
    python examples/08_memory_agent.py

REQUIREMENTS
------------
    pip install -r requirements.txt && pip install -e .
    export ANTHROPIC_API_KEY=sk-ant-...
=============================================================================
"""

import asyncio
import logging

from fastmcp import FastMCP

from mcp_agent_framework import AnthropicClient, AgentConfig, Message
from mcp_agent_framework.patterns import SingleAgentLoop
from mcp_agent_framework.memory import EpisodicMemory, SemanticMemory

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s — %(message)s",
)

# ------------------------------------------------------------------
# Memory stores — created once, shared across all agent.run() calls.
#
# This is the critical design: the stores live outside the agent. They
# are NOT recreated on each run(). Facts written in Turn 1 are still
# present when Turn 2 queries them.
# ------------------------------------------------------------------
episodic = EpisodicMemory(max_entries=100)
semantic = SemanticMemory()   # bag-of-words by default — zero extra deps

# ------------------------------------------------------------------
# MCP server — exposes two tools to the agent:
#   remember(fact)    → writes into both stores
#   recall(query)     → reads from both stores
# ------------------------------------------------------------------
app = FastMCP("memory_tools")


@app.tool
async def remember(fact: str) -> str:
    """Store a fact in memory for later retrieval."""
    await episodic.add(fact, metadata={"type": "fact"})
    await semantic.add(fact, metadata={"type": "fact"})
    return f"Remembered: {fact}"


@app.tool
async def recall(query: str) -> str:
    """Search memory for relevant facts."""
    # get_recent() gives the most recent events — good for "what just happened"
    recent   = await episodic.get_recent(5)
    # semantic.search() finds facts by meaning — good for "what do I know about X"
    relevant = await semantic.search(query, top_k=3)

    parts = []
    if recent:
        parts.append(
            "Recent facts:\n" + "\n".join(f"- {e.content}" for e in recent)
        )
    if relevant:
        parts.append(
            "Relevant facts:\n" + "\n".join(f"- {e.content}" for e in relevant)
        )

    return "\n\n".join(parts) if parts else "No relevant memories found."


async def main() -> None:
    # ------------------------------------------------------------------
    # Step 1 — Build the agent.
    #
    # max_iterations=8 allows the agent to call remember() several times
    # in a single run() call (once per fact it decides to store).
    # ------------------------------------------------------------------
    config = AgentConfig(
        mcp_server_config=app,
        system_prompt=(
            "You are a helpful assistant with memory. "
            "When the user shares personal facts, call remember() to store each one. "
            "When the user asks what you know about them, call recall() to retrieve it."
        ),
        max_iterations=8,
    )
    agent = SingleAgentLoop(
        llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
        config=config,
    )

    # ------------------------------------------------------------------
    # Turn 1 — Teach the agent some facts.
    #
    # The agent will call remember() for each fact. After this call the
    # episodic and semantic stores have 3 entries in them.
    # ------------------------------------------------------------------
    turn1_message = "My name is Alex. I prefer Python over JavaScript. I'm building a RAG system."

    print("Turn 1: Teaching the agent some facts")
    print("-" * 40)
    r1 = await agent.run(turn1_message)
    print(r1)

    # ------------------------------------------------------------------
    # Turn 2 — Test memory recall.
    #
    # The `history` argument re-injects Turn 1 so the agent treats Turn 2
    # as a continuation of the same conversation.
    #
    # Crucially, even WITHOUT history the agent COULD recall the facts by
    # calling recall() — the memory stores are still populated. The history
    # just gives it conversational context so it understands the question.
    # ------------------------------------------------------------------
    history = [
        Message(role="user",      content=turn1_message),
        Message(role="assistant", content=r1),
    ]

    print("\nTurn 2: Testing memory recall")
    print("-" * 40)
    r2 = await agent.run(
        "What do you remember about me and my project?",
        history=history,
    )
    print(r2)

    # ------------------------------------------------------------------
    # Bonus: inspect the stores directly to see what was stored.
    # ------------------------------------------------------------------
    print("\n--- Memory store contents (direct inspection) ---")
    all_episodic = await episodic.list_all()
    print(f"EpisodicMemory has {len(all_episodic)} entries:")
    for entry in all_episodic:
        print(f"  [{entry.id[:8]}] {entry.content}")

    all_semantic = await semantic.list_all()
    print(f"SemanticMemory has {len(all_semantic)} entries:")
    for entry in all_semantic:
        print(f"  [{entry.id[:8]}] {entry.content}")


if __name__ == "__main__":
    asyncio.run(main())
