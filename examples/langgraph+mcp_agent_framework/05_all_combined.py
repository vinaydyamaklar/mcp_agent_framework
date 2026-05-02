"""
=============================================================================
LangGraph + MCP Agent Framework — 05: All Capabilities Combined
=============================================================================

WHAT THIS EXAMPLE BUILDS
-------------------------
A production-grade research assistant that uses all four capabilities:

  ✓ CHECKPOINTING   — every step saved to SQLite; survives process restarts
  ✓ INTERRUPTS      — human approves before the agent sends a final report
  ✓ TIME TRAVEL     — inspect history, rewind and rerun with a different model
  ✓ STREAMING       — tokens stream to the user as the agent types

THE GRAPH
---------
  START
    │
    ▼
  [research_node]   ← SingleAgentLoop searches knowledge base, streams tokens
    │
    ▼
  [review_node]     ← interrupt(): human approves/rejects the draft report
    │
    ▼
  [finalise_node]   ← formats and saves the approved report
    │
    ▼
  END

ARCHITECTURE LESSON
--------------------
  This framework handles:   MCP connections, provider normalisation, patterns
  LangGraph handles:        state persistence, interrupt/resume, streaming transport

  They do not overlap — LangGraph is the outer shell, this framework is the engine.

RUNNING
-------
    pip install "langgraph[sqlite]"       # includes aiosqlite
    pip install -r ../../requirements.txt && pip install -e ../..
    export ANTHROPIC_API_KEY=sk-ant-...
    python 05_all_combined.py
=============================================================================
"""

from __future__ import annotations

import asyncio
import os
from typing import Annotated, Any, TypedDict

import anthropic
from fastmcp import FastMCP
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, StreamWriter, interrupt

from mcp_agent_framework import AgentConfig, AnthropicClient
from mcp_agent_framework.memory import SemanticMemory
from mcp_agent_framework.patterns import SingleAgentLoop
from mcp_agent_framework.types import Message

# =============================================================================
# Knowledge base + MCP server
# =============================================================================

knowledge = SemanticMemory()
app       = FastMCP("combined_demo")

ARTICLES = {
    "vector_search": (
        "Vector search finds semantically similar content using embeddings. "
        "HNSW is the most widely used ANN algorithm — excellent recall at low latency. "
        "Cosine similarity is the standard distance metric for text embeddings."
    ),
    "chunking": (
        "Chunking splits documents before embedding. Recursive text chunking respects "
        "paragraph and sentence boundaries — best general-purpose choice. "
        "Chunk size 400–512 characters with 10–15% overlap is a reliable default."
    ),
    "agentic_rag": (
        "Agentic RAG gives the agent control over retrieval: it plans searches, "
        "evaluates sufficiency, re-searches if gaps remain, and synthesises results. "
        "The key addition over naive RAG is the self-evaluation loop."
    ),
    "langgraph": (
        "LangGraph models agents as stateful graphs. Key features: checkpointing "
        "(persist state to SQLite/Postgres), interrupts (pause for human review), "
        "time travel (replay from any prior checkpoint), and streaming."
    ),
}


@app.tool
async def search_knowledge(query: str, top_k: int = 3) -> str:
    """Search the knowledge base for relevant information."""
    results = await knowledge.search(query, top_k=top_k)
    if not results:
        return "No relevant results found."
    return "\n\n".join(
        f"[{r.metadata.get('source', '?')}]\n{r.content}" for r in results
    )


@app.tool
async def list_topics() -> str:
    """List all available topics in the knowledge base."""
    return f"Available topics: {', '.join(ARTICLES)}"


# =============================================================================
# Streaming LLM helper (same pattern as example 04)
# =============================================================================

async def _stream_llm(prompt: str, system: str, writer: StreamWriter) -> str:
    """Call Anthropic streaming API and push tokens via LangGraph's writer."""
    client   = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    full     = ""
    async with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for chunk in stream.text_stream:
            writer(chunk)
            full += chunk
    return full


# =============================================================================
# State
# =============================================================================

class ResearchState(TypedDict):
    messages:       Annotated[list, add_messages]
    research_query: str
    draft_report:   str
    final_report:   str
    approved:       bool


# =============================================================================
# Node 1 — research_node
#
# Uses SingleAgentLoop to search the knowledge base and draft a report.
# Tokens stream to the user as they are generated.
# =============================================================================

async def research_node(state: ResearchState, writer: StreamWriter) -> dict:
    """
    Search the knowledge base and draft a report.
    Tokens are streamed to the caller via writer() as the agent types.
    """
    query = state["research_query"]

    # ── Phase 1: use SingleAgentLoop for multi-tool research (no streaming) ──
    config = AgentConfig(
        mcp_server_config=app,
        system_prompt=(
            "You are a research assistant. Use list_topics() first, then "
            "search_knowledge() for each relevant sub-topic. Gather all facts needed "
            "to write a comprehensive report. Return ALL retrieved facts, nothing else."
        ),
        max_iterations=8,
    )
    gathered_facts = await SingleAgentLoop(
        llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
        config=config,
    ).run(f"Research everything relevant to: {query}")

    # ── Phase 2: stream the actual report writing ─────────────────────────────
    writer("\n[Writing report — streaming tokens as generated]\n\n")
    draft = await _stream_llm(
        prompt=(
            f"Based on these research facts, write a clear, well-structured report "
            f"answering: '{query}'\n\nFacts:\n{gathered_facts}"
        ),
        system="You are a technical writer. Write a concise, accurate report with headers.",
        writer=writer,
    )

    return {
        "messages":     [{"role": "assistant", "content": draft}],
        "draft_report": draft,
    }


# =============================================================================
# Node 2 — review_node
#
# Human reviews the draft report before it is finalised.
# interrupt() pauses the graph — state is saved, process can exit.
# Resume with Command(resume={"approved": True/False, "notes": "..."}).
# =============================================================================

def review_node(state: ResearchState) -> dict:
    """Pause for human review of the draft report."""
    decision = interrupt({
        "message":      "Draft report ready for review. Approve to finalise.",
        "draft_report": state["draft_report"][:500] + "...",   # preview
        "instructions": "Pass Command(resume={'approved': True}) to approve, "
                        "or Command(resume={'approved': False, 'notes': 'reason'}) to reject.",
    })

    approved = decision.get("approved", False)
    notes    = decision.get("notes", "")
    return {
        "approved": approved,
        "messages": [{"role": "assistant", "content": f"Review decision: {'approved' if approved else 'rejected'}. {notes}"}],
    }


# =============================================================================
# Node 3 — finalise_node
#
# Runs only after human approval. Formats the final report.
# =============================================================================

async def finalise_node(state: ResearchState, writer: StreamWriter) -> dict:
    """Format and return the approved report, or explain the rejection."""
    if not state.get("approved", False):
        msg = "Report was not approved. Please revise and resubmit."
        writer(msg)
        return {"final_report": msg, "messages": [{"role": "assistant", "content": msg}]}

    writer("\n[Finalising approved report — streaming]\n\n")
    final = await _stream_llm(
        prompt=(
            f"Format this report cleanly with a title, introduction, and clear sections:\n\n"
            f"{state['draft_report']}"
        ),
        system="You are a document formatter. Add a title and structure. Keep all content.",
        writer=writer,
    )

    return {
        "final_report": final,
        "messages": [{"role": "assistant", "content": final}],
    }


# =============================================================================
# Build graph
# =============================================================================

def build_graph(checkpointer):
    builder = StateGraph(ResearchState)
    builder.add_node("research",  research_node)
    builder.add_node("review",    review_node)
    builder.add_node("finalise",  finalise_node)
    builder.add_edge(START,       "research")
    builder.add_edge("research",  "review")
    builder.add_edge("review",    "finalise")
    builder.add_edge("finalise",  END)
    # interrupt_before="review" is optional — interrupt() inside the node also works
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# Demo
# =============================================================================

async def main() -> None:
    # ── Ingest knowledge base ─────────────────────────────────────────────────
    print("Loading knowledge base...")
    for source, text in ARTICLES.items():
        await knowledge.add(text, metadata={"source": source})

    # ── Use MemorySaver (swap for AsyncSqliteSaver for real persistence) ──────
    # For SQLite persistence:
    #   from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    #   async with AsyncSqliteSaver.from_conn_string("research_agent.db") as cp:
    checkpointer = MemorySaver()
    graph        = build_graph(checkpointer)
    config       = {"configurable": {"thread_id": "research-001"}}

    query = "How does agentic RAG work and how does LangGraph improve it?"
    print(f"\nQuery: {query}\n")

    # ── Phase 1: Research + streaming draft ──────────────────────────────────
    print("=" * 60)
    print("CAPABILITY 1 + 4: Checkpointing + Streaming")
    print("Agent researches and streams the draft report token-by-token")
    print("=" * 60)

    async for chunk in graph.astream(
        {
            "messages":       [{"role": "user", "content": query}],
            "research_query": query,
            "draft_report":   "",
            "final_report":   "",
            "approved":       False,
        },
        config,
        stream_mode="custom",
    ):
        print(chunk, end="", flush=True)

    # ── Phase 2: Human review (interrupt) ────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("CAPABILITY 2: Human-in-the-loop Interrupt")
    print("Graph is paused — human reviewing the draft")
    print("=" * 60)

    saved = graph.get_state(config)
    print(f"[State saved] {len(saved.values.get('messages', []))} messages checkpointed")
    print("[Simulating human review — approving in 1 second...]")
    await asyncio.sleep(1)

    # Resume with approval — tokens stream again during finalise_node
    print("\n" + "=" * 60)
    print("Human approved — resuming and streaming final report")
    print("=" * 60)
    async for chunk in graph.astream(
        Command(resume={"approved": True}),
        config,
        stream_mode="custom",
    ):
        print(chunk, end="", flush=True)

    # ── Phase 3: Time travel ──────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("CAPABILITY 3: Time Travel — full checkpoint history")
    print("=" * 60)
    history = list(graph.get_state_history(config))
    print(f"\n  {len(history)} checkpoints saved for thread '{config['configurable']['thread_id']}'")
    for i, snap in enumerate(history[:4]):    # show first 4
        cid  = snap.config["configurable"].get("checkpoint_id", "—")[:20]
        msgs = len(snap.values.get("messages", []))
        next_ = str(snap.next) if snap.next else "(done)"
        print(f"  [{i}] checkpoint={cid}...  messages={msgs}  next={next_}")

    if len(history) >= 2:
        print(f"\n  To branch from checkpoint[1]: ")
        print(f"  graph.update_state(new_config, history[1].values)")
        print(f"  graph.ainvoke(new_input, new_config)  ← runs a new branch")

    print("\n[Done] All four capabilities demonstrated.")


if __name__ == "__main__":
    asyncio.run(main())
