"""
=============================================================================
Example 11 — Agentic RAG (multi-round retrieval with self-evaluation)
=============================================================================

WHAT THIS EXAMPLE TEACHES
--------------------------
Agentic RAG vs naive RAG:

  NAIVE RAG (example 10):
      question → one search → inject results → answer
      Problem: what if the first search misses something? LLM may hallucinate.

  AGENTIC RAG (this example):
      question → PLAN sub-topics → SEARCH each sub-topic → EVALUATE whether
      you have enough → RE-SEARCH if gaps remain → SYNTHESISE grounded answer

The agent has AGENCY over retrieval. It is not a one-shot wrapper around
search — it reasons about what it does and doesn't know.

THE AGENTIC LOOP
----------------
  ┌───────────────────────────────────────────────────────────────────┐
  │  OBSERVE   →  get_document_stats(): what is in the knowledge base? │
  │  PLAN      →  identify sub-topics the question covers              │
  │  SEARCH    →  search_knowledge() for each sub-topic               │
  │  KEYWORD   →  search_keyword() for exact terms (names, codes)     │
  │  EVALUATE  →  check_sufficiency(): do I have enough to answer?    │
  │  RE-SEARCH →  if gaps remain, run more targeted searches           │
  │  SYNTHESISE→  write a grounded answer with cited sources           │
  └───────────────────────────────────────────────────────────────────┘

THE KEY TOOL: check_sufficiency
--------------------------------
This tool is the heart of agentic RAG. The agent passes its question and
all retrieved text to the tool. An LLM evaluates whether the context is
complete and either confirms readiness or returns specific follow-up queries.

Without this tool the agent might stop after one search. With it, the agent
knows when it's done — and what to look for when it isn't.

TWO RETRIEVAL MODES
--------------------
  semantic  — cosine similarity on embeddings. Best for concept questions
              ("what is X", "how does Y work"). Finds meaning, not exact words.
  keyword   — BM25 scoring (TF-IDF-style). Best for exact terms: product
              names, error codes, proper nouns. Immune to paraphrase variation.

Using both covers each retrieval mode's blind spots.

RUNNING
-------
    python examples/11_agentic_rag.py

REQUIREMENTS
------------
    pip install -r requirements.txt
    pip install -e .
    export ANTHROPIC_API_KEY=sk-ant-...
=============================================================================
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastmcp import FastMCP

from mcp_agent_framework import AnthropicClient, AgentConfig, Message, AgenticRAGStore
from mcp_agent_framework.patterns import SingleAgentLoop

# AgenticRAGStore provides: search_semantic, search_keyword, search_hybrid (RRF)
# All live in mcp_agent_framework.rag — import directly to customise:
#   from mcp_agent_framework.rag import AgenticRAGStore, RecursiveTextChunker


# =============================================================================
# SELF-EVALUATION — check_sufficiency
#
# The agent calls this tool after each retrieval round. An LLM assesses
# whether the collected context is complete enough to answer the question.
# If not, it returns specific follow-up queries so the agent knows what
# to search for next. This is what makes the loop "agentic".
# =============================================================================

_SUFFICIENCY_CLIENT = None  # lazily initialised


async def evaluate_sufficiency(question: str, context: str) -> dict[str, Any]:
    """Ask an LLM: given this context, can we fully answer this question?"""
    global _SUFFICIENCY_CLIENT
    if _SUFFICIENCY_CLIENT is None:
        _SUFFICIENCY_CLIENT = AnthropicClient("claude-haiku-4-5-20251001")

    prompt = (
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Does the context above contain enough information to fully answer the question?\n"
        "If YES: reply with exactly 'SUFFICIENT'.\n"
        "If NO: reply with 'INSUFFICIENT: ' followed by 1-3 specific follow-up search "
        "queries that would fill the gaps, separated by ' | '.\n"
        "Reply with nothing else."
    )
    response = await _SUFFICIENCY_CLIENT.complete(
        [Message(role="user", content=prompt)]
    )
    text = (response.content or "").strip()
    if text.upper().startswith("SUFFICIENT"):
        return {"sufficient": True, "follow_up_queries": []}
    # parse follow-up queries from "INSUFFICIENT: q1 | q2 | q3"
    queries_part = text.split(":", 1)[-1].strip() if ":" in text else ""
    follow_ups = [q.strip() for q in queries_part.split("|") if q.strip()]
    return {"sufficient": False, "follow_up_queries": follow_ups}


# =============================================================================
# MCP SERVER — tools the agent uses during the agentic loop
# =============================================================================

rag_store = AgenticRAGStore()
app       = FastMCP("agentic_rag_server")


@app.tool
async def get_document_stats() -> str:
    """
    Show what is in the knowledge base: number of chunks and source files.
    Call this first to understand what knowledge is available.
    """
    s = rag_store.stats()
    sources = "\n".join(f"  - {src}" for src in s["sources"])
    return f"Total chunks: {s['total_chunks']}\nSources:\n{sources}"


@app.tool
async def search_knowledge(query: str, top_k: int = 5) -> str:
    """
    Semantic search: finds chunks conceptually related to the query.
    Best for: concept questions, 'how does X work', 'what is Y'.
    """
    results = await rag_store.search_semantic(query, top_k=top_k)
    if not results:
        return "No relevant results found."
    parts = [f"[{r.source}]\n{r.text}" for r in results]
    return "\n\n---\n\n".join(parts)


@app.tool
async def search_keyword(query: str, top_k: int = 5) -> str:
    """
    Keyword search (BM25): finds chunks containing the exact query terms.
    Best for: proper nouns, product names, error codes, technical acronyms.
    """
    results = rag_store.search_keyword(query, top_k=top_k)
    if not results:
        return "No keyword matches found."
    parts = [f"[{r.source}]\n{r.text}" for r in results]
    return "\n\n---\n\n".join(parts)


@app.tool
async def check_sufficiency(question: str, context_so_far: str) -> str:
    """
    Self-evaluation: given the context retrieved so far, is it enough to
    fully answer the question? Returns 'SUFFICIENT' or follow-up queries.
    Call after each search round before writing the final answer.
    """
    result = await evaluate_sufficiency(question, context_so_far)
    if result["sufficient"]:
        return "SUFFICIENT — you have enough context to answer the question."
    queries = " | ".join(result["follow_up_queries"])
    return f"INSUFFICIENT — search for these to fill the gaps: {queries}"


# =============================================================================
# SAMPLE DOCUMENTS (same corpus as example 10)
# =============================================================================

DOCUMENTS = {
    "vector_databases.txt": """
Vector databases store data as high-dimensional vectors (embeddings) and
retrieve by similarity rather than exact match. Popular options include
Pinecone, Weaviate, Qdrant, Milvus, and pgvector (Postgres extension).

Unlike relational databases that use B-tree indexes for exact lookups,
vector databases use approximate nearest-neighbour (ANN) algorithms such as
HNSW (Hierarchical Navigable Small World) or IVF (Inverted File Index).
HNSW offers excellent recall with low query latency and is the most widely
adopted algorithm today.

Similarity is measured by distance metrics: cosine similarity (angle between
vectors, range 0–1), dot product (fast, requires normalised vectors), and
Euclidean distance (L2, sensitive to vector magnitude). Cosine similarity is
the default for text embeddings because it ignores length.

For a startup, Qdrant is a strong choice: open-source, fast, supports both
in-memory and persistent storage, and has a generous free cloud tier.
Pinecone is fully managed and requires zero infrastructure. pgvector works
inside Postgres — ideal when you already have a Postgres database.
""",
    "chunking_strategies.txt": """
Chunking is the process of splitting documents into smaller pieces before
embedding. Chunk quality directly determines retrieval quality.

Fixed-size chunking splits by character count with optional overlap. Fastest
but cuts sentences mid-way, producing semantically noisy embeddings.

Recursive text chunking tries paragraph breaks first, then sentence breaks,
then word breaks. Produces coherent chunks without an embedding model.
Best general-purpose choice for plain text, markdown, or HTML.

Semantic chunking uses sentence embeddings to find topic boundaries.
Sentences with similar embeddings are grouped; a large cosine-distance jump
marks a chunk boundary. Highest quality but requires an embedding model.

Chunk size trade-offs: 400–512 characters with 10–15% overlap is the sweet
spot for most text corpora.
""",
    "rag_patterns.txt": """
RAG (Retrieval-Augmented Generation) grounds LLM responses in a document
corpus. The basic pattern: question → retrieve chunks → inject into prompt
→ LLM answers using retrieved content.

HyDE (Hypothetical Document Embeddings): ask the LLM to write a hypothetical
answer, embed THAT answer, then retrieve using its embedding. The hypothesis
is often semantically closer to the actual document than the raw question.

Multi-query retrieval generates 3–5 paraphrases of the question, retrieves
for each, then deduplicates and re-ranks. Reduces query-wording sensitivity.

Hybrid retrieval combines semantic and keyword scores using Reciprocal Rank
Fusion (RRF): score = Σ 1/(rank + k). Consistently outperforms either alone.

Self-RAG adds a sufficiency check: after retrieval the model assesses whether
it has enough to answer. If not, it formulates follow-up queries. This is the
step that turns basic RAG into agentic RAG.

Agentic RAG gives the agent full control over retrieval: it plans what to
search, searches multiple times, evaluates sufficiency, then synthesises.
""",
}


# =============================================================================
# MAIN
# =============================================================================

_AGENTIC_SYSTEM_PROMPT = """\
You are a senior AI engineer answering technical questions about RAG systems.

AGENTIC SEARCH LOOP — follow this exact process for every question:

1. OBSERVE  — call get_document_stats() to see what sources are available.
2. PLAN     — identify the 2–4 sub-topics or concepts the question covers.
3. SEARCH   — for each sub-topic, call search_knowledge() with a focused query.
              If the question contains exact product names or acronyms, also
              call search_keyword() for those terms.
4. EVALUATE — call check_sufficiency(question, <all_retrieved_text_joined>).
              If it returns INSUFFICIENT, run additional searches on the
              suggested follow-up queries before continuing.
5. SYNTHESISE — write a grounded, concise answer citing source file names.
              Do not add information that was not in the retrieved context.
"""


async def main() -> None:
    # ── Step 1: Ingest ───────────────────────────────────────────────────────
    print("Ingesting documents...")
    for source, text in DOCUMENTS.items():
        n = await rag_store.add_document(text, source)
        print(f"  {source} → {n} chunks")

    # ── Step 2: Build agentic agent ──────────────────────────────────────────
    config = AgentConfig(
        mcp_server_config=app,
        system_prompt=_AGENTIC_SYSTEM_PROMPT,
        max_iterations=20,   # allow multiple search rounds
    )
    agent = SingleAgentLoop(
        llm_client=AnthropicClient("claude-sonnet-4-6"),
        config=config,
    )

    # ── Step 3: Ask a multi-faceted question that needs multiple searches ─────
    question = (
        "I'm building a production RAG system in Python. Please explain: "
        "(1) which chunking strategy to use and why, "
        "(2) which vector database fits a small startup, "
        "(3) how agentic RAG improves on basic RAG, and "
        "(4) what retrieval technique works best for exact product names."
    )

    print(f"\n{'=' * 60}")
    print("Question (requires 4 sub-topics — watch the agent plan & search):")
    print(f"\n{question}")
    print("\n[The agent will: observe → plan → search × N → evaluate → synthesise]\n")
    print("-" * 60)

    answer = await agent.run(question)
    print(f"\nFinal Answer:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
