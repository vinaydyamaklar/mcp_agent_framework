"""
=============================================================================
Example 10 — RAG (Retrieval-Augmented Generation)
=============================================================================

WHAT THIS EXAMPLE TEACHES
--------------------------
Basic RAG: chunk documents → store in a knowledge base → retrieve on demand
→ LLM answers using only retrieved facts.

THE PIPELINE
------------
  1. CHUNK   — split raw text into overlapping windows (RecursiveTextChunker)
  2. STORE   — embed and store each chunk in SemanticMemory
  3. EXPOSE  — wrap the knowledge base in an MCP tool (search_knowledge)
  4. ANSWER  — SingleAgentLoop uses the tool; grounded answer, not hallucination

WHY CHUNKING MATTERS
--------------------
An LLM can only read ~100 k tokens at once. A document library can have
millions. You cannot send everything — you must retrieve only what is
relevant. But retrieval quality depends on chunk quality: split mid-sentence
and the embedding is semantically noisy; keep paragraphs together and the
embedding is clean.

This example uses RecursiveTextChunker — tries paragraph breaks first, then
sentence breaks, then word breaks. Best general-purpose choice for plain text.

THE DIFFERENCE FROM A PLAIN LLM CALL
--------------------------------------
Without RAG:  LLM answers from training data → may hallucinate, out of date
With RAG:     LLM answers from YOUR documents → grounded, citable, current

RUNNING
-------
    python examples/10_rag.py

REQUIREMENTS
------------
    pip install -r requirements.txt
    pip install -e .
    export ANTHROPIC_API_KEY=sk-ant-...
=============================================================================
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from fastmcp import FastMCP

from mcp_agent_framework import AnthropicClient, AgentConfig
from mcp_agent_framework.memory import SemanticMemory
from mcp_agent_framework.patterns import SingleAgentLoop


# =============================================================================
# CHUNKER — RecursiveTextChunker
#
# Splits text by trying coarser separators first (paragraphs → sentences →
# words). Each resulting piece that is still too large is split recursively
# with the next finer separator. Overlap preserves cross-boundary context.
# =============================================================================

class RecursiveTextChunker:
    """Split text into overlapping chunks, respecting semantic boundaries."""

    _SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+", r"(?<=,)\s+", " ", ""]

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 60) -> None:
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        return self._split(text.strip(), self._SEPARATORS)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text else []

        sep = separators[0]
        rest = separators[1:]

        pieces = re.split(sep, text) if sep else list(text)
        chunks: list[str] = []
        current = ""

        for piece in pieces:
            if len(current) + len(piece) + 1 > self.chunk_size and current:
                chunks.append(current.strip())
                # carry overlap forward
                words = current.split()
                overlap_text = " ".join(words[max(0, len(words) - 20):])
                current = overlap_text + " " + piece
            else:
                current = (current + " " + piece).strip() if current else piece

        if current.strip():
            # recursively split any piece still too large
            if len(current) > self.chunk_size and rest:
                chunks.extend(self._split(current.strip(), rest))
            else:
                chunks.append(current.strip())

        return chunks


# =============================================================================
# KNOWLEDGE BASE — thin wrapper around SemanticMemory
#
# SemanticMemory stores text with embeddings and retrieves by cosine similarity.
# The RAGStore adds chunking: one document-string in → multiple chunks stored.
# =============================================================================

class RAGStore:
    """Chunk documents and store them in SemanticMemory for retrieval."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 60) -> None:
        self._memory  = SemanticMemory()
        self._chunker = RecursiveTextChunker(chunk_size, chunk_overlap)

    async def add_document(self, text: str, source: str) -> int:
        """Chunk text and store all chunks. Returns number of chunks created."""
        chunks = self._chunker.split(text)
        for i, chunk in enumerate(chunks):
            await self._memory.add(chunk, metadata={"source": source, "chunk": i})
        return len(chunks)

    async def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """Return top-k chunks most relevant to the query."""
        results = await self._memory.search(query, top_k=top_k)
        return [
            {"text": r.content, "source": r.metadata.get("source", "unknown")}
            for r in results
        ]


# =============================================================================
# SAMPLE DOCUMENTS — a small knowledge base about vector databases
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
Pinecone is fully managed and requires zero infrastructure — ideal if you
want to skip DevOps entirely. pgvector works inside Postgres and is perfect
when you already have a Postgres database and want to avoid a second system.
""",
    "chunking_strategies.txt": """
Chunking is the process of splitting documents into smaller pieces before
embedding. Chunk quality directly determines retrieval quality.

Fixed-size chunking splits by character count with optional overlap. It is
the fastest approach and requires no decisions beyond chunk size and overlap.
The downside: it cuts sentences mid-way, producing semantically noisy
embeddings. Best for structured data or quick prototypes.

Recursive text chunking tries progressively finer separators: paragraph
breaks first, then sentence breaks, then word breaks. It produces coherent
chunks without requiring an embedding model for chunking itself. This is the
best general-purpose choice for plain text, markdown, or HTML.

Semantic chunking uses sentence embeddings to find topic boundaries. Sentences
with similar embeddings are grouped together; a large cosine-distance jump
between consecutive sentences marks a chunk boundary. Highest quality but
requires an embedding model and is slower.

Layout-aware chunking for PDFs recognises headers, tables, and figure
captions as typed chunks. The heading becomes metadata for all chunks within
its section, dramatically improving retrieval relevance for structured PDFs.

Chunk size trade-offs: small chunks (128–256 chars) are precise but lose
context; large chunks (1024+ chars) give the LLM more context but retrieve
less specifically. 400–512 characters with 10–15% overlap is the sweet spot
for most text corpora.
""",
    "rag_patterns.txt": """
RAG (Retrieval-Augmented Generation) grounds LLM responses in a document
corpus. The basic pattern: user asks a question → retrieve relevant chunks
→ inject chunks into the prompt → LLM answers using retrieved content.

Naive RAG limitation: one retrieval round may miss relevant chunks if the
question phrasing differs from the document phrasing.

HyDE (Hypothetical Document Embeddings) improves retrieval for vague
questions: ask the LLM to write a hypothetical answer first, embed THAT
answer, then use it for retrieval. The hypothesis is often semantically
closer to the actual document than the raw question.

Multi-query retrieval generates 3–5 paraphrases of the question, retrieves
for each, then deduplicates by chunk ID and re-ranks. Reduces sensitivity
to exact query phrasing.

Hybrid retrieval combines semantic (cosine similarity) and keyword (BM25)
scores using Reciprocal Rank Fusion (RRF): score = Σ 1/(rank + k). The
combination consistently outperforms either alone, especially on mixed
technical/conversational corpora.

Self-RAG adds a sufficiency check: after retrieval, the model assesses
whether it has enough information to answer. If not, it formulates follow-up
queries and retrieves again. This is the key step that turns basic RAG into
agentic RAG.
""",
}


# =============================================================================
# MCP SERVER — exposes the knowledge base as a tool
# =============================================================================

store = RAGStore(chunk_size=400, chunk_overlap=60)
app   = FastMCP("rag_server")


@app.tool
async def search_knowledge(query: str, top_k: int = 4) -> str:
    """
    Search the knowledge base for chunks relevant to the query.
    Returns the most relevant document excerpts with source labels.
    """
    results = await store.search(query, top_k=top_k)
    if not results:
        return "No relevant information found in the knowledge base."
    parts = [f"[{r['source']}]\n{r['text']}" for r in results]
    return "\n\n---\n\n".join(parts)


# =============================================================================
# MAIN
# =============================================================================

async def main() -> None:
    # ── Step 1: Ingest documents ─────────────────────────────────────────────
    print("Ingesting documents into knowledge base...")
    for source, text in DOCUMENTS.items():
        n = await store.add_document(text, source)
        print(f"  {source} → {n} chunks")

    # ── Step 2: Build the RAG agent ──────────────────────────────────────────
    config = AgentConfig(
        mcp_server_config=app,
        system_prompt=(
            "You are a helpful assistant with access to a knowledge base about RAG systems "
            "and vector databases. Always call search_knowledge before answering a factual "
            "question. Base your answer strictly on the retrieved content and cite sources."
        ),
        max_iterations=6,
    )
    agent = SingleAgentLoop(
        llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
        config=config,
    )

    # ── Step 3: Ask questions ────────────────────────────────────────────────
    questions = [
        "Which vector database should I choose for a startup with no DevOps team?",
        "What is the difference between fixed-size and recursive text chunking?",
        "How does HyDE improve RAG retrieval for vague questions?",
    ]

    for question in questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {question}")
        print("-" * 60)
        answer = await agent.run(question)
        print(f"A: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
