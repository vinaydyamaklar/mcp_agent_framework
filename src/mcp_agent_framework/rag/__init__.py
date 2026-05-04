"""
mcp_agent_framework.rag — RAG (Retrieval-Augmented Generation) utilities.

``RecursiveTextChunker``
    Split documents into overlapping chunks at natural language boundaries.
    Tries paragraph → sentence → word splits; never breaks mid-word.

``RAGStore``
    Basic semantic knowledge base. Chunk → embed → cosine search.
    Accepts an optional ``embed_fn`` for real embedding models.

``AgenticRAGStore``
    Extended store with three retrieval modes:
    - ``search_semantic()``  — cosine similarity
    - ``search_keyword()``   — BM25 exact-term ranking
    - ``search_hybrid()``    — Reciprocal Rank Fusion of both

``RetrievalResult``
    Returned by all search methods: text, source label, score.

Quick start::

    from mcp_agent_framework.rag import AgenticRAGStore, RecursiveTextChunker

    store = AgenticRAGStore()                    # demo: bag-of-words embeddings
    await store.add_document(text, source="docs.md")
    results = await store.search_hybrid("HNSW latency", top_k=5)

Production (real embeddings)::

    import openai
    _oc = openai.AsyncOpenAI()

    async def embed(text: str) -> list[float]:
        r = await _oc.embeddings.create(model="text-embedding-3-small", input=text)
        return r.data[0].embedding

    store = AgenticRAGStore(embed_fn=embed)
"""

from mcp_agent_framework.rag.chunker import RecursiveTextChunker
from mcp_agent_framework.rag.store import (
    AgenticRAGStore,
    RAGStore,
    RetrievalResult,
)

__all__ = [
    "RecursiveTextChunker",
    "RAGStore",
    "AgenticRAGStore",
    "RetrievalResult",
]
