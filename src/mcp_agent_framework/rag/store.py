"""
RAG stores — knowledge base implementations for Retrieval-Augmented Generation.

Two classes:

``RAGStore``
    Basic semantic-only store. Chunk → embed → cosine search.
    Good for simple Q&A over a document corpus.

``AgenticRAGStore``
    Extends RAGStore with BM25 keyword search and hybrid RRF retrieval.
    Use when agents need to self-direct their retrieval strategy:
    semantic for concept questions, keyword for exact terms, hybrid for both.

Both accept an optional ``embed_fn`` that is forwarded to ``SemanticMemory``.
Pass a real async embedding function for production (OpenAI, Cohere, etc.).
The default bag-of-words fallback works for demos only.

Usage::

    # Basic RAG
    store = RAGStore()
    await store.add_document(text, source="docs.md")
    results = await store.search("what is HNSW?", top_k=3)

    # Agentic RAG (semantic + keyword + hybrid)
    store = AgenticRAGStore()
    await store.add_document(text, source="docs.md")
    sem     = await store.search_semantic("approximate nearest neighbour", top_k=3)
    kw      = store.search_keyword("HNSW", top_k=3)
    hybrid  = await store.search_hybrid("HNSW algorithm performance", top_k=5)

    # Production: plug in a real embedding model
    import openai
    _client = openai.AsyncOpenAI()

    async def openai_embed(text: str) -> list[float]:
        resp = await _client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return resp.data[0].embedding

    store = AgenticRAGStore(embed_fn=openai_embed)
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from mcp_agent_framework.memory import SemanticMemory
from mcp_agent_framework.rag.chunker import RecursiveTextChunker


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """One retrieved chunk with its source and relevance score."""
    text:   str
    source: str
    score:  float = 0.0


# ---------------------------------------------------------------------------
# Internal chunk record (used by BM25)
# ---------------------------------------------------------------------------

@dataclass
class _Chunk:
    text:   str
    source: str
    index:  int


# ---------------------------------------------------------------------------
# RAGStore — semantic search only
# ---------------------------------------------------------------------------

class RAGStore:
    """
    Basic RAG knowledge base: chunk documents, store embeddings, retrieve by
    cosine similarity.

    Args:
        chunk_size:    Target characters per chunk (default 400).
        chunk_overlap: Overlap characters between adjacent chunks (default 60).
        embed_fn:      Async callable ``(text: str) -> list[float]``.
                       Pass a real embedding model for production.
                       Defaults to the framework's bag-of-words fallback.

    Example::

        store = RAGStore(embed_fn=openai_embed)
        n = await store.add_document(long_text, source="manual.pdf")
        results = await store.search("how do I configure retries?", top_k=4)
        for r in results:
            print(r.source, r.score, r.text[:80])
    """

    def __init__(
        self,
        chunk_size:    int = 400,
        chunk_overlap: int = 60,
        embed_fn: Callable[[str], Awaitable[list[float]]] | None = None,
    ) -> None:
        self._semantic = SemanticMemory(embed_fn=embed_fn)
        self._chunker  = RecursiveTextChunker(chunk_size, chunk_overlap)
        self._sources:  set[str] = set()
        self._n_chunks: int = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def add_document(self, text: str, source: str) -> int:
        """
        Chunk *text* and store all chunks under *source* label.

        Args:
            text:   Raw document text.
            source: Human-readable label (filename, URL, doc ID).

        Returns:
            Number of chunks created.
        """
        chunks = self._chunker.split(text)
        for i, chunk in enumerate(chunks):
            await self._semantic.add(chunk, metadata={"source": source, "chunk_index": i})
        self._sources.add(source)
        self._n_chunks += len(chunks)
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def search(self, query: str, top_k: int = 4) -> list[RetrievalResult]:
        """
        Retrieve the *top_k* chunks most semantically similar to *query*.

        Returns:
            List of :class:`RetrievalResult` sorted highest-score first.
        """
        entries = await self._semantic.search(query, top_k=top_k)
        return [
            RetrievalResult(
                text=e.content,
                source=e.metadata.get("source", "unknown"),
            )
            for e in entries
        ]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return a summary of the knowledge base contents."""
        return {
            "total_chunks": self._n_chunks,
            "sources":      sorted(self._sources),
        }


# ---------------------------------------------------------------------------
# RRF helper
# ---------------------------------------------------------------------------

def _rrf_merge(
    lists: list[list[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    """
    Reciprocal Rank Fusion: merge *lists* of ranked results into one list.

    Score for each result = Σ 1 / (k + rank_in_list).
    Results appearing in multiple lists accumulate contributions.

    Args:
        lists: Two or more ranked result lists (each sorted best-first).
        k:     Smoothing constant (default 60 — standard RRF value).

    Returns:
        Merged list sorted by combined RRF score, highest first.
        ``RetrievalResult.score`` is set to the combined RRF score.
    """
    scores: dict[str, float] = {}
    best:   dict[str, RetrievalResult] = {}

    for ranked_list in lists:
        for rank, result in enumerate(ranked_list):
            key = result.text  # deduplicate by exact text
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in best:
                best[key] = result

    merged = []
    for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        r = best[key]
        merged.append(RetrievalResult(text=r.text, source=r.source, score=score))
    return merged


# ---------------------------------------------------------------------------
# AgenticRAGStore — semantic + BM25 keyword + hybrid RRF
# ---------------------------------------------------------------------------

class AgenticRAGStore(RAGStore):
    """
    Extended knowledge base with three retrieval modes:

    ``search_semantic(query)``
        Cosine similarity on embeddings. Best for concept questions and
        paraphrased queries ("how does X work", "what is Y").

    ``search_keyword(query)``
        BM25 ranking. Best for exact technical terms, product names, error
        codes, and proper nouns that semantic search may mis-rank.

    ``search_hybrid(query)``
        Reciprocal Rank Fusion of semantic + keyword results. Consistently
        outperforms either alone on mixed corpora. Best default for production.

    All three return :class:`RetrievalResult` lists sorted best-first.

    Args:
        chunk_size:    Target characters per chunk (default 400).
        chunk_overlap: Overlap characters between adjacent chunks (default 60).
        embed_fn:      Async callable ``(text: str) -> list[float]``.
                       Defaults to bag-of-words fallback.

    Example::

        store = AgenticRAGStore(embed_fn=openai_embed)
        await store.add_document(doc_text, source="api_reference.md")

        # Let the agent choose the right mode per query
        results = await store.search_hybrid("HNSW latency benchmark", top_k=5)
    """

    def __init__(
        self,
        chunk_size:    int = 400,
        chunk_overlap: int = 60,
        embed_fn: Callable[[str], Awaitable[list[float]]] | None = None,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, embed_fn=embed_fn)
        self._bm25_chunks: list[_Chunk] = []   # parallel corpus for BM25

    # ------------------------------------------------------------------
    # Override add_document to also populate BM25 index
    # ------------------------------------------------------------------

    async def add_document(self, text: str, source: str) -> int:
        chunks = self._chunker.split(text)
        for i, chunk in enumerate(chunks):
            await self._semantic.add(chunk, metadata={"source": source, "chunk_index": i})
            self._bm25_chunks.append(_Chunk(chunk, source, i))
        self._sources.add(source)
        self._n_chunks += len(chunks)
        return len(chunks)

    # ------------------------------------------------------------------
    # Semantic (alias of parent's search for clarity)
    # ------------------------------------------------------------------

    async def search_semantic(self, query: str, top_k: int = 4) -> list[RetrievalResult]:
        """Semantic cosine search. Alias of :meth:`search`."""
        return await self.search(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Keyword search (BM25)
    # ------------------------------------------------------------------

    def search_keyword(self, query: str, top_k: int = 4) -> list[RetrievalResult]:
        """
        BM25 keyword ranking.

        Scores every stored chunk using the BM25 formula:
            score = Σ_term  IDF(term) * TF(term, doc) * (k+1)
                            / (TF(term, doc) + k * (1 − b + b * dl/avgdl))
        with k=1.5, b=0.75 (standard BM25 parameters).

        Args:
            query:  Space-separated keyword query.
            top_k:  Maximum results to return.

        Returns:
            List of :class:`RetrievalResult` sorted by BM25 score descending.
            ``score`` is the raw BM25 score (unbounded positive float).
        """
        if not self._bm25_chunks:
            return []

        k, b = 1.5, 0.75
        query_terms = set(query.lower().split())

        doc_tfs: list[Counter] = [Counter(c.text.lower().split()) for c in self._bm25_chunks]
        avgdl = sum(sum(tf.values()) for tf in doc_tfs) / max(len(doc_tfs), 1)

        # Document frequency per term
        df: Counter = Counter()
        for tf in doc_tfs:
            for term in tf:
                df[term] += 1
        N = len(self._bm25_chunks)

        scored: list[tuple[float, _Chunk]] = []
        for chunk, tf in zip(self._bm25_chunks, doc_tfs):
            dl    = sum(tf.values())
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                idf      = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
                tf_score = (tf[term] * (k + 1)) / (
                    tf[term] + k * (1 - b + b * dl / max(avgdl, 1))
                )
                score += idf * tf_score
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievalResult(text=c.text, source=c.source, score=s)
            for s, c in scored[:top_k]
        ]

    # ------------------------------------------------------------------
    # Hybrid search (RRF)
    # ------------------------------------------------------------------

    async def search_hybrid(
        self,
        query:     str,
        top_k:     int = 5,
        rrf_k:     int = 60,
    ) -> list[RetrievalResult]:
        """
        Hybrid search: merge semantic and keyword results with Reciprocal Rank
        Fusion (RRF).

        Fetches ``top_k * 2`` candidates from each retriever to give RRF enough
        signal, then returns the top ``top_k`` merged results.

        Args:
            query:  The search query.
            top_k:  Number of results to return after merging.
            rrf_k:  RRF smoothing constant (default 60).

        Returns:
            List of :class:`RetrievalResult` sorted by combined RRF score.
        """
        fetch = top_k * 2  # fetch more to give RRF enough candidates
        sem_results = await self.search_semantic(query, top_k=fetch)
        kw_results  = self.search_keyword(query,        top_k=fetch)
        merged      = _rrf_merge([sem_results, kw_results], k=rrf_k)
        return merged[:top_k]
