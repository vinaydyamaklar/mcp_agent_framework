"""
semantic.py — In-memory semantic search using cosine similarity.

For production, pass ``embed_fn=your_async_embedding_fn``.  The default
bag-of-words fallback works for demos but has poor recall for paraphrased
queries because two synonym-heavy texts will share few tokens and therefore
score low despite meaning the same thing.  Swap in a real embedding model
(e.g. OpenAI ``text-embedding-3-small``, ``sentence-transformers``, or any
other async callable that returns a list[float]) to get genuine semantic recall.

When using the default embed function, the store automatically re-embeds all
entries at search time using a *shared vocabulary* built from every stored entry
plus the query.  This ensures all vectors occupy the same dimensional space and
cosine similarity is meaningful.  The trade-off is O(n) re-embedding on every
search — acceptable for demos; use a real embedding service for production.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from .base import AbstractMemoryStore, MemoryEntry


# ---------------------------------------------------------------------------
# Default embedding function (zero-dependency bag-of-words)
# ---------------------------------------------------------------------------

async def _default_embed(text: str) -> list[float]:
    """Simple bag-of-words TF embedding scoped to a single text.

    **Not intended to be called directly by external code.**  SemanticMemory
    uses :func:`_embed_with_shared_vocab` when the default embed_fn is active so
    that all vectors share the same dimensional space.

    Tokenises *text* by whitespace, builds a sorted vocabulary, then returns a
    term-frequency vector where each dimension corresponds to one vocabulary
    word.  Limitations: no IDF weighting, stemming, or stop-word removal; purely
    lexical.
    """
    words = text.lower().split()
    if not words:
        return [0.0]
    vocab = sorted(set(words))
    vec = [words.count(w) / len(words) for w in vocab]
    return vec


def _embed_with_shared_vocab(texts: list[str]) -> list[list[float]]:
    """Build TF vectors for *texts* over a shared vocabulary.

    All returned vectors have the same length (``len(shared_vocab)``), making
    cosine similarity between any pair well-defined.

    Args:
        texts: List of strings to vectorise together.

    Returns:
        Parallel list of float vectors, one per input string.
    """
    # Build shared vocabulary across all texts
    all_words: list[str] = []
    tokenised: list[list[str]] = []
    for t in texts:
        words = t.lower().split()
        tokenised.append(words)
        all_words.extend(words)

    vocab = sorted(set(all_words))
    if not vocab:
        return [[0.0]] * len(texts)

    vectors: list[list[float]] = []
    for words in tokenised:
        total = len(words) if words else 1
        vec = [words.count(w) / total for w in vocab]
        vectors.append(vec)
    return vectors


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def _pad_to_same_length(
    a: list[float], b: list[float]
) -> tuple[list[float], list[float]]:
    """Zero-pad the shorter of *a* and *b* so both have the same length.

    Required when two vectors were built from different vocabularies (e.g. when
    a custom embed_fn is used and vectors may differ in size).

    Args:
        a: First float vector.
        b: Second float vector.

    Returns:
        A ``(a_padded, b_padded)`` tuple where both lists have length
        ``max(len(a), len(b))``.
    """
    len_a, len_b = len(a), len(b)
    if len_a < len_b:
        a = a + [0.0] * (len_b - len_a)
    elif len_b < len_a:
        b = b + [0.0] * (len_a - len_b)
    return a, b


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity between two float vectors.

    Vectors need not have the same length — the shorter one is zero-padded
    automatically via :func:`_pad_to_same_length`.

    Uses ``dot / (|a| * |b| + 1e-9)`` to avoid division by zero when either
    vector is all-zeros.

    Args:
        a: First float vector.
        b: Second float vector.

    Returns:
        Cosine similarity in the range ``[-1.0, 1.0]``.
    """
    a, b = _pad_to_same_length(a, b)
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    return dot / (mag_a * mag_b + 1e-9)


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------

# Sentinel object used to detect whether the caller passed a custom embed_fn.
_USE_DEFAULT = object()


class SemanticMemory(AbstractMemoryStore):
    """In-memory store that retrieves entries by cosine similarity.

    For production, pass ``embed_fn=your_async_embedding_fn``.  The default
    bag-of-words fallback works for demos but has poor recall for paraphrased
    queries.

    **Default embed_fn behaviour (no external deps):**

    When no ``embed_fn`` is supplied, the store uses a shared-vocabulary TF
    approach: at search time it builds a single vocabulary from *all* stored
    entry texts plus the query, then represents every text as a term-frequency
    vector over that shared vocabulary.  This guarantees all vectors live in the
    same dimensional space so cosine similarity is meaningful.

    **Custom embed_fn:**

    Pass any async callable ``(text: str) -> list[float]``.  The store caches
    the embedding computed at ``add()`` time and reuses it at search time.  If
    your model returns fixed-size vectors (e.g. 1536-dim OpenAI embeddings) the
    cosine calculation is exact.

    Args:
        embed_fn: An async callable ``(text: str) -> list[float]``.  Defaults
            to the built-in shared-vocabulary bag-of-words implementation.
        similarity_threshold: Minimum cosine similarity required for an entry
            to appear in search results.  Defaults to ``0.0`` (return all
            entries, just ranked).

    Example::

        # Demo usage with default embedder
        mem = SemanticMemory()
        await mem.add("The cat sat on the mat")
        await mem.add("Python is a programming language")
        results = await mem.search("cat mat")
        assert "cat" in results[0].content   # cat/mat entry ranks first

        # Production usage with OpenAI embeddings
        import openai
        client = openai.AsyncOpenAI()

        async def openai_embed(text: str) -> list[float]:
            resp = await client.embeddings.create(
                model="text-embedding-3-small", input=text
            )
            return resp.data[0].embedding

        mem = SemanticMemory(embed_fn=openai_embed)
    """

    def __init__(
        self,
        embed_fn: Callable[[str], Awaitable[list[float]]] | None = None,
        similarity_threshold: float = 0.0,
    ) -> None:
        # Track whether caller provided a custom embed function.
        # When None, we use the shared-vocabulary path (no per-entry caching
        # of embeddings because the vocabulary grows with each add()).
        self._custom_embed_fn: Callable[[str], Awaitable[list[float]]] | None = embed_fn
        self._similarity_threshold = similarity_threshold
        self._entries: dict[str, MemoryEntry] = {}  # id -> entry

    # ------------------------------------------------------------------
    # AbstractMemoryStore implementation
    # ------------------------------------------------------------------

    async def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a new entry.

        When a custom ``embed_fn`` was provided, the embedding is computed now
        and cached on the entry.  With the default bag-of-words embedder,
        embeddings are deferred to search time so they always use the full
        shared vocabulary.

        Args:
            content: Text to store.
            metadata: Optional key-value data attached to the entry.

        Returns:
            The ``id`` of the newly created entry.
        """
        entry = MemoryEntry(content=content, metadata=metadata or {})

        if self._custom_embed_fn is not None:
            # Cache embedding now — vocabulary is fixed for real embedding models.
            entry.embedding = await self._custom_embed_fn(content)

        self._entries[entry.id] = entry
        return entry.id

    async def search(
        self, query: str, top_k: int = 5
    ) -> list[MemoryEntry]:
        """Return up to *top_k* entries ranked by cosine similarity to *query*.

        Entries whose similarity falls below ``similarity_threshold`` are
        excluded.  Ties in similarity are broken arbitrarily.

        When the default bag-of-words embedder is active, all entry texts and
        the query are vectorised together over a shared vocabulary so that
        dimensions are aligned and similarity is meaningful.

        Args:
            query: Query text to embed and compare against stored entries.
            top_k: Maximum number of results to return.

        Returns:
            List of :class:`~base.MemoryEntry` objects, sorted
            highest-similarity first.
        """
        if not self._entries:
            return []

        entries = list(self._entries.values())

        if self._custom_embed_fn is not None:
            # Use cached (or freshly computed) embeddings.
            query_embedding = await self._custom_embed_fn(query)
            scored: list[tuple[float, MemoryEntry]] = []
            for entry in entries:
                emb = entry.embedding
                if emb is None:
                    emb = await self._custom_embed_fn(entry.content)
                    entry.embedding = emb
                score = _cosine_similarity(query_embedding, emb)
                if score >= self._similarity_threshold:
                    scored.append((score, entry))
        else:
            # Default path: shared-vocabulary TF vectors computed together.
            all_texts = [query] + [e.content for e in entries]
            all_vectors = _embed_with_shared_vocab(all_texts)
            query_vec = all_vectors[0]
            scored = []
            for i, entry in enumerate(entries):
                score = _cosine_similarity(query_vec, all_vectors[i + 1])
                if score >= self._similarity_threshold:
                    scored.append((score, entry))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Return the entry with *entry_id*, or ``None``."""
        return self._entries.get(entry_id)

    async def delete(self, entry_id: str) -> bool:
        """Remove entry *entry_id*.  Returns ``True`` if it existed."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            return True
        return False

    async def clear(self) -> None:
        """Remove all entries from this store."""
        self._entries.clear()

    async def list_all(self) -> list[MemoryEntry]:
        """Return all entries in insertion order (dict preserves order, Py 3.7+)."""
        return list(self._entries.values())
