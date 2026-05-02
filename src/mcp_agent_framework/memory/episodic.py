"""
episodic.py — Timestamped event log with recency-biased keyword search.

EpisodicMemory is designed to store things that *happened*: conversation turns,
agent observations, tool call results, error traces, etc.  Entries are kept in
insertion (chronological) order and searched via a combined score that rewards
both keyword relevance and recency.

Why recency matters here:
  In an agentic loop the most recent observations are almost always more
  actionable than older ones.  The recency bonus prevents a single highly
  keyword-matched ancient entry from drowning out fresh, partially-matching
  context.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from .base import AbstractMemoryStore, MemoryEntry


class EpisodicMemory(AbstractMemoryStore):
    """Timestamped fact store for "things that happened".

    Search combines keyword overlap and recency bias:

    - ``keyword_overlap`` — count of words shared between the lower-cased query
      tokens and the lower-cased entry content tokens.
    - ``recency_bonus`` — a value in ``[0, 0.5]`` proportional to the entry's
      position in the insertion-ordered list (later = newer = higher bonus).

    Final score::

        score = keyword_overlap + (index / total) * 0.5

    Useful for: conversation history, agent observations, action logs.

    Args:
        max_entries: If set, the store will evict the *oldest* entry whenever a
            new one would exceed this limit.  ``None`` means unlimited.

    Example::

        mem = EpisodicMemory(max_entries=1000)
        await mem.add("User asked about the weather in Paris")
        await mem.add("Tool returned: 15°C, cloudy")
        recent = await mem.get_recent(5)
        relevant = await mem.search("weather Paris", top_k=3)
    """

    def __init__(self, max_entries: int | None = None) -> None:
        # deque gives O(1) popleft() for oldest-entry eviction when the store
        # is full; list.pop(0) is O(N) because it shifts every remaining element.
        self._entries: deque[MemoryEntry] = deque()
        self._by_id: dict[str, MemoryEntry] = {}
        self._max_entries = max_entries

    # ------------------------------------------------------------------
    # AbstractMemoryStore implementation
    # ------------------------------------------------------------------

    async def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Append a new event entry.

        If ``max_entries`` was set and the store is full, the oldest entry is
        silently evicted before the new one is appended.

        Args:
            content: Text describing the event.
            metadata: Optional context (e.g. ``{"role": "assistant"}``).

        Returns:
            The ``id`` of the newly created entry.
        """
        entry = MemoryEntry(content=content, metadata=metadata or {})

        if self._max_entries is not None and len(self._entries) >= self._max_entries:
            # Evict oldest (front of deque) — O(1) with deque, was O(N) with list.pop(0)
            oldest = self._entries.popleft()
            self._by_id.pop(oldest.id, None)

        self._entries.append(entry)
        self._by_id[entry.id] = entry
        return entry.id

    async def search(
        self, query: str, top_k: int = 5
    ) -> list[MemoryEntry]:
        """Return up to *top_k* entries ranked by keyword overlap + recency.

        Scoring formula::

            keyword_overlap = len(query_words & entry_words)
            recency_bonus   = (entry_index / total_entries) * 0.5
            score           = keyword_overlap + recency_bonus

        Entries with a score of exactly ``0`` (no keyword overlap and lowest
        possible recency) are still included — all stored events are candidates.
        Callers who want strict keyword filtering should post-filter the results.

        Args:
            query: Space-tokenised keyword query.
            top_k: Maximum number of results.

        Returns:
            List of :class:`~base.MemoryEntry` objects, sorted best-first.
        """
        if not self._entries:
            return []

        query_words = set(query.lower().split())
        total = len(self._entries)
        scored: list[tuple[float, MemoryEntry]] = []

        for index, entry in enumerate(self._entries):
            entry_words = set(entry.content.lower().split())
            keyword_overlap = len(query_words & entry_words)
            recency_bonus = (index / total) * 0.5
            score = keyword_overlap + recency_bonus
            scored.append((score, entry))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Return the entry with *entry_id*, or ``None``."""
        return self._by_id.get(entry_id)

    async def delete(self, entry_id: str) -> bool:
        """Remove entry *entry_id*.

        Returns:
            ``True`` if the entry existed and was removed, ``False`` otherwise.
        """
        if entry_id not in self._by_id:
            return False
        entry = self._by_id.pop(entry_id)
        self._entries.remove(entry)
        return True

    async def clear(self) -> None:
        """Remove all entries from this store."""
        self._entries.clear()
        self._by_id.clear()

    async def list_all(self) -> list[MemoryEntry]:
        """Return all entries in chronological (insertion) order."""
        return list(self._entries)

    # ------------------------------------------------------------------
    # Episodic-specific helpers
    # ------------------------------------------------------------------

    async def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Return the *n* most recently added entries, newest first.

        Useful for building a sliding-window context window: grab the last N
        conversation turns and inject them into the next prompt.

        Args:
            n: How many entries to return.

        Returns:
            A list of at most *n* entries, most recent entry at index 0.
        """
        # deque does not support slicing; convert to list first.
        return list(reversed(list(self._entries)[-n:]))
