"""
procedural.py — Task-to-procedure lookup using Jaccard similarity.

ProceduralMemory stores *how to do things*: SOP documents, step-by-step
workflows, API recipes, tool usage guides.  Given a free-text task description
it retrieves the procedure whose title/description best matches the request.

Why Jaccard rather than cosine here?
  Procedures are often described with precise, domain-specific vocabulary
  ("deploy Kubernetes pod", "send Slack message").  Jaccard on word *sets*
  rewards exact token overlap without penalising documents that happen to be
  longer, making it a good fit for short, keyword-dense procedure titles.
  For longer procedure bodies a cosine-based store (SemanticMemory) would be
  more appropriate.
"""

from __future__ import annotations

from typing import Any

from .base import AbstractMemoryStore, MemoryEntry


def _jaccard(a: str, b: str) -> float:
    """Compute Jaccard similarity between the word sets of two strings.

    ``J(A, B) = |A ∩ B| / |A ∪ B|``

    Returns ``0.0`` when both strings are empty (undefined case treated as no
    overlap).

    Args:
        a: First string.
        b: Second string.

    Returns:
        Jaccard index in ``[0.0, 1.0]``.
    """
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


class ProceduralMemory(AbstractMemoryStore):
    """Stores "how to do things" — procedures, recipes, workflows.

    Given a task description, retrieve the procedure that best matches it.
    Uses Jaccard similarity on word sets.

    The ``content`` field of each entry should be a short, descriptive title or
    summary of the procedure.  Detailed steps, tool names, parameters, etc.
    belong in ``metadata``.

    Example::

        mem = ProceduralMemory()
        await mem.add(
            "How to write a blog post",
            metadata={"steps": ["research", "outline", "draft", "edit", "publish"]},
        )
        await mem.add(
            "How to deploy a Docker container",
            metadata={"steps": ["build image", "tag", "push", "run"]},
        )

        results = await mem.search("write an article for my website")
        procedure = results[0]
        steps = procedure.metadata.get("steps", [])
        # steps == ["research", "outline", "draft", "edit", "publish"]

        # Or use the convenience method:
        best = await mem.get_by_task("deploy my app with Docker")
        print(best.metadata["steps"])
    """

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}  # id -> entry

    # ------------------------------------------------------------------
    # AbstractMemoryStore implementation
    # ------------------------------------------------------------------

    async def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a new procedure.

        Args:
            content: Short title or description of the procedure.  This text
                is what Jaccard similarity is computed against during search.
            metadata: Structured procedure data — steps list, parameters, links,
                version info, etc.

        Returns:
            The ``id`` of the newly created entry.
        """
        entry = MemoryEntry(content=content, metadata=metadata or {})
        self._entries[entry.id] = entry
        return entry.id

    async def search(
        self, query: str, top_k: int = 5
    ) -> list[MemoryEntry]:
        """Return up to *top_k* procedures ranked by Jaccard similarity.

        Entries with zero Jaccard overlap are excluded from results, keeping
        the return list clean.  If the store is empty, or if no entry has any
        word in common with *query*, an empty list is returned.

        Args:
            query: Task description to match against stored procedure titles.
            top_k: Maximum number of results to return.

        Returns:
            List of :class:`~base.MemoryEntry` objects sorted by descending
            Jaccard score.
        """
        if not self._entries:
            return []

        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._entries.values():
            score = _jaccard(query, entry.content)
            if score > 0.0:
                scored.append((score, entry))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Return the entry with *entry_id*, or ``None``."""
        return self._entries.get(entry_id)

    async def delete(self, entry_id: str) -> bool:
        """Remove entry *entry_id*.

        Returns:
            ``True`` if the entry existed and was removed, ``False`` otherwise.
        """
        if entry_id in self._entries:
            del self._entries[entry_id]
            return True
        return False

    async def clear(self) -> None:
        """Remove all entries from this store."""
        self._entries.clear()

    async def list_all(self) -> list[MemoryEntry]:
        """Return all procedures in insertion order."""
        return list(self._entries.values())

    # ------------------------------------------------------------------
    # Procedural-specific helpers
    # ------------------------------------------------------------------

    async def get_by_task(self, task: str) -> MemoryEntry | None:
        """Return the single best-matching procedure, or ``None`` if store is empty.

        Convenience wrapper around :meth:`search` for callers that only want
        one result and want explicit ``None`` instead of an empty list.

        Args:
            task: Natural-language task description.

        Returns:
            The highest-scoring :class:`~base.MemoryEntry`, or ``None``.
        """
        results = await self.search(task, top_k=1)
        return results[0] if results else None
