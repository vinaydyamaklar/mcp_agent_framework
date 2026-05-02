"""
base.py — Core data structures and abstract interface for the memory subsystem.

All concrete memory store implementations must subclass AbstractMemoryStore and
implement every abstract method so the rest of the framework can interact with
different store types through a single, stable interface.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryEntry:
    """A single unit of stored memory.

    Attributes:
        id: Auto-generated UUID string that uniquely identifies this entry.
        content: The raw text content being stored.
        metadata: Arbitrary key-value pairs attached to the entry (e.g. tags,
            source, tool_name, step numbers).
        created_at: Unix timestamp (float) recorded at instantiation time.
        embedding: Optional pre-computed vector representation.  When present,
            semantic stores use it for similarity search; otherwise they compute
            one on demand.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    embedding: list[float] | None = None


class AbstractMemoryStore(ABC):
    """Async abstract base class for all memory store implementations.

    Concrete subclasses (SemanticMemory, EpisodicMemory, ProceduralMemory) each
    fulfil this contract so callers can depend on the interface rather than a
    specific implementation.

    All methods are *async* so implementations can use network I/O (vector
    databases, embedding APIs, etc.) without blocking the event loop.
    """

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a new entry and return its unique id.

        Args:
            content: The text to store.
            metadata: Optional extra data attached to the entry.

        Returns:
            The ``MemoryEntry.id`` of the newly created entry.
        """

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Remove the entry with the given id.

        Args:
            entry_id: UUID of the entry to remove.

        Returns:
            ``True`` if the entry existed and was deleted, ``False`` otherwise.
        """

    @abstractmethod
    async def clear(self) -> None:
        """Delete **all** entries from this store."""

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Retrieve the *top_k* entries most relevant to *query*.

        The relevance metric is implementation-defined (cosine similarity,
        recency + keyword overlap, Jaccard, etc.).

        Args:
            query: Natural-language or keyword query string.
            top_k: Maximum number of results to return.

        Returns:
            List of matching ``MemoryEntry`` objects, sorted best-first.
        """

    @abstractmethod
    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Fetch a single entry by its id.

        Args:
            entry_id: UUID of the target entry.

        Returns:
            The ``MemoryEntry``, or ``None`` if not found.
        """

    @abstractmethod
    async def list_all(self) -> list[MemoryEntry]:
        """Return every entry currently in the store, in insertion order."""
