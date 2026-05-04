"""
RecursiveTextChunker — split documents into overlapping chunks.

Tries natural language boundaries largest-to-smallest:
  \\n\\n  →  \\n  →  sentence ends  →  word boundaries

Never splits mid-word. Overlap preserves context across chunk boundaries.

Usage::

    chunker = RecursiveTextChunker(chunk_size=400, chunk_overlap=60)
    chunks  = chunker.split(document_text)

Chunk size guide:
  128–256 chars  — precise retrieval, loses surrounding context
  400–512 chars  — best default for most text corpora
  768–1024 chars — more context per chunk, diluted embeddings
"""
from __future__ import annotations

import re


class RecursiveTextChunker:
    """
    Split text into overlapping chunks at natural language boundaries.

    The splitter tries separators from coarsest to finest:
      1. ``\\n\\n`` — paragraph breaks (preferred)
      2. ``\\n``    — line breaks
      3. ``(?<=[.!?])\\s+`` — sentence ends
      4. ``(?<=,)\\s+``     — clause ends
      5. `` ``              — word boundaries
      6. ``""``             — character level (last resort)

    Any piece that still exceeds ``chunk_size`` after splitting on the current
    separator is recursively split with the next finer separator.

    Args:
        chunk_size:    Target maximum characters per chunk.
        chunk_overlap: Characters shared between adjacent chunks, carried
                       forward as a word-boundary-aligned overlap window.
                       Prevents losing concepts split across a boundary.

    Example::

        chunker = RecursiveTextChunker(chunk_size=500, chunk_overlap=50)
        chunks  = chunker.split(long_document)
        # each chunk is <= 500 chars (approximately); adjacent chunks share ~50 chars
    """

    _SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+", r"(?<=,)\s+", " ", ""]

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 60) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        """Split *text* into overlapping chunks. Returns an empty list for blank input."""
        text = text.strip()
        if not text:
            return []
        return self._split(text, self._SEPARATORS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        sep  = separators[0]
        rest = separators[1:]

        pieces = re.split(sep, text) if sep else list(text)
        chunks: list[str] = []
        current = ""

        for piece in pieces:
            candidate = (current + " " + piece).strip() if current else piece
            if len(candidate) > self.chunk_size and current:
                # current is full — flush it
                chunks.append(current.strip())
                # carry overlap forward aligned to word boundaries
                current = self._overlap_tail(current) + " " + piece
            else:
                current = candidate

        if current.strip():
            if len(current) > self.chunk_size and rest:
                chunks.extend(self._split(current.strip(), rest))
            else:
                chunks.append(current.strip())

        return [c for c in chunks if c]

    def _overlap_tail(self, text: str) -> str:
        """Return the last N characters of *text* aligned to a word boundary."""
        if len(text) <= self.chunk_overlap:
            return text
        tail = text[-self.chunk_overlap:]
        # walk forward to the first space so we don't start mid-word
        space = tail.find(" ")
        return tail[space + 1:] if space != -1 else tail
