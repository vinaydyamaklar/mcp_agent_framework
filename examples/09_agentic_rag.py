"""
=============================================================================
Example 09 — Robust Agentic RAG System
=============================================================================

WHAT THIS EXAMPLE TEACHES
--------------------------
This is the most complete example in the suite. It builds a production-grade
Agentic RAG (Retrieval-Augmented Generation) system from scratch using only
this framework — no LangChain, no LlamaIndex.

You will learn:

    CHUNKING TECHNIQUES (Section 1–4)
    ──────────────────────────────────
    1. FixedSizeChunker       — simplest, fastest, but cuts sentences mid-way
    2. RecursiveTextChunker   — semantic-boundary-aware (same idea as LangChain's
                                RecursiveCharacterTextSplitter, but pure Python)
    3. SemanticChunker        — groups sentences with similar embeddings together
    4. LayoutAwarePDFChunker  — structure-preserving for PDFs: headers, tables,
                                figure captions as typed chunks

    RETRIEVAL TECHNIQUES (Section 5–9)
    ────────────────────────────────────
    5. SemanticRetriever      — cosine-similarity vector search
    6. KeywordRetriever       — TF-IDF / BM25-style lexical search (zero deps)
    7. HybridRetriever        — combine semantic + keyword scores (best of both)
    8. HyDERetriever          — Hypothetical Document Embeddings: generate a
                                hypothetical answer, embed THAT for retrieval
    9. MultiQueryRetriever    — generate N paraphrases, search each, merge results

    THE AGENTIC LOOP (Section 10)
    ──────────────────────────────
    10. Self-evaluation loop  — agent assesses: "do I have enough information?"
        ┌─────────────────────────────────────────────────────────────────┐
        │  OBSERVE   →   the question and available knowledge             │
        │  PLAN      →   decide which searches to run and in what order   │
        │  SEARCH    →   call search tools with targeted queries          │
        │  EVALUATE  →   do these results answer the question completely? │
        │  RE-SEARCH →   if gaps remain, search again with new angles     │
        │  SYNTHESISE→   build a grounded answer with citations           │
        └─────────────────────────────────────────────────────────────────┘

    DEMOS (Section 11–16)
    ──────────────────────
    Demo A: Show all 4 chunking techniques on the same text
    Demo B: Compare all retrieval techniques on the same query
    Demo C: Simple one-shot RAG (SingleAgentLoop, 1 search)
    Demo D: Agentic RAG (PlannerExecutorPattern, multi-search, self-eval)
    Demo E: EvaluatorOptimizer RAG (generates then critiques its own answer)
    Demo F: Use-case matrix — which chunker + which retriever for which scenario

THE DIFFERENCE BETWEEN NAIVE RAG AND AGENTIC RAG
-------------------------------------------------
NAIVE RAG (what most tutorials show):
    user asks question → one search → inject results → LLM answers
    Problem: what if the first search misses something? The LLM guesses.

AGENTIC RAG (what this example builds):
    user asks question → agent PLANS how to research it → searches multiple
    times with different queries → EVALUATES whether gaps remain → searches
    again if needed → synthesises a grounded answer with citations.

    The agent has AGENCY over retrieval. It is not a wrapper around search.

CHUNKING — WHY IT MATTERS
--------------------------
An LLM can only read ~100k tokens at once. A document library might have
10 million tokens. You cannot send everything. You must chunk, embed, and
retrieve only what is relevant.

Bad chunking ruins retrieval. If you split mid-sentence, or cut a table in
half, or separate a header from its paragraph, the chunks are semantically
incomplete. The embeddings are noisy. Retrieval quality drops.

CHUNKING COMPARISON TABLE
--------------------------
Technique          | Speed | Coherence | PDF Structure | Use When
-------------------+-------+-----------+---------------+------------------------
FixedSize          | ★★★★★ | ★★        | ✗             | Quick prototypes
RecursiveText      | ★★★★  | ★★★★      | partial       | Most plain text
Semantic           | ★★    | ★★★★★     | ✗             | High-quality prose
LayoutAwarePDF     | ★★★   | ★★★★★     | ✓ full        | PDFs with tables/figures

RETRIEVAL COMPARISON TABLE
---------------------------
Technique          | Semantic | Keyword | Cross-doc | Robustness | Use When
-------------------+----------+---------+-----------+------------+------------------
Semantic           | ★★★★★   | ★★      | ★★★★★    | exact-spell | Default choice
Keyword (BM25)     | ★★       | ★★★★★   | ★★        | ★★★★★      | Known terms
Hybrid             | ★★★★★   | ★★★★    | ★★★★★    | ★★★★★      | Production default
HyDE               | ★★★★★   | ★★★     | ★★★★★    | ★★★        | Vague questions
MultiQuery         | ★★★★★   | ★★★     | ★★★★★    | ★★★★       | Rephrasing needed

RUNNING
-------
    # For plain text examples (no PDF):
    python examples/09_agentic_rag.py

    # Run specific demos only:
    python examples/09_agentic_rag.py --demos A,B,C

    # For PDF examples (requires pdfplumber):
    pip install pdfplumber
    python examples/09_agentic_rag.py --pdf path/to/your.pdf

REQUIREMENTS
------------
    pip install -r requirements.txt
    pip install -e .
    pip install pdfplumber          # only needed for PDF ingestion
    export ANTHROPIC_API_KEY=sk-ant-...
=============================================================================
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from fastmcp import FastMCP

from mcp_agent_framework import AgentConfig, AnthropicClient, Message
from mcp_agent_framework.memory import SemanticMemory
from mcp_agent_framework.patterns import (
    PlannerExecutorPattern,
    SingleAgentLoop,
    EvaluatorOptimizerPattern,
)
from mcp_agent_framework.patterns.evaluation import LLMEvaluator, RubricEvaluator, RubricCriterion

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1 — DocumentChunk
#
# Every piece of text that enters the RAG system becomes a DocumentChunk.
# The metadata travels with the chunk all the way to retrieval, so the LLM
# can cite: "According to page 4, section 'Architecture Overview'..."
# =============================================================================

@dataclass
class DocumentChunk:
    """
    A single retrievable unit of text with its provenance metadata.

    chunk_id    — stable hash of (source + chunk_index + text prefix). Reproducible.
    text        — the actual text content of this chunk
    source      — filename, URL, or label identifying the document
    chunk_index — position of this chunk within its source (0-based)
    chunk_type  — "paragraph", "header", "table", "figure_caption", "code"
    metadata    — everything else: page number, section title, bounding box, etc.
    """
    text:        str
    source:      str
    chunk_index: int
    chunk_type:  str = "paragraph"
    metadata:    dict[str, Any] = field(default_factory=dict)
    chunk_id:    str = field(init=False)

    def __post_init__(self) -> None:
        # Deterministic ID so re-ingesting the same document is idempotent
        raw = f"{self.source}:{self.chunk_index}:{self.text[:64]}"
        self.chunk_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_storage_text(self) -> str:
        """
        The text stored in SemanticMemory. Prepend header + section so the
        embedding captures document structure, not just raw paragraph text.
        """
        parts = []
        if self.metadata.get("section_title"):
            parts.append(f"[Section: {self.metadata['section_title']}]")
        if self.metadata.get("page"):
            parts.append(f"[Page {self.metadata['page']}]")
        if self.chunk_type == "table":
            parts.append("[TABLE]")
        elif self.chunk_type == "figure_caption":
            parts.append("[FIGURE CAPTION]")
        parts.append(self.text)
        return " ".join(parts)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"DocumentChunk(id={self.chunk_id}, type={self.chunk_type}, "
            f"source={self.source!r}, idx={self.chunk_index}, text={preview!r}...)"
        )


# =============================================================================
# SECTION 2 — FixedSizeChunker
#
# The simplest possible chunker: split by character count with optional overlap.
#
# WHY USE IT: Maximum speed, zero decisions to make, easiest to reason about.
# WHY AVOID IT: Cuts sentences mid-way; the embedding of half-a-sentence is
# semantically noisy. Use RecursiveTextChunker unless speed is critical.
#
# WHEN TO USE: Quick prototyping, structured data (CSV, JSON), or when text
# is already pre-segmented (e.g. one paragraph per row in a database).
# =============================================================================

class FixedSizeChunker:
    """
    Split text into fixed-length windows with optional character overlap.

    This is the simplest chunking approach. It does NOT respect sentence or
    paragraph boundaries — a sentence may be cut in the middle.

    Args:
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Characters to repeat from end of one chunk at the start
                       of the next. Typical: 10-15% of chunk_size.

    Example output for chunk_size=100, overlap=20 on a 250-char string:
        Chunk 0: chars 0–100
        Chunk 1: chars 80–180     (20-char overlap with chunk 0)
        Chunk 2: chars 160–250

    Best used for:
        • Rapid prototyping (no setup, instant results)
        • Pre-segmented content (one paragraph / one row already)
        • Structured data ingested as text (CSV, JSON, log lines)
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        chunks: list[str] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += step
        return chunks

    def split_documents(
        self, text: str, source: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                text=chunk, source=source, chunk_index=i,
                chunk_type="paragraph",
                metadata={**(extra_metadata or {}), "char_start": i * (self.chunk_size - self.chunk_overlap)},
            )
            for i, chunk in enumerate(self.split(text))
        ]


# =============================================================================
# SECTION 3 — RecursiveTextChunker
#
# Splits plain text into semantically coherent chunks by trying progressively
# finer separators. Same algorithm as LangChain RecursiveCharacterTextSplitter
# but implemented in ~80 lines of pure Python with zero dependencies.
#
# WHEN TO USE: Default choice for plain text, markdown, HTML stripped to text,
# any unstructured prose document. Gets the best semantic coherence without
# needing an embedding model (unlike SemanticChunker).
# =============================================================================

class RecursiveTextChunker:
    """
    Split text into overlapping chunks, preferring semantic boundaries.

    The algorithm tries separators in priority order (coarsest → finest):
        \\n\\n   — paragraph breaks  (preferred: keeps paragraphs together)
        \\n     — line breaks
        . /!/? — sentence endings
        ,      — clause boundaries
        ' '    — word boundaries    (last resort before character split)
        ''     — character split    (only if a single word > chunk_size)

    For each piece produced by a separator, if it still exceeds chunk_size,
    the algorithm RECURSES with the next finer separator. This ensures the
    coarsest possible semantic unit is always used.

    OVERLAP: each chunk shares `chunk_overlap` characters with the previous
    chunk. This preserves cross-boundary context — a sentence spanning two
    chunks is readable in full in the second chunk.

        Chunk 1: [======================]
        Chunk 2:             [======================]
                             ^----overlap----^

    Best used for:
        • Plain text, markdown, HTML (stripped), source code prose
        • Most general-purpose document ingestion
        • When you cannot afford an embedding model for chunking (SemanticChunker)
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+", r"(?<=,)\s+", " ", ""]

    def __init__(
        self,
        chunk_size:    int = 512,
        chunk_overlap: int = 64,
        separators:    list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators    = separators or self.DEFAULT_SEPARATORS

    def split(self, text: str) -> list[str]:
        """Split text into chunks. Returns list of strings."""
        text = text.strip()
        if not text:
            return []
        chunks = self._recursive_split(text, self.separators)
        return self._merge_with_overlap(chunks)

    def split_documents(
        self, text: str, source: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Split text and wrap each piece in a DocumentChunk."""
        raw_chunks = self.split(text)
        return [
            DocumentChunk(
                text=chunk, source=source, chunk_index=i,
                chunk_type="paragraph",
                metadata={**(extra_metadata or {}), "char_index": i * self.chunk_size},
            )
            for i, chunk in enumerate(raw_chunks)
        ]

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]
        sep = separators[0]
        if sep == "":
            return list(text)
        pieces = re.split(sep, text)
        pieces = [p for p in pieces if p.strip()]
        result: list[str] = []
        for piece in pieces:
            if len(piece) <= self.chunk_size:
                result.append(piece)
            else:
                result.extend(self._recursive_split(piece, separators[1:]))
        return result

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        if not pieces:
            return []
        chunks:  list[str] = []
        current: list[str] = []
        current_len = 0
        for piece in pieces:
            piece_len = len(piece)
            if current_len + piece_len + (1 if current else 0) > self.chunk_size and current:
                chunks.append("\n".join(current))
                while current and current_len > self.chunk_overlap:
                    removed = current.pop(0)
                    current_len -= len(removed) + 1
            current.append(piece)
            current_len += piece_len + (1 if len(current) > 1 else 0)
        if current:
            chunks.append("\n".join(current))
        return chunks


# =============================================================================
# SECTION 4 — SemanticChunker
#
# Groups sentences together whose embeddings are similar to each other.
# Produces the most semantically coherent chunks — each chunk is a cluster of
# related sentences, regardless of where line breaks happen to fall.
#
# This chunker is MORE EXPENSIVE than RecursiveTextChunker because it needs
# to embed each sentence. In production you would use a real embedding model.
# This implementation uses the same bag-of-words TF approach as SemanticMemory
# (cosine similarity in pure Python, zero extra deps) so it runs locally.
#
# WHEN TO USE:
#   • High-quality prose where topical shifts happen within paragraphs
#   • Academic papers, long reports with multiple sub-topics per section
#   • When you have a real embedding model and quality > speed
# =============================================================================

class SemanticChunker:
    """
    Group sentences into chunks based on embedding similarity.

    Algorithm:
        1. Split text into individual sentences (regex-based)
        2. Compute a TF (term-frequency) vector for each sentence
        3. Calculate cosine similarity between consecutive sentences
        4. When similarity DROPS below a threshold, start a new chunk
           (the similarity drop signals a topic shift)
        5. Apply size limits: if a group exceeds chunk_size, split further

    The `breakpoint_threshold` controls how sensitive the chunker is to
    topic shifts. Lower = more chunks (finer granularity).

    Args:
        chunk_size:           Target max characters per chunk.
        chunk_overlap:        Overlap sentences between consecutive chunks.
        breakpoint_threshold: Cosine similarity below which a new chunk starts.
                              Range: 0–1. Typical: 0.3–0.6.
        embed_fn:             Optional async embed function. Signature:
                              async (text: str) -> list[float]
                              If None, uses bag-of-words TF cosine similarity.

    Best used for:
        • Academic papers, essays, long-form reports
        • Any text where topics shift within paragraphs
        • When you have an OpenAI / Cohere embedding model available
    """

    def __init__(
        self,
        chunk_size:           int   = 512,
        chunk_overlap:        int   = 1,       # in sentences, not chars
        breakpoint_threshold: float = 0.4,
        embed_fn: Callable[[str], Awaitable[list[float]]] | None = None,
    ) -> None:
        self.chunk_size           = chunk_size
        self.chunk_overlap        = chunk_overlap
        self.breakpoint_threshold = breakpoint_threshold
        self._embed_fn            = embed_fn or self._bow_embed

    # ── Public ────────────────────────────────────────────────────────────────

    async def split(self, text: str) -> list[str]:
        """Split text by semantic similarity. Returns list of chunk strings."""
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        embeddings = []
        for s in sentences:
            result = self._embed_fn(s)
            # Support both sync and async embed functions
            if asyncio.iscoroutine(result):
                result = await result
            embeddings.append(result)
        similarities = [
            self._cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        # Find breakpoints where similarity drops below threshold
        breakpoints: list[int] = []
        for i, sim in enumerate(similarities):
            if sim < self.breakpoint_threshold:
                breakpoints.append(i + 1)  # index of sentence that starts new chunk

        # Build chunks from breakpoints
        chunks: list[str] = []
        prev = 0
        for bp in breakpoints:
            group = sentences[prev:bp]
            chunk_text = " ".join(group)
            if len(chunk_text) > self.chunk_size:
                # Sub-split with RecursiveTextChunker if chunk is still too large
                sub = RecursiveTextChunker(self.chunk_size, self.chunk_size // 8)
                chunks.extend(sub.split(chunk_text))
            else:
                chunks.append(chunk_text)
            # Overlap: carry last `chunk_overlap` sentences into next chunk
            prev = max(bp - self.chunk_overlap, bp)
        # Add the remaining sentences as the final chunk
        if prev < len(sentences):
            chunks.append(" ".join(sentences[prev:]))

        return [c for c in chunks if c.strip()]

    async def split_documents(
        self, text: str, source: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        raw_chunks = await self.split(text)
        return [
            DocumentChunk(
                text=chunk, source=source, chunk_index=i,
                chunk_type="paragraph",
                metadata={**(extra_metadata or {}), "chunker": "semantic"},
            )
            for i, chunk in enumerate(raw_chunks)
        ]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex. Good enough for English prose."""
        text = text.strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _bow_embed(text: str) -> list[float]:
        """
        Bag-of-words term-frequency vector. No external dependencies.
        Shared vocabulary is NOT precomputed — each call returns a Counter
        but _cosine_similarity handles variable-length vectors.
        """
        words = re.findall(r"\w+", text.lower())
        counts = Counter(words)
        vocab = sorted(counts.keys())
        return [counts[w] for w in vocab]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors of potentially different lengths."""
        n = max(len(a), len(b))
        a = a + [0.0] * (n - len(a))
        b = b + [0.0] * (n - len(b))
        dot   = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)


# =============================================================================
# SECTION 5 — LayoutAwarePDFChunker
#
# Extracts text from PDFs while preserving document structure.
# Uses pdfplumber which gives per-character font size, font name, and
# bounding box data.
#
# DETECTION HEURISTICS:
#   HEADER:         font size > page_avg + delta OR all-caps short line
#   TABLE:          pdfplumber grid extraction — kept as atomic chunks
#   FIGURE:         image bounding boxes from page.images
#   FIGURE CAPTION: text immediately below image bbox (< 30px gap)
#   PARAGRAPH:      everything else
#
# WHEN TO USE: Any PDF that has tables, headers, figures — technical reports,
# academic papers, product documentation, financial filings.
# =============================================================================

class LayoutAwarePDFChunker:
    """
    Extract and chunk a PDF while preserving its visual structure.

    Each chunk is typed:
        "header"          — section heading text
        "paragraph"       — body text under a section
        "table"           — complete table (never split)
        "figure_caption"  — caption text below a figure/image

    Every chunk carries layout metadata:
        page            — 1-based page number
        section_title   — the most recent header text seen on this page
        bbox            — (x0, top, x1, bottom) bounding box on the page
        table_rows      — row count (table chunks only)
        table_cols      — column count (table chunks only)

    Layout-aware chunking diagram:
        Page structure detected:
        ┌─────────────────────────────────┐
        │  [HEADER: 18pt bold]            │  font_size > avg + 1.5pt
        │  Introduction to Neural Nets    │
        │                                 │
        │  [PARAGRAPH: 11pt regular]      │  standard body text
        │  Neural networks are computing  │
        │  systems inspired by biological │
        │                                 │
        │  [TABLE: grid structure]        │  extracted via pdfplumber.find_tables()
        │  Layer | Neurons | Activation   │  kept as one atomic chunk
        │  Input |   784   | None         │
        │                                 │
        │  [FIGURE: image object]         │  page.images bounding box
        │  [CAPTION: 9pt italic, 20px↓]  │  text within 30px below image
        └─────────────────────────────────┘

    Args:
        chunk_size:            Max characters per paragraph chunk.
        chunk_overlap:         Overlap between paragraph chunks.
        min_header_size_delta: Font size must exceed page avg by this much
                               to be classified as a header. Default: 1.5pt.

    Best used for:
        • Technical documentation PDFs
        • Academic papers with figures and tables
        • Financial reports, legal documents
        • Any PDF where tables must not be split
    """

    def __init__(
        self,
        chunk_size:    int   = 512,
        chunk_overlap: int   = 64,
        min_header_size_delta: float = 1.5,
    ) -> None:
        self._text_chunker      = RecursiveTextChunker(chunk_size, chunk_overlap)
        self._header_size_delta = min_header_size_delta

    def chunk_pdf(self, pdf_path: str) -> list[DocumentChunk]:
        """Open a PDF and return all chunks with layout metadata."""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required for PDF chunking.\n"
                "Install it with:  pip install pdfplumber"
            )

        chunks: list[DocumentChunk] = []
        source = pdf_path.split("/")[-1]
        page_num = 0

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_chunks = self._process_page(page, page_num, source, len(chunks))
                chunks.extend(page_chunks)

        logger.info("[pdf-chunker] %s → %d chunks across %d pages", source, len(chunks), page_num)
        return chunks

    def _process_page(self, page, page_num: int, source: str, base_idx: int) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        # Step 1: extract tables as atomic chunks
        table_bboxes: list[tuple] = []
        try:
            tables = page.extract_tables({"vertical_strategy": "lines", "horizontal_strategy": "lines"})
            for table_obj in (page.find_tables() or []):
                table_bboxes.append(table_obj.bbox)
            for i, table_data in enumerate(tables or []):
                if not table_data:
                    continue
                bbox = table_bboxes[i] if i < len(table_bboxes) else (0, 0, 0, 0)
                table_text = self._format_table(table_data)
                chunks.append(DocumentChunk(
                    text=table_text, source=source,
                    chunk_index=base_idx + len(chunks),
                    chunk_type="table",
                    metadata={
                        "page": page_num, "bbox": bbox,
                        "table_rows": len(table_data),
                        "table_cols": max(len(r) for r in table_data) if table_data else 0,
                    },
                ))
        except Exception as exc:
            logger.debug("[pdf-chunker] table error page %d: %s", page_num, exc)

        # Step 2: find image bounding boxes
        image_bboxes: list[tuple] = []
        try:
            for img in (page.images or []):
                image_bboxes.append((img["x0"], img["top"], img["x1"], img["bottom"]))
        except Exception:
            pass

        # Step 3: extract words + font metadata
        avg_font_size = self._get_avg_font_size(page)
        words = page.extract_words(extra_attrs=["size", "fontname"]) or []
        lines = self._words_to_lines(words)

        # Step 4: classify lines and flush paragraphs at section boundaries
        current_section: str | None = None
        current_para_lines: list[str] = []
        current_para_bbox: tuple | None = None

        def _flush_paragraph() -> None:
            nonlocal current_para_lines, current_para_bbox
            if not current_para_lines:
                return
            para_text = " ".join(current_para_lines)
            if self._is_in_table(current_para_bbox, table_bboxes):
                current_para_lines = []; current_para_bbox = None
                return
            chunk_type = "paragraph"
            if current_para_bbox and self._is_near_image(current_para_bbox, image_bboxes):
                chunk_type = "figure_caption"
            sub_chunks = self._text_chunker.split(para_text)
            for sub_idx, sub_text in enumerate(sub_chunks):
                chunks.append(DocumentChunk(
                    text=sub_text, source=source,
                    chunk_index=base_idx + len(chunks),
                    chunk_type=chunk_type,
                    metadata={
                        "page": page_num,
                        "section_title": current_section,
                        "bbox": current_para_bbox,
                        "sub_chunk": sub_idx,
                    },
                ))
            current_para_lines = []; current_para_bbox = None

        for line_text, line_avg_size, line_bbox in lines:
            if not line_text.strip() or self._is_in_table(line_bbox, table_bboxes):
                continue
            if self._is_header(line_text, line_avg_size, avg_font_size):
                _flush_paragraph()
                current_section = line_text.strip()
                chunks.append(DocumentChunk(
                    text=line_text.strip(), source=source,
                    chunk_index=base_idx + len(chunks),
                    chunk_type="header",
                    metadata={"page": page_num, "bbox": line_bbox},
                ))
            else:
                current_para_lines.append(line_text.strip())
                if current_para_bbox is None:
                    current_para_bbox = line_bbox
                else:
                    current_para_bbox = (
                        min(current_para_bbox[0], line_bbox[0]),
                        min(current_para_bbox[1], line_bbox[1]),
                        max(current_para_bbox[2], line_bbox[2]),
                        max(current_para_bbox[3], line_bbox[3]),
                    )

        _flush_paragraph()
        return chunks

    def _get_avg_font_size(self, page) -> float:
        try:
            sizes = [c["size"] for c in (page.chars or []) if c.get("size")]
            return sum(sizes) / len(sizes) if sizes else 11.0
        except Exception:
            return 11.0

    def _words_to_lines(self, words: list[dict]) -> list[tuple[str, float, tuple]]:
        if not words:
            return []
        lines: list[tuple[str, float, tuple]] = []
        current_words: list[dict] = [words[0]]
        for word in words[1:]:
            if abs(word["top"] - current_words[-1]["top"]) <= 2:
                current_words.append(word)
            else:
                lines.append(self._words_to_line_tuple(current_words))
                current_words = [word]
        if current_words:
            lines.append(self._words_to_line_tuple(current_words))
        return lines

    def _words_to_line_tuple(self, words: list[dict]) -> tuple[str, float, tuple]:
        text     = " ".join(w["text"] for w in words)
        sizes    = [w.get("size", 11.0) for w in words if w.get("size")]
        avg_size = sum(sizes) / len(sizes) if sizes else 11.0
        x0  = min(w["x0"]    for w in words)
        top = min(w["top"]   for w in words)
        x1  = max(w["x1"]    for w in words)
        bot = max(w["bottom"] for w in words)
        return text, avg_size, (x0, top, x1, bot)

    def _is_header(self, text: str, font_size: float, avg_font_size: float) -> bool:
        text = text.strip()
        if not text:
            return False
        if font_size > avg_font_size + self._header_size_delta:
            return True
        if len(text) < 80 and text == text.upper() and text[-1] not in ".!?,;:" and len(text.split()) > 1:
            return True
        return False

    def _is_in_table(self, bbox: tuple | None, table_bboxes: list[tuple]) -> bool:
        if bbox is None:
            return False
        for tb in table_bboxes:
            if bbox[0] < tb[2] and bbox[2] > tb[0] and bbox[1] < tb[3] and bbox[3] > tb[1]:
                return True
        return False

    def _is_near_image(self, bbox: tuple, image_bboxes: list[tuple]) -> bool:
        for img in image_bboxes:
            if bbox[0] < img[2] and bbox[2] > img[0] and 0 <= (bbox[1] - img[3]) <= 30:
                return True
        return False

    def _format_table(self, table_data: list[list]) -> str:
        rows: list[str] = []
        for row in table_data:
            cells = [str(cell).strip() if cell is not None else "" for cell in row]
            rows.append(" | ".join(cells))
        return "\n".join(rows)


# =============================================================================
# SECTION 6 — Retrieval Techniques
#
# All retrieval classes share the same interface:
#
#     await retriever.search(query: str, top_k: int) -> list[DocumentChunk]
#
# This allows hot-swapping retrieval strategies without changing the agent.
# =============================================================================

# ── 6a. SemanticRetriever ─────────────────────────────────────────────────────
#
# Uses cosine similarity between dense vector embeddings.
# Best for: conceptual questions, meaning-based queries, cross-lingual search.
# Weakness: struggles with exact keyword matches (product names, IDs, codes).
#
# Example: "what are the main benefits of vector search?"
#          → finds chunks talking about ANN, embeddings, similarity even if the
#            words don't overlap with the query

class SemanticRetriever:
    """
    Dense vector retrieval using cosine similarity.

    The SemanticMemory store computes bag-of-words TF vectors by default.
    For production, inject a real embedding model via the RAGStore's embed_fn.

    Strengths:
        ✓ Finds semantically similar content even without shared keywords
        ✓ Handles synonyms, paraphrases, concept-level matches
        ✓ Excellent for open-ended conceptual questions

    Weaknesses:
        ✗ Misses exact keyword matches (e.g. product codes, proper nouns)
        ✗ Bag-of-words TF (default) less powerful than real dense embeddings
        ✗ Computationally expensive with real embeddings + large corpora
    """

    def __init__(self, memory: SemanticMemory, registry: dict[str, DocumentChunk]) -> None:
        self._memory   = memory
        self._registry = registry

    async def search(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        entries = await self._memory.search(query, top_k=top_k)
        results: list[DocumentChunk] = []
        for entry in entries:
            cid = entry.metadata.get("chunk_id")
            if cid and cid in self._registry:
                results.append(self._registry[cid])
        return results


# ── 6b. KeywordRetriever ──────────────────────────────────────────────────────
#
# TF-IDF / BM25-style lexical search. Scores chunks by term frequency of
# query words, weighted by inverse document frequency (IDF).
#
# BM25 formula (simplified):
#   score(doc, query) = Σ_term [ IDF(term) × TF(term, doc) × (k+1)
#                                / (TF(term, doc) + k × (1 - b + b × |doc|/avgdl)) ]
#
# where k=1.5, b=0.75 are standard BM25 tuning parameters.
#
# Best for: exact keyword searches, product codes, named entities, technical terms.
# Example: "pgvector PostgreSQL extension" → reliably finds chunks mentioning
#          exactly those terms even if their meaning is unrelated to the query embedding.

class KeywordRetriever:
    """
    BM25-style term-frequency-inverse-document-frequency lexical retrieval.

    Scores each chunk by how often the query terms appear in it, weighted by
    how rare each term is across the entire corpus (IDF: rare terms matter more).

    BM25 parameters:
        k1: term frequency saturation. Higher = longer docs get more credit.
            Typical: 1.2–2.0. Default: 1.5
        b:  length normalisation. 0=no normalisation, 1=full.
            Typical: 0.75. Default: 0.75

    Strengths:
        ✓ Exact keyword matching (product codes, names, IDs, technical terms)
        ✓ No embedding model required — runs in pure Python
        ✓ Predictable and debuggable: you can see exactly why a chunk ranked high
        ✓ Fast even on large corpora

    Weaknesses:
        ✗ Misses synonyms and paraphrases (no semantic understanding)
        ✗ Sensitive to spelling variations
        ✗ Poor for questions whose answer uses different vocabulary than the question
    """

    def __init__(self, registry: dict[str, DocumentChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self._registry = registry
        self._k1       = k1
        self._b        = b
        self._idf:    dict[str, float] = {}
        self._tf_map: dict[str, Counter] = {}  # chunk_id → term counts
        self._avg_dl  = 0.0
        self._build_index()

    def _build_index(self) -> None:
        """Pre-compute TF per chunk and IDF across the corpus."""
        df: Counter = Counter()   # document frequency per term
        for chunk_id, chunk in self._registry.items():
            terms = self._tokenise(chunk.text)
            self._tf_map[chunk_id] = Counter(terms)
            for term in set(terms):
                df[term] += 1

        N = max(len(self._registry), 1)
        self._avg_dl = (
            sum(sum(tf.values()) for tf in self._tf_map.values()) / N
        )
        # IDF with smoothing: log((N - df + 0.5) / (df + 0.5) + 1)
        for term, freq in df.items():
            self._idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)

    def rebuild(self) -> None:
        """Call this if chunks were added after construction."""
        self._tf_map.clear()
        self._idf.clear()
        self._build_index()

    async def search(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        """Score all chunks against the query using BM25 and return top-k."""
        query_terms = self._tokenise(query)
        if not query_terms:
            return []

        scores: list[tuple[float, str]] = []
        for chunk_id, tf in self._tf_map.items():
            dl    = sum(tf.values())
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                idf   = self._idf.get(term, 0.0)
                tf_d  = tf[term]
                denom = tf_d + self._k1 * (1 - self._b + self._b * dl / max(self._avg_dl, 1))
                score += idf * (tf_d * (self._k1 + 1)) / denom
            if score > 0:
                scores.append((score, chunk_id))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            self._registry[cid]
            for _, cid in scores[:top_k]
            if cid in self._registry
        ]

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())


# ── 6c. HybridRetriever ───────────────────────────────────────────────────────
#
# Combines semantic and keyword scores with a weighted sum (Reciprocal Rank Fusion).
# Gets the best of both worlds: conceptual matching AND keyword precision.
#
# Fusion formula used: Reciprocal Rank Fusion (RRF)
#   score(doc) = Σ_retriever [ 1 / (rank + k) ]
#   where k=60 is a damping constant that reduces the impact of top ranks.
#
# This is more robust than a weighted linear combination because it handles
# the fact that semantic scores and BM25 scores live in different numeric ranges.
#
# Best for: production default. Use when you're not sure which retrieval mode
# will work better — hybrid almost always beats either single-mode retriever.

class HybridRetriever:
    """
    Combine semantic vector search + BM25 keyword search via Reciprocal Rank Fusion.

    Reciprocal Rank Fusion (RRF):
        1. Get ranked results from BOTH retrievers
        2. Assign each document a score: 1 / (rank + k) from each retriever
        3. Sum the scores for documents appearing in both result sets
        4. Documents that rank high in BOTH retrievers float to the top

    This handles the incompatible score scales between cosine similarity (0–1)
    and BM25 (unbounded positive) without normalisation gymnastics.

    Args:
        semantic_retriever:  SemanticRetriever instance
        keyword_retriever:   KeywordRetriever instance
        rrf_k:               RRF damping constant. Default: 60 (standard value).
        semantic_weight:     Multiplier for semantic rank contribution. Default: 1.0
        keyword_weight:      Multiplier for keyword rank contribution. Default: 1.0

    Strengths:
        ✓ Best of both worlds in a single query
        ✓ Robust to spelling variations (semantic covers mismatches)
        ✓ Robust to synonym/paraphrase queries (both methods contribute)
        ✓ Production-proven: used by Weaviate, Pinecone, Chroma hybrid search

    Weaknesses:
        ✗ 2× the API calls / compute vs single-mode
        ✗ Harder to debug ("why did this chunk rank so high?")
    """

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        keyword_retriever:  KeywordRetriever,
        rrf_k:             float = 60.0,
        semantic_weight:   float = 1.0,
        keyword_weight:    float = 1.0,
    ) -> None:
        self._semantic = semantic_retriever
        self._keyword  = keyword_retriever
        self._rrf_k    = rrf_k
        self._sw       = semantic_weight
        self._kw       = keyword_weight

    async def search(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        """Fuse semantic and keyword rankings via RRF."""
        # Retrieve a larger set first, then fuse down to top_k
        fetch_k = top_k * 3
        sem_results = await self._semantic.search(query, top_k=fetch_k)
        kw_results  = await self._keyword.search(query, top_k=fetch_k)

        scores: dict[str, float] = {}
        for rank, chunk in enumerate(sem_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + \
                self._sw / (rank + self._rrf_k)
        for rank, chunk in enumerate(kw_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + \
                self._kw / (rank + self._rrf_k)

        # Build ordered list of all candidate chunks
        all_chunks: dict[str, DocumentChunk] = {
            c.chunk_id: c for c in sem_results + kw_results
        }

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [all_chunks[cid] for cid, _ in ranked[:top_k] if cid in all_chunks]


# ── 6d. HyDERetriever ─────────────────────────────────────────────────────────
#
# Hypothetical Document Embeddings (HyDE) — Gao et al. 2023.
#
# Key insight: a user's QUESTION and the ANSWER to that question have very
# different linguistic patterns. The question asks for information; the answer
# PROVIDES information. These live in different parts of the embedding space.
#
# Solution: ask the LLM to generate a HYPOTHETICAL ANSWER (even if incorrect).
# Embed that hypothesis instead of the question. The hypothesis is linguistically
# similar to real answers in the corpus, so it retrieves better.
#
# Example:
#   Query: "how does cosine similarity work?"
#   Hypothesis: "Cosine similarity measures the angle between two vectors in
#                high-dimensional space. It is computed as the dot product..."
#   → the hypothesis embedding is close to actual technical explanations in the corpus

class HyDERetriever:
    """
    Hypothetical Document Embeddings retrieval.

    Instead of embedding the user's question, this retriever:
        1. Asks the LLM to generate a short hypothetical passage that WOULD
           answer the question (even if it makes up details)
        2. Embeds that hypothetical passage for retrieval
        3. Returns chunks similar to the hypothetical answer

    This bridges the query-document linguistic gap: questions and answers
    use different vocabulary; a hypothetical answer has the same vocabulary
    as a real answer.

    Args:
        base_retriever:  Any retriever to use after hypothesis generation.
                         Typically SemanticRetriever or HybridRetriever.
        llm_client:      LLM client to generate the hypothesis.
        hypothesis_model: Model to use for hypothesis generation.
                          Use a fast/cheap model — quality matters less here.

    Strengths:
        ✓ Best retrieval quality for open-ended questions
        ✓ Handles highly abstract queries
        ✓ Bridges vocabulary mismatch between questions and documents

    Weaknesses:
        ✗ One extra LLM call per query (latency + cost)
        ✗ If the hypothesis is wildly wrong, retrieval quality degrades
        ✗ Overkill for simple keyword lookups
    """

    _HYPOTHESIS_PROMPT = (
        "Generate a short passage (2–4 sentences) that would directly answer "
        "the following question. Write it as if it were an excerpt from an "
        "authoritative document. Do not preface it with 'The answer is' — "
        "just write the passage.\n\nQuestion: {question}"
    )

    def __init__(
        self,
        base_retriever: SemanticRetriever | HybridRetriever,
        llm_client,
        hypothesis_model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._base     = base_retriever
        self._llm      = llm_client

    async def search(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        """Generate a hypothesis, then retrieve using it."""
        prompt = self._HYPOTHESIS_PROMPT.format(question=query)
        try:
            hypothesis_response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
            )
            hypothesis = hypothesis_response.content or query
            logger.debug("[HyDE] hypothesis: %s", hypothesis[:120])
        except Exception as exc:
            logger.warning("[HyDE] hypothesis generation failed: %s — falling back to direct query", exc)
            hypothesis = query

        # Retrieve using the hypothesis text instead of the raw query
        return await self._base.search(hypothesis, top_k=top_k)


# ── 6e. MultiQueryRetriever ───────────────────────────────────────────────────
#
# Generates N paraphrases of the question, retrieves for each, then deduplicates.
# Reduces sensitivity to the exact wording of the original query.
#
# Example: "what are vector databases?"
#   Paraphrase 1: "explain what a vector database is"
#   Paraphrase 2: "vector database definition and use cases"
#   Paraphrase 3: "how do databases store and search embeddings?"
# → The union of results is more complete than any single phrasing.

class MultiQueryRetriever:
    """
    Generate multiple query paraphrases and merge the retrieved results.

    Algorithm:
        1. Ask the LLM to generate `n_queries` paraphrases of the question
        2. Run the base retriever on each paraphrase
        3. Deduplicate by chunk_id (same chunk from multiple queries → keep once)
        4. Re-rank the deduplicated set by frequency (chunks retrieved by more
           queries rank higher) and then by first-retrieval rank

    Args:
        base_retriever:  Any retriever to use for each paraphrase.
        llm_client:      LLM client to generate paraphrases.
        n_queries:       How many query paraphrases to generate. Default: 3.

    Strengths:
        ✓ Reduces sensitivity to exact query wording
        ✓ Often recovers chunks missed by single-query retrieval
        ✓ Works well when users ask vague or ambiguous questions

    Weaknesses:
        ✗ n_queries × retriever calls (latency multiplier)
        ✗ One extra LLM call for paraphrase generation
        ✗ Overkill for specific, well-formed queries
    """

    _PARAPHRASE_PROMPT = (
        "Generate {n} different ways to ask the following question. "
        "Each paraphrase should approach the question from a different angle "
        "or use different vocabulary. Return only the questions, one per line, "
        "no numbering or prefixes.\n\nOriginal: {question}"
    )

    def __init__(
        self,
        base_retriever: SemanticRetriever | HybridRetriever,
        llm_client,
        n_queries: int = 3,
    ) -> None:
        self._base     = base_retriever
        self._llm      = llm_client
        self._n        = n_queries

    async def search(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        """Generate paraphrases, retrieve for each, deduplicate, re-rank."""
        paraphrases = [query]
        try:
            prompt = self._PARAPHRASE_PROMPT.format(n=self._n, question=query)
            resp = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
            )
            if resp.content:
                new_queries = [q.strip() for q in resp.content.strip().splitlines() if q.strip()]
                paraphrases.extend(new_queries[:self._n])
                logger.debug("[MultiQuery] generated %d paraphrases", len(new_queries))
        except Exception as exc:
            logger.warning("[MultiQuery] paraphrase generation failed: %s", exc)

        # Retrieve for each paraphrase
        seen_ids: dict[str, int] = {}     # chunk_id → frequency count
        all_chunks: dict[str, DocumentChunk] = {}
        for pq in paraphrases:
            results = await self._base.search(pq, top_k=top_k)
            for chunk in results:
                seen_ids[chunk.chunk_id] = seen_ids.get(chunk.chunk_id, 0) + 1
                all_chunks[chunk.chunk_id] = chunk

        # Re-rank: higher frequency = seen by more queries = more likely relevant
        ranked = sorted(seen_ids.items(), key=lambda x: x[1], reverse=True)
        return [all_chunks[cid] for cid, _ in ranked[:top_k] if cid in all_chunks]


# =============================================================================
# SECTION 7 — Self-Evaluation Loop
#
# This is the "intelligence" that makes Agentic RAG different from basic RAG.
# Before finalising its answer, the agent evaluates whether it has sufficient
# information to answer completely and confidently.
#
# The self-evaluation ask two questions:
#   1. Sufficiency:  "Do I have enough evidence to answer all parts?"
#   2. Confidence:   "How certain am I? Are there gaps or contradictions?"
#
# If sufficiency is low, the agent triggers another search round.
# This loop runs up to max_eval_rounds times.
#
# WHY IT MATTERS:
#   Without self-evaluation, the agent answers even when its retrieved context
#   is incomplete. The result is hallucination or partial answers presented as
#   complete ones.
#
#   With self-evaluation, the agent only synthesises when it has sufficient
#   evidence. If it cannot find enough information, it says so honestly.
# =============================================================================

@dataclass
class SufficiencyResult:
    """
    Output of the self-evaluation step.

    sufficient:    True if the agent has enough information to answer
    confidence:    0.0–1.0 score of how confident the agent is
    missing_info:  What information is still needed (empty if sufficient)
    suggested_queries: Follow-up search queries to fill the gaps
    reasoning:     Brief explanation of the sufficiency assessment
    """
    sufficient:        bool
    confidence:        float
    missing_info:      list[str]
    suggested_queries: list[str]
    reasoning:         str


async def evaluate_sufficiency(
    question: str,
    retrieved_chunks: list[DocumentChunk],
    llm_client,
    confidence_threshold: float = 0.7,
) -> SufficiencyResult:
    """
    Ask the LLM to evaluate whether the retrieved context is sufficient.

    This is the self-evaluation step in the agentic RAG loop. The LLM is
    given the original question and the retrieved chunks, and asked:
      - Can I answer this question completely with this context?
      - What is my confidence level?
      - What specific information is missing?
      - What follow-up searches would help?

    Args:
        question:             The original user question.
        retrieved_chunks:     All chunks retrieved so far.
        llm_client:           LLM client to perform the evaluation.
        confidence_threshold: Minimum confidence to declare 'sufficient'.

    Returns:
        SufficiencyResult with the evaluation outcome.
    """
    if not retrieved_chunks:
        return SufficiencyResult(
            sufficient=False, confidence=0.0,
            missing_info=["No information retrieved yet"],
            suggested_queries=[question],
            reasoning="No context available — search required",
        )

    context_text = "\n\n".join(
        f"[Chunk from {c.source}, type={c.chunk_type}]\n{c.text[:400]}"
        for c in retrieved_chunks[:8]  # limit to 8 chunks to avoid prompt overflow
    )

    eval_prompt = f"""You are evaluating whether the retrieved context is sufficient to answer a question.

QUESTION: {question}

RETRIEVED CONTEXT:
{context_text}

Evaluate:
1. Can you answer the question COMPLETELY using only this context?
2. Confidence (0.0-1.0): how certain are you the answer is accurate and complete?
3. What specific information is missing? (empty list if nothing is missing)
4. What follow-up search queries would retrieve the missing information?
5. Brief reasoning.

Respond in this exact format:
SUFFICIENT: yes/no
CONFIDENCE: 0.XX
MISSING: item1 | item2 | item3 (or "none")
QUERIES: query1 | query2 (or "none")
REASONING: your brief reasoning here
"""

    try:
        response = await llm_client.complete(
            messages=[{"role": "user", "content": eval_prompt}],
        )
        text = response.content or ""
        return _parse_sufficiency_response(text, confidence_threshold)
    except Exception as exc:
        logger.warning("[self-eval] evaluation failed: %s — assuming sufficient", exc)
        return SufficiencyResult(
            sufficient=True, confidence=0.5,
            missing_info=[], suggested_queries=[],
            reasoning=f"Evaluation error: {exc}",
        )


def _parse_sufficiency_response(text: str, threshold: float) -> SufficiencyResult:
    """Parse the structured self-evaluation response."""
    lines = {
        line.split(":")[0].strip(): ":".join(line.split(":")[1:]).strip()
        for line in text.strip().splitlines()
        if ":" in line
    }

    sufficient_raw = lines.get("SUFFICIENT", "no").lower().strip()
    sufficient     = sufficient_raw in ("yes", "true", "1")

    try:
        confidence = float(lines.get("CONFIDENCE", "0.5"))
    except ValueError:
        confidence = 0.5

    if confidence < threshold:
        sufficient = False

    missing_raw = lines.get("MISSING", "none")
    missing     = [] if missing_raw.lower() == "none" else [m.strip() for m in missing_raw.split("|")]

    queries_raw = lines.get("QUERIES", "none")
    queries     = [] if queries_raw.lower() == "none" else [q.strip() for q in queries_raw.split("|")]

    reasoning = lines.get("REASONING", "")

    return SufficiencyResult(
        sufficient=sufficient,
        confidence=confidence,
        missing_info=missing,
        suggested_queries=queries,
        reasoning=reasoning,
    )


# =============================================================================
# SECTION 8 — RAGStore
#
# The ingestion + retrieval layer. Wraps all four retrieval strategies behind
# a unified interface, and owns the SemanticMemory + chunk registry.
# =============================================================================

class RAGStore:
    """
    Complete RAG ingestion and retrieval pipeline with multiple strategies.

    Ingestion:
        await rag.ingest_text(text, source)        — chunk with RecursiveTextChunker
        await rag.ingest_text_fixed(text, source)  — chunk with FixedSizeChunker
        await rag.ingest_text_semantic(text, source) — chunk with SemanticChunker
        await rag.ingest_pdf(path)                 — chunk with LayoutAwarePDFChunker
        await rag.ingest_chunks(chunks)            — ingest pre-built chunk list

    Retrieval:
        await rag.search_semantic(query, top_k)    — dense vector search
        await rag.search_keyword(query, top_k)     — BM25 lexical search
        await rag.search_hybrid(query, top_k)      — semantic + keyword fusion
        await rag.search_hyde(query, top_k, llm)   — hypothetical doc embedding
        await rag.search_multiquery(query, top_k, llm) — multi-paraphrase search

    The active retriever used by the MCP tools can be hot-swapped via:
        rag.set_retrieval_mode("semantic" | "keyword" | "hybrid" | "hyde" | "multiquery")
    """

    def __init__(
        self,
        chunk_size:    int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self._memory           = SemanticMemory()
        self._text_chunker     = RecursiveTextChunker(chunk_size, chunk_overlap)
        self._fixed_chunker    = FixedSizeChunker(chunk_size, chunk_overlap)
        self._semantic_chunker = SemanticChunker(chunk_size)
        self._pdf_chunker      = LayoutAwarePDFChunker(chunk_size, chunk_overlap)
        self._registry:        dict[str, DocumentChunk] = {}

        # Retriever instances — built lazily once chunks exist
        self._semantic_retriever: SemanticRetriever | None = None
        self._keyword_retriever:  KeywordRetriever  | None = None
        self._hybrid_retriever:   HybridRetriever   | None = None
        self._retrieval_mode:     str               = "hybrid"

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def ingest_text(
        self, text: str, source: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> int:
        """Ingest plain text with RecursiveTextChunker (default)."""
        chunks = self._text_chunker.split_documents(text, source, extra_metadata)
        return await self.ingest_chunks(chunks)

    async def ingest_text_fixed(
        self, text: str, source: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> int:
        """Ingest plain text with FixedSizeChunker."""
        chunks = self._fixed_chunker.split_documents(text, source, extra_metadata)
        for c in chunks:
            c.metadata["chunker"] = "fixed_size"
        return await self.ingest_chunks(chunks)

    async def ingest_text_semantic(
        self, text: str, source: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> int:
        """Ingest plain text with SemanticChunker."""
        chunks = await self._semantic_chunker.split_documents(text, source, extra_metadata)
        return await self.ingest_chunks(chunks)

    async def ingest_pdf(self, pdf_path: str) -> int:
        """Ingest PDF with LayoutAwarePDFChunker."""
        chunks = self._pdf_chunker.chunk_pdf(pdf_path)
        return await self.ingest_chunks(chunks)

    async def ingest_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Store a pre-built list of DocumentChunks and rebuild indexes."""
        for chunk in chunks:
            self._registry[chunk.chunk_id] = chunk
            await self._memory.add(
                chunk.to_storage_text(),
                metadata={
                    "chunk_id":   chunk.chunk_id,
                    "source":     chunk.source,
                    "chunk_type": chunk.chunk_type,
                    "page":       chunk.metadata.get("page"),
                    "section":    chunk.metadata.get("section_title"),
                },
            )
        self._build_retrievers()
        return len(chunks)

    def _build_retrievers(self) -> None:
        """Rebuild all retriever indexes after new chunks are added."""
        self._semantic_retriever = SemanticRetriever(self._memory, self._registry)
        self._keyword_retriever  = KeywordRetriever(self._registry)
        self._hybrid_retriever   = HybridRetriever(
            self._semantic_retriever, self._keyword_retriever
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def search_semantic(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        if not self._semantic_retriever:
            return []
        return await self._semantic_retriever.search(query, top_k)

    async def search_keyword(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        if not self._keyword_retriever:
            return []
        return await self._keyword_retriever.search(query, top_k)

    async def search_hybrid(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        if not self._hybrid_retriever:
            return []
        return await self._hybrid_retriever.search(query, top_k)

    async def search_hyde(self, query: str, top_k: int, llm_client) -> list[DocumentChunk]:
        if not self._semantic_retriever:
            return []
        retriever = HyDERetriever(self._semantic_retriever, llm_client)
        return await retriever.search(query, top_k)

    async def search_multiquery(self, query: str, top_k: int, llm_client) -> list[DocumentChunk]:
        if not self._hybrid_retriever:
            return []
        retriever = MultiQueryRetriever(self._hybrid_retriever, llm_client)
        return await retriever.search(query, top_k)

    async def search(self, query: str, top_k: int = 5, llm_client=None) -> list[DocumentChunk]:
        """Search using the currently active retrieval mode."""
        if self._retrieval_mode == "keyword":
            return await self.search_keyword(query, top_k)
        elif self._retrieval_mode == "hybrid":
            return await self.search_hybrid(query, top_k)
        elif self._retrieval_mode == "hyde" and llm_client:
            return await self.search_hyde(query, top_k, llm_client)
        elif self._retrieval_mode == "multiquery" and llm_client:
            return await self.search_multiquery(query, top_k, llm_client)
        else:
            return await self.search_semantic(query, top_k)

    def set_retrieval_mode(self, mode: str) -> None:
        """Hot-swap retrieval strategy: semantic | keyword | hybrid | hyde | multiquery"""
        valid = {"semantic", "keyword", "hybrid", "hyde", "multiquery"}
        if mode not in valid:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {valid}")
        self._retrieval_mode = mode
        logger.info("[rag-store] retrieval mode → %s", mode)

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        return self._registry.get(chunk_id)

    def stats(self) -> dict[str, Any]:
        sources = {c.source for c in self._registry.values()}
        by_type: dict[str, int] = {}
        for c in self._registry.values():
            by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
        by_chunker: dict[str, int] = {}
        for c in self._registry.values():
            chunker = c.metadata.get("chunker", "recursive")
            by_chunker[chunker] = by_chunker.get(chunker, 0) + 1
        return {
            "total_chunks": len(self._registry),
            "sources":      list(sources),
            "by_type":      by_type,
            "by_chunker":   by_chunker,
        }


# =============================================================================
# SECTION 9 — MCP Server
#
# Expose the RAGStore as MCP tools so any agent pattern can call them.
# The tools support the FULL agentic loop:
#   - search_knowledge()   — primary retrieval (uses active mode)
#   - search_by_mode()     — explicit mode selection per query
#   - evaluate_my_context()— self-evaluation: "is this enough to answer?"
#   - ingest_text/pdf()    — runtime knowledge base updates
#   - get_document_stats() — observability
# =============================================================================

rag_store = RAGStore(chunk_size=600, chunk_overlap=80)
_llm_for_hyde = AnthropicClient("claude-haiku-4-5-20251001")  # cheap model for hypothesis gen

app = FastMCP("agentic_rag_tools")


@app.tool
async def search_knowledge(query: str, top_k: int = 5) -> str:
    """
    Search the knowledge base for information relevant to the query.
    Uses hybrid retrieval (semantic + keyword fusion) by default.

    Returns the top-k most relevant chunks with source, page, section, and
    chunk type so you can cite them precisely.

    Use this tool multiple times with DIFFERENT query angles if the first
    search does not fully cover all aspects of the question.
    """
    chunks = await rag_store.search(query, top_k=top_k)
    return _format_chunks(chunks, query)


@app.tool
async def search_by_mode(
    query:    str,
    mode:     str = "hybrid",
    top_k:    int = 5,
) -> str:
    """
    Search the knowledge base using a specific retrieval technique.

    mode options:
        "semantic"   — cosine-similarity vector search
                       Best for: conceptual questions, synonyms, paraphrases
        "keyword"    — BM25 lexical search
                       Best for: exact terms, product names, codes, IDs
        "hybrid"     — semantic + keyword fusion (recommended default)
                       Best for: most real-world queries
        "hyde"       — hypothetical document embedding
                       Best for: vague or abstract questions
        "multiquery" — multi-paraphrase retrieval
                       Best for: when query wording might be suboptimal

    Returns chunks with full provenance metadata.
    """
    old_mode = rag_store._retrieval_mode
    rag_store.set_retrieval_mode(mode)
    try:
        if mode in ("hyde", "multiquery"):
            chunks = await rag_store.search(query, top_k=top_k, llm_client=_llm_for_hyde)
        else:
            chunks = await rag_store.search(query, top_k=top_k)
    finally:
        rag_store.set_retrieval_mode(old_mode)
    return _format_chunks(chunks, query)


@app.tool
async def check_sufficiency(
    question:        str,
    retrieved_so_far: str,
) -> str:
    """
    Self-evaluate: do the results retrieved so far sufficiently answer the question?

    Call this AFTER searching to decide whether to search again or synthesise.
    Returns: whether you have enough information, confidence score, what is
    missing, and suggested follow-up queries if gaps remain.

    Use this to implement the self-evaluation loop:
        1. search_knowledge(question)
        2. check_sufficiency(question, results)
        3. If not sufficient: search_knowledge(suggested_query)
        4. check_sufficiency again
        5. When sufficient: synthesise your final answer
    """
    # Parse the retrieved text back into minimal chunk objects for evaluation
    # In production you would pass structured data; here we use text heuristics
    lines  = retrieved_so_far.strip().splitlines()
    chunks = [
        DocumentChunk(
            text=retrieved_so_far,
            source="retrieved_context",
            chunk_index=0,
        )
    ]

    result = await evaluate_sufficiency(
        question=question,
        retrieved_chunks=chunks,
        llm_client=_llm_for_hyde,
        confidence_threshold=0.65,
    )

    output = [
        f"Sufficient: {'YES' if result.sufficient else 'NO'}",
        f"Confidence: {result.confidence:.2f}",
        f"Reasoning:  {result.reasoning}",
    ]
    if result.missing_info:
        output.append(f"Missing:    {'; '.join(result.missing_info)}")
    if result.suggested_queries:
        output.append(f"Try these searches:")
        for q in result.suggested_queries:
            output.append(f"  - {q}")
    return "\n".join(output)


@app.tool
async def ingest_text(text: str, source: str = "user_input") -> str:
    """Add plain text to the knowledge base at runtime using RecursiveTextChunker."""
    count = await rag_store.ingest_text(text, source=source)
    return f"Ingested {count} chunks from source '{source}'."


@app.tool
async def ingest_pdf(file_path: str) -> str:
    """
    Ingest a PDF into the knowledge base with full layout awareness.
    Headers, tables, figure captions, and paragraphs are typed chunks.
    """
    try:
        count = await rag_store.ingest_pdf(file_path)
        stats = rag_store.stats()
        return (
            f"Ingested {count} chunks from '{file_path}'.\n"
            f"Knowledge base now has {stats['total_chunks']} total chunks.\n"
            f"Chunk types: {stats['by_type']}"
        )
    except ImportError as exc:
        return f"Cannot ingest PDF: {exc}"
    except FileNotFoundError:
        return f"File not found: {file_path}"


@app.tool
async def get_chunk_detail(chunk_id: str) -> str:
    """Retrieve the full text and metadata for a chunk by ID."""
    chunk = rag_store.get_chunk(chunk_id)
    if not chunk:
        return f"Chunk '{chunk_id}' not found."
    return (
        f"Chunk ID:   {chunk.chunk_id}\n"
        f"Source:     {chunk.source}\n"
        f"Type:       {chunk.chunk_type}\n"
        f"Page:       {chunk.metadata.get('page', 'N/A')}\n"
        f"Section:    {chunk.metadata.get('section_title', 'N/A')}\n"
        f"Index:      {chunk.chunk_index}\n\n"
        f"--- Full Text ---\n{chunk.text}"
    )


@app.tool
async def get_document_stats() -> str:
    """Show knowledge base statistics: chunk counts by source, type, and chunker."""
    stats = rag_store.stats()
    lines = [
        f"Total chunks:  {stats['total_chunks']}",
        f"Sources:       {', '.join(stats['sources']) or 'none'}",
        "Chunk types:",
    ]
    for ctype, count in stats["by_type"].items():
        lines.append(f"  {ctype:20s} {count}")
    if stats.get("by_chunker"):
        lines.append("Chunkers used:")
        for chunker, count in stats["by_chunker"].items():
            lines.append(f"  {chunker:20s} {count}")
    return "\n".join(lines)


# ── Helper ────────────────────────────────────────────────────────────────────

def _format_chunks(chunks: list[DocumentChunk], query: str = "") -> str:
    if not chunks:
        return f"No relevant information found for: '{query}'. Try a different search query or mode."
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        meta = []
        if chunk.metadata.get("page"):
            meta.append(f"p.{chunk.metadata['page']}")
        if chunk.metadata.get("section_title"):
            meta.append(f"§ {chunk.metadata['section_title']}")
        meta.append(chunk.chunk_type)
        meta_str = " | ".join(meta)
        parts.append(
            f"[Result {i}] {chunk.source} ({meta_str}) [id:{chunk.chunk_id}]\n"
            f"{chunk.text}\n"
        )
    return "\n".join(parts)


# =============================================================================
# SECTION 10 — Sample Documents
# =============================================================================

SAMPLE_DOCS: list[tuple[str, str]] = [
    (
        "python_overview.txt",
        """
Python Programming Language — Overview

Python is a high-level, interpreted programming language known for its
clear syntax and readability. Created by Guido van Rossum and first released
in 1991, Python has grown to become one of the most popular programming
languages in the world.

Design Philosophy
Python's design philosophy emphasises code readability, particularly through
the use of significant whitespace. Its syntax allows programmers to express
concepts in fewer lines of code than would be possible in languages like C++
or Java. The language constructs and object-oriented approach aim to help
programmers write clear, logical code for small and large-scale projects.

Key Features
Python is dynamically typed — you do not declare variable types explicitly.
It uses garbage collection to manage memory automatically. The language
supports multiple programming paradigms, including structured, object-
oriented, and functional programming.

Python's standard library is very large, covering areas such as string
operations, internet protocols, software engineering tools, and operating
system interfaces.

Ecosystem
The Python Package Index (PyPI) hosts over 400,000 packages. Major
frameworks built on Python include Django and FastAPI for web development,
NumPy, Pandas, and PyTorch for data science and machine learning. Python is
the dominant language in machine learning and AI research as of 2026.
        """,
    ),
    (
        "vector_databases.txt",
        """
Vector Databases — Technical Overview

A vector database is a database management system that stores data as
high-dimensional vectors — numerical representations of data objects.
These vectors are called embeddings. The key operation vector databases
optimise is Approximate Nearest Neighbour (ANN) search.

How Embeddings Work
An embedding model (such as OpenAI's text-embedding-3-small or Google's
text-embedding-004) converts text, images, or other data into a vector of
floating-point numbers. Semantically similar items produce vectors that are
close together in the high-dimensional space.

For example, "king" and "queen" would have nearby embeddings because they
share semantic context. "king" and "database" would be far apart.

Distance Metrics
Cosine similarity measures the angle between two vectors, ignoring magnitude.
It is the most common metric for text similarity tasks. Values range from
-1 (opposite meaning) to 1 (identical meaning).

Euclidean distance (L2 norm) measures the straight-line distance between
two points in the vector space. It is sensitive to vector magnitude.

Dot product is similar to cosine similarity but includes magnitude effects.
It is efficient to compute and works well when vectors are normalised.

Popular Vector Databases (as of 2026)
pgvector — PostgreSQL extension. Adds a vector column type and cosine/L2
indexes. Good choice if you already run PostgreSQL.

Chroma — Embedded database, runs in-process like SQLite. Good for
prototyping and small-scale applications.

Pinecone — Fully managed cloud vector database. Best for production at scale
without wanting to manage infrastructure.

Weaviate — Open-source, supports multi-modal embeddings, GraphQL query
interface, and has a cloud offering.

Chunking and RAG
In Retrieval-Augmented Generation, documents are split into chunks, each
chunk is embedded, and the embeddings are stored in the vector database.
At query time, the user's question is embedded using the same model, and
the nearest-neighbour search returns the most semantically relevant chunks.
These chunks become the context window for the language model's response.
        """,
    ),
    (
        "rag_architecture.txt",
        """
RAG System Architecture — Design Patterns

Retrieval-Augmented Generation (RAG) combines a retrieval system with a
generative language model. The retrieval system grounds the model's responses
in factual, up-to-date information from a document corpus.

Basic RAG Pipeline
The simplest RAG pipeline has three stages:

1. Ingestion: Documents are chunked, embedded, and stored in a vector
   database. This is an offline process that runs when documents change.

2. Retrieval: At query time, the user's question is embedded. The vector
   database returns the top-k most similar chunks by cosine similarity.

3. Generation: The retrieved chunks are injected into the LLM prompt as
   context. The LLM generates an answer grounded in this context.

Chunking Strategies
Fixed-size chunking splits documents into fixed-length windows (e.g. 512
tokens) with optional overlap. Simple but can cut sentences mid-way.

Recursive character splitting tries paragraph boundaries first, then
sentence boundaries, then word boundaries. Produces more coherent chunks.

Semantic chunking groups sentences with similar embeddings together. More
expensive to compute but produces semantically coherent chunks.

Layout-aware chunking for PDFs preserves headers, tables, and figure
captions as typed chunks. The heading becomes metadata for all chunks
within its section, improving retrieval relevance.

Advanced RAG Patterns

Hypothetical Document Embeddings (HyDE): generate a hypothetical answer,
embed it, then use that embedding for retrieval. The hypothesis is often
closer to the actual document than the raw question.

Multi-query retrieval: generate 3-5 paraphrases of the question, retrieve
for each, then deduplicate and re-rank. Reduces query-wording sensitivity.

Re-ranking: after initial retrieval with ANN search, apply a cross-encoder
re-ranker (like Cohere Rerank) to score retrieved chunks against the full
question. More accurate but more expensive.

Self-RAG: the model decides when to retrieve, what to retrieve, and
evaluates the relevance and support of each retrieved passage.

Agentic RAG: the agent has full control over the retrieval process. It
can search multiple times, ask follow-up questions about the corpus, assess
whether it has enough information, and synthesise across multiple documents.
        """,
    ),
]


# =============================================================================
# SECTION 11 — Demo A: Chunking Technique Comparison
#
# Apply all 4 chunkers to the same document and show the differences.
# This is purely educational — no LLM calls needed.
# =============================================================================

async def demo_a_chunking_comparison() -> None:
    print("\n" + "=" * 70)
    print("DEMO A — Chunking Technique Comparison (same text, 4 chunkers)")
    print("=" * 70)

    sample_text = SAMPLE_DOCS[1][1].strip()  # vector_databases.txt
    source = "vector_databases.txt"

    chunkers = [
        ("FixedSize (512, overlap=50)",   FixedSizeChunker(512, 50)),
        ("Recursive (512, overlap=64)",   RecursiveTextChunker(512, 64)),
        # SemanticChunker is async so we call it separately below
    ]

    for name, chunker in chunkers:
        chunks = chunker.split_documents(sample_text, source)
        print(f"\n{name}:")
        print(f"  → {len(chunks)} chunks produced")
        for i, chunk in enumerate(chunks[:3]):
            preview = chunk.text[:100].replace("\n", " ")
            print(f"  [{i}] type={chunk.chunk_type}  '{preview}...'")

    # SemanticChunker — async
    sem_chunker = SemanticChunker(chunk_size=512, breakpoint_threshold=0.3)
    sem_chunks  = await sem_chunker.split_documents(sample_text, source)
    print(f"\nSemantic (threshold=0.30):")
    print(f"  → {len(sem_chunks)} chunks produced")
    for i, chunk in enumerate(sem_chunks[:3]):
        preview = chunk.text[:100].replace("\n", " ")
        print(f"  [{i}] '{preview}...'")

    print("""
  Takeaway:
    FixedSize   — predictable count, but cuts mid-sentence (noisy embeddings)
    Recursive   — paragraph-aligned chunks, best general-purpose choice
    Semantic    — topic-coherent groups, more or fewer chunks based on content
    LayoutPDF   — not shown here (needs a PDF), but adds type + page metadata
    """)


# =============================================================================
# SECTION 12 — Demo B: Retrieval Technique Comparison
#
# Run all 5 retrieval strategies against the same query.
# Show how different techniques surface different chunks.
# =============================================================================

async def demo_b_retrieval_comparison() -> None:
    print("\n" + "=" * 70)
    print("DEMO B — Retrieval Technique Comparison (same query, 5 strategies)")
    print("=" * 70)

    query    = "what distance metrics are used in vector databases?"
    top_k    = 3
    llm      = AnthropicClient("claude-haiku-4-5-20251001")

    modes = [
        ("Semantic  (vector cosine sim)", "semantic"),
        ("Keyword   (BM25 lexical)",      "keyword"),
        ("Hybrid    (RRF fusion)",         "hybrid"),
        ("HyDE      (hypothetical doc)",   "hyde"),
        ("MultiQuery (paraphrase fan-out)","multiquery"),
    ]

    print(f"\nQuery: '{query}'")
    print("-" * 60)

    for label, mode in modes:
        print(f"\n{label}:")
        if mode in ("hyde", "multiquery"):
            chunks = await rag_store.search(query, top_k=top_k, llm_client=llm)
        else:
            rag_store.set_retrieval_mode(mode)
            chunks = await rag_store.search(query, top_k=top_k)
        for i, chunk in enumerate(chunks, 1):
            preview = chunk.text[:80].replace("\n", " ")
            print(f"  [{i}] {chunk.source}  '{preview}...'")

    rag_store.set_retrieval_mode("hybrid")  # restore default


# =============================================================================
# SECTION 13 — Demo C: Simple One-Shot RAG
#
# The baseline: one search → one LLM call → answer.
# Fast but can miss information not captured by a single query angle.
# =============================================================================

async def demo_c_simple_rag() -> None:
    print("\n" + "=" * 70)
    print("DEMO C — Simple One-Shot RAG (SingleAgentLoop, 1–2 searches)")
    print("=" * 70)

    config = AgentConfig(
        mcp_server_config=app,
        system_prompt=(
            "You are a knowledge base assistant. "
            "Always call search_knowledge() before answering. "
            "Cite the source in your answer."
        ),
        max_iterations=4,
    )
    agent = SingleAgentLoop(
        llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
        config=config,
    )

    q = "What is cosine similarity and why is it used in vector databases?"
    print(f"\nQuestion: {q}")
    print("-" * 60)
    answer = await agent.run(q)
    print(answer)
    print("""
  Observation:
    The agent made 1–2 tool calls. Fast, but if the answer required
    cross-document synthesis, it might have missed some context.
    See Demo D for the agentic approach.
    """)


# =============================================================================
# SECTION 14 — Demo D: Agentic RAG with Self-Evaluation Loop
#
# THE AGENTIC LOOP STRUCTURE:
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  OBSERVE:   Read the question. Understand what is being asked.          │
# │             Call get_document_stats() to know what's in the corpus.     │
# │                                                                         │
# │  PLAN:      Decompose the question into sub-topics.                     │
# │             PlannerExecutorPattern generates this as an ExecutionPlan.  │
# │                                                                         │
# │  SEARCH:    For each sub-topic, call search_knowledge() or              │
# │             search_by_mode() with the most appropriate mode.            │
# │                                                                         │
# │  EVALUATE:  Call check_sufficiency() to assess: do I have enough?       │
# │             check_sufficiency() runs evaluate_sufficiency() which asks  │
# │             the LLM: "Can I answer completely with this context?"       │
# │                                                                         │
# │  RE-SEARCH: If not sufficient, call search_knowledge() with the         │
# │             suggested follow-up queries from check_sufficiency().       │
# │             Repeat until sufficient or max_iterations reached.          │
# │                                                                         │
# │  SYNTHESISE: Build the final answer using only retrieved context.       │
# │             Cite every factual claim with source + chunk_id.            │
# └─────────────────────────────────────────────────────────────────────────┘
# =============================================================================

async def demo_d_agentic_rag_with_self_eval() -> None:
    print("\n" + "=" * 70)
    print("DEMO D — Agentic RAG with Self-Evaluation Loop")
    print("=" * 70)
    print("""
  Agentic loop flow:
    OBSERVE → PLAN → SEARCH → EVALUATE → [RE-SEARCH if gaps] → SYNTHESISE

  Key MCP tools driving the loop:
    search_knowledge()   — primary retrieval
    search_by_mode()     — retrieval with explicit technique selection
    check_sufficiency()  — self-evaluation: am I done searching?
    get_document_stats() — observe: what is in the knowledge base?
    """)

    config = AgentConfig(
        mcp_server_config=app,
        system_prompt=(
            "You are a senior AI engineer answering technical questions about RAG systems.\n\n"
            "AGENTIC SEARCH LOOP — follow this process:\n"
            "1. OBSERVE: Call get_document_stats() to understand what is available.\n"
            "2. PLAN: Identify the sub-topics the question covers.\n"
            "3. SEARCH: Call search_knowledge() for each sub-topic with a targeted query.\n"
            "   - For conceptual questions, use search_by_mode() with mode='semantic'\n"
            "   - For exact terms (names, codes), use search_by_mode() with mode='keyword'\n"
            "   - Default: search_knowledge() uses hybrid mode automatically\n"
            "4. EVALUATE: After each search round, call check_sufficiency() with the question "
            "and all retrieved text so far. If it says 'NO' with follow-up queries, search again.\n"
            "5. SYNTHESISE: Only finalise your answer once check_sufficiency() says 'YES' "
            "or you have completed 3 search rounds. Always cite sources."
        ),
        max_iterations=16,
    )

    agentic_agent = PlannerExecutorPattern(
        planner_client=AnthropicClient("claude-sonnet-4-6"),
        executor_client=AnthropicClient("claude-haiku-4-5-20251001"),
        config=config,
        max_replan_attempts=1,
    )

    question = (
        "I'm building a production RAG system in Python. "
        "Explain: (1) which chunking strategy I should use and why, "
        "(2) which vector database to choose for a startup, "
        "(3) how agentic RAG improves on basic RAG, "
        "and (4) what Python libraries I should consider."
    )

    print(f"\nQuestion: {question}")
    print("-" * 60)
    print("[The agent will PLAN searches, EXECUTE them, self-EVALUATE, and SYNTHESISE.]\n")

    answer = await agentic_agent.run(question)
    print(f"\nFinal Answer:\n{answer}")


# =============================================================================
# SECTION 15 — Demo E: EvaluatorOptimizer RAG
#
# The agent writes an answer, evaluates its own output against a rubric,
# then rewrites if the quality score is below threshold.
#
# This is different from the self-evaluation loop in Demo D:
#   Demo D:  self-evaluation of RETRIEVAL sufficiency ("do I have enough context?")
#   Demo E:  self-evaluation of ANSWER QUALITY ("is my answer well-structured?")
#
# These can be combined: first ensure retrieval sufficiency (Demo D approach),
# then run the answer through an EvaluatorOptimizer loop (Demo E).
# =============================================================================

async def demo_e_evaluator_optimizer_rag() -> None:
    print("\n" + "=" * 70)
    print("DEMO E — EvaluatorOptimizer RAG (write → evaluate → rewrite)")
    print("=" * 70)
    print("""
  EvaluatorOptimizer loop:
    1. Generator agent drafts an answer (searches the knowledge base)
    2. Evaluator scores the answer against rubric criteria
    3. If score < threshold: evaluator's feedback → generator rewrites
    4. Repeat up to max_rounds times

  This adds quality assurance on TOP of retrieval.
  Think of it as: first search well (Demo D), then write well (Demo E).
    """)

    # The generator agent: searches the knowledge base and drafts an answer
    generator_config = AgentConfig(
        mcp_server_config=app,
        system_prompt=(
            "You are a technical writer. "
            "Search the knowledge base to answer the question, then write a clear, "
            "well-structured response. Cite sources. Be specific and concrete."
        ),
        max_iterations=8,
    )
    generator_client = AnthropicClient("claude-haiku-4-5-20251001")

    # The evaluator: scores on four criteria
    evaluator = RubricEvaluator(
        llm_client=AnthropicClient("claude-sonnet-4-6"),
        criteria=[
            RubricCriterion(
                name="accuracy",
                description="Are all factual claims correct and supported by the retrieved context?",
                weight=2.0,   # highest weight: accuracy matters most
            ),
            RubricCriterion(
                name="completeness",
                description="Does the answer address all parts of the question?",
                weight=1.5,
            ),
            RubricCriterion(
                name="citations",
                description="Are sources cited for key claims (source name and/or section)?",
                weight=1.0,
            ),
            RubricCriterion(
                name="clarity",
                description="Is the answer clearly structured and easy to follow?",
                weight=1.0,
            ),
        ],
        pass_threshold=0.75,  # 75% of max weighted score required to pass
    )

    agent = EvaluatorOptimizerPattern(
        generator_client=generator_client,
        evaluator=evaluator,
        config=generator_config,
        max_rounds=3,
    )

    q = "What is the difference between semantic chunking and recursive text chunking for RAG systems?"
    print(f"\nQuestion: {q}")
    print("-" * 60)
    answer = await agent.run(q)
    print(f"\nFinal Answer (after evaluation/rewrite loop):\n{answer}")


# =============================================================================
# SECTION 16 — Demo F: Use-Case Matrix
#
# Show which chunker + which retrieval technique is best for which scenario.
# This is a reference guide, not a live demo (no LLM calls).
# =============================================================================

def demo_f_use_case_matrix() -> None:
    print("\n" + "=" * 70)
    print("DEMO F — Use-Case Matrix: Chunker + Retriever recommendations")
    print("=" * 70)

    matrix = [
        {
            "use_case":   "Customer support FAQ (short Q&A pairs)",
            "chunker":    "FixedSize (small, e.g. 256 chars)",
            "retriever":  "Hybrid",
            "why": (
                "FAQ entries are already self-contained. FixedSize is fast and "
                "won't split meaningful content. Hybrid catches both exact product "
                "names (keyword) and paraphrased questions (semantic)."
            ),
        },
        {
            "use_case":   "Technical documentation (long prose manuals)",
            "chunker":    "RecursiveText (512–1024 chars, overlap=10%)",
            "retriever":  "Semantic or Hybrid",
            "why": (
                "Prose has paragraphs as natural units. RecursiveText respects them. "
                "Semantic retrieval handles the diverse vocabulary of technical writing. "
                "Add Hybrid if the docs contain specific API names or error codes."
            ),
        },
        {
            "use_case":   "Academic papers / research reports",
            "chunker":    "SemanticChunker (topic-aware grouping)",
            "retriever":  "HyDE or MultiQuery",
            "why": (
                "Academic papers have multiple sub-topics per section. SemanticChunker "
                "groups related sentences together. HyDE helps because queries are often "
                "abstract ('what methods improve retrieval precision?') while papers use "
                "formal/different vocabulary."
            ),
        },
        {
            "use_case":   "PDF financial reports with tables",
            "chunker":    "LayoutAwarePDF (keeps tables atomic)",
            "retriever":  "Hybrid with table-type filter",
            "why": (
                "Financial PDFs have precise numerical tables that must not be split. "
                "LayoutAwarePDF keeps tables as single chunks. Hybrid retrieval ensures "
                "both the table structure (keyword: 'revenue 2025') and its context "
                "(semantic: 'quarterly performance') are findable."
            ),
        },
        {
            "use_case":   "Legal contracts (exact clause lookup)",
            "chunker":    "RecursiveText (with clause-level separators)",
            "retriever":  "Keyword (BM25) primarily",
            "why": (
                "Legal queries are almost always keyword-based: 'indemnification clause', "
                "'governing law', 'Section 14.2'. BM25 is more reliable here than semantic "
                "search for exact clause retrieval. Add Semantic as secondary for broader "
                "questions like 'what are my obligations under this contract?'"
            ),
        },
        {
            "use_case":   "Conversational knowledge assistant (vague user questions)",
            "chunker":    "RecursiveText",
            "retriever":  "MultiQuery + Hybrid",
            "why": (
                "End users ask vague, conversational questions like 'tell me about the "
                "new security features'. MultiQuery generates paraphrases to overcome "
                "vocabulary mismatch. Hybrid ensures both the conceptual match (semantic) "
                "and specific feature names (keyword) are covered."
            ),
        },
        {
            "use_case":   "Code documentation / API reference",
            "chunker":    "FixedSize or RecursiveText with code-aware separators",
            "retriever":  "Keyword primarily, Hybrid secondary",
            "why": (
                "Developers search for exact function names, parameter names, error codes. "
                "Keyword retrieval with BM25 is highly effective. Semantic retrieval helps "
                "for 'how do I handle rate limits?' type questions that don't directly name "
                "the relevant API method."
            ),
        },
    ]

    for item in matrix:
        print(f"\n  USE CASE: {item['use_case']}")
        print(f"  Chunker:  {item['chunker']}")
        print(f"  Retriever:{item['retriever']}")
        print(f"  Why:      {item['why']}")

    print("""
  ═══════════════════════════════════════════════════════════════
  QUICK DECISION RULES:
    • Always start with RecursiveText + Hybrid — works for 80% of cases
    • Switch to LayoutAwarePDF if you have PDFs with tables or figures
    • Add HyDE if users ask vague or abstract questions
    • Add MultiQuery if retrieval misses obvious content
    • Use SemanticChunker only when your embedding model is high quality
    • Use Keyword (BM25) when queries are always exact identifiers/terms
  ═══════════════════════════════════════════════════════════════
    """)


# =============================================================================
# SECTION 17 — Main
# =============================================================================

async def main() -> None:
    # ------------------------------------------------------------------
    # Parse demo selection: python 09_agentic_rag.py --demos A,B,C
    # Default: run all demos A–F
    # ------------------------------------------------------------------
    demo_arg  = "A,B,C,D,E,F"
    if "--demos" in sys.argv:
        idx = sys.argv.index("--demos")
        if idx + 1 < len(sys.argv):
            demo_arg = sys.argv[idx + 1]
    selected_demos = {d.strip().upper() for d in demo_arg.split(",")}

    # ------------------------------------------------------------------
    # Step 1: Ingest sample documents
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Agentic RAG — Ingesting sample knowledge base")
    print("=" * 70)

    total = 0
    for source, text in SAMPLE_DOCS:
        count = await rag_store.ingest_text(text.strip(), source=source)
        print(f"  ✓ {source}: {count} chunks (RecursiveTextChunker)")
        total += count

    # Optional: ingest a real PDF
    if "--pdf" in sys.argv:
        idx = sys.argv.index("--pdf")
        if idx + 1 < len(sys.argv):
            pdf_path = sys.argv[idx + 1]
            print(f"\nIngesting PDF: {pdf_path}")
            try:
                pdf_count = await rag_store.ingest_pdf(pdf_path)
                print(f"  ✓ PDF: {pdf_count} chunks (LayoutAwarePDFChunker)")
                total += pdf_count
            except ImportError:
                print("  ✗ pdfplumber not installed. Run: pip install pdfplumber")
            except FileNotFoundError:
                print(f"  ✗ File not found: {pdf_path}")

    stats = rag_store.stats()
    print(f"\nKnowledge base ready: {total} total chunks")
    print(f"Sources:  {', '.join(stats['sources'])}")
    print(f"By type:  {stats['by_type']}")

    # ------------------------------------------------------------------
    # Run selected demos
    # ------------------------------------------------------------------
    if "A" in selected_demos:
        await demo_a_chunking_comparison()

    if "B" in selected_demos:
        await demo_b_retrieval_comparison()

    if "C" in selected_demos:
        await demo_c_simple_rag()

    if "D" in selected_demos:
        await demo_d_agentic_rag_with_self_eval()

    if "E" in selected_demos:
        await demo_e_evaluator_optimizer_rag()

    if "F" in selected_demos:
        demo_f_use_case_matrix()

    # ------------------------------------------------------------------
    # Final comparison summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Architecture Summary")
    print("=" * 70)
    print("""
  CHUNKING DECISION TREE:
    Has PDF with tables/figures?  → LayoutAwarePDFChunker
    Have a production embedder?   → SemanticChunker
    General prose/markdown?       → RecursiveTextChunker  ← default
    Prototyping / structured data → FixedSizeChunker

  RETRIEVAL DECISION TREE:
    Simple exact lookup?          → Keyword (BM25)
    Best general purpose?         → Hybrid (RRF fusion)  ← default
    Vague/abstract questions?     → HyDE
    Noisy/varied query wording?   → MultiQuery
    Pure concept matching?        → Semantic

  AGENTIC LOOP DECISION TREE:
    Single question, small KB?    → SingleAgentLoop (Demo C)
    Multi-part, multi-document?   → PlannerExecutorPattern (Demo D)
    Quality matters most?         → EvaluatorOptimizer on top (Demo E)
    Critical factual accuracy?    → Add self-evaluation (check_sufficiency)

  COMBINING EVERYTHING (production pattern):
    LayoutAwarePDF / RecursiveText  ← ingest with the right chunker
    → HybridRetriever               ← retrieve with RRF fusion
    → PlannerExecutorPattern        ← plan + multi-search
    → check_sufficiency loop        ← verify before synthesising
    → EvaluatorOptimizer            ← verify answer quality
    → SingleAgentLoop for simple Qs ← avoid over-engineering
    """)


if __name__ == "__main__":
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    asyncio.run(main())
