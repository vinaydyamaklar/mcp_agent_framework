"""
memory — Pluggable memory subsystem for the MCP agent framework.

Three concrete store types are provided, each suited to a different kind of
agent memory:

``SemanticMemory``
    Stores arbitrary text and retrieves it by cosine similarity to a query
    vector.  Pass a real async embedding function for production use; the
    built-in bag-of-words fallback is good enough for demos and tests.

``EpisodicMemory``
    An ordered event log.  Retrieval combines keyword overlap with a recency
    bonus so the most recent observations surface naturally.  Use for
    conversation history, action logs, and tool call records.

``ProceduralMemory``
    A task-to-procedure lookup store.  Given a free-text task description it
    returns the best-matching stored procedure (SOP, workflow, recipe) using
    Jaccard similarity on word sets.

All three implement ``AbstractMemoryStore``, so agent code can depend on that
interface and swap implementations freely.

Quick start::

    from mcp_agent_framework.memory import SemanticMemory, EpisodicMemory, ProceduralMemory

    sem = SemanticMemory()
    epi = EpisodicMemory(max_entries=500)
    pro = ProceduralMemory()

    await sem.add("Paris is the capital of France")
    await epi.add("User asked: what is the capital of France?")
    await pro.add(
        "Answer geography questions",
        metadata={"steps": ["look up fact", "cite source", "reply concisely"]},
    )
"""

from .base import AbstractMemoryStore, MemoryEntry
from .episodic import EpisodicMemory
from .procedural import ProceduralMemory
from .semantic import SemanticMemory

__all__ = [
    "AbstractMemoryStore",
    "MemoryEntry",
    "SemanticMemory",
    "EpisodicMemory",
    "ProceduralMemory",
]
