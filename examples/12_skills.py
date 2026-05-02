"""
=============================================================================
Example 12 — Agentic Skills
=============================================================================

WHAT THIS EXAMPLE SHOWS
-----------------------
Skills are named, reusable agentic capabilities that sit between raw tools
and full coordination patterns:

  Tools     — single functions  (search, write_file, send_email)
  Skills    — named capabilities that compose tools + patterns together
  Patterns  — coordination strategies (ReAct, planner-executor, etc.)

Think of a Skill as "a verb your agent system knows how to perform".
You define it once, register it, and any agent can discover and call it.

THE DEMO
--------
We define three skills and register them in a SkillRegistry:

  1. research_topic    — runs SingleAgentLoop on a knowledge base
  2. summarise_text    — condenses any text using the LLM
  3. compare_topics    — invokes research_topic twice in parallel, then compares

Then a SkillAwareAgent is given the registry. It sees two MCP tools:
  - list_skills()                 → discovers all registered skills
  - invoke_skill(name, json)      → runs a skill by name

The agent decides which skills to call to answer the user's question.

RUNNING
-------
    pip install -r requirements.txt && pip install -e ..
    export ANTHROPIC_API_KEY=sk-ant-...
    python 12_skills.py
=============================================================================
"""

from __future__ import annotations

import asyncio
import os

import anthropic
from fastmcp import FastMCP

from mcp_agent_framework import AgentConfig, AnthropicClient
from mcp_agent_framework.memory import SemanticMemory
from mcp_agent_framework.patterns import SingleAgentLoop
from mcp_agent_framework.skills import Skill, SkillAwareAgent, SkillRegistry

# =============================================================================
# Knowledge base (shared across skills)
# =============================================================================

knowledge = SemanticMemory()
kb_app    = FastMCP("knowledge_base")

ARTICLES = {
    "vector_search": (
        "Vector search finds semantically similar content using dense embeddings. "
        "HNSW (Hierarchical Navigable Small World) is the dominant ANN algorithm. "
        "Cosine similarity is the standard metric for text embeddings."
    ),
    "bm25": (
        "BM25 is a probabilistic keyword ranking function. It scores documents by "
        "term frequency (TF) and inverse document frequency (IDF), with length "
        "normalisation. It is the gold standard for sparse retrieval."
    ),
    "hybrid_search": (
        "Hybrid search combines dense (vector) and sparse (BM25) retrieval. "
        "Reciprocal Rank Fusion (RRF) is a simple, effective way to merge ranked lists. "
        "Hybrid consistently outperforms either method alone on recall benchmarks."
    ),
    "chunking": (
        "Chunking splits documents before embedding. Recursive text chunking respects "
        "paragraph and sentence boundaries — best general-purpose choice. "
        "Chunk size 400-512 characters with 10-15% overlap is a reliable default."
    ),
}


@kb_app.tool
async def search_knowledge(query: str) -> str:
    """Search the knowledge base for relevant information."""
    results = await knowledge.search(query, top_k=2)
    if not results:
        return "No relevant results found."
    return "\n\n".join(
        f"[{r.metadata.get('source', '?')}] {r.content}" for r in results
    )


@kb_app.tool
async def list_topics() -> str:
    """List all available topics in the knowledge base."""
    return f"Available topics: {', '.join(ARTICLES)}"


# =============================================================================
# Skill 1 — research_topic
# Uses SingleAgentLoop to research a topic using the knowledge base.
# =============================================================================

async def _research_topic(inputs: dict) -> str:
    topic = inputs.get("topic", "")
    if not topic:
        return "Error: 'topic' input is required."

    config = AgentConfig(
        mcp_server_config=kb_app,
        system_prompt=(
            "You are a research assistant with access to a knowledge base. "
            "Use list_topics() to see what's available, then search_knowledge() "
            "for relevant content. Synthesise a clear, accurate answer."
        ),
        max_iterations=6,
    )
    return await SingleAgentLoop(
        llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
        config=config,
    ).run(f"Research and explain: {topic}")


research_skill = Skill(
    name="research_topic",
    description=(
        "Research any topic related to information retrieval, vector search, "
        "chunking, or hybrid search. Returns a detailed explanation."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The topic to research (e.g. 'vector search', 'BM25')",
            }
        },
        "required": ["topic"],
    },
    handler=_research_topic,
    tags=["research", "read-only"],
)


# =============================================================================
# Skill 2 — summarise_text
# Condenses any text to a target length. No tools needed — pure LLM.
# =============================================================================

async def _summarise_text(inputs: dict) -> str:
    text       = inputs.get("text", "")
    max_words  = inputs.get("max_words", 100)
    if not text:
        return "Error: 'text' input is required."

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp   = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system="You are a concise technical writer.",
        messages=[{
            "role":    "user",
            "content": f"Summarise the following text in at most {max_words} words:\n\n{text}",
        }],
    )
    return resp.content[0].text


summarise_skill = Skill(
    name="summarise_text",
    description=(
        "Condense any text to a shorter summary. Useful after research to create "
        "a concise final answer. Pass max_words to control length (default 100)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "text":      {"type": "string", "description": "The text to summarise"},
            "max_words": {"type": "integer", "description": "Maximum words (default 100)", "default": 100},
        },
        "required": ["text"],
    },
    handler=_summarise_text,
    tags=["writing", "read-only"],
)


# =============================================================================
# Skill 3 — compare_topics
# Researches two topics in parallel and compares them.
# Shows skill composition: one skill invoking two others concurrently.
# =============================================================================

async def _compare_topics(inputs: dict) -> str:
    topic_a = inputs.get("topic_a", "")
    topic_b = inputs.get("topic_b", "")
    if not topic_a or not topic_b:
        return "Error: both 'topic_a' and 'topic_b' inputs are required."

    # Research both topics in parallel — this is skill composition
    result_a, result_b = await asyncio.gather(
        _research_topic({"topic": topic_a}),
        _research_topic({"topic": topic_b}),
    )

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp   = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        system="You are a technical writer. Write a clear, structured comparison.",
        messages=[{
            "role":    "user",
            "content": (
                f"Compare {topic_a!r} and {topic_b!r}.\n\n"
                f"Research on {topic_a}:\n{result_a}\n\n"
                f"Research on {topic_b}:\n{result_b}\n\n"
                "Write a comparison covering: what each is, key differences, when to use each."
            ),
        }],
    )
    return resp.content[0].text


compare_skill = Skill(
    name="compare_topics",
    description=(
        "Research two topics in parallel and produce a structured comparison. "
        "Covers what each is, key differences, and when to use each."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "topic_a": {"type": "string", "description": "First topic"},
            "topic_b": {"type": "string", "description": "Second topic"},
        },
        "required": ["topic_a", "topic_b"],
    },
    handler=_compare_topics,
    tags=["research", "comparison", "read-only"],
)


# =============================================================================
# Demo
# =============================================================================

async def main() -> None:
    # Ingest knowledge base
    print("Loading knowledge base...")
    for source, text in ARTICLES.items():
        await knowledge.add(text, metadata={"source": source})

    # Build the registry
    registry = SkillRegistry()
    registry.register(research_skill)
    registry.register(summarise_skill)
    registry.register(compare_skill)

    print(f"\nRegistry contains {len(registry)} skills:")
    for skill in registry.list_skills():
        tags = f"  [{', '.join(skill.tags)}]" if skill.tags else ""
        print(f"  - {skill.name}{tags}: {skill.description[:60]}...")

    # ── Demo 1: direct registry use (no LLM needed) ──────────────────────────
    print("\n" + "=" * 60)
    print("Demo 1 — Direct skill invocation (no LLM)")
    print("=" * 60)
    result = await registry.invoke("research_topic", {"topic": "hybrid search"})
    print(f"\nresearch_topic('hybrid search'):\n{result[:300]}...")

    # ── Demo 2: parallel invocation via invoke_many ───────────────────────────
    print("\n" + "=" * 60)
    print("Demo 2 — Parallel skill invocation (invoke_many)")
    print("=" * 60)
    results = await registry.invoke_many([
        ("research_topic", {"topic": "vector search"}),
        ("research_topic", {"topic": "BM25"}),
    ])
    print(f"\nTwo topics researched in parallel:")
    for i, r in enumerate(results):
        print(f"  [{i}] {r[:120]}...")

    # ── Demo 3: SkillAwareAgent — LLM decides which skills to call ────────────
    print("\n" + "=" * 60)
    print("Demo 3 — SkillAwareAgent (LLM chooses and calls skills)")
    print("=" * 60)
    agent = SkillAwareAgent(
        llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
        registry=registry,
        system_prompt=(
            "You are a search infrastructure expert. "
            "Use list_skills() to discover capabilities, then invoke the right skills "
            "to answer the user's question thoroughly."
        ),
        max_iterations=10,
    )

    question = "How does vector search compare to BM25, and when should I use hybrid search?"
    print(f"\nQuestion: {question}\n")
    answer = await agent.run(question)
    print(f"Answer:\n{answer}")

    # ── Demo 4: tag filtering ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Demo 4 — Tag filtering")
    print("=" * 60)
    research_skills = registry.list_skills(tag="research")
    writing_skills  = registry.list_skills(tag="writing")
    print(f"\nSkills tagged 'research': {[s.name for s in research_skills]}")
    print(f"Skills tagged 'writing':  {[s.name for s in writing_skills]}")


if __name__ == "__main__":
    asyncio.run(main())
