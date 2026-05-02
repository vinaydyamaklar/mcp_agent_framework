"""
=============================================================================
Example 07 — Parallel Pattern (fan-out to multiple agents)
=============================================================================

WHAT THIS EXAMPLE TEACHES
--------------------------
ParallelPattern runs N independent agent tasks concurrently using
asyncio.gather, then feeds all results to a synthesiser LLM that merges
them into one coherent final answer.

The wall-clock time is roughly the time of the SLOWEST single task, not the
sum of all tasks. This is the key win: 3 tasks that each take 4 seconds run
in ~4 s total instead of ~12 s.

This script measures elapsed time for you so you can see this directly.

THE FAN-OUT DESIGN
------------------
Each ParallelTask bundles three things:
    name   — a short label ("python_research", "go_research", …)
    prompt — what this specific agent should research
    agent  — its own SingleAgentLoop instance

Because each task has its own agent it can use a DIFFERENT LLM client
and connect to a DIFFERENT MCP server. One task could call a Python docs
API, another a Go docs API, another a general web search. They are
fully independent.

FAULT TOLERANCE
---------------
If one task raises an exception it does NOT cancel the others. The error
is captured in ParallelResult.error and the synthesiser is told "Task X
FAILED: <reason>". This means the pattern always returns a result, even
if some agents fail.

HOW SYNTHESIS WORKS
-------------------
After all tasks complete, a separate synthesiser LLM receives a prompt
containing every task's name, output (or error), and the original user
question. It produces the unified final answer. The synthesiser is given
its own client and config — you might use a larger model here since it
needs to reason across all the gathered evidence.

THE ELAPSED-TIME OUTPUT
-----------------------
At the end of main() you will see:

    [Total elapsed: X.Xs for 3 parallel agents]

In a production system with real LLM calls and tool latency, this number
would be dramatically lower than 3× the per-agent time.

MCP TOOLS
---------
One stub tool `search(query)` simulates a 0.1 s search latency. Each
agent will call it once for its specific research topic.

RUNNING
-------
    python examples/07_parallel_agents.py

REQUIREMENTS
------------
    pip install -r requirements.txt && pip install -e .
    export ANTHROPIC_API_KEY=sk-ant-...
=============================================================================
"""

import asyncio
import logging
import time

from fastmcp import FastMCP

from mcp_agent_framework import AnthropicClient, AgentConfig
from mcp_agent_framework.patterns import SingleAgentLoop, ParallelPattern, ParallelTask

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s — %(message)s",
)

# ------------------------------------------------------------------
# Stub MCP server with a single search tool.
#
# The 0.1 s sleep simulates real network latency so the parallelism
# benefit is observable even in a tiny demo.
# ------------------------------------------------------------------
app = FastMCP("research_tools")


@app.tool
def search(query: str) -> str:
    """Search for information."""
    time.sleep(0.1)  # simulate small latency
    return f"Search results for '{query}': [Results would appear here in production]"


async def main() -> None:
    # ------------------------------------------------------------------
    # Step 1 — Shared config for the research agents.
    #
    # All three research agents share the same MCP server and system
    # prompt. In production you could give each task its own config if
    # they need different tools or different personas.
    # ------------------------------------------------------------------
    config = AgentConfig(
        mcp_server_config=app,
        system_prompt="You are a research agent. Use the search tool to find information.",
        max_iterations=3,
    )

    # ------------------------------------------------------------------
    # Step 2 — Build three independent tasks.
    #
    # Key design: each ParallelTask gets its OWN SingleAgentLoop instance.
    # They all run concurrently via asyncio.gather, so they never block
    # each other.
    #
    # In production you could give each task a completely different LLM:
    #   python_task uses GeminiClient("gemini-2.5-flash")
    #   go_task     uses AnthropicClient("claude-haiku-4-5-20251001")
    #   rust_task   uses OpenAIClient("gpt-4o-mini")
    # ------------------------------------------------------------------
    def make_agent() -> SingleAgentLoop:
        return SingleAgentLoop(
            llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
            config=config,
        )

    tasks = [
        ParallelTask(
            "python_research",
            "Research Python's strengths for data science in 2026",
            make_agent(),
        ),
        ParallelTask(
            "go_research",
            "Research Go's strengths for systems programming in 2026",
            make_agent(),
        ),
        ParallelTask(
            "rust_research",
            "Research Rust's strengths for performance-critical code in 2026",
            make_agent(),
        ),
    ]

    # ------------------------------------------------------------------
    # Step 3 — Synthesiser config.
    #
    # The synthesiser sees all three research outputs and merges them.
    # mcp_server_config={} because the synthesiser only reads text —
    # it does not need to call any tools.
    # ------------------------------------------------------------------
    synthesiser = AnthropicClient("claude-haiku-4-5-20251001")
    synth_config = AgentConfig(
        mcp_server_config={},
        system_prompt="You are a synthesis expert. Combine research findings into clear, balanced recommendations.",
        max_iterations=1,
    )

    # ------------------------------------------------------------------
    # Step 4 — Assemble and run.
    # ------------------------------------------------------------------
    agent = ParallelPattern(
        synthesiser_client=synthesiser,
        tasks=tasks,
        config=synth_config,
    )

    question = "Compare Python, Go, and Rust — which should I learn first as a backend developer?"
    print(f"Question: {question}")
    print(f"Running {len(tasks)} research agents in parallel...")
    print("=" * 60)
    print()

    t0 = time.monotonic()
    result = await agent.run(question)
    elapsed = time.monotonic() - t0

    # This is the key teaching moment: N agents ran in the time of ~1 agent.
    print(f"\n[Total elapsed: {elapsed:.1f}s for {len(tasks)} parallel agents]")
    print(f"\nSynthesised Answer:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
