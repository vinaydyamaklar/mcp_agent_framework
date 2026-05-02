"""
=============================================================================
Example 06 — Planner-Executor Pattern
=============================================================================

WHAT THIS EXAMPLE TEACHES
--------------------------
PlannerExecutorPattern separates *thinking about what to do* from *doing it*.
This two-phase design is inspired by classical AI planning systems and is
extremely useful for multi-step tasks:

    PLAN:     A strong model reads the task and outputs a structured
              ExecutionPlan (a list of typed ExecutionStep objects).
    EXECUTE:  A fast/cheap model works through each step one at a time,
              calling MCP tools as needed. It sees the results from prior
              steps so later steps can build on earlier ones.
    REPLAN:   If a step fails and max_replan_attempts > 0, the planner is
              asked to revise the *remaining* steps given the error context.
    SYNTHESISE: The planner merges all step results into one final answer.

WHY TWO MODELS?
---------------
Planning requires strong reasoning (understanding goals, decomposing them
into logical sequences). Execution is narrower — each step is a focused
sub-task, so a smaller, faster model works well and is cheaper.

This script uses:
    planner  = claude-sonnet-4-6    (strong reasoning + structured output)
    executor = claude-haiku-4-5     (fast + cheap per step)

You can pass the same model for both if you prefer simplicity.

THE KEY LEARNING MOMENT
-----------------------
The script explicitly PRINTS the generated plan before execution starts.
This reveals the two-phase design: you will see something like:

    === PLAN (3 steps) ===
    Goal: Research pros and cons of Python vs Go for web APIs
    Step 1: Search for Python web API strengths
    Step 2: Search for Go web API strengths
    Step 3: Compare and write recommendation

Then the executor works through each step, calling search_topic() and
summarise_findings() tools. Finally the planner synthesises the results.

MCP TOOLS IN THIS EXAMPLE
--------------------------
Two stub tools are defined using FastMCP:
    search_topic(topic)           — returns fake research data
    summarise_findings(findings)  — returns a stub summary

In a real application these would call a web search API, a RAG system,
a database, etc.  The pattern does not change — swap the stubs for real
implementations when ready.

HOW TO READ THE OUTPUT
----------------------
You will see three clear phases in the terminal:

    [planner-executor] plan: N steps for goal: ...
    [planner-executor] executing step 1 ...
    [planner-executor] executing step 2 ...
    [planner-executor] synthesising N step results

RUNNING
-------
    python examples/06_planner_executor.py

REQUIREMENTS
------------
    pip install -r requirements.txt && pip install -e .
    export ANTHROPIC_API_KEY=sk-ant-...
=============================================================================
"""

import asyncio
import logging

from fastmcp import FastMCP

from mcp_agent_framework import AnthropicClient, AgentConfig
from mcp_agent_framework.patterns import PlannerExecutorPattern

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s — %(message)s",
)

# ------------------------------------------------------------------
# Stub MCP tools
#
# FastMCP lets you define tools as plain Python functions decorated
# with @app.tool. The framework serialises them to MCP tool schema
# automatically and calls them when the executor agent requests them.
# ------------------------------------------------------------------
app = FastMCP("research_tools")


@app.tool
def search_topic(topic: str) -> str:
    """Search for information about a topic."""
    # In production: call a search API, RAG system, or database.
    return (
        f"Research results for '{topic}': "
        f"[Relevant facts about {topic} would appear here in a real implementation]"
    )


@app.tool
def summarise_findings(findings: str) -> str:
    """Summarise a set of research findings."""
    return f"Summary: Key points extracted from findings: {findings[:100]}..."


async def main() -> None:
    # ------------------------------------------------------------------
    # Step 1 — Two clients with different model sizes.
    # ------------------------------------------------------------------
    planner  = AnthropicClient("claude-sonnet-4-6")          # powerful for planning
    executor = AnthropicClient("claude-haiku-4-5-20251001")  # cheap for execution

    # ------------------------------------------------------------------
    # Step 2 — AgentConfig connects executor steps to the MCP tools.
    #
    # Passing a FastMCP app object directly is the in-process mode — no
    # network round-trip, tools run in the same Python process.
    # ------------------------------------------------------------------
    config = AgentConfig(
        mcp_server_config=app,
        system_prompt="You are a research assistant that plans and executes research tasks.",
        max_iterations=5,
    )

    # ------------------------------------------------------------------
    # Step 3 — Assemble the pattern.
    #
    # max_replan_attempts=1 means if a step fails, the planner gets ONE
    # chance to revise the remaining steps before the error is recorded
    # and execution continues.
    # ------------------------------------------------------------------
    agent = PlannerExecutorPattern(
        planner_client=planner,
        executor_client=executor,
        config=config,
        max_replan_attempts=1,
    )

    task = "Research the pros and cons of Python vs Go for building web APIs, then write a recommendation."
    print(f"Task: {task}")
    print("=" * 60)
    print()
    print("[The agent will first generate a plan, then execute each step.]")
    print()

    # ------------------------------------------------------------------
    # Step 4 — Run.
    #
    # The INFO log lines emitted by the framework let you watch the plan
    # being generated and each step being executed in real time.
    # ------------------------------------------------------------------
    result = await agent.run(task)

    print(f"\nFinal Answer:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
