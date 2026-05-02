"""
=============================================================================
Example 05 — Evaluator-Optimizer Pattern
=============================================================================

WHAT THIS EXAMPLE TEACHES
--------------------------
The EvaluatorOptimizerPattern implements a "generate → evaluate → rewrite"
loop. It is the agent equivalent of a human writer doing multiple drafts:

    Round 1:  Generator produces an initial draft.
    Evaluate: Evaluator scores it (0–1) and writes feedback.
    Round 2:  Generator sees the score and feedback, rewrites.
    Evaluate: Score again. If ≥ pass_threshold → done. Otherwise repeat.
    Round N:  Give up and return the last draft if max_rounds is reached.

WHY TWO CLIENTS?
----------------
You can — and often should — use a cheap, fast model as the generator and a
slightly more careful model as the evaluator. The evaluator only reads text
and returns a JSON score; it does not need to be large. Here both clients
use the same model for simplicity.

HOW LLMEvaluator WORKS
-----------------------
LLMEvaluator calls `complete_structured()` on its LLM client. It sends the
task description + the draft to the LLM and asks for:
    {"score": <0-10 integer>, "feedback": "<actionable suggestions>"}

Internally it normalises the score to 0–1 (divides by 10). If the normalised
score >= pass_threshold the draft is accepted.

THE FEEDBACK LOOP
-----------------
If the draft fails, EvaluatorOptimizerPattern appends two messages to the
working history before the next round:
    1. assistant: <the draft that failed>
    2. user:      "Your response scored X%. Feedback: <feedback>"

So on round 2 the generator sees exactly what went wrong. This is why the
pattern often converges in 2 rounds even with a high threshold.

HOW TO READ THE OUTPUT
----------------------
Watch for the logging output (set LOG_LEVEL=DEBUG to see all rounds):

    [evaluator-optimizer] round 1/3  score=0.72  passed=False
    [evaluator-optimizer] round 2/3  score=0.88  passed=True

The script also prints the final accepted draft.

RUNNING
-------
    python examples/05_evaluator_optimizer.py

REQUIREMENTS
------------
    pip install -r requirements.txt && pip install -e .
    export ANTHROPIC_API_KEY=sk-ant-...
=============================================================================
"""

import asyncio
import logging

from mcp_agent_framework import AnthropicClient, AgentConfig, Message
from mcp_agent_framework.patterns import EvaluatorOptimizerPattern
from mcp_agent_framework.patterns.evaluation import LLMEvaluator

# Show round-by-round scoring in the terminal.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s — %(message)s",
)


async def main() -> None:
    # ------------------------------------------------------------------
    # Step 1 — Create two LLM clients.
    #
    # Using the same model for both is fine.  In production you might use
    # a stronger model as the evaluator to catch subtle quality issues.
    # ------------------------------------------------------------------
    generator_client = AnthropicClient("claude-haiku-4-5-20251001")
    evaluator_client = AnthropicClient("claude-haiku-4-5-20251001")

    # ------------------------------------------------------------------
    # Step 2 — Configure the evaluator.
    #
    # pass_threshold=0.8 means the LLM must rate the draft >= 8/10 for it
    # to be accepted.  Lower this (e.g. 0.6) if you want faster convergence
    # in demos; raise it (e.g. 0.9) if you need high quality output.
    # ------------------------------------------------------------------
    evaluator = LLMEvaluator(
        llm_client=evaluator_client,
        pass_threshold=0.8,   # 80 % score required to pass
    )

    # ------------------------------------------------------------------
    # Step 3 — Build an AgentConfig.
    #
    # mcp_server_config={} means NO MCP tools are connected — this is a
    # pure text-generation task.  The generator will never call a tool;
    # it just writes prose.  You can add MCP tools here if the task
    # requires information retrieval, file access, etc.
    # ------------------------------------------------------------------
    config = AgentConfig(
        mcp_server_config={},    # empty = no tools
        system_prompt="You are a technical writer who creates clear, concise explanations.",
        max_iterations=3,
    )

    # ------------------------------------------------------------------
    # Step 4 — Assemble the pattern.
    #
    # max_rounds=3 caps the loop at 3 generate-evaluate cycles regardless
    # of whether the draft ever passes.  The final draft is always returned.
    # ------------------------------------------------------------------
    agent = EvaluatorOptimizerPattern(
        generator_client=generator_client,
        evaluator=evaluator,
        config=config,
        max_rounds=3,
    )

    # ------------------------------------------------------------------
    # Step 5 — Run it and print the result.
    # ------------------------------------------------------------------
    task = "Explain what a Python decorator is in exactly 3 sentences, suitable for a beginner."
    print(f"Task: {task}")
    print("=" * 60)
    print()
    print("Watch the logs above for round-by-round scores.")
    print("The INFO lines show:  round X/3  score=Y  passed=True/False")
    print()

    result = await agent.run(task)

    print(f"\nFinal result:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
