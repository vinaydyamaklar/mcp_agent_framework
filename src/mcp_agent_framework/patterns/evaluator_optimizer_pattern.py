"""
Pattern: Evaluator-Optimizer

Generate → Evaluate → Rewrite loop.

The generator produces content using a full SingleAgentLoop (so it has
tool access). The evaluator scores it and gives feedback. If the score
is below threshold, the feedback is fed back to the generator as context
for a rewrite. Repeats until the content passes or max_rounds is reached.

                ┌──────────────────────────────────────────┐
                │              EvaluatorOptimizer           │
                │                                          │
  user_message ─►  Round 1: SingleAgentLoop → draft        │
                │      ↓                                   │
                │  Evaluator: score + feedback              │
                │      ↓ (if failed)                       │
                │  Round 2: "Your draft scored X. Feedback: │
                │  {feedback}. Please revise." → new draft  │
                │      ↓                                   │
                │  Evaluator: score + feedback              │
                │      ↓ (if passed or max_rounds reached) │
                │  Return final draft                       │
                └──────────────────────────────────────────┘

When to use:
- Writing tasks where quality matters (blog posts, reports, code)
- Data extraction that needs verification
- Any task where "good enough" has a measurable definition
"""
from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.observability import RunContext
from mcp_agent_framework.observability.tracer import TraceEventType
from mcp_agent_framework.patterns.evaluation import AbstractEvaluator
from mcp_agent_framework.patterns.single_agent_loop import SingleAgentLoop
from mcp_agent_framework.types import AgentConfig, Message

logger = logging.getLogger(__name__)


class EvaluatorOptimizerPattern:
    """
    Iterative generate-evaluate-rewrite pattern.

    A ``SingleAgentLoop`` (the *generator*) produces a draft. An
    ``AbstractEvaluator`` scores it. If the draft fails the threshold and
    there are rounds remaining, structured feedback is appended to the
    conversation and the generator is asked to revise. The loop continues
    until the draft passes or ``max_rounds`` is exhausted.

    Design decisions
    ----------------
    * ``SingleAgentLoop`` is instantiated fresh on every round so that each
      round gets a clean MCP connection (connect → work → disconnect).
    * ``working_history`` grows across rounds so the generator always has the
      full revision history as context — it can see what it tried and why it
      was rejected.
    * Trace events (``PATTERN_START`` / ``PATTERN_END``) are emitted when a
      ``RunContext`` is provided, giving observability into the outer loop
      without coupling to a specific tracer backend.

    Parameters
    ----------
    generator_client:
        The LLM client used by the internal ``SingleAgentLoop`` to produce
        drafts. Must implement ``BaseLLMClient``.
    evaluator:
        Any ``AbstractEvaluator`` implementation. It receives each draft plus
        the original user message and returns a scored ``EvaluationResult``.
    config:
        ``AgentConfig`` forwarded to the ``SingleAgentLoop`` each round
        (MCP server config, system prompt, iteration cap, etc.).
    max_rounds:
        Maximum number of generate-evaluate cycles before returning the last
        draft regardless of quality. Defaults to 3.
    context:
        Optional ``RunContext`` for emitting ``PATTERN_START`` / ``PATTERN_END``
        trace events. Pass ``None`` (the default) to skip tracing entirely.

    Example
    -------
    ::

        from mcp_agent_framework.clients.anthropic_client import AnthropicClient
        from mcp_agent_framework.patterns.evaluation import LLMEvaluator
        from mcp_agent_framework.patterns.evaluator_optimizer_pattern import (
            EvaluatorOptimizerPattern,
        )
        from mcp_agent_framework.types import AgentConfig

        config = AgentConfig(
            mcp_server_config={"mcpServers": {"fs": {"url": "http://localhost:8001/mcp"}}},
            system_prompt="You are a senior technical writer.",
        )
        evaluator = LLMEvaluator(
            client=AnthropicClient(),
            criteria="Clear, concise, accurate, and well-structured.",
            threshold=0.80,
        )
        pattern = EvaluatorOptimizerPattern(
            generator_client=AnthropicClient(),
            evaluator=evaluator,
            config=config,
            max_rounds=3,
        )
        final_draft = await pattern.run("Write a blog post about async Python.")
        print(final_draft)
    """

    def __init__(
        self,
        generator_client: BaseLLMClient,
        evaluator: AbstractEvaluator,
        config: AgentConfig,
        max_rounds: int = 3,
        context: RunContext | None = None,
    ) -> None:
        self._generator  = generator_client
        self._evaluator  = evaluator
        self._config     = config
        self._max_rounds = max_rounds
        self._context    = context

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        user_message: str,
        history: list[Message] | None = None,
    ) -> str:
        """
        Run the generate-evaluate-rewrite loop and return the best draft.

        The method always returns a string — even if no round passes the
        threshold, it returns whatever the generator produced on the final
        round.

        Parameters
        ----------
        user_message:
            The user's request that seeds generation on round 0 and is also
            passed to the evaluator as the ``task`` argument on every round.
        history:
            Optional prior conversation context. On round 0 the generator
            receives this as-is. Revision messages (draft + feedback) are
            appended to a working copy across subsequent rounds.

        Returns
        -------
        str
            The generator's output from the last round that either passed
            evaluation or exhausted ``max_rounds``.
        """
        start_ts = time.monotonic()

        if self._context:
            await self._context.emit(
                TraceEventType.PATTERN_START,
                {
                    "pattern_name": "EvaluatorOptimizerPattern",
                    "user_message": user_message,
                    "max_rounds": self._max_rounds,
                },
            )

        # working_history accumulates the full revision trail so the generator
        # always sees its prior attempts and the feedback it received.
        working_history: list[Message] = list(history or [])
        # current_prompt starts as the original request; after each failed
        # round it becomes the revision instruction (score + feedback).
        # The original task is preserved separately for the evaluator so
        # evaluation context never drifts across rounds.
        original_task  = user_message
        current_prompt = user_message
        draft = ""

        for round_idx in range(self._max_rounds):
            logger.debug(
                "[evaluator-optimizer] starting round %d/%d",
                round_idx + 1,
                self._max_rounds,
            )

            # -----------------------------------------------------------------
            # Generation step — fresh SingleAgentLoop per round so each round
            # gets a clean MCP connection lifecycle.
            #
            # IMPORTANT: working_history must end with an *assistant* turn
            # before this call so that SingleAgentLoop.run() can safely append
            # the *user* turn (current_prompt) without creating two consecutive
            # user messages (which Anthropic's API rejects with HTTP 400).
            # On round 0 the history is either empty or caller-supplied and
            # ends correctly.  On round 1+ we append the assistant draft below
            # before updating current_prompt, guaranteeing this invariant.
            # -----------------------------------------------------------------
            draft = await SingleAgentLoop(
                llm_client=self._generator,
                config=self._config,
            ).run(current_prompt, history=working_history)

            # -----------------------------------------------------------------
            # Evaluation step — always uses original_task so the evaluator
            # judges quality against the real goal, not the revision prompt.
            # -----------------------------------------------------------------
            result = await self._evaluator.evaluate(
                content=draft,
                task=original_task,
                iteration=round_idx,
            )

            logger.info(
                "[evaluator-optimizer] round %d/%d  score=%.2f  passed=%s",
                round_idx + 1,
                self._max_rounds,
                result.score,
                result.passed,
            )

            # -----------------------------------------------------------------
            # Termination check — pass OR last round
            # -----------------------------------------------------------------
            if result.passed or round_idx == self._max_rounds - 1:
                if self._context:
                    elapsed_ms = (time.monotonic() - start_ts) * 1_000
                    await self._context.emit(
                        TraceEventType.PATTERN_END,
                        {
                            "pattern_name": "EvaluatorOptimizerPattern",
                            "rounds_completed": round_idx + 1,
                            "final_score": result.score,
                            "passed": result.passed,
                            "elapsed_ms": elapsed_ms,
                        },
                    )
                return draft

            # -----------------------------------------------------------------
            # Prepare next round: append draft as assistant turn so history
            # ends with "assistant", then set the revision instruction as the
            # next user_message.  This avoids consecutive user messages.
            # -----------------------------------------------------------------
            working_history.append(
                Message(role="assistant", content=draft)
            )
            current_prompt = (
                f"Your response scored {result.score:.1%}. "
                f"Please improve it.\n\nFeedback: {result.feedback}"
            )

        # Unreachable — loop always returns inside the for body — but satisfies
        # type checkers that expect an explicit return on every path.
        return draft  # pragma: no cover

    async def run_stream(
        self,
        user_message: str,
        history: list[Message] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Run all evaluate-optimize rounds synchronously; stream the final round's generation.
        Intermediate rounds (where drafts are rejected) run non-streaming for simplicity.
        The last round always streams, giving the caller live tokens for the final output.
        """
        from mcp_agent_framework.types import StreamEvent

        working_history: list[Message] = list(history or [])
        original_task = user_message
        current_prompt = user_message

        for round_idx in range(self._max_rounds):
            is_last = (round_idx == self._max_rounds - 1)

            if is_last:
                # Stream the final round
                gen = SingleAgentLoop(llm_client=self._generator, config=self._config)
                async for event in gen.run_stream(current_prompt, history=working_history):
                    yield event
                return

            # Non-streaming intermediate rounds
            draft = await SingleAgentLoop(
                llm_client=self._generator, config=self._config
            ).run(current_prompt, history=working_history)

            result = await self._evaluator.evaluate(
                content=draft, task=original_task, iteration=round_idx
            )

            if result.passed:
                # Passed early — yield the draft as a single text event
                yield StreamEvent(type="text", delta=draft)
                return

            working_history.append(Message(role="assistant", content=draft))
            current_prompt = (
                f"Your response scored {result.score:.1%}. "
                f"Please improve it.\n\nFeedback: {result.feedback}"
            )
