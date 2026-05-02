"""
RubricEvaluator — weighted criteria, with optional rule-based checkers.

Each criterion is scored 0-10 by the LLM (or by a checker_fn for mechanical
rules). The final score is the weighted average of all criterion scores,
normalised to 0-1.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.types import Message
from mcp_agent_framework.patterns.evaluation.base_evaluator import AbstractEvaluator, EvaluationResult

logger = logging.getLogger(__name__)

_CRITERION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score":    {"type": "number"},
        "feedback": {"type": "string"},
    },
    "required": ["score", "feedback"],
}

_CRITERION_PROMPT = """You are a quality evaluator. Score the following response against ONE specific criterion.

Task: {task}

Response to evaluate:
{content}

Criterion to score: {criterion_name}
Criterion description: {criterion_description}

Rate only this criterion on a scale of 0 to 10.

Return a JSON object with:
- score: integer 0-10 (10 = perfectly meets the criterion)
- feedback: specific, actionable suggestion for this criterion (empty string if score is 10)
"""


@dataclass
class RubricCriterion:
    """A single named criterion used by RubricEvaluator."""
    name: str
    description: str
    weight: float = 1.0


class RubricEvaluator(AbstractEvaluator):
    """
    Evaluate content against explicit, human-defined criteria.

    Each criterion is scored 0-10 by the LLM (or by a checker_fn for mechanical
    rules). Final score is the weighted average of all criterion scores,
    normalised to 0-1.

    checker_fns: dict mapping criterion name → sync bool function.
    When provided, replaces the LLM call for that criterion with a direct bool
    check (True = full score of 1.0, False = 0.0). Good for: word limits,
    required phrases, format constraints, etc.

    Example::

        evaluator = RubricEvaluator(
            llm_client=client,
            criteria=[
                RubricCriterion("clarity",  "Is the writing clear and easy to understand?", weight=2.0),
                RubricCriterion("accuracy", "Are the facts correct?",                        weight=3.0),
                RubricCriterion("length",   "Is it under 200 words?",                        weight=1.0),
            ],
            checker_fns={"length": lambda t: len(t.split()) <= 200},
        )
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        criteria: list[RubricCriterion],
        pass_threshold: float = 0.7,
        checker_fns: dict[str, Callable[[str], bool]] | None = None,
    ):
        """
        Args:
            llm_client:      Any BaseLLMClient implementation.
            criteria:        Ordered list of RubricCriterion objects.
            pass_threshold:  Weighted-average score >= this → passed=True.
            checker_fns:     Maps criterion name to a sync callable that returns
                             True (pass) or False (fail) without calling the LLM.
        """
        if not criteria:
            raise ValueError("RubricEvaluator requires at least one criterion.")
        self._llm          = llm_client
        self._criteria     = criteria
        self._threshold    = pass_threshold
        self._checker_fns  = checker_fns or {}

    async def evaluate(self, content: str, task: str, iteration: int = 0) -> EvaluationResult:
        """
        Score *content* against every criterion, then aggregate.

        Args:
            content:   Generated text to evaluate.
            task:      Original user request.
            iteration: Rewrite round (informational only).

        Returns:
            EvaluationResult whose score is the weighted average across criteria.
            details contains per-criterion scores.
            feedback concatenates LLM feedback for every failing criterion.
        """
        async def _score_one(criterion: RubricCriterion) -> tuple[str, float, str]:
            """Score a single criterion; returns (name, score 0–1, feedback)."""
            checker = self._checker_fns.get(criterion.name)
            if checker is not None:
                passed = checker(content)
                score = 1.0 if passed else 0.0
                fb = "" if passed else f"'{criterion.name}' check failed."
                logger.debug(
                    "[RubricEvaluator] criterion='%s' checker=%s score=%.1f",
                    criterion.name, passed, score,
                )
                return criterion.name, score, fb

            prompt_text = _CRITERION_PROMPT.format(
                task=task,
                content=content,
                criterion_name=criterion.name,
                criterion_description=criterion.description,
            )
            messages = [Message(role="user", content=prompt_text)]
            try:
                result = await self._llm.complete_structured(messages, _CRITERION_SCHEMA)
                raw    = float(result["score"])
                score  = max(0.0, min(1.0, raw / 10.0))
                fb     = result.get("feedback", "")
                logger.debug(
                    "[RubricEvaluator] criterion='%s' raw=%.1f norm=%.3f",
                    criterion.name, raw, score,
                )
                return criterion.name, score, fb
            except Exception as err:
                logger.warning(
                    "[RubricEvaluator] LLM scoring failed for criterion '%s': %s",
                    criterion.name, err,
                )
                return criterion.name, 0.5, f"Evaluation of '{criterion.name}' failed: {err}"

        # Score all criteria concurrently — each LLM call is independent.
        scored = await asyncio.gather(*[_score_one(c) for c in self._criteria])
        criterion_scores:   dict[str, float] = {name: s   for name, s, _  in scored}
        criterion_feedback: dict[str, str]   = {name: fb  for name, _, fb in scored}

        # Weighted average
        total_weight = sum(c.weight for c in self._criteria)
        weighted_sum = sum(
            criterion_scores[c.name] * c.weight for c in self._criteria
        )
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Collect feedback only for criteria that failed (score < 1.0 after norm)
        failed_feedback = [
            f"[{c.name}] {criterion_feedback[c.name]}"
            for c in self._criteria
            if criterion_feedback.get(c.name) and criterion_scores[c.name] < 1.0
        ]
        combined_feedback = "\n".join(failed_feedback)

        logger.debug(
            "[RubricEvaluator] iteration=%d final_score=%.3f passed=%s",
            iteration, final_score, final_score >= self._threshold,
        )

        return EvaluationResult(
            score=final_score,
            passed=final_score >= self._threshold,
            feedback=combined_feedback,
            details={
                "criterion_scores": criterion_scores,
                "iteration": iteration,
            },
        )
