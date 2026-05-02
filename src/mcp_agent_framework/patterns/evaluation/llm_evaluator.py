"""
LLMEvaluator — scores generated content using complete_structured().

The LLM is asked to rate the response 0-10 and provide actionable feedback.
That feedback is fed back to the generator on the next iteration.
"""
from __future__ import annotations

import logging
from typing import Any

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.types import Message
from mcp_agent_framework.patterns.evaluation.base_evaluator import AbstractEvaluator, EvaluationResult

logger = logging.getLogger(__name__)

_DEFAULT_EVAL_PROMPT = """You are a quality evaluator. Given a task and a response to that task,
evaluate the quality of the response on a scale of 0 to 10.

Be specific and actionable in your feedback — the generator will use your feedback to rewrite.

Task: {task}

Response to evaluate:
{content}

Return a JSON object with:
- score: integer 0-10 (10 = perfect)
- feedback: specific, actionable improvement suggestions (empty string if score is 10)
"""

_EVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score":    {"type": "number"},
        "feedback": {"type": "string"},
    },
    "required": ["score", "feedback"],
}


class LLMEvaluator(AbstractEvaluator):
    """
    Evaluate generated content by asking the LLM to score it.

    Uses complete_structured() so the score and feedback are always
    reliably parseable, regardless of provider.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        pass_threshold: float = 0.7,
        evaluation_prompt: str | None = None,
    ):
        """
        Args:
            llm_client:         Any BaseLLMClient implementation.
            pass_threshold:     score/10 >= this value → EvaluationResult.passed = True.
            evaluation_prompt:  Override the default prompt template.
                                Must contain {task} and {content} placeholders.
        """
        self._llm       = llm_client
        self._threshold = pass_threshold
        self._prompt    = evaluation_prompt or _DEFAULT_EVAL_PROMPT

    async def evaluate(self, content: str, task: str, iteration: int = 0) -> EvaluationResult:
        """
        Score *content* against *task* using the LLM.

        Args:
            content:   Generated text to evaluate.
            task:      Original user request.
            iteration: Rewrite round (informational; not used by this evaluator).

        Returns:
            EvaluationResult with score normalised to 0-1.
        """
        prompt_text = self._prompt.format(task=task, content=content)
        messages = [Message(role="user", content=prompt_text)]

        try:
            result = await self._llm.complete_structured(messages, _EVAL_SCHEMA)
            raw_score: float = float(result["score"])
            score_normalised = max(0.0, min(1.0, raw_score / 10.0))
            feedback: str = result.get("feedback", "")
            logger.debug(
                "[LLMEvaluator] iteration=%d raw_score=%.1f normalised=%.3f",
                iteration, raw_score, score_normalised,
            )
            return EvaluationResult(
                score=score_normalised,
                passed=score_normalised >= self._threshold,
                feedback=feedback,
                details={"raw_score": raw_score, "iteration": iteration},
            )
        except Exception as err:
            logger.warning("[LLMEvaluator] complete_structured failed: %s", err)
            return EvaluationResult(
                score=0.5,
                passed=False,
                feedback=f"Evaluation failed: {err}",
                details={"iteration": iteration, "error": str(err)},
            )
