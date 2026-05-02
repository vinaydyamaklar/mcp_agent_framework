"""
Abstract base class and result type for all evaluators.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """The outcome of a single evaluation pass."""
    score: float                          # 0.0–1.0
    passed: bool                          # score >= threshold
    feedback: str                         # human-readable explanation fed back to generator
    details: dict[str, Any] = field(default_factory=dict)


class AbstractEvaluator(ABC):
    """
    Common interface every evaluator must implement.

    Evaluators are used by generator-evaluator patterns to score a piece of
    generated content and decide whether it needs to be rewritten.
    """

    @abstractmethod
    async def evaluate(
        self,
        content: str,
        task: str,          # the original user request / task
        iteration: int = 0, # which rewrite round (0 = first attempt)
    ) -> EvaluationResult:
        """
        Score *content* against *task*.

        Args:
            content:   The generated text to evaluate.
            task:      The original user request that triggered generation.
            iteration: Which rewrite round this is (0 = first attempt).
                       Evaluators may use this to adjust strictness over time.

        Returns:
            EvaluationResult with score, passed flag, and actionable feedback.
        """
        ...
