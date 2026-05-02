"""
Retry policy with exponential backoff and jitter for transient failure recovery.

Usage example::

    policy = RetryPolicy(max_retries=3, base_delay=0.5)
    result = await policy.execute(lambda: some_async_call())
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class RetryExhaustedError(RuntimeError):
    """Raised when all retry attempts have been exhausted without success.

    Attributes:
        attempts: Total number of attempts made (including the first try).
        last_exception: The exception raised on the final attempt.
    """

    def __init__(self, attempts: int, last_exception: Exception) -> None:
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"All {attempts} attempt(s) exhausted. "
            f"Last error: {type(last_exception).__name__}: {last_exception}"
        )


@dataclass
class RetryPolicy:
    """Configurable retry policy with exponential backoff and optional jitter.

    Exponential backoff spaces out retries so a misbehaving downstream service
    gets increasing breathing room on each attempt.  Jitter (randomised delay
    offset) is enabled by default to avoid the *thundering herd* problem: when
    many callers back off by the *exact* same amount they all hammer the
    recovering service again at the same instant.  Adding a small random spread
    staggers those retries in time.

    Why a zero-argument callable instead of a coroutine?
    -------------------------------------------------------
    A coroutine object can only be awaited once — after it completes (or
    raises) it is exhausted and re-awaiting it raises a ``RuntimeError``.  By
    accepting a *factory* (``coro_fn: Callable[[], Awaitable[T]]``) we create a
    fresh coroutine on every attempt, which is the correct pattern for retries.

    Args:
        max_retries: Maximum number of additional attempts after the first
            failure.  Total attempts = ``max_retries + 1``.
        base_delay: Initial sleep duration in seconds before the first retry.
        max_delay: Upper bound on the computed delay, preventing runaway waits.
        exponential_base: Multiplier applied per attempt
            (``delay = base_delay * exponential_base ** attempt``).
        jitter: When ``True``, adds a random offset in
            ``[0, delay * 0.1]`` to spread concurrent retries.
        retryable_exceptions: Tuple of exception types that should trigger a
            retry.  Any exception *not* in this tuple propagates immediately
            without retrying.  Defaults to catching all ``Exception`` subclasses.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.base_delay < 0:
            raise ValueError(f"base_delay must be >= 0, got {self.base_delay}")
        if self.max_delay < self.base_delay:
            raise ValueError(
                f"max_delay ({self.max_delay}) must be >= base_delay ({self.base_delay})"
            )
        if self.exponential_base <= 0:
            raise ValueError(f"exponential_base must be > 0, got {self.exponential_base}")

    async def execute(self, coro_fn: Callable[[], Awaitable[T]]) -> T:
        """Execute *coro_fn* with retry logic, returning its result on success.

        On every failure whose exception type matches ``retryable_exceptions``
        the policy sleeps for an exponentially growing delay (with optional
        jitter) before the next attempt.  Once ``max_retries`` retries have
        been exhausted, a :class:`RetryExhaustedError` is raised carrying the
        total attempt count and the last exception.

        Any exception type *not* listed in ``retryable_exceptions`` bypasses
        retry logic entirely and is re-raised immediately so callers are not
        silently swallowing programming errors.

        Args:
            coro_fn: A zero-argument callable that returns a new awaitable on
                each call.  **Do not pass a coroutine directly** — a coroutine
                can only be awaited once and cannot be retried.

        Returns:
            The value returned by ``coro_fn`` on a successful attempt.

        Raises:
            RetryExhaustedError: When all attempts fail with a retryable
                exception.
            Exception: Immediately, for any non-retryable exception type.
        """
        total_attempts = self.max_retries + 1
        last_exc: Exception | None = None

        for attempt in range(total_attempts):
            try:
                return await coro_fn()
            except self.retryable_exceptions as exc:
                last_exc = exc
                is_last_attempt = attempt == total_attempts - 1
                if is_last_attempt:
                    raise RetryExhaustedError(
                        attempts=total_attempts, last_exception=exc
                    ) from exc

                # Compute exponential backoff capped at max_delay.
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay,
                )
                # Jitter spreads concurrent retries to avoid thundering herd:
                # without it every caller wakes at the exact same moment and
                # bombards the recovering service together.
                if self.jitter:
                    delay += random.uniform(0, delay * 0.1)

                await asyncio.sleep(delay)

        # This line is unreachable but satisfies type checkers.
        raise RetryExhaustedError(attempts=total_attempts, last_exception=last_exc)  # type: ignore[arg-type]
