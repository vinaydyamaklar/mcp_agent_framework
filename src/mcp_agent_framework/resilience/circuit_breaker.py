"""
Circuit breaker pattern for preventing cascading failures to downstream services.

Usage example::

    cb = CircuitBreaker(name="payments-api", failure_threshold=5, recovery_timeout=30.0)
    result = await cb.call(lambda: payments_client.charge(order))
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    """The three states of a circuit breaker.

    CLOSED:    Normal operation.  Calls pass through; failures are counted.
    OPEN:      The failure threshold was breached.  All calls are rejected
               immediately without touching the downstream service, giving it
               time to recover.
    HALF_OPEN: A probe state entered after ``recovery_timeout`` seconds.  A
               limited number of calls are allowed through.  Consecutive
               successes close the circuit again; a single failure re-opens it.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is attempted while the circuit is OPEN.

    Attributes:
        name: Identifier of the circuit breaker that rejected the call.
        failure_count: Number of failures that tripped the circuit.
        opened_at: ``time.monotonic()`` timestamp when the circuit opened.
    """

    def __init__(self, name: str, failure_count: int, opened_at: float) -> None:
        self.name = name
        self.failure_count = failure_count
        self.opened_at = opened_at
        age = time.monotonic() - opened_at
        super().__init__(
            f"Circuit '{name}' is OPEN (tripped after {failure_count} failure(s), "
            f"{age:.1f}s ago). Call rejected."
        )


class CircuitBreaker:
    """Async circuit breaker that short-circuits calls to a failing dependency.

    The circuit breaker pattern works like an electrical fuse: when too many
    faults accumulate the circuit "trips" (opens) and subsequent calls fail
    immediately without reaching the struggling downstream service.  This
    prevents your process from wasting threads/connections on a service that
    cannot respond, stops error storms from propagating, and gives the
    downstream service a quiet period to recover.

    **When to use circuit breaker vs retry:**
    Use *retry* for transient blips (network hiccup, momentary overload) where
    the service is likely healthy and will succeed on the next attempt.  Use a
    *circuit breaker* when the service may be completely down or severely
    degraded for an extended period — retrying in that case amplifies load and
    worsens the outage.  The two patterns complement each other: a retry policy
    wraps individual calls, while the circuit breaker wraps the service as a
    whole and trips when the retry budget keeps getting consumed.

    State machine::

        CLOSED ──(failures >= threshold)──► OPEN
          ▲                                   │
          │                         (recovery_timeout elapsed)
          │                                   ▼
          └──(successes >= success_threshold)── HALF_OPEN
                                               │
                              (any failure) ───┘ (back to OPEN)

    Args:
        name: Human-readable identifier used in error messages and logging.
        failure_threshold: Consecutive failures in CLOSED state needed to open
            the circuit.
        recovery_timeout: Seconds the circuit stays OPEN before allowing a
            probe call (transitioning to HALF_OPEN).
        success_threshold: Consecutive successes required in HALF_OPEN state
            to close the circuit and resume normal operation.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._opened_at: float | None = None
        # The lock protects state reads/writes only — it is NEVER held while
        # awaiting coro_fn() so we don't block other callers during I/O.
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state (read-only snapshot)."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Number of consecutive failures recorded in the current CLOSED window."""
        return self._failure_count

    async def call(self, coro_fn: Callable[[], Awaitable[T]]) -> T:
        """Pass a call through the circuit breaker, enforcing the current state.

        The lock is acquired only for the brief state-inspection and
        state-mutation phases.  It is **released before** ``coro_fn()`` is
        awaited so that concurrent callers are not serialised behind a slow
        I/O operation.

        Args:
            coro_fn: Zero-argument callable returning a fresh awaitable each
                time it is called.  Do not pass a coroutine object directly.

        Returns:
            The value produced by ``coro_fn`` on success.

        Raises:
            CircuitOpenError: When the circuit is OPEN and the recovery timeout
                has not yet elapsed.
            Exception: Any exception raised by ``coro_fn`` itself (and records
                a failure against the circuit).
        """
        # ── Phase 1: inspect state under lock ────────────────────────────────
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._opened_at is None:
                    raise RuntimeError(
                        f"CircuitBreaker '{self.name}': OPEN state but opened_at is None. "
                        "This is a bug — opened_at must be set when the circuit opens."
                    )
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self.recovery_timeout:
                    # Allow one probe call by moving to HALF_OPEN.
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                else:
                    raise CircuitOpenError(
                        name=self.name,
                        failure_count=self._failure_count,
                        opened_at=self._opened_at,
                    )
            # CLOSED or HALF_OPEN: fall through; lock released before the call.

        # ── Phase 2: execute coro_fn WITHOUT holding the lock ─────────────────
        try:
            result = await coro_fn()
        except Exception as exc:
            await self._record_failure()
            raise

        # ── Phase 3: record success under lock ───────────────────────────────
        await self._record_success()
        return result

    async def _record_success(self) -> None:
        """Update state after a successful call."""
        async with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._opened_at = None
                    self._success_count = 0

    async def _record_failure(self) -> None:
        """Update state after a failed call."""
        async with self._lock:
            self._failure_count += 1
            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — snap back to OPEN immediately.
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._success_count = 0
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    async def reset(self) -> None:
        """Forcibly reset the circuit to CLOSED state.

        Clears all failure and success counters.  Intended for use in tests
        and manual operator recovery — do not call this in normal application
        logic as it bypasses the automatic state machine.
        """
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._opened_at = None
