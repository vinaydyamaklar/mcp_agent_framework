# Lesson 15 — Resilience

**Unit 5: Scale**

---

## What you will learn

- Why distributed systems fail and why agents must handle it gracefully
- `RetryPolicy` — exponential backoff with jitter
- `CircuitBreaker` — the CLOSED / OPEN / HALF_OPEN state machine
- The thundering herd problem and how jitter solves it
- The `-O` flag bug and why `assert` is never the right guard

---

## The concept

In development, everything works. In production:

- APIs return HTTP 429 (rate limited) when you send too many requests
- Tool servers go down for 30 seconds during deploys
- The network drops packets intermittently
- External services have 99.5% uptime — meaning ~44 hours of downtime per year

Without resilience, one failure crashes your agent. With resilience, your agent retries intelligently and stops hitting services that are clearly broken.

---

## `RetryPolicy` — exponential backoff with jitter

```python
from mcp_agent_framework import RetryPolicy

policy = RetryPolicy(
    max_retries=3,
    base_delay=1.0,      # first retry waits ~1 second
    max_delay=60.0,      # never wait more than 60 seconds
    exponential_base=2.0, # each retry doubles the wait
    jitter=True,          # add randomness to prevent thundering herd
)
```

**The delay calculation:**
```
attempt 1 fails → wait: base_delay × exponential_base^0 + jitter = 1s + ~0.5s = ~1.5s
attempt 2 fails → wait: base_delay × exponential_base^1 + jitter = 2s + ~1s   = ~3s
attempt 3 fails → wait: base_delay × exponential_base^2 + jitter = 4s + ~2s   = ~6s
attempt 4 → raise RetryExhaustedError
```

Capped at `max_delay` to prevent waiting forever.

**The thundering herd problem:**
If 100 agents all fail at the same time (e.g., rate limit hit), and they all retry at exactly `t + 1.0s`, they all hit the service again simultaneously — causing another rate limit. With jitter (random fraction added to the delay), each agent waits a slightly different amount. The 100 retries spread out over a 1–3 second window, the service recovers, and all 100 succeed.

**The `__post_init__` validation:**
```python
def __post_init__(self) -> None:
    if self.max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if self.base_delay < 0:
        raise ValueError("base_delay must be >= 0")
    if self.max_delay < self.base_delay:
        raise ValueError("max_delay must be >= base_delay")
    if self.exponential_base <= 0:
        raise ValueError("exponential_base must be > 0")
```

Validation happens at construction. You find invalid configs immediately at startup, not at runtime during a user request.

---

## `CircuitBreaker` — stop hammering a broken service

```python
from mcp_agent_framework import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,     # open after 5 consecutive failures
    recovery_timeout=30.0,   # wait 30 seconds before trying again
    success_threshold=2,     # 2 successes in HALF_OPEN → back to CLOSED
)
```

**The three states:**

```
CLOSED (normal operation)
    ↓ 5 consecutive failures
OPEN (failing fast)
    ↓ 30 seconds pass
HALF_OPEN (testing recovery)
    ↓ 2 successes → CLOSED
    ↓ 1 failure   → OPEN again
```

**CLOSED:** All calls go through. Failures are counted. 5 consecutive failures → trips to OPEN.

**OPEN:** All calls fail immediately with `CircuitOpenError` — no actual call made. The circuit "opens" to protect both your system and the failing service from being hammered. After `recovery_timeout` seconds, automatically transitions to HALF_OPEN.

**HALF_OPEN:** One test call is allowed through. If it succeeds (and the next `success_threshold - 1` calls succeed), the circuit closes. If it fails, back to OPEN.

**The key insight:** If 10 consecutive calls fail, the 11th will almost certainly fail too. Better to fail fast (circuit OPEN, no API call, instant error) than waste time and money waiting for timeouts. The circuit re-closes automatically — it's self-healing.

---

## The `-O` flag bug

In the previous version of `circuit_breaker.py`, there was:

```python
assert self._opened_at is not None, "Circuit opened but _opened_at is None"
```

This looks defensive. But Python's `-O` (optimise) flag strips all `assert` statements. In production, Python is often run with optimisations. The assert was silently removed, and a `NoneType` error would surface elsewhere with no context.

**The fix:**
```python
if self._opened_at is None:
    raise RuntimeError(
        "Circuit is OPEN but _opened_at is None — this is a bug in CircuitBreaker."
    )
```

`RuntimeError` is never stripped. This is a general rule: **never use `assert` for runtime invariants in production code.** Use explicit `if ... raise` instead. Reserve `assert` for debugging and tests only.

---

## Using resilience in practice

Wrap your LLM calls and tool server calls:

```python
from mcp_agent_framework import RetryPolicy, CircuitBreaker, CircuitOpenError, RetryExhaustedError

policy  = RetryPolicy(max_retries=3, base_delay=1.0)
breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

async def resilient_complete(client, messages, tools):
    for attempt in range(policy.max_retries + 1):
        try:
            if breaker.state == CircuitState.OPEN:
                raise CircuitOpenError("LLM service circuit is open")

            async with breaker:
                return await client.complete(messages, tools)

        except RateLimitError:
            if attempt == policy.max_retries:
                raise RetryExhaustedError(f"Failed after {policy.max_retries} retries")
            delay = policy.calculate_delay(attempt)
            await asyncio.sleep(delay)
```

In a production system, you wrap the LLM client calls, external API calls, and database calls separately — each with their own circuit breaker state.

---

## Read these files

```
src/mcp_agent_framework/resilience/retry.py
src/mcp_agent_framework/resilience/circuit_breaker.py
```

In `retry.py`, find `__post_init__` validation and the delay calculation.

In `circuit_breaker.py`, find the three state transitions. Find the `async with self._lock` wrapping `reset()` — this ensures thread-safe state changes under concurrent calls.

---

## Build this

Create a fake "flaky API" and observe retry and circuit breaker behaviour:

```python
import random, asyncio
from mcp_agent_framework import RetryPolicy, RetryExhaustedError
from mcp_agent_framework import CircuitBreaker, CircuitOpenError, CircuitState

call_count = 0
fail_rate  = 0.7   # 70% failure rate

async def flaky_api_call(query: str) -> str:
    global call_count
    call_count += 1
    if random.random() < fail_rate:
        raise ConnectionError(f"API timed out (attempt #{call_count})")
    return f"Result for: {query}"

# Part 1: Test RetryPolicy
policy = RetryPolicy(max_retries=4, base_delay=0.1, exponential_base=2.0, jitter=False)

# Part 2: Add CircuitBreaker on top
breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0)
```

Questions to answer:
1. How many calls does the retry policy make before giving up?
2. After how many failures does the circuit open?
3. How long after opening does HALF_OPEN appear?
4. Set `fail_rate=0.0` (100% success) and observe the circuit close again.

---

## Key terms

| Term | Meaning |
|------|---------|
| Exponential backoff | Each retry waits 2× longer than the previous |
| Jitter | Random delay added to prevent thundering herd |
| `RetryExhaustedError` | All retry attempts failed |
| `CircuitOpenError` | Circuit is OPEN — call rejected immediately |
| CLOSED state | Normal — calls go through, failures tracked |
| OPEN state | Failing fast — all calls rejected immediately |
| HALF_OPEN state | Testing recovery — one call allowed through |
| Thundering herd | Many clients retrying at the same moment, overwhelming a recovering service |

---

## Connects to

- **Lesson 16** — observability: circuit breaker state changes are traced events
- **Lesson 4** — MCP tool calls are the most common retry target
- **Lesson 3** — LLM client calls (rate limits, 500 errors) are the second most common

---

*Lesson 15 of 21 — Applied AI Engineering*
