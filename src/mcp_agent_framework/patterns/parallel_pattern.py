"""
Pattern: Parallel Fan-Out + Synthesise

Multiple independent subtasks are dispatched concurrently (fan-out), each
running inside its own SingleAgentLoop. Once all tasks complete (or fail),
a dedicated synthesiser LLM merges their outputs into a single coherent answer.

Use this when:
  - The overall problem can be split into independent sub-problems (no data
    dependency between tasks).
  - Latency matters: running N tasks in parallel is ~N× faster than sequential.
  - Each subtask benefits from a specialised LLM or a different set of MCP tools.
  - You want fault-tolerance: one task failing should not abort the others.

ASCII diagram:

                        user message
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │  Task A      │ │  Task B      │ │  Task C      │   (fan-out, concurrent)
      │  agent loop  │ │  agent loop  │ │  agent loop  │
      │  + MCP srv A │ │  + MCP srv B │ │  + MCP srv B │
      └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
             │                │                │
             └────────────────┼────────────────┘
                              │  results (+ errors)
                              ▼
                    ┌──────────────────┐
                    │   Synthesiser    │   (single LLM, sees all results)
                    │       LLM        │
                    └──────────────────┘
                              │
                              ▼
                       final answer

Key design decisions:
  - asyncio.gather is used so ALL tasks start immediately and run concurrently.
  - return_exceptions is NOT used; instead each task is wrapped in _run_task
    which catches all exceptions and returns a ParallelResult with error set.
    This guarantees gather never raises, so every result is always available.
  - The synthesiser receives structured context: task names, outputs, and any
    error messages, letting it produce an informed answer even if some tasks fail.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.observability.run_context import RunContext
from mcp_agent_framework.patterns.single_agent_loop import SingleAgentLoop
from mcp_agent_framework.types import AgentConfig, Message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ParallelTask:
    """
    One independent subtask to run inside the parallel fan-out phase.

    Attributes:
        name:   A short human-readable label used in logs and in the synthesis
                prompt (e.g. "ml_research", "market_trends").
        prompt: The specific instruction given to this task's agent.  The
                user's original question is appended automatically as
                "User question context: …" so the task stays relevant.
        agent:  A fully configured SingleAgentLoop.  Because each task owns
                its own agent it can use a completely different LLM client
                (e.g. Gemini for one task, Claude for another) and connect to
                a different MCP server — tasks are fully independent.
    """
    name: str
    prompt: str
    agent: SingleAgentLoop


@dataclass
class ParallelResult:
    """
    The outcome of one parallel subtask.

    Attributes:
        name:       Mirrors ParallelTask.name — used to correlate with the
                    original task in logs and synthesis.
        output:     The agent's final text response.  Empty string on failure.
        error:      None when the task succeeded; error message string when it
                    raised an exception.
        elapsed_ms: Wall-clock time the task took, in milliseconds.
    """
    name: str
    output: str
    error: str | None = None
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

_DEFAULT_SYNTHESIS_TEMPLATE = (
    "You have the following research results from parallel tasks:\n\n"
    "{task_results}"
    "Original question: {user_message}\n\n"
    "Synthesise these results into a comprehensive, coherent answer. "
    "If some tasks failed, work with the available results."
)


class ParallelPattern:
    """
    Fan out N subtasks to run concurrently via asyncio.gather, then
    synthesise all results into a single final answer.

    Key properties:
    - Each ParallelTask can use a DIFFERENT SingleAgentLoop with a different
      LLM client and different MCP server — tasks are fully independent.
    - One task failing does NOT cancel the others — errors surface in synthesis.
    - The synthesiser sees each task's name, output, and any errors.

    Usage:
        tasks = [
            ParallelTask("ml_research",   "Research recent ML breakthroughs", research_agent),
            ParallelTask("market_trends", "Research AI market trends",         market_agent),
            ParallelTask("safety",        "Research AI safety developments",   research_agent),
        ]
        agent = ParallelPattern(
            synthesiser_client=AnthropicClient("claude-sonnet-4-6"),
            tasks=tasks,
            config=AgentConfig(mcp_server_config={}, system_prompt="You are a research synthesiser."),
        )
        report = await agent.run("Give me a comprehensive AI overview for 2026.")
    """

    def __init__(
        self,
        synthesiser_client: BaseLLMClient,
        tasks: list[ParallelTask],
        config: AgentConfig,
        synthesis_prompt_template: str | None = None,
        context: RunContext | None = None,
    ):
        """
        Initialise the ParallelPattern.

        Args:
            synthesiser_client:        LLM client used exclusively for the
                                       synthesis step (not the subtasks).
            tasks:                     List of ParallelTask definitions to run
                                       concurrently in the fan-out phase.
            config:                    AgentConfig whose system_prompt and
                                       max_iterations govern the synthesiser.
                                       The mcp_server_config on this config is
                                       not used (each task's agent has its own).
            synthesis_prompt_template: Optional override for the default
                                       synthesis prompt.  Must contain
                                       ``{task_results}`` and
                                       ``{user_message}`` placeholders.
            context:                   Optional RunContext for distributed
                                       tracing / observability.
        """
        self._synthesiser = synthesiser_client
        self._tasks = tasks
        self._config = config
        template = synthesis_prompt_template or _DEFAULT_SYNTHESIS_TEMPLATE
        # Validate template placeholders at construction time to catch mistakes
        # before the first run() call.
        missing = [p for p in ("{task_results}", "{user_message}") if p not in template]
        if missing:
            raise ValueError(
                f"synthesis_prompt_template is missing required placeholder(s): "
                f"{', '.join(missing)}"
            )
        self._synthesis_prompt_template = template
        self._context = context

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, user_message: str, history: list[Message] | None = None) -> str:
        """
        Execute the full fan-out + synthesise pipeline.

        Phase 1 — Fan-out:
            All tasks are dispatched concurrently with asyncio.gather.  Each
            task receives its configured prompt plus the original user message
            as context.  Failures are caught per-task and recorded; no task
            cancels another.

        Phase 2 — Synthesise:
            The synthesiser LLM receives a structured prompt listing every
            task name, its output (or error), and the original user question.
            It produces the final answer that is returned to the caller.

        Args:
            user_message: The user's top-level question or instruction.
            history:      Optional prior conversation messages passed to the
                          synthesiser (NOT to the subtask agents).

        Returns:
            The synthesiser's final text response.
        """
        logger.debug("[parallel] starting %d tasks", len(self._tasks))

        # ---- Phase 1: fan-out ----
        results: list[ParallelResult] = await asyncio.gather(
            *[self._run_task(task, user_message) for task in self._tasks],
        )

        success_count = sum(1 for r in results if r.error is None)
        failed_count = len(results) - success_count

        logger.info(
            "[parallel] %d tasks completed in parallel — %d succeeded, %d failed",
            len(results),
            success_count,
            failed_count,
        )

        for result in results:
            status = "OK" if result.error is None else f"FAILED: {result.error}"
            logger.debug(
                "[parallel] task '%s' — %s (%.1f ms)", result.name, status, result.elapsed_ms
            )

        # ---- Phase 2: synthesise ----
        synthesis_input = self._build_synthesis_prompt(results, user_message)

        synth_messages = list(history or [])
        synth_messages.append(Message(role="user", content=synthesis_input))

        logger.debug("[parallel] calling synthesiser (%s)", self._synthesiser.provider_name())
        response = await self._synthesiser.complete(
            messages=synth_messages,
            tools=None,
            system=self._config.system_prompt or None,
        )

        return response.content or ""

    async def run_stream(
        self,
        user_message: str,
        history: list[Message] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Fan-out runs synchronously; stream the synthesis response."""
        from mcp_agent_framework.types import StreamEvent

        results: list[ParallelResult] = await asyncio.gather(
            *[self._run_task(task, user_message) for task in self._tasks],
        )

        synthesis_input = self._build_synthesis_prompt(results, user_message)
        synth_messages = list(history or [])
        synth_messages.append(Message(role="user", content=synthesis_input))

        async for event in self._synthesiser.stream(
            synth_messages,
            tools=None,
            system=self._config.system_prompt or None,
        ):
            yield event

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_task(self, task: ParallelTask, context_msg: str) -> ParallelResult:
        """
        Run a single subtask and return a ParallelResult regardless of success
        or failure.

        The task's prompt is combined with the user's original message so each
        subtask agent understands the broader goal while staying focused on its
        specific instruction.

        Args:
            task:        The ParallelTask to execute.
            context_msg: The original user message appended as context.

        Returns:
            ParallelResult with output populated on success or error set on
            failure.  elapsed_ms is always populated.
        """
        t0 = time.monotonic()
        full_prompt = f"{task.prompt}\n\nUser question context: {context_msg}"
        try:
            output = await task.agent.run(full_prompt)
            elapsed = (time.monotonic() - t0) * 1000
            logger.debug("[parallel] task '%s' finished in %.1f ms", task.name, elapsed)
            return ParallelResult(
                name=task.name,
                output=output,
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            logger.error(
                "[parallel] task '%s' failed after %.1f ms: %s",
                task.name,
                elapsed,
                exc,
            )
            return ParallelResult(
                name=task.name,
                output="",
                error=str(exc),
                elapsed_ms=elapsed,
            )

    def _build_synthesis_prompt(
        self, results: list[ParallelResult], user_message: str
    ) -> str:
        """
        Construct the synthesis prompt by formatting each task result into a
        human-readable block and injecting it into the template.

        Failed tasks are labelled explicitly so the synthesiser knows to treat
        their absence gracefully.

        Args:
            results:      All ParallelResult objects from the fan-out phase.
            user_message: The original user question.

        Returns:
            The fully rendered synthesis prompt string.
        """
        task_blocks: list[str] = []
        for result in results:
            if result.error is not None:
                task_blocks.append(f"Task '{result.name}' FAILED: {result.error}\n\n")
            else:
                task_blocks.append(f"Task '{result.name}':\n{result.output}\n\n")

        return self._synthesis_prompt_template.format(
            task_results="".join(task_blocks),
            user_message=user_message,
        )
