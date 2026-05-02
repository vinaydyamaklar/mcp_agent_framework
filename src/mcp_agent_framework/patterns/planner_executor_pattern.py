"""
Pattern: Planner-Executor

The LLM first generates a structured plan (list of steps), then executes
each step using a SingleAgentLoop. If a step fails, it replans with the
error context. Finally, a synthesiser call combines all step results.

Two-phase design:
  PLAN:    planner_client.complete_structured() → ExecutionPlan
  EXECUTE: for each step → SingleAgentLoop(executor_client).run(step_prompt)
  REPLAN:  on failure → planner_client.complete_structured() with error context
  SYNTH:   planner_client.complete() to summarise all results

Why use separate planner and executor clients?
  Powerful model for planning (complex reasoning, structured output).
  Fast/cheap model for execution (each step is narrower in scope).
  You can pass the same client for both if desired.

                ┌────────────────────────────────────────────┐
                │             PlannerExecutor                 │
                │                                            │
  user_message ─► PLAN: "Step 1: ..., Step 2: ..., ..."      │
                │      ↓                                     │
                │  EXECUTE step 1 → result_1                 │
                │  EXECUTE step 2 → result_2  (or REPLAN)    │
                │  EXECUTE step N → result_N                 │
                │      ↓                                     │
                │  SYNTHESISE: "Given steps and results..."   │
                │      ↓                                     │
                │  final answer                              │
                └────────────────────────────────────────────┘
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.observability import RunContext
from mcp_agent_framework.patterns.single_agent_loop import SingleAgentLoop
from mcp_agent_framework.types import AgentConfig, Message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ExecutionStep(BaseModel):
    """
    A single step in the execution plan.

    Attributes:
        step_number: 1-based position of this step in the plan.
        description: Human-readable description of what must be done.
        tool_hint:   Optional suggestion about which MCP tool the executor
                     should prefer when carrying out this step.
    """

    step_number: int
    description: str
    tool_hint: str | None = None   # optional hint about which tool to use


class ExecutionPlan(BaseModel):
    """
    A structured plan produced by the planner LLM.

    Attributes:
        goal:      Restatement of the user's objective as the planner
                   understands it.
        steps:     Ordered list of ExecutionStep objects.
        reasoning: Optional chain-of-thought that the planner used to derive
                   the steps. Useful for debugging and tracing.
    """

    goal: str
    steps: list[ExecutionStep]
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Helper functions (module-level, not methods)
# ---------------------------------------------------------------------------

def build_step_prompt(
    step: ExecutionStep,
    completed_results: list[str],
    goal: str,
) -> str:
    """
    Build the prompt sent to the executor agent for a single plan step.

    Args:
        step:              The step to execute.
        completed_results: Results from all previously completed steps,
                           formatted as "Step N: <result>".
        goal:              The overall user goal, included for context.

    Returns:
        A self-contained prompt string suitable for
        ``SingleAgentLoop.run()``.
    """
    prior_context = (
        "\n".join(completed_results)
        if completed_results
        else "No steps completed yet."
    )

    tool_hint_line = (
        f"\nPreferred tool: {step.tool_hint}" if step.tool_hint else ""
    )

    return (
        f"Overall goal: {goal}\n"
        f"\n"
        f"Steps completed so far:\n{prior_context}\n"
        f"\n"
        f"Your task for this step (step {step.step_number}):\n"
        f"{step.description}"
        f"{tool_hint_line}\n"
        f"\n"
        f"Complete only this step. Be concise and return the result directly."
    )


def build_replan_prompt(
    goal: str,
    completed: list[str],
    failed_step: ExecutionStep,
    error: str,
    remaining: list[ExecutionStep],
) -> str:
    """
    Build the prompt asking the planner to revise the remaining steps after
    a step failure.

    Args:
        goal:         The original user goal.
        completed:    Results from steps completed before the failure.
        failed_step:  The ExecutionStep that raised an exception.
        error:        String representation of the exception.
        remaining:    The steps that had not yet been attempted (including
                      the failed step).

    Returns:
        A prompt string for ``planner_client.complete_structured()``.
        The response is expected to conform to ``ExecutionPlan``.
    """
    completed_text = (
        "\n".join(completed) if completed else "No steps completed yet."
    )
    remaining_descriptions = "\n".join(
        f"  - Step {s.step_number}: {s.description}" for s in remaining
    )

    return (
        f"You are replanning because a step failed during execution.\n"
        f"\n"
        f"Original goal: {goal}\n"
        f"\n"
        f"Steps already completed successfully:\n{completed_text}\n"
        f"\n"
        f"Failed step (step {failed_step.step_number}): {failed_step.description}\n"
        f"Error: {error}\n"
        f"\n"
        f"Remaining steps that were not yet attempted:\n{remaining_descriptions}\n"
        f"\n"
        f"Please produce a revised ExecutionPlan that achieves the original goal "
        f"given the completed work above. The new plan should start from where "
        f"execution failed and may modify or replace the remaining steps to work "
        f"around the error."
    )


def build_synthesis_prompt(goal: str, results: list[str]) -> str:
    """
    Build the prompt asking the planner to synthesise all step results into
    a final answer.

    Args:
        goal:    The original user goal.
        results: List of step result strings ("Step N: <result>" or
                 "Step N failed: <error>").

    Returns:
        A prompt string for ``planner_client.complete()``.
    """
    results_text = "\n".join(results) if results else "No results available."

    return (
        f"You have just executed a multi-step plan to achieve the following goal:\n"
        f"{goal}\n"
        f"\n"
        f"Here are the results from each step:\n{results_text}\n"
        f"\n"
        f"Please synthesise these results into a single, coherent final answer "
        f"that directly addresses the original goal. Be concise and accurate. "
        f"Do not repeat each step verbatim; instead produce a unified response."
    )


# ---------------------------------------------------------------------------
# Pattern class
# ---------------------------------------------------------------------------

class PlannerExecutorPattern:
    """
    Two-model agentic pattern: a planner LLM decomposes the task, an executor
    LLM carries out each step, and the planner synthesises the final answer.

    Key behaviours:
    - The planner uses ``complete_structured`` so the plan is always a valid
      ``ExecutionPlan`` object with typed steps.
    - Each step is executed by a ``SingleAgentLoop`` (full ReAct loop with
      tools) using the executor client.
    - On failure the planner is asked to revise the *remaining* steps only,
      preserving already-completed work.
    - After exhausting ``max_replan_attempts`` for a step, execution continues
      with the error recorded so the synthesiser can still produce a partial
      answer.

    Usage::

        planner = AnthropicClient()   # e.g. claude-3-5-sonnet — strong reasoning
        executor = AnthropicClient()  # e.g. claude-3-haiku — fast & cheap
        config = AgentConfig(
            mcp_server_config={"mcpServers": {"tools": {"url": "http://localhost:8001/mcp"}}},
            system_prompt="You are a helpful research assistant.",
        )
        agent = PlannerExecutorPattern(planner, executor, config)
        result = await agent.run("Research and summarise the latest MCP spec changes.")
        print(result)
    """

    def __init__(
        self,
        planner_client: BaseLLMClient,
        executor_client: BaseLLMClient,
        config: AgentConfig,
        max_replan_attempts: int = 2,
        context: RunContext | None = None,
    ):
        """
        Initialise the pattern.

        Args:
            planner_client:      LLM client used for PLAN, REPLAN, and SYNTH
                                 phases. Should support structured output.
            executor_client:     LLM client used inside each ``SingleAgentLoop``
                                 for the EXECUTE phase.
            config:              Shared ``AgentConfig`` (MCP server, system
                                 prompt, max_iterations for each executor loop).
            max_replan_attempts: How many times the planner may revise a
                                 failing step before giving up and recording the
                                 error. Defaults to 2.
            context:             Optional ``RunContext`` for tracing / telemetry.
                                 Pass ``None`` to disable tracing.
        """
        self._planner  = planner_client
        self._executor = executor_client
        self._config   = config
        self._max_replan_attempts = max_replan_attempts
        self._context  = context

    async def run(
        self,
        user_message: str,
        history: list[Message] | None = None,
    ) -> str:
        """
        Execute the full Planner-Executor cycle for *user_message*.

        Phases
        ------
        1. **PLAN** — ask the planner for a structured ``ExecutionPlan``.
        2. **EXECUTE** — iterate over steps, running each through a
           ``SingleAgentLoop``. On failure, attempt to replan up to
           ``max_replan_attempts`` times before recording the error and
           moving on.
        3. **SYNTHESISE** — ask the planner to merge all step results into
           a single coherent answer.

        Args:
            user_message: The user's request / question.
            history:      Optional prior conversation context prepended to
                          the planning messages.

        Returns:
            The planner's synthesised final answer, or a newline-joined
            summary of step results if the synthesis call returns no content.
        """
        # ------------------------------------------------------------------
        # Phase 1 — PLAN
        # ------------------------------------------------------------------
        plan_messages = list(history or []) + [
            Message(
                role="user",
                content=f"Create a step-by-step plan to: {user_message}",
            )
        ]

        plan_data = await self._planner.complete_structured(
            plan_messages,
            ExecutionPlan,
            system=self._config.system_prompt or None,
        )
        plan_obj = ExecutionPlan(**plan_data)

        logger.info(
            "[planner-executor] plan: %d steps for goal: %s",
            len(plan_obj.steps),
            plan_obj.goal,
        )
        if plan_obj.reasoning:
            logger.debug("[planner-executor] planner reasoning: %s", plan_obj.reasoning)

        # ------------------------------------------------------------------
        # Phase 2 — EXECUTE (with inline REPLAN on failure)
        # ------------------------------------------------------------------
        completed_results: list[str] = []

        i = 0
        while i < len(plan_obj.steps):
            step = plan_obj.steps[i]
            step_prompt = build_step_prompt(step, completed_results, user_message)

            for attempt in range(self._max_replan_attempts + 1):
                try:
                    logger.debug(
                        "[planner-executor] executing step %d (attempt %d): %s",
                        step.step_number,
                        attempt + 1,
                        step.description,
                    )
                    result = await SingleAgentLoop(
                        self._executor, self._config
                    ).run(step_prompt)
                    completed_results.append(f"Step {step.step_number}: {result}")
                    logger.info(
                        "[planner-executor] step %d done: %s",
                        step.step_number,
                        result[:120],
                    )
                    break

                except Exception as exc:
                    logger.warning(
                        "[planner-executor] step %d failed (attempt %d/%d): %s",
                        step.step_number,
                        attempt + 1,
                        self._max_replan_attempts + 1,
                        exc,
                    )

                    if attempt < self._max_replan_attempts:
                        # REPLAN: ask the planner to revise remaining steps
                        remaining = plan_obj.steps[i:]
                        replan_prompt = build_replan_prompt(
                            goal=user_message,
                            completed=completed_results,
                            failed_step=step,
                            error=str(exc),
                            remaining=remaining,
                        )
                        replan_messages = [
                            Message(role="user", content=replan_prompt)
                        ]
                        new_plan_data = await self._planner.complete_structured(
                            replan_messages,
                            ExecutionPlan,
                            system=self._config.system_prompt or None,
                        )
                        new_plan = ExecutionPlan(**new_plan_data)

                        # Replace remaining steps with the revised plan and
                        # renumber so step_number values are contiguous from 1.
                        plan_obj.steps[i:] = new_plan.steps
                        for j, s in enumerate(plan_obj.steps):
                            s.step_number = j + 1
                        # Update step reference to the (possibly rewritten) step
                        step = plan_obj.steps[i]
                        step_prompt = build_step_prompt(
                            step, completed_results, user_message
                        )

                        logger.info(
                            "[planner-executor] replanned: %d steps remaining",
                            len(plan_obj.steps[i:]),
                        )
                    else:
                        # All replan attempts exhausted — record failure and continue
                        completed_results.append(
                            f"Step {step.step_number} failed: {exc}"
                        )

            i += 1

        # ------------------------------------------------------------------
        # Phase 3 — SYNTHESISE
        # ------------------------------------------------------------------
        synthesis_prompt = build_synthesis_prompt(user_message, completed_results)
        synth_messages = [Message(role="user", content=synthesis_prompt)]

        logger.info("[planner-executor] synthesising %d step results", len(completed_results))

        synth_response = await self._planner.complete(
            synth_messages,
            system=self._config.system_prompt or None,
        )

        return synth_response.content or "\n".join(completed_results)
