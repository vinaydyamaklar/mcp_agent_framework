"""
=============================================================================
LangGraph + MCP Agent Framework — 02: Interrupts & Human-in-the-Loop at Scale
=============================================================================

WHAT THIS EXAMPLE TEACHES
--------------------------
This framework's HumanInLoopPattern uses input() — it blocks the event loop
waiting for a human. In production that means:
  - The process hangs until a human is at the terminal
  - You can't pause overnight and resume in the morning
  - You can't route approvals through Slack, email, or a web UI

LangGraph's interrupt() solves all of this:
  1. Agent reaches a decision point → calls interrupt(payload)
  2. LangGraph saves the full graph state to the checkpointer
  3. ainvoke() returns immediately with the interrupt payload
  4. Hours/days later, a human reviews and sends Command(resume=decision)
  5. LangGraph loads the saved state and resumes from exactly where it paused

THE FLOW
--------
  ┌─────────────┐    ┌──────────────────┐    ┌────────────────┐
  │  plan_node  │───►│ human_review_node│───►│  execute_node  │
  │  (agent     │    │ interrupt() here  │    │  (runs only    │
  │   plans     │    │ saves state       │    │   if approved) │
  │   action)   │    │ waits for human   │    │                │
  └─────────────┘    └──────────────────┘    └────────────────┘

RUNNING
-------
    pip install langgraph
    pip install -r ../../requirements.txt && pip install -e ../..
    export ANTHROPIC_API_KEY=sk-ant-...
    python 02_interrupts_hitl.py
=============================================================================
"""

from __future__ import annotations

import asyncio
from typing import Annotated, TypedDict

from fastmcp import FastMCP
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from mcp_agent_framework import AgentConfig, AnthropicClient
from mcp_agent_framework.patterns import SingleAgentLoop
from mcp_agent_framework.types import Message

# =============================================================================
# MCP server — tools that have side effects requiring human approval
# =============================================================================

app = FastMCP("hitl_demo")

@app.tool
async def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. Requires human approval before executing."""
    # In a real system this would call an email API
    return f"Email sent to {to} — Subject: '{subject}'"

@app.tool
async def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': [simulated results about {query}]"


# =============================================================================
# State
# =============================================================================

class State(TypedDict):
    messages:       Annotated[list, add_messages]
    planned_action: dict    # what the agent wants to do
    action_result:  str     # result after execution


# =============================================================================
# Node 1 — plan_node
# The agent figures out what action it wants to take.
# =============================================================================

async def plan_node(state: State) -> dict:
    """Agent reads the task and decides what action to propose."""
    user_msg = state["messages"][-1]
    content  = user_msg["content"] if isinstance(user_msg, dict) else user_msg.content

    # Use the LLM to decide what to do (simplified: just extract intent)
    client  = AnthropicClient("claude-haiku-4-5-20251001")
    history = [Message(role="user", content=f"What single action should be taken for this task? Task: {content}\nReply with: ACTION: <tool_name> | ARGS: <brief description>")]
    resp    = await client.complete(history)
    plan    = resp.content or "ACTION: search_web | ARGS: general search"

    return {
        "planned_action": {"description": plan, "task": content},
        "messages": [{"role": "assistant", "content": f"Proposed action: {plan}"}],
    }


# =============================================================================
# Node 2 — human_review_node
#
# This is the key difference from HumanInLoopPattern's input() approach.
# interrupt() does NOT block — it saves state and returns control to the caller.
# The process can terminate. Resume later with Command(resume=decision).
# =============================================================================

def human_review_node(state: State) -> dict:
    """Pause and surface the proposed action to a human for review."""

    # interrupt() saves graph state and surfaces the payload to the caller.
    # ainvoke() returns at this point — the process is free to do other work
    # (or terminate entirely). Resume later with Command(resume=...).
    human_decision = interrupt({
        "message":         "Action requires approval before execution.",
        "proposed_action": state["planned_action"]["description"],
        "task":            state["planned_action"]["task"],
        "options":         {"approve": True, "reject": "reason string"},
    })

    # human_decision is whatever was passed to Command(resume=...)
    approved = human_decision.get("approved", False)
    reason   = human_decision.get("reason", "")

    return {
        "planned_action": {
            **state["planned_action"],
            "approved": approved,
            "rejection_reason": reason,
        }
    }


# =============================================================================
# Node 3 — execute_node
# Runs the agent loop only if the human approved the action.
# =============================================================================

async def execute_node(state: State) -> dict:
    """Execute the action — only reached if human approved."""
    action   = state["planned_action"]
    approved = action.get("approved", False)

    if not approved:
        reason = action.get("rejection_reason", "No reason given.")
        return {
            "action_result": f"Action rejected by human. Reason: {reason}",
            "messages": [{"role": "assistant", "content": f"Action was rejected: {reason}"}],
        }

    # Run the full agent loop now that we have approval
    config = AgentConfig(
        mcp_server_config=app,
        system_prompt="Execute the task using the available tools.",
        max_iterations=5,
    )
    result = await SingleAgentLoop(
        llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
        config=config,
    ).run(action["task"])

    return {
        "action_result": result,
        "messages": [{"role": "assistant", "content": result}],
    }


# =============================================================================
# Build graph
# =============================================================================

def build_graph(checkpointer):
    builder = StateGraph(State)
    builder.add_node("plan",   plan_node)
    builder.add_node("review", human_review_node)
    builder.add_node("execute", execute_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan",    "review")
    builder.add_edge("review",  "execute")
    builder.add_edge("execute", END)
    return builder.compile(checkpointer=checkpointer, interrupt_before=["review"])


# =============================================================================
# Demo
# =============================================================================

async def main() -> None:
    checkpointer = MemorySaver()
    graph        = build_graph(checkpointer)
    config       = {"configurable": {"thread_id": "hitl-demo-1"}}

    task = "Search for the latest Python 3.13 release notes and send a summary to team@example.com"

    # ── Step 1: Start the run — will pause at human_review_node ──────────────
    print("=" * 60)
    print("Step 1: Agent plans action, then pauses for human review")
    print("=" * 60)
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": task}]},
        config,
    )
    # At this point ainvoke() has returned — the graph is paused.
    # In production: send a Slack message / email / push notification to the
    # human reviewer with the interrupt payload.
    current = graph.get_state(config)
    pending = current.tasks[0].interrupts[0].value if current.tasks else {}
    print(f"\n[PAUSED — awaiting human review]")
    print(f"  Proposed: {pending.get('proposed_action', 'N/A')}")
    print(f"  Task:     {pending.get('task', 'N/A')}")

    # ── Step 2: Human reviews (simulated) and approves ───────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Human approves the action")
    print("=" * 60)
    result = await graph.ainvoke(
        Command(resume={"approved": True}),
        config,
    )
    print(f"\nResult after approval:\n{result.get('action_result', '')}")

    # ── Step 3: Show what rejection looks like ───────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Same task, but human REJECTS this time")
    print("=" * 60)
    config2 = {"configurable": {"thread_id": "hitl-demo-2"}}
    await graph.ainvoke(
        {"messages": [{"role": "user", "content": task}]},
        config2,
    )
    result2 = await graph.ainvoke(
        Command(resume={"approved": False, "reason": "Do not send emails without legal review."}),
        config2,
    )
    print(f"\nResult after rejection:\n{result2.get('action_result', '')}")


if __name__ == "__main__":
    asyncio.run(main())
