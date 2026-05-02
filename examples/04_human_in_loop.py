"""
=============================================================================
EXAMPLE 04: Human-in-the-Loop — approval gates before dangerous actions
=============================================================================

WHY DOES HUMAN-IN-THE-LOOP MATTER?
------------------------------------
Agents are powerful exactly because they act autonomously — but that power
cuts both ways. An agent that can delete files can also delete the WRONG
files. An agent that can send emails can send emails to the wrong people.
An agent that can write to a database can corrupt production data.

"Human-in-the-loop" means the agent PAUSES before executing a sensitive
action and asks a human: "Is this OK?" The human can:
  - Approve:  the tool runs normally
  - Reject:   the tool is skipped; the agent is told why and can try
              an alternative approach
  - Modify:   (via a custom callback) change the arguments before execution

This pattern is essential for any agent with write access to the real world.

HOW requires_approval WORKS
-----------------------------
You pass a set of tool names that need approval:

    requires_approval={"delete_file", "send_email"}

Any tool NOT in that set runs automatically (like a normal SingleAgentLoop).
Tools IN the set are paused and the approval callback is called first.

Pass requires_approval=None (the default) to require approval for ALL tools.
Pass requires_approval=set() (empty set) to skip approval for everything
(equivalent to SingleAgentLoop).

CUSTOMISING THE APPROVAL CALLBACK
-----------------------------------
The default built-in callback prints to stdout and reads from stdin — perfect
for CLI scripts like this one.

For a web app you would pass a custom async callback:

    async def web_approval(tool_name: str, args: dict) -> bool | str:
        # Store the pending action in your database
        # Send a notification (email, Slack, push notification)
        # Wait for the user to click Approve/Reject in the UI
        # Return True (approved) or a rejection reason string
        await db.insert_pending_approval(tool_name, args)
        return await wait_for_user_decision(tool_name)

    agent = HumanInLoopPattern(
        llm_client=...,
        config=...,
        approval_callback=web_approval,
        requires_approval={"delete_file"},
    )

The callback signature: (tool_name: str, arguments: dict) → bool | str | dict
  - True         → approve, run the tool
  - False        → reject with a generic message
  - str          → reject, and tell the LLM this reason (it can try again)
  - dict         → approve but replace the arguments with this modified dict

Run:
    ANTHROPIC_API_KEY=sk-ant-... python examples/04_human_in_loop.py

When prompted, type:
    y      → approve the action (tool runs)
    n      → reject (you will then be asked for a reason)
    anything else at the y/n prompt → treated as 'n'
=============================================================================
"""

import asyncio
import os

from fastmcp import FastMCP

from mcp_agent_framework import AgentConfig, AnthropicClient, HumanInLoopPattern

# ---------------------------------------------------------------------------
# 1. Define an in-process MCP server with two tools:
#    read_file  — safe, runs automatically
#    delete_file — dangerous, requires human approval
# ---------------------------------------------------------------------------

app = FastMCP("file_tools")


@app.tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path."""
    # Stub — returns fake content so this example works without real files.
    return f"[Contents of '{path}']\nHello, world! This is the file content.\n(3 lines, 256 bytes)"


@app.tool
def delete_file(path: str) -> str:
    """Permanently delete a file at the given path. This action cannot be undone."""
    # Stub — in a real app this would call os.remove(path) or similar.
    return f"File '{path}' has been permanently deleted."


# ---------------------------------------------------------------------------
# 2. Run the agent with the approval gate
# ---------------------------------------------------------------------------

async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Get your key at https://console.anthropic.com and run:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-..."
        )
        return

    config = AgentConfig(
        mcp_server_config=app,
        system_prompt=(
            "You are a file management assistant. "
            "When asked to read and then delete a file, always read it first, "
            "then delete it. Report what you found and what actions you took."
        ),
        max_iterations=8,
    )

    # HumanInLoopPattern wraps the ReAct loop with an approval gate.
    # requires_approval={"delete_file"} means:
    #   - read_file  → runs automatically, no prompt
    #   - delete_file → pauses and asks for your approval
    agent = HumanInLoopPattern(
        llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
        config=config,
        requires_approval={"delete_file"},
        # approval_callback is not set → uses the built-in CLI prompt
    )

    task = "Read the file README.md, then delete the file temp.log"

    print("=" * 60)
    print("Human-in-the-Loop Agent Demo")
    print("=" * 60)
    print(f"Task: {task}")
    print()
    print("The agent will:")
    print("  1. Call read_file(README.md)  — runs automatically (no approval needed)")
    print("  2. Call delete_file(temp.log) — PAUSES and asks for your approval")
    print()
    print("When prompted:")
    print("  Type 'y' + Enter to approve")
    print("  Type 'n' + Enter to reject (you will be asked for a reason)")
    print("=" * 60)
    print()

    result = await agent.run(task)

    print()
    print("=" * 60)
    print("Agent's final response:")
    print("=" * 60)
    print(result)
    print()
    print("Key takeaway: read_file ran silently; delete_file required your sign-off.")
    print("Change requires_approval=set() to remove all gates (like SingleAgentLoop).")
    print("Change requires_approval=None to gate EVERY tool call.")


if __name__ == "__main__":
    asyncio.run(main())
