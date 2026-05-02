"""
=============================================================================
Example — Context-Aware MCP Server
=============================================================================

WHAT THIS EXAMPLE TEACHES
--------------------------
How to build a request-scoped MCP server where tools close over per-request
context: the current user, their permissions, and their data store.

The problem without context:
    MCP servers are typically singletons. Tools use global state or accept
    user_id as a tool argument. Both approaches leak state across requests
    and put auth logic inside tool logic.

The solution — MCPServerBase(context=...):
    Create a fresh server instance per request. Inject whatever that request
    needs as context. Tools close over self.ctx naturally. Auth, tenant
    isolation, and per-user DB connections just work.

SCENARIO
--------
A customer support agent where:
    - Each request is scoped to one support agent (user)
    - Tools check self.ctx.permissions before executing
    - Tools read from self.ctx.tickets (per-user ticket store)
    - A second server handles escalation — different permission set

RUNNING
-------
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/context_aware_server.py

=============================================================================
"""

import asyncio
import logging
from dataclasses import dataclass, field

from mcp_agent_framework import AgentConfig, AnthropicClient
from mcp_agent_framework.patterns import SingleAgentLoop
from mcp_agent_framework.server import MCPServerBase

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s - %(message)s",
)

# ------------------------------------------------------------------
# Per-request context — injected fresh on every request
# ------------------------------------------------------------------

@dataclass
class SupportContext:
    """Everything a support agent's tools need for one request."""
    user_id:     str
    user_name:   str
    permissions: set[str]
    tickets:     dict[str, dict]   # simulates a per-user DB view


# ------------------------------------------------------------------
# Server 1 — standard support agent (read + respond)
# ------------------------------------------------------------------

class SupportMCPServer(MCPServerBase):
    def __init__(self, ctx: SupportContext):
        super().__init__("support", context=ctx)

        @self.tool
        async def list_tickets(status: str = "open") -> str:
            """List support tickets. status can be 'open', 'closed', or 'all'."""
            tickets = self.ctx.tickets
            if status != "all":
                tickets = {k: v for k, v in tickets.items() if v["status"] == status}
            if not tickets:
                return f"No {status} tickets found."
            lines = [
                f"#{tid} [{t['status']}] {t['subject']} — Priority: {t['priority']}"
                for tid, t in tickets.items()
            ]
            return "\n".join(lines)

        @self.tool
        async def get_ticket(ticket_id: str) -> str:
            """Get full details of a specific ticket by ID."""
            ticket = self.ctx.tickets.get(ticket_id)
            if not ticket:
                return f"Ticket #{ticket_id} not found."
            return (
                f"Ticket #{ticket_id}\n"
                f"Subject:  {ticket['subject']}\n"
                f"Status:   {ticket['status']}\n"
                f"Priority: {ticket['priority']}\n"
                f"Customer: {ticket['customer']}\n"
                f"Notes:    {ticket.get('notes', 'None')}"
            )

        @self.tool
        async def add_note(ticket_id: str, note: str) -> str:
            """Add a note to a ticket. Requires 'tickets:write' permission."""
            if "tickets:write" not in self.ctx.permissions:
                return f"Permission denied. {self.ctx.user_name} does not have 'tickets:write'."
            ticket = self.ctx.tickets.get(ticket_id)
            if not ticket:
                return f"Ticket #{ticket_id} not found."
            ticket["notes"] = note
            return f"Note added to ticket #{ticket_id}."

        @self.tool
        async def close_ticket(ticket_id: str) -> str:
            """Close a ticket. Requires 'tickets:close' permission."""
            if "tickets:close" not in self.ctx.permissions:
                return f"Permission denied. {self.ctx.user_name} does not have 'tickets:close'."
            ticket = self.ctx.tickets.get(ticket_id)
            if not ticket:
                return f"Ticket #{ticket_id} not found."
            ticket["status"] = "closed"
            return f"Ticket #{ticket_id} closed."

        @self.tool
        async def escalate_ticket(ticket_id: str, reason: str) -> str:
            """Escalate a ticket to senior support. Requires 'tickets:escalate' permission."""
            if "tickets:escalate" not in self.ctx.permissions:
                return f"Permission denied. {self.ctx.user_name} does not have 'tickets:escalate'."
            ticket = self.ctx.tickets.get(ticket_id)
            if not ticket:
                return f"Ticket #{ticket_id} not found."
            ticket["status"] = "escalated"
            ticket["escalation_reason"] = reason
            return f"Ticket #{ticket_id} escalated. Reason: {reason}"


# ------------------------------------------------------------------
# Shared ticket store — simulates a DB
# Normally you'd query a real DB per request; here we share one store
# so you can see state changes across demo runs.
# ------------------------------------------------------------------

TICKETS: dict[str, dict] = {
    "T-001": {
        "subject":  "Payment failed on renewal",
        "status":   "open",
        "priority": "high",
        "customer": "alice@example.com",
        "notes":    None,
    },
    "T-002": {
        "subject":  "Cannot login after password reset",
        "status":   "open",
        "priority": "medium",
        "customer": "bob@example.com",
        "notes":    None,
    },
    "T-003": {
        "subject":  "Feature request: bulk export",
        "status":   "open",
        "priority": "low",
        "customer": "carol@example.com",
        "notes":    None,
    },
}


# ------------------------------------------------------------------
# Demo runners
# ------------------------------------------------------------------

async def run_as_junior(task: str) -> str:
    """Junior agent: read-only. Cannot close or escalate."""
    ctx = SupportContext(
        user_id="agent-001",
        user_name="Junior Agent Sam",
        permissions={"tickets:read"},          # read only
        tickets=TICKETS,
    )
    server = SupportMCPServer(ctx)
    agent = SingleAgentLoop(
        llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
        config=AgentConfig(
            mcp_server_config=server.mcp,
            system_prompt=(
                f"You are {ctx.user_name}, a junior support agent. "
                "Help with the task using available tools."
            ),
            max_iterations=6,
        ),
    )
    return await agent.run(task)


async def run_as_senior(task: str) -> str:
    """Senior agent: full permissions — can close and escalate."""
    ctx = SupportContext(
        user_id="agent-002",
        user_name="Senior Agent Jordan",
        permissions={"tickets:read", "tickets:write", "tickets:close", "tickets:escalate"},
        tickets=TICKETS,
    )
    server = SupportMCPServer(ctx)
    agent = SingleAgentLoop(
        llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
        config=AgentConfig(
            mcp_server_config=server.mcp,
            system_prompt=(
                f"You are {ctx.user_name}, a senior support agent. "
                "Help with the task using available tools."
            ),
            max_iterations=6,
        ),
    )
    return await agent.run(task)


async def main() -> None:
    sep = "=" * 60

    # Demo 1 — Junior tries to close a ticket (should be denied)
    print(f"\n{sep}")
    print("DEMO 1: Junior agent tries to close a high-priority ticket")
    print(sep)
    result = await run_as_junior(
        "Look at ticket T-001 and close it if it's high priority."
    )
    print(result)

    # Demo 2 — Junior lists and reads tickets (should work fine)
    print(f"\n{sep}")
    print("DEMO 2: Junior agent summarises all open tickets")
    print(sep)
    result = await run_as_junior(
        "List all open tickets and give me a one-line summary of each."
    )
    print(result)

    # Demo 3 — Senior closes and escalates (full permissions)
    print(f"\n{sep}")
    print("DEMO 3: Senior agent handles the payment failure ticket")
    print(sep)
    result = await run_as_senior(
        "Add a note to T-001 saying 'Investigating payment gateway logs', "
        "then escalate it with reason 'Payment system issue requires engineering review'."
    )
    print(result)

    # Demo 4 — Same server class, completely different context
    print(f"\n{sep}")
    print("DEMO 4: Read-only audit agent sees current ticket state")
    print(sep)
    result = await run_as_junior(
        "List all tickets including closed and escalated ones."
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
