"""
Composed MCP Server — mount multiple MCP servers under one.

Use this when: you want ONE endpoint that exposes tools from MULTIPLE
underlying MCP servers. A single client connects here and sees all tools.

This is the server-side counterpart to OrchestratorWorkerPattern.
OrchestratorWorkerPattern connects to workers directly from the client.
ComposedMCPServer merges workers on the server side into one endpoint.

                    ┌───────────────────────────────┐
                    │     ComposedMCPServer          │
                    │  (one FastMCP instance)        │
                    │                                │
                    │  mounts: ┌──────────────────┐  │
                    │          │  Nova MCP Server  │  │
                    │          └──────────────────┘  │
                    │          ┌──────────────────┐  │
                    │          │  Brown MCP Server │  │
                    │          └──────────────────┘  │
                    └───────────────────┬────────────┘
                                        │  single endpoint
                                        ▼
                               MCP Client / LLM
                         (sees all tools from all servers)

Real-world example from this course (lesson 25):
    nova_config  = {"url": "http://localhost:8001/mcp"}
    brown_config = {"url": "http://localhost:8002/mcp"}
    server = build_composed_server({"nova": nova_config, "brown": brown_config})
    server.run(transport="http", port=8003)
"""
from __future__ import annotations

import logging
from typing import Any

from fastmcp import Client, FastMCP

logger = logging.getLogger(__name__)


def build_composed_server(
    servers: dict[str, dict[str, Any]],
    name: str = "ComposedMCPServer",
    version: str = "0.1.0",
    combined_prompt: str | None = None,
) -> FastMCP:
    """
    Create a single FastMCP server that proxies multiple upstream MCP servers.

    Args:
        servers:
            Dict of server_name → MCP server config. Each config follows the
            standard MCP format for a single server, e.g.:
              {"url": "http://localhost:8001/mcp"}               # HTTP
              {"command": "python", "args": ["server.py"]}       # stdio

        name:
            Name of the composed server (shown in MCP handshake).

        version:
            Version string of the composed server.

        combined_prompt:
            Optional system prompt registered as an MCP prompt on the composed
            server. Use this to give the orchestrating LLM full instructions
            for how to use all the mounted tools together.

    Returns:
        A FastMCP instance ready to call .run() on.

    Usage:
        server = build_composed_server(
            servers={
                "nova":  {"url": "http://localhost:8001/mcp"},
                "brown": {"url": "http://localhost:8002/mcp"},
            },
            name="Nova+Brown",
            combined_prompt="Phase 1: use Nova tools for research. Phase 2: use Brown tools to write.",
        )
        server.run(transport="http", port=8003)
    """
    composed = FastMCP(name=name, version=version)

    for server_name, server_config in servers.items():
        # Wrap in the mcpServers format FastMCP Client expects
        client_config = {"mcpServers": {server_name: server_config}}
        client = Client(client_config)
        proxy  = FastMCP.as_proxy(client)
        composed.mount(proxy)
        logger.debug("[composed] mounted server '%s'", server_name)

    if combined_prompt:
        _register_prompt(composed, combined_prompt)

    return composed


def _register_prompt(mcp: FastMCP, prompt_text: str) -> None:
    """Register the combined workflow prompt on the composed server."""

    @mcp.prompt()
    def workflow_instructions() -> str:
        """Full workflow instructions for the orchestrating agent."""
        return prompt_text
