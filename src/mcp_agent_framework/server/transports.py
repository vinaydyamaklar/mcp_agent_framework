"""
MCP Transports — how a server and client physically talk to each other.

There are two transports in MCP. Choosing the wrong one is the most common
source of confusion when getting started.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSPORT 1: stdio  (standard input/output)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

How it works:
    The CLIENT spawns the server as a subprocess. They talk via stdin/stdout.
    No network ports. No URLs. The server process lives and dies with the client.

When to use:
    ✓ Running tools locally (file system, local DB, shell commands)
    ✓ Development and testing
    ✓ Desktop apps (Claude Desktop, Cursor, VS Code)
    ✓ When you want zero network configuration

Config on the client side looks like:
    {
        "mcpServers": {
            "my_server": {
                "command": "python",
                "args": ["path/to/server.py"],
                "env": {"MY_KEY": "value"}   ← optional env vars
            }
        }
    }

The server code just calls:
    server.run()          # or server.run(transport="stdio")

Diagram:
    ┌──────────────────────────────────┐
    │  Client Process                  │
    │  (your Python script, Claude     │
    │   Desktop, Cursor, etc.)         │
    │                                  │
    │    spawns ──→ [ server.py ]      │
    │    stdin/stdout ←──────────────→ │
    └──────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSPORT 2: HTTP  (Streamable HTTP / SSE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

How it works:
    The server runs as a standalone web service on a port. Any number of
    clients can connect to it over the network. Server and client are
    independent processes — the server keeps running when the client disconnects.

When to use:
    ✓ Production deployments
    ✓ Multiple clients sharing one server (e.g. multiple agents, team usage)
    ✓ Server runs on a different machine (cloud, Docker, remote)
    ✓ You want to keep the server alive independent of any one client
    ✓ Composed servers (nova + brown behind one URL)

Config on the client side looks like:
    {
        "mcpServers": {
            "my_server": {
                "url": "http://localhost:8001/mcp"
            }
        }
    }

The server code calls:
    server.run(transport="http", host="0.0.0.0", port=8001)

Diagram:
    ┌─────────────┐         HTTP          ┌──────────────────┐
    │   Client A  │ ──────────────────→   │                  │
    └─────────────┘                       │   server.py      │
    ┌─────────────┐         HTTP          │   :8001          │
    │   Client B  │ ──────────────────→   │  (always on)     │
    └─────────────┘                       └──────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Local script / dev / one client?       → StdioTransport
    Production / cloud / multiple clients? → HttpTransport
    Multiple servers merged into one URL?  → HttpTransport on ComposedMCPServer
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StdioTransport:
    """
    stdio transport — client spawns this server as a subprocess.

    No configuration needed on the server side.
    Configuration lives in the CLIENT's mcp_server_config:

        {
            "mcpServers": {
                "my_server": {
                    "command": "python",
                    "args": ["server.py"],
                    "env": {"API_KEY": "..."}
                }
            }
        }

    Usage:
        server = MyServer()
        server.run(StdioTransport())    ← blocks until client disconnects
    """
    pass  # No fields — stdio has no configuration on the server side


@dataclass
class HttpTransport:
    """
    HTTP transport — server runs as a standalone web service.

    Client connects with:
        {"mcpServers": {"name": {"url": "http://<host>:<port>/mcp"}}}

    Args:
        host:  Interface to bind to.
               "127.0.0.1" → only local connections (development)
               "0.0.0.0"   → all interfaces (Docker, cloud, production)
        port:  TCP port. Convention in this course:
               8001 → Nova (research server)
               8002 → Brown (writing server)
               8003 → Composed (Nova + Brown together)
        path:  URL path for the MCP endpoint. Default "/mcp" is the convention.

    Usage:
        server = MyServer()
        server.run(HttpTransport(host="0.0.0.0", port=8001))
    """
    host: str = "127.0.0.1"
    port: int = 8000
    path: str = "/mcp"
