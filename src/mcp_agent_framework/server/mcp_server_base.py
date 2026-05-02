"""
MCPServerBase - the template for every MCP server you will ever build.

This is where your capabilities live: tools, resources, memory, and prompts.
Subclass this, register your capabilities in __init__, then call .run().

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE FOUR CAPABILITY TYPES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. TOOLS  (@mcp.tool)
   Functions the LLM can CALL. The LLM decides when to call them.
   These are actions - scrape a URL, write a file, query a database.
   The LLM provides arguments; your function executes and returns a result.

   Agent sees:  "I have a tool called scrape_url(url: str)"
   Agent does:  calls it, gets back content, continues reasoning

2. RESOURCES  (@mcp.resource)
   Data the CLIENT can READ on demand. Unlike tools, resources are not
   called by the LLM autonomously - a client explicitly requests them by URI.
   Think of them as files or API endpoints your server exposes.

   Use for: configuration, large static data, file contents, DB records.
   URI format: "scheme://path"  e.g. "file://config.json", "db://users/123"

   Client reads:  await mcp_client.read_resource("file://config.json")
   Server serves: the file contents as text or bytes

3. PROMPTS  (@mcp.prompt)
   Pre-written instruction templates stored on the server.
   Clients load them by name and inject them as the system/user message.
   Use these to store complex workflow instructions alongside the tools
   that implement them - they stay in sync.

   Client loads:  await mcp_client.get_prompt("full_workflow")
   Server returns: the full instruction text with optional arguments filled in

4. MEMORY  (in-process state + @mcp.tool or @mcp.resource)
   MCP has no built-in "memory" type. Memory is implemented as:
   a) An in-process store (dict, list, vector DB connection) held on the server
   b) Tools that write to it:  remember(key, value), add_to_history(text)
   c) Tools/resources that read from it:  recall(query), get_history()

   The server instance lives between calls, so self._memory persists
   across multiple tool invocations in the same server process.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL EXAMPLE - see bottom of this file
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from fastmcp import FastMCP

from mcp_agent_framework.server.transports import HttpTransport, StdioTransport

logger = logging.getLogger(__name__)

Transport = StdioTransport | HttpTransport


class MCPServerBase:
    """
    Base class for all MCP servers in this framework.

    Subclass it, register capabilities in __init__, call .run().

    Internally wraps FastMCP. You can always access the raw FastMCP instance
    via self.mcp if you need anything not exposed here.

    Args:
        name:    Server name passed to FastMCP.
        version: Server version string.
        context: Optional per-request context object stored as ``self.ctx``.
                 Use this for multi-tenant systems where tools need access to
                 request-scoped state (user ID, DB connection, permissions, etc.).
                 Defaults to None — existing subclasses are unaffected.

    Context example::

        @dataclass
        class RequestContext:
            user_id: str
            permissions: set[str]
            db: Database

        class CRMServer(MCPServerBase):
            def __init__(self, ctx: RequestContext):
                super().__init__("crm", context=ctx)

                @self.tool
                async def get_customer(customer_id: str) -> str:
                    if "crm:read" not in self.ctx.permissions:
                        return "Permission denied."
                    return await self.ctx.db.fetch(customer_id)
    """

    def __init__(self, name: str, version: str = "0.1.0", context: Any = None):
        self.mcp = FastMCP(name=name, version=version)
        self._name = name
        self.ctx: Any = context  # per-request context; None by default (backward compatible)

    # ──────────────────────────────────────────────────────────────────
    # 1. TOOLS  - actions the LLM calls
    # ──────────────────────────────────────────────────────────────────

    def tool(self, fn: Callable | None = None, *, description: str | None = None):
        """
        Register a function as a callable tool.

        The LLM sees the function name, description (from docstring or override),
        and parameters (from type hints). It can call this tool at any time
        during its reasoning loop.

        Usage:
            class MyServer(MCPServerBase):
                def __init__(self):
                    super().__init__("my_server")

                    @self.tool
                    def add(a: int, b: int) -> int:
                        \"\"\"Add two numbers.\"\"\"
                        return a + b

                    @self.tool(description="Fetch the contents of a URL")
                    async def fetch(url: str) -> str:
                        async with httpx.AsyncClient() as client:
                            r = await client.get(url)
                            return r.text

        Supports both sync and async functions.
        Works as @self.tool or @self.tool(description="...").
        """
        if fn is not None:
            # Used as @self.tool (no parentheses)
            return self.mcp.tool()(fn)
        else:
            # Used as @self.tool(description="...")
            return self.mcp.tool(description=description)

    # ──────────────────────────────────────────────────────────────────
    # 2. RESOURCES  - data the client reads by URI
    # ──────────────────────────────────────────────────────────────────

    def resource(self, uri: str, *, description: str | None = None, mime_type: str = "text/plain"):
        """
        Expose a piece of data as a readable resource at a URI.

        The client reads it explicitly with: await client.read_resource(uri)
        The LLM does NOT call resources automatically - a tool or the client
        code must decide to read a resource.

        URI conventions:
            "file://path/to/file"      ← file contents
            "config://settings"        ← application config
            "memory://conversation"    ← stored memory
            "db://table/record_id"     ← database record
            "api://endpoint"           ← cached API response

        Usage:
            @self.resource("config://app_settings")
            def get_config() -> str:
                return json.dumps({"model": "claude-sonnet-4-6", "max_tokens": 4096})

            @self.resource("file://article_guideline", mime_type="text/markdown")
            async def get_guideline() -> str:
                return Path("guideline.md").read_text()

        For dynamic URIs (e.g. db://users/{user_id}), use a template:
            @self.resource("db://users/{user_id}")
            async def get_user(user_id: str) -> str:
                return await db.get_user(user_id)
        """
        return self.mcp.resource(uri, description=description, mime_type=mime_type)

    # ──────────────────────────────────────────────────────────────────
    # 3. PROMPTS  - instruction templates stored on the server
    # ──────────────────────────────────────────────────────────────────

    def prompt(self, fn: Callable | None = None, *, name: str | None = None):
        """
        Register a prompt template.

        Clients load it by name and inject it as the starting instruction
        for the agent. Keeps complex workflow instructions co-located with
        the tools that implement them.

        Usage:
            @self.prompt
            def research_workflow() -> str:
                \"\"\"Full instructions for the research workflow.\"\"\"
                return \"\"\"
                Step 1: Extract all URLs from the guidelines file.
                Step 2: Scrape web URLs in parallel.
                ...
                \"\"\"

            # With a parameter:
            @self.prompt
            def article_writer(tone: str = "professional") -> str:
                return f"Write in a {tone} tone. Use short paragraphs..."

        Client loads it:
            prompt = await client.get_prompt("research_workflow")
        """
        if fn is not None:
            return self.mcp.prompt()(fn)
        else:
            return self.mcp.prompt(name=name)

    # ──────────────────────────────────────────────────────────────────
    # 4. MEMORY  - in-process state across tool calls
    # ──────────────────────────────────────────────────────────────────

    def add_memory(
        self,
        store: Any,
        *,
        write_tool_name: str = "memory_store",
        read_tool_name: str  = "memory_recall",
        resource_uri: str    = "memory://store",
    ) -> None:
        """
        Wire an in-process memory store to the server as tools + a resource.

        The store can be any object - a dict, a list, a vector DB client, etc.
        This method registers three capabilities:
          - Tool `write_tool_name`: the LLM calls to save something
          - Tool `read_tool_name`:  the LLM calls to retrieve something
          - Resource `resource_uri`: clients can read the full store directly

        For simple key-value memory, pass a dict:
            class MyServer(MCPServerBase):
                def __init__(self):
                    super().__init__("my_server")
                    self._memory: dict[str, str] = {}
                    self.add_memory(self._memory)

        For custom stores, subclass and override. The default implementation
        treats `store` as a plain dict with str keys and str values.

        For advanced memory (vector search, episodic memory, mem0, etc.)
        register your own tools manually with @self.tool instead.
        """
        import json

        mcp = self.mcp

        @mcp.tool(name=write_tool_name)
        def _write(key: str, value: str) -> str:
            """Save a value to memory under a key."""
            store[key] = value
            return f"Stored '{key}'."

        @mcp.tool(name=read_tool_name)
        def _read(key: str) -> str:
            """Recall a value from memory by key."""
            if key not in store:
                return f"No memory found for key '{key}'."
            return str(store[key])

        @mcp.resource(resource_uri)
        def _read_all() -> str:
            """Return the full memory store as JSON."""
            return json.dumps(store, indent=2, default=str)

    # ──────────────────────────────────────────────────────────────────
    # Transport & lifecycle
    # ──────────────────────────────────────────────────────────────────

    def run(self, transport: Transport | None = None) -> None:
        """
        Start the server. Blocks until the server is stopped.

        Args:
            transport: StdioTransport() or HttpTransport(host=..., port=...).
                       Defaults to StdioTransport if not provided.

        Usage:
            # stdio - client spawns this process
            server.run()
            server.run(StdioTransport())

            # HTTP - standalone service, clients connect over network
            server.run(HttpTransport(host="0.0.0.0", port=8001))
        """
        if transport is None or isinstance(transport, StdioTransport):
            logger.info("[%s] starting on stdio", self._name)
            self.mcp.run(transport="stdio")
        elif isinstance(transport, HttpTransport):
            logger.info("[%s] starting on http %s:%d%s", self._name, transport.host, transport.port, transport.path)
            self.mcp.run(transport="http", host=transport.host, port=transport.port)
        else:
            raise ValueError(f"Unknown transport: {transport}")


# ══════════════════════════════════════════════════════════════════════
# FULL WORKING EXAMPLE
# ══════════════════════════════════════════════════════════════════════
#
# Run this file directly to start the example server:
#   python mcp_server_base.py                  → starts on stdio
#   python mcp_server_base.py --http 8001      → starts on HTTP :8001
#
# Connect from any MCP client:
#   stdio: {"mcpServers": {"example": {"command": "python", "args": ["mcp_server_base.py"]}}}
#   http:  {"mcpServers": {"example": {"url": "http://localhost:8001/mcp"}}}
# ══════════════════════════════════════════════════════════════════════

class ExampleServer(MCPServerBase):
    """
    A minimal server demonstrating all four capability types.
    Drop this into any project as a starting point.
    """

    def __init__(self):
        super().__init__(name="example_server", version="0.1.0")

        # In-process memory store (persists for the life of this server process)
        self._memory: dict[str, str] = {}

        # ── TOOL ──────────────────────────────────────────────────────
        @self.tool
        def add(a: int, b: int) -> int:
            """Add two numbers and return the result."""
            return a + b

        @self.tool
        async def fetch_url(url: str) -> str:
            """Fetch the text content of a URL. Returns the raw text."""
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True, timeout=30)
                return response.text[:5000]  # first 5k chars

        # ── RESOURCE ──────────────────────────────────────────────────
        @self.resource("config://server_info")
        def server_info() -> str:
            """Current server configuration."""
            import json
            return json.dumps({"name": "example_server", "version": "0.1.0"})

        @self.resource("memory://all", mime_type="application/json")
        def read_all_memory() -> str:
            """Read the full memory store."""
            import json
            return json.dumps(self._memory, indent=2)

        # ── MEMORY ────────────────────────────────────────────────────
        # Wire the dict as LLM-callable memory tools
        self.add_memory(
            self._memory,
            write_tool_name="remember",
            read_tool_name="recall",
            resource_uri="memory://store",
        )

        # ── PROMPT ────────────────────────────────────────────────────
        @self.prompt
        def assistant_instructions() -> str:
            """System instructions for agents connecting to this server."""
            return (
                "You are a helpful assistant with access to web fetching and memory tools.\n"
                "Use 'fetch_url' to retrieve web content.\n"
                "Use 'remember' to save important facts and 'recall' to retrieve them.\n"
            )


if __name__ == "__main__":
    import sys
    server = ExampleServer()
    if "--http" in sys.argv:
        idx  = sys.argv.index("--http")
        port = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 8000
        server.run(HttpTransport(host="0.0.0.0", port=port))
    else:
        server.run(StdioTransport())
