# Lesson 4 — MCP: Tools for Your Agent

**Unit 1: Foundations**

---

## What you will learn

- What the Model Context Protocol (MCP) is and why it exists
- How FastMCP lets you define tools in seconds
- The four things an MCP server can expose: tools, resources, prompts, memory
- How clients connect to servers (in-process, subprocess, HTTP)
- How `list_tools()` and `call_tool()` work under the hood

---

## The concept

### The problem before MCP

Before MCP, every AI framework invented its own tool system. LangChain tools. OpenAI function calling. Anthropic tool use. Each had a different format, different schema, different way to register. If you wrote a "search database" tool for one framework, you couldn't reuse it in another.

**MCP is Anthropic's open standard for tool connectivity.** It separates:
- **Tool definition** — what the tool is called, what it does, what inputs it takes
- **Tool execution** — the code that actually runs when the tool is called
- **Tool transport** — how the LLM-side code connects to the tool-side code

This separation means a Python agent can call a Node.js tool server. Your data team can publish tools without touching your agent code. Tools become a shared infrastructure layer.

### FastMCP — MCP without boilerplate

FastMCP is the Python library that makes building MCP servers ergonomic. One decorator is all you need:

```python
from fastmcp import FastMCP

app = FastMCP("my_tools")

@app.tool
async def search_database(query: str, limit: int = 10) -> str:
    """Search the product database. Returns JSON array of matching products."""
    rows = db.execute("SELECT * FROM products WHERE name LIKE ?", [f"%{query}%"])
    return json.dumps([dict(r) for r in rows[:limit]])
```

That decorator does three things automatically:
1. **Generates the JSON Schema** from the function signature (`query: str`, `limit: int`)
2. **Registers the function** as callable via MCP protocol
3. **Makes it discoverable** via `mcp.list_tools()`

The docstring becomes the tool's `description` — the LLM reads this to decide when to use the tool. **Write good docstrings. They are prompts.**

### The four capability types

**Tools** — functions the LLM can call. Input comes in, output goes back. Everything in this course uses tools.

**Resources** — read-only data the LLM can access (files, database records, documentation). Think of them as tools that only read.

**Prompts** — reusable prompt templates registered on the server, retrievable by name.

**Memory** — not a standard MCP concept but the framework extends MCP servers with memory stores (Lesson 7).

---

## Connecting: three transport modes

### In-process (development default)

```python
app = FastMCP("my_server")

@app.tool
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# Pass the FastMCP instance directly to AgentConfig
config = AgentConfig(mcp_server_config=app)
```

The agent and tool server run in the same Python process. No network. Fastest option. Perfect for development, testing, and single-process apps.

### Subprocess (stdio transport)

```python
config = AgentConfig(mcp_server_config={
    "mcpServers": {
        "my_server": {
            "command": "python",
            "args": ["tools/my_server.py"]
        }
    }
})
```

The framework spawns a subprocess running your tool server. Communication happens over stdin/stdout. The subprocess is isolated — a crash in the tool server doesn't crash the agent.

### HTTP transport (production)

```python
config = AgentConfig(mcp_server_config={
    "mcpServers": {
        "my_server": {
            "url": "http://localhost:8001/mcp"
        }
    }
})
```

The tool server runs as a standalone HTTP service. Can be on a different machine, in Docker, on Kubernetes. Multiple agents can share one tool server. This is the production architecture.

---

## `_tool_utils.py` — how the loop uses MCP

```python
# src/mcp_agent_framework/patterns/_tool_utils.py

async def list_tools(mcp: Client) -> list[MCPTool]:
    """Get available tools from the MCP server, converted to MCPTool objects."""
    raw_tools = await mcp.list_tools()
    return [
        MCPTool(
            name=t.name,
            description=t.description or "",
            input_schema=t.inputSchema or {},
        )
        for t in raw_tools
    ]

async def call_tool(mcp: Client, tool_call: ToolCall) -> str:
    """Execute a tool call and return the result as a string."""
    result = await mcp.call_tool(tool_call.name, tool_call.arguments)
    # FastMCP returns a list of content objects; extract the text
    if result and hasattr(result[0], "text"):
        return result[0].text
    return str(result)
```

`list_tools()` is called **once** at the start of each `run()`. The tool list is fixed for the duration of the run — the LLM doesn't re-query mid-loop.

`call_tool()` is called **once per tool call** the LLM requests. The result is always converted to a string before being put in a `Message`.

---

## The composed server pattern

When you have multiple MCP servers (one for web search, one for your database, one for email), you can merge them into a single endpoint:

```python
from mcp_agent_framework import build_composed_server

web_server = FastMCP("web")
db_server  = FastMCP("database")
email_server = FastMCP("email")

# The agent sees all tools from all three servers as one flat list
combined = build_composed_server([web_server, db_server, email_server])
config = AgentConfig(mcp_server_config=combined)
```

The agent doesn't know or care that tools came from different servers. It just sees a flat list of available tools.

---

## Read these files

```
src/mcp_agent_framework/server/mcp_server_base.py    ← MCPServerBase: subclass to build your own
src/mcp_agent_framework/server/transports.py         ← StdioTransport, HttpTransport
src/mcp_agent_framework/server/composed_server.py    ← build_composed_server()
src/mcp_agent_framework/patterns/_tool_utils.py      ← list_tools() and call_tool()
```

In `mcp_server_base.py`, notice that `MCPServerBase` is itself a thin wrapper around `FastMCP`. It adds a `register_tools()` method that your subclass overrides to add tools programmatically rather than with decorators.

---

## Run this

```bash
python examples/01_hello_agent.py
```

Add a print statement before and after `list_tools()` to see what the MCP server advertises. Print the tool names and their descriptions.

---

## Build this

Create a `FileSystemServer` that exposes 4 tools:

```python
app = FastMCP("filesystem")

@app.tool
async def read_file(path: str) -> str:
    """Read the contents of a file. Returns the file contents as a string."""
    ...

@app.tool
async def list_directory(path: str) -> str:
    """List files and folders in a directory. Returns newline-separated names."""
    ...

@app.tool
async def write_file(path: str, content: str) -> str:
    """Write content to a file. Returns 'OK' on success."""
    ...

@app.tool
async def file_exists(path: str) -> str:
    """Check if a file or directory exists. Returns 'true' or 'false'."""
    ...
```

Wire it to a `SingleAgentLoop` (from Lesson 5). Ask it: *"Create a file called hello.txt with 'Hello, World!' in it, then read it back and confirm the contents."*

---

## Key terms

| Term | Meaning |
|------|---------|
| MCP | Model Context Protocol — open standard for tool connectivity |
| FastMCP | Python library for building MCP servers with minimal boilerplate |
| `@app.tool` | Decorator that registers a function as an MCP tool |
| `input_schema` | JSON Schema describing the tool's arguments — the LLM reads this |
| In-process transport | FastMCP instance used directly, no network |
| Stdio transport | Tool server as subprocess, stdin/stdout communication |
| HTTP transport | Tool server as standalone service, REST communication |

---

## Connects to

- **Lesson 5** — the Single Agent Loop connects to an MCP server at the start of every `run()`
- **Lesson 6** — tool calling: the full lifecycle from tool list to tool result
- **Lesson 7** — memory is exposed as MCP tools
- **Lesson 8** — orchestrator workers each have their own MCP server
- **Lesson 20** — skills are ultimately exposed as MCP tools via `SkillAwareAgent`

---

*Lesson 4 of 20 — Applied AI Engineering*
