# MCP Agent Framework
### A complete guide - from zero to production agents

> Written for Python developers who have never built an AI agent before.  
> Every file explained. Every concept demystified. No jargon without explanation.

---

## Tools & Frameworks

| Tool / Framework | Role in this project |
|---|---|
| [FastMCP](https://github.com/jlowin/fastmcp) | MCP server toolkit - define tools as plain Python functions with `@app.tool`; handles schema generation and protocol wiring |
| [MCP (Model Context Protocol)](https://modelcontextprotocol.io) | Open protocol for LLM ↔ tool communication; `fastmcp.Client` is the async client used by every agent pattern |
| [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) (`anthropic`) | Claude API client - `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5` |
| [OpenAI SDK](https://github.com/openai/openai-python) (`openai`) | GPT API client; also used for Grok (xAI), Ollama, Together AI, and any OpenAI-compatible endpoint via `base_url` |
| [Google GenAI SDK](https://github.com/googleapis/python-genai) (`google-genai`) | Gemini API client - `gemini-2.5-flash`, `gemini-2.5-pro` |
| [Pydantic](https://docs.pydantic.dev) | Structured output schema definition (`ExecutionPlan`, `ExecutionStep`) and JSON schema extraction |
| [asyncio](https://docs.python.org/3/library/asyncio.html) | Python standard library - all I/O is async; `asyncio.gather` drives parallel tool execution and fan-out patterns |
| [Python `collections.deque`](https://docs.python.org/3/library/collections.html#collections.deque) | O(1) front-eviction in `EpisodicMemory` |

---

## Table of Contents

1. [What this framework is and why it exists](#1-what-this-framework-is-and-why-it-exists)
2. [The mental map - how the layers fit together](#2-the-mental-map)
3. [Quick start - your first agent in 5 minutes](#3-quick-start)
4. [Layer 1 - Types (the shared language)](#4-layer-1--types)
5. [Layer 2 - Clients (talking to AI providers)](#5-layer-2--clients)
6. [Layer 3 - Model Registry (organise your models)](#6-layer-3--model-registry)
7. [Layer 4 - MCP Servers (what your agent can do)](#7-layer-4--mcp-servers)
8. [Layer 5 - Patterns (how agents coordinate)](#8-layer-5--patterns)
9. [Pattern decision guide - which pattern for which job](#9-pattern-decision-guide)
10. [Layer 6 - Evaluation (measuring quality)](#10-layer-6--evaluation)
11. [Layer 7 - Memory (what agents remember)](#11-layer-7--memory)
12. [Layer 8 - Resilience (surviving failures)](#12-layer-8--resilience)
13. [Layer 9 - Observability (seeing what's happening)](#13-layer-9--observability)
14. [Building a RAG system](#14-building-a-rag-system)
15. [Building an Agentic RAG system](#15-building-an-agentic-rag-system)
16. [Building a Multi-Agent Platform](#16-building-a-multi-agent-platform)
17. [File reference - every file explained](#17-file-reference)
18. [How to copy this into your own project](#18-how-to-copy-this-into-your-own-project)
19. [Skills — named, reusable agentic capabilities](#19-skills--named-reusable-agentic-capabilities)
20. [Applied AI Engineering Curriculum](#20-applied-ai-engineering-curriculum)
21. [LangGraph - how it compares](#langgraph---how-it-compares)

---

## 1. What this framework is and why it exists

### The problem this solves

You want to build an AI application. You pick Anthropic's Claude. You write code that calls the Anthropic API. Three months later, you want to try OpenAI's GPT-5 to compare. You have to rewrite your entire calling code because the APIs are completely different.

Now multiply that by: tool calling (different format per provider), structured output (different mechanism per provider), streaming (different event types per provider). You end up maintaining three completely different codebases for the same application logic.

**This framework's job:** absorb all those provider differences so your application code never knows which provider it's talking to. You write your application logic once. You swap providers with one line.

### What MCP is (in plain English)

MCP stands for **Model Context Protocol**. It is a standard way for AI models to call external tools and services.

Think of it like this: your AI model is a brain. It can think and reason, but it can't actually *do* anything in the world - it can't search the web, read files, query a database, or send emails. MCP is the protocol that lets the brain reach out and use tools that can do those things.

An **MCP server** is a program that exposes tools. Your AI agent connects to it and can call those tools.

```
Your AI Agent
     │
     │  "I need to search the web"
     ▼
MCP Server (has web_search tool)
     │
     │  calls actual search API
     ▼
Returns results back to the agent
```

### What "agentic" means

A normal AI call is: you ask a question → it answers. Done.

An **agent** is different. It loops:
1. You give it a task
2. It thinks about what tools it needs
3. It calls those tools
4. It looks at the results
5. It decides if it needs more tools or if it's done
6. Repeat until done

This loop is called **ReAct** (Reason + Act). The agent reasons about what to do, acts by calling a tool, observes the result, reasons again. This is how agents can tackle complex tasks that require multiple steps.

---

## 2. The mental map

Before diving into files, here is the complete picture of how the layers relate:

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR APPLICATION                          │
│  (imports from mcp_agent_framework and builds on top)        │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    PATTERNS                                   │
│  SingleAgentLoop  │  OrchestratorWorker  │  Hierarchy        │
│  HumanInLoop      │  EvaluatorOptimizer  │  PlannerExecutor  │
│  ParallelPattern                                             │
│                                                              │
│  "How do my agents coordinate with each other?"              │
└───────────┬─────────────────────────┬───────────────────────┘
            │                         │
┌───────────▼──────────┐   ┌──────────▼──────────────────────┐
│    MODEL REGISTRY    │   │         MCP SERVERS              │
│                      │   │                                  │
│  "Which AI model     │   │  "What can my agent DO?"         │
│   do I call?"        │   │  (tools, resources, memory)      │
└───────────┬──────────┘   └──────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────────┐
│                       CLIENTS                                 │
│  AnthropicClient  │  OpenAIClient  │  GeminiClient           │
│                                                               │
│  "Translate between framework's canonical format             │
│   and each provider's specific API format"                   │
└───────────────────────────────────────────────────────────────┘

Supporting layers (used by all of the above):
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│    MEMORY    │  │  RESILIENCE  │  │ OBSERVABILITY│  │  EVALUATION  │
│ Semantic     │  │ RetryPolicy  │  │ RunContext   │  │ LLMEvaluator │
│ Episodic     │  │ CircuitBreak │  │ LoggingTracer│  │ RubricEval   │
│ Procedural   │  │              │  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

**The key insight:** every layer only knows about the layer below it, never above it. Patterns use Clients. Clients don't know about Patterns. This means you can swap any layer without touching anything else.

---

## 3. Quick start

### Install

```bash
pip install -r requirements.txt
pip install -e .
```

### Your first agent (30 lines)

```python
import asyncio
import os
from fastmcp import FastMCP
from mcp_agent_framework import AnthropicClient, SingleAgentLoop, AgentConfig

# Step 1: Define what your agent can DO (an MCP server with tools)
app = FastMCP("my_tools")

@app.tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny and 22°C."

# Step 2: Configure the agent
config = AgentConfig(
    mcp_server_config=app,              # which tools are available
    system_prompt="You are a helpful assistant.",
    max_iterations=5,
)

# Step 3: Create the agent with a model
agent = SingleAgentLoop(
    llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
    config=config,
)

# Step 4: Run it
async def main():
    result = await agent.run("What's the weather in Paris?")
    print(result)

asyncio.run(main())
```

That's it. The agent will:
1. Receive your question
2. Decide it needs to call `get_weather`
3. Call it with `city="Paris"`
4. Read the result
5. Return a natural language answer

---

## 4. Layer 1 - Types

**File:** `src/mcp_agent_framework/types.py`

This file defines the **shared language** of the entire framework. Every layer uses these types. Think of them as the data contracts that prevent layers from being coupled to each other.

### The types explained

**`Message`** - one turn in a conversation

```python
# A user message
Message(role="user", content="What is the capital of France?")

# An assistant reply
Message(role="assistant", content="The capital of France is Paris.")

# A tool result (after the agent called a tool)
Message(role="tool", content="Paris weather: sunny 22°C", tool_call_id="call_abc123", name="get_weather")
```

The `role` can be `"user"`, `"assistant"`, or `"tool"`. Notice there's no `"system"` role in the message - system prompts are passed separately to clients because different providers handle them differently.

**`ToolCall`** - the model wants to call a tool

```python
ToolCall(
    id="call_abc123",          # unique ID - links the call to its result
    name="get_weather",        # which tool to call
    arguments={"city": "Paris"}  # what to pass it
)
```

When the model returns tool calls, you execute them and send back results with the same `id`. The `id` is how the model knows which result belongs to which call.

**`MCPTool`** - a tool definition the model can read

```python
MCPTool(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema={                    # JSON Schema describing the parameters
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
)
```

This is what gets sent to the model so it knows what tools exist and how to call them.

**`LLMResponse`** - what comes back from any model

```python
LLMResponse(
    content="The weather in Paris is sunny.",   # text response (may be None if tool calls returned)
    tool_calls=[ToolCall(...)],                 # tool calls (may be None if no tools needed)
    stop_reason=StopReason.END_TURN,            # why the model stopped
    input_tokens=150,
    output_tokens=42,
)
```

**`StopReason`** - why the model stopped generating

```python
StopReason.END_TURN    # model finished normally - has a text answer
StopReason.TOOL_USE    # model wants to call tools - no final answer yet
StopReason.MAX_TOKENS  # ran out of token budget
```

The agent loop uses `stop_reason` to know whether to continue looping or return the answer.

**`AgentConfig`** - configuration for any agent pattern

```python
AgentConfig(
    mcp_server_config=app,            # FastMCP object OR {"mcpServers": {...}} dict
    system_prompt="You are...",       # optional system prompt
    max_iterations=10,                # safety cap on the tool-call loop
    extra={},                         # pattern-specific extra config
)
```

---

## 5. Layer 2 - Clients

**Directory:** `src/mcp_agent_framework/clients/`

The clients are the layer that talks to AI providers. They translate between the framework's canonical format (using `Message`, `ToolCall`, `MCPTool`, `LLMResponse`) and each provider's specific API format.

### Why this layer exists

Every AI provider has a completely different wire format:

| Thing | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| Tool schema key | `input_schema` | `parameters` | `parameters_json_schema` |
| Stop reason for tool use | `"tool_use"` | `"tool_calls"` | *(not in finish_reason - check parts)* |
| Tool arguments type | dict | JSON string (!) | dict |
| System prompt | top-level param | role="system" message | `system_instruction` in config |
| Structured output | forced tool use | `response_format` json_schema | `response_mime_type` |

Without a client abstraction, you'd have five `if provider == "anthropic"` blocks in every file. The client layer absorbs all of this.

### `base_client.py` - the contract every client must fulfil

```python
class BaseLLMClient(ABC):
    
    @abstractmethod
    async def complete(messages, tools, system) -> LLMResponse:
        """Send messages, get a response. The core call."""
    
    @abstractmethod
    async def complete_structured(messages, response_schema, system) -> dict:
        """Send messages, get a response guaranteed to match a schema."""
    
    async def stream_complete(messages, tools, system) -> AsyncIterator[str]:
        """Stream tokens. Default falls back to complete() as one chunk."""
    
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name for logs. e.g. 'anthropic/claude-sonnet-4-6'"""
```

Every pattern in the framework only calls these methods. They never import `anthropic` or `openai` directly. This is how provider swapping works - the patterns don't know or care which provider they're using.

### `anthropic_client.py` - Claude

```python
client = AnthropicClient(
    model="claude-sonnet-4-6",    # default
    api_key="sk-...",             # reads ANTHROPIC_API_KEY from env if not passed
    max_tokens=8096,
    temperature=0.7,              # any extra kwargs forwarded to the API
)
```

**Structured output:** Uses forced tool use. A synthetic tool named `"structured_output"` is created with your schema as its `input_schema`. The model is forced to call it exactly once via `tool_choice={"type": "tool", "name": "structured_output"}`. The tool call arguments *are* the structured output - already a dict, no JSON parsing needed.

### `openai_client.py` - GPT, Grok, Ollama, Together

```python
# OpenAI
client = OpenAIClient("gpt-5.4")

# Grok (same API, different base_url)
client = OpenAIClient("grok-3", base_url="https://api.x.ai/v1", api_key=os.environ["XAI_KEY"])

# Ollama (local - no API key needed)
client = OpenAIClient("llama3", api_key="ollama", base_url="http://localhost:11434/v1")

# Together AI
client = OpenAIClient("mistral", api_key=os.environ["TOGETHER_KEY"], base_url="https://api.together.xyz/v1")
```

**Structured output:** Uses `response_format={"type": "json_schema", "json_schema": {"name": "response", "schema": ..., "strict": True}}`. The model is constrained by the schema at the decoding level - it physically cannot produce invalid JSON.

### `gemini_client.py` - Google Gemini

```python
client = GeminiClient(
    model="gemini-2.5-flash",    # default
    api_key="...",               # reads GOOGLE_API_KEY from env if not passed
)
```

**Important April 2026 detail:** uses `parameters_json_schema=tool.input_schema` in `FunctionDeclaration` - passing the JSON Schema dict directly. The old `types.Schema` conversion is gone.

**Structured output:** Uses `GenerateContentConfig(response_mime_type="application/json", response_schema=schema)`. Cannot be combined with tools in the same call.

### `schema_utils.py` - shared schema helper

```python
from mcp_agent_framework import schema_from

# Pydantic v2 model → JSON Schema dict
schema = schema_from(MyPydanticModel)

# Pydantic v1 model → JSON Schema dict  
schema = schema_from(MyV1Model)

# Plain dict → returned as-is
schema = schema_from({"type": "object", "properties": {...}})
```

This utility is used by all three clients' `complete_structured()` methods. You can also use it directly.

### Structured output - the full picture

```python
from pydantic import BaseModel
from mcp_agent_framework import AnthropicClient, Message

class Article(BaseModel):
    title: str
    summary: str
    tags: list[str]

client = AnthropicClient()
messages = [Message(role="user", content="Write a short article about Python.")]

# Returns a plain dict - identical result regardless of provider
result = await client.complete_structured(messages, Article)

# Parse into your model
article = Article(**result)
print(article.title)
```

Switch `AnthropicClient()` to `OpenAIClient()` or `GeminiClient()` - the call site doesn't change. The mechanism underneath changes (forced tool use vs json_schema vs MIME type), but the result is the same dict.

---

## 6. Layer 3 - Model Registry

**File:** `src/mcp_agent_framework/registry/model_registry.py`

The Model Registry is the answer to: "My application uses 5 different models for different tasks. How do I organise them?"

### The idea

Register all your models at startup with human-readable names. Call them anywhere by name. The name is the only thing your application code knows about - not the provider, not the model version.

```python
from mcp_agent_framework import ModelRegistry, AnthropicClient, OpenAIClient, GeminiClient

registry = ModelRegistry()
registry.register("fast",  AnthropicClient("claude-haiku-4-5-20251001"), tags=["cheap", "fast"])
registry.register("smart", AnthropicClient("claude-sonnet-4-6"),         tags=["balanced"])
registry.register("best",  AnthropicClient("claude-opus-4-6"),           tags=["powerful"])
registry.register("local", OpenAIClient("llama3", base_url="http://localhost:11434/v1"))
registry.register("flash", GeminiClient("gemini-2.5-flash"),             tags=["cheap"])
```

Now when you need a model:

```python
# Call by name
response = await registry.complete("smart", messages)

# Find cheapest models for bulk tasks
cheap_models = registry.find_by_tag("cheap")  # ["fast", "flash"]

# Hot-swap without restarting - useful for A/B testing or upgrades
registry.swap("smart", AnthropicClient("claude-opus-4-6"))
```

### `auto_execute=False` (default) - you manage tool calls

```python
response = await registry.complete("smart", messages, tools=tools)

if response.has_tool_calls:
    # You are in control - decide what to execute and when
    messages.append(response.to_message())   # append assistant's turn
    for tc in response.tool_calls:
        result = await mcp.call_tool(tc.name, tc.arguments)
        messages.append(tc.to_tool_result_message(result))
    # Call again with updated messages
    response = await registry.complete("smart", messages, tools=tools)

print(response.text)
```

**Why `False` is the default:** It gives you full visibility and control. You see every tool call before it executes. You can log it, validate it, rate-limit it, or modify it. Invisible auto-execution is fine for prototypes; production systems benefit from explicit control.

### `auto_execute=True` - registry manages the loop

```python
async def execute_tool(name: str, args: dict) -> str:
    return await mcp.call_tool(name, args)

response = await registry.complete(
    "smart", messages, tools=tools,
    auto_execute=True,
    tool_executor=execute_tool,
    max_iterations=10,
)
print(response.text)   # guaranteed to be the final text answer
```

The registry runs the full tool-call loop internally. Tools from a single LLM response are executed in **parallel** (`asyncio.gather`) - this matters because the model often returns multiple tool calls at once and they're independent.

### Structured output via the registry

```python
result = await registry.complete_structured("smart", messages, MyModel)
# result.data  - plain dict
# result.model_name  - which model was used
obj = MyModel(**result.data)
```

### Response types

```python
response: ModelResponse
response.text            # final text (empty string if None)
response.content         # raw string or None
response.tool_calls      # list[ToolCall] or None
response.has_tool_calls  # bool convenience property
response.stop_reason     # StopReason enum
response.usage           # UsageInfo(input_tokens, output_tokens, total_tokens)
response.to_message()    # convert to Message for appending to history

result: StructuredModelResponse
result.data              # plain dict matching your schema
result.model_name        # which model produced this
```

---

## 7. Layer 4 - MCP Servers

**Directory:** `src/mcp_agent_framework/server/`

MCP Servers are where you define **what your agent can do**. Tools, resources, prompts, and memory all live here.

### `mcp_server_base.py` - building a server

```python
from mcp_agent_framework import MCPServerBase

class MyServer(MCPServerBase):
    def __init__(self):
        super().__init__("my_server")
        self._db = {}   # in-process state
        
        # Register a tool - the LLM can call this
        @self.tool
        def search_database(query: str) -> str:
            """Search the database for records matching the query."""
            results = [v for k, v in self._db.items() if query.lower() in k.lower()]
            return "\n".join(results) if results else "No results found."
        
        # Register a resource - clients read this on demand (not via LLM)
        @self.resource("db://stats")
        def database_stats() -> str:
            return f"Database has {len(self._db)} records."
        
        # Register a prompt - instruction template stored on the server
        @self.prompt
        def search_instructions() -> str:
            return "When searching, try multiple query variations if the first returns no results."
```

### The four capability types

**Tools** - functions the LLM can call to get information or take actions.

```python
@self.tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    # actually send it...
    return "Email sent successfully."
```

The docstring becomes the tool description the LLM reads. Well-written docstrings directly improve agent performance.

**Resources** - data endpoints that clients read explicitly (not through the LLM).

```python
@self.resource("config://settings")
def get_settings() -> str:
    return json.dumps({"max_results": 10, "language": "en"})
```

Use resources for configuration, documentation, and data that the client should fetch once - not for things the LLM needs to call repeatedly.

**Prompts** - instruction templates stored on the server.

```python
@self.prompt
def analysis_instructions() -> str:
    return """When analysing data:
    1. First check for null values
    2. Look for outliers
    3. Report summary statistics"""
```

Clients can load these by name and inject them into their system prompts.

**Memory** - in-process state that persists between tool calls within a session.

```python
@self.add_memory
def _setup_memory():
    # The MCPServerBase.add_memory() method wires a dict as key-value memory tools
    # (remember/recall/forget tools are automatically registered)
    pass
```

### `transports.py` - how clients connect to servers

**StdioTransport** - client spawns server as a subprocess.

```python
from mcp_agent_framework import StdioTransport

config = AgentConfig(
    mcp_server_config={
        "mcpServers": {
            "my_server": {
                "command": "python",
                "args": ["path/to/my_server.py"],
            }
        }
    }
)
```

Use when: developing locally, building desktop apps, when each client needs its own isolated server process.

**HttpTransport** - server runs independently, clients connect over HTTP.

```python
# Start the server separately:
server = MyServer()
server.run(HttpTransport(host="0.0.0.0", port=8001))

# Connect from your agent:
config = AgentConfig(
    mcp_server_config={
        "mcpServers": {
            "my_server": {"url": "http://localhost:8001/mcp"}
        }
    }
)
```

Use when: production deployments, multiple clients sharing one server, horizontal scaling.

**In-process (for examples and testing)** - pass a `FastMCP` object directly.

```python
from fastmcp import FastMCP
app = FastMCP("tools")

@app.tool
def my_tool(x: str) -> str: ...

config = AgentConfig(mcp_server_config=app)   # no network required
```

### `composed_server.py` - merge multiple servers

```python
from mcp_agent_framework import build_composed_server

# Merge search_server and database_server into one endpoint
# The agent sees all tools from both as if they were one server
composed = build_composed_server([search_server, database_server], name="combined")
composed.run(HttpTransport(port=8001))
```

### Context-aware servers — per-request state

Standard MCP servers are singletons. All requests share the same instance, which means tools can't safely access per-request state: the current user, their permissions, their database connection.

The fix: pass `context` to `MCPServerBase` and create a fresh instance per request. Tools close over `self.ctx` naturally.

```python
from dataclasses import dataclass
from mcp_agent_framework.server import MCPServerBase

@dataclass
class RequestContext:
    user_id:     str
    permissions: set[str]
    db:          object   # per-request DB connection

class CRMServer(MCPServerBase):
    def __init__(self, ctx: RequestContext):
        super().__init__("crm", context=ctx)

        @self.tool
        async def get_customer(customer_id: str) -> str:
            """Get customer record by ID."""
            if "crm:read" not in self.ctx.permissions:
                return "Permission denied."
            return await self.ctx.db.fetch(customer_id)

        @self.tool
        async def delete_customer(customer_id: str) -> str:
            """Delete a customer record."""
            if "crm:admin" not in self.ctx.permissions:
                return "Permission denied. Admin access required."
            await self.ctx.db.delete(customer_id)
            return f"Customer {customer_id} deleted."

# Per-request — fresh instance, fresh context, no shared state
async def handle_request(user_id: str, message: str) -> str:
    ctx = RequestContext(
        user_id=user_id,
        permissions=await load_permissions(user_id),
        db=await db_pool.acquire(),
    )
    server = CRMServer(ctx)
    agent = SingleAgentLoop(
        llm_client=AnthropicClient(),
        config=AgentConfig(mcp_server_config=server.mcp),
    )
    return await agent.run(message)
```

The agent sees the same tools regardless of who is calling — but each tool enforces the caller's permissions via `self.ctx`. No global state, no thread-locals, no leakage between requests.

See `examples/context_aware_server.py` for a full working demo.

---

## 8. Layer 5 - Patterns

**Directory:** `src/mcp_agent_framework/patterns/`

Patterns are reusable agent architectures. Every pattern has the same interface:

```python
result: str = await pattern.run(user_message, history=optional_prior_messages)
```

### Pattern 1: `single_agent_loop.py` - the foundation

One LLM. One MCP server. Loop until done.

```
user message → LLM thinks → calls tools → sees results → thinks again → done
```

This is the ReAct pattern. Every other pattern is built on top of this.

```python
agent = SingleAgentLoop(
    llm_client=AnthropicClient("claude-sonnet-4-6"),
    config=AgentConfig(mcp_server_config=app, system_prompt="You are helpful."),
)
result = await agent.run("Summarise the files in /tmp")
```

**When to use:** Any task where one model + one set of tools is sufficient.

### Pattern 2: `orchestration_pattern.py` - one brain, many hands

One orchestrator LLM that can see tools from multiple specialised worker servers.

```
                    ┌─── search_worker (web search tools)
orchestrator LLM ───┤─── database_worker (SQL tools)
                    └─── file_worker (file system tools)
```

```python
from mcp_agent_framework import OrchestratorWorkerPattern, WorkerConfig

agent = OrchestratorWorkerPattern(
    llm_client=AnthropicClient("claude-sonnet-4-6"),
    config=AgentConfig(mcp_server_config={}, system_prompt="You coordinate specialists."),
    workers=[
        WorkerConfig("search",   {"mcpServers": {"search":   {"url": "http://search-service/mcp"}}}),
        WorkerConfig("database", {"mcpServers": {"database": {"url": "http://db-service/mcp"}}}),
    ]
)
result = await agent.run("Find all customers who bought product X and their order history")
```

The orchestrator sees all tools from all workers. It routes tool calls to the correct worker internally - you don't manage routing manually.

**When to use:** Tasks that need tools from different domains. Research + writing. Analytics + reporting. The orchestrator decides which tools to call; the workers provide the capabilities.

### Pattern 3: `hierarchy_pattern.py` - agents calling agents

A parent agent treats child agents as tools. The parent calls `call_agent__<name>(task=...)` and the child runs its own full agent loop.

```
parent agent
   │
   ├── call_agent__researcher("Find latest papers on X")
   │        └── researcher agent (its own loop + tools)
   │                 └── returns synthesised research
   │
   └── call_agent__writer("Write a report based on: {research}")
            └── writer agent (its own loop + tools)
                     └── returns formatted report
```

```python
from mcp_agent_framework import HierarchicalAgentPattern, ChildAgentConfig

agent = HierarchicalAgentPattern(
    llm_client=AnthropicClient("claude-opus-4-6"),
    config=AgentConfig(mcp_server_config={}, system_prompt="You delegate to specialists."),
    children=[
        ChildAgentConfig("researcher", "Does deep research on topics", researcher_agent),
        ChildAgentConfig("writer",     "Writes polished content",      writer_agent),
    ]
)
```

**When to use:** Complex tasks with distinct sub-tasks, each requiring their own reasoning and tool access. When one flat agent loop becomes too complex. When you want different models for different roles.

### Pattern 4: `human_in_loop_pattern.py` - with approval gates

Wraps any agent with a human checkpoint before specified tools execute.

```python
agent = HumanInLoopPattern(
    llm_client=AnthropicClient("claude-sonnet-4-6"),
    config=config,
    requires_approval={"delete_file", "send_email", "execute_sql"},
    # requires_approval=None means ALL tools require approval
    # requires_approval=set() means NO tools require approval (same as SingleAgentLoop)
)
```

The default approval handler is CLI-based (prints tool + args, waits for `y/n`). For web apps, provide a custom async callback:

```python
async def my_approval_callback(tool_name: str, arguments: dict) -> bool | str | dict:
    # Return True to approve
    # Return False or "reason" to reject
    # Return {"modified_arg": "new_value"} to approve with modified arguments
    await notify_slack(f"Agent wants to call {tool_name} with {arguments}")
    return await wait_for_human_decision()

agent = HumanInLoopPattern(..., approval_callback=my_approval_callback)
```

**When to use:** Any agent that takes irreversible actions (deleting data, sending messages, making purchases, deploying code).

### Pattern 5: `evaluator_optimizer_pattern.py` - generate → evaluate → rewrite

The generator produces content. The evaluator scores it. If the score is below threshold, feedback is sent back and the generator rewrites. Repeats until it passes or hits `max_rounds`.

```python
from mcp_agent_framework import EvaluatorOptimizerPattern, LLMEvaluator

evaluator = LLMEvaluator(
    llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
    pass_threshold=0.8,   # 80% score required
)

agent = EvaluatorOptimizerPattern(
    generator_client=AnthropicClient("claude-sonnet-4-6"),
    evaluator=evaluator,
    config=AgentConfig(mcp_server_config=app, system_prompt="You are a technical writer."),
    max_rounds=3,
)
result = await agent.run("Write a blog post about async Python for beginners.")
```

**Why use separate models for generation and evaluation?** The evaluator can be a cheaper model. Evaluation is a simpler task (scoring + feedback) than generation. Using `claude-haiku` for evaluation and `claude-sonnet` for generation cuts evaluation cost by ~10x.

**When to use:** Any task where quality is important and "first draft" is rarely good enough. Blog posts, reports, code, documentation.

### Pattern 6: `planner_executor_pattern.py` - plan then execute

The planner generates a structured plan (list of steps). Each step is executed independently. On failure, the planner generates a revised plan for the remaining steps.

```python
from mcp_agent_framework import PlannerExecutorPattern

agent = PlannerExecutorPattern(
    planner_client=AnthropicClient("claude-opus-4-6"),   # powerful for planning
    executor_client=AnthropicClient("claude-haiku-4-5-20251001"),  # cheap for execution
    config=config,
    max_replan_attempts=2,
)
result = await agent.run("Research quantum computing and write a technical summary.")
```

Under the hood:
1. `complete_structured(messages, ExecutionPlan)` → a structured list of steps (using Pydantic)
2. Each step runs as a separate `SingleAgentLoop.run()` - with prior step results as context
3. If a step fails → `complete_structured(replan_prompt, ExecutionPlan)` → revised remaining steps
4. All step results are synthesised into a final answer

**Why structured plan?** The plan is a Pydantic model, not free text. Every step has `step_number`, `description`, and optional `tool_hint`. This guarantees the planning output is parseable and the executor always knows what to do next.

**When to use:** Long, multi-step tasks. Research pipelines. Content creation workflows. Any task where you'd naturally write out steps on paper before starting.

### Pattern 7: `parallel_pattern.py` - fan-out and synthesise

Multiple agents run the same or different subtasks concurrently. All results are synthesised by a dedicated model.

```python
from mcp_agent_framework import ParallelPattern, ParallelTask, SingleAgentLoop

tasks = [
    ParallelTask("python",  "Research Python's strengths in 2026",  SingleAgentLoop(client, config)),
    ParallelTask("go",      "Research Go's strengths in 2026",      SingleAgentLoop(client, config)),
    ParallelTask("rust",    "Research Rust's strengths in 2026",    SingleAgentLoop(client, config)),
]

agent = ParallelPattern(
    synthesiser_client=AnthropicClient("claude-sonnet-4-6"),
    tasks=tasks,
    config=synth_config,
)
result = await agent.run("Which language should I learn first for systems programming?")
```

Each `ParallelTask` runs its `SingleAgentLoop` via `asyncio.gather` - all three run simultaneously. One task failing doesn't cancel the others. The synthesiser receives all results (with errors clearly labelled) and produces a coherent final answer.

**When to use:** Research requiring multiple independent sources. Comparative analysis. Batch processing where subtasks are independent. Anywhere sequential execution wastes time.

---

## 9. Pattern decision guide

Reading about seven patterns is one thing. Knowing which one to reach for when you sit down to build something is another. This section gives you the decision logic, a quick-reference table, and a set of real use cases mapped to patterns.

---

### The decision flowchart

```
Your task arrives.
        │
        ▼
Does it need a human to approve
anything before it acts?
  YES → HumanInLoopPattern
  NO  → continue
        │
        ▼
Does it naturally break into
independent subtasks that can
run at the same time?
  YES → ParallelPattern  (or ParallelPattern + one of the below per task)
  NO  → continue
        │
        ▼
Does it require a multi-step plan
where each step depends on the
results of the previous one?
  YES → PlannerExecutorPattern
  NO  → continue
        │
        ▼
Does output quality matter enough
to evaluate and rewrite?
  YES → EvaluatorOptimizerPattern
  NO  → continue
        │
        ▼
Does the task need multiple
specialised agents - each with
their own reasoning + tools?
  YES → HierarchicalAgentPattern
  NO  → continue
        │
        ▼
Does the task need tools from
multiple separate domains/servers
but one model coordinates?
  YES → OrchestratorWorkerPattern
  NO  → SingleAgentLoop
```

---

### Quick-reference table

| Pattern | Core idea | Cost | Complexity | Latency |
|---|---|---|---|---|
| **SingleAgentLoop** | One LLM + one server | Low | Lowest | Lowest |
| **OrchestratorWorker** | One LLM + many tool servers | Medium | Low | Low |
| **Hierarchy** | Agents calling agents | High | High | Medium |
| **HumanInLoop** | Agent + approval gate | Low + human time | Low | Depends on human |
| **EvaluatorOptimizer** | Write → score → rewrite | Medium (N rounds) | Low | Medium |
| **PlannerExecutor** | Plan first, execute per step | Medium | Medium | Medium |
| **Parallel** | N agents simultaneously | N × per-agent cost | Medium | ~1 agent's time |

---

### Pattern 1: `SingleAgentLoop` - use cases

**The rule of thumb:** if you can describe the task in one sentence and it fits in one context window, use this.

**Use case 1 - Customer support bot**
```
Task: "Answer customer questions about our product using the knowledge base."
Tools: search_kb(query), get_product_specs(sku), check_order_status(order_id)
Why this pattern: one set of tools, one domain, one answer per question.
```

**Use case 2 - Code reviewer**
```
Task: "Review this pull request and flag issues."
Tools: read_file(path), list_files(directory), get_git_diff(pr_id)
Why this pattern: reads files, reasons about code, returns a report. No sub-delegation needed.
```

**Use case 3 - Data extraction**
```
Task: "Extract all invoice line items from this PDF."
Tools: read_pdf(path), parse_table(content)
Why this pattern: a single focused extraction task. combine with complete_structured() for typed output.
```

**Use case 4 - SQL assistant**
```
Task: "How many orders were placed in Q1 2026 by enterprise customers?"
Tools: run_sql(query), list_tables(), describe_table(name)
Why this pattern: the model writes SQL, runs it, reads results, refines if needed. Self-contained.
```

**Use case 5 - Personal assistant / CLI tool**
```
Task: "Summarise my emails from today and draft replies to the urgent ones."
Tools: list_emails(date), read_email(id), draft_reply(id, body), get_calendar(date)
Why this pattern: one person, one assistant, one session. No coordination overhead needed.
```

---

### Pattern 2: `OrchestratorWorkerPattern` - use cases

**The rule of thumb:** when tools live in separate services/domains but one LLM can coordinate them all.

**Use case 1 - Full-stack developer agent**
```
Task: "Add a new 'dark mode' feature to the application."
Workers:
  - frontend_worker: read/write React components
  - backend_worker: read/write Python API endpoints
  - database_worker: read/write migration files
  - git_worker: create branch, commit, open PR
Why this pattern: one orchestrator LLM drives changes across four separate codebases/tools.
The orchestrator decides the order; the workers provide the capabilities.
```

**Use case 2 - Market intelligence pipeline**
```
Task: "Give me a competitive analysis of our top 3 competitors this week."
Workers:
  - web_worker: scrape public pages, news articles
  - social_worker: fetch Twitter/LinkedIn mentions
  - internal_worker: query internal CRM and sales data
  - analytics_worker: run statistical analysis functions
Why this pattern: data lives in completely separate systems. One analyst LLM coordinates all of them.
```

**Use case 3 - DevOps automation agent**
```
Task: "Deploy version 2.4.1 to production and verify it's healthy."
Workers:
  - ci_worker: trigger GitHub Actions, check status
  - kubernetes_worker: apply manifests, check pod health
  - monitoring_worker: check Datadog metrics, query logs
  - notifications_worker: post to Slack, update status page
Why this pattern: deployment spans 4 separate toolchains. One orchestrator makes the decisions.
```

**Use case 4 - E-commerce order agent**
```
Task: "Process order #9821 - check inventory, reserve stock, charge card, send confirmation."
Workers:
  - inventory_worker: check and reserve stock
  - payment_worker: charge card, issue receipt
  - email_worker: send transactional emails
  - erp_worker: create order record in ERP
Why this pattern: each step hits a different system. The orchestrator ensures the right sequence.
```

---

### Pattern 3: `HierarchicalAgentPattern` - use cases

**The rule of thumb:** when sub-tasks are complex enough to need their own multi-step reasoning loop - not just a tool call.

**Use case 1 - Research and publishing pipeline**
```
Task: "Write and publish a technical deep-dive on Rust's ownership model."
Children:
  - researcher: runs its own loop searching papers, docs, blog posts, saving findings
  - outline_writer: reads findings, produces structured outline with section headings
  - section_writer: writes each section (loops over sections, may do additional research)
  - editor: reviews full draft, produces revision notes
  - publisher: formats for the CMS, uploads, schedules
Why hierarchy (not orchestrator): each child is a full reasoning loop, not a single tool call.
The researcher may do 10 tool calls. The writer may do 5 rewrites. Each is its own agent.
```

**Use case 2 - Software architecture review**
```
Task: "Review our microservices architecture and produce a modernisation plan."
Children:
  - code_analyst: reads all service codebases, catalogues patterns and anti-patterns
  - dependency_mapper: traces service-to-service calls, builds dependency graph
  - security_reviewer: checks auth flows, data handling, secret management
  - report_writer: synthesises all findings into an executive report
Why hierarchy: each reviewer needs to read dozens of files and reason across them.
A single flat agent would have too many tools and too much context to manage.
```

**Use case 3 - Automated due diligence**
```
Task: "Perform technical due diligence on this startup's codebase before acquisition."
Children:
  - architecture_agent: reviews system design, scalability
  - security_agent: scans for vulnerabilities, checks compliance
  - code_quality_agent: assesses test coverage, tech debt, documentation
  - team_analysis_agent: reviews commit history, contributor patterns
  - risk_summariser: aggregates all findings into a risk-rated report
Why hierarchy: due diligence is genuinely a team of specialists. Each sub-domain is its own discipline.
```

**Use case 4 - Agentic tutoring system**
```
Task: "Teach me Python async programming."
Children:
  - curriculum_agent: assesses student level, designs a personalised learning path
  - lesson_agent: delivers one lesson at a time with examples and exercises
  - quiz_agent: generates and evaluates quiz questions, tracks understanding
  - feedback_agent: identifies gaps from quiz results, adjusts the curriculum
Why hierarchy: each child runs a multi-turn interaction loop. The parent coordinates the experience.
```

---

### Pattern 4: `HumanInLoopPattern` - use cases

**The rule of thumb:** any action that is irreversible, affects other people, moves money, or requires accountability.

**Use case 1 - Automated email campaigns**
```
Task: "Draft and send the weekly newsletter to 50,000 subscribers."
Approval required for: send_email_campaign(), update_subscriber_list()
Not required for: get_analytics(), draft_content(), check_unsubscribe_list()
Why: sending to 50,000 people is irreversible. One bad draft is a PR disaster.
```

**Use case 2 - Financial operations agent**
```
Task: "Process the end-of-month vendor payments."
Approval required for: initiate_wire_transfer(), approve_payment(), void_invoice()
Not required for: list_pending_invoices(), get_vendor_details(), calculate_totals()
Why: money leaving the company requires human sign-off. Reads are fine; writes need approval.
```

**Use case 3 - Infrastructure management**
```
Task: "Investigate the production latency spike and fix it."
Approval required for: scale_down_service(), restart_pods(), rollback_deployment()
Not required for: read_logs(), query_metrics(), describe_deployment()
Why: destructive infrastructure changes can cause outages. Investigation is safe; remediation is not.
```

**Use case 4 - HR automation**
```
Task: "Process this week's new hire onboarding requests."
Approval required for: create_employee_record(), grant_system_access(), send_offer_letter()
Not required for: check_application_status(), look_up_role_requirements()
Why: HR actions have legal and personal implications. The agent prepares; a human confirms.
```

**Use case 5 - Content moderation**
```
Task: "Review flagged content and take appropriate action."
Approval required for: ban_user(), delete_content(), escalate_to_legal()
Not required for: read_flagged_items(), check_user_history(), classify_violation_type()
Why: banning users and legal escalations need human judgement. Classification is safe to automate.
```

---

### Pattern 5: `EvaluatorOptimizerPattern` - use cases

**The rule of thumb:** when "pretty good" isn't good enough and you can define what "good" means.

**Use case 1 - Marketing copy generation**
```
Task: "Write a product description for our new standing desk."
Evaluator criteria: persuasiveness, benefit clarity, SEO keyword inclusion, tone of voice match
pass_threshold: 0.85
Why: marketing copy goes in front of customers. A/B testing is expensive. Get it right first.
```

**Use case 2 - API documentation writer**
```
Task: "Document the /payments endpoint for our REST API."
Evaluator criteria: technical accuracy, completeness (all params documented), code example quality
checker_fns: {"has_code_example": lambda t: "```" in t, "mentions_auth": lambda t: "Authorization" in t}
Why: bad docs cost support hours. Mechanical checks catch obvious gaps; LLM checks nuance.
```

**Use case 3 - Legal document summariser**
```
Task: "Summarise this 80-page contract for a non-legal audience."
Evaluator criteria: key obligations captured, risk flags identified, plain language used, no legalese
pass_threshold: 0.9  (high threshold - this is legal, mistakes matter)
Why: if the summary misses a key clause, someone signs something they didn't understand.
```

**Use case 4 - Interview question generator**
```
Task: "Create 10 technical interview questions for a senior Python engineer role."
Evaluator criteria: appropriate difficulty level, tests real skills, not trivia, includes expected answer
checker_fns: {"has_ten_questions": lambda t: t.count("?") >= 10}
Why: bad interview questions waste everyone's time and produce false signals.
```

**Use case 5 - Data analysis report**
```
Task: "Analyse Q1 sales data and write an executive summary."
Evaluator criteria: data-backed claims, no unsupported assertions, actionable recommendations
pass_threshold: 0.8
Why: executives make decisions based on these reports. Vague or unsupported claims cause bad decisions.
```

---

### Pattern 6: `PlannerExecutorPattern` - use cases

**The rule of thumb:** when the task has natural sequential steps where each step's output feeds the next, and you can't know all the steps upfront.

**Use case 1 - Full software feature implementation**
```
Task: "Implement user authentication with JWT tokens for our FastAPI app."
Plan generated: [
  1. Read existing auth code to understand current state
  2. Design the JWT schema and token lifecycle
  3. Implement token generation endpoint
  4. Implement token validation middleware
  5. Add refresh token endpoint
  6. Write tests for all auth flows
  7. Update API documentation
]
Why: you can't write tests before the implementation. Steps genuinely depend on each other.
Replanning: if step 3 hits an unexpected constraint (e.g. existing session system), replan steps 4–7.
```

**Use case 2 - Competitive research report**
```
Task: "Produce a 2026 competitive landscape report for our SaaS pricing tool."
Plan generated: [
  1. Identify top 5 competitors from search
  2. For each competitor: scrape pricing pages, feature lists, recent announcements
  3. Analyse pricing strategies and patterns
  4. Map feature parity and gaps
  5. Write executive summary with strategic recommendations
]
Why: you need competitor names (step 1) before you can research them (step 2).
Each step contextualises the next.
```

**Use case 3 - Database migration**
```
Task: "Migrate our user table from Postgres to the new schema."
Plan generated: [
  1. Read current schema and all dependent queries
  2. Design new schema with migration mapping
  3. Write migration script
  4. Run migration on staging, verify row counts
  5. Run on production during low-traffic window
  6. Verify data integrity post-migration
  7. Update application code to use new schema
]
Why: migrations are strictly ordered. Replanning handles unexpected row count mismatches (step 4 fails → replan steps 5–7 to add a data repair step).
```

**Use case 4 - Onboarding a new team member**
```
Task: "Onboard Jamie as a new backend engineer - set up all access and send orientation."
Plan generated: [
  1. Create GitHub account and add to org
  2. Create Slack account and add to channels
  3. Create AWS IAM user with appropriate policies
  4. Set up local dev environment guide based on current stack
  5. Schedule intro meetings with relevant team members
  6. Send welcome email with all credentials and links
]
Why: step 6 depends on all other steps completing. If step 3 fails (IAM limits), replan to use a different access method.
```

---

### Pattern 7: `ParallelPattern` - use cases

**The rule of thumb:** when you have N independent questions that each need their own research, and waiting for them sequentially wastes time.

**Use case 1 - Multi-market news briefing**
```
Task: "Give me a morning briefing covering tech, finance, and geopolitics."
Parallel tasks:
  - tech_agent:       "Summarise top tech news from the last 24 hours"
  - finance_agent:    "Summarise key market movements and economic news"
  - geopolitics_agent: "Summarise major geopolitical developments"
Result: synthesiser combines three independent summaries into one briefing.
Time saved: 3 agents × 15 seconds each = 45 seconds sequential → 15 seconds parallel.
```

**Use case 2 - Multi-language product localisation**
```
Task: "Translate and localise our new product announcement for US, Germany, and Japan."
Parallel tasks:
  - en_us_agent: translates + adapts for US market tone and references
  - de_agent:    translates + adapts for German market (formal tone, different regulations)
  - ja_agent:    translates + adapts for Japanese market (cultural nuances, honorifics)
Result: synthesiser does a final consistency check across all three.
Why parallel: each localisation is independent. No reason to wait.
```

**Use case 3 - Security audit**
```
Task: "Audit our application for security vulnerabilities."
Parallel tasks:
  - auth_agent:       "Audit authentication and authorisation flows"
  - injection_agent:  "Check for SQL injection and XSS vulnerabilities"
  - secrets_agent:    "Scan for hardcoded secrets and exposed credentials"
  - deps_agent:       "Check for vulnerable dependencies (CVE database)"
  - infra_agent:      "Review infrastructure configuration for misconfigurations"
Result: synthesiser produces a unified risk-rated security report.
Time saved: 5 independent audits running simultaneously.
```

**Use case 4 - Candidate screening**
```
Task: "Screen the 20 applicants for the Senior Engineer role."
Parallel tasks: (one agent per applicant)
  - applicant_1_agent: "Review CV, GitHub, and cover letter for applicant 1"
  - applicant_2_agent: "Review CV, GitHub, and cover letter for applicant 2"
  ... × 20
Result: synthesiser ranks candidates and identifies top 5 for interviews.
Time saved: 20 sequential reviews (20 × 30s = 10min) → ~30s in parallel.
```

**Use case 5 - Multi-source due diligence**
```
Task: "Should we acquire this startup? Check their tech, market, team, and financials."
Parallel tasks:
  - tech_agent:     "Analyse their GitHub, architecture, tech stack quality"
  - market_agent:   "Research market size, competition, customer reviews"
  - team_agent:     "Research founders' backgrounds, team experience, Glassdoor"
  - financial_agent: "Analyse their public financial data, funding history, burn rate"
Result: synthesiser produces a go/no-go recommendation with evidence.
Each dimension is independent - run them all at once.
```

---

### Combining patterns - real-world recipes

Most non-trivial applications combine patterns. Here are the most common combinations:

**Recipe 1: Safe agentic workflow**
```
HumanInLoop( SingleAgentLoop )
→ Agent proposes actions, human approves before execution.
Used for: any agent that writes to production systems.
```

**Recipe 2: High-quality content pipeline**
```
EvaluatorOptimizer( SingleAgentLoop with writing tools )
→ Agent writes, evaluator scores, agent rewrites until it passes.
Used for: blog posts, reports, marketing copy, documentation.
```

**Recipe 3: Research + publish**
```
PlannerExecutor where each step is a SingleAgentLoop
→ Planner breaks research into steps; each step is an independent agent run.
Used for: multi-source research, due diligence, competitive analysis.
```

**Recipe 4: Parallel research + synthesis**
```
ParallelPattern( N × SingleAgentLoop )  →  EvaluatorOptimizer( synthesiser )
→ Gather from N sources in parallel, then refine the synthesis until it's polished.
Used for: news briefings, market reports, multi-domain analyses.
```

**Recipe 5: Enterprise multi-agent platform**
```
HierarchicalAgentPattern(
    parent = OrchestratorWorkerPattern( tool servers ),
    children = [
        PlannerExecutorPattern( research_agent ),
        EvaluatorOptimizerPattern( writer_agent ),
        SingleAgentLoop( publisher_agent ),
    ]
)
→ Parent coordinates. Children handle complex sub-domains autonomously.
Used for: full content pipelines, automated research desks, agentic software development.
```

**Recipe 6: Safe parallel execution**
```
HumanInLoop( ParallelPattern( N agents ) )
→ All parallel tasks complete, human reviews before any irreversible action is taken.
Used for: batch processing with high-stakes actions (bulk emails, bulk payments).
```

---

### The one-line rule for each pattern

| If you're asking yourself... | Use this |
|---|---|
| "I just need the agent to answer questions and use some tools." | `SingleAgentLoop` |
| "My tools live in different servers - search, DB, files, etc." | `OrchestratorWorkerPattern` |
| "Each sub-task is complex enough to need its own agent." | `HierarchicalAgentPattern` |
| "I need a human to sign off before anything irreversible happens." | `HumanInLoopPattern` |
| "I need the output to be genuinely good, not just 'good enough'." | `EvaluatorOptimizerPattern` |
| "I can't do step 3 until step 2 is done - it's inherently sequential." | `PlannerExecutorPattern` |
| "These N tasks are independent. Why am I waiting for them one by one?" | `ParallelPattern` |

---

## 10. Layer 6 - Evaluation

**Directory:** `src/mcp_agent_framework/patterns/evaluation/`

Evaluation is how you measure whether agent output is good enough. Used by `EvaluatorOptimizerPattern` but available standalone.

### `base_evaluator.py` - the contract

```python
@dataclass
class EvaluationResult:
    score: float    # 0.0 to 1.0
    passed: bool    # score >= threshold
    feedback: str   # actionable improvements - fed back to the generator
    details: dict   # per-criterion scores and notes

class AbstractEvaluator(ABC):
    async def evaluate(self, content: str, task: str, iteration: int = 0) -> EvaluationResult:
        ...
```

Build your own evaluator by subclassing `AbstractEvaluator` and implementing `evaluate()`.

### `llm_evaluator.py` - score with an LLM

Uses `complete_structured()` to ask a model to score content on a 0–10 scale with specific feedback. The score is normalised to 0–1.

```python
evaluator = LLMEvaluator(
    llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
    pass_threshold=0.7,
    evaluation_prompt="Custom prompt with {task} and {content} placeholders",  # optional
)
result = await evaluator.evaluate(
    content="The generated blog post...",
    task="Write a 500-word blog post about Python for beginners.",
)
print(f"Score: {result.score:.0%}")   # e.g. "Score: 82%"
print(f"Feedback: {result.feedback}")
```

### `rubric_evaluator.py` - score with explicit criteria

When you have specific, measurable requirements, use `RubricEvaluator` to evaluate each one separately. Mix LLM-scored criteria with rule-based checks.

```python
from mcp_agent_framework import RubricEvaluator, RubricCriterion

evaluator = RubricEvaluator(
    llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
    criteria=[
        RubricCriterion("clarity",    "Is the writing clear and easy to understand?", weight=2.0),
        RubricCriterion("accuracy",   "Are all facts verifiably correct?",            weight=3.0),
        RubricCriterion("word_count", "Is it between 400 and 600 words?",             weight=1.0),
    ],
    checker_fns={
        # Rule-based: no LLM needed for mechanical checks → costs nothing
        "word_count": lambda text: 400 <= len(text.split()) <= 600,
    },
    pass_threshold=0.75,
)
```

The final score is a weighted average. The `details` field in the result contains per-criterion scores so you can see exactly what failed.

---

## 11. Layer 7 - Memory

**Directory:** `src/mcp_agent_framework/memory/`

Memory is how agents remember things across turns. Without memory, every `agent.run()` call starts from scratch - the agent has no idea what happened before.

### The three memory types

Think of them like three different ways humans remember things:

| Type | Human analogy | What it stores | How you retrieve |
|---|---|---|---|
| **Semantic** | Long-term memory | Facts, knowledge, documents | By meaning (similar content) |
| **Episodic** | Short-term / diary | Events, observations, history | By recency or keyword |
| **Procedural** | Muscle memory | How-to knowledge, procedures | By task description |

### `semantic.py` - remember by meaning

```python
from mcp_agent_framework import SemanticMemory

memory = SemanticMemory()   # zero-dependency default (bag-of-words similarity)

# Store facts
await memory.add("Python is great for data science and machine learning")
await memory.add("Go is excellent for high-performance web servers")
await memory.add("Rust provides memory safety without garbage collection")

# Search by meaning - not just keyword match
results = await memory.search("Which language is best for building APIs?")
for entry in results:
    print(entry.content)
```

**For production quality**, pass a real embedding function:

```python
import anthropic

async def embed(text: str) -> list[float]:
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(...)  # use your provider's embedding API
    return response.embedding

memory = SemanticMemory(embed_fn=embed)
```

The default bag-of-words fallback works for demos. It fails on paraphrasing (searching "build a server" won't find "create an API" without embedding). Pass a real embedding function for production.

### `episodic.py` - remember what happened recently

```python
from mcp_agent_framework import EpisodicMemory

memory = EpisodicMemory(max_entries=1000)   # evicts oldest when full

await memory.add("User said they prefer Python over JavaScript")
await memory.add("User is building a RAG system")
await memory.add("User asked about vector databases")

# Get the 5 most recent events
recent = await memory.get_recent(5)

# Search by keyword + recency
results = await memory.search("Python preferences")
```

Episodic memory scores entries by: keyword overlap + recency bonus. More recent entries score higher even with fewer keyword matches. This prevents old entries from drowning out recent context.

### `procedural.py` - remember how to do things

```python
from mcp_agent_framework import ProceduralMemory

memory = ProceduralMemory()

await memory.add(
    "How to write a technical blog post",
    metadata={"steps": ["research", "outline", "draft", "technical_review", "edit", "publish"]}
)

await memory.add(
    "How to debug a Python application",
    metadata={"steps": ["reproduce", "add_logging", "isolate", "fix", "test", "deploy"]}
)

# Retrieve the right procedure for a task
procedure = await memory.get_by_task("write an article about databases")
if procedure:
    steps = procedure.metadata.get("steps", [])
    print(f"Procedure: {procedure.content}")
    print(f"Steps: {steps}")
```

### Wiring memory to an MCP server

Memory stores are Python objects. To make them available to an LLM, you wire them through MCP server tools:

```python
from fastmcp import FastMCP
from mcp_agent_framework import EpisodicMemory, SemanticMemory

episodic = EpisodicMemory(max_entries=500)
semantic = SemanticMemory()

app = FastMCP("memory_server")

@app.tool
async def remember(fact: str) -> str:
    """Store a fact for later retrieval."""
    await episodic.add(fact)
    await semantic.add(fact)
    return f"Remembered: {fact}"

@app.tool
async def recall(query: str) -> str:
    """Search memory for relevant information."""
    recent   = await episodic.get_recent(3)
    relevant = await semantic.search(query, top_k=3)
    # combine and return...
```

The key insight: memory stores are created **once** outside the tool functions. They persist across all calls to `remember()` and `recall()`. The agent can accumulate facts over the course of a long conversation.

---

## 12. Layer 8 - Resilience

**Directory:** `src/mcp_agent_framework/resilience/`

Production AI applications fail. Rate limits hit. Networks blip. APIs return 500s. Resilience is how you recover gracefully.

### `retry.py` - automatic retry with backoff

```python
from mcp_agent_framework import RetryPolicy

policy = RetryPolicy(
    max_retries=3,         # try up to 4 times total (1 original + 3 retries)
    base_delay=1.0,        # wait 1 second before first retry
    max_delay=60.0,        # never wait more than 60 seconds
    exponential_base=2.0,  # double the wait each retry (1s, 2s, 4s, 8s...)
    jitter=True,           # add small random variation to prevent thundering herd
)

# Wrap any async call
result = await policy.execute(
    lambda: client.complete(messages, tools=tools)
)
```

**Why jitter?** Without it, if 1000 clients all get a rate limit error at the same time, they all retry at exactly the same moment and hammer the API together. Jitter spreads them out.

**Why a callable, not a coroutine?** Coroutines are one-shot - you can't `await` the same coroutine twice. A callable is a factory that creates a new coroutine on each attempt. Always pass `lambda: ...` or a function reference.

### `circuit_breaker.py` - stop hammering a broken service

```python
from mcp_agent_framework import CircuitBreaker

breaker = CircuitBreaker(
    name="anthropic_api",
    failure_threshold=5,    # open after 5 consecutive failures
    recovery_timeout=30.0,  # try again after 30 seconds
    success_threshold=2,    # close after 2 consecutive successes in half-open
)

try:
    result = await breaker.call(lambda: client.complete(messages))
except CircuitOpenError as e:
    # Circuit is open - fail fast, don't waste time on doomed calls
    return "Service temporarily unavailable. Please try again in a moment."
```

**The three states:**
- **CLOSED** - normal operation, calls go through
- **OPEN** - too many failures, calls are rejected immediately (fail fast)
- **HALF_OPEN** - recovery probe: let a few calls through to test if service recovered

The circuit breaker protects you from spending 30 seconds waiting for a timeout on every call when the service is down. It fails fast instead, freeing resources for other work.

---

## 13. Layer 9 - Observability

**Directory:** `src/mcp_agent_framework/observability/`

Observability is how you see what your agent is doing - which models were called, which tools ran, how long things took, where errors occurred.

### `tracer.py` - capture events

```python
from mcp_agent_framework import LoggingTracer, BaseTracer, TraceEvent

# Default - logs everything to Python logging at DEBUG level
tracer = LoggingTracer()

# Custom - build your own (send to Datadog, OpenTelemetry, your database, etc.)
class MyTracer(BaseTracer):
    async def on_event(self, event: TraceEvent) -> None:
        await my_analytics_db.insert({
            "run_id":     event.run_id,
            "event_type": event.event_type,
            "timestamp":  event.timestamp,
            "data":       event.data,
        })
```

### `run_context.py` - attach a tracer to an agent run

```python
from mcp_agent_framework import RunContext, LoggingTracer

context = RunContext(tracer=LoggingTracer())

# Pass to any pattern that supports it
agent = EvaluatorOptimizerPattern(
    generator_client=client,
    evaluator=evaluator,
    config=config,
    context=context,    # optional - omit for zero overhead
)
```

When `context=None` (the default for all patterns), there is **zero overhead** - no event objects are created, no `if` checks are paid beyond one null check. Observability is entirely opt-in.

**`context.child()`** creates a child context for nested patterns. Child contexts have their own `run_id` but share the parent's `run_id` in `parent_run_id`. This lets you reconstruct the full call tree for a complex multi-agent run.

---

## 14. Building a RAG system

RAG = Retrieval-Augmented Generation. The pattern: user asks a question → retrieve relevant documents → give documents to the LLM as context → LLM answers using the documents.

Here's a complete RAG system using this framework:

```python
import asyncio
from fastmcp import FastMCP
from mcp_agent_framework import (
    AnthropicClient, SingleAgentLoop, AgentConfig,
    SemanticMemory, Message,
)

# ── Step 1: Build a knowledge base ───────────────────────────────
knowledge_base = SemanticMemory()

DOCUMENTS = [
    "Python was created by Guido van Rossum and first released in 1991.",
    "Python's design philosophy emphasises code readability with significant indentation.",
    "Python is widely used in data science, machine learning, and web development.",
    "The Zen of Python: Beautiful is better than ugly. Explicit is better than implicit.",
    "Python uses dynamic typing and garbage collection.",
]

async def load_documents():
    for doc in DOCUMENTS:
        await knowledge_base.add(doc, metadata={"source": "python_facts"})

# ── Step 2: Build an MCP server that exposes retrieval as a tool ──
app = FastMCP("rag_server")

@app.tool
async def search_knowledge_base(query: str, max_results: int = 3) -> str:
    """Search the knowledge base for information relevant to the query."""
    results = await knowledge_base.search(query, top_k=max_results)
    if not results:
        return "No relevant information found in the knowledge base."
    
    formatted = []
    for i, entry in enumerate(results, 1):
        formatted.append(f"[{i}] {entry.content}")
    return "\n".join(formatted)

# ── Step 3: Create an agent that uses the knowledge base ──────────
async def main():
    await load_documents()
    
    config = AgentConfig(
        mcp_server_config=app,
        system_prompt="""You are a helpful assistant with access to a knowledge base.
        Always use the search_knowledge_base tool before answering factual questions.
        Base your answers on the retrieved information.""",
        max_iterations=5,
    )
    
    agent = SingleAgentLoop(
        llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
        config=config,
    )
    
    # Ask questions
    questions = [
        "Who created Python?",
        "What is Python used for?",
        "What does the Zen of Python say about beauty?",
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        answer = await agent.run(q)
        print(f"A: {answer}")

asyncio.run(main())
```

**What makes this a "RAG" system:**
1. Documents are stored in `SemanticMemory` (the knowledge base)
2. The agent has a `search_knowledge_base` tool that retrieves relevant documents
3. The LLM's answer is grounded in retrieved documents, not just its training data

---

## 15. Building an Agentic RAG system

Agentic RAG goes further - the agent doesn't just retrieve once and answer. It retrieves, reasons about gaps, retrieves again, synthesises, and may even update the knowledge base.

```python
from mcp_agent_framework import (
    AnthropicClient, PlannerExecutorPattern, AgentConfig,
    SemanticMemory, EpisodicMemory,
)

# ── Knowledge base + session memory ──────────────────────────────
knowledge_base = SemanticMemory()    # persistent documents
session_memory = EpisodicMemory()   # what happened in this session

app = FastMCP("agentic_rag_server")

@app.tool
async def search_knowledge(query: str) -> str:
    """Search the knowledge base for relevant information."""
    results = await knowledge_base.search(query, top_k=5)
    return "\n".join(f"- {e.content}" for e in results) if results else "No results."

@app.tool
async def check_session_context(topic: str) -> str:
    """Check what has already been researched in this session."""
    recent = await session_memory.get_recent(10)
    relevant = await session_memory.search(topic, top_k=3)
    combined = {e.id: e for e in recent + relevant}
    return "\n".join(f"- {e.content}" for e in combined.values()) if combined else "Nothing yet."

@app.tool
async def note_finding(finding: str) -> str:
    """Record an important finding from your research."""
    await session_memory.add(finding)
    return f"Noted: {finding}"

@app.tool
async def add_to_knowledge_base(fact: str, source: str = "agent_research") -> str:
    """Add a new fact to the permanent knowledge base."""
    await knowledge_base.add(fact, metadata={"source": source})
    return f"Added to knowledge base: {fact}"

# ── Agentic RAG with a planner ────────────────────────────────────
async def agentic_rag_query(question: str):
    config = AgentConfig(
        mcp_server_config=app,
        system_prompt="""You are a research agent with access to a knowledge base.
        For complex questions:
        1. Search the knowledge base for relevant facts
        2. Note your findings as you go
        3. If you find new important facts, add them to the knowledge base
        4. Synthesise a complete, accurate answer""",
        max_iterations=15,
    )
    
    agent = PlannerExecutorPattern(
        planner_client=AnthropicClient("claude-sonnet-4-6"),
        executor_client=AnthropicClient("claude-haiku-4-5-20251001"),
        config=config,
        max_replan_attempts=1,
    )
    
    return await agent.run(question)
```

**What makes this "agentic":**
- The agent plans how to answer (what to search for, in what order)
- It can do multiple rounds of retrieval if the first search isn't enough
- It tracks what it's already found to avoid redundant searches
- It can update the knowledge base with new facts it discovers
- On failure, it replans the remaining steps

---

## 16. Building a Multi-Agent Platform

A multi-agent platform uses multiple specialised agents that work together. Here is a complete research platform:

```python
from fastmcp import FastMCP
from mcp_agent_framework import (
    AnthropicClient, AgentConfig,
    SingleAgentLoop, HierarchicalAgentPattern, ChildAgentConfig,
    ParallelPattern, ParallelTask,
    EvaluatorOptimizerPattern, LLMEvaluator,
    SemanticMemory, EpisodicMemory,
)

# ── Shared memory ──────────────────────────────────────────────────
shared_kb    = SemanticMemory()
session_log  = EpisodicMemory(max_entries=500)

# ── Specialist 1: Web Researcher ──────────────────────────────────
research_app = FastMCP("research_tools")

@research_app.tool
async def web_search(query: str) -> str:
    # In production: call a real search API
    return f"Search results for '{query}': [results here]"

@research_app.tool
async def save_finding(content: str) -> str:
    await shared_kb.add(content)
    await session_log.add(f"Research finding: {content[:80]}...")
    return "Saved."

researcher = SingleAgentLoop(
    llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
    config=AgentConfig(
        mcp_server_config=research_app,
        system_prompt="You are a meticulous researcher. Find facts, verify them, save them.",
        max_iterations=10,
    )
)

# ── Specialist 2: Data Analyst ─────────────────────────────────────
analyst_app = FastMCP("analyst_tools")

@analyst_app.tool
async def query_knowledge_base(question: str) -> str:
    results = await shared_kb.search(question, top_k=5)
    return "\n".join(f"- {e.content}" for e in results) if results else "No data."

@analyst_app.tool
def calculate_statistics(data: str) -> str:
    return f"Statistical analysis of: {data[:100]}... [computed stats here]"

analyst = SingleAgentLoop(
    llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
    config=AgentConfig(
        mcp_server_config=analyst_app,
        system_prompt="You analyse data and extract insights. Use the knowledge base.",
        max_iterations=8,
    )
)

# ── Specialist 3: Report Writer ────────────────────────────────────
writer_app = FastMCP("writer_tools")

@writer_app.tool
async def get_all_findings() -> str:
    all_entries = await shared_kb.list_all()
    return "\n".join(f"- {e.content}" for e in all_entries)

writer = SingleAgentLoop(
    llm_client=AnthropicClient("claude-sonnet-4-6"),
    config=AgentConfig(
        mcp_server_config=writer_app,
        system_prompt="You write clear, well-structured reports. Pull from the knowledge base.",
        max_iterations=5,
    )
)

# ── Evaluator for writer output ────────────────────────────────────
writing_evaluator = LLMEvaluator(
    llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
    pass_threshold=0.8,
)

# Wrap writer in evaluator-optimizer
polished_writer = EvaluatorOptimizerPattern(
    generator_client=AnthropicClient("claude-sonnet-4-6"),
    evaluator=writing_evaluator,
    config=AgentConfig(
        mcp_server_config=writer_app,
        system_prompt="You write and refine clear, accurate technical reports.",
        max_iterations=5,
    ),
    max_rounds=2,
)

# ── The Platform: Parallel Research → Hierarchical Synthesis ──────
async def research_platform(topic: str) -> str:
    # Phase 1: Parallel research across subtopics
    research_tasks = [
        ParallelTask("background",  f"Research background and history of: {topic}",      researcher),
        ParallelTask("current",     f"Research current state and trends of: {topic}",    researcher),
        ParallelTask("future",      f"Research future outlook and predictions for: {topic}", researcher),
    ]
    
    parallel_agent = ParallelPattern(
        synthesiser_client=AnthropicClient("claude-haiku-4-5-20251001"),
        tasks=research_tasks,
        config=AgentConfig(mcp_server_config={}, system_prompt="Summarise parallel research."),
    )
    
    research_summary = await parallel_agent.run(f"Comprehensive research on: {topic}")
    
    # Phase 2: Analysis + polished report via hierarchy
    coordinator_app = FastMCP("coordinator")
    
    @coordinator_app.tool
    def get_research_summary() -> str:
        return research_summary
    
    platform = HierarchicalAgentPattern(
        llm_client=AnthropicClient("claude-opus-4-6"),
        config=AgentConfig(
            mcp_server_config=coordinator_app,
            system_prompt="""You coordinate a research team. 
            Delegate analysis to the analyst, then writing to the writer.""",
            max_iterations=10,
        ),
        children=[
            ChildAgentConfig("analyst", "Analyses data and extracts insights", analyst),
            ChildAgentConfig("writer",  "Writes polished reports",              polished_writer),
        ]
    )
    
    return await platform.run(f"Produce a comprehensive report on: {topic}")

# Run it
asyncio.run(research_platform("the impact of AI on software development in 2026"))
```

**What this platform does:**
1. Three researcher agents run **in parallel** - background, current state, future outlook
2. Results are synthesised into a research summary
3. The **hierarchical coordinator** delegates to the analyst (data insights) then the writer (report)
4. The writer goes through **evaluate → rewrite** cycles until quality passes 80%

**Each agent uses the right model for its role:**
- Parallel researchers: `claude-haiku` (fast, cheap, many calls)
- Analyst: `claude-haiku` (structured analysis)
- Writer: `claude-sonnet` (nuanced writing)
- Coordinator: `claude-opus` (complex orchestration reasoning)
- Evaluator: `claude-haiku` (cheap scoring)

This is the economic reality of production multi-agent systems: use expensive models only where reasoning quality matters.

---

## 17. File reference

Every file, explained in one paragraph.

### Core

| File | What it does |
|---|---|
| `types.py` | The shared language - `Message`, `ToolCall`, `MCPTool`, `LLMResponse`, `AgentConfig`, `StopReason`. Every layer uses these. Change nothing here. |
| `__init__.py` | The public face of the framework. Every class you import comes from here. The docstring at the top is a quick-reference guide. |

### `clients/`

| File | What it does |
|---|---|
| `base_client.py` | The abstract contract - `complete()`, `complete_structured()`, `stream_complete()`. Every pattern only calls these methods. Never import a provider SDK in a pattern. |
| `anthropic_client.py` | Translates between canonical types and Anthropic's API. Structured output via forced tool use. Default model: `claude-sonnet-4-6`. |
| `openai_client.py` | Translates for OpenAI's API. Also works for Grok, Ollama, Together AI via `base_url`. Structured output via `response_format` json_schema. Default model: `gpt-5.4`. |
| `gemini_client.py` | Translates for Google's Gemini API (`google-genai` SDK). Tool schemas use `parameters_json_schema`. Structured output via `response_mime_type`. Default model: `gemini-2.5-flash`. |
| `schema_utils.py` | Converts Pydantic v1 models, Pydantic v2 models, and plain dicts to JSON Schema. Used by all three clients' `complete_structured()`. `StructuredOutputError` raised when a model fails to return valid structured output. |

### `registry/`

| File | What it does |
|---|---|
| `model_registry.py` | `ModelRegistry` - register models by name, call by name, swap without restart. `ModelResponse` and `StructuredModelResponse` - the return types. The auto-execute loop with parallel tool execution. Tag-based queries. |

### `server/`

| File | What it does |
|---|---|
| `mcp_server_base.py` | `MCPServerBase` - base class for all MCP servers. Decorators for tools, resources, prompts. `ExampleServer` shows all four capability types in one file. |
| `transports.py` | `StdioTransport` (subprocess) and `HttpTransport` (standalone service). Decision guide in docstrings. |
| `composed_server.py` | `build_composed_server()` - merge multiple MCP servers into one endpoint. Single client sees all tools. |

### `patterns/`

| File | What it does |
|---|---|
| `_tool_utils.py` | Shared `list_tools()` and `call_tool()` helpers. Used by new patterns to avoid copy-paste. |
| `single_agent_loop.py` | `SingleAgentLoop` - the ReAct loop. Foundation of all other patterns. |
| `orchestration_pattern.py` | `OrchestratorWorkerPattern` - one LLM + multiple MCP worker servers. |
| `hierarchy_pattern.py` | `HierarchicalAgentPattern` - parent LLM treats child agents as tools via `call_agent__<name>`. |
| `human_in_loop_pattern.py` | `HumanInLoopPattern` - approval gate before tool execution. |
| `evaluator_optimizer_pattern.py` | `EvaluatorOptimizerPattern` - generate → evaluate → rewrite loop. |
| `planner_executor_pattern.py` | `PlannerExecutorPattern` - structured plan → step-by-step execution → replan on failure. `ExecutionPlan` and `ExecutionStep` Pydantic models. |
| `parallel_pattern.py` | `ParallelPattern` - fan-out via `asyncio.gather`, synthesise results. `ParallelTask` and `ParallelResult`. |

### `patterns/evaluation/`

| File | What it does |
|---|---|
| `base_evaluator.py` | `AbstractEvaluator` - the interface. `EvaluationResult` - score, passed, feedback, details. |
| `llm_evaluator.py` | `LLMEvaluator` - uses `complete_structured()` to score 0–10 and get feedback. Normalises to 0–1. |
| `rubric_evaluator.py` | `RubricEvaluator` - weighted criteria with optional rule-based `checker_fns` for mechanical checks. |

### `memory/`

| File | What it does |
|---|---|
| `base.py` | `AbstractMemoryStore` - the interface (add/search/get/delete/clear). `MemoryEntry` - id, content, metadata, created_at, optional embedding. |
| `semantic.py` | `SemanticMemory` - cosine similarity search. Pluggable `embed_fn`. Pure Python default (bag-of-words) for zero-dependency demos. |
| `episodic.py` | `EpisodicMemory` - ordered event log. Recency + keyword scoring. `get_recent(n)` convenience method. Optional `max_entries` cap. |
| `procedural.py` | `ProceduralMemory` - Jaccard similarity on task descriptions. `get_by_task(task)` returns single best match. |

### `resilience/`

| File | What it does |
|---|---|
| `retry.py` | `RetryPolicy` - exponential backoff with jitter. Takes callable (not coroutine). `RetryExhaustedError` on final failure. |
| `circuit_breaker.py` | `CircuitBreaker` - CLOSED/OPEN/HALF_OPEN state machine. `asyncio.Lock` for coroutine safety. `reset()` for testing. |

### `observability/`

| File | What it does |
|---|---|
| `tracer.py` | `TraceEventType` enum, `TraceEvent` dataclass, `BaseTracer` abstract class, `LoggingTracer` default. |
| `run_context.py` | `RunContext` - run_id, parent_run_id, optional tracer. `emit()` no-ops when no tracer. `child()` for nested patterns. |

### `examples/`

| File | What it teaches |
|---|---|
| `01_hello_agent.py` | Minimal: one tool, one model, one question. The ReAct loop. |
| `02_structured_output.py` | `complete_structured()` with Pydantic. Extract typed data from text. |
| `03_model_registry.py` | Register models, call by name, find by tag, compare outputs. |
| `04_human_in_loop.py` | Approval gates. `requires_approval` set. CLI prompt. |
| `05_evaluator_optimizer.py` | Generate → evaluate → rewrite. Score printed each round. |
| `06_planner_executor.py` | Plan printed before execution. Two-model setup. |
| `07_parallel_agents.py` | Three agents in parallel. Elapsed time printed. |
| `08_memory_agent.py` | Agent that remembers across turns. Episodic + semantic. |
| `10_rag.py` | RAG pipeline: RecursiveTextChunker, SemanticMemory, retrieval. |
| `11_agentic_rag.py` | Agentic RAG: BM25 + semantic search, self-evaluation loop. |
| `12_skills.py` | Skills: Skill, SkillRegistry, SkillAwareAgent. |

---

## 18. How to copy this into your own project

This framework is designed to be dropped into any application. Here is the fastest path from copy-paste to working agent.

### Option A - install from local path

```bash
cd /path/to/mcp_agent_framework
pip install -r requirements.txt
pip install -e .
```

### Option B - copy the source into your project

```bash
cp -r mcp_agent_framework/src/mcp_agent_framework your_project/mcp_agent_framework
```

Copy `requirements.txt` into your project (or merge it with your own):
```
fastmcp>=2.0.0
mcp>=1.0.0
pydantic>=2.0.0
anthropic>=0.40.0
openai>=1.50.0
google-genai>=1.0.0
```

### Minimal starter - RAG + single agent

Copy just what you need:

```
your_project/
├── agents/
│   ├── __init__.py
│   └── research_agent.py   ← your agent code here
├── knowledge_base/
│   └── loader.py           ← load your documents into SemanticMemory
├── server/
│   └── tools.py            ← your MCP server with domain tools
└── main.py
```

```python
# main.py
import asyncio
from mcp_agent_framework import (
    AnthropicClient, SingleAgentLoop, AgentConfig, SemanticMemory,
)
from your_project.server.tools import app   # your FastMCP app

async def main():
    agent = SingleAgentLoop(
        llm_client=AnthropicClient(),
        config=AgentConfig(mcp_server_config=app, system_prompt="You are helpful."),
    )
    answer = await agent.run("Your question here")
    print(answer)

asyncio.run(main())
```

### Growing from single-agent to multi-agent

**Start here:** `SingleAgentLoop` - one tool, one model, one task.

**Add memory when:** your agent needs to remember across sessions → `SemanticMemory` + `EpisodicMemory` wired as MCP tools.

**Add evaluation when:** output quality matters → `LLMEvaluator` + `EvaluatorOptimizerPattern`.

**Scale to orchestrator-worker when:** your tools span multiple domains → `OrchestratorWorkerPattern`.

**Scale to hierarchy when:** sub-tasks need their own reasoning loops → `HierarchicalAgentPattern`.

**Scale to parallel when:** independent tasks are wasting time running sequentially → `ParallelPattern`.

**Add resilience when:** going to production → wrap `client.complete()` with `RetryPolicy`, wrap external service calls with `CircuitBreaker`.

**Add observability when:** debugging is hard → pass `RunContext(tracer=LoggingTracer())` to your patterns.

The framework is designed so you can start at the simplest point and add complexity only when you need it. Each step is a single class swap or a new wrapper - no rewrites required.

---

## 19. Skills — named, reusable agentic capabilities

Skills are the missing layer between raw tools and full coordination patterns.

```
Tools     — single functions           (search_web, write_file, send_email)
Skills    — named agentic capabilities (research_topic, write_report, compare_options)
Patterns  — coordination strategies   (ReAct, planner-executor, orchestrator-worker)
```

A **Skill** is a verb your agent system knows how to perform. It has a name, a description the LLM can read, a JSON Schema for its inputs, and an async handler that does the actual work. The handler can call anything — a SingleAgentLoop, a PlannerExecutorPattern, another skill, or raw API calls.

### Why Skills?

Without skills, every time you build a new agent you re-implement the same "research" or "summarise" logic from scratch. With skills, you define it once, register it, and any agent in the system can discover and call it.

Industry context: Claude Code has `/skills`, OpenAI has `GPT Actions`, LangGraph has `subgraphs`. The concept is universal — reusable named capabilities that agents can compose.

### The three core classes

**`Skill`** — a named, typed, invokable capability:

```python
from mcp_agent_framework.skills import Skill

async def _do_research(inputs: dict) -> str:
    agent = SingleAgentLoop(...)
    return await agent.run(inputs["topic"])

skill = Skill(
    name="research_topic",
    description="Deep research on any topic. Returns a detailed explanation.",
    input_schema={
        "type": "object",
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
    },
    handler=_do_research,
    tags=["research", "read-only"],
)
```

**`SkillRegistry`** — central store for all skills. Register once, invoke anywhere:

```python
from mcp_agent_framework.skills import SkillRegistry

registry = SkillRegistry()
registry.register(research_skill)
registry.register(summarise_skill)
registry.register(compare_skill)

# Direct invocation (no LLM involved)
result = await registry.invoke("research_topic", {"topic": "vector search"})

# Parallel invocation
results = await registry.invoke_many([
    ("research_topic", {"topic": "vector search"}),
    ("research_topic", {"topic": "BM25"}),
])

# Filter by tag
read_only = registry.list_skills(tag="read-only")
```

**`SkillAwareAgent`** — an agent that auto-wires the registry as two MCP tools:
- `list_skills()` — the LLM calls this to discover what capabilities exist
- `invoke_skill(name, inputs_json)` — the LLM calls this to run a skill

```python
from mcp_agent_framework.skills import SkillAwareAgent

agent = SkillAwareAgent(
    llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
    registry=registry,
    system_prompt="Use list_skills() to discover capabilities, then invoke them.",
)
answer = await agent.run("Compare vector search vs BM25 in detail.")
```

The LLM decides which skills to call, in what order, and how to combine results — exactly like it decides which tools to call in SingleAgentLoop.

### Skill composition

Skills can call other skills. This is how you build complex capabilities from simple ones:

```python
async def _compare(inputs: dict) -> str:
    # Research both topics in parallel (skill calling skill)
    a, b = await asyncio.gather(
        registry.invoke("research_topic", {"topic": inputs["topic_a"]}),
        registry.invoke("research_topic", {"topic": inputs["topic_b"]}),
    )
    # Then synthesise
    return await registry.invoke("summarise_text", {"text": f"{a}\n\n{b}"})

compare_skill = Skill(
    name="compare_topics",
    description="Research two topics and compare them side by side.",
    input_schema={...},
    handler=_compare,
)
```

This is the same pattern as LangGraph subgraphs — reusable units of agent work that compose cleanly.

### Example

See `examples/12_skills.py` for a full working demo covering:
- Three skills: `research_topic`, `summarise_text`, `compare_topics`
- Direct invocation via `registry.invoke()`
- Parallel invocation via `registry.invoke_many()`
- `SkillAwareAgent` with the LLM picking skills autonomously
- Tag-based filtering

---

## 20. Applied AI Engineering Curriculum

This framework is the textbook. These 21 lessons take you from "what is an agent?" to building production multi-agent systems with skills, RAG, resilience, LangGraph integration, and multi-modal pipelines.

Estimated time: 40–60 hours of focused study and hands-on work.

| Lesson | Topic | Key concept | Example file |
|--------|-------|-------------|--------------|
| 1  | Why agents exist | The problem they solve; tools vs agents vs workflows | `01_hello_agent.py` |
| 2  | Types: the shared language | `Message`, `LLMResponse`, `ToolCall`, `StreamEvent`; why one type system matters | — |
| 3  | Clients: talking to LLMs | `AnthropicClient`, `OpenAIClient`, `GeminiClient`; `stream()`; extended thinking | `02_structured_output.py` |
| 4  | MCP: tools for your agent | FastMCP; `@app.tool`; in-process, stdio, HTTP transports; context-aware servers | `01_hello_agent.py` |
| 5  | The single agent loop | ReAct loop; `run()` vs `run_stream()`; `system_prompt`; `max_iterations` | `01_hello_agent.py` |
| 6  | Tool calling deep dive | Full lifecycle: `list_tools` → `ToolCall` → `call_tool` → result message | `01_hello_agent.py` |
| 7  | Memory | `SemanticMemory`, `EpisodicMemory`, `ProceduralMemory`; memory as MCP tools | `08_memory_agent.py` |
| 8  | Orchestrator pattern | One LLM, multiple MCP workers; tool routing; `asyncio.gather` | — |
| 9  | Hierarchy pattern | Parent delegates to child agent loops; `call_agent__` synthetic tools | — |
| 10 | Human-in-the-loop | Approval callbacks; `requires_approval`; sync vs async gates | `04_human_in_loop.py` |
| 11 | Evaluation | `LLMEvaluator`, `RubricEvaluator`, `RubricCriterion`; scoring quality | `05_evaluator_optimizer.py` |
| 12 | EvaluatorOptimizer | Generate → evaluate → rewrite loop; working history; convergence | `05_evaluator_optimizer.py` |
| 13 | PlannerExecutor | Structured `ExecutionPlan`; dynamic replan on step failure; synthesis | `06_planner_executor.py` |
| 14 | Parallel pattern | Fan-out / gather; independent subtasks; fault-tolerant synthesis | `07_parallel_agents.py` |
| 15 | Resilience | `RetryPolicy` with jitter; `CircuitBreaker`; production reliability | — |
| 16 | Observability | `RunContext`, `LoggingTracer`, `TraceEventType`; custom tracer backends | — |
| 17 | RAG | Chunking, embedding, cosine retrieval; `RecursiveTextChunker` | `10_rag.py` |
| 18 | Agentic RAG | BM25, self-evaluation loop, multi-round retrieval | `11_agentic_rag.py` |
| 19 | LangGraph integration | Checkpointing, interrupts, time travel; when LangGraph beats this framework | `langgraph+mcp_agent_framework/` |
| 20 | Skills | `Skill`, `SkillRegistry`, `SkillAwareAgent`; composable agentic capabilities | `12_skills.py` |
| 21 | Multi-modal pipeline | LLM + image tools; rembg, DALL-E 3, Pillow; audience-driven scene generation | `product_image_pipeline.py` |

### How to use this curriculum

Work through each lesson in order. For each lesson:
1. Read the relevant README section to understand the concept
2. Read the source file(s) named in the table of contents above
3. Run the example and observe the output
4. Modify the example — break something, fix it, add a feature
5. Build one small project using only the concepts from that lesson

The lessons are cumulative. By Lesson 21 you will have built every component of a production-grade multi-agent system from scratch.

---

## LangGraph - how it compares

[LangGraph](https://github.com/langchain-ai/langgraph) is a standalone framework for building stateful, graph-structured agents. It is one of the most widely used agent frameworks and has several features that go beyond what this framework implements directly.

**What LangGraph is famous for:**

| LangGraph feature | What it does | Equivalent here |
|---|---|---|
| **Interrupts** | Pause a running graph mid-execution, surface state to a human or external system, then resume from exactly where it stopped - across process restarts | `HumanInLoopPattern` handles approval callbacks but does not persist + resume across restarts |
| **Checkpointing** | Every graph node's state is saved to a persistent backend (Postgres, SQLite, Redis) after each step. You can replay, branch, or resume any prior run by ID | No built-in persistence - state lives in memory for the duration of a `run()` call |
| **Conditional edges** | Route execution to different nodes based on LLM output, tool results, or custom logic - the graph topology itself is dynamic | Implemented in each pattern's Python loop; not a declarative graph |
| **Human-in-the-loop at scale** | Interrupts + checkpointing combine to make async human review practical in production: the graph pauses, a human reviews hours later, execution resumes | `HumanInLoopPattern` is synchronous - the event loop blocks waiting for the approval callback |
| **Time travel / branching** | Rewind to any earlier checkpoint and run a different branch from that point (useful for A/B testing agent behaviour or recovering from bad decisions) | Not supported |
| **Streaming token-by-token** | Built-in support for streaming individual tokens out of any node as they are generated | `run_stream()` on every pattern yields `StreamEvent` objects live; `AnthropicClient(enable_thinking=True)` also streams reasoning tokens |
| **Studio UI** | LangGraph Studio gives a visual debugger: see the graph topology, step through node executions, inspect state at each checkpoint | Not included |

**Why this framework doesn't use LangGraph:**

This framework was built from scratch intentionally - as a teaching tool. Every loop, every state transition, every tool-routing decision is visible in plain Python. The goal is for you to understand *what LangGraph (and similar frameworks) are actually doing under the hood* before you pick one up.

Once you understand how a `SingleAgentLoop` works at the Python level, LangGraph's `StateGraph` + `ToolNode` pattern becomes immediately readable. The concepts are the same; LangGraph adds production infrastructure (persistence, interrupts, Studio) on top.

**When to reach for LangGraph instead:**

- You need persistent, resumable runs (agent pauses overnight, resumes next day)
- You need async human approval in production (interrupt → human reviews → resume)
- You want time travel / branching for debugging or A/B testing agent behaviour
- You want a visual graph editor and step-through debugger

**When this framework is the right tool:**

- You are learning how agents work and want to read every line
- You need a lightweight, dependency-minimal base with no framework lock-in
- You are integrating with MCP servers specifically
- You need full control over provider differences (Anthropic, OpenAI, Gemini)

---

*Framework version: 0.1.0 - April 2026*  
*Python 3.11+ required*  
*Built with: fastmcp, anthropic, openai, google-genai, pydantic*
