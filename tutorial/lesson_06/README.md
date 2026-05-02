# Lesson 6 — Tool Calling Deep Dive

**Unit 2: Core Patterns**

---

## What you will learn

- The exact lifecycle of a single tool call, from LLM decision to result
- How the LLM "decides" to use a tool (it's not magic — it's pattern matching)
- What happens when a tool raises an exception
- How multiple tool calls in one response work
- How to write tool descriptions that guide the LLM correctly

---

## The concept

Tool calling feels like magic the first time you see it. The LLM "knows" to call `search_database` with the right arguments. But understanding exactly what happens removes the mystery — and lets you debug and design better.

### Step 1 — The LLM sees the tool list

Before the LLM sees your user message, it receives the tool schemas:

```json
[
  {
    "name": "search_database",
    "description": "Search the product database by name or category. Returns JSON array of matches.",
    "input_schema": {
      "type": "object",
      "properties": {
        "query":  {"type": "string", "description": "Search term"},
        "limit":  {"type": "integer", "description": "Max results (default 10)"}
      },
      "required": ["query"]
    }
  }
]
```

This is what `list_tools()` fetches and what the client encodes into the API call. **The LLM reads the `description` and `input_schema` to decide if and how to use the tool.** The description is your communication channel with the LLM.

### Step 2 — The LLM "decides" to use a tool

The LLM is not executing code or running logic. It is completing a sequence. When it sees a tool that matches the user's need, the probability distribution over next tokens heavily favours outputting a tool call in the required JSON format.

Concretely, this is what the raw model output looks like:

```json
{
  "stop_reason": "tool_use",
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_01XFb",
      "name": "search_database",
      "input": {"query": "red running shoes", "limit": 5}
    }
  ]
}
```

The client decodes this into:
```python
LLMResponse(
    content=None,
    tool_calls=[ToolCall(id="toolu_01XFb", name="search_database", arguments={"query": "red running shoes", "limit": 5})],
    stop_reason=StopReason.TOOL_USE,
)
```

### Step 3 — Your code executes the tool

```python
result = await call_tool(mcp, tool_call)
# result is a string: '[{"id":1,"name":"Red Nike Pegasus",...},...]'
```

`call_tool()` calls the MCP server, which calls the actual function, and returns the result as a string.

### Step 4 — The result goes back as a message

```python
messages.append(Message(
    role="tool",
    content=result,          # the string result
    tool_call_id=tool_call.id,  # "toolu_01XFb" — links result to request
    name=tool_call.name,    # "search_database"
))
```

The `tool_call_id` is critical. It links this result to the specific tool call that requested it. Without it, the LLM cannot match results to requests when multiple tools were called.

### Step 5 — The LLM sees the result and continues

On the next iteration, the LLM receives the full history including the tool result. It now "knows" what the search returned and can either use the data to answer, or call more tools.

---

## Multiple tool calls in one response

The LLM can request multiple tools in a single response:

```json
{
  "stop_reason": "tool_use",
  "content": [
    {"type": "tool_use", "id": "tc_1", "name": "search_database", "input": {"query": "shoes"}},
    {"type": "tool_use", "id": "tc_2", "name": "get_inventory",   "input": {"category": "footwear"}}
  ]
}
```

The loop handles this with `for tool_call in response.tool_calls`. Both tools are called before the next LLM iteration. The LLM gets both results together.

**Can you run them in parallel?** Yes — and you should. Sequential execution wastes time when the tools are independent:

```python
# Sequential (default in SingleAgentLoop for simplicity)
for tc in response.tool_calls:
    result = await call_tool(mcp, tc)

# Parallel (used in OrchestratorWorkerPattern)
results = await asyncio.gather(*[call_tool(mcp, tc) for tc in response.tool_calls])
```

`OrchestratorWorkerPattern` uses `asyncio.gather` for parallelism.

---

## What happens when a tool fails

If `call_tool()` raises an exception, or the tool returns an error string, that error goes back to the LLM as the tool result. A well-behaved LLM will see the error and either:
1. Try the tool again with different arguments
2. Try a different tool
3. Explain to the user that it couldn't complete the task

This means your tools should return *meaningful error strings* rather than raising exceptions when possible:

```python
@app.tool
async def search_database(query: str) -> str:
    """Search the product database."""
    try:
        results = db.search(query)
        if not results:
            return f"No products found matching '{query}'. Try a broader search term."
        return json.dumps(results)
    except DatabaseError as e:
        return f"Database error: {e}. The database may be temporarily unavailable."
```

The LLM gets a useful error message it can relay to the user or react to.

---

## Writing tool descriptions that work

The description is a prompt. Treat it that way.

**Bad:**
```python
async def search(q: str) -> str:
    """Search."""
```

**Good:**
```python
async def search_products(query: str, category: str = "") -> str:
    """
    Search the product catalogue by keyword.

    Use this when the user asks about specific products, wants to find items,
    or asks what we sell. Pass the user's search terms directly as the query.
    Optionally filter by category (e.g. 'shoes', 'electronics', 'clothing').

    Returns a JSON array of matching products with id, name, price, and stock.
    Returns an empty array if nothing matches — try a shorter/broader query.
    """
```

The good description tells the LLM:
- When to use it
- How to map user intent to arguments
- What the output looks like
- What to do if it fails

---

## The `input_schema` in detail

JSON Schema controls what the LLM passes as arguments. Key fields:

```json
{
  "type": "object",
  "properties": {
    "query":    {"type": "string",  "description": "The search query"},
    "limit":    {"type": "integer", "description": "Max results", "default": 10},
    "category": {"type": "string",  "enum": ["shoes", "clothing", "electronics"]}
  },
  "required": ["query"]
}
```

- `"required"` — arguments the LLM must always provide
- `"description"` on properties — per-argument guidance for the LLM
- `"enum"` — constrain the LLM to specific values (prevents hallucinated values)
- `"default"` — shown to the LLM but not enforced by the schema

FastMCP generates this schema automatically from your function signature and type hints. Annotate well:

```python
async def search_products(
    query: str,
    limit: int = 10,
    category: Literal["shoes", "clothing", "electronics"] | None = None,
) -> str:
```

`Literal["shoes", ...]` becomes `"enum"` in the schema automatically.

---

## Read these files

```
src/mcp_agent_framework/patterns/_tool_utils.py         ← list_tools(), call_tool()
src/mcp_agent_framework/patterns/single_agent_loop.py   ← the for loop over tool_calls
src/mcp_agent_framework/clients/anthropic_client.py     ← _encode_tool(), _decode_response()
```

In `anthropic_client.py`, find `_decode_response()` and trace exactly how a `ToolUseBlock` becomes a `ToolCall`. Find `_encode_tool()` and trace how `MCPTool` becomes an Anthropic `ToolParam`.

---

## Run this

Add detailed logging to `01_hello_agent.py`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

You will see:
```
DEBUG list_tools: found 3 tools — [search_knowledge, list_topics, ...]
DEBUG iteration 1
DEBUG [tool] search_knowledge → "Vector databases store..."
DEBUG iteration 2
```

Match these log lines to the steps in the lifecycle above.

---

## Build this

Build a tool that intentionally fails sometimes and observe how the agent recovers:

```python
import random

@app.tool
async def flaky_search(query: str) -> str:
    """Search the knowledge base. May occasionally return a timeout error."""
    if random.random() < 0.4:   # fails 40% of the time
        return f"Error: search timed out for query '{query}'. Please try again."
    return f"Results for '{query}': [relevant information here]"
```

Ask the agent a question. Watch how it handles the error — does it retry? Does it use a different query? Does it give up and explain? Try rephrasing the error message (make it more or less informative) and see how that changes the agent's behaviour.

---

## Key terms

| Term | Meaning |
|------|---------|
| Tool call lifecycle | Decision → encode → execute → result → message |
| `tool_call_id` | Links a tool result back to the specific request that created it |
| `input_schema` | JSON Schema describing tool arguments — the LLM reads this |
| `required` | Schema field: arguments the LLM must always provide |
| Tool description | A prompt that tells the LLM when and how to use the tool |

---

## Connects to

- **Lesson 7** — memory tools follow the same lifecycle: the agent calls `remember()` and `recall()` exactly like any other tool
- **Lesson 8** — the orchestrator's "workers" are tool calls too — each worker is a tool from the orchestrator's perspective
- **Lesson 15** — resilience: wrapping `call_tool()` with retry logic for flaky tool servers
- **Lesson 20** — `invoke_skill()` is just a tool call that happens to run a whole agent pattern

---

*Lesson 6 of 20 — Applied AI Engineering*
