# Lesson 1 — Why Agents Exist

**Unit 1: Foundations**

---

## What you will learn

- What makes an agent different from a regular LLM call or a script
- The three properties every agent must have
- Why deterministic code fails for open-ended tasks
- How to read a simple agent's source and trace its execution

---

## The concept

Every software system before agents was *deterministic*: you write code, it runs the same way every time. Input X always produces output Y. This works perfectly for calculators, sorting algorithms, and CRUD APIs.

Agents are different. An agent *observes* its environment, *reasons* about what to do, and *acts* — then observes the result of that action. The loop repeats until the task is done or the agent decides it is done.

### Why deterministic code fails here

Imagine you are building a "customer support assistant" that answers questions about your documentation. The deterministic approach:

```python
if "refund" in question:
    return refund_policy_text
elif "shipping" in question:
    return shipping_policy_text
# ... 500 more elif branches ...
```

This breaks the moment a user asks something slightly unexpected. You cannot enumerate all possible questions in advance.

An agent generalises. It reads the docs, understands the question, reasons about what information is relevant, and constructs an answer. That generalisation is what makes agents powerful — and what makes them require a different engineering mindset.

### The three properties of every agent

**1. Tools — the agent can *do* things, not just say things**

A plain LLM produces text. An agent with tools can search a database, send an email, write a file, call an API. The agent decides when and how to use these tools.

**2. A loop — the agent keeps going until the task is complete**

One LLM call often isn't enough. The agent observes the result of a tool call, reasons about whether it's done, and continues if not. This is the loop.

**3. Stopping criteria — the agent decides when it's done**

The loop must end. Either the model decides it has enough information to give a final answer, or a safety limit (max iterations) stops it. Without stopping criteria, agents run forever.

### The simplest possible agent

```python
messages = [user_message]
while True:
    response = llm.call(messages, tools=available_tools)
    if response.is_text:
        return response.text      # done
    # otherwise, execute the tool calls
    for tool_call in response.tool_calls:
        result = execute(tool_call)
        messages.append(result)
    messages.append(response)
```

That is the entire ReAct pattern. Everything else in this framework is built on top of this loop.

---

## Read these files

Start here — these are the foundational types every other file uses:

```
src/mcp_agent_framework/types.py
```

Key things to notice:
- `Message` — a single turn (user / assistant / tool). The conversation history is just a list of these.
- `ToolCall` — what the LLM sends back when it wants to use a tool. Has an `id`, `name`, and `arguments`.
- `LLMResponse` — the normalised response from any provider. Has `content` (text) or `tool_calls` (tool requests), and a `stop_reason`.
- `StopReason` — `END_TURN` (model finished), `TOOL_USE` (model wants tools), `MAX_TOKENS` (hit limit).
- `AgentConfig` — what you pass when creating any agent pattern.

Then read the simplest agent loop:

```
src/mcp_agent_framework/patterns/single_agent_loop.py
```

It is under 120 lines. Read all of it.

---

## Run this example

```bash
cd /path/to/mcp_agent_framework
pip install -r requirements.txt
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...

python examples/01_hello_agent.py
```

Watch the output. The agent:
1. Lists available tools
2. Decides to call one (or more)
3. Gets the result
4. Produces a final text answer

---

## Build this

Write a 30-line agent **from scratch** — no patterns, just raw `AnthropicClient`:

```python
import asyncio
from fastmcp import FastMCP, Client
from mcp_agent_framework import AnthropicClient
from mcp_agent_framework.types import Message, StopReason

app = FastMCP("math_server")

@app.tool
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

async def main():
    client = AnthropicClient("claude-haiku-4-5-20251001")
    async with Client(app) as mcp:
        tools = [...]  # list_tools from mcp
        messages = [Message(role="user", content="What is 17 plus 25?")]

        while True:
            response = await client.complete(messages, tools=tools)
            messages.append(Message(role="assistant", content=response.content, tool_calls=response.tool_calls))

            if response.stop_reason != StopReason.TOOL_USE:
                print(response.content)
                break

            for tc in response.tool_calls:
                result = await mcp.call_tool(tc.name, tc.arguments)
                messages.append(Message(role="tool", content=str(result), tool_call_id=tc.id, name=tc.name))

asyncio.run(main())
```

Fill in the gaps (the `list_tools` call). Run it. You will feel the loop manually before `SingleAgentLoop` hides it from you.

---

## Key terms

| Term | Meaning |
|------|---------|
| Agent | An LLM + tools + a loop |
| Tool | A function the LLM can request to call |
| ReAct | Reason + Act — the loop pattern |
| Stop reason | Why the LLM stopped: text done, tool needed, or token limit |
| Context window | The LLM's working memory — the message list it can see |

---

## Connects to

- **Lesson 2** — deep dive into the ReAct loop mechanics
- **Lesson 3** — tools and MCP in detail
- **Lesson 4** — the client layer that makes providers interchangeable

---

*Lesson 1 of 20 — Applied AI Engineering*
