"""
=============================================================================
LangGraph + MCP Agent Framework — 04: Streaming Token-by-Token
=============================================================================

WHAT THIS EXAMPLE TEACHES
--------------------------
SingleAgentLoop.run() waits for the full response before returning — the user
sees nothing until the LLM finishes thinking. For long answers this means
seconds of silence.

Real streaming: tokens appear on screen as the LLM generates them, just like
ChatGPT's typing effect.

TWO LAYERS OF STREAMING
------------------------
  Layer 1 — LLM streaming:
    The provider SDK has a streaming mode that yields text chunks as they
    are generated. AnthropicClient needs a `stream_complete()` method that
    uses `client.messages.stream()` internally.

  Layer 2 — LangGraph streaming:
    `graph.astream()` with `stream_mode="custom"` lets a node push arbitrary
    chunks to the caller via `config["writer"](chunk)`. This propagates
    the LLM token stream through the graph to the outer caller.

    For simpler use cases, `stream_mode="values"` streams the entire state
    after each node completes — useful for watching multi-node pipelines
    progress in real time.

HOW TO ADD STREAMING TO BaseLLMClient
--------------------------------------
The framework's BaseLLMClient.complete() returns a full LLMResponse.
For streaming, add stream_complete() that yields str chunks:

    # In AnthropicClient:
    async def stream_complete(self, messages, system=None):
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[self._encode_message(m) for m in messages],
            system=system,
        ) as stream:
            async for text in stream.text_stream:
                yield text

This example implements a standalone streaming node to show the pattern
without requiring changes to the framework source.

RUNNING
-------
    pip install langgraph
    pip install -r ../../requirements.txt && pip install -e ../..
    export ANTHROPIC_API_KEY=sk-ant-...
    python 04_streaming.py
=============================================================================
"""

from __future__ import annotations

import asyncio
import os
from typing import Annotated, Any, AsyncIterator, TypedDict

import anthropic
from fastmcp import FastMCP
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter

from mcp_agent_framework.types import Message

# =============================================================================
# MCP server
# =============================================================================

app = FastMCP("streaming_demo")

@app.tool
async def get_explanation(topic: str) -> str:
    """Retrieve a topic explanation to include in the response."""
    explanations = {
        "streaming": (
            "Streaming in LLMs works by having the model emit tokens one at a time "
            "as they are sampled from the probability distribution, rather than "
            "buffering the entire response. This reduces time-to-first-token (TTFT) "
            "dramatically and gives users immediate feedback."
        ),
        "langgraph": (
            "LangGraph models agent logic as a directed graph where nodes are Python "
            "functions and edges define execution flow. State is a typed dict that "
            "flows between nodes and is persisted by the checkpointer after each step."
        ),
    }
    return explanations.get(topic.lower(), f"No explanation for '{topic}'.")


# =============================================================================
# Streaming LLM helper
#
# This is the pattern to add to AnthropicClient as stream_complete().
# Shown standalone here so no framework source files need to change.
# =============================================================================

async def stream_anthropic(
    messages: list[Message],
    system:   str | None = None,
    model:    str = "claude-haiku-4-5-20251001",
) -> AsyncIterator[str]:
    """
    Yield text chunks from Anthropic's streaming API.

    This is what BaseLLMClient.stream_complete() would look like if added
    to the framework. Each yielded string is a text delta (typically 1–5 words).
    """
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    encoded = [
        {"role": m.role, "content": m.content or ""}
        for m in messages
        if m.role in ("user", "assistant")
    ]
    params: dict[str, Any] = {
        "model":      model,
        "max_tokens": 1024,
        "messages":   encoded,
    }
    if system:
        params["system"] = system

    async with client.messages.stream(**params) as stream:
        async for text in stream.text_stream:
            yield text


# =============================================================================
# State
# =============================================================================

class State(TypedDict):
    messages:      Annotated[list, add_messages]
    full_response: str


# =============================================================================
# Streaming node
#
# StreamWriter is LangGraph's mechanism for pushing custom chunks from inside
# a node to the caller of graph.astream(stream_mode="custom").
# The caller receives these chunks in real time — before the node returns.
# =============================================================================

async def streaming_agent_node(state: State, writer: StreamWriter) -> dict:
    """
    Run the agent and stream tokens to the caller as they are generated.

    writer(chunk) pushes each token immediately through the LangGraph stream.
    The full response is also stored in state for checkpointing.
    """
    latest  = state["messages"][-1]
    content = latest["content"] if isinstance(latest, dict) else latest.content

    system = (
        "You are a helpful assistant explaining LangGraph and streaming concepts. "
        "Give detailed, thoughtful answers. Do not use the tool — answer from knowledge."
    )

    full_response = ""
    # Stream tokens directly to the caller
    async for chunk in stream_anthropic(
        [Message(role="user", content=content)],
        system=system,
    ):
        writer(chunk)           # pushed to caller immediately
        full_response += chunk  # accumulated for state storage

    return {
        "messages":      [{"role": "assistant", "content": full_response}],
        "full_response": full_response,
    }


# =============================================================================
# Second node — shows stream_mode="values" (full state after each node)
# =============================================================================

async def summarise_node(state: State) -> dict:
    """Add a one-line summary after the main response."""
    response = state.get("full_response", "")
    summary  = f"\n\n[Summary: {len(response.split())} words generated via token-by-token streaming]"
    return {
        "messages":      [{"role": "assistant", "content": summary}],
        "full_response": response + summary,
    }


# =============================================================================
# Build graph
# =============================================================================

def build_graph(checkpointer):
    builder = StateGraph(State)
    builder.add_node("stream_agent", streaming_agent_node)
    builder.add_node("summarise",    summarise_node)
    builder.add_edge(START,          "stream_agent")
    builder.add_edge("stream_agent", "summarise")
    builder.add_edge("summarise",    END)
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# Demo
# =============================================================================

async def main() -> None:
    checkpointer = MemorySaver()
    graph        = build_graph(checkpointer)
    config       = {"configurable": {"thread_id": "streaming-demo-1"}}

    question = "Explain how token-by-token streaming works in LLMs and why it matters for user experience."

    print("=" * 60)
    print("Demo 1 — stream_mode='custom': tokens arrive as they are generated")
    print("=" * 60)
    print(f"\nQ: {question}\nA: ", end="", flush=True)

    # stream_mode="custom" receives whatever writer(chunk) sends from the node
    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": question}]},
        config,
        stream_mode="custom",
    ):
        print(chunk, end="", flush=True)   # print each token immediately

    print("\n")

    # ── Demo 2: stream_mode="values" — full state after each node ────────────
    print("=" * 60)
    print("Demo 2 — stream_mode='values': full state snapshot after each node")
    print("(useful for watching multi-node pipelines progress)")
    print("=" * 60)

    config2 = {"configurable": {"thread_id": "streaming-demo-2"}}
    async for state_snapshot in graph.astream(
        {"messages": [{"role": "user", "content": "What is LangGraph in one sentence?"}]},
        config2,
        stream_mode="values",
    ):
        msg_count = len(state_snapshot.get("messages", []))
        response  = state_snapshot.get("full_response", "")
        node_name = "stream_agent" if msg_count <= 2 else "summarise"
        print(f"\n[After node '{node_name}'] {msg_count} messages, {len(response)} chars")


if __name__ == "__main__":
    asyncio.run(main())
