"""
Example 13 — Streaming and Reasoning

Demonstrates all four combinations of streaming and extended thinking:

  Combo 1: No thinking, no streaming  — response.content arrives all at once
  Combo 2: No thinking, with streaming — text tokens arrive live
  Combo 3: Thinking, no streaming     — response.reasoning + response.content at end
  Combo 4: Thinking, with streaming   — thinking tokens then text tokens, both live

Run:
    python examples/13_streaming.py

No MCP server needed — uses a bare LLM call for clarity.
"""
from __future__ import annotations

import asyncio
import os

from mcp_agent_framework import AnthropicClient
from mcp_agent_framework.types import Message


QUESTION = "In two sentences, explain why cosine similarity works well for comparing text embeddings."


# ---------------------------------------------------------------------------
# Combo 1: No thinking, no streaming
# ---------------------------------------------------------------------------
async def combo1_response_only() -> None:
    print("\n" + "=" * 60)
    print("COMBO 1 — Response only, no streaming")
    print("=" * 60)

    client = AnthropicClient()
    response = await client.complete(
        messages=[Message(role="user", content=QUESTION)]
    )

    print(f"content   : {response.content}")
    print(f"reasoning : {response.reasoning}")   # always None here


# ---------------------------------------------------------------------------
# Combo 2: No thinking, with streaming
# ---------------------------------------------------------------------------
async def combo2_streaming_no_thinking() -> None:
    print("\n" + "=" * 60)
    print("COMBO 2 — Response only, with streaming")
    print("=" * 60)

    client = AnthropicClient()

    print("text: ", end="", flush=True)
    async for event in client.stream(
        messages=[Message(role="user", content=QUESTION)]
    ):
        if event.type == "text":
            print(event.delta, end="", flush=True)
    print()   # newline after stream ends


# ---------------------------------------------------------------------------
# Combo 3: Thinking enabled, no streaming
# ---------------------------------------------------------------------------
async def combo3_thinking_no_streaming() -> None:
    print("\n" + "=" * 60)
    print("COMBO 3 — Reasoning + response, no streaming")
    print("=" * 60)

    client = AnthropicClient(enable_thinking=True, thinking_budget=2000)
    response = await client.complete(
        messages=[Message(role="user", content=QUESTION)]
    )

    reasoning_preview = (response.reasoning or "")[:200]
    print(f"reasoning : {reasoning_preview}{'...' if len(response.reasoning or '') > 200 else ''}")
    print(f"content   : {response.content}")


# ---------------------------------------------------------------------------
# Combo 4: Thinking enabled, with streaming
# ---------------------------------------------------------------------------
async def combo4_thinking_and_streaming() -> None:
    print("\n" + "=" * 60)
    print("COMBO 4 — Reasoning + response, with streaming")
    print("=" * 60)

    client = AnthropicClient(enable_thinking=True, thinking_budget=2000)

    print("thinking: ", end="", flush=True)
    in_text = False
    async for event in client.stream(
        messages=[Message(role="user", content=QUESTION)]
    ):
        if event.type == "thinking":
            print(event.delta, end="", flush=True)
        elif event.type == "text":
            if not in_text:
                print("\ntext    : ", end="", flush=True)
                in_text = True
            print(event.delta, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY before running this example.")

    await combo1_response_only()
    await combo2_streaming_no_thinking()
    await combo3_thinking_no_streaming()
    await combo4_thinking_and_streaming()

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
