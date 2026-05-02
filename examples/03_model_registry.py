"""
=============================================================================
EXAMPLE 03: Model Registry — register once, call anywhere, swap any time
=============================================================================

WHY DO YOU WANT A REGISTRY?
----------------------------
Imagine you have 10 agent workflows and each one calls an LLM. If you
hardcode "claude-haiku-4-5-20251001" in all 10 places, then upgrading the
model means touching 10 files. If you point an A/B test at "smart" and later
want to swap the model behind "smart", you do it in one line.

The registry is a simple name → client lookup table:

    registry.register("fast",  AnthropicClient("claude-haiku-4-5-20251001"))
    registry.register("smart", AnthropicClient("claude-sonnet-4-6"))

Your workflow code calls registry.complete("smart", ...) — it never needs to
know which specific model is behind "smart". You can swap it later with
registry.swap("smart", new_client) at runtime, zero restarts.

WHAT ARE TAGS?
--------------
Tags are free-form labels you attach when registering. They let you query
"give me all cheap models" or "give me all vision-capable models" without
keeping a separate list yourself.

    registry.register("fast",  ..., tags=["cheap", "fast"])
    registry.register("flash", ..., tags=["cheap", "fast", "gemini"])

    registry.find_by_tag("cheap")  → ["fast", "flash"]

This is useful for fallback strategies: if the "smart" model is unavailable,
find any model tagged "balanced" and use that instead.

auto_execute=True vs auto_execute=False
---------------------------------------
When you call registry.complete():

  auto_execute=False (DEFAULT)
    The registry makes ONE call to the model and returns immediately.
    If the model wants to call tools, the tool_calls list is populated and
    YOU are responsible for running them and continuing the conversation.
    Use this when you need full control over the tool-call loop, or when
    you are just doing a text-only call (no tools).

  auto_execute=True
    The registry runs the full ReAct loop for you: calls the model, executes
    any tool calls using the tool_executor you provide, feeds the results
    back, and keeps going until the model stops calling tools.
    Use this for convenience when you do not need custom logic per tool call.

Run:
    ANTHROPIC_API_KEY=sk-ant-... python examples/03_model_registry.py
=============================================================================
"""

import asyncio
import os

from mcp_agent_framework import AnthropicClient, Message, ModelRegistry

# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Get your key at https://console.anthropic.com and run:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-..."
        )
        return

    # ---------------------------------------------------------------------------
    # 1. Register models
    # ---------------------------------------------------------------------------
    section("1. Registering models")

    registry = ModelRegistry()

    registry.register(
        "fast",
        AnthropicClient("claude-haiku-4-5-20251001"),
        description="Fastest and cheapest Claude. Good for simple tasks.",
        tags=["cheap", "fast"],
    )
    registry.register(
        "smart",
        AnthropicClient("claude-sonnet-4-6"),
        description="Balanced quality and cost. Good for most production tasks.",
        tags=["balanced"],
    )

    # Show what is in the registry
    print("Registered models:")
    for entry in registry.list():
        print(f"  name={entry.name!r:8}  model={entry.client.provider_name()!r:45}  tags={entry.tags}")

    # ---------------------------------------------------------------------------
    # 2. Query by tag
    # ---------------------------------------------------------------------------
    section("2. Querying by tag")

    cheap_models = registry.find_by_tag("cheap")
    print(f"Models tagged 'cheap':    {cheap_models}")

    balanced_models = registry.find_by_tag("balanced")
    print(f"Models tagged 'balanced': {balanced_models}")

    unknown_tag = registry.find_by_tag("vision")
    print(f"Models tagged 'vision':   {unknown_tag}  (empty — none registered with that tag)")

    # ---------------------------------------------------------------------------
    # 3. Single call with auto_execute=False (default)
    # ---------------------------------------------------------------------------
    section("3. Single call — auto_execute=False (default)")

    # No tools registered, so the model will just answer directly.
    # This is the simplest usage: one question, one answer.
    messages = [Message(role="user", content="What is 2 + 2? Answer with just the number.")]

    response = await registry.complete("fast", messages)

    print(f"Answer:         {response.text!r}")
    print(f"Stop reason:    {response.stop_reason}")
    print(f"Has tool calls: {response.has_tool_calls}")
    if response.usage:
        print(f"Tokens used:    in={response.usage.input_tokens}  out={response.usage.output_tokens}")

    # ---------------------------------------------------------------------------
    # 4. Compare two models on the same prompt
    # ---------------------------------------------------------------------------
    section("4. Same question, two different models")

    prompt = [Message(role="user", content="In one sentence, what is a neural network?")]

    fast_resp  = await registry.complete("fast",  prompt)
    smart_resp = await registry.complete("smart", prompt)

    print(f"fast  → {fast_resp.text}")
    print()
    print(f"smart → {smart_resp.text}")

    # ---------------------------------------------------------------------------
    # 5. Hot-swap a model
    # ---------------------------------------------------------------------------
    section("5. Hot-swapping a model at runtime")

    print("Before swap: 'fast' is", registry.get("fast").provider_name())

    # Imagine the haiku model goes down and you want to fall back to sonnet.
    # swap() replaces the client behind "fast" without touching any call sites.
    registry.swap("fast", AnthropicClient("claude-sonnet-4-6"))
    print("After  swap: 'fast' is", registry.get("fast").provider_name())
    print("All call sites that use registry.complete('fast', ...) now use sonnet — zero code changes.")


if __name__ == "__main__":
    asyncio.run(main())
