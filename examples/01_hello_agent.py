"""
=============================================================================
EXAMPLE 01: Hello Agent - the simplest possible AI agent
=============================================================================

WHAT IS AN AGENT?
-----------------
An "agent" is a program that lets an AI model take actions in the world, not
just answer questions. Instead of a single question → answer exchange, the
model can call tools (functions you define), look at the results, and decide
what to do next - all in a loop.

WHAT IS THE REACT LOOP?
-----------------------
ReAct stands for Reason + Act. The loop works like this:

  1. OBSERVE  - The model sees your question and the list of available tools.
  2. THINK    - It reasons about what action (if any) would help answer the question.
  3. ACT      - It calls a tool. Your code runs the tool and returns the result.
  4. REPEAT   - The model sees the tool result and either calls another tool
                or writes its final answer.

This "observe → think → act → repeat" cycle is what separates an agent from
a simple chatbot. The model can take MULTIPLE steps before finishing.

WHY DOES IT NEED AN API KEY?
----------------------------
Every call to the model (Claude, GPT, Gemini, etc.) goes to a cloud API that
charges by the number of tokens processed. The API key is how the provider
authenticates you and bills you. Keep it in environment variables - NEVER
hardcode it in source code.

IN-PROCESS SERVER
-----------------
We create the MCP server using FastMCP directly in this file. No separate
process or network connection needed. The framework's Client() recognises a
FastMCP object and wires it up automatically.

Run:
    ANTHROPIC_API_KEY=sk-ant-... python examples/01_hello_agent.py
=============================================================================
"""

import asyncio
import os

from fastmcp import FastMCP

from mcp_agent_framework import AgentConfig, AnthropicClient, Message, SingleAgentLoop

# ---------------------------------------------------------------------------
# 1. Define an in-process MCP server with one tool
# ---------------------------------------------------------------------------
# FastMCP is the server framework. Any function decorated with @app.tool
# becomes a capability the agent can call. The docstring becomes the
# description the model reads to decide when to use the tool.

app = FastMCP("weather_tools")


@app.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Stub - returns fake data so this example works without a real weather API.
    # In a real app you would call OpenWeatherMap, WeatherAPI, etc. here.
    weather_db = {
        "paris":  "Sunny and 22°C with a light breeze.",
        "tokyo":  "Partly cloudy and 18°C. Humidity 65%.",
        "london": "Overcast and 12°C. Light rain expected.",
    }
    return weather_db.get(city.lower(), f"Weather data for {city} is unavailable right now.")


# ---------------------------------------------------------------------------
# 2. Run the agent
# ---------------------------------------------------------------------------

async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Get your key at https://console.anthropic.com and run:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-..."
        )
        return

    # AnthropicClient wraps the Anthropic SDK. We use haiku - it is the
    # cheapest and fastest Claude model, perfect for examples and demos.
    client = AnthropicClient("claude-haiku-4-5-20251001")

    # AgentConfig bundles everything the agent needs to know:
    #   mcp_server_config - pass the FastMCP app object for in-process use
    #   system_prompt     - instructions given to the model before the conversation
    #   max_iterations    - safety cap: stop after N tool-call rounds
    config = AgentConfig(
        mcp_server_config=app,
        system_prompt="You are a helpful weather assistant. Always call the get_weather tool to get accurate data.",
        max_iterations=5,
    )

    # SingleAgentLoop is the ReAct pattern: one LLM + one set of tools,
    # looping until the model stops requesting tool calls.
    agent = SingleAgentLoop(llm_client=client, config=config)

    question = "What's the weather like in Paris and Tokyo?"
    print(f"Question: {question}")
    print("-" * 60)

    # agent.run() handles the entire ReAct loop for you:
    #   - Sends the question to Claude with the tool list
    #   - Calls get_weather for each city when Claude requests it
    #   - Feeds the results back to Claude
    #   - Returns Claude's final text answer
    result = await agent.run(question)

    print(result)
    print("-" * 60)
    print("Done! The agent called get_weather for each city, got the results,")
    print("and composed the answer above - all automatically.")


if __name__ == "__main__":
    asyncio.run(main())
