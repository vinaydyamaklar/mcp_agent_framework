"""
SkillAwareAgent — an agent that knows about a SkillRegistry.

This wraps any agent pattern (defaulting to SingleAgentLoop) and auto-wires
the skill registry as two MCP tools visible to the LLM:

  list_skills()          — returns a description of every available skill
  invoke_skill(name, inputs_json) — calls a skill by name with JSON inputs

The LLM can then decide to call a skill the same way it would call any tool.
The skill runs its underlying pattern (SingleAgentLoop, PlannerExecutor, etc.)
and the result comes back as a tool response.

Architecture:

  User message
      │
      ▼
  SkillAwareAgent
      │  builds in-process FastMCP server with:
      │    - list_skills tool
      │    - invoke_skill tool  (each call runs the real skill handler)
      │    - + any extra tools from a second MCP server you pass in
      ▼
  SingleAgentLoop (or your chosen pattern)
      │
      └── tool calls → registry.invoke(name, inputs)
"""
from __future__ import annotations

import json
from typing import Any

from fastmcp import FastMCP

from mcp_agent_framework.clients.base_client import BaseLLMClient
from mcp_agent_framework.patterns.single_agent_loop import SingleAgentLoop
from mcp_agent_framework.skills.skill import SkillRegistry
from mcp_agent_framework.types import AgentConfig


class SkillAwareAgent:
    """
    An agent that exposes a SkillRegistry as MCP tools so the LLM can
    discover and invoke skills by name.

    Usage:
        registry = SkillRegistry()
        registry.register(research_skill)
        registry.register(summarise_skill)

        agent = SkillAwareAgent(
            llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
            registry=registry,
            system_prompt="You are a research assistant. Use skills to answer questions.",
        )
        result = await agent.run("Research quantum computing and write a summary.")

    How the LLM uses skills:
        1. LLM calls list_skills() → sees all available skills with descriptions
        2. LLM calls invoke_skill("research_topic", '{"topic": "quantum computing"}')
        3. invoke_skill runs the skill handler (which may itself run a SingleAgentLoop)
        4. Result comes back as a tool response — LLM continues from there

    Composability:
        Pass extra_mcp_server to give the agent additional domain tools alongside
        the skill tools. The agent sees all tools in the same namespace.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        registry: SkillRegistry,
        system_prompt: str = "",
        max_iterations: int = 50,
        extra_mcp_server: Any | None = None,
    ) -> None:
        self._llm            = llm_client
        self._registry       = registry
        self._system_prompt  = system_prompt
        self._max_iterations = max_iterations
        self._extra          = extra_mcp_server

        self._mcp = self._build_mcp_server()

    def _build_mcp_server(self) -> FastMCP:
        """Build an in-process FastMCP server that exposes the registry."""
        registry = self._registry

        # Use a unique server name so multiple agents don't collide
        app = FastMCP("skill_registry")

        @app.tool
        async def list_skills(tag: str = "") -> str:
            """
            List all available skills. Pass a tag to filter by category.
            Returns skill names, descriptions, and input schemas.
            """
            skills = registry.list_skills(tag=tag or None)
            if not skills:
                return "No skills are registered yet."
            lines = []
            for skill in skills:
                schema_str = json.dumps(skill.input_schema, indent=2)
                tags_str   = f"  tags: {skill.tags}" if skill.tags else ""
                lines.append(
                    f"Skill: {skill.name}\n"
                    f"  Description: {skill.description}{tags_str}\n"
                    f"  Input schema:\n{schema_str}"
                )
            return "\n\n".join(lines)

        @app.tool
        async def invoke_skill(name: str, inputs_json: str = "{}") -> str:
            """
            Invoke a skill by name with JSON-encoded inputs.

            Args:
                name:        The skill name (from list_skills).
                inputs_json: JSON string with the input arguments.
                             Example: '{"topic": "vector search", "depth": "deep"}'
            """
            try:
                inputs = json.loads(inputs_json)
            except json.JSONDecodeError as e:
                return f"Error: inputs_json is not valid JSON — {e}"

            if name not in registry:
                available = [s.name for s in registry.list_skills()]
                return (
                    f"Error: no skill named '{name}'. "
                    f"Available skills: {available}. "
                    "Call list_skills() to see details."
                )

            try:
                result = await registry.invoke(name, inputs)
                return result
            except Exception as e:
                return f"Error running skill '{name}': {e}"

        # If the caller provided an extra MCP server, mount it
        # Note: FastMCP supports importing tools from another app
        if self._extra is not None:
            app.mount("/extra", self._extra)

        return app

    async def run(self, user_message: str) -> str:
        """
        Run the agent on a user message. The LLM has access to list_skills and
        invoke_skill tools and will use them as needed.
        """
        config = AgentConfig(
            mcp_server_config=self._mcp,
            system_prompt=self._system_prompt or (
                "You are a capable assistant. "
                "Use list_skills() to discover available capabilities, "
                "then invoke_skill() to execute them. "
                "Compose skills as needed to answer the user's request."
            ),
            max_iterations=self._max_iterations,
        )
        agent = SingleAgentLoop(llm_client=self._llm, config=config)
        return await agent.run(user_message)
