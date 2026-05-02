"""
Skills — named, reusable agent capabilities.

A Skill sits between a raw Tool and a full Pattern:
  - A Tool is a single function (search, write_file, send_email)
  - A Pattern is a coordination strategy (ReAct, planner-executor, etc.)
  - A Skill is a named capability that *uses* tools and patterns to do something
    specific and reusable ("research_topic", "write_report", "summarise_url")

Think of skills as the verbs your agent system knows how to perform.
Each skill encapsulates: what it does, what input it takes, and how it runs.

Industry context:
  Claude Code calls them "/skills", OpenAI calls them "GPT Actions",
  LangGraph calls them "subgraphs". The concept is universal — reusable
  agentic capabilities you can drop in anywhere.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class Skill:
    """
    A named, reusable agentic capability.

    Attributes:
        name:         Unique identifier, used to invoke the skill.
                      Must be a valid Python identifier (no spaces).
        description:  Human + LLM-readable description of what this skill does
                      and when to use it. This is what the LLM sees when it
                      decides which skill to invoke.
        input_schema: JSON Schema dict describing the expected input. Same
                      format as MCP tool input_schema — the LLM uses this to
                      construct the correct call.
        handler:      Async callable that receives the input dict and returns
                      a string result. This is where the agent/pattern lives.
        tags:         Optional labels for grouping and filtering skills
                      (e.g. ["research", "read-only"]).
        version:      Optional version string for tracking skill evolution.

    Example:
        async def _do_research(inputs: dict) -> str:
            agent = SingleAgentLoop(...)
            return await agent.run(inputs["topic"])

        skill = Skill(
            name="research_topic",
            description="Deep-dive research on any topic using web search and knowledge base.",
            input_schema={
                "type": "object",
                "properties": {"topic": {"type": "string", "description": "What to research"}},
                "required": ["topic"],
            },
            handler=_do_research,
            tags=["research", "read-only"],
        )
    """
    name:         str
    description:  str
    input_schema: dict[str, Any]
    handler:      Callable[[dict[str, Any]], Awaitable[str]]
    tags:         list[str] = field(default_factory=list)
    version:      str = "1.0.0"

    def __post_init__(self) -> None:
        if not self.name.replace("_", "").isalnum():
            raise ValueError(
                f"Skill name {self.name!r} must contain only letters, digits, and underscores."
            )
        if not self.description.strip():
            raise ValueError("Skill description must not be empty.")

    async def invoke(self, inputs: dict[str, Any]) -> str:
        """Execute this skill with the given inputs."""
        return await self.handler(inputs)


class SkillRegistry:
    """
    Central registry for all skills in an agent system.

    Acts like a plugin system — register skills once, invoke by name from
    anywhere. Agents and MCP tools can query the registry to discover what
    capabilities are available.

    Usage:
        registry = SkillRegistry()
        registry.register(research_skill)
        registry.register(summarise_skill)

        # List what's available
        for skill in registry.list_skills():
            print(skill.name, "-", skill.description)

        # Invoke by name
        result = await registry.invoke("research_topic", {"topic": "vector search"})

    Thread/task safety:
        SkillRegistry is not thread-safe for concurrent registrations.
        Register all skills at startup; invoke concurrently freely.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a skill. Raises ValueError if the name is already taken."""
        if skill.name in self._skills:
            raise ValueError(
                f"A skill named {skill.name!r} is already registered. "
                "Use replace() to overwrite it explicitly."
            )
        self._skills[skill.name] = skill

    def replace(self, skill: Skill) -> None:
        """Register or overwrite a skill regardless of whether it exists."""
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> None:
        """Remove a skill by name. Raises KeyError if not found."""
        if name not in self._skills:
            raise KeyError(f"No skill named {name!r} is registered.")
        del self._skills[name]

    def get(self, name: str) -> Skill:
        """Return a skill by name. Raises KeyError if not found."""
        if name not in self._skills:
            raise KeyError(
                f"No skill named {name!r}. "
                f"Available: {list(self._skills)}"
            )
        return self._skills[name]

    def list_skills(self, tag: str | None = None) -> list[Skill]:
        """
        Return all registered skills, optionally filtered by tag.

        Args:
            tag: If provided, only return skills whose tags list includes this value.
        """
        skills = list(self._skills.values())
        if tag:
            skills = [s for s in skills if tag in s.tags]
        return skills

    def as_mcp_tools(self) -> list[dict[str, Any]]:
        """
        Return skill descriptions in MCP tool format.

        Useful when you want to expose the registry itself as tools to an LLM,
        so the LLM can discover available skills and call them directly.
        """
        return [
            {
                "name":         skill.name,
                "description":  skill.description,
                "inputSchema":  skill.input_schema,
            }
            for skill in self._skills.values()
        ]

    async def invoke(self, name: str, inputs: dict[str, Any]) -> str:
        """
        Invoke a skill by name with the given inputs.

        Raises:
            KeyError:  if no skill with that name is registered.
            Exception: any exception raised by the skill handler.
        """
        return await self.get(name).invoke(inputs)

    async def invoke_many(
        self, calls: list[tuple[str, dict[str, Any]]]
    ) -> list[str]:
        """
        Invoke multiple skills concurrently.

        Args:
            calls: list of (skill_name, inputs) pairs.

        Returns:
            list of results in the same order as calls.
        """
        tasks = [self.invoke(name, inputs) for name, inputs in calls]
        return list(await asyncio.gather(*tasks))

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __repr__(self) -> str:
        names = list(self._skills)
        return f"SkillRegistry({names})"
