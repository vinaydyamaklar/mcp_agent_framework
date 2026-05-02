"""
Skills — named, reusable agentic capabilities.

The skills layer sits between raw tools and full agent patterns:

  Tools     — single functions (search, write_file, send_email)
  Skills    — named capabilities that compose tools + patterns
  Patterns  — coordination strategies (ReAct, planner-executor, etc.)

Key classes:
    Skill            — a named, typed, invokable capability
    SkillRegistry    — discover, register, and invoke skills by name
    SkillAwareAgent  — agent that auto-wires the registry as MCP tools

Quick start:
    from mcp_agent_framework.skills import Skill, SkillRegistry, SkillAwareAgent

    async def _research(inputs):
        # your SingleAgentLoop / pattern here
        return "research result"

    skill = Skill(
        name="research_topic",
        description="Research any topic deeply.",
        input_schema={"type": "object", "properties": {"topic": {"type": "string"}}},
        handler=_research,
    )
    registry = SkillRegistry()
    registry.register(skill)

    agent = SkillAwareAgent(llm_client=..., registry=registry)
    result = await agent.run("Research vector databases")
"""

from mcp_agent_framework.skills.skill import Skill, SkillRegistry
from mcp_agent_framework.skills.skill_aware_agent import SkillAwareAgent

__all__ = [
    "Skill",
    "SkillRegistry",
    "SkillAwareAgent",
]
