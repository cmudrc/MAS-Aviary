"""Agent factory — creates smolagents ToolCallingAgent instances from config."""

from typing import Any, Callable

from smolagents import Tool, ToolCallingAgent
from smolagents.models import Model

from src.agents.agent_registry import AgentRegistry
from src.config.loader import AppConfig, load_yaml
from src.tools.tool_loader import load_tools_for_agent


def create_agent(
    name: str,
    role: str,
    system_prompt: str,
    tools: list[Tool],
    model: Model,
    max_steps: int = 8,
    final_answer_checks: list[Callable] | None = None,
) -> ToolCallingAgent:
    """Create a single ToolCallingAgent from explicit parameters.

    This is the programmatic Python API for agent creation.
    """
    agent = ToolCallingAgent(
        tools=tools,
        model=model,
        name=name,
        description=role,
        instructions=system_prompt,
        max_steps=max_steps,
        add_base_tools=False,
        final_answer_checks=final_answer_checks,
    )
    return agent


def create_agent_from_dict(
    agent_def: dict[str, Any],
    model: Model,
    config: AppConfig,
    final_answer_checks: list[Callable] | None = None,
) -> ToolCallingAgent:
    """Create a ToolCallingAgent from a single agent definition dict.

    Expected keys in agent_def: name, role, system_prompt, tools (list of
    tool name strings), max_steps (optional, default 8).
    """
    tool_names = agent_def.get("tools", [])
    # Only load tools when explicitly listed.  An empty list means "no tools"
    # (orchestrator tools, for example, are injected later by the strategy).
    # load_tools_for_agent([]) in real-MCP mode returns ALL tools, which would
    # give the orchestrator worker tools it shouldn't have.
    tools = load_tools_for_agent(tool_names, config) if tool_names else []

    return create_agent(
        name=agent_def["name"],
        role=agent_def["role"],
        system_prompt=agent_def.get("system_prompt", ""),
        tools=tools,
        model=model,
        max_steps=agent_def.get("max_steps", 8),
        final_answer_checks=final_answer_checks,
    )


def create_agents_from_yaml(
    yaml_path: str,
    model: Model,
    config: AppConfig,
) -> AgentRegistry:
    """Load agent definitions from a YAML file and return a populated AgentRegistry.

    The YAML file should have an 'agents' key containing a list of agent dicts.
    """
    data = load_yaml(yaml_path)
    agent_defs = data.get("agents", [])

    registry = AgentRegistry()
    for agent_def in agent_defs:
        agent = create_agent_from_dict(agent_def, model, config)
        registry.register(agent)

    return registry
