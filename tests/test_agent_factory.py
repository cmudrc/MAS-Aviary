"""Tests for agent factory and registry.

Uses mock tools only — no LLM needed for agent *creation*.
We provide a dummy model to satisfy the constructor.
"""

import pytest
import yaml
from smolagents import ToolCallingAgent
from smolagents.models import Model

from src.agents.agent_factory import (
    create_agent,
    create_agent_from_dict,
    create_agents_from_yaml,
)
from src.agents.agent_registry import AgentRegistry
from src.config.loader import AppConfig
from src.tools.mock_tools import EchoTool

# ---- Fixtures -----------------------------------------------------------------

class DummyModel(Model):
    """Minimal model stub that satisfies the ToolCallingAgent constructor."""
    def __init__(self):
        super().__init__(model_id="dummy")

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs):
        from smolagents.types import ChatMessage
        return ChatMessage(role="assistant", content="dummy response")


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def mock_config():
    return AppConfig()  # defaults to mock mode


# ---- AgentRegistry tests ------------------------------------------------------

class TestAgentRegistry:
    def test_register_and_get(self, dummy_model):
        registry = AgentRegistry()
        agent = ToolCallingAgent(
            tools=[EchoTool()], model=dummy_model, name="test_agent",
            description="test", add_base_tools=False,
        )
        registry.register(agent)
        assert registry.get("test_agent") is agent

    def test_duplicate_name_raises(self, dummy_model):
        registry = AgentRegistry()
        agent = ToolCallingAgent(
            tools=[EchoTool()], model=dummy_model, name="dup",
            description="test", add_base_tools=False,
        )
        registry.register(agent)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(agent)

    def test_unknown_name_raises(self):
        registry = AgentRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("ghost")

    def test_list_names(self, dummy_model):
        registry = AgentRegistry()
        for name in ["a", "b", "c"]:
            agent = ToolCallingAgent(
                tools=[], model=dummy_model, name=name,
                description=name, add_base_tools=False,
            )
            registry.register(agent)
        assert registry.list_names() == ["a", "b", "c"]

    def test_len_and_contains(self, dummy_model):
        registry = AgentRegistry()
        agent = ToolCallingAgent(
            tools=[], model=dummy_model, name="x",
            description="x", add_base_tools=False,
        )
        registry.register(agent)
        assert len(registry) == 1
        assert "x" in registry
        assert "y" not in registry


# ---- create_agent tests -------------------------------------------------------

class TestCreateAgent:
    def test_creates_tool_calling_agent(self, dummy_model):
        agent = create_agent(
            name="planner",
            role="Plans tasks",
            system_prompt="You are a planner.",
            tools=[EchoTool()],
            model=dummy_model,
            max_steps=5,
        )
        assert isinstance(agent, ToolCallingAgent)
        assert agent.name == "planner"
        assert agent.description == "Plans tasks"
        assert agent.max_steps == 5

    def test_default_max_steps(self, dummy_model):
        agent = create_agent(
            name="test", role="test", system_prompt="", tools=[], model=dummy_model,
        )
        assert agent.max_steps == 8


# ---- create_agent_from_dict tests ---------------------------------------------

class TestCreateAgentFromDict:
    def test_from_dict_with_mock_tools(self, dummy_model, mock_config):
        agent_def = {
            "name": "executor",
            "role": "Executes tasks",
            "system_prompt": "Reasoning: medium\nYou are an executor.",
            "tools": ["echo_tool", "calculator_tool"],
            "max_steps": 6,
        }
        agent = create_agent_from_dict(agent_def, dummy_model, mock_config)
        assert agent.name == "executor"
        # smolagents adds final_answer automatically, so +1
        assert "echo_tool" in agent.tools
        assert "calculator_tool" in agent.tools

    def test_from_dict_no_tools(self, dummy_model, mock_config):
        agent_def = {"name": "bare", "role": "No tools", "tools": []}
        agent = create_agent_from_dict(agent_def, dummy_model, mock_config)
        assert agent.name == "bare"
        # smolagents always adds final_answer
        assert "final_answer" in agent.tools


# ---- create_agents_from_yaml tests --------------------------------------------

class TestCreateAgentsFromYaml:
    def test_loads_agents_yaml(self, dummy_model, mock_config):
        registry = create_agents_from_yaml("config/agents.yaml", dummy_model, mock_config)
        assert isinstance(registry, AgentRegistry)
        assert len(registry) == 3
        assert "planner" in registry
        assert "executor" in registry
        assert "reviewer" in registry

    def test_agent_properties_from_yaml(self, dummy_model, mock_config):
        registry = create_agents_from_yaml("config/agents.yaml", dummy_model, mock_config)
        planner = registry.get("planner")
        assert planner.max_steps == 5
        executor = registry.get("executor")
        assert "calculator_tool" in executor.tools
        assert "state_tool" in executor.tools

    def test_custom_yaml(self, tmp_path, dummy_model, mock_config):
        custom = tmp_path / "custom_agents.yaml"
        custom.write_text(yaml.dump({
            "agents": [
                {"name": "solo", "role": "Single agent", "tools": ["echo_tool"], "max_steps": 3},
            ]
        }))
        registry = create_agents_from_yaml(str(custom), dummy_model, mock_config)
        assert len(registry) == 1
        assert registry.get("solo").max_steps == 3
