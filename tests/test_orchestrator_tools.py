"""Tests for orchestrator tools — ListAvailableTools, CreateAgent, AssignTask.

No GPU needed. Uses DummyModel stub and mock tools.
"""

import json

import pytest
from smolagents import ToolCallingAgent
from smolagents.models import Model

from src.tools.mock_tools import CalculatorTool, EchoTool, StateTool
from src.tools.orchestrator_tools import (
    AssignTask,
    CreateAgent,
    ListAvailableTools,
    OrchestratorContext,
    _parse_tool_names,
)

# ---- Fixtures ----------------------------------------------------------------

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
def available_tools():
    """All mock tools available for worker assignment."""
    echo = EchoTool()
    calc = CalculatorTool()
    state = StateTool()
    return {echo.name: echo, calc.name: calc, state.name: state}


@pytest.fixture
def context(dummy_model, available_tools):
    """Fresh OrchestratorContext for each test."""
    return OrchestratorContext(
        available_tools=available_tools,
        agents={"orchestrator": "placeholder"},  # pre-existing system agent
        model=dummy_model,
        system_agent_names={"orchestrator"},
        max_agents=3,
        worker_max_steps=5,
    )


# ---- ListAvailableTools tests ------------------------------------------------

class TestListAvailableTools:
    def test_returns_all_tools(self, context):
        tool = ListAvailableTools(context)
        result = json.loads(tool.forward())
        assert result["total_count"] == 3
        names = {t["name"] for t in result["tools"]}
        assert names == {"echo_tool", "calculator_tool", "state_tool"}

    def test_includes_descriptions(self, context):
        tool = ListAvailableTools(context)
        result = json.loads(tool.forward())
        for t in result["tools"]:
            assert "description" in t
            assert len(t["description"]) > 0

    def test_includes_input_schema(self, context):
        tool = ListAvailableTools(context)
        result = json.loads(tool.forward())
        calc = next(t for t in result["tools"] if t["name"] == "calculator_tool")
        assert "expression" in calc["input_schema"]

    def test_empty_tools(self, dummy_model):
        ctx = OrchestratorContext(
            available_tools={},
            agents={},
            model=dummy_model,
        )
        tool = ListAvailableTools(ctx)
        result = json.loads(tool.forward())
        assert result["total_count"] == 0
        assert result["tools"] == []


# ---- CreateAgent tests -------------------------------------------------------

class TestCreateAgent:
    def test_creates_agent_successfully(self, context):
        tool = CreateAgent(context)
        result = json.loads(tool(
            name="coder",
            persona="You write Python code",
            tools=["calculator_tool"],
        ))
        assert result["success"] is True
        assert result["agent_name"] == "coder"
        assert result["tools_assigned"] == ["calculator_tool"]
        assert result["error"] is None

    def test_agent_registered_in_pool(self, context):
        tool = CreateAgent(context)
        tool(name="coder", persona="Writes code", tools=["echo_tool"])
        assert "coder" in context.agents
        assert isinstance(context.agents["coder"], ToolCallingAgent)

    def test_agent_added_to_created_list(self, context):
        tool = CreateAgent(context)
        tool(name="coder", persona="Writes code", tools=["echo_tool"])
        assert "coder" in context.created_agents

    def test_agent_has_correct_tools(self, context):
        tool = CreateAgent(context)
        tool(name="coder", persona="Writes code", tools=["calculator_tool", "echo_tool"])
        agent = context.agents["coder"]
        assert "calculator_tool" in agent.tools
        assert "echo_tool" in agent.tools

    def test_duplicate_name_returns_success(self, context):
        """Re-creating an existing agent is a graceful no-op (not an error)."""
        tool = CreateAgent(context)
        tool(name="coder", persona="v1", tools=["echo_tool"])
        result = json.loads(tool(name="coder", persona="v2", tools=["echo_tool"]))
        assert result["success"] is True
        assert "already exists" in result["note"]

    def test_rejects_unknown_tool(self, context):
        tool = CreateAgent(context)
        result = json.loads(tool(
            name="coder",
            persona="Writes code",
            tools=["nonexistent_tool"],
        ))
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_rejects_when_max_agents_reached(self, context):
        tool = CreateAgent(context)
        # max_agents=3 in fixture
        tool(name="a1", persona="Agent 1", tools=["echo_tool"])
        tool(name="a2", persona="Agent 2", tools=["echo_tool"])
        tool(name="a3", persona="Agent 3", tools=["echo_tool"])
        result = json.loads(tool(name="a4", persona="Agent 4", tools=["echo_tool"]))
        assert result["success"] is False
        assert "Maximum agent limit" in result["error"]

    def test_tools_as_json_string(self, context):
        tool = CreateAgent(context)
        result = json.loads(tool(
            name="coder",
            persona="Writes code",
            tools='["echo_tool", "calculator_tool"]',
        ))
        assert result["success"] is True
        assert result["tools_assigned"] == ["echo_tool", "calculator_tool"]

    def test_tools_as_csv_string(self, context):
        tool = CreateAgent(context)
        result = json.loads(tool(
            name="coder",
            persona="Writes code",
            tools="echo_tool, calculator_tool",
        ))
        assert result["success"] is True
        assert result["tools_assigned"] == ["echo_tool", "calculator_tool"]

    def test_agent_uses_shared_model(self, context):
        tool = CreateAgent(context)
        tool(name="coder", persona="Writes code", tools=["echo_tool"])
        agent = context.agents["coder"]
        assert agent.model is context.model


# ---- AssignTask tests --------------------------------------------------------

class TestAssignTask:
    def test_assigns_task_successfully(self, context):
        # First create an agent
        CreateAgent(context)(name="worker", persona="A worker", tools=["echo_tool"])
        tool = AssignTask(context)
        result = json.loads(tool(agent_name="worker", task="Do something"))
        assert result["success"] is True
        assert result["agent_name"] == "worker"
        assert result["task"] == "Do something"
        assert result["queue_position"] == 1

    def test_records_assignment_in_queue(self, context):
        CreateAgent(context)(name="w1", persona="Worker", tools=["echo_tool"])
        tool = AssignTask(context)
        tool(agent_name="w1", task="Task 1")
        assert len(context.assignments) == 1
        assert context.assignments[0]["agent_name"] == "w1"
        assert context.assignments[0]["task"] == "Task 1"

    def test_multiple_assignments(self, context):
        CreateAgent(context)(name="w1", persona="Worker 1", tools=["echo_tool"])
        CreateAgent(context)(name="w2", persona="Worker 2", tools=["echo_tool"])
        tool = AssignTask(context)
        tool(agent_name="w1", task="Task A")
        r2 = json.loads(tool(agent_name="w2", task="Task B"))
        assert r2["queue_position"] == 2
        assert len(context.assignments) == 2

    def test_rejects_unknown_agent(self, context):
        tool = AssignTask(context)
        result = json.loads(tool(agent_name="ghost", task="Do something"))
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_rejects_system_agent(self, context):
        tool = AssignTask(context)
        result = json.loads(tool(agent_name="orchestrator", task="Do something"))
        assert result["success"] is False
        assert "system agent" in result["error"]

    def test_assignment_records_turn(self, context):
        context.turn_counter = 7
        CreateAgent(context)(name="w", persona="Worker", tools=["echo_tool"])
        tool = AssignTask(context)
        tool(agent_name="w", task="Do it")
        assert context.assignments[0]["assigned_at_turn"] == 7


# ---- _parse_tool_names tests ------------------------------------------------

class TestParseToolNames:
    def test_list_input(self):
        assert _parse_tool_names(["a", "b"]) == ["a", "b"]

    def test_json_string(self):
        assert _parse_tool_names('["a", "b"]') == ["a", "b"]

    def test_csv_string(self):
        assert _parse_tool_names("a, b, c") == ["a", "b", "c"]

    def test_single_string(self):
        assert _parse_tool_names("tool_name") == ["tool_name"]

    def test_strips_whitespace(self):
        assert _parse_tool_names(["  a ", " b"]) == ["a", "b"]

    def test_filters_empty(self):
        assert _parse_tool_names(["a", "", "b"]) == ["a", "b"]
