"""Tests for the new template-based SequentialStrategy.

No GPU needed. Uses DummyModel stub and mock tools.
"""

import json
import time

import pytest
from smolagents import ToolCallingAgent
from smolagents.models import Model

from src.coordination.history import AgentMessage
from src.coordination.strategies.sequential import (
    SequentialStrategy,
    _parse_planner_output,
    _validate_interface,
)
from src.tools.mock_tools import CalculatorTool, EchoTool, StateTool

# ---- Fixtures ----------------------------------------------------------------

class DummyModel(Model):
    """Minimal model stub."""

    def __init__(self):
        super().__init__(model_id="dummy")

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs):
        from smolagents.models import ChatMessage
        return ChatMessage(role="assistant", content="dummy response")


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def worker_tools():
    return {
        "echo_tool": EchoTool(),
        "calculator_tool": CalculatorTool(),
        "state_tool": StateTool(),
    }


def _make_config(worker_tools, **overrides):
    """Build a minimal config dict for SequentialStrategy."""
    seq = {
        "decomposition_mode": "human",
        "pipeline_template": "linear",
        "validate_interfaces": False,
        "stage_max_steps": 4,
    }
    seq.update(overrides)
    return {
        "sequential": seq,
        "termination": {
            "keyword": "TASK_COMPLETE",
            "max_turns": 20,
            "max_consecutive_errors": 3,
        },
        "_worker_tools": worker_tools,
        "stage_defaults": {
            "base_instructions": "You are one stage in a sequential pipeline.",
        },
    }


def _msg(agent, content, turn, error=None):
    return AgentMessage(
        agent_name=agent,
        content=content,
        turn_number=turn,
        timestamp=time.time(),
        error=error,
    )


# ---- Human mode initialization -----------------------------------------------

class TestHumanModeInit:
    def test_creates_agents_from_linear(self, dummy_model, worker_tools):
        strategy = SequentialStrategy()
        config = _make_config(worker_tools, pipeline_template="linear")
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        assert set(agents.keys()) == {"planner", "executor", "reviewer"}
        assert strategy.stage_order == ["planner", "executor", "reviewer"]

    def test_creates_agents_from_v_model(self, dummy_model, worker_tools):
        strategy = SequentialStrategy()
        config = _make_config(worker_tools, pipeline_template="v_model")
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        assert len(agents) == 5
        assert strategy.stage_order == [
            "requirements_analyst", "system_designer",
            "detailed_designer", "implementer", "integration_verifier",
        ]

    def test_creates_agents_from_mbse(self, dummy_model, worker_tools):
        strategy = SequentialStrategy()
        config = _make_config(worker_tools, pipeline_template="mbse")
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        assert len(agents) == 5

    def test_agents_are_tool_calling_agents(self, dummy_model, worker_tools):
        strategy = SequentialStrategy()
        config = _make_config(worker_tools)
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        for agent in agents.values():
            assert isinstance(agent, ToolCallingAgent)

    def test_tool_restriction_no_tools(self, dummy_model, worker_tools):
        """Planner stage should have no domain tools."""
        strategy = SequentialStrategy()
        config = _make_config(worker_tools, pipeline_template="linear")
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        planner = agents["planner"]
        # Planner should have no domain tools (only final_answer from smolagents)
        domain_names = {"echo_tool", "calculator_tool", "state_tool"}
        agent_tool_names = set(planner.tools.keys())
        assert not (agent_tool_names & domain_names)

    def test_tool_restriction_all_tools(self, dummy_model, worker_tools):
        """Executor stage should have all domain tools."""
        strategy = SequentialStrategy()
        config = _make_config(worker_tools, pipeline_template="linear")
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        executor = agents["executor"]
        assert "calculator_tool" in executor.tools
        assert "echo_tool" in executor.tools
        assert "state_tool" in executor.tools

    def test_custom_template(self, dummy_model, worker_tools):
        strategy = SequentialStrategy()
        config = _make_config(
            worker_tools,
            pipeline_template="custom",
            custom_stages=[
                {"name": "a", "role": "Do A", "allowed_tools": [],
                 "interface_output": "A output"},
                {"name": "b", "role": "Do B", "allowed_tools": ["*"],
                 "interface_output": "B output"},
            ],
        )
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        assert strategy.stage_order == ["a", "b"]
        assert len(agents) == 2

    def test_template_stored(self, dummy_model, worker_tools):
        strategy = SequentialStrategy()
        config = _make_config(worker_tools, pipeline_template="linear")
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        assert strategy.template is not None
        assert strategy.template.name == "linear"


# ---- LLM mode initialization ------------------------------------------------

class _MockPlannerModel(Model):
    """Model that returns a JSON pipeline when called."""

    def __init__(self, output: str):
        super().__init__(model_id="mock-planner")
        self._output = output

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs):
        from smolagents.models import ChatMessage
        return ChatMessage(role="assistant", content=self._output)


class TestLLMMode:
    def test_unparseable_output_raises(self, worker_tools):
        """LLM mode must raise on unparseable output, not fall back."""
        model = _MockPlannerModel("This is not JSON at all")
        strategy = SequentialStrategy()
        config = _make_config(worker_tools, decomposition_mode="llm")
        config["_model"] = model
        config["_task"] = "Test task"
        agents = {}
        with pytest.raises(ValueError, match="could not be parsed"):
            strategy.initialize(agents, config)

    def test_invalid_decomposition_mode_raises(self, dummy_model, worker_tools):
        strategy = SequentialStrategy()
        config = _make_config(worker_tools, decomposition_mode="invalid")
        config["_model"] = dummy_model
        agents = {}
        with pytest.raises(ValueError, match="Unknown decomposition_mode"):
            strategy.initialize(agents, config)


# ---- next_step tests ---------------------------------------------------------

class TestNextStep:
    def _init(self, dummy_model, worker_tools, **overrides):
        strategy = SequentialStrategy()
        config = _make_config(worker_tools, **overrides)
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        return strategy, agents

    def test_first_step_returns_first_stage(self, dummy_model, worker_tools):
        strategy, agents = self._init(dummy_model, worker_tools)
        action = strategy.next_step([], {"task": "Calculate 2+2"})
        assert action.action_type == "invoke_agent"
        assert action.agent_name == "planner"

    def test_first_step_uses_task(self, dummy_model, worker_tools):
        strategy, agents = self._init(dummy_model, worker_tools)
        action = strategy.next_step([], {"task": "Calculate 2+2"})
        assert "Calculate 2+2" in action.input_context

    def test_advances_to_next_stage(self, dummy_model, worker_tools):
        strategy, agents = self._init(dummy_model, worker_tools)
        a1 = strategy.next_step([], {"task": "Do it"})
        assert a1.agent_name == "planner"
        history = [_msg("planner", "Here's my plan", 1)]
        a2 = strategy.next_step(history, {})
        assert a2.agent_name == "executor"

    def test_passes_previous_output_with_interface(self, dummy_model, worker_tools):
        strategy, agents = self._init(dummy_model, worker_tools)
        strategy.next_step([], {"task": "Do it"})
        history = [_msg("planner", "Step 1: compute", 1)]
        a2 = strategy.next_step(history, {})
        assert "planner" in a2.input_context
        assert "Step 1: compute" in a2.input_context
        assert "expected format" in a2.input_context.lower()

    def test_terminates_after_all_stages(self, dummy_model, worker_tools):
        strategy, agents = self._init(dummy_model, worker_tools)
        state = {"task": "Do it"}
        history = []
        for i in range(3):
            action = strategy.next_step(history, state)
            assert action.action_type == "invoke_agent"
            history.append(_msg(action.agent_name, f"output_{i}", i + 1))

        action = strategy.next_step(history, state)
        assert action.action_type == "terminate"
        assert action.metadata["reason"] == "pipeline_complete"

    def test_metadata_has_stage_info(self, dummy_model, worker_tools):
        strategy, agents = self._init(dummy_model, worker_tools)
        action = strategy.next_step([], {"task": "Do it"})
        assert action.metadata["stage_name"] == "planner"
        assert action.metadata["stage_index"] == 0
        assert action.metadata["total_stages"] == 3
        assert action.metadata["pipeline_template"] == "linear"

    def test_v_model_5_stages_in_order(self, dummy_model, worker_tools):
        strategy, agents = self._init(
            dummy_model, worker_tools, pipeline_template="v_model",
        )
        state = {"task": "Build it"}
        expected = [
            "requirements_analyst", "system_designer",
            "detailed_designer", "implementer", "integration_verifier",
        ]
        history = []
        for i, name in enumerate(expected):
            action = strategy.next_step(history, state)
            assert action.agent_name == name
            history.append(_msg(name, f"output_{i}", i + 1))


# ---- is_complete tests -------------------------------------------------------

class TestIsComplete:
    def _init(self, dummy_model, worker_tools, **overrides):
        strategy = SequentialStrategy()
        config = _make_config(worker_tools, **overrides)
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        return strategy

    def test_complete_after_all_stages(self, dummy_model, worker_tools):
        strategy = self._init(dummy_model, worker_tools)
        state = {"task": "Do it"}
        history = []
        for i in range(3):
            strategy.next_step(history, state)
            history.append(_msg(f"agent_{i}", f"out_{i}", i + 1))
        assert strategy.is_complete(history, state) is True

    def test_complete_on_keyword_mid_pipeline(self, dummy_model, worker_tools):
        strategy = self._init(dummy_model, worker_tools)
        strategy.next_step([], {"task": "Do it"})
        history = [_msg("planner", "TASK_COMPLETE", 1)]
        assert strategy.is_complete(history, {}) is True

    def test_complete_on_max_turns(self, dummy_model, worker_tools):
        strategy = self._init(dummy_model, worker_tools)
        history = [_msg("a", "msg", i) for i in range(20)]
        assert strategy.is_complete(history, {}) is True

    def test_not_complete_mid_pipeline(self, dummy_model, worker_tools):
        strategy = self._init(dummy_model, worker_tools)
        strategy.next_step([], {"task": "Do it"})
        history = [_msg("planner", "plan", 1)]
        assert strategy.is_complete(history, {}) is False


# ---- Interface validation ----------------------------------------------------

class TestInterfaceValidation:
    def _init(self, dummy_model, worker_tools):
        strategy = SequentialStrategy()
        config = _make_config(worker_tools, validate_interfaces=True)
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        return strategy

    def test_valid_interface(self, dummy_model, worker_tools):
        strategy = self._init(dummy_model, worker_tools)
        strategy.next_step([], {"task": "Do it"})
        # Planner output contains "plan" which overlaps with interface description
        history = [_msg("planner", "Here is the structured plan", 1)]
        strategy.next_step(history, {})
        # Should have validated planner's output.
        assert len(strategy.interface_results) == 1
        assert strategy.interface_results[0]["valid"] is True

    def test_invalid_interface_empty_output(self, dummy_model, worker_tools):
        strategy = self._init(dummy_model, worker_tools)
        strategy.next_step([], {"task": "Do it"})
        history = [_msg("planner", "", 1)]
        strategy.next_step(history, {})
        assert len(strategy.interface_results) == 1
        assert strategy.interface_results[0]["valid"] is False


# ---- Planner output parsing ---------------------------------------------------

class TestParsePlannerOutput:
    def test_valid_json(self):
        raw = json.dumps([
            {"stage_name": "s1", "role": "Do A",
             "allowed_tools": [], "interface_output": "Out A"},
            {"stage_name": "s2", "role": "Do B",
             "allowed_tools": ["*"], "interface_output": "Out B"},
        ])
        result = _parse_planner_output(raw)
        assert len(result) == 2
        assert result[0]["stage_name"] == "s1"

    def test_json_wrapped_in_text(self):
        raw = (
            "Here's my decomposition:\n"
            '[{"stage_name": "s1", "role": "Do A", '
            '"allowed_tools": [], "interface_output": "Out"}]\n'
            "Done!"
        )
        result = _parse_planner_output(raw)
        assert len(result) == 1

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="could not be parsed"):
            _parse_planner_output("This is plain text with no JSON")

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_planner_output("[{bad json}]")

    def test_missing_stage_name_raises(self):
        raw = json.dumps([{"role": "Do A"}])
        with pytest.raises(ValueError, match="missing 'stage_name'"):
            _parse_planner_output(raw)

    def test_missing_role_raises(self):
        raw = json.dumps([{"stage_name": "s1"}])
        with pytest.raises(ValueError, match="missing 'role'"):
            _parse_planner_output(raw)


# ---- Interface validation helper ---------------------------------------------

class TestValidateInterface:
    def test_non_empty_with_overlap(self):
        assert _validate_interface("Here is my structured plan", "A structured plan") is True

    def test_empty_output(self):
        assert _validate_interface("", "A structured plan") is False

    def test_whitespace_only(self):
        assert _validate_interface("   ", "A plan") is False

    def test_no_expected_description(self):
        assert _validate_interface("Something", "") is True
