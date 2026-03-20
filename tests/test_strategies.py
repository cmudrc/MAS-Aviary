"""Tests for coordination strategies.

Pure logic — mock agents (just names), no real LLM.
"""

import time

import pytest
from smolagents.models import Model

from src.coordination.history import AgentMessage
from src.coordination.strategies.graph_routed import GraphRoutedStrategy
from src.coordination.strategies.sequential import SequentialStrategy
from src.coordination.strategy import CoordinationAction
from src.tools.mock_tools import CalculatorTool, EchoTool, StateTool


def _msg(agent: str, content: str, turn: int) -> AgentMessage:
    return AgentMessage(
        agent_name=agent, content=content, turn_number=turn, timestamp=time.time(),
    )


class _DummyModel(Model):
    """Minimal model stub for strategy tests."""
    def __init__(self):
        super().__init__(model_id="dummy")

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs):
        from smolagents.models import ChatMessage
        return ChatMessage(role="assistant", content="dummy response")


# ---- CoordinationAction structure ---------------------------------------------

class TestCoordinationAction:
    def test_invoke_action(self):
        action = CoordinationAction(
            action_type="invoke_agent",
            agent_name="planner",
            input_context="Do something",
        )
        assert action.action_type == "invoke_agent"
        assert action.agent_name == "planner"
        assert action.metadata == {}

    def test_terminate_action(self):
        action = CoordinationAction(
            action_type="terminate",
            agent_name=None,
            input_context="",
        )
        assert action.action_type == "terminate"


# ---- SequentialStrategy -------------------------------------------------------

class TestSequentialStrategy:
    """Tests for the template-based sequential pipeline strategy."""

    @pytest.fixture
    def model(self):
        return _DummyModel()

    @pytest.fixture
    def worker_tools(self):
        return {
            "echo_tool": EchoTool(),
            "calculator_tool": CalculatorTool(),
            "state_tool": StateTool(),
        }

    def _make_config(self, worker_tools, **overrides):
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

    def test_initialize_creates_agents(self, model, worker_tools):
        s = SequentialStrategy()
        config = self._make_config(worker_tools)
        config["_model"] = model
        agents = {}
        s.initialize(agents, config)
        assert set(agents.keys()) == {"planner", "executor", "reviewer"}

    def test_first_step_uses_task(self, model, worker_tools):
        s = SequentialStrategy()
        config = self._make_config(worker_tools)
        config["_model"] = model
        agents = {}
        s.initialize(agents, config)
        action = s.next_step([], {"task": "Calculate 2+2"})
        assert action.action_type == "invoke_agent"
        assert action.agent_name == "planner"
        assert "Calculate 2+2" in action.input_context

    def test_advances_to_next_stage(self, model, worker_tools):
        s = SequentialStrategy()
        config = self._make_config(worker_tools)
        config["_model"] = model
        agents = {}
        s.initialize(agents, config)
        s.next_step([], {"task": "task"})
        history = [_msg("planner", "Step 1: do X", 1)]
        action = s.next_step(history, {})
        assert action.agent_name == "executor"

    def test_full_pipeline_then_terminate(self, model, worker_tools):
        s = SequentialStrategy()
        config = self._make_config(worker_tools)
        config["_model"] = model
        agents = {}
        s.initialize(agents, config)

        history = []
        for i, expected_agent in enumerate(["planner", "executor", "reviewer"]):
            action = s.next_step(history, {"task": "task"})
            assert action.action_type == "invoke_agent"
            assert action.agent_name == expected_agent
            history.append(_msg(expected_agent, f"output_{i}", i))

        action = s.next_step(history, {})
        assert action.action_type == "terminate"
        assert action.metadata["reason"] == "pipeline_complete"

    def test_is_complete_after_pipeline(self, model, worker_tools):
        s = SequentialStrategy()
        config = self._make_config(worker_tools)
        config["_model"] = model
        agents = {}
        s.initialize(agents, config)
        assert s.is_complete([], {}) is False

        history = []
        for i in range(3):
            s.next_step(history, {"task": "t"})
            history.append(_msg(f"agent_{i}", f"out_{i}", i + 1))
        assert s.is_complete(history, {}) is True

    def test_v_model_template(self, model, worker_tools):
        s = SequentialStrategy()
        config = self._make_config(worker_tools, pipeline_template="v_model")
        config["_model"] = model
        agents = {}
        s.initialize(agents, config)
        assert len(agents) == 5
        assert s.stage_order[0] == "requirements_analyst"

    def test_invalid_decomposition_mode_raises(self, model, worker_tools):
        s = SequentialStrategy()
        config = self._make_config(worker_tools, decomposition_mode="invalid")
        config["_model"] = model
        agents = {}
        with pytest.raises(ValueError, match="Unknown decomposition_mode"):
            s.initialize(agents, config)


# ---- GraphRoutedStrategy ------------------------------------------------------

class TestGraphRoutedStrategy:
    @pytest.fixture
    def agents(self):
        return {"planner": "mock", "executor": "mock", "reviewer": "mock"}

    @pytest.fixture
    def config(self):
        return {
            "graph_routed": {
                "transitions": {
                    "planner": ["executor"],
                    "executor": ["reviewer"],
                    "reviewer": ["planner", "executor"],
                },
                "routing_mode": "rule_based",
                "routing_rules": {
                    "needs revision": "planner",
                    "needs calculation": "executor",
                    "default": "reviewer",
                },
            }
        }

    def test_initialize(self, agents, config):
        s = GraphRoutedStrategy()
        s.initialize(agents, config)
        assert s.start_agent == "planner"
        assert s.routing_mode == "rule_based"

    def test_initialize_unknown_agent_raises(self):
        s = GraphRoutedStrategy()
        config = {"graph_routed": {"transitions": {"ghost": ["planner"]}}}
        with pytest.raises(ValueError, match="unknown agent"):
            s.initialize({"planner": "mock"}, config)

    def test_first_step_starts_with_start_agent(self, agents, config):
        s = GraphRoutedStrategy()
        s.initialize(agents, config)
        action = s.next_step([], {"task": "Solve X"})
        assert action.action_type == "invoke_agent"
        assert action.agent_name == "planner"
        assert action.input_context == "Solve X"

    def test_rule_based_routing_keyword(self, agents, config):
        s = GraphRoutedStrategy()
        s.initialize(agents, config)
        # After reviewer, "needs calculation" should route to executor
        history = [_msg("reviewer", "This needs calculation to verify", 1)]
        action = s.next_step(history, {})
        assert action.agent_name == "executor"

    def test_rule_based_routing_default(self, agents, config):
        s = GraphRoutedStrategy()
        s.initialize(agents, config)
        # After reviewer, no keyword match, default is "reviewer" but
        # reviewer's allowed transitions are [planner, executor],
        # and "reviewer" is not in allowed, so falls back to first allowed
        history = [_msg("reviewer", "Looks fine", 1)]
        action = s.next_step(history, {})
        # default is "reviewer" but not in allowed [planner, executor] → first allowed
        assert action.agent_name == "planner"

    def test_terminate_when_no_transitions(self, agents, config):
        """An agent with no outgoing transitions should trigger terminate."""
        config["graph_routed"]["transitions"]["executor"] = []
        s = GraphRoutedStrategy()
        s.initialize(agents, config)
        history = [_msg("executor", "done", 1)]
        action = s.next_step(history, {})
        assert action.action_type == "terminate"

    def test_planner_to_executor(self, agents, config):
        s = GraphRoutedStrategy()
        s.initialize(agents, config)
        history = [_msg("planner", "Execute subtask A", 1)]
        action = s.next_step(history, {})
        assert action.agent_name == "executor"

    def test_is_complete_no_transitions(self, agents, config):
        config["graph_routed"]["transitions"]["reviewer"] = []
        s = GraphRoutedStrategy()
        s.initialize(agents, config)
        history = [_msg("reviewer", "all done", 3)]
        assert s.is_complete(history, {}) is True

    def test_is_complete_has_transitions(self, agents, config):
        s = GraphRoutedStrategy()
        s.initialize(agents, config)
        history = [_msg("planner", "planned", 1)]
        assert s.is_complete(history, {}) is False
