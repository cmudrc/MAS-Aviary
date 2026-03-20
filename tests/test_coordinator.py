"""Tests for the Coordinator runner.

Uses a FinalAnswerModel so ToolCallingAgents created by the strategy
complete in one step with controlled output. No GPU needed.
"""

import json
import time

from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    Model,
)

from src.coordination.coordinator import Coordinator
from src.coordination.strategies.sequential import SequentialStrategy
from src.coordination.strategy import CoordinationResult
from src.tools.mock_tools import CalculatorTool, EchoTool, StateTool


class _FinalAnswerModel(Model):
    """Model that immediately returns final_answer with configurable output."""

    def __init__(self, answer: str = "dummy output"):
        super().__init__(model_id="final-answer-model")
        self._answer = answer
        self._call_count = 0

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        self._call_count += 1
        answer = f"{self._answer} [call {self._call_count}]"
        tc = ChatMessageToolCall(
            id=f"call_{self._call_count}",
            type="function",
            function=ChatMessageToolCallFunction(
                name="final_answer",
                arguments=json.dumps({"answer": answer}),
            ),
        )
        return ChatMessage(role="assistant", content="", tool_calls=[tc])


class _ErrorModel(Model):
    """Model that always raises an exception."""

    def __init__(self):
        super().__init__(model_id="error-model")

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        raise RuntimeError("model crashed")


def _worker_tools():
    return {
        "echo_tool": EchoTool(),
        "calculator_tool": CalculatorTool(),
        "state_tool": StateTool(),
    }


def _make_config(model, pipeline_template="linear", max_turns=20, **overrides):
    seq = {
        "decomposition_mode": "human",
        "pipeline_template": pipeline_template,
        "validate_interfaces": False,
        "stage_max_steps": 2,
    }
    seq.update(overrides)
    return {
        "sequential": seq,
        "termination": {
            "keyword": "TASK_COMPLETE",
            "max_turns": max_turns,
            "max_consecutive_errors": 3,
        },
        "_worker_tools": _worker_tools(),
        "_model": model,
        "stage_defaults": {
            "base_instructions": "You are one stage in a sequential pipeline.",
        },
    }


# ---- Basic coordination loop ---------------------------------------------------


class TestCoordinatorSequential:
    def test_runs_sequential_pipeline(self):
        model = _FinalAnswerModel("stage output")
        config = _make_config(model)
        agents = {}
        strategy = SequentialStrategy()
        coord = Coordinator(agents, strategy, config)
        result = coord.run("Calculate 2+2")

        assert isinstance(result, CoordinationResult)
        assert len(result.history) == 3
        assert result.history[0].agent_name == "planner"
        assert result.history[1].agent_name == "executor"
        assert result.history[2].agent_name == "reviewer"

    def test_terminates_on_keyword(self):
        model = _FinalAnswerModel("TASK_COMPLETE")
        config = _make_config(model)
        agents = {}
        strategy = SequentialStrategy()
        coord = Coordinator(agents, strategy, config)
        result = coord.run("test")

        # First agent (planner) returns TASK_COMPLETE, termination checker
        # stops the loop
        assert len(result.history) == 1
        assert result.history[0].agent_name == "planner"
        assert "TASK_COMPLETE" in result.history[0].content

    def test_terminates_on_max_turns(self):
        model = _FinalAnswerModel("still working")
        config = _make_config(model, pipeline_template="v_model", max_turns=3)
        agents = {}
        strategy = SequentialStrategy()
        coord = Coordinator(agents, strategy, config)
        result = coord.run("test")

        # v_model has 5 stages, max_turns=3 stops after 3
        assert len(result.history) == 3


# ---- Error handling -----------------------------------------------------------


class TestCoordinatorErrors:
    def test_agent_exception_logged_in_message(self):
        model = _ErrorModel()
        config = _make_config(model, max_turns=5)
        config["termination"]["max_consecutive_errors"] = 1
        agents = {}
        strategy = SequentialStrategy()
        coord = Coordinator(agents, strategy, config)
        result = coord.run("test")

        assert len(result.history) >= 1
        err_msg = result.history[0]
        assert err_msg.error is not None
        assert "crashed" in err_msg.error

    def test_consecutive_errors_terminate(self):
        model = _ErrorModel()
        config = _make_config(model, pipeline_template="v_model", max_turns=20)
        config["termination"]["max_consecutive_errors"] = 3
        agents = {}
        strategy = SequentialStrategy()
        coord = Coordinator(agents, strategy, config)
        result = coord.run("test")

        # Should stop after 3 consecutive errors
        assert len(result.history) == 3
        assert all(m.error is not None for m in result.history)


# ---- Message structure ---------------------------------------------------------


class TestCoordinatorMessages:
    def test_turn_numbers_sequential(self):
        model = _FinalAnswerModel("out")
        config = _make_config(model)
        agents = {}
        strategy = SequentialStrategy()
        coord = Coordinator(agents, strategy, config)
        result = coord.run("test")

        assert result.history[0].turn_number == 1
        assert result.history[1].turn_number == 2

    def test_timestamps_are_recent(self):
        model = _FinalAnswerModel("out")
        config = _make_config(model)
        agents = {}
        strategy = SequentialStrategy()
        coord = Coordinator(agents, strategy, config)
        before = time.time()
        result = coord.run("test")
        after = time.time()

        assert before <= result.history[0].timestamp <= after

    def test_duration_is_positive(self):
        model = _FinalAnswerModel("out")
        config = _make_config(model)
        agents = {}
        strategy = SequentialStrategy()
        coord = Coordinator(agents, strategy, config)
        result = coord.run("test")

        assert result.history[0].duration_seconds >= 0
