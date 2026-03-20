"""End-to-end integration tests.

Fast tests use mock agents (no GPU required).
Slow tests (marked @pytest.mark.slow) use the real LLM.
"""

import json

import pytest
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    Model,
)

from src.config.loader import load_config, load_yaml
from src.coordination.coordinator import Coordinator
from src.coordination.history import AgentMessage
from src.coordination.strategies.graph_routed import GraphRoutedStrategy
from src.coordination.strategies.sequential import SequentialStrategy
from src.logging.logger import InstrumentationLogger
from src.tools.mock_tools import CalculatorTool, EchoTool, StateTool

# ---- Model stubs for fast integration tests -----------------------------------

class _FinalAnswerModel(Model):
    """Model that returns final_answer immediately with configurable output."""

    def __init__(self, answer: str = "dummy output"):
        super().__init__(model_id="final-answer-model")
        self._answer = answer
        self._call_count = 0

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs):
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


def _worker_tools():
    return {
        "echo_tool": EchoTool(),
        "calculator_tool": CalculatorTool(),
        "state_tool": StateTool(),
    }


def _make_sequential_config(model, **overrides):
    """Build a config dict for sequential strategy integration tests."""
    seq = {
        "decomposition_mode": "human",
        "pipeline_template": "linear",
        "validate_interfaces": False,
        "stage_max_steps": 2,
    }
    seq.update(overrides)
    return {
        "sequential": seq,
        "termination": {
            "keyword": "TASK_COMPLETE",
            "max_turns": 20,
            "max_consecutive_errors": 3,
        },
        "_worker_tools": _worker_tools(),
        "_model": model,
        "stage_defaults": {
            "base_instructions": "You are one stage in a pipeline.",
        },
    }


# ---- Mock agent for graph-routed (pre-existing agent approach) ----------------

class _MockAgent:
    """Canned-response agent matching the ToolCallingAgent.run() interface."""

    def __init__(self, name: str, responses: list[str]):
        self.name = name
        self._responses = responses
        self._call_count = 0

    def run(self, task: str) -> str:
        self._call_count += 1
        idx = min(self._call_count - 1, len(self._responses) - 1)
        return f"{self._responses[idx]} [call {self._call_count}]"


def _make_agents():
    """Create a standard 3-agent set for graph-routed integration."""
    return {
        "planner": _MockAgent("planner", [
            "Plan: Step 1 calculate, Step 2 verify",
        ]),
        "executor": _MockAgent("executor", [
            "Calculated: 2+2=4",
            "Verified: result correct",
        ]),
        "reviewer": _MockAgent("reviewer", [
            "Review: looks correct. TASK_COMPLETE",
        ]),
    }


# ---- Full-stack integration with sequential strategy -------------------------

class TestIntegrationSequential:
    def test_full_run_with_logger(self, tmp_path):
        model = _FinalAnswerModel("TASK_COMPLETE result")
        config = _make_sequential_config(model)
        agents = {}
        strategy = SequentialStrategy()

        logger = InstrumentationLogger(
            {"logging": {"output_dir": str(tmp_path)}},
        )
        coord = Coordinator(agents, strategy, config, logger=logger)
        result = coord.run("Calculate 2+2 and verify")

        # First agent returns TASK_COMPLETE, termination stops
        assert len(result.history) >= 1
        assert result.history[0].agent_name == "planner"

        # Keyword terminated
        assert "TASK_COMPLETE" in result.final_output

        # Metrics populated
        assert result.metrics["total_messages"] >= 1
        assert result.metrics["error_count"] == 0

        # Export works
        path = logger.export_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["history"]) >= 1
        assert data["metrics"]["total_messages"] >= 1

    def test_full_pipeline_three_stages(self, tmp_path):
        """Run all 3 stages without keyword termination."""
        model = _FinalAnswerModel("stage output")
        config = _make_sequential_config(model)
        config["termination"]["keyword"] = "NEVER_MATCH"
        agents = {}
        strategy = SequentialStrategy()
        logger = InstrumentationLogger(
            {"logging": {"output_dir": str(tmp_path)}},
        )
        coord = Coordinator(agents, strategy, config, logger=logger)
        result = coord.run("Calculate 2+2 and verify")

        assert len(result.history) == 3
        agent_names = [m.agent_name for m in result.history]
        assert agent_names == ["planner", "executor", "reviewer"]

        assert result.metrics["total_messages"] == 3
        assert result.metrics["error_count"] == 0

        path = logger.export_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["history"]) == 3

    def test_turns_have_required_fields(self):
        model = _FinalAnswerModel("output")
        config = _make_sequential_config(model)
        config["termination"]["keyword"] = "NEVER_MATCH"
        agents = {}
        strategy = SequentialStrategy()
        logger = InstrumentationLogger()
        coord = Coordinator(agents, strategy, config, logger=logger)
        result = coord.run("Test task")

        for msg in result.history:
            assert isinstance(msg, AgentMessage)
            assert msg.agent_name in agents
            assert msg.turn_number > 0
            assert msg.timestamp > 0
            assert msg.duration_seconds >= 0
            assert msg.error is None


# ---- Full-stack integration with graph-routed strategy -----------------------

class TestIntegrationGraphRouted:
    def test_graph_routed_run(self, tmp_path):
        agents = _make_agents()
        config = load_yaml("config/coordination.yaml")
        config["strategy"] = "graph_routed"
        strategy = GraphRoutedStrategy()

        logger = InstrumentationLogger(
            {"logging": {"output_dir": str(tmp_path)}},
        )
        coord = Coordinator(agents, strategy, config, logger=logger)
        result = coord.run("Calculate prime factors of 42")

        # Should produce at least 1 message
        assert len(result.history) >= 1
        assert result.metrics["total_messages"] >= 1
        assert result.metrics["error_count"] == 0

        # All messages reference valid agents
        for msg in result.history:
            assert msg.agent_name in agents


# ---- Config loading integration ---------------------------------------------

class TestConfigIntegration:
    def test_default_config_loads_cleanly(self):
        config = load_config("config/default.yaml")
        assert config.llm.model_id == "Qwen/Qwen3-8B"
        assert config.mcp.mode in ("mock", "real")

    def test_coordination_config_loads_cleanly(self):
        coord_config = load_yaml("config/coordination.yaml")
        assert "strategy" in coord_config
        assert "termination" in coord_config
        assert "sequential" in coord_config

    def test_agents_config_loads_cleanly(self):
        agents_data = load_yaml("config/agents.yaml")
        assert "agents" in agents_data
        assert len(agents_data["agents"]) == 3


# ---- Metrics integration ----------------------------------------------------

class TestMetricsIntegration:
    def test_metrics_from_real_run(self):
        model = _FinalAnswerModel("output")
        config = _make_sequential_config(model)
        config["termination"]["keyword"] = "NEVER_MATCH"
        agents = {}
        strategy = SequentialStrategy()
        logger = InstrumentationLogger()
        coord = Coordinator(agents, strategy, config, logger=logger)
        result = coord.run("Compute something")

        metrics = result.metrics
        assert metrics["total_messages"] == 3
        assert metrics["total_duration_seconds"] >= 0
        assert metrics["error_count"] == 0
        assert metrics["error_rate"] == 0.0
        assert 0.0 <= metrics["coordination_efficiency"] <= 1.0

    def test_json_export_round_trip(self, tmp_path):
        model = _FinalAnswerModel("output")
        config = _make_sequential_config(model)
        config["termination"]["keyword"] = "NEVER_MATCH"
        agents = {}
        strategy = SequentialStrategy()
        logger = InstrumentationLogger(
            {"logging": {"output_dir": str(tmp_path)}},
        )
        coord = Coordinator(agents, strategy, config, logger=logger)
        result = coord.run("Test task")

        # Export
        from src.logging.exporter import export_run
        path = str(tmp_path / "integration_run.json")
        export_run(result.history, result.metrics, path)

        # Re-read
        with open(path) as f:
            data = json.load(f)

        assert data["metrics"]["total_messages"] == len(data["history"])
        assert all("agent_name" in m for m in data["history"])
        assert all("content" in m for m in data["history"])


# ---- Real LLM integration (slow) -------------------------------------------

@pytest.mark.slow
class TestIntegrationRealLLM:
    """These tests load the real LLM. Run with: pytest -m slow"""

    def test_from_config_sequential(self, tmp_path):
        config = load_config("config/default.yaml")
        logger = InstrumentationLogger(
            {"logging": {"output_dir": str(tmp_path)}},
        )
        coordinator = Coordinator.from_config(
            config, logger=logger, strategy_override="sequential",
        )
        result = coordinator.run("What is 2 plus 2?")

        assert len(result.history) >= 1
        assert result.final_output != ""
        assert result.metrics["total_messages"] >= 1

        # Export
        path = logger.export_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["history"]) >= 1
