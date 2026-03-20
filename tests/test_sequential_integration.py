"""Integration tests for the Sequential coordination strategy.

Fast tests use FinalAnswerModel (no GPU). Slow test uses real LLM.
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
from src.coordination.strategies.sequential import SequentialStrategy
from src.logging.logger import InstrumentationLogger
from src.logging.sequential_metrics import (
    compute_sequential_metrics,
    compute_template_comparison,
)
from src.tools.mock_tools import CalculatorTool, EchoTool, StateTool

# ---- Fixtures ----------------------------------------------------------------

class _FinalAnswerModel(Model):
    """Model that immediately returns final_answer with configurable output."""

    def __init__(self, answer: str = "stage output"):
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


def _make_config(model, pipeline_template="linear", **seq_overrides):
    seq = {
        "decomposition_mode": "human",
        "pipeline_template": pipeline_template,
        "validate_interfaces": False,
        "stage_max_steps": 2,
    }
    seq.update(seq_overrides)
    return {
        "sequential": seq,
        "termination": {
            "keyword": "NEVER_MATCH",
            "max_turns": 30,
            "max_consecutive_errors": 3,
        },
        "_worker_tools": _worker_tools(),
        "_model": model,
        "stage_defaults": {
            "base_instructions": "You are one stage in a sequential pipeline.",
        },
    }


# ---- Fast integration tests (no GPU) ----------------------------------------

class TestSequentialLinear:
    """Full pipeline with linear template."""

    def test_three_stages_in_order(self, tmp_path):
        model = _FinalAnswerModel("output")
        config = _make_config(model)
        agents = {}
        strategy = SequentialStrategy()
        logger = InstrumentationLogger(
            {"logging": {"output_dir": str(tmp_path)}},
        )
        coord = Coordinator(agents, strategy, config, logger=logger)
        result = coord.run("Calculate 15 times 7")

        assert len(result.history) == 3
        names = [m.agent_name for m in result.history]
        assert names == ["planner", "executor", "reviewer"]

    def test_agents_are_in_dict(self, tmp_path):
        model = _FinalAnswerModel("output")
        config = _make_config(model)
        agents = {}
        strategy = SequentialStrategy()
        coord = Coordinator(agents, strategy, config)
        coord.run("Test")

        assert "planner" in agents
        assert "executor" in agents
        assert "reviewer" in agents

    def test_no_errors(self, tmp_path):
        model = _FinalAnswerModel("output")
        config = _make_config(model)
        agents = {}
        strategy = SequentialStrategy()
        logger = InstrumentationLogger(
            {"logging": {"output_dir": str(tmp_path)}},
        )
        coord = Coordinator(agents, strategy, config, logger=logger)
        result = coord.run("Calculate 15 times 7")

        assert all(m.error is None for m in result.history)
        assert result.metrics["error_count"] == 0

    def test_export_json(self, tmp_path):
        model = _FinalAnswerModel("output")
        config = _make_config(model)
        agents = {}
        strategy = SequentialStrategy()
        logger = InstrumentationLogger(
            {"logging": {"output_dir": str(tmp_path)}},
        )
        coord = Coordinator(agents, strategy, config, logger=logger)
        coord.run("Test task")

        path = logger.export_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["history"]) == 3
        assert data["metrics"]["total_messages"] == 3


class TestSequentialVModel:
    """Full pipeline with v_model template (5 stages)."""

    def test_five_stages_in_order(self, tmp_path):
        model = _FinalAnswerModel("output")
        config = _make_config(model, pipeline_template="v_model")
        agents = {}
        strategy = SequentialStrategy()
        logger = InstrumentationLogger(
            {"logging": {"output_dir": str(tmp_path)}},
        )
        coord = Coordinator(agents, strategy, config, logger=logger)
        result = coord.run("Calculate 15 times 7")

        assert len(result.history) == 5
        names = [m.agent_name for m in result.history]
        assert names == [
            "requirements_analyst", "system_designer",
            "detailed_designer", "implementer", "integration_verifier",
        ]

    def test_all_durations_positive(self, tmp_path):
        model = _FinalAnswerModel("output")
        config = _make_config(model, pipeline_template="v_model")
        agents = {}
        strategy = SequentialStrategy()
        coord = Coordinator(agents, strategy, config)
        result = coord.run("Test")

        for msg in result.history:
            assert msg.duration_seconds >= 0


class TestSequentialMBSE:
    """Full pipeline with mbse template (5 stages)."""

    def test_five_stages_in_order(self, tmp_path):
        model = _FinalAnswerModel("output")
        config = _make_config(model, pipeline_template="mbse")
        agents = {}
        strategy = SequentialStrategy()
        logger = InstrumentationLogger(
            {"logging": {"output_dir": str(tmp_path)}},
        )
        coord = Coordinator(agents, strategy, config, logger=logger)
        result = coord.run("Design a system")

        assert len(result.history) == 5
        names = [m.agent_name for m in result.history]
        assert names == [
            "stakeholder_analyst", "system_architect",
            "subsystem_designer", "implementer", "validator",
        ]


class TestSequentialCustom:
    """Full pipeline with custom template."""

    def test_custom_two_stages(self, tmp_path):
        model = _FinalAnswerModel("output")
        config = _make_config(model, pipeline_template="custom")
        config["sequential"]["custom_stages"] = [
            {"name": "analyzer", "role": "Analyze", "allowed_tools": [],
             "interface_output": "Analysis"},
            {"name": "builder", "role": "Build", "allowed_tools": ["*"],
             "interface_output": "Result"},
        ]
        agents = {}
        strategy = SequentialStrategy()
        coord = Coordinator(agents, strategy, config)
        result = coord.run("Custom task")

        assert len(result.history) == 2
        assert result.history[0].agent_name == "analyzer"
        assert result.history[1].agent_name == "builder"


# ---- Metrics integration ----------------------------------------------------

class TestSequentialMetricsIntegration:
    """Verify metrics computation works with real run data."""

    def test_compute_metrics_from_run(self, tmp_path):
        model = _FinalAnswerModel("output")
        config = _make_config(model)
        agents = {}
        strategy = SequentialStrategy()
        coord = Coordinator(agents, strategy, config)
        result = coord.run("Calculate something")

        metrics = compute_sequential_metrics(
            messages=result.history,
            stage_order=strategy.stage_order,
            pipeline_template="linear",
        )
        assert metrics["stage_count"] == 3
        assert metrics["pipeline_template"] == "linear"
        assert metrics["total_turns"] == 3
        assert metrics["propagation_time"] >= 0

    def test_template_comparison_two_templates(self, tmp_path):
        runs = {}
        for template in ("linear", "v_model"):
            model = _FinalAnswerModel("output")
            config = _make_config(model, pipeline_template=template)
            agents = {}
            strategy = SequentialStrategy()
            coord = Coordinator(agents, strategy, config)
            result = coord.run("Calculate something")

            m = compute_sequential_metrics(
                result.history, strategy.stage_order, template,
            )
            runs.setdefault(template, []).append({
                "metrics": m,
                "success": True,
            })

        comparison = compute_template_comparison(runs)
        assert "linear" in comparison["per_template"]
        assert "v_model" in comparison["per_template"]
        assert comparison["per_template"]["linear"]["runs"] == 1
        assert comparison["per_template"]["v_model"]["runs"] == 1


# ---- Config file loading -----------------------------------------------------

class TestSequentialConfigLoading:
    """Verify the sequential config files load correctly."""

    def test_sequential_yaml_loads(self):
        config = load_yaml("config/sequential.yaml")
        assert config["strategy"] == "sequential"
        assert config["sequential"]["decomposition_mode"] == "human"
        assert config["sequential"]["pipeline_template"] == "aviary"

    def test_sequential_agents_yaml_loads(self):
        config = load_yaml("config/sequential_agents.yaml")
        assert "stage_defaults" in config
        assert "templates" in config
        assert "linear" in config["templates"]
        assert "v_model" in config["templates"]
        assert "mbse" in config["templates"]

    def test_agents_yaml_templates_have_stages(self):
        config = load_yaml("config/sequential_agents.yaml")
        for name in ("linear", "v_model", "mbse"):
            template = config["templates"][name]
            assert "stages" in template
            assert len(template["stages"]) >= 3


# ---- Real LLM integration (slow) -------------------------------------------

@pytest.mark.slow
class TestSequentialRealLLM:
    """Run sequential strategy with real LLM. Requires GPU."""

    def test_linear_with_real_llm(self, tmp_path):
        config = load_config("config/default.yaml")
        logger = InstrumentationLogger(
            {"logging": {"output_dir": str(tmp_path)}},
        )
        coordinator = Coordinator.from_config(
            config, logger=logger, strategy_override="sequential",
        )
        result = coordinator.run(
            "Calculate the sum of 15 and 27, then verify the result"
        )

        assert len(result.history) >= 1
        assert result.final_output != ""

        # Verify metrics
        assert result.metrics["total_messages"] >= 1

        # Export
        path = logger.export_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["history"]) >= 1
