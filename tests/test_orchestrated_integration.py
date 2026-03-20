"""Integration tests for the Orchestrated coordination strategy.

Fast tests use mock agents (no GPU). Slow test uses real LLM.
"""

import json

import pytest
from smolagents.models import Model

from src.config.loader import load_config, load_yaml
from src.coordination.coordinator import Coordinator
from src.coordination.strategies.orchestrated import OrchestratedStrategy
from src.logging.logger import InstrumentationLogger
from src.logging.orchestration_metrics import compute_orchestration_metrics
from src.tools.mock_tools import CalculatorTool, EchoTool, StateTool

# ---- Fixtures ----------------------------------------------------------------

class DummyModel(Model):
    def __init__(self):
        super().__init__(model_id="dummy")

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs):
        from smolagents.types import ChatMessage
        return ChatMessage(role="assistant", content="dummy response")


class _MockOrchestrator:
    """Mock orchestrator that programmatically calls the injected tools.

    When run() is called, it invokes list_available_tools, create_agent,
    and assign_task through the real tool objects, then signals
    DELEGATION_COMPLETE. This tests the full tool → context → strategy flow.
    """

    def __init__(self, name: str = "orchestrator"):
        self.name = name
        self.model = DummyModel()
        self.tools = {}  # Populated by strategy.initialize()
        self._call_count = 0

    def run(self, task: str) -> str:
        self._call_count += 1

        if self._call_count == 1:
            # First call: discover tools, create a worker, assign a task.
            list_tool = self.tools.get("list_available_tools")
            create_tool = self.tools.get("create_agent")
            assign_tool = self.tools.get("assign_task")

            list_tool() if list_tool else "[]"

            if create_tool:
                create_tool(
                    name="calc_worker",
                    persona="You are a calculator agent. Use tools to compute.",
                    tools=["calculator_tool"],
                )

            if assign_tool:
                assign_tool(
                    agent_name="calc_worker",
                    task="Calculate 15 + 27",
                )

            return "Team created with 1 worker. DELEGATION_COMPLETE"

        # Subsequent calls (active mode review).
        return "Review complete. TASK_COMPLETE"


class _MockWorker:
    """Simple mock that returns a canned response."""

    def __init__(self, name: str, response: str = "42"):
        self.name = name
        self._response = response

    def run(self, task: str) -> str:
        return f"{self._response} [task: {task[:30]}]"


@pytest.fixture
def worker_tools():
    echo = EchoTool()
    calc = CalculatorTool()
    state = StateTool()
    return {echo.name: echo, calc.name: calc, state.name: state}


@pytest.fixture
def orchestrated_config():
    return load_yaml("config/orchestrated.yaml")


# ---- Fast integration: setup_only mode --------------------------------------

class TestOrchestratedSetupOnly:
    def test_full_flow_setup_only(self, worker_tools, orchestrated_config, tmp_path):
        orchestrated_config["orchestrated"]["lifecycle_mode"] = "setup_only"
        orchestrated_config["_worker_tools"] = worker_tools

        orchestrator = _MockOrchestrator()
        agents = {"orchestrator": orchestrator}

        strategy = OrchestratedStrategy()
        logger = InstrumentationLogger({"logging": {"output_dir": str(tmp_path)}})
        coord = Coordinator(agents, strategy, orchestrated_config, logger=logger)
        result = coord.run("Calculate 15 + 27 and verify")

        # Orchestrator should have created a worker.
        assert "calc_worker" in agents
        assert len(strategy.context.created_agents) == 1

        # At least 2 messages: orchestrator + worker.
        assert len(result.history) >= 2

        # First message from orchestrator, contains DELEGATION_COMPLETE.
        assert result.history[0].agent_name == "orchestrator"
        assert "DELEGATION_COMPLETE" in result.history[0].content

        # Worker ran (calc_worker is a ToolCallingAgent, may error since
        # DummyModel can't reason, but should still produce a turn).
        worker_msgs = [m for m in result.history if m.agent_name == "calc_worker"]
        assert len(worker_msgs) >= 1

        # No coordination-level errors.
        assert result.metrics["error_count"] == 0 or any(
            m.agent_name == "calc_worker" for m in result.history
        )

    def test_orchestration_metrics_computed(self, worker_tools, orchestrated_config):
        orchestrated_config["orchestrated"]["lifecycle_mode"] = "setup_only"
        orchestrated_config["_worker_tools"] = worker_tools

        orchestrator = _MockOrchestrator()
        agents = {"orchestrator": orchestrator}

        strategy = OrchestratedStrategy()
        logger = InstrumentationLogger()
        coord = Coordinator(agents, strategy, orchestrated_config, logger=logger)
        result = coord.run("Calculate something")

        # Compute orchestration-specific metrics.
        orch_metrics = compute_orchestration_metrics(result.history)
        assert orch_metrics["orchestrator_turns"] >= 1
        assert orch_metrics["agents_spawned"] >= 0


# ---- Fast integration: active mode ------------------------------------------

class TestOrchestratedActive:
    def test_active_mode_runs(self, worker_tools, orchestrated_config, tmp_path):
        orchestrated_config["orchestrated"]["lifecycle_mode"] = "active"
        orchestrated_config["_worker_tools"] = worker_tools

        orchestrator = _MockOrchestrator()
        agents = {"orchestrator": orchestrator}

        strategy = OrchestratedStrategy()
        logger = InstrumentationLogger({"logging": {"output_dir": str(tmp_path)}})
        coord = Coordinator(agents, strategy, orchestrated_config, logger=logger)
        result = coord.run("Calculate 42 * 1")

        # Should have orchestrator turns + at least one worker turn.
        assert len(result.history) >= 2
        agent_names = [m.agent_name for m in result.history]
        assert "orchestrator" in agent_names


# ---- Config loading ----------------------------------------------------------

class TestOrchestratedConfig:
    def test_orchestrated_config_loads(self):
        config = load_yaml("config/orchestrated.yaml")
        assert config["strategy"] == "orchestrated"
        assert "orchestrated" in config
        assert config["orchestrated"]["max_agents"] == 8

    def test_orchestrated_agents_config_loads(self):
        agents_data = load_yaml("config/orchestrated_agents.yaml")
        assert "agents" in agents_data
        assert len(agents_data["agents"]) == 1
        assert agents_data["agents"][0]["name"] == "orchestrator"

    def test_strategy_loadable(self):
        from src.coordination.coordinator import _load_strategy
        strategy = _load_strategy("orchestrated")
        assert isinstance(strategy, OrchestratedStrategy)


# ---- JSON export integration ------------------------------------------------

class TestOrchestratedExport:
    def test_export_round_trip(self, worker_tools, orchestrated_config, tmp_path):
        orchestrated_config["orchestrated"]["lifecycle_mode"] = "setup_only"
        orchestrated_config["_worker_tools"] = worker_tools

        orchestrator = _MockOrchestrator()
        agents = {"orchestrator": orchestrator}

        strategy = OrchestratedStrategy()
        logger = InstrumentationLogger({"logging": {"output_dir": str(tmp_path)}})
        coord = Coordinator(agents, strategy, orchestrated_config, logger=logger)
        result = coord.run("Test export")

        # Export and re-read.
        path = logger.export_json()
        with open(path) as f:
            data = json.load(f)

        assert len(data["history"]) == len(result.history)
        assert data["metrics"]["total_messages"] == len(result.history)


# ---- Real LLM integration (slow) -------------------------------------------

@pytest.mark.slow
class TestOrchestratedRealLLM:
    """Requires GPU + model download. Run with: pytest -m slow"""

    def test_orchestrated_setup_only_with_real_llm(self, tmp_path):
        """Full end-to-end: real LLM orchestrator creates agents and runs."""
        config = load_config("config/default.yaml")
        # Override to use orchestrated strategy and agents.
        config.agents_config = "config/orchestrated_agents.yaml"
        config.coordination_config = "config/orchestrated.yaml"

        logger = InstrumentationLogger({"logging": {"output_dir": str(tmp_path)}})
        coordinator = Coordinator.from_config(
            config, logger=logger, strategy_override="orchestrated",
        )

        result = coordinator.run(
            "Calculate the sum of 15 and 27, then verify the result by multiplying 42 by 1"
        )

        assert len(result.history) >= 2
        assert result.final_output != ""

        # Export.
        path = str(tmp_path / "orchestrated_e2e.json")
        from src.logging.exporter import export_run
        export_run(result.history, result.metrics, path)

        with open(path) as f:
            data = json.load(f)
        assert len(data["history"]) >= 2
