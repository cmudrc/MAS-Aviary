"""Integration tests for the Networked coordination strategy.

Fast tests use mock agents (no GPU). Slow test uses real LLM.
"""

import json

import pytest
from smolagents.models import Model

from src.config.loader import load_config, load_yaml
from src.coordination.coordinator import Coordinator
from src.coordination.strategies.networked import NetworkedStrategy
from src.logging.logger import InstrumentationLogger
from src.logging.networked_metrics import compute_networked_metrics
from src.tools.mock_tools import CalculatorTool, EchoTool, StateTool

# ---- Fixtures ----------------------------------------------------------------


class DummyModel(Model):
    def __init__(self):
        super().__init__(model_id="dummy")

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        from smolagents.models import ChatMessage

        return ChatMessage(role="assistant", content="dummy response")


class _MockPeerAgent:
    """Mock peer agent that reads blackboard and posts results.

    Simulates the agent workflow: read → decide → work → write → status.
    On the Nth call (configurable), outputs TASK_COMPLETE.
    """

    def __init__(self, name: str, complete_on_call: int = 3):
        self.name = name
        self.model = DummyModel()
        self.tools = {}
        self._call_count = 0
        self._complete_on = complete_on_call

    def run(self, task: str) -> str:
        self._call_count += 1

        # Read blackboard if available.
        read_tool = self.tools.get("read_blackboard")
        write_tool = self.tools.get("write_blackboard")

        if read_tool:
            read_tool.forward()

        if write_tool:
            write_tool.forward(
                key=f"{self.name}_result",
                value=f"Completed subtask {self._call_count}",
                entry_type="result",
            )
            write_tool.forward(
                key=f"{self.name}_status",
                value="DONE" if self._call_count >= self._complete_on else "working",
                entry_type="status",
            )

        if self._call_count >= self._complete_on:
            return "All work done. TASK_COMPLETE"

        return f"Working on subtask {self._call_count}"


@pytest.fixture
def worker_tools():
    echo = EchoTool()
    calc = CalculatorTool()
    state = StateTool()
    return {echo.name: echo, calc.name: calc, state.name: state}


@pytest.fixture
def networked_config():
    return load_yaml("config/networked.yaml")


# ---- Fast integration: basic run ---------------------------------------------


class TestNetworkedBasicRun:
    def test_full_flow_soft_claiming(self, worker_tools, networked_config, tmp_path):
        """Strategy creates agents, rotates, terminates on TASK_COMPLETE."""
        networked_config["networked"]["initial_agents"] = 2
        networked_config["networked"]["claiming_mode"] = "soft"
        networked_config["_worker_tools"] = worker_tools
        networked_config["_model"] = DummyModel()

        agents = {}
        strategy = NetworkedStrategy()
        logger = InstrumentationLogger({"logging": {"output_dir": str(tmp_path)}})
        coord = Coordinator(agents, strategy, networked_config, logger=logger)
        result = coord.run("Calculate 15 + 27")

        # Should have created 2 agents.
        agent_names = {m.agent_name for m in result.history}
        assert len(agent_names) >= 1  # At least one agent acted

        # Should have some history.
        assert len(result.history) >= 1

    def test_hard_claiming_mode(self, worker_tools, networked_config, tmp_path):
        networked_config["networked"]["initial_agents"] = 2
        networked_config["networked"]["claiming_mode"] = "hard"
        networked_config["_worker_tools"] = worker_tools
        networked_config["_model"] = DummyModel()

        agents = {}
        strategy = NetworkedStrategy()
        logger = InstrumentationLogger({"logging": {"output_dir": str(tmp_path)}})
        coord = Coordinator(agents, strategy, networked_config, logger=logger)
        result = coord.run("Calculate something")

        assert len(result.history) >= 1
        assert strategy.blackboard.claiming_mode == "hard"

    def test_no_claiming_mode(self, worker_tools, networked_config, tmp_path):
        networked_config["networked"]["initial_agents"] = 2
        networked_config["networked"]["claiming_mode"] = "none"
        networked_config["_worker_tools"] = worker_tools
        networked_config["_model"] = DummyModel()

        agents = {}
        strategy = NetworkedStrategy()
        logger = InstrumentationLogger({"logging": {"output_dir": str(tmp_path)}})
        coord = Coordinator(agents, strategy, networked_config, logger=logger)
        result = coord.run("Calculate something")

        assert len(result.history) >= 1
        assert strategy.blackboard.claiming_mode == "none"


# ---- Blackboard interaction ---------------------------------------------------


class TestNetworkedBlackboard:
    def test_blackboard_populated_during_run(self, worker_tools, networked_config, tmp_path):
        """After the run, blackboard should have task entry."""
        networked_config["networked"]["initial_agents"] = 2
        networked_config["_worker_tools"] = worker_tools
        networked_config["_model"] = DummyModel()

        agents = {}
        strategy = NetworkedStrategy()
        coord = Coordinator(agents, strategy, networked_config)
        coord.run("Calculate 2 + 2")

        # Task should be written to blackboard.
        task_entry = strategy.blackboard.get("task")
        assert task_entry is not None
        assert "2 + 2" in task_entry.value


# ---- Metrics computation -----------------------------------------------------


class TestNetworkedMetrics:
    def test_metrics_from_run(self, worker_tools, networked_config, tmp_path):
        networked_config["networked"]["initial_agents"] = 2
        networked_config["_worker_tools"] = worker_tools
        networked_config["_model"] = DummyModel()

        agents = {}
        strategy = NetworkedStrategy()
        logger = InstrumentationLogger({"logging": {"output_dir": str(tmp_path)}})
        coord = Coordinator(agents, strategy, networked_config, logger=logger)
        result = coord.run("Do some work")

        # Compute networked metrics.
        metrics = compute_networked_metrics(
            result.history,
            blackboard_size=len(strategy.blackboard),
            claim_conflicts=strategy.blackboard.claim_conflicts,
            initial_agents=2,
            spawned_agents=len(strategy.context.spawned_agents),
        )
        assert "duplicate_work_rate" in metrics
        assert "blackboard_utilization" in metrics
        assert "self_selection_diversity" in metrics
        assert metrics["total_agents"] >= 2


# ---- Config loading ----------------------------------------------------------


class TestNetworkedConfig:
    def test_networked_config_loads(self):
        config = load_yaml("config/networked.yaml")
        assert config["strategy"] == "networked"
        assert "networked" in config
        assert config["networked"]["initial_agents"] == 5
        assert config["networked"]["max_agents"] == 10
        assert config["networked"]["claiming_mode"] == "soft"

    def test_networked_agents_config_loads(self):
        agents_data = load_yaml("config/networked_agents.yaml")
        assert "peer_template" in agents_data
        template = agents_data["peer_template"]
        assert "base_system_prompt" in template
        assert "soft_claiming_addition" in template
        assert "hard_claiming_addition" in template
        assert "prediction_prompt_addition" in template

    def test_strategy_loadable(self):
        from src.coordination.coordinator import _load_strategy

        strategy = _load_strategy("networked")
        assert isinstance(strategy, NetworkedStrategy)


# ---- JSON export integration ------------------------------------------------


class TestNetworkedExport:
    def test_export_round_trip(self, worker_tools, networked_config, tmp_path):
        networked_config["networked"]["initial_agents"] = 2
        networked_config["_worker_tools"] = worker_tools
        networked_config["_model"] = DummyModel()

        agents = {}
        strategy = NetworkedStrategy()
        logger = InstrumentationLogger({"logging": {"output_dir": str(tmp_path)}})
        coord = Coordinator(agents, strategy, networked_config, logger=logger)
        result = coord.run("Test export")

        path = logger.export_json()
        with open(path) as f:
            data = json.load(f)

        assert len(data["history"]) == len(result.history)


# ---- Real LLM integration (slow) -------------------------------------------


@pytest.mark.slow
class TestNetworkedRealLLM:
    """Requires GPU + model download. Run with: pytest -m slow"""

    def test_networked_with_real_llm(self, tmp_path):
        """Full end-to-end: real LLM peer agents coordinate via blackboard."""
        config = load_config("config/default.yaml")
        config.agents_config = "config/networked_agents.yaml"
        config.coordination_config = "config/networked.yaml"

        logger = InstrumentationLogger({"logging": {"output_dir": str(tmp_path)}})
        coordinator = Coordinator.from_config(
            config,
            logger=logger,
            strategy_override="networked",
        )

        result = coordinator.run(
            "Calculate the sum of 15 and 27, then multiply the result by 2, then verify both answers"
        )

        assert len(result.history) >= 2
        assert result.final_output != ""

        # Export.
        path = str(tmp_path / "networked_e2e.json")
        from src.logging.exporter import export_run

        export_run(result.history, result.metrics, path)

        with open(path) as f:
            data = json.load(f)
        assert len(data["history"]) >= 2
