"""Tests for the NetworkedStrategy coordination class.

No GPU needed. Uses DummyModel stub and mock tools.
Tests cover initialization, next_step, is_complete, toggle combinations,
agent spawning, and prediction verification.
"""

import time

import pytest
from smolagents import ToolCallingAgent
from smolagents.models import Model

from src.coordination.blackboard import Blackboard
from src.coordination.history import AgentMessage
from src.coordination.strategies.networked import NetworkedStrategy
from src.tools.mock_tools import CalculatorTool, EchoTool, StateTool
from src.tools.networked_tools import PEER_TOOL_NAMES

# ---- Fixtures ----------------------------------------------------------------


class DummyModel(Model):
    """Minimal model stub."""

    def __init__(self):
        super().__init__(model_id="dummy")

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        from smolagents.types import ChatMessage

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
    """Build a minimal valid config dict for NetworkedStrategy."""
    net = {
        "initial_agents": 3,
        "max_agents": 6,
        "agent_max_steps": 4,
        "claiming_mode": "soft",
        "peer_monitoring_visible": True,
        "trans_specialist_knowledge": True,
        "predictive_knowledge": False,
    }
    net.update(overrides)
    return {
        "networked": net,
        "termination": {
            "keyword": "TASK_COMPLETE",
            "max_turns": 20,
            "max_consecutive_errors": 3,
        },
        "context": {
            "max_recent_messages": 10,
            "max_context_tokens": 4000,
        },
        "_worker_tools": worker_tools,
        "peer_template": {
            "base_system_prompt": "You are a peer agent.",
            "soft_claiming_addition": "Claim subtasks before working.",
            "hard_claiming_addition": "You MUST claim and respect locks.",
            "prediction_prompt_addition": "Predict what another agent will do.",
        },
    }


def _make_agent_stub(model, name="stub"):
    """Create a minimal ToolCallingAgent that satisfies the interface."""
    return ToolCallingAgent(
        tools=[EchoTool()],
        model=model,
        name=name,
        add_base_tools=False,
        max_steps=2,
    )


def _make_message(agent_name, content, turn, error=None):
    return AgentMessage(
        agent_name=agent_name,
        content=content,
        turn_number=turn,
        timestamp=time.time(),
        error=error,
    )


# ---- Initialization tests ----------------------------------------------------


class TestInitialize:
    def test_creates_initial_agents(self, dummy_model, worker_tools):
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools, initial_agents=3)
        # Start with an empty agents dict (strategy creates peers).
        agents = {}
        # Need a model source — inject via config.
        config["_model"] = dummy_model
        strategy.initialize(agents, config)
        assert len(agents) == 3
        assert set(agents.keys()) == {"agent_1", "agent_2", "agent_3"}

    def test_agents_are_tool_calling_agents(self, dummy_model, worker_tools):
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools)
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        for agent in agents.values():
            assert isinstance(agent, ToolCallingAgent)

    def test_agents_have_domain_tools(self, dummy_model, worker_tools):
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools)
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        agent = agents["agent_1"]
        assert "echo_tool" in agent.tools
        assert "calculator_tool" in agent.tools

    def test_agents_have_peer_tools(self, dummy_model, worker_tools):
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools)
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        agent = agents["agent_1"]
        for peer_tool in PEER_TOOL_NAMES:
            assert peer_tool in agent.tools

    def test_blackboard_created(self, dummy_model, worker_tools):
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools)
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        assert strategy.blackboard is not None
        assert isinstance(strategy.blackboard, Blackboard)

    def test_blackboard_claiming_mode(self, dummy_model, worker_tools):
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools, claiming_mode="hard")
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        assert strategy.blackboard.claiming_mode == "hard"

    def test_agent_order(self, dummy_model, worker_tools):
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools, initial_agents=3)
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        assert strategy.agent_order == ["agent_1", "agent_2", "agent_3"]

    def test_model_from_existing_agent(self, dummy_model, worker_tools):
        """Model can be extracted from pre-existing agents."""
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools, initial_agents=2)
        # No _model in config, but provide an agent with a model.
        stub = _make_agent_stub(dummy_model, "seed")
        agents = {"seed": stub}
        strategy.initialize(agents, config)
        # Should have 2 new agents + the seed.
        assert len(agents) == 3  # seed + agent_1 + agent_2


# ---- Prompt assembly tests ---------------------------------------------------


class TestPromptAssembly:
    def test_soft_claiming_prompt(self, dummy_model, worker_tools):
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools, claiming_mode="soft")
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        assert "Claim subtasks before working" in strategy.peer_prompt

    def test_hard_claiming_prompt(self, dummy_model, worker_tools):
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools, claiming_mode="hard")
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        assert "MUST claim and respect locks" in strategy.peer_prompt

    def test_no_claiming_no_addition(self, dummy_model, worker_tools):
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools, claiming_mode="none")
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        assert "Claim subtasks" not in strategy.peer_prompt
        assert "MUST claim" not in strategy.peer_prompt

    def test_prediction_prompt_added(self, dummy_model, worker_tools):
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools, predictive_knowledge=True)
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        assert "Predict what another agent" in strategy.peer_prompt

    def test_prediction_prompt_not_added(self, dummy_model, worker_tools):
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools, predictive_knowledge=False)
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        assert "Predict what another agent" not in strategy.peer_prompt


# ---- next_step tests ---------------------------------------------------------


class TestNextStep:
    def _init_strategy(self, dummy_model, worker_tools, **overrides):
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools, initial_agents=2, **overrides)
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        return strategy, agents

    def test_returns_invoke_agent(self, dummy_model, worker_tools):
        strategy, agents = self._init_strategy(dummy_model, worker_tools)
        action = strategy.next_step([], {"task": "Do something"})
        assert action.action_type == "invoke_agent"
        assert action.agent_name in agents

    def test_rotation_cycles_agents(self, dummy_model, worker_tools):
        strategy, agents = self._init_strategy(dummy_model, worker_tools)
        state = {"task": "Do something"}

        a1 = strategy.next_step([], state)
        m1 = _make_message(a1.agent_name, "done", 1)
        a2 = strategy.next_step([m1], state)
        m2 = _make_message(a2.agent_name, "done", 2)
        a3 = strategy.next_step([m1, m2], state)

        # With 2 agents, 3rd call should cycle back.
        names = [a1.agent_name, a2.agent_name, a3.agent_name]
        assert names[0] != names[1]  # different agents
        assert names[2] == names[0]  # cycled back

    def test_context_includes_task(self, dummy_model, worker_tools):
        strategy, agents = self._init_strategy(dummy_model, worker_tools)
        action = strategy.next_step([], {"task": "Calculate 2+2"})
        assert "Calculate 2+2" in action.input_context

    def test_context_includes_blackboard(self, dummy_model, worker_tools):
        strategy, agents = self._init_strategy(dummy_model, worker_tools)
        # Write something to blackboard first.
        strategy.blackboard.write("test_key", "test_value", "agent_1", "status")
        action = strategy.next_step([], {"task": "Do it"})
        assert "Blackboard State" in action.input_context
        assert "test_key" in action.input_context

    def test_context_includes_history(self, dummy_model, worker_tools):
        strategy, agents = self._init_strategy(dummy_model, worker_tools)
        history = [_make_message("agent_1", "I did work", 1)]
        action = strategy.next_step(history, {"task": "Do it"})
        assert "I did work" in action.input_context

    def test_metadata_has_turn_info(self, dummy_model, worker_tools):
        strategy, agents = self._init_strategy(dummy_model, worker_tools)
        action = strategy.next_step([], {"task": "Do it"})
        assert "turn" in action.metadata
        assert "total_agents" in action.metadata

    def test_writes_task_to_blackboard(self, dummy_model, worker_tools):
        strategy, agents = self._init_strategy(dummy_model, worker_tools)
        strategy.next_step([], {"task": "Build a box"})
        task_entry = strategy.blackboard.get("task")
        assert task_entry is not None
        assert "Build a box" in task_entry.value

    def test_no_agents_returns_error(self, dummy_model, worker_tools):
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools, initial_agents=0)
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        action = strategy.next_step([], {"task": "Do it"})
        assert action.action_type == "error"

    def test_auto_writes_result_to_blackboard(self, dummy_model, worker_tools):
        # After a successful turn, next_step() should auto-post the result.
        strategy, agents = self._init_strategy(dummy_model, worker_tools)
        history = [_make_message("agent_1", "STL produced successfully", 1)]
        strategy.next_step(history, {"task": "Do it"})
        entry = strategy.blackboard.get("agent_1_result")
        assert entry is not None
        assert "STL produced successfully" in entry.value
        assert entry.entry_type == "result"
        assert entry.author == "agent_1"

    def test_auto_write_skips_error_turns(self, dummy_model, worker_tools):
        # Error turns must not be written to the blackboard.
        strategy, agents = self._init_strategy(dummy_model, worker_tools)
        history = [_make_message("agent_1", "", 1, error="timeout")]
        strategy.next_step(history, {"task": "Do it"})
        assert strategy.blackboard.get("agent_1_result") is None

    def test_auto_write_truncates_long_content(self, dummy_model, worker_tools):
        # Content over 800 chars should be truncated with ellipsis.
        strategy, agents = self._init_strategy(dummy_model, worker_tools)
        long_content = "x" * 1000
        history = [_make_message("agent_1", long_content, 1)]
        strategy.next_step(history, {"task": "Do it"})
        entry = strategy.blackboard.get("agent_1_result")
        assert entry is not None
        assert len(entry.value) <= 803  # 800 chars + "..."
        assert entry.value.endswith("...")

    def test_auto_write_visible_in_next_agent_context(self, dummy_model, worker_tools):
        # The auto-written result must appear in the next agent's context.
        strategy, agents = self._init_strategy(dummy_model, worker_tools)
        history = [_make_message("agent_1", "Evaluation: PCD=0.035", 1)]
        action = strategy.next_step(history, {"task": "Do it"})
        # agent_2 is next in rotation; its context should include agent_1's result.
        assert "PCD=0.035" in action.input_context


# ---- is_complete tests -------------------------------------------------------


class TestIsComplete:
    def _init_strategy(self, dummy_model, worker_tools, **overrides):
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools, initial_agents=2, **overrides)
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)
        return strategy

    def test_complete_on_task_complete_keyword(self, dummy_model, worker_tools):
        strategy = self._init_strategy(dummy_model, worker_tools)
        history = [_make_message("agent_1", "All done. TASK_COMPLETE", 1)]
        assert strategy.is_complete(history, {}) is True

    def test_not_complete_without_keyword(self, dummy_model, worker_tools):
        strategy = self._init_strategy(dummy_model, worker_tools)
        history = [_make_message("agent_1", "Still working", 1)]
        assert strategy.is_complete(history, {}) is False

    def test_complete_on_max_turns(self, dummy_model, worker_tools):
        strategy = self._init_strategy(dummy_model, worker_tools)
        history = [_make_message("a", "msg", i) for i in range(20)]
        assert strategy.is_complete(history, {}) is True

    def test_complete_on_all_agents_done(self, dummy_model, worker_tools):
        strategy = self._init_strategy(dummy_model, worker_tools)
        # Both agents report DONE on blackboard.
        strategy.blackboard.write("s1", "DONE", "agent_1", "status")
        strategy.blackboard.write("s2", "DONE", "agent_2", "status")
        history = [_make_message("agent_1", "done", 1)]
        assert strategy.is_complete(history, {}) is True

    def test_not_complete_if_only_some_done(self, dummy_model, worker_tools):
        strategy = self._init_strategy(dummy_model, worker_tools)
        strategy.blackboard.write("s1", "DONE", "agent_1", "status")
        # agent_2 hasn't reported DONE.
        history = [_make_message("agent_1", "done", 1)]
        assert strategy.is_complete(history, {}) is False

    def test_complete_on_consecutive_errors(self, dummy_model, worker_tools):
        strategy = self._init_strategy(dummy_model, worker_tools)
        state = {"task": "Do it"}
        # Simulate 3 consecutive errors via next_step tracking.
        history = [
            _make_message("agent_1", "", 1, error="fail"),
            _make_message("agent_2", "", 2, error="fail"),
            _make_message("agent_1", "", 3, error="fail"),
        ]
        # next_step tracks consecutive errors internally.
        for i in range(3):
            strategy.next_step(history[: i + 1], state)
        assert strategy.is_complete(history, state) is True

    def test_empty_history_not_complete(self, dummy_model, worker_tools):
        strategy = self._init_strategy(dummy_model, worker_tools)
        assert strategy.is_complete([], {}) is False

    def test_complete_on_mark_task_done(self, dummy_model, worker_tools):
        strategy = self._init_strategy(dummy_model, worker_tools)
        # Simulate an agent calling mark_task_done — writes task_complete entry.
        strategy.blackboard.write("task_complete", "DONE: finished", "agent_1", "status")
        history = [_make_message("agent_1", "called mark_task_done", 1)]
        assert strategy.is_complete(history, {}) is True

    def test_not_complete_without_task_complete_entry(self, dummy_model, worker_tools):
        strategy = self._init_strategy(dummy_model, worker_tools)
        # Other DONE writes to different keys don't trigger this path.
        strategy.blackboard.write("some_result", "DONE: partial", "agent_1", "status")
        history = [_make_message("agent_1", "posted result", 1)]
        assert strategy.is_complete(history, {}) is False


# ---- Spawned agents in rotation ----------------------------------------------


class TestSpawnedAgentsInRotation:
    def test_spawned_agent_added_to_rotation(self, dummy_model, worker_tools):
        strategy = NetworkedStrategy()
        config = _make_config(worker_tools, initial_agents=2, max_agents=5)
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)

        # Simulate spawn by directly using SpawnPeer.
        from src.tools.networked_tools import SpawnPeer

        spawn = SpawnPeer(strategy.context, agent_name="agent_1")
        spawn.forward("Need help with fillets")

        # The new agent should appear in rotation.
        state = {"task": "Do it"}
        seen = set()
        for i in range(6):  # enough turns to see all 3 agents
            action = strategy.next_step(
                [_make_message("a", "x", j) for j in range(i)],
                state,
            )
            seen.add(action.agent_name)
        assert "agent_3" in seen


# ---- Toggle combination tests (parameterized) -------------------------------

_CLAIMING_MODES = ["none", "soft", "hard"]
_BOOL_TOGGLES = [True, False]


class TestToggleCombinations:
    @pytest.mark.parametrize("claiming", _CLAIMING_MODES)
    @pytest.mark.parametrize("peer_mon", _BOOL_TOGGLES)
    @pytest.mark.parametrize("trans_spec", _BOOL_TOGGLES)
    @pytest.mark.parametrize("predictive", _BOOL_TOGGLES)
    def test_strategy_initializes_with_all_toggle_combos(
        self,
        dummy_model,
        worker_tools,
        claiming,
        peer_mon,
        trans_spec,
        predictive,
    ):
        """Strategy should initialize cleanly for all 48 toggle combos."""
        strategy = NetworkedStrategy()
        config = _make_config(
            worker_tools,
            initial_agents=2,
            claiming_mode=claiming,
            peer_monitoring_visible=peer_mon,
            trans_specialist_knowledge=trans_spec,
            predictive_knowledge=predictive,
        )
        config["_model"] = dummy_model
        agents = {}
        strategy.initialize(agents, config)

        # Basic sanity: agents created, blackboard exists.
        assert len(agents) == 2
        assert strategy.blackboard is not None
        assert strategy.blackboard.claiming_mode == claiming

        # Can produce a next_step action.
        action = strategy.next_step([], {"task": "Test task"})
        assert action.action_type == "invoke_agent"
