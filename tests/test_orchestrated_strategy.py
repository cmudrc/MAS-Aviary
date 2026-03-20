"""Tests for OrchestratedStrategy.

No GPU needed. Uses DummyModel and mock tools throughout.
"""


import pytest
from smolagents import ToolCallingAgent
from smolagents.models import Model

from src.coordination.history import AgentMessage, ToolCallRecord
from src.coordination.strategies.orchestrated import (
    DELEGATION_COMPLETE,
    OrchestratedStrategy,
)
from src.tools.mock_tools import CalculatorTool, EchoTool, StateTool

# ---- Fixtures ----------------------------------------------------------------

class DummyModel(Model):
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
def orchestrator_agent(dummy_model):
    """Pre-built orchestrator agent (tools injected by strategy)."""
    return ToolCallingAgent(
        tools=[],
        model=dummy_model,
        name="orchestrator",
        description="Creates and manages a team",
        instructions="You are an orchestrator.",
        max_steps=10,
        add_base_tools=False,
    )


@pytest.fixture
def worker_tools():
    """Available worker tools dict."""
    echo = EchoTool()
    calc = CalculatorTool()
    state = StateTool()
    return {echo.name: echo, calc.name: calc, state.name: state}


@pytest.fixture
def base_config():
    """Minimal valid orchestrated config."""
    return {
        "orchestrated": {
            "authority_mode": "orchestrator",
            "information_mode": "transparent",
            "lifecycle_mode": "setup_only",
            "max_agents": 8,
            "max_orchestrator_turns": 5,
            "worker_max_steps": 8,
        },
        "termination": {
            "keyword": "TASK_COMPLETE",
            "max_turns": 30,
        },
    }


@pytest.fixture
def strategy_with_tools(orchestrator_agent, worker_tools, base_config):
    """Initialized strategy with worker tools injected via config."""
    base_config["_worker_tools"] = worker_tools
    agents = {"orchestrator": orchestrator_agent}
    strategy = OrchestratedStrategy()
    strategy.initialize(agents, base_config)
    return strategy


# ---- Initialization tests ----------------------------------------------------

class TestInitialize:
    def test_reads_config(self, orchestrator_agent, base_config, worker_tools):
        base_config["_worker_tools"] = worker_tools
        base_config["orchestrated"]["max_agents"] = 4
        strategy = OrchestratedStrategy()
        strategy.initialize({"orchestrator": orchestrator_agent}, base_config)
        assert strategy._max_agents == 4
        assert strategy._lifecycle_mode == "setup_only"

    def test_identifies_orchestrator(self, strategy_with_tools):
        assert strategy_with_tools.orchestrator_name == "orchestrator"

    def test_injects_orchestrator_tools(self, orchestrator_agent, base_config, worker_tools):
        base_config["_worker_tools"] = worker_tools
        agents = {"orchestrator": orchestrator_agent}
        strategy = OrchestratedStrategy()
        strategy.initialize(agents, base_config)
        assert "list_available_tools" in orchestrator_agent.tools
        assert "create_agent" in orchestrator_agent.tools
        assert "assign_task" in orchestrator_agent.tools

    def test_missing_orchestrator_raises(self, base_config, worker_tools):
        base_config["_worker_tools"] = worker_tools
        strategy = OrchestratedStrategy()
        with pytest.raises(ValueError, match="not found"):
            strategy.initialize({}, base_config)

    def test_phase_starts_at_creation(self, strategy_with_tools):
        assert strategy_with_tools.phase == "creation"

    def test_context_has_available_tools(self, strategy_with_tools):
        ctx = strategy_with_tools.context
        assert "echo_tool" in ctx.available_tools
        assert "calculator_tool" in ctx.available_tools


# ---- Phase 1: Team creation -------------------------------------------------

class TestCreationPhase:
    def test_first_step_invokes_orchestrator(self, strategy_with_tools):
        action = strategy_with_tools.next_step([], {"task": "Do something"})
        assert action.action_type == "invoke_agent"
        assert action.agent_name == "orchestrator"
        assert action.input_context == "Do something"

    def test_subsequent_steps_invoke_orchestrator(self, strategy_with_tools):
        strategy_with_tools.next_step([], {"task": "Task"})
        msg = AgentMessage(
            agent_name="orchestrator", content="Thinking...",
            turn_number=1, timestamp=1.0,
        )
        action2 = strategy_with_tools.next_step([msg], {"task": "Task"})
        assert action2.action_type == "invoke_agent"
        assert action2.agent_name == "orchestrator"

    def test_delegation_complete_transitions(self, strategy_with_tools):
        # Simulate orchestrator creating an agent and assigning a task.
        ctx = strategy_with_tools.context
        ctx.created_agents.append("worker1")
        ctx.agents["worker1"] = "placeholder_agent"
        ctx.assignments.append({"agent_name": "worker1", "task": "Do work", "assigned_at_turn": 1})

        msg = AgentMessage(
            agent_name="orchestrator",
            content=f"Team ready. {DELEGATION_COMPLETE}",
            turn_number=1, timestamp=1.0,
        )
        strategy_with_tools.next_step([msg], {"task": "Task"})
        assert strategy_with_tools.phase == "execution"

    def test_max_orchestrator_turns_transitions(self, orchestrator_agent, base_config, worker_tools):
        base_config["_worker_tools"] = worker_tools
        base_config["orchestrated"]["max_orchestrator_turns"] = 2
        agents = {"orchestrator": orchestrator_agent}
        strategy = OrchestratedStrategy()
        strategy.initialize(agents, base_config)

        # Simulate orchestrator creating two agents with assignments.
        strategy.context.created_agents.extend(["w1", "w2"])
        strategy.context.agents["w1"] = "placeholder"
        strategy.context.agents["w2"] = "placeholder"
        strategy.context.assignments.extend([
            {"agent_name": "w1", "task": "T1", "assigned_at_turn": 1},
            {"agent_name": "w2", "task": "T2", "assigned_at_turn": 2},
        ])

        history = []
        for i in range(2):
            action = strategy.next_step(history, {"task": "Task"})
            assert action.agent_name == "orchestrator"
            history.append(AgentMessage(
                agent_name="orchestrator", content="working",
                turn_number=i + 1, timestamp=float(i),
            ))

        # Third call exceeds max_orchestrator_turns=2 → transitions to execution.
        action = strategy.next_step(history, {"task": "Task"})
        assert strategy.phase == "execution"
        assert action.agent_name == "w1"  # first worker runs

    def test_no_assignments_terminates(self, strategy_with_tools):
        # Orchestrator finished but created no assignments.
        msg = AgentMessage(
            agent_name="orchestrator",
            content=DELEGATION_COMPLETE,
            turn_number=1, timestamp=1.0,
        )
        action = strategy_with_tools.next_step([msg], {"task": "Task"})
        assert action.action_type == "terminate"
        assert strategy_with_tools.phase == "done"


# ---- Phase 2: Execution (setup_only) ----------------------------------------

class TestSetupOnlyExecution:
    def _setup_with_assignments(self, strategy_with_tools):
        """Manually set up created agents and assignments."""
        ctx = strategy_with_tools.context
        # Create mock workers in the agent pool.
        ctx.created_agents.extend(["w1", "w2"])
        ctx.agents["w1"] = "mock_agent_w1"
        ctx.agents["w2"] = "mock_agent_w2"
        ctx.assignments.extend([
            {"agent_name": "w1", "task": "Task 1", "assigned_at_turn": 1},
            {"agent_name": "w2", "task": "Task 2", "assigned_at_turn": 2},
        ])
        # Transition to execution.
        strategy_with_tools._phase = "execution"
        return strategy_with_tools

    def test_runs_workers_in_order(self, strategy_with_tools):
        s = self._setup_with_assignments(strategy_with_tools)
        action1 = s.next_step([], {"task": "T"})
        assert action1.agent_name == "w1"
        assert "Task 1" in action1.input_context

        msg1 = AgentMessage(agent_name="w1", content="done1", turn_number=1, timestamp=1.0)
        action2 = s.next_step([msg1], {"task": "T"})
        assert action2.agent_name == "w2"
        assert "Task 2" in action2.input_context

    def test_passes_context_to_next(self, strategy_with_tools):
        s = self._setup_with_assignments(strategy_with_tools)
        s.next_step([], {"task": "T"})  # w1
        msg1 = AgentMessage(agent_name="w1", content="output_from_w1", turn_number=1, timestamp=1.0)
        action2 = s.next_step([msg1], {"task": "T"})
        assert "output_from_w1" in action2.input_context

    def test_terminates_after_all_assignments(self, strategy_with_tools):
        s = self._setup_with_assignments(strategy_with_tools)
        # Run both workers.
        s.next_step([], {"task": "T"})
        msg1 = AgentMessage(agent_name="w1", content="d1", turn_number=1, timestamp=1.0)
        s.next_step([msg1], {"task": "T"})
        msg2 = AgentMessage(agent_name="w2", content="d2", turn_number=2, timestamp=2.0)
        action3 = s.next_step([msg1, msg2], {"task": "T"})
        assert action3.action_type == "terminate"
        assert s.phase == "done"


# ---- Phase 2: Execution (active) --------------------------------------------

class TestActiveExecution:
    def _setup_active(self, orchestrator_agent, base_config, worker_tools):
        base_config["_worker_tools"] = worker_tools
        base_config["orchestrated"]["lifecycle_mode"] = "active"
        agents = {"orchestrator": orchestrator_agent}
        strategy = OrchestratedStrategy()
        strategy.initialize(agents, base_config)

        ctx = strategy.context
        ctx.created_agents.extend(["w1"])
        ctx.agents["w1"] = "mock"
        ctx.assignments.append({"agent_name": "w1", "task": "T1", "assigned_at_turn": 1})
        strategy._phase = "execution"
        return strategy

    def test_runs_worker_then_orchestrator_review(self, orchestrator_agent, base_config, worker_tools):
        s = self._setup_active(orchestrator_agent, base_config, worker_tools)

        # First action: worker.
        action1 = s.next_step([], {"task": "T"})
        assert action1.agent_name == "w1"

        # After worker, orchestrator reviews.
        msg1 = AgentMessage(agent_name="w1", content="worker output", turn_number=1, timestamp=1.0)
        action2 = s.next_step([msg1], {"task": "T"})
        assert action2.agent_name == "orchestrator"

    def test_terminates_after_review(self, orchestrator_agent, base_config, worker_tools):
        s = self._setup_active(orchestrator_agent, base_config, worker_tools)

        s.next_step([], {"task": "T"})  # w1
        msg1 = AgentMessage(agent_name="w1", content="done", turn_number=1, timestamp=1.0)
        s.next_step([msg1], {"task": "T"})  # orchestrator review
        msg2 = AgentMessage(agent_name="orchestrator", content="reviewed", turn_number=2, timestamp=2.0)
        action3 = s.next_step([msg1, msg2], {"task": "T"})
        assert action3.action_type == "terminate"


# ---- Information mode --------------------------------------------------------

class TestInformationMode:
    def test_transparent_includes_full_output(self, orchestrator_agent, base_config, worker_tools):
        base_config["_worker_tools"] = worker_tools
        base_config["orchestrated"]["information_mode"] = "transparent"
        strategy = OrchestratedStrategy()
        strategy.initialize({"orchestrator": orchestrator_agent}, base_config)

        msg = AgentMessage(
            agent_name="worker1", content="Full detailed output here",
            turn_number=1, timestamp=1.0, duration_seconds=2.5,
        )
        context = strategy._format_context_for_orchestrator([msg])
        assert "Full detailed output here" in context
        assert "2.5s" in context

    def test_opaque_strips_output(self, orchestrator_agent, base_config, worker_tools):
        base_config["_worker_tools"] = worker_tools
        base_config["orchestrated"]["information_mode"] = "opaque"
        strategy = OrchestratedStrategy()
        strategy.initialize({"orchestrator": orchestrator_agent}, base_config)

        msg = AgentMessage(
            agent_name="worker1", content="Detailed secret output",
            turn_number=1, timestamp=1.0, duration_seconds=2.5,
        )
        context = strategy._format_context_for_orchestrator([msg])
        assert "Detailed secret output" not in context
        assert "SUCCESS" in context

    def test_opaque_shows_failed(self, orchestrator_agent, base_config, worker_tools):
        base_config["_worker_tools"] = worker_tools
        base_config["orchestrated"]["information_mode"] = "opaque"
        strategy = OrchestratedStrategy()
        strategy.initialize({"orchestrator": orchestrator_agent}, base_config)

        msg = AgentMessage(
            agent_name="worker1", content="",
            turn_number=1, timestamp=1.0, error="Something broke",
        )
        context = strategy._format_context_for_orchestrator([msg])
        assert "FAILED" in context

    def test_transparent_includes_tool_calls(self, orchestrator_agent, base_config, worker_tools):
        base_config["_worker_tools"] = worker_tools
        base_config["orchestrated"]["information_mode"] = "transparent"
        strategy = OrchestratedStrategy()
        strategy.initialize({"orchestrator": orchestrator_agent}, base_config)

        msg = AgentMessage(
            agent_name="worker1", content="output",
            turn_number=1, timestamp=1.0,
            tool_calls=[ToolCallRecord(
                tool_name="calculator_tool",
                inputs={"expression": "2+2"},
                output="4",
                duration_seconds=0.1,
            )],
        )
        context = strategy._format_context_for_orchestrator([msg])
        assert "calculator_tool" in context


# ---- Authority transfer ------------------------------------------------------

class TestAuthorityTransfer:
    def test_scores_computed_correctly(self, strategy_with_tools):
        history = [
            AgentMessage(
                agent_name="worker1", content="ok", turn_number=1, timestamp=1.0,
                tool_calls=[
                    ToolCallRecord("calc", {}, "4", 0.1),
                    ToolCallRecord("calc", {}, "err", 0.1, error="failed"),
                ],
            ),
            AgentMessage(
                agent_name="worker1", content="ok", turn_number=2, timestamp=2.0,
            ),
        ]
        scores = strategy_with_tools.compute_authority_scores(history)
        assert "worker1" in scores
        # tool_error_rate = 1/2 = 0.5 → (1-0.5)*0.5 = 0.25
        # retry_rate = 0/2 = 0 → (1-0)*0.3 = 0.3
        # completion_rate = 2/2 = 1.0 → 1.0*0.2 = 0.2
        # total = 0.75
        assert scores["worker1"] == 0.75

    def test_transfer_triggers_at_threshold(self, orchestrator_agent, base_config, worker_tools):
        base_config["_worker_tools"] = worker_tools
        base_config["orchestrated"]["authority_mode"] = "delegated"
        base_config["orchestrated"]["authority_transfer_after"] = 2
        strategy = OrchestratedStrategy()
        strategy.initialize({"orchestrator": orchestrator_agent}, base_config)

        # Add a created worker to context.
        strategy.context.created_agents.append("best_worker")
        strategy.context.agents["best_worker"] = "placeholder"

        # History where worker outperforms orchestrator.
        history = [
            AgentMessage(agent_name="orchestrator", content="ok", turn_number=1,
                         timestamp=1.0, error="had an error"),
            AgentMessage(agent_name="best_worker", content="great", turn_number=2,
                         timestamp=2.0),
            AgentMessage(agent_name="best_worker", content="perfect", turn_number=3,
                         timestamp=3.0),
        ]

        # First prompt: not enough yet.
        result1 = strategy.check_authority_transfer(history)
        assert result1 is None

        # Second prompt: triggers transfer.
        result2 = strategy.check_authority_transfer(history)
        assert result2 is not None
        assert result2["event"] == "authority_transfer"
        assert result2["to"] == "best_worker"

    def test_no_transfer_when_orchestrator_mode(self, strategy_with_tools):
        result = strategy_with_tools.check_authority_transfer([])
        assert result is None

    def test_manual_authority_sets_orchestrator(self, dummy_model, base_config, worker_tools):
        base_config["_worker_tools"] = worker_tools
        base_config["orchestrated"]["authority_mode"] = "manual"
        base_config["orchestrated"]["manual_authority_agent"] = "special_agent"

        # Create the manual authority agent.
        special = ToolCallingAgent(
            tools=[], model=dummy_model, name="special_agent",
            description="Manual authority", add_base_tools=False,
        )
        orchestrator = ToolCallingAgent(
            tools=[], model=dummy_model, name="orchestrator",
            description="Default orch", add_base_tools=False,
        )
        agents = {"orchestrator": orchestrator, "special_agent": special}

        strategy = OrchestratedStrategy()
        strategy.initialize(agents, base_config)
        assert strategy.orchestrator_name == "special_agent"
        # Special agent should have orchestrator tools.
        assert "create_agent" in special.tools


# ---- is_complete tests -------------------------------------------------------

class TestIsComplete:
    def test_done_phase_is_complete(self, strategy_with_tools):
        strategy_with_tools._phase = "done"
        assert strategy_with_tools.is_complete([], {}) is True

    def test_task_complete_keyword(self, strategy_with_tools):
        msg = AgentMessage(
            agent_name="w", content="TASK_COMPLETE all done",
            turn_number=1, timestamp=1.0,
        )
        assert strategy_with_tools.is_complete([msg], {}) is True

    def test_max_turns(self, orchestrator_agent, base_config, worker_tools):
        base_config["_worker_tools"] = worker_tools
        base_config["termination"]["max_turns"] = 3
        strategy = OrchestratedStrategy()
        strategy.initialize({"orchestrator": orchestrator_agent}, base_config)

        msgs = [
            AgentMessage(agent_name="a", content="x", turn_number=i, timestamp=float(i))
            for i in range(4)
        ]
        assert strategy.is_complete(msgs, {}) is True

    def test_not_complete_during_creation(self, strategy_with_tools):
        msg = AgentMessage(
            agent_name="orchestrator", content="still thinking",
            turn_number=1, timestamp=1.0,
        )
        assert strategy_with_tools.is_complete([msg], {}) is False
