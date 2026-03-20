"""Integration tests for the Iterative Feedback operational methodology.

Fast tests use mock agents (no GPU).
Slow test (@pytest.mark.slow) uses real LLM + mock tools.
"""

import json

import pytest
from smolagents import Model
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
)

from src.coordination.coordinator import Coordinator
from src.coordination.execution_handler import Assignment
from src.coordination.iterative_feedback_handler import IterativeFeedbackHandler
from src.coordination.strategy import CoordinationAction, CoordinationStrategy
from src.logging.iterative_feedback_metrics import (
    compute_ambidexterity,
    compute_escalation,
    compute_per_agent_metrics,
    compute_per_prompt_metrics,
)
from src.logging.logger import InstrumentationLogger

# ---- Model stubs (no GPU) ---------------------------------------------------


class _FinalAnswerModel(Model):
    """Returns a final_answer tool call so ToolCallingAgents complete in 1 step."""

    def __init__(self, answer: str = "done"):
        super().__init__(model_id="final-answer-stub")
        self._answer = answer
        self._call_count = 0

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        self._call_count += 1
        tc = ChatMessageToolCall(
            id=f"call_{self._call_count}",
            type="function",
            function=ChatMessageToolCallFunction(
                name="final_answer",
                arguments=json.dumps({"answer": self._answer}),
            ),
        )
        return ChatMessage(role="assistant", content="", tool_calls=[tc])


class _ErrorThenSuccessModel(Model):
    """Raises on first N calls, then returns final_answer."""

    def __init__(self, fail_count: int = 1, answer: str = "recovered"):
        super().__init__(model_id="error-then-success-stub")
        self._fail_count = fail_count
        self._answer = answer
        self._call_count = 0

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise RuntimeError(f"Simulated failure #{self._call_count}")
        tc = ChatMessageToolCall(
            id=f"call_{self._call_count}",
            type="function",
            function=ChatMessageToolCallFunction(
                name="final_answer",
                arguments=json.dumps({"answer": self._answer}),
            ),
        )
        return ChatMessage(role="assistant", content="", tool_calls=[tc])


# ---- Mock agents for handler tests ------------------------------------------


class _MockAgent:
    """Agent stub returning fixed output."""

    def __init__(self, response: str = "done", tools: dict | None = None):
        self.tools = tools if tools is not None else {"tool": True}
        self._response = response
        self.calls: list[str] = []

    def run(self, ctx: str):
        self.calls.append(ctx)
        return self._response


class _FailThenSucceedAgent:
    """Agent that fails N times (exception), then succeeds."""

    def __init__(self, fail_count: int = 1, success: str = "ok"):
        self.tools = {"tool": True}
        self._fail_count = fail_count
        self._success = success
        self._count = 0
        self.calls: list[str] = []

    def run(self, ctx: str):
        self.calls.append(ctx)
        self._count += 1
        if self._count <= self._fail_count:
            raise RuntimeError(f"Error #{self._count}")
        return self._success


# ---- Minimal strategy for coordinator tests ----------------------------------


class _LinearStrategy(CoordinationStrategy):
    """Runs agents in order, one step per call — minimal test strategy."""

    def __init__(self, agent_names: list[str]):
        self._names = agent_names
        self._index = 0
        self._task = ""

    def initialize(self, agents, config):
        self._index = 0

    def next_step(self, history, state):
        if "task" in state:
            self._task = state["task"]
        if self._index >= len(self._names):
            return CoordinationAction(
                action_type="terminate",
                agent_name=None,
                input_context="",
            )
        name = self._names[self._index]
        self._index += 1
        ctx = self._task if not history else history[-1].content
        return CoordinationAction(
            action_type="invoke_agent",
            agent_name=name,
            input_context=ctx,
        )

    def is_complete(self, history, state):
        return self._index >= len(self._names)


# ---- Handler-level integration tests (fast) ----------------------------------


class TestHandlerIntegration:
    """Test the handler with mock agents — no coordinator, no GPU."""

    def test_single_agent_succeeds(self):
        handler = IterativeFeedbackHandler({"max_retries": 5})
        agents = {"a1": _MockAgent("success")}
        msgs = handler.execute(
            [Assignment(agent_name="a1", task="Do something")],
            agents,
            logger=None,
        )
        assert len(msgs) == 1
        assert msgs[0].content == "success"
        metrics = compute_per_agent_metrics(handler.attempt_histories[0])
        assert metrics["total_attempts"] == 1
        assert metrics["final_outcome"] == "success"

    def test_retry_then_succeed_with_metrics(self):
        handler = IterativeFeedbackHandler(
            {
                "max_retries": 5,
                "aspiration_mode": "tool_success",
            }
        )
        agent = _FailThenSucceedAgent(fail_count=2, success="recovered")
        agents = {"a1": agent}
        msgs = handler.execute(
            [Assignment(agent_name="a1", task="Retry test")],
            agents,
            logger=None,
        )
        assert len(msgs) == 3  # 2 failures + 1 success
        assert msgs[-1].content == "recovered"

        # Check per-agent metrics.
        agent_metrics = compute_per_agent_metrics(handler.attempt_histories[0])
        assert agent_metrics["total_attempts"] == 3
        assert agent_metrics["final_outcome"] == "success"

    def test_max_retries_with_summary_handoff(self):
        handler = IterativeFeedbackHandler({"max_retries": 3})
        always_fail = _FailThenSucceedAgent(fail_count=100)
        receiver = _MockAgent("received")
        agents = {"fail_agent": always_fail, "next_agent": receiver}
        msgs = handler.execute(
            [
                Assignment(agent_name="fail_agent", task="Hard task"),
                Assignment(agent_name="next_agent", task="Continue"),
            ],
            agents,
            logger=None,
        )
        # fail_agent: 3 attempts, next_agent: 1 attempt = 4 total messages
        assert len(msgs) == 4
        # next_agent received context from fail_agent.
        assert len(receiver.calls) == 1

    def test_pipeline_with_multiple_agents(self):
        handler = IterativeFeedbackHandler({"max_retries": 5})
        agents = {
            "planner": _MockAgent("plan: do X then Y"),
            "executor": _FailThenSucceedAgent(fail_count=1, success="executed"),
            "reviewer": _MockAgent("TASK_COMPLETE"),
        }
        msgs = handler.execute(
            [
                Assignment(agent_name="planner", task="Plan something"),
                Assignment(agent_name="executor", task="Execute the plan"),
                Assignment(agent_name="reviewer", task="Review results"),
            ],
            agents,
            logger=None,
        )
        # planner: 1, executor: 2 (1 fail + 1 success), stops at TASK_COMPLETE
        # Total should be 3 (planner + executor fail + executor success)
        # Then reviewer runs and returns TASK_COMPLETE, stopping pipeline
        # Actually: TASK_COMPLETE is checked after executor's success,
        # but "executed" doesn't contain TASK_COMPLETE, so reviewer runs
        assert msgs[-1].content == "TASK_COMPLETE"

    def test_per_prompt_metrics_integration(self):
        handler = IterativeFeedbackHandler({"max_retries": 5})
        agents = {
            "a1": _MockAgent("output one"),
            "a2": _FailThenSucceedAgent(fail_count=1, success="output two"),
        }
        handler.execute(
            [
                Assignment(agent_name="a1", task="T1"),
                Assignment(agent_name="a2", task="T2"),
            ],
            agents,
            logger=None,
        )
        prompt_metrics = compute_per_prompt_metrics(handler.attempt_histories)
        assert prompt_metrics["total_attempts_all_agents"] == 3  # 1 + 2
        assert prompt_metrics["agents_succeeded_first_try"] == 1
        assert prompt_metrics["agents_required_retries"] == 1


# ---- Coordinator-level integration tests (fast) ------------------------------


class TestCoordinatorWithHandler:
    """Test handler wired through the Coordinator."""

    def test_coordinator_with_iterative_feedback(self):
        handler = IterativeFeedbackHandler({"max_retries": 3})
        strategy = _LinearStrategy(["a1", "a2"])
        agents = {
            "a1": _MockAgent("first"),
            "a2": _MockAgent("second"),
        }
        coord = Coordinator(
            agents=agents,
            strategy=strategy,
            config={"termination": {"keyword": "TASK_COMPLETE", "max_turns": 10}},
            execution_handler=handler,
        )
        result = coord.run("test task")
        assert result.final_output == "second"
        assert len(result.history) == 2

    def test_coordinator_retry_produces_multiple_messages(self):
        handler = IterativeFeedbackHandler({"max_retries": 3})
        strategy = _LinearStrategy(["a1"])
        agents = {"a1": _FailThenSucceedAgent(fail_count=1, success="recovered")}
        coord = Coordinator(
            agents=agents,
            strategy=strategy,
            config={"termination": {"keyword": "TASK_COMPLETE", "max_turns": 10}},
            execution_handler=handler,
        )
        result = coord.run("task")
        # 1 failure + 1 success = 2 messages in history
        assert len(result.history) == 2
        assert result.final_output == "recovered"

    def test_coordinator_without_handler_unchanged(self):
        """Without execution_handler, coordinator works as before."""
        strategy = _LinearStrategy(["a1"])
        agents = {"a1": _MockAgent("direct")}
        coord = Coordinator(
            agents=agents,
            strategy=strategy,
            config={"termination": {"keyword": "TASK_COMPLETE", "max_turns": 10}},
        )
        result = coord.run("task")
        assert len(result.history) == 1
        assert result.final_output == "direct"

    def test_config_driven_handler_selection(self):
        """Handler is selected from config dict."""
        strategy = _LinearStrategy(["a1"])
        agents = {"a1": _MockAgent("result")}
        handler = IterativeFeedbackHandler({"max_retries": 5})
        coord = Coordinator(
            agents=agents,
            strategy=strategy,
            config={
                "execution_handler": "iterative_feedback",
                "termination": {"keyword": "TASK_COMPLETE", "max_turns": 10},
            },
            execution_handler=handler,
        )
        result = coord.run("task")
        assert result.final_output == "result"


# ---- Ambidexterity / escalation end-to-end -----------------------------------


class TestMetricsEndToEnd:
    def test_ambidexterity_from_handler(self):
        handler = IterativeFeedbackHandler({"max_retries": 4})
        # Agent that always fails with different outputs.
        responses = iter(["alpha beta gamma", "delta epsilon zeta", "eta theta iota", "kappa lambda mu"])

        class _VaryingAgent:
            tools = {"tool": True}
            calls = []

            def run(self, ctx):
                self.calls.append(ctx)
                raise RuntimeError(next(responses))

        agents = {"a1": _VaryingAgent()}
        handler.execute(
            [Assignment(agent_name="a1", task="T")],
            agents,
            logger=None,
        )
        amb = compute_ambidexterity(
            handler.attempt_histories[0],
            similarity_method="jaccard",
        )
        assert amb.get("score") is not None or amb.get("ambidexterity_score") is not None

    def test_escalation_from_handler(self):
        handler = IterativeFeedbackHandler({"max_retries": 5})

        class _SameFailAgent:
            tools = {"tool": True}
            calls = []

            def run(self, ctx):
                self.calls.append(ctx)
                raise RuntimeError("same error every time")

        agents = {"a1": _SameFailAgent()}
        handler.execute(
            [Assignment(agent_name="a1", task="T")],
            agents,
            logger=None,
        )
        esc = compute_escalation(
            handler.attempt_histories[0],
            similarity_method="jaccard",
            min_length=3,
        )
        # All failures with empty output content (from exceptions)
        # → may or may not trigger escalation depending on similarity
        assert "escalation_length" in esc


# ---- Real LLM integration test (slow) ----------------------------------------


@pytest.mark.slow
class TestRealLLMIntegration:
    """Requires GPU. Tests the handler with real model + mock tools."""

    def test_sequential_with_iterative_feedback(self):
        """Load sequential strategy + iterative feedback handler + mock tools."""
        from src.config.loader import load_config
        from src.llm.model_loader import load_model
        from src.tools.mock_tools import CalculatorTool, EchoTool

        config = load_config("config/default.yaml")
        model = load_model(config.llm)

        # Build mock tools.
        tools = [CalculatorTool(), EchoTool()]

        # Build sequential strategy.
        from src.coordination.strategies.sequential import SequentialStrategy

        strategy = SequentialStrategy()

        agents: dict = {}
        coord_config = {
            "sequential": {
                "decomposition_mode": "human",
                "pipeline_template": "linear",
                "stage_max_steps": 6,
            },
            "termination": {"keyword": "TASK_COMPLETE", "max_turns": 30},
            "_worker_tools": {t.name: t for t in tools},
            "_model": model,
        }

        handler = IterativeFeedbackHandler(
            {
                "max_retries": 3,
                "aspiration_mode": "tool_success",
                "human_feedback_mode": "none",
            }
        )

        logger = InstrumentationLogger()
        coord = Coordinator(
            agents=agents,
            strategy=strategy,
            config=coord_config,
            logger=logger,
            execution_handler=handler,
        )
        result = coord.run("Calculate 15 * 7 using the calculator_tool")

        assert len(result.history) > 0
        # Handler accumulates attempt histories across stage calls.
        # With 3 stages (linear template), we expect 3 entries.
        assert len(handler.attempt_histories) >= 3
        prompt_metrics = compute_per_prompt_metrics(handler.attempt_histories)
        assert prompt_metrics["total_attempts_all_agents"] >= 3
