"""Tests for IterativeFeedbackHandler.

No GPU needed. Uses mock agents that return controlled responses.
"""

from src.coordination.execution_handler import Assignment
from src.coordination.feedback_extraction import AttemptFeedback
from src.coordination.iterative_feedback_handler import IterativeFeedbackHandler
from src.logging.logger import InstrumentationLogger

# ---- Mock Agents -------------------------------------------------------------


class MockAgent:
    """Agent that returns a fixed response."""

    def __init__(self, response: str = "done", tools: dict | None = None):
        self.tools = tools or {"my_tool": True}
        self._response = response
        self.calls: list[str] = []

    def run(self, input_context: str):
        self.calls.append(input_context)
        return self._response


class SequenceAgent:
    """Agent that returns a different response on each call."""

    def __init__(self, responses: list[str], tools: dict | None = None):
        self.tools = tools or {"my_tool": True}
        self._responses = list(responses)
        self._index = 0
        self.calls: list[str] = []

    def run(self, input_context: str):
        self.calls.append(input_context)
        resp = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return resp


class ErrorThenSuccessAgent:
    """Agent that raises on first N calls, then succeeds."""

    def __init__(self, fail_count: int = 1, success_response: str = "ok", tools: dict | None = None):
        self.tools = tools or {"my_tool": True}
        self._fail_count = fail_count
        self._success = success_response
        self._call = 0
        self.calls: list[str] = []

    def run(self, input_context: str):
        self.calls.append(input_context)
        self._call += 1
        if self._call <= self._fail_count:
            raise RuntimeError(f"fail #{self._call}")
        return self._success


class ToollessAgent:
    """Agent with no tools — pure reasoning."""

    def __init__(self, response: str = "reasoning output"):
        self.tools = {}
        self._response = response
        self.calls: list[str] = []

    def run(self, input_context: str):
        self.calls.append(input_context)
        return self._response


# ---- Helpers -----------------------------------------------------------------


def _cfg(**overrides) -> dict:
    """Build a config dict with defaults."""
    base = {
        "max_retries": 5,
        "feedback_window": 3,
        "retry_toolless_agents": False,
        "aspiration_mode": "tool_success",
        "human_feedback_mode": "none",
        "termination_keyword": "TASK_COMPLETE",
    }
    base.update(overrides)
    return base


# ---- Basic execution ---------------------------------------------------------


class TestBasicExecution:
    def test_agent_succeeds_first_try(self):
        handler = IterativeFeedbackHandler(_cfg())
        agents = {"a1": MockAgent("success")}
        assignments = [Assignment(agent_name="a1", task="Do it")]
        msgs = handler.execute(assignments, agents, logger=None)
        assert len(msgs) == 1
        assert msgs[0].content == "success"
        assert msgs[0].is_retry is False

    def test_multiple_assignments_in_order(self):
        handler = IterativeFeedbackHandler(_cfg())
        agents = {
            "a1": MockAgent("result_a"),
            "a2": MockAgent("result_b"),
        }
        assignments = [
            Assignment(agent_name="a1", task="Task A"),
            Assignment(agent_name="a2", task="Task B"),
        ]
        msgs = handler.execute(assignments, agents, logger=None)
        assert len(msgs) == 2
        assert msgs[0].content == "result_a"
        assert msgs[1].content == "result_b"

    def test_previous_output_passed_as_context(self):
        handler = IterativeFeedbackHandler(_cfg())
        a1 = MockAgent("first_output")
        a2 = MockAgent("second_output")
        agents = {"a1": a1, "a2": a2}
        assignments = [
            Assignment(agent_name="a1", task="T1"),
            Assignment(agent_name="a2", task="T2"),
        ]
        handler.execute(assignments, agents, logger=None)
        assert "first_output" in a2.calls[0]

    def test_missing_agent(self):
        handler = IterativeFeedbackHandler(_cfg())
        msgs = handler.execute(
            [Assignment(agent_name="ghost", task="Task")],
            {},
            logger=None,
        )
        assert len(msgs) == 1
        assert msgs[0].error is not None
        assert "not found" in msgs[0].error

    def test_empty_assignments(self):
        handler = IterativeFeedbackHandler(_cfg())
        msgs = handler.execute([], {}, logger=None)
        assert msgs == []

    def test_logs_all_turns(self):
        handler = IterativeFeedbackHandler(_cfg())
        logger = InstrumentationLogger()
        agents = {"a1": MockAgent("done")}
        handler.execute(
            [Assignment(agent_name="a1", task="T")],
            agents,
            logger=logger,
        )
        assert logger.turn_count == 1


# ---- Retry behaviour ---------------------------------------------------------


class TestRetryBehaviour:
    def test_agent_fails_then_succeeds(self):
        handler = IterativeFeedbackHandler(_cfg())
        agent = ErrorThenSuccessAgent(fail_count=2, success_response="ok")
        agents = {"a1": agent}
        msgs = handler.execute(
            [Assignment(agent_name="a1", task="Try")],
            agents,
            logger=None,
        )
        # 2 failures + 1 success = 3 messages
        assert len(msgs) == 3
        assert msgs[0].error is not None
        assert msgs[1].error is not None
        assert msgs[2].content == "ok"
        assert msgs[2].is_retry is True

    def test_feedback_injected_into_retry(self):
        handler = IterativeFeedbackHandler(_cfg())
        agent = ErrorThenSuccessAgent(fail_count=1, success_response="fixed")
        agents = {"a1": agent}
        handler.execute(
            [Assignment(agent_name="a1", task="Task")],
            agents,
            logger=None,
        )
        # Second call should include feedback text.
        assert "previous attempt" in agent.calls[1].lower() or "attempt" in agent.calls[1].lower()

    def test_max_retries_exhausted(self):
        handler = IterativeFeedbackHandler(_cfg(max_retries=3))
        agent = ErrorThenSuccessAgent(fail_count=100)  # always fails
        agents = {"a1": agent}
        msgs = handler.execute(
            [Assignment(agent_name="a1", task="Task")],
            agents,
            logger=None,
        )
        assert len(msgs) == 3

    def test_max_retries_capped_at_20(self):
        handler = IterativeFeedbackHandler(_cfg(max_retries=50))
        assert handler._max_retries == 20

    def test_penultimate_warning_injected(self):
        handler = IterativeFeedbackHandler(_cfg(max_retries=4))
        agent = ErrorThenSuccessAgent(fail_count=100)
        agents = {"a1": agent}
        handler.execute(
            [Assignment(agent_name="a1", task="Task")],
            agents,
            logger=None,
        )
        # Attempt index 2 (3rd attempt, 0-indexed) = max_retries-2 = penultimate
        assert "second-to-last" in agent.calls[2].lower()

    def test_final_attempt_forces_summary(self):
        handler = IterativeFeedbackHandler(_cfg(max_retries=3))
        agent = ErrorThenSuccessAgent(fail_count=100)
        agents = {"a1": agent}
        handler.execute(
            [Assignment(agent_name="a1", task="Task")],
            agents,
            logger=None,
        )
        # Last call should contain summary instruction.
        assert "final attempt" in agent.calls[-1].lower()

    def test_summary_passed_to_next_agent(self):
        handler = IterativeFeedbackHandler(_cfg(max_retries=2))
        failing = ErrorThenSuccessAgent(fail_count=100)
        receiver = MockAgent("got it")
        agents = {"fail_agent": failing, "next_agent": receiver}
        assignments = [
            Assignment(agent_name="fail_agent", task="Hard task"),
            Assignment(agent_name="next_agent", task="Continue"),
        ]
        handler.execute(assignments, agents, logger=None)
        # next_agent should receive context from fail_agent's last output
        # (which was an error, so content is "")
        assert len(receiver.calls) == 1

    def test_attempt_histories_populated(self):
        handler = IterativeFeedbackHandler(_cfg(max_retries=3))
        agent = ErrorThenSuccessAgent(fail_count=1)
        agents = {"a1": agent}
        handler.execute(
            [Assignment(agent_name="a1", task="T")],
            agents,
            logger=None,
        )
        assert len(handler.attempt_histories) == 1
        assert len(handler.attempt_histories[0]) == 2  # 1 fail + 1 success


# ---- Aspiration modes --------------------------------------------------------


class TestAspirationModes:
    def test_tool_success_moves_on(self):
        """Agent with no tool errors → aspiration met, no retry."""
        handler = IterativeFeedbackHandler(_cfg(aspiration_mode="tool_success"))
        agent = MockAgent("result")
        agents = {"a1": agent}
        msgs = handler.execute(
            [Assignment(agent_name="a1", task="T")],
            agents,
            logger=None,
        )
        assert len(msgs) == 1

    def test_any_output_moves_on(self):
        handler = IterativeFeedbackHandler(_cfg(aspiration_mode="any_output"))
        agent = MockAgent("some output")
        agents = {"a1": agent}
        msgs = handler.execute(
            [Assignment(agent_name="a1", task="T")],
            agents,
            logger=None,
        )
        assert len(msgs) == 1

    def test_any_output_retries_on_empty(self):
        handler = IterativeFeedbackHandler(
            _cfg(
                aspiration_mode="any_output",
                max_retries=3,
            )
        )
        agent = SequenceAgent(["", "", "finally"])
        agents = {"a1": agent}
        msgs = handler.execute(
            [Assignment(agent_name="a1", task="T")],
            agents,
            logger=None,
        )
        assert len(msgs) == 3
        assert msgs[-1].content == "finally"

    def test_no_tool_errors_or_max(self):
        handler = IterativeFeedbackHandler(
            _cfg(
                aspiration_mode="no_tool_errors_or_max",
                max_retries=3,
            )
        )
        agent = MockAgent("output")
        agents = {"a1": agent}
        msgs = handler.execute(
            [Assignment(agent_name="a1", task="T")],
            agents,
            logger=None,
        )
        assert len(msgs) == 1

    def test_custom_aspiration(self):
        def custom_check(fb: AttemptFeedback) -> bool:
            return "PASS" in fb.output_content

        handler = IterativeFeedbackHandler(
            _cfg(
                aspiration_mode="custom",
                aspiration_threshold=custom_check,
                max_retries=5,
            )
        )
        agent = SequenceAgent(["nope", "nope", "PASS"])
        agents = {"a1": agent}
        msgs = handler.execute(
            [Assignment(agent_name="a1", task="T")],
            agents,
            logger=None,
        )
        assert len(msgs) == 3
        assert "PASS" in msgs[-1].content


# ---- Toolless agents ---------------------------------------------------------


class TestToollessAgents:
    def test_toolless_no_retry_default(self):
        handler = IterativeFeedbackHandler(_cfg(retry_toolless_agents=False))
        agent = ToollessAgent("reasoning")
        agents = {"a1": agent}
        msgs = handler.execute(
            [Assignment(agent_name="a1", task="Think")],
            agents,
            logger=None,
        )
        assert len(msgs) == 1
        assert len(agent.calls) == 1

    def test_toolless_with_retry_enabled(self):
        handler = IterativeFeedbackHandler(
            _cfg(
                retry_toolless_agents=True,
                aspiration_mode="any_output",
                max_retries=3,
            )
        )
        agent = SequenceAgent(["", "", "got it"], tools={})
        agents = {"a1": agent}
        msgs = handler.execute(
            [Assignment(agent_name="a1", task="Think")],
            agents,
            logger=None,
        )
        assert len(msgs) == 3


# ---- Termination keyword -----------------------------------------------------


class TestTermination:
    def test_task_complete_stops_processing(self):
        handler = IterativeFeedbackHandler(_cfg())
        agents = {
            "a1": MockAgent("TASK_COMPLETE done"),
            "a2": MockAgent("should not run"),
        }
        assignments = [
            Assignment(agent_name="a1", task="T1"),
            Assignment(agent_name="a2", task="T2"),
        ]
        msgs = handler.execute(assignments, agents, logger=None)
        assert len(msgs) == 1
        assert "TASK_COMPLETE" in msgs[0].content


# ---- Human feedback modes ----------------------------------------------------


class TestHumanFeedback:
    def test_none_mode_no_callback(self):
        handler = IterativeFeedbackHandler(_cfg(human_feedback_mode="none"))
        agent = ErrorThenSuccessAgent(fail_count=1)
        agents = {"a1": agent}
        msgs = handler.execute(
            [Assignment(agent_name="a1", task="T")],
            agents,
            logger=None,
        )
        # Should retry without blocking.
        assert len(msgs) == 2

    def test_real_time_callback_invoked(self):
        feedback_calls = []

        def mock_callback(fb: AttemptFeedback) -> str:
            feedback_calls.append(fb)
            return "try something else"

        handler = IterativeFeedbackHandler(
            _cfg(human_feedback_mode="real_time"),
            human_feedback_callback=mock_callback,
        )
        agent = ErrorThenSuccessAgent(fail_count=1)
        agents = {"a1": agent}
        handler.execute(
            [Assignment(agent_name="a1", task="T")],
            agents,
            logger=None,
        )
        assert len(feedback_calls) == 1
        # Human feedback should appear in retry context.
        assert "try something else" in agent.calls[1]

    def test_real_time_skip_keyword(self):
        def skip_callback(fb: AttemptFeedback) -> str:
            return "SKIP"

        handler = IterativeFeedbackHandler(
            _cfg(human_feedback_mode="real_time", max_retries=10),
            human_feedback_callback=skip_callback,
        )
        agent = ErrorThenSuccessAgent(fail_count=100)
        agents = {"a1": agent}
        msgs = handler.execute(
            [Assignment(agent_name="a1", task="T")],
            agents,
            logger=None,
        )
        # Should stop after first failure + SKIP.
        assert len(msgs) == 1

    def test_between_prompt_guidance_injected(self):
        handler = IterativeFeedbackHandler(
            _cfg(
                human_feedback_mode="between_prompt",
                human_guidance="Use simple geometry only",
            )
        )
        agent = MockAgent("done")
        agents = {"a1": agent}
        handler.execute(
            [Assignment(agent_name="a1", task="T")],
            agents,
            logger=None,
        )
        assert "Use simple geometry only" in agent.calls[0]


# ---- Feedback window ---------------------------------------------------------


class TestFeedbackWindow:
    def test_only_recent_feedback_in_context(self):
        handler = IterativeFeedbackHandler(
            _cfg(
                max_retries=6,
                feedback_window=2,
            )
        )
        agent = ErrorThenSuccessAgent(fail_count=4)
        agents = {"a1": agent}
        handler.execute(
            [Assignment(agent_name="a1", task="T")],
            agents,
            logger=None,
        )
        # On the 5th call (attempt 4, 0-indexed), context should contain
        # feedback from at most 2 previous attempts.
        last_call = agent.calls[-1]
        # Should have "attempt" mentions but limited by window.
        assert "attempt" in last_call.lower()


# ---- Turn numbering ----------------------------------------------------------


class TestTurnNumbering:
    def test_turn_numbers_increment(self):
        handler = IterativeFeedbackHandler(_cfg())
        agents = {
            "a1": MockAgent("r1"),
            "a2": MockAgent("r2"),
        }
        assignments = [
            Assignment(agent_name="a1", task="T1"),
            Assignment(agent_name="a2", task="T2"),
        ]
        msgs = handler.execute(assignments, agents, logger=None)
        assert msgs[0].turn_number == 1
        assert msgs[1].turn_number == 2

    def test_turn_offset(self):
        handler = IterativeFeedbackHandler(_cfg())
        agents = {"a1": MockAgent("r")}
        msgs = handler.execute(
            [Assignment(agent_name="a1", task="T")],
            agents,
            logger=None,
            turn_offset=10,
        )
        assert msgs[0].turn_number == 11

    def test_retry_turns_counted(self):
        handler = IterativeFeedbackHandler(_cfg(max_retries=3))
        agent = ErrorThenSuccessAgent(fail_count=2)
        agents = {"a1": agent}
        msgs = handler.execute(
            [Assignment(agent_name="a1", task="T")],
            agents,
            logger=None,
        )
        # 3 messages, turns 1, 2, 3
        assert [m.turn_number for m in msgs] == [1, 2, 3]
