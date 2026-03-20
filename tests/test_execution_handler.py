"""Tests for ExecutionHandler ABC and PlaceholderExecutor.

No GPU needed. Uses mock agents that return fixed strings.
"""

import pytest

from src.coordination.execution_handler import (
    Assignment,
    ExecutionHandler,
    PlaceholderExecutor,
)
from src.logging.logger import InstrumentationLogger

# ---- Mock Agent --------------------------------------------------------------


class MockAgent:
    """Minimal agent stub that returns a fixed response from .run()."""

    def __init__(self, name: str, response: str = "done"):
        self.name = name
        self._response = response
        self._calls = []

    def run(self, input_context: str):
        self._calls.append(input_context)
        return self._response


class ErrorAgent:
    """Agent stub that raises on .run()."""

    def __init__(self, name: str, error: str = "boom"):
        self.name = name
        self._error = error

    def run(self, input_context: str):
        raise RuntimeError(self._error)


# ---- PlaceholderExecutor tests -----------------------------------------------


class TestPlaceholderExecutor:
    def test_executes_agents_in_order(self):
        agents = {
            "a1": MockAgent("a1", "result_a"),
            "a2": MockAgent("a2", "result_b"),
        }
        assignments = [
            Assignment(agent_name="a1", task="Task A"),
            Assignment(agent_name="a2", task="Task B"),
        ]
        executor = PlaceholderExecutor()
        messages = executor.execute(assignments, agents, logger=None)
        assert len(messages) == 2
        assert messages[0].agent_name == "a1"
        assert messages[0].content == "result_a"
        assert messages[1].agent_name == "a2"
        assert messages[1].content == "result_b"

    def test_passes_output_as_context(self):
        a1 = MockAgent("a1", "first_output")
        a2 = MockAgent("a2", "second_output")
        agents = {"a1": a1, "a2": a2}
        assignments = [
            Assignment(agent_name="a1", task="Task A"),
            Assignment(agent_name="a2", task="Task B"),
        ]
        executor = PlaceholderExecutor()
        executor.execute(assignments, agents, logger=None)
        # Second agent should receive context from first
        assert "first_output" in a2._calls[0]
        assert "Task B" in a2._calls[0]

    def test_first_agent_gets_task_only(self):
        a1 = MockAgent("a1", "result")
        agents = {"a1": a1}
        assignments = [Assignment(agent_name="a1", task="Just the task")]
        executor = PlaceholderExecutor()
        executor.execute(assignments, agents, logger=None)
        # First agent should only get the task, no "Context from previous"
        assert a1._calls[0] == "Just the task"

    def test_logs_all_turns(self):
        agents = {
            "a1": MockAgent("a1", "r1"),
            "a2": MockAgent("a2", "r2"),
        }
        assignments = [
            Assignment(agent_name="a1", task="T1"),
            Assignment(agent_name="a2", task="T2"),
        ]
        logger = InstrumentationLogger()
        executor = PlaceholderExecutor()
        executor.execute(assignments, agents, logger=logger)
        assert logger.turn_count == 2

    def test_stops_on_task_complete(self):
        agents = {
            "a1": MockAgent("a1", "TASK_COMPLETE done"),
            "a2": MockAgent("a2", "should not run"),
        }
        assignments = [
            Assignment(agent_name="a1", task="T1"),
            Assignment(agent_name="a2", task="T2"),
        ]
        executor = PlaceholderExecutor()
        messages = executor.execute(assignments, agents, logger=None)
        assert len(messages) == 1
        assert "TASK_COMPLETE" in messages[0].content

    def test_stops_on_max_turns(self):
        agents = {f"a{i}": MockAgent(f"a{i}", f"r{i}") for i in range(10)}
        assignments = [Assignment(agent_name=f"a{i}", task=f"T{i}") for i in range(10)]
        executor = PlaceholderExecutor(max_turns=3)
        messages = executor.execute(assignments, agents, logger=None)
        assert len(messages) == 3

    def test_turn_offset(self):
        agents = {"a1": MockAgent("a1", "result")}
        assignments = [Assignment(agent_name="a1", task="Task")]
        executor = PlaceholderExecutor()
        messages = executor.execute(
            assignments,
            agents,
            logger=None,
            turn_offset=5,
        )
        assert messages[0].turn_number == 6

    def test_handles_missing_agent(self):
        agents = {}  # no agents registered
        assignments = [Assignment(agent_name="ghost", task="Task")]
        executor = PlaceholderExecutor()
        messages = executor.execute(assignments, agents, logger=None)
        assert len(messages) == 1
        assert messages[0].error is not None
        assert "not found" in messages[0].error

    def test_handles_agent_error(self):
        agents = {"a1": ErrorAgent("a1", "kaboom")}
        assignments = [Assignment(agent_name="a1", task="Task")]
        executor = PlaceholderExecutor()
        messages = executor.execute(assignments, agents, logger=None)
        assert len(messages) == 1
        assert messages[0].error is not None
        assert "kaboom" in messages[0].error

    def test_continues_after_error(self):
        agents = {
            "a1": ErrorAgent("a1", "fail"),
            "a2": MockAgent("a2", "success"),
        }
        assignments = [
            Assignment(agent_name="a1", task="T1"),
            Assignment(agent_name="a2", task="T2"),
        ]
        executor = PlaceholderExecutor()
        messages = executor.execute(assignments, agents, logger=None)
        assert len(messages) == 2
        assert messages[0].error is not None
        assert messages[1].error is None
        assert messages[1].content == "success"

    def test_duration_is_recorded(self):
        agents = {"a1": MockAgent("a1", "result")}
        assignments = [Assignment(agent_name="a1", task="Task")]
        executor = PlaceholderExecutor()
        messages = executor.execute(assignments, agents, logger=None)
        assert messages[0].duration_seconds >= 0

    def test_empty_assignments(self):
        executor = PlaceholderExecutor()
        messages = executor.execute([], {}, logger=None)
        assert messages == []


# ---- ExecutionHandler ABC tests ----------------------------------------------


class TestExecutionHandlerABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ExecutionHandler()

    def test_subclass_must_implement_execute(self):
        class Incomplete(ExecutionHandler):
            pass

        with pytest.raises(TypeError):
            Incomplete()
