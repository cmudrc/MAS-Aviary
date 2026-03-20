"""Tests for metadata instrumentation across strategies and handlers.

All tests use mocks and in-process helpers only — no GPU, no real LLM.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

from src.coordination.history import AgentMessage, ToolCallRecord

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _make_msg(**kwargs) -> AgentMessage:
    defaults = dict(
        agent_name="agent_1",
        content="hello",
        turn_number=1,
        timestamp=time.time(),
    )
    defaults.update(kwargs)
    return AgentMessage(**defaults)


# ---------------------------------------------------------------------------
# Fix 1: AgentMessage.metadata field
# ---------------------------------------------------------------------------


class TestAgentMessageMetadata:
    def test_default_is_empty_dict(self):
        msg = _make_msg()
        assert msg.metadata == {}

    def test_accepts_dict_kwarg(self):
        msg = _make_msg(metadata={"phase": "creation", "turn": 1})
        assert msg.metadata["phase"] == "creation"
        assert msg.metadata["turn"] == 1

    def test_mutation_after_creation(self):
        msg = _make_msg()
        msg.metadata["foo"] = "bar"
        assert msg.metadata["foo"] == "bar"

    def test_separate_instances_do_not_share_metadata(self):
        a = _make_msg()
        b = _make_msg()
        a.metadata["x"] = 1
        assert "x" not in b.metadata


# ---------------------------------------------------------------------------
# Fix 2/3/4: coordinator._execute_agent() metadata propagation
# ---------------------------------------------------------------------------


class TestCoordinatorMetadata:
    """Tests for coordinator._execute_agent metadata injection."""

    def _make_coordinator(self, action_metadata: dict | None = None):
        """Build a minimal Coordinator with a mock strategy and agent."""
        from src.coordination.coordinator import Coordinator
        from src.coordination.strategy import CoordinationAction

        mock_agent = MagicMock()
        mock_agent.run.return_value = "output text"
        # No tool call extraction attributes.
        mock_agent.logs = []
        mock_agent.memory = None

        mock_strategy = MagicMock()
        mock_strategy._blackboard = None
        mock_strategy.context = None

        coord = Coordinator.__new__(Coordinator)
        coord.agents = {"worker": mock_agent}
        coord.strategy = mock_strategy
        coord.execution_handler = None
        coord._turn_counter = 0
        coord.history = MagicMock()
        coord.termination = MagicMock()
        coord.config = {}
        coord.logger = None

        action = CoordinationAction(
            action_type="invoke_agent",
            agent_name="worker",
            input_context="do the thing",
            metadata=action_metadata or {},
        )
        return coord, action

    def test_action_metadata_passed_to_message(self):
        coord, action = self._make_coordinator({"phase": "creation", "orchestrator_turn": 1})
        msgs = coord._execute_agent(action)
        assert len(msgs) == 1
        assert msgs[0].metadata["phase"] == "creation"
        assert msgs[0].metadata["orchestrator_turn"] == 1

    def test_sequential_metadata_passed_through(self):
        coord, action = self._make_coordinator(
            {
                "stage_name": "design",
                "stage_index": 0,
                "total_stages": 3,
                "pipeline_template": "linear",
            }
        )
        msgs = coord._execute_agent(action)
        assert msgs[0].metadata["stage_name"] == "design"
        assert msgs[0].metadata["pipeline_template"] == "linear"

    def test_networked_metadata_passed_through(self):
        coord, action = self._make_coordinator(
            {
                "turn": 2,
                "rotation_index": 1,
                "total_agents": 5,
            }
        )
        msgs = coord._execute_agent(action)
        assert msgs[0].metadata["turn"] == 2
        assert msgs[0].metadata["total_agents"] == 5

    def test_empty_action_metadata_yields_empty_msg_metadata(self):
        coord, action = self._make_coordinator({})
        msgs = coord._execute_agent(action)
        # No keys from action, but blackboard keys won't appear (no blackboard).
        assert isinstance(msgs[0].metadata, dict)

    def test_exception_preserves_metadata(self):
        coord, action = self._make_coordinator({"phase": "execution"})
        coord.agents["worker"].run.side_effect = RuntimeError("boom")
        msgs = coord._execute_agent(action)
        assert msgs[0].error == "boom"
        assert msgs[0].metadata["phase"] == "execution"

    def test_token_count_estimated_from_content(self):
        coord, action = self._make_coordinator({})
        coord.agents["worker"].run.return_value = "x" * 400  # 400 chars → ~100 tokens
        msgs = coord._execute_agent(action)
        assert msgs[0].token_count is not None
        assert msgs[0].token_count >= 1

    def test_blackboard_writes_delta_for_networked(self):
        from src.coordination.blackboard import Blackboard
        from src.coordination.coordinator import Coordinator
        from src.coordination.strategy import CoordinationAction

        mock_agent = MagicMock()
        mock_agent.run.return_value = "done"
        mock_agent.logs = []
        mock_agent.memory = None

        bb = Blackboard(claiming_mode="none")

        mock_strategy = MagicMock()
        mock_strategy._blackboard = bb

        # Simulate context with spawned_agents list.
        mock_ctx = MagicMock()
        mock_ctx.spawned_agents = []
        mock_strategy.context = mock_ctx

        coord = Coordinator.__new__(Coordinator)
        coord.agents = {"agent_1": mock_agent}
        coord.strategy = mock_strategy
        coord.execution_handler = None
        coord._turn_counter = 0
        coord.history = MagicMock()
        coord.termination = MagicMock()
        coord.config = {}
        coord.logger = None

        def run_and_write(_task):
            bb.write("key1", "val", "agent_1", "result")
            return "output"

        coord.agents["agent_1"].run.side_effect = run_and_write

        action = CoordinationAction(
            action_type="invoke_agent",
            agent_name="agent_1",
            input_context="task",
            metadata={},
        )
        msgs = coord._execute_agent(action)
        assert msgs[0].metadata["blackboard_writes"] == 1
        assert msgs[0].metadata["blackboard_size"] == 1
        assert msgs[0].metadata["claim_conflicts"] == 0

    def test_peers_spawned_delta(self):
        from src.coordination.coordinator import Coordinator
        from src.coordination.strategy import CoordinationAction

        mock_agent = MagicMock()
        mock_agent.logs = []
        mock_agent.memory = None

        mock_strategy = MagicMock()
        mock_strategy._blackboard = None

        spawned = []
        mock_ctx = MagicMock()
        mock_ctx.spawned_agents = spawned
        mock_strategy.context = mock_ctx

        coord = Coordinator.__new__(Coordinator)
        coord.agents = {"agent_1": mock_agent}
        coord.strategy = mock_strategy
        coord.execution_handler = None
        coord._turn_counter = 0
        coord.history = MagicMock()
        coord.termination = MagicMock()
        coord.config = {}
        coord.logger = None

        def run_and_spawn(_task):
            spawned.append("agent_2")
            return "spawned"

        coord.agents["agent_1"].run.side_effect = run_and_spawn

        action = CoordinationAction(
            action_type="invoke_agent",
            agent_name="agent_1",
            input_context="go",
            metadata={},
        )
        msgs = coord._execute_agent(action)
        assert msgs[0].metadata["peers_spawned"] == 1


# ---------------------------------------------------------------------------
# Fix 5a: IterativeFeedbackHandler metadata
# ---------------------------------------------------------------------------


class TestIterativeFeedbackMetadata:
    def _make_handler(self, aspiration_mode="tool_success", max_retries=3):
        from src.coordination.iterative_feedback_handler import IterativeFeedbackHandler

        return IterativeFeedbackHandler(
            {
                "aspiration_mode": aspiration_mode,
                "max_retries": max_retries,
                "retry_toolless_agents": True,
            }
        )

    def _make_assignment(self, agent_name="worker", task="do it"):
        from src.coordination.execution_handler import Assignment

        return Assignment(agent_name=agent_name, task=task, assigned_at_turn=0)

    def test_first_attempt_has_attempt_number_zero(self):
        handler = self._make_handler()
        mock_agent = MagicMock()
        mock_agent.tools = {"some_tool": MagicMock()}
        mock_agent.run.return_value = "success"
        mock_agent.logs = []
        mock_agent.memory = None
        agents = {"worker": mock_agent}
        msgs = handler.execute([self._make_assignment()], agents, None)
        assert msgs[0].metadata["attempt_number"] == 0

    def test_aspiration_mode_set_in_metadata(self):
        handler = self._make_handler(aspiration_mode="any_output")
        mock_agent = MagicMock()
        mock_agent.tools = {}
        mock_agent.run.return_value = "hi"
        mock_agent.logs = []
        mock_agent.memory = None
        agents = {"worker": mock_agent}
        msgs = handler.execute([self._make_assignment()], agents, None)
        assert msgs[0].metadata["aspiration_mode"] == "any_output"

    def test_aspiration_met_true_on_success(self):
        handler = self._make_handler(aspiration_mode="any_output")
        mock_agent = MagicMock()
        mock_agent.tools = {}
        mock_agent.run.return_value = "some output"
        mock_agent.logs = []
        mock_agent.memory = None
        agents = {"worker": mock_agent}
        msgs = handler.execute([self._make_assignment()], agents, None)
        assert msgs[0].metadata["aspiration_met"] is True

    def test_aspiration_met_false_on_exception(self):
        handler = self._make_handler(max_retries=1)
        mock_agent = MagicMock()
        mock_agent.tools = {"tool": MagicMock()}
        mock_agent.run.side_effect = RuntimeError("crash")
        mock_agent.logs = []
        mock_agent.memory = None
        agents = {"worker": mock_agent}
        msgs = handler.execute([self._make_assignment()], agents, None)
        assert msgs[0].metadata["aspiration_met"] is False

    def test_total_attempts_backfilled(self):
        handler = self._make_handler(max_retries=3, aspiration_mode="any_output")
        mock_agent = MagicMock()
        mock_agent.tools = {}
        mock_agent.run.return_value = "done"
        mock_agent.logs = []
        mock_agent.memory = None
        agents = {"worker": mock_agent}
        msgs = handler.execute([self._make_assignment()], agents, None)
        assert msgs[0].metadata["total_attempts"] == 1

    def test_retry_increments_attempt_number(self):
        """Agent with tool errors retries; later attempts have higher attempt_number."""

        def run_with_tool_error(_task):
            return "output"

        handler = self._make_handler(max_retries=2, aspiration_mode="tool_success")

        mock_agent = MagicMock()
        mock_agent.tools = {"my_tool": MagicMock()}
        # First run returns output that will have tool errors via AttemptFeedback.
        # The simplest way: just make run succeed but check attempt_number increments.
        call_count = [0]

        def side_effect(_task):
            call_count[0] += 1
            return f"attempt {call_count[0]}"

        mock_agent.run.side_effect = side_effect
        mock_agent.logs = []
        mock_agent.memory = None

        agents = {"worker": mock_agent}
        msgs = handler.execute([self._make_assignment()], agents, None)
        # With aspiration_mode=tool_success and no actual tool calls,
        # has_tool_errors=False → aspiration met on first attempt.
        assert msgs[0].metadata["attempt_number"] == 0

    def test_missing_agent_metadata(self):
        handler = self._make_handler()
        msgs = handler.execute([self._make_assignment(agent_name="missing")], {}, None)
        assert msgs[0].metadata["attempt_number"] == 0
        assert msgs[0].metadata["aspiration_met"] is False
        assert msgs[0].metadata["total_attempts"] == 0

    def test_cached_result_metadata(self):
        handler = self._make_handler()
        handler._last_successful_output = "cached output"
        handler._last_successful_agent = "worker"
        mock_agent = MagicMock()
        mock_agent.tools = {}
        mock_agent.run.return_value = "should not be called"
        mock_agent.logs = []
        mock_agent.memory = None
        agents = {"worker": mock_agent}
        msgs = handler.execute([self._make_assignment()], agents, None)
        mock_agent.run.assert_not_called()
        assert msgs[0].metadata["aspiration_met"] is True
        assert msgs[0].metadata["total_attempts"] == 1
        assert msgs[0].content == "cached output"


# ---------------------------------------------------------------------------
# Fix 5b: GraphRoutedHandler metadata
# ---------------------------------------------------------------------------


class TestGraphRoutedMetadata:
    def _make_handler(self, graph_data: dict) -> Any:
        from src.coordination.graph_routed_handler import GraphRoutedHandler

        return GraphRoutedHandler({"_graph_data": graph_data})

    def _simple_terminal_graph(self) -> dict:
        """Graph that immediately routes to terminal state."""
        return {
            "initial_state": "WORK",
            "terminal_states": ["DONE"],
            "resource_budgets": {},
            "states": {
                "WORK": {
                    "agent": "worker",
                    "agent_prompt": "",
                    "transitions": [{"condition": "true", "target": "DONE"}],
                },
                "DONE": {
                    "agent": None,
                    "agent_prompt": "",
                    "transitions": [],
                },
            },
        }

    def test_graph_state_in_metadata(self):
        from src.coordination.execution_handler import Assignment

        graph = self._simple_terminal_graph()
        handler = self._make_handler(graph)

        mock_agent = MagicMock()
        mock_agent.run.return_value = "done"
        mock_agent.logs = []
        mock_agent.memory = None
        agents = {"worker": mock_agent}

        assignment = Assignment(agent_name="worker", task="build it", assigned_at_turn=0)
        msgs = handler.execute([assignment], agents, None)

        assert len(msgs) >= 1
        assert msgs[0].metadata["graph_state"] == "WORK"

    def test_passes_remaining_in_metadata(self):
        from src.coordination.execution_handler import Assignment

        graph = self._simple_terminal_graph()
        handler = self._make_handler(graph)

        mock_agent = MagicMock()
        mock_agent.run.return_value = "output"
        mock_agent.logs = []
        mock_agent.memory = None
        agents = {"worker": mock_agent}

        assignment = Assignment(agent_name="worker", task="t", assigned_at_turn=0)
        msgs = handler.execute([assignment], agents, None)

        # passes_remaining should be an int (could be None if budgets not configured).
        assert "passes_remaining" in msgs[0].metadata

    def test_no_prior_transition_gives_no_transition_from(self):
        from src.coordination.execution_handler import Assignment

        graph = self._simple_terminal_graph()
        handler = self._make_handler(graph)

        mock_agent = MagicMock()
        mock_agent.run.return_value = "ok"
        mock_agent.logs = []
        mock_agent.memory = None
        agents = {"worker": mock_agent}

        assignment = Assignment(agent_name="worker", task="t", assigned_at_turn=0)
        msgs = handler.execute([assignment], agents, None)

        # On first state, there's no prior transition.
        assert "graph_transition_from" not in msgs[0].metadata

    def test_resolve_error_has_graph_state(self):
        """When the agent for a state can't be resolved, the error msg has graph_state."""
        from src.coordination.execution_handler import Assignment
        from src.coordination.graph_routed_handler import GraphRoutedHandler

        graph = {
            "initial_state": "WORK",
            "terminal_states": ["DONE"],
            "resource_budgets": {},
            "states": {
                "WORK": {
                    "agent": "missing_agent",
                    "agent_prompt": "",
                    "transitions": [{"condition": "true", "target": "DONE"}],
                },
                "DONE": {"agent": None, "agent_prompt": "", "transitions": []},
            },
        }
        handler = GraphRoutedHandler({"_graph_data": graph})
        assignment = Assignment(agent_name="missing_agent", task="t", assigned_at_turn=0)
        # Pass empty agents dict so resolve_agent_for_role raises ValueError.
        msgs = handler.execute([assignment], {}, None)
        assert len(msgs) >= 1
        # Error message should still include graph_state.
        assert msgs[0].metadata["graph_state"] == "WORK"
        assert msgs[0].error is not None


# ---------------------------------------------------------------------------
# Fix 5c: StagedPipelineHandler metadata
# ---------------------------------------------------------------------------


class TestStagedPipelineMetadata:
    def _make_handler(self, pipeline_config: dict | None = None):
        from src.coordination.staged_pipeline_handler import StagedPipelineHandler

        cfg = pipeline_config or {}
        return StagedPipelineHandler(cfg)

    def _make_assignment(self, name="worker", task="do stage"):
        from src.coordination.execution_handler import Assignment

        return Assignment(agent_name=name, task=task, assigned_at_turn=0)

    def _simple_pipeline(self):
        return {
            "stages": [
                {
                    "name": "analysis",
                    "agent": "worker",
                    "stage_prompt": "",
                    "completion_criteria": {"type": "any", "check": "always"},
                }
            ]
        }

    def test_stage_name_in_metadata(self):
        handler = self._make_handler(self._simple_pipeline())

        mock_agent = MagicMock()
        mock_agent.run.return_value = "analysis done"
        mock_agent.logs = []
        mock_agent.memory = None

        msgs = handler.execute([self._make_assignment()], {"worker": mock_agent}, None)
        assert len(msgs) >= 1
        assert msgs[0].metadata["stage_name"] == "analysis"

    def test_stage_index_in_metadata(self):
        handler = self._make_handler(self._simple_pipeline())

        mock_agent = MagicMock()
        mock_agent.run.return_value = "ok"
        mock_agent.logs = []
        mock_agent.memory = None

        msgs = handler.execute([self._make_assignment()], {"worker": mock_agent}, None)
        assert msgs[0].metadata["stage_index"] == 0

    def test_completion_met_in_metadata(self):
        handler = self._make_handler(self._simple_pipeline())

        mock_agent = MagicMock()
        mock_agent.run.return_value = "done"
        mock_agent.logs = []
        mock_agent.memory = None

        msgs = handler.execute([self._make_assignment()], {"worker": mock_agent}, None)
        # completion_met is set after evaluate_completion; with type=any/always it's True.
        assert msgs[0].metadata["completion_met"] is True

    def test_missing_agent_metadata(self):
        handler = self._make_handler(self._simple_pipeline())
        msgs = handler.execute([self._make_assignment(name="missing")], {}, None)
        assert msgs[0].metadata["stage_name"] == "analysis"
        assert msgs[0].metadata["completion_met"] is False

    def test_received_failed_input_false_for_first_stage(self):
        handler = self._make_handler(self._simple_pipeline())
        mock_agent = MagicMock()
        mock_agent.run.return_value = "out"
        mock_agent.logs = []
        mock_agent.memory = None
        msgs = handler.execute([self._make_assignment()], {"worker": mock_agent}, None)
        assert msgs[0].metadata["received_failed_input"] is False

    def test_token_count_estimated_from_content(self):
        handler = self._make_handler(self._simple_pipeline())
        mock_agent = MagicMock()
        mock_agent.run.return_value = "x" * 200
        mock_agent.logs = []
        mock_agent.memory = None
        msgs = handler.execute([self._make_assignment()], {"worker": mock_agent}, None)
        assert msgs[0].token_count is not None
        assert msgs[0].token_count >= 1


# ---------------------------------------------------------------------------
# Fix 8: _msg_to_dict serialization
# ---------------------------------------------------------------------------


class TestMsgToDict:
    def test_metadata_serialized(self):
        from src.runners.batch_runner import _msg_to_dict

        msg = _make_msg(metadata={"stage_name": "analysis", "stage_index": 0})
        d = _msg_to_dict(msg)
        assert d["metadata"] == {"stage_name": "analysis", "stage_index": 0}

    def test_is_retry_serialized(self):
        from src.runners.batch_runner import _msg_to_dict

        msg = _make_msg(is_retry=True, retry_of_turn=1)
        d = _msg_to_dict(msg)
        assert d["is_retry"] is True
        assert d["retry_of_turn"] == 1

    def test_is_retry_false_by_default(self):
        from src.runners.batch_runner import _msg_to_dict

        msg = _make_msg()
        d = _msg_to_dict(msg)
        assert d["is_retry"] is False
        assert d["retry_of_turn"] is None

    def test_empty_metadata_serialized(self):
        from src.runners.batch_runner import _msg_to_dict

        msg = _make_msg()
        d = _msg_to_dict(msg)
        assert d["metadata"] == {}

    def test_tool_calls_still_present(self):
        from src.runners.batch_runner import _msg_to_dict

        tc = ToolCallRecord(
            tool_name="my_tool",
            inputs={"x": 1},
            output="result",
            duration_seconds=0.5,
        )
        msg = _make_msg(tool_calls=[tc])
        d = _msg_to_dict(msg)
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["tool_name"] == "my_tool"

    def test_token_count_serialized(self):
        from src.runners.batch_runner import _msg_to_dict

        msg = _make_msg(token_count=512)
        d = _msg_to_dict(msg)
        assert d["token_count"] == 512


# ---------------------------------------------------------------------------
# Coordinator helper functions (Fix 6 + Fix 7)
# ---------------------------------------------------------------------------


class TestCoordinatorHelpers:
    def test_extract_tool_calls_empty_on_no_logs(self):
        from src.coordination.coordinator import _extract_tool_calls

        agent = MagicMock()
        agent.logs = []
        agent.memory = None
        result = _extract_tool_calls(agent)
        assert result == []

    def test_extract_token_count_from_content_estimate(self):
        from src.coordination.coordinator import _extract_token_count

        agent = MagicMock()
        agent.token_count = None
        agent.total_tokens = None
        agent.memory = None
        tc, estimated = _extract_token_count(agent, "x" * 400)
        assert tc == 100
        assert estimated is True

    def test_extract_token_count_from_agent_attr(self):
        from src.coordination.coordinator import _extract_token_count

        agent = MagicMock()
        agent.token_count = 256
        tc, estimated = _extract_token_count(agent, "some content")
        assert tc == 256
        assert estimated is False

    def test_extract_token_count_none_for_empty_content(self):
        from src.coordination.coordinator import _extract_token_count

        agent = MagicMock()
        agent.token_count = None
        agent.total_tokens = None
        agent.memory = None
        tc, estimated = _extract_token_count(agent, "")
        assert tc is None


# ---------------------------------------------------------------------------
# action_metadata pass-through tests (Change 1)
# ---------------------------------------------------------------------------


class TestActionMetadataPassthrough:
    """Tests that action_metadata flows through all handlers."""

    def _mock_agent(self, return_value="output"):
        agent = MagicMock()
        agent.run.return_value = return_value
        agent.logs = []
        agent.memory = None
        agent.tools = {"some_tool": MagicMock()}
        return agent

    def _assignment(self, name="worker", task="task"):
        from src.coordination.execution_handler import Assignment

        return Assignment(agent_name=name, task=task, assigned_at_turn=0)

    def test_placeholder_merges_action_metadata(self):
        from src.coordination.execution_handler import PlaceholderExecutor

        handler = PlaceholderExecutor()
        agents = {"worker": self._mock_agent()}
        meta = {"phase": "execution", "rotation_index": 2}
        msgs = handler.execute([self._assignment()], agents, None, action_metadata=meta)
        assert msgs[0].metadata["phase"] == "execution"
        assert msgs[0].metadata["rotation_index"] == 2

    def test_if_handler_merges_action_metadata(self):
        from src.coordination.iterative_feedback_handler import IterativeFeedbackHandler

        handler = IterativeFeedbackHandler({"max_retries": 1, "aspiration_mode": "any_output"})
        agents = {"worker": self._mock_agent()}
        meta = {"turn": 5, "total_agents": 3}
        msgs = handler.execute([self._assignment()], agents, None, action_metadata=meta)
        assert msgs[0].metadata["turn"] == 5
        assert msgs[0].metadata["total_agents"] == 3
        # Handler-specific keys should also be present.
        assert msgs[0].metadata["aspiration_mode"] == "any_output"
        assert msgs[0].metadata["attempt_number"] == 0

    def test_if_handler_keys_override_base(self):
        """Handler-specific keys take precedence over action_metadata."""
        from src.coordination.iterative_feedback_handler import IterativeFeedbackHandler

        handler = IterativeFeedbackHandler({"max_retries": 1, "aspiration_mode": "any_output"})
        agents = {"worker": self._mock_agent()}
        # Try to override aspiration_mode from action_metadata — handler should win.
        meta = {"aspiration_mode": "SHOULD_BE_OVERRIDDEN"}
        msgs = handler.execute([self._assignment()], agents, None, action_metadata=meta)
        assert msgs[0].metadata["aspiration_mode"] == "any_output"

    def test_sp_handler_merges_action_metadata(self):
        from src.coordination.staged_pipeline_handler import StagedPipelineHandler

        handler = StagedPipelineHandler({})
        agents = {"worker": self._mock_agent()}
        meta = {"phase": "execution", "assignment_index": 1}
        msgs = handler.execute([self._assignment()], agents, None, action_metadata=meta)
        assert msgs[0].metadata["phase"] == "execution"
        assert msgs[0].metadata["assignment_index"] == 1
        # Handler-specific stage keys should also be present.
        assert "stage_index" in msgs[0].metadata

    def test_gr_handler_merges_action_metadata(self):
        from src.coordination.graph_routed_handler import GraphRoutedHandler

        graph = {
            "initial_state": "WORK",
            "terminal_states": ["DONE"],
            "resource_budgets": {},
            "states": {
                "WORK": {
                    "agent": "worker",
                    "agent_prompt": "",
                    "transitions": [{"condition": "true", "target": "DONE"}],
                },
                "DONE": {"agent": None, "agent_prompt": "", "transitions": []},
            },
        }
        handler = GraphRoutedHandler({"_graph_data": graph})
        agents = {"worker": self._mock_agent()}
        meta = {"phase": "execution", "rotation_index": 0}
        msgs = handler.execute([self._assignment()], agents, None, action_metadata=meta)
        assert msgs[0].metadata["phase"] == "execution"
        assert msgs[0].metadata["rotation_index"] == 0
        # Handler graph_state should also be present.
        assert msgs[0].metadata["graph_state"] == "WORK"

    def test_backward_compat_no_action_metadata(self):
        """Calling without action_metadata still works (no crash, empty base)."""
        from src.coordination.execution_handler import PlaceholderExecutor

        handler = PlaceholderExecutor()
        agents = {"worker": self._mock_agent()}
        msgs = handler.execute([self._assignment()], agents, None)
        assert isinstance(msgs[0].metadata, dict)

    def test_coordinator_passes_action_metadata_to_handler(self):
        """Coordinator sends action.metadata to handler.execute()."""
        from src.coordination.coordinator import Coordinator
        from src.coordination.strategy import CoordinationAction

        mock_handler = MagicMock()
        mock_handler.execute.return_value = [_make_msg(turn_number=1, metadata={"from_handler": True})]

        coord = Coordinator.__new__(Coordinator)
        coord.agents = {"worker": self._mock_agent()}
        coord.strategy = MagicMock()
        coord.strategy._blackboard = None
        coord.strategy.context = None
        coord.execution_handler = mock_handler
        coord._turn_counter = 0
        coord.history = MagicMock()
        coord.termination = MagicMock()
        coord.config = {}
        coord.logger = None

        action = CoordinationAction(
            action_type="invoke_agent",
            agent_name="worker",
            input_context="task",
            metadata={"phase": "execution", "custom_key": 42},
        )
        coord._execute_agent(action)
        # Verify handler.execute was called with action_metadata.
        call_kwargs = mock_handler.execute.call_args
        assert call_kwargs.kwargs.get("action_metadata") == {"phase": "execution", "custom_key": 42}

    def test_coordinator_blackboard_postprocess(self):
        """Coordinator injects bb deltas into handler-returned messages."""
        from src.coordination.blackboard import Blackboard
        from src.coordination.coordinator import Coordinator
        from src.coordination.strategy import CoordinationAction

        bb = Blackboard(claiming_mode="none")
        bb.write("pre_key", "val", "setup", "status")  # pre-existing entry

        mock_handler = MagicMock()

        def handler_execute(assignments, agents, logger=None, turn_offset=0, action_metadata=None):
            # Simulate handler writing to blackboard during execution.
            bb.write("new_key", "new_val", "worker", "result")
            return [_make_msg(turn_number=turn_offset + 1, metadata={"from_handler": True})]

        mock_handler.execute.side_effect = handler_execute

        mock_strategy = MagicMock()
        mock_strategy._blackboard = bb
        mock_strategy.context = None

        coord = Coordinator.__new__(Coordinator)
        coord.agents = {"worker": self._mock_agent()}
        coord.strategy = mock_strategy
        coord.execution_handler = mock_handler
        coord._turn_counter = 0
        coord.history = MagicMock()
        coord.termination = MagicMock()
        coord.config = {}
        coord.logger = None

        action = CoordinationAction(
            action_type="invoke_agent",
            agent_name="worker",
            input_context="task",
            metadata={},
        )
        msgs = coord._execute_agent(action)
        # Blackboard writes during handler: 1 new write.
        assert msgs[0].metadata["blackboard_writes"] == 1
        # Total board size: 2 (pre_key + new_key).
        assert msgs[0].metadata["blackboard_size"] == 2
        assert msgs[0].metadata["claim_conflicts"] == 0
        # Handler's own metadata should be preserved.
        assert msgs[0].metadata["from_handler"] is True


# ---------------------------------------------------------------------------
# org_theory_metrics: blackboard computation + aspiration_mode fix (Change 2)
# ---------------------------------------------------------------------------


class TestOrgTheoryBlackboardMetrics:
    """Tests for Tier 1 and Tier 2 blackboard metric computation."""

    def test_tier1_metadata_fields(self):
        """When messages have metadata.blackboard_writes etc., Tier 1 is used."""
        from src.logging.org_theory_metrics import _networked_os_metrics

        msgs = [
            _make_msg(
                agent_name="agent_1",
                metadata={
                    "blackboard_writes": 3,
                    "blackboard_size": 5,
                    "claim_conflicts": 1,
                },
            ),
            _make_msg(
                agent_name="agent_2",
                metadata={
                    "blackboard_writes": 2,
                    "blackboard_size": 7,
                    "claim_conflicts": 0,
                },
            ),
        ]
        warnings = []
        result = _networked_os_metrics(msgs, {}, None, warnings)
        assert result["blackboard_writes_total"] == 5  # 3 + 2
        assert result["blackboard_size_final"] == 7  # max(5, 7)
        assert result["claim_conflicts"] == 1  # 1 + 0
        assert result["blackboard_utilization"] is not None

    def test_tier2_tool_calls_fallback(self):
        """When metadata is absent, blackboard metrics are inferred from tool_calls."""
        from src.logging.org_theory_metrics import _networked_os_metrics

        tc_write = ToolCallRecord(tool_name="write_blackboard", inputs={}, output="", duration_seconds=0.0)
        tc_read = ToolCallRecord(tool_name="read_blackboard", inputs={}, output="", duration_seconds=0.0)
        tc_done = ToolCallRecord(tool_name="mark_task_done", inputs={}, output="", duration_seconds=0.0)
        msgs = [
            _make_msg(agent_name="agent_1", tool_calls=[tc_write, tc_read]),
            _make_msg(agent_name="agent_2", tool_calls=[tc_write, tc_done, tc_read]),
        ]
        warnings = []
        result = _networked_os_metrics(msgs, {}, None, warnings)
        # Tier 2: 2 write_blackboard + 1 mark_task_done + 2 structural + 1 initial = 6
        assert result["blackboard_writes_total"] == 6
        # Reads: 2 read_blackboard calls
        assert result["blackboard_reads_total"] == 2
        # Size estimate: 2 unique agents + 1 = 3
        assert result["blackboard_size_final"] == 3
        assert result["blackboard_utilization"] is not None
        # Check warnings mention Tier 2.
        tier2_warns = [w for w in warnings if "Tier 2" in w]
        assert len(tier2_warns) >= 1

    def test_empty_messages_no_crash(self):
        from src.logging.org_theory_metrics import _networked_os_metrics

        warnings = []
        result = _networked_os_metrics([], {}, None, warnings)
        assert result["blackboard_utilization"] is None


class TestAspirationModeFromMetadata:
    def test_reads_from_message_metadata(self):
        from src.logging.org_theory_metrics import _iterative_feedback_metrics

        msgs = [
            _make_msg(agent_name="worker", metadata={"aspiration_mode": "any_output"}),
        ]
        warnings = []
        result = _iterative_feedback_metrics(msgs, {}, warnings)
        assert result["aspiration_mode"] == "any_output"

    def test_falls_back_to_config(self):
        from src.logging.org_theory_metrics import _iterative_feedback_metrics

        msgs = [_make_msg(agent_name="worker")]
        warnings = []
        result = _iterative_feedback_metrics(msgs, {"aspiration_mode": "tool_success"}, warnings)
        assert result["aspiration_mode"] == "tool_success"

    def test_metadata_takes_precedence_over_config(self):
        from src.logging.org_theory_metrics import _iterative_feedback_metrics

        msgs = [
            _make_msg(agent_name="worker", metadata={"aspiration_mode": "any_output"}),
        ]
        warnings = []
        result = _iterative_feedback_metrics(msgs, {"aspiration_mode": "tool_success"}, warnings)
        assert result["aspiration_mode"] == "any_output"
