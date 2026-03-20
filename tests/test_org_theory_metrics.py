"""Tests for src/logging/org_theory_metrics.py.

All tests run without GPU (no model loading). Similarity is tested via the
tfidf path (sklearn required) with fallback to jaccard if sklearn missing.
"""

from __future__ import annotations

import pytest

from src.coordination.history import AgentMessage, ToolCallRecord
from src.logging.org_theory_metrics import (
    _convergence_classification,
    _duplicate_work_rate,
    _get,
    _graph_routed_metrics,
    _iterative_feedback_metrics,
    _networked_os_metrics,
    _orchestrated_os_metrics,
    _sequential_os_metrics,
    _staged_pipeline_metrics,
    compute_org_theory_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(
    agent_name: str,
    content: str = "",
    turn_number: int = 1,
    timestamp: float = 0.0,
    duration_seconds: float = 1.0,
    error: str | None = None,
    is_retry: bool = False,
    retry_of_turn: int | None = None,
) -> AgentMessage:
    return AgentMessage(
        agent_name=agent_name,
        content=content,
        turn_number=turn_number,
        timestamp=timestamp,
        duration_seconds=duration_seconds,
        error=error,
        is_retry=is_retry,
        retry_of_turn=retry_of_turn,
    )


# ---------------------------------------------------------------------------
# _get helper
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_from_dataclass(self):
        m = _msg("alice", content="hello")
        assert _get(m, "agent_name") == "alice"
        assert _get(m, "content") == "hello"

    def test_get_from_dict(self):
        d = {"agent_name": "bob", "content": "world"}
        assert _get(d, "agent_name") == "bob"
        assert _get(d, "missing_key", "default") == "default"

    def test_get_default_on_missing_attr(self):
        m = _msg("x")
        assert _get(m, "nonexistent_field", 42) == 42

    def test_get_none_when_no_default(self):
        d = {}
        assert _get(d, "key") is None


# ---------------------------------------------------------------------------
# _convergence_classification
# ---------------------------------------------------------------------------


class TestConvergenceClassification:
    def test_high_convergence_failure(self):
        assert _convergence_classification(0.8, False) == "joint_myopia"

    def test_high_convergence_success(self):
        assert _convergence_classification(0.7, True) == "effective_consensus"

    def test_low_convergence_failure(self):
        assert _convergence_classification(0.3, False) == "incoherent_team"

    def test_low_convergence_success(self):
        assert _convergence_classification(0.2, True) == "healthy_diversity"

    def test_none_score_returns_none(self):
        assert _convergence_classification(None, True) is None

    def test_none_eval_returns_direction(self):
        result = _convergence_classification(0.7, None)
        assert result == "high_convergence"

    def test_threshold_boundary_high(self):
        # Exactly at 0.6 is considered high
        assert _convergence_classification(0.6, True) == "effective_consensus"

    def test_threshold_boundary_low(self):
        # Just below 0.6 is low
        assert _convergence_classification(0.59, True) == "healthy_diversity"


# ---------------------------------------------------------------------------
# _duplicate_work_rate
# ---------------------------------------------------------------------------


class TestDuplicateWorkRate:
    def test_identical_texts(self):
        texts = ["aircraft parameters", "aircraft parameters", "aircraft parameters"]
        rate = _duplicate_work_rate(texts)
        assert rate == pytest.approx(1.0, abs=0.01)

    def test_completely_different_texts(self):
        texts = ["alpha beta gamma", "xyz uvw qrs", "one two three four five"]
        rate = _duplicate_work_rate(texts)
        # Should be very low — these are distinct enough
        assert rate is not None
        assert 0.0 <= rate <= 1.0

    def test_single_text_returns_none(self):
        assert _duplicate_work_rate(["only one"]) is None

    def test_empty_list_returns_none(self):
        assert _duplicate_work_rate([]) is None

    def test_two_identical(self):
        rate = _duplicate_work_rate(["same text here", "same text here"])
        assert rate == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# _orchestrated_os_metrics
# ---------------------------------------------------------------------------


class TestOrchestratedOSMetrics:
    def _make_messages(self):
        return [
            _msg("orchestrator", "created agents", turn_number=1, duration_seconds=10.0, timestamp=100.0),
            _msg("cad_coder", "generated code", turn_number=2, duration_seconds=5.0, timestamp=110.0),
            _msg("evaluator", "metrics computed", turn_number=3, duration_seconds=4.0, timestamp=115.0),
            _msg("orchestrator", "review", turn_number=4, duration_seconds=2.0, timestamp=120.0),
        ]

    def test_agents_spawned(self):
        msgs = self._make_messages()
        w = []
        result = _orchestrated_os_metrics(msgs, {}, None, w)
        assert result["agents_spawned"] == 2  # cad_coder, evaluator

    def test_orchestrator_turns(self):
        msgs = self._make_messages()
        w = []
        result = _orchestrated_os_metrics(msgs, {}, None, w)
        assert result["orchestrator_turns"] == 2

    def test_worker_turns(self):
        msgs = self._make_messages()
        w = []
        result = _orchestrated_os_metrics(msgs, {}, None, w)
        assert result["worker_turns"] == 2

    def test_orchestrator_overhead_ratio(self):
        msgs = self._make_messages()
        w = []
        result = _orchestrated_os_metrics(msgs, {}, None, w)
        # 2 orch / 4 total = 0.5
        assert result["orchestrator_overhead_ratio"] == pytest.approx(0.5)

    def test_cost_per_agent(self):
        msgs = self._make_messages()
        w = []
        result = _orchestrated_os_metrics(msgs, {}, None, w)
        # orch time = 10 + 2 = 12; agents = 2 → 6.0
        assert result["cost_per_agent"] == pytest.approx(6.0)

    def test_reasoning_iterations(self):
        msgs = self._make_messages()
        w = []
        result = _orchestrated_os_metrics(msgs, {}, None, w)
        # Only 1 orchestrator turn before first worker
        assert result["reasoning_iterations"] == 1

    def test_null_fields_with_warnings(self):
        msgs = self._make_messages()
        w = []
        result = _orchestrated_os_metrics(msgs, {}, None, w)
        assert result["orchestrator_token_growth"] is None
        assert result["authority_transfers"] is None
        assert result["information_ratio"] is None
        assert len(w) >= 3

    def test_authority_holder(self):
        msgs = self._make_messages()
        w = []
        result = _orchestrated_os_metrics(msgs, {}, None, w)
        assert result["authority_holder"] == "orchestrator"

    def test_empty_messages(self):
        w = []
        result = _orchestrated_os_metrics([], {}, None, w)
        assert result["agents_spawned"] == 0
        assert result["orchestrator_overhead_ratio"] == 0.0

    def test_all_orchestrator_no_workers(self):
        msgs = [
            _msg("orchestrator", "a", turn_number=1),
            _msg("orchestrator", "b", turn_number=2),
        ]
        w = []
        result = _orchestrated_os_metrics(msgs, {}, None, w)
        assert result["agents_spawned"] == 0
        assert result["worker_turns"] == 0
        assert result["cost_per_agent"] is None


# ---------------------------------------------------------------------------
# _networked_os_metrics
# ---------------------------------------------------------------------------


class TestNetworkedOSMetrics:
    def _make_messages(self):
        return [
            _msg("agent_1", "I will write simulation parameters for aircraft", turn_number=1),
            _msg("agent_2", "I will write simulation parameters for aircraft too", turn_number=2),
            _msg("agent_3", "I will evaluate the generated STL file", turn_number=3),
            _msg("agent_4", "The task is complete, metrics are good", turn_number=4),
        ]

    def test_agents_spawned(self):
        msgs = self._make_messages()
        w = []
        result = _networked_os_metrics(msgs, {}, True, w)
        assert result["agents_spawned"] == 4

    def test_convergence_score_computed(self):
        msgs = self._make_messages()
        w = []
        result = _networked_os_metrics(msgs, {}, True, w)
        # Score should be between 0 and 1
        assert result["convergence_score"] is not None
        assert 0.0 <= result["convergence_score"] <= 1.0

    def test_convergence_classification_healthy_diversity(self):
        # Low similarity texts + success → healthy_diversity
        msgs = [
            _msg("agent_1", "alpha beta gamma delta epsilon zeta eta theta"),
            _msg("agent_2", "one two three four five six seven eight nine ten"),
        ]
        w = []
        result = _networked_os_metrics(msgs, {}, True, w)
        # Low convergence + success
        assert result["convergence_classification"] in (
            "healthy_diversity",
            "effective_consensus",
            "incoherent_team",
            "joint_myopia",
            "high_convergence",
            "low_convergence",
        )

    def test_convergence_classification_joint_myopia(self):
        # High similarity texts + failure → joint_myopia
        text = "aircraft parameters fuel_burn gtow wing_mass results"
        msgs = [
            _msg("agent_1", text),
            _msg("agent_2", text),
            _msg("agent_3", text),
        ]
        w = []
        result = _networked_os_metrics(msgs, {}, False, w)
        assert result["convergence_classification"] == "joint_myopia"

    def test_blackboard_fields_computed_via_fallback(self):
        """Blackboard fields are now computed via Tier 2 fallback, not None."""
        msgs = self._make_messages()
        w = []
        result = _networked_os_metrics(msgs, {}, None, w)
        # Blackboard fields are now computed (Tier 2 fallback).
        assert result["blackboard_writes_total"] is not None
        assert result["blackboard_size_final"] is not None
        assert result["blackboard_reads_total"] is not None
        assert result["blackboard_utilization"] is not None
        # claim_conflicts defaults to 0 when no metadata.
        assert result["claim_conflicts"] == 0
        # These remain None (no data sources).
        for key in ("prediction_count", "prediction_accuracy_mean", "self_selection_diversity"):
            assert result[key] is None
        assert len(w) >= 3

    def test_duplicate_work_rate_computed(self):
        text = "aircraft parameters results"
        msgs = [_msg(f"agent_{i}", text) for i in range(1, 4)]
        w = []
        result = _networked_os_metrics(msgs, {}, None, w)
        assert result["duplicate_work_rate"] is not None
        assert result["duplicate_work_rate"] == pytest.approx(1.0, abs=0.01)

    def test_non_peer_agents_ignored(self):
        msgs = [
            _msg("orchestrator", "directing"),
            _msg("agent_1", "peer work"),
        ]
        w = []
        result = _networked_os_metrics(msgs, {}, None, w)
        assert result["agents_spawned"] == 1


# ---------------------------------------------------------------------------
# _sequential_os_metrics
# ---------------------------------------------------------------------------


class TestSequentialOSMetrics:
    def _make_messages(self):
        return [
            _msg("design_planning", "Plan created", turn_number=1, duration_seconds=34.0, timestamp=100.0),
            _msg("code_writing", "Code written", turn_number=2, duration_seconds=30.0, timestamp=134.0),
            _msg("code_execution", "Code executed", turn_number=3, duration_seconds=71.0, timestamp=164.0),
            _msg("output_review", "ACCEPTABLE", turn_number=4, duration_seconds=42.0, timestamp=235.0),
        ]

    def test_stage_count(self):
        msgs = self._make_messages()
        w = []
        result = _sequential_os_metrics(msgs, {}, w)
        assert result["stage_count"] == 4

    def test_per_stage_duration(self):
        msgs = self._make_messages()
        w = []
        result = _sequential_os_metrics(msgs, {}, w)
        assert result["per_stage_duration"] == pytest.approx([34.0, 30.0, 71.0, 42.0])

    def test_stage_bottleneck(self):
        msgs = self._make_messages()
        w = []
        result = _sequential_os_metrics(msgs, {}, w)
        assert result["stage_bottleneck"] == "code_execution"

    def test_propagation_time(self):
        msgs = self._make_messages()
        w = []
        result = _sequential_os_metrics(msgs, {}, w)
        # last timestamp (235) - first timestamp (100) = 135
        assert result["propagation_time"] == pytest.approx(135.0)

    def test_per_stage_propagation_equals_per_stage_duration(self):
        msgs = self._make_messages()
        w = []
        result = _sequential_os_metrics(msgs, {}, w)
        assert result["per_stage_propagation"] == result["per_stage_duration"]

    def test_template_used_from_config(self):
        msgs = self._make_messages()
        w = []
        result = _sequential_os_metrics(msgs, {"pipeline_template": "aviary"}, w)
        assert result["template_used"] == "aviary"

    def test_template_used_none_when_absent(self):
        msgs = self._make_messages()
        w = []
        result = _sequential_os_metrics(msgs, {}, w)
        assert result["template_used"] is None

    def test_null_fields_with_warnings(self):
        msgs = self._make_messages()
        w = []
        result = _sequential_os_metrics(msgs, {}, w)
        for key in (
            "per_stage_tokens",
            "tool_utilization_per_stage",
            "stage_independence_score",
            "tool_restriction_violations",
        ):
            assert result[key] is None
        assert len(w) >= 4

    def test_single_message(self):
        msgs = [_msg("stage_1", "done", duration_seconds=5.0, timestamp=10.0)]
        w = []
        result = _sequential_os_metrics(msgs, {}, w)
        assert result["stage_count"] == 1
        assert result["propagation_time"] is None  # need 2+ timestamps


# ---------------------------------------------------------------------------
# Modularity metrics (tool_utilization, stage_independence, tool_violations)
# ---------------------------------------------------------------------------


class TestModularityMetrics:
    """Tests for modularity metrics in _sequential_os_metrics."""

    def _tc(self, name: str) -> "ToolCallRecord":
        return ToolCallRecord(
            tool_name=name,
            inputs={},
            output="ok",
            duration_seconds=0.1,
        )

    def _msgs_with_tools(self):
        """4-stage sample pipeline with realistic tool_calls."""
        return [
            AgentMessage(
                agent_name="design_planning",
                content="Plan",
                turn_number=1,
                timestamp=100.0,
                duration_seconds=34.0,
                token_count=500,
                tool_calls=[],
            ),
            AgentMessage(
                agent_name="code_writing",
                content="Code",
                turn_number=2,
                timestamp=134.0,
                duration_seconds=30.0,
                token_count=600,
                tool_calls=[],
            ),
            AgentMessage(
                agent_name="code_execution",
                content="Executed",
                turn_number=3,
                timestamp=164.0,
                duration_seconds=71.0,
                token_count=800,
                tool_calls=[
                    self._tc("run_simulation"),
                    self._tc("get_results"),
                ],
            ),
            AgentMessage(
                agent_name="output_review",
                content="TASK_COMPLETE",
                turn_number=4,
                timestamp=235.0,
                duration_seconds=42.0,
                token_count=300,
                tool_calls=[self._tc("final_answer")],
            ),
        ]

    def _sample_allowed_tools(self):
        return {
            "design_planning": [],
            "code_writing": [],
            "code_execution": ["run_simulation", "get_results"],
            "output_review": [],
        }

    def test_tool_utilization_populated(self):
        """tool_utilization_per_stage shows tools used per stage."""
        msgs = self._msgs_with_tools()
        w = []
        result = _sequential_os_metrics(msgs, {}, w)
        util = result["tool_utilization_per_stage"]
        assert util is not None
        assert util["design_planning"] == []
        assert util["code_writing"] == []
        assert sorted(util["code_execution"]) == ["get_results", "run_simulation"]
        # final_answer is excluded
        assert util["output_review"] == []

    def test_tool_utilization_none_without_tool_calls(self):
        """When no messages have tool_calls, utilization is None."""
        msgs = [
            _msg("stage_a", "Plan"),
            _msg("stage_b", "Execute"),
        ]
        w = []
        result = _sequential_os_metrics(msgs, {}, w)
        assert result["tool_utilization_per_stage"] is None

    def test_stage_independence_with_allowed_tools(self):
        """When _stage_allowed_tools is provided, compute from restrictions."""
        msgs = self._msgs_with_tools()
        config = {"_stage_allowed_tools": self._sample_allowed_tools()}
        w = []
        result = _sequential_os_metrics(msgs, config, w)
        # No violations: code_execution uses only its allowed tools
        assert result["tool_restriction_violations"] == 0
        assert result["stage_independence_score"] == 1.0

    def test_stage_independence_with_violation(self):
        """Detect stages using tools outside their allowed list."""
        msgs = [
            AgentMessage(
                agent_name="planner",
                content="Plan",
                turn_number=1,
                timestamp=100.0,
                duration_seconds=10.0,
                tool_calls=[self._tc("run_simulation")],  # not allowed
            ),
            AgentMessage(
                agent_name="executor",
                content="Done",
                turn_number=2,
                timestamp=110.0,
                duration_seconds=20.0,
                tool_calls=[self._tc("run_simulation")],
            ),
        ]
        config = {
            "_stage_allowed_tools": {
                "planner": [],  # planner shouldn't use run_simulation
                "executor": ["*"],
            }
        }
        w = []
        result = _sequential_os_metrics(msgs, config, w)
        assert result["tool_restriction_violations"] == 1
        assert result["stage_independence_score"] == 0.5

    def test_wildcard_allows_any_tool(self):
        """Stages with allowed_tools=["*"] never count as violations."""
        msgs = [
            AgentMessage(
                agent_name="worker",
                content="Done",
                turn_number=1,
                timestamp=100.0,
                duration_seconds=10.0,
                tool_calls=[self._tc("any_tool"), self._tc("another_tool")],
            ),
        ]
        config = {"_stage_allowed_tools": {"worker": ["*"]}}
        w = []
        result = _sequential_os_metrics(msgs, config, w)
        assert result["tool_restriction_violations"] == 0
        assert result["stage_independence_score"] == 1.0

    def test_stage_independence_from_overlap_without_config(self):
        """Without _stage_allowed_tools, compute from cross-stage tool overlap."""
        msgs = [
            AgentMessage(
                agent_name="stage_a",
                content="A",
                turn_number=1,
                timestamp=100.0,
                duration_seconds=10.0,
                tool_calls=[self._tc("tool_x"), self._tc("tool_y")],
            ),
            AgentMessage(
                agent_name="stage_b",
                content="B",
                turn_number=2,
                timestamp=110.0,
                duration_seconds=10.0,
                tool_calls=[self._tc("tool_x"), self._tc("tool_z")],
            ),
        ]
        w = []
        result = _sequential_os_metrics(msgs, {}, w)
        # stage_b uses tool_x which was also used by stage_a → 1 cross-ref
        # total_refs = 4 (2 from stage_a, 2 from stage_b)
        # cross_refs = 1 (tool_x in stage_b, present in stage_a)
        # independence = 1 - 1/4 = 0.75
        assert result["stage_independence_score"] == 0.75
        assert result["tool_restriction_violations"] is None

    def test_stage_independence_no_overlap(self):
        """Complete tool separation = independence 1.0."""
        msgs = [
            AgentMessage(
                agent_name="stage_a",
                content="A",
                turn_number=1,
                timestamp=100.0,
                duration_seconds=10.0,
                tool_calls=[self._tc("tool_x")],
            ),
            AgentMessage(
                agent_name="stage_b",
                content="B",
                turn_number=2,
                timestamp=110.0,
                duration_seconds=10.0,
                tool_calls=[self._tc("tool_y")],
            ),
        ]
        w = []
        result = _sequential_os_metrics(msgs, {}, w)
        assert result["stage_independence_score"] == 1.0

    def test_per_stage_tokens(self):
        """per_stage_tokens is populated when token_count is on messages."""
        msgs = self._msgs_with_tools()
        w = []
        result = _sequential_os_metrics(msgs, {}, w)
        assert result["per_stage_tokens"] == [500, 600, 800, 300]

    def test_per_stage_tokens_none_without_token_count(self):
        """per_stage_tokens is None when no token_count on messages."""
        msgs = [
            _msg("stage_a", "A"),
            _msg("stage_b", "B"),
        ]
        w = []
        result = _sequential_os_metrics(msgs, {}, w)
        assert result["per_stage_tokens"] is None

    def test_allowed_tools_no_tool_calls_still_computes(self):
        """With _stage_allowed_tools but no tool_calls, violations=0, independence=1."""
        msgs = [
            _msg("design_planning", "Plan", turn_number=1),
            _msg("code_execution", "Done", turn_number=2),
        ]
        config = {
            "_stage_allowed_tools": {
                "design_planning": [],
                "code_execution": ["run_simulation"],
            }
        }
        w = []
        result = _sequential_os_metrics(msgs, config, w)
        assert result["tool_restriction_violations"] == 0
        assert result["stage_independence_score"] == 1.0


# ---------------------------------------------------------------------------
# _iterative_feedback_metrics
# ---------------------------------------------------------------------------


class TestIterativeFeedbackMetrics:
    def test_single_attempt_per_agent(self):
        msgs = [
            _msg("agent_a", "first output", turn_number=1),
            _msg("agent_b", "second output", turn_number=2),
        ]
        w = []
        result = _iterative_feedback_metrics(msgs, {"aspiration_mode": "tool_success"}, w)
        assert result["max_attempts_used"] == 1
        assert result["aspiration_mode"] == "tool_success"
        assert result["ambidexterity_score"] is None  # single attempt — no variance
        assert result["escalation_length"] == 0
        assert result["escalation_detected"] is False

    def test_aspiration_mode_from_config(self):
        msgs = [_msg("agent_a", "output")]
        w = []
        result = _iterative_feedback_metrics(msgs, {"aspiration_mode": "any_output"}, w)
        assert result["aspiration_mode"] == "any_output"

    def test_mean_attempts_single_try(self):
        msgs = [
            _msg("stage_a", "output_a", turn_number=1),
            _msg("stage_b", "output_b", turn_number=2),
        ]
        w = []
        result = _iterative_feedback_metrics(msgs, {}, w)
        assert result["mean_attempts_to_success"] == pytest.approx(1.0)
        # Should warn about no retries
        assert any("no retries" in ww.lower() for ww in w)

    def test_escalation_detection_with_retries(self):
        # Simulate 4 identical retries from same agent
        repeated = "aircraft parameters fuel_burn 7000 gtow 67365"
        msgs = [
            _msg("agent_a", repeated, turn_number=1, is_retry=False),
            _msg("agent_a", repeated, turn_number=2, is_retry=True),
            _msg("agent_a", repeated, turn_number=3, is_retry=True),
            _msg("agent_a", repeated, turn_number=4, is_retry=True),
            _msg("agent_a", repeated, turn_number=5, is_retry=True),
        ]
        w = []
        result = _iterative_feedback_metrics(msgs, {}, w)
        assert result["escalation_length"] >= 3
        # 4+ consecutive similar retries → detected
        assert result["escalation_detected"] is True

    def test_no_escalation_when_diverse_retries(self):
        msgs = [
            _msg("agent_a", "completely different text alpha beta gamma", turn_number=1),
            _msg("agent_a", "entirely new approach delta epsilon zeta eta", turn_number=2),
            _msg("agent_a", "yet another solution one two three four five", turn_number=3),
        ]
        w = []
        result = _iterative_feedback_metrics(msgs, {}, w)
        assert result["escalation_length"] == 0
        assert result["escalation_detected"] is False

    def test_ambidexterity_high_variance(self):
        # Alternating very similar then very different outputs → high variance
        msgs = [
            _msg("agent_a", "identical output text here", turn_number=1),
            _msg("agent_a", "identical output text here", turn_number=2),
            _msg("agent_a", "completely unrelated content xyz", turn_number=3),
        ]
        w = []
        result = _iterative_feedback_metrics(msgs, {}, w)
        assert result["ambidexterity_score"] is not None
        assert result["ambidexterity_score"] >= 0.0

    def test_early_stopping_count_null(self):
        msgs = [_msg("agent_a", "done")]
        w = []
        result = _iterative_feedback_metrics(msgs, {}, w)
        assert result["early_stopping_count"] is None
        assert any("early_stopping" in ww for ww in w)

    def test_empty_messages(self):
        w = []
        result = _iterative_feedback_metrics([], {}, w)
        assert result["max_attempts_used"] == 0


# ---------------------------------------------------------------------------
# _graph_routed_metrics
# ---------------------------------------------------------------------------


class TestGraphRoutedMetrics:
    def _make_messages(self):
        # Simulate a graph traversal: classifier → designer → coder → executor
        return [
            _msg("classifier", "simple task", turn_number=1),
            _msg("designer", "design complete", turn_number=2),
            _msg("coder", "code written", turn_number=3),
            _msg("code_reviewer", "looks good", turn_number=4),
            _msg("executor", "executed", turn_number=5),
            _msg("output_reviewer", "ACCEPTABLE", turn_number=6),
        ]

    def test_total_transitions(self):
        msgs = self._make_messages()
        w = []
        result = _graph_routed_metrics(msgs, {}, w)
        assert result["total_transitions"] == 5

    def test_routing_accuracy_with_output_reviewer(self):
        msgs = self._make_messages()
        w = []
        result = _graph_routed_metrics(msgs, {}, w)
        # 1 output_reviewer turn / 5 transitions = 0.2
        assert result["routing_accuracy"] is not None
        assert 0.0 <= result["routing_accuracy"] <= 1.0

    def test_misroute_rate_no_revisits(self):
        msgs = self._make_messages()
        w = []
        result = _graph_routed_metrics(msgs, {}, w)
        # No revisited states → misroute_rate = 0
        assert result["misroute_rate"] == pytest.approx(0.0)

    def test_misroute_rate_with_revisits(self):
        msgs = [
            _msg("classifier", "classify", turn_number=1),
            _msg("coder", "code v1", turn_number=2),
            _msg("classifier", "reclassify", turn_number=3),  # revisit
            _msg("coder", "code v2", turn_number=4),  # revisit
            _msg("output_reviewer", "done", turn_number=5),
        ]
        w = []
        result = _graph_routed_metrics(msgs, {}, w)
        # 2 revisits / 4 transitions = 0.5
        assert result["misroute_rate"] == pytest.approx(0.5)

    def test_null_fields_with_warnings(self):
        msgs = self._make_messages()
        w = []
        result = _graph_routed_metrics(msgs, {}, w)
        # Without resource metadata in messages, attention metrics default to None.
        for key in (
            "resource_utilization",
            "context_utilization",
            "complexity_budget",
            "budget_constrained",
            "initial_complexity",
            "final_complexity",
            "complexity_escalations",
            "missed_routes",
        ):
            assert result[key] is None
        # Warnings for fields that could not be inferred:
        # missed_routes (always) + initial_complexity (no metadata).
        assert len(w) >= 2

    def test_attention_metrics_from_metadata(self):
        """When messages carry resource metadata, attention metrics are populated."""
        msgs = [
            AgentMessage(
                agent_name="classifier",
                content="simple",
                turn_number=1,
                timestamp=0.0,
                duration_seconds=1.0,
                metadata={
                    "complexity": "simple",
                    "passes_remaining": 5,
                    "passes_max": 6,
                    "context_used": 10,
                    "context_budget": 2000,
                },
            ),
            AgentMessage(
                agent_name="coder",
                content="code",
                turn_number=2,
                timestamp=1.0,
                duration_seconds=1.0,
                metadata={
                    "complexity": "simple",
                    "passes_remaining": 4,
                    "passes_max": 6,
                    "context_used": 30,
                    "context_budget": 2000,
                },
            ),
            AgentMessage(
                agent_name="executor",
                content="success",
                turn_number=3,
                timestamp=2.0,
                duration_seconds=1.0,
                metadata={
                    "complexity": "simple",
                    "passes_remaining": 3,
                    "passes_max": 6,
                    "context_used": 50,
                    "context_budget": 2000,
                },
            ),
        ]
        w = []
        result = _graph_routed_metrics(msgs, {}, w)
        assert result["initial_complexity"] == "simple"
        assert result["final_complexity"] == "simple"
        assert result["resource_utilization"] == pytest.approx(0.5)  # 3/6
        assert result["context_utilization"] == pytest.approx(0.025)  # 50/2000
        assert result["complexity_escalations"] == 0
        assert result["complexity_budget"] == 6
        assert result["budget_constrained"] is False

    def test_mental_model_from_config(self):
        msgs = self._make_messages()
        w = []
        result = _graph_routed_metrics(msgs, {"mental_model_enabled": True}, w)
        assert result["mental_model_enabled"] is True

    def test_empty_messages(self):
        w = []
        result = _graph_routed_metrics([], {}, w)
        assert result["total_transitions"] == 0
        assert result["misroute_rate"] is None


# ---------------------------------------------------------------------------
# _staged_pipeline_metrics
# ---------------------------------------------------------------------------


class TestStagedPipelineMetrics:
    def _make_messages(self, errors=None):
        errors = errors or [None, None, None, None]
        stages = ["design_planning", "code_writing", "code_execution", "output_review"]
        contents = [
            "Plan: build a box 0.59 x 0.44 x 0.37",
            "Code:\n```python\nprob.setup()\nprob.run_driver()\n```",
            "Execution succeeded. fuel_burn=7000.5",
            "ACCEPTABLE",
        ]
        return [
            _msg(stage, content, turn_number=i + 1, duration_seconds=float(10 + i * 5), error=errors[i])
            for i, (stage, content) in enumerate(zip(stages, contents))
        ]

    def test_stage_count(self):
        msgs = self._make_messages()
        w = []
        result = _staged_pipeline_metrics(msgs, {}, w)
        assert result["stage_count"] == 4

    def test_completion_rate_all_met(self):
        msgs = self._make_messages()
        w = []
        result = _staged_pipeline_metrics(msgs, {}, w)
        assert result["completion_rate"] == pytest.approx(1.0)

    def test_completion_rate_with_error(self):
        msgs = self._make_messages(errors=[None, None, "SyntaxError", None])
        w = []
        result = _staged_pipeline_metrics(msgs, {}, w)
        # 3 of 4 stages met criteria
        assert result["completion_rate"] == pytest.approx(0.75)

    def test_per_stage_completion_structure(self):
        msgs = self._make_messages()
        w = []
        result = _staged_pipeline_metrics(msgs, {}, w)
        comp = result["per_stage_completion"]
        assert len(comp) == 4
        assert all("stage" in c and "met" in c for c in comp)

    def test_error_propagation_chain(self):
        # Stage 2 fails → stage 3 also fails (propagation)
        msgs = [
            _msg("stage_1", "success", turn_number=1, duration_seconds=5.0),
            _msg("stage_2", "error occurred", turn_number=2, duration_seconds=5.0, error="RuntimeError"),
            _msg("stage_3", "also failed", turn_number=3, duration_seconds=5.0, error="NameError"),
            _msg("stage_4", "recovered", turn_number=4, duration_seconds=5.0),
        ]
        w = []
        result = _staged_pipeline_metrics(msgs, {}, w)
        assert result["error_propagation_count"] >= 1
        assert result["first_failure_stage"] == 1  # 0-indexed

    def test_no_errors_propagation(self):
        msgs = self._make_messages()
        w = []
        result = _staged_pipeline_metrics(msgs, {}, w)
        assert result["error_propagation_count"] == 0
        assert result["first_failure_stage"] is None
        assert result["propagation_depth"] == 0

    def test_propagation_rate_computation(self):
        # Stage 1 fails, stage 2 fails (propagation), stage 3 succeeds (recovery)
        msgs = [
            _msg("s0", "ok", turn_number=1, duration_seconds=1.0),
            _msg("s1", "x", turn_number=2, duration_seconds=1.0, error="Err"),
            _msg("s2", "also failed", turn_number=3, duration_seconds=1.0, error="Err2"),
            _msg("s3", "success content", turn_number=4, duration_seconds=1.0),
        ]
        w = []
        result = _staged_pipeline_metrics(msgs, {}, w)
        # P(fail|prev failed) with 1 follow-up fail and 1 follow-up success = 0.5
        assert result["propagation_rate"] == pytest.approx(0.5)
        assert result["recovery_rate"] == pytest.approx(0.5)

    def test_empty_messages(self):
        w = []
        result = _staged_pipeline_metrics([], {}, w)
        assert result["stage_count"] == 0
        assert result["completion_rate"] is None


# ---------------------------------------------------------------------------
# compute_org_theory_metrics dispatch routing
# ---------------------------------------------------------------------------


class TestDispatchRouting:
    def _orch_msgs(self):
        return [
            _msg("orchestrator", "delegating", turn_number=1, duration_seconds=10.0),
            _msg("worker_a", "working", turn_number=2, duration_seconds=5.0),
        ]

    def _net_msgs(self):
        return [
            _msg("agent_1", "peer work", turn_number=1),
            _msg("agent_2", "more work", turn_number=2),
        ]

    def _seq_msgs(self):
        return [
            _msg("stage_a", "plan", turn_number=1, duration_seconds=5.0, timestamp=10.0),
            _msg("stage_b", "code", turn_number=2, duration_seconds=7.0, timestamp=15.0),
        ]

    def test_orchestrated_placeholder_routing(self):
        result = compute_org_theory_metrics(self._orch_msgs(), "orchestrated", "placeholder")
        # OS metrics present
        assert "agents_spawned" in result
        assert "orchestrator_overhead_ratio" in result
        # Handler metrics NOT present (placeholder)
        assert "aspiration_mode" not in result
        assert "total_transitions" not in result
        assert "stage_count" not in result

    def test_orchestrated_iterative_feedback_routing(self):
        result = compute_org_theory_metrics(self._orch_msgs(), "orchestrated", "iterative_feedback")
        assert "agents_spawned" in result  # OS
        assert "aspiration_mode" in result  # handler
        assert "escalation_detected" in result

    def test_orchestrated_graph_routed_routing(self):
        msgs = [
            _msg("orchestrator", "plan", turn_number=1),
            _msg("coder", "code", turn_number=2),
        ]
        result = compute_org_theory_metrics(msgs, "orchestrated", "graph_routed")
        assert "orchestrator_turns" in result  # OS
        assert "total_transitions" in result  # handler
        assert "routing_accuracy" in result

    def test_orchestrated_staged_pipeline_routing(self):
        result = compute_org_theory_metrics(self._orch_msgs(), "orchestrated", "staged_pipeline")
        assert "orchestrator_overhead_ratio" in result  # OS
        assert "completion_rate" in result  # handler
        assert "propagation_rate" in result

    def test_networked_graph_routed_routing(self):
        msgs = [
            _msg("agent_1", "classify", turn_number=1),
            _msg("agent_2", "code", turn_number=2),
        ]
        result = compute_org_theory_metrics(msgs, "networked", "graph_routed")
        assert "convergence_score" in result  # OS
        assert "total_transitions" in result  # handler

    def test_sequential_iterative_feedback_routing(self):
        result = compute_org_theory_metrics(self._seq_msgs(), "sequential", "iterative_feedback")
        assert "stage_count" in result  # OS
        assert "propagation_time" in result
        assert "ambidexterity_score" in result  # handler

    def test_sequential_staged_pipeline_routing(self):
        result = compute_org_theory_metrics(self._seq_msgs(), "sequential", "staged_pipeline")
        assert "stage_count" in result  # sequential OS
        assert "completion_rate" in result  # staged pipeline handler

    def test_unknown_os_warns(self):
        result = compute_org_theory_metrics(self._orch_msgs(), "unknown_os", "placeholder")
        assert "warnings" in result
        assert any("unknown_os" in w.lower() for w in result["warnings"])

    def test_unknown_handler_warns(self):
        result = compute_org_theory_metrics(self._orch_msgs(), "orchestrated", "unknown_handler")
        assert "warnings" in result
        assert any("unknown_handler" in w.lower() for w in result["warnings"])

    def test_warnings_key_always_present(self):
        result = compute_org_theory_metrics([], "orchestrated", "placeholder")
        assert "warnings" in result
        assert isinstance(result["warnings"], list)

    def test_none_config_handled(self):
        result = compute_org_theory_metrics(self._seq_msgs(), "sequential", "staged_pipeline", config=None)
        assert "stage_count" in result

    def test_dict_messages_accepted(self):
        # Serialised dict format (from batch_summary.json)
        msgs = [
            {
                "agent_name": "orchestrator",
                "content": "delegating",
                "turn_number": 1,
                "duration_seconds": 5.0,
                "timestamp": 100.0,
                "error": None,
                "tool_calls": [],
            },
            {
                "agent_name": "worker",
                "content": "working",
                "turn_number": 2,
                "duration_seconds": 3.0,
                "timestamp": 105.0,
                "error": None,
                "tool_calls": [],
            },
        ]
        result = compute_org_theory_metrics(msgs, "orchestrated", "placeholder")
        assert result["agents_spawned"] == 1
        assert result["orchestrator_turns"] == 1


# ---------------------------------------------------------------------------
# Integration: convergence_score with known texts
# ---------------------------------------------------------------------------


class TestConvergenceScoreIntegration:
    def test_identical_texts_high_convergence(self):
        text = "aircraft parameters fuel_burn 7000 gtow 67365 wing_mass 5200"
        msgs = [_msg(f"agent_{i}", text) for i in range(1, 4)]
        # Provide eval_success so classification resolves to a named outcome
        result = compute_org_theory_metrics(msgs, "networked", "placeholder", config={"_eval_success": True})
        assert result["convergence_score"] == pytest.approx(1.0, abs=0.01)
        assert result["convergence_classification"] == "effective_consensus"

    def test_dissimilar_texts_low_convergence(self):
        msgs = [
            _msg("agent_1", "alpha beta gamma delta epsilon zeta eta theta iota kappa"),
            _msg("agent_2", "one two three four five six seven eight nine ten eleven twelve"),
            _msg("agent_3", "red blue green yellow orange purple violet indigo magenta cyan"),
        ]
        result = compute_org_theory_metrics(msgs, "networked", "placeholder")
        assert result["convergence_score"] is not None
        assert result["convergence_score"] < 0.6


# ---------------------------------------------------------------------------
# Integration: escalation_length with known sequences
# ---------------------------------------------------------------------------


class TestEscalationIntegration:
    def test_escalation_detected_after_4_similar_retries(self):
        text = "aircraft parameters fuel_burn 7000 gtow 67365 DONE complete finished"
        msgs = [
            _msg("stage_a", text, turn_number=i)
            for i in range(1, 7)  # 6 identical attempts → escalation_length >= 4
        ]
        result = compute_org_theory_metrics(msgs, "sequential", "iterative_feedback")
        assert result["escalation_length"] >= 4
        assert result["escalation_detected"] is True

    def test_no_escalation_single_attempt(self):
        msgs = [_msg("stage_a", "some output")]
        result = compute_org_theory_metrics(msgs, "sequential", "iterative_feedback")
        assert result["escalation_length"] == 0
        assert result["escalation_detected"] is False


# ---------------------------------------------------------------------------
# Integration: ambidexterity_score with known patterns
# ---------------------------------------------------------------------------


class TestAmbidexterityIntegration:
    def test_high_variance_ambidexterity(self):
        # Agent alternates between very similar and very different outputs
        msgs = [
            _msg("agent_a", "aircraft parameters fuel_burn 7000 gtow 67365"),  # attempt 1
            _msg("agent_a", "aircraft parameters fuel_burn 7000 gtow 67365"),  # attempt 2 — identical (high sim)
            _msg("agent_a", "spherical assembly design completely new approach xyz uvw"),  # attempt 3 — different
        ]
        result = compute_org_theory_metrics(msgs, "networked", "iterative_feedback")
        # Variance of [high_sim, low_sim] should be > 0
        assert result["ambidexterity_score"] is not None
        assert result["ambidexterity_score"] > 0.0

    def test_zero_variance_pure_exploitation(self):
        # All outputs identical → all similarities = 1.0 → variance = 0
        text = "aircraft parameters fuel_burn gtow wing_mass converged"
        msgs = [_msg("agent_a", text, turn_number=i) for i in range(1, 4)]
        result = compute_org_theory_metrics(msgs, "networked", "iterative_feedback")
        assert result["ambidexterity_score"] is not None
        assert result["ambidexterity_score"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Integration: error_propagation_rate with known stage sequences
# ---------------------------------------------------------------------------


class TestErrorPropagationIntegration:
    def test_full_propagation_chain(self):
        # All stages after first failure also fail
        msgs = [
            _msg("s0", "success", turn_number=1, duration_seconds=1.0),
            _msg("s1", "Error in code", turn_number=2, duration_seconds=1.0, error="SyntaxError"),
            _msg("s2", "Traceback error", turn_number=3, duration_seconds=1.0, error="RuntimeError"),
            _msg("s3", "FAILED execution", turn_number=4, duration_seconds=1.0, error="NameError"),
        ]
        result = compute_org_theory_metrics(msgs, "sequential", "staged_pipeline")
        # propagation_rate = 2/2 = 1.0 (both follow-up stages failed)
        assert result["propagation_rate"] == pytest.approx(1.0)
        assert result["recovery_rate"] == pytest.approx(0.0)
        assert result["first_failure_stage"] == 1

    def test_no_propagation_isolated_failures(self):
        msgs = [
            _msg("s0", "success", turn_number=1, duration_seconds=1.0),
            _msg("s1", "Error occurred", turn_number=2, duration_seconds=1.0, error="Err"),
            _msg("s2", "Recovered successfully", turn_number=3, duration_seconds=1.0),
            _msg("s3", "Success output ACCEPTABLE", turn_number=4, duration_seconds=1.0),
        ]
        result = compute_org_theory_metrics(msgs, "sequential", "staged_pipeline")
        assert result["propagation_rate"] == pytest.approx(0.0)
        assert result["recovery_rate"] == pytest.approx(1.0)

    def test_no_failures(self):
        msgs = [
            _msg("s0", "design output plan", turn_number=1, duration_seconds=1.0),
            _msg("s1", "code block here", turn_number=2, duration_seconds=1.0),
            _msg("s2", "execution success metrics", turn_number=3, duration_seconds=1.0),
        ]
        result = compute_org_theory_metrics(msgs, "sequential", "staged_pipeline")
        assert result["propagation_rate"] is None
        assert result["recovery_rate"] is None
        assert result["first_failure_stage"] is None
