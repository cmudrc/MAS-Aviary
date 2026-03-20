"""Unit tests for graph-routed metrics computation."""

from src.coordination.graph_routed_handler import TransitionRecord
from src.logging.graph_routed_metrics import (
    compute_cross_prompt_metrics,
    compute_per_prompt_metrics,
    compute_routing_quality,
)

# ---- Helpers ---------------------------------------------------------------


def _tr(
    from_state: str,
    to_state: str,
    condition: str = "always",
    agent: str | None = None,
    passes: int = 10,
    context: int = 0,
    cycles: int = 0,
) -> TransitionRecord:
    return TransitionRecord(
        from_state=from_state,
        to_state=to_state,
        condition_matched=condition,
        agent_invoked=agent,
        passes_remaining=passes,
        context_used=context,
        cycle_count=cycles,
    )


# ---- Per-prompt metrics ----------------------------------------------------


class TestPerPromptMetrics:
    def test_empty_transitions(self):
        m = compute_per_prompt_metrics([])
        assert m["total_transitions"] == 0
        assert m["unique_states_visited"] == 0
        assert m["path_efficiency"] == 1.0

    def test_simple_linear_path(self):
        transitions = [
            _tr("A", "B", "always", "w1", passes=5),
            _tr("B", "C", "always", "w2", passes=4),
        ]
        m = compute_per_prompt_metrics(transitions)
        assert m["total_transitions"] == 2
        assert m["unique_states_visited"] == 3  # A, B, C
        assert m["cycle_count"] == 0
        assert m["path_efficiency"] == 1.0

    def test_transition_count(self):
        transitions = [
            _tr("A", "B"),
            _tr("B", "C"),
            _tr("C", "D"),
            _tr("D", "E"),
        ]
        m = compute_per_prompt_metrics(transitions)
        assert m["total_transitions"] == 4

    def test_state_histogram(self):
        transitions = [
            _tr("A", "B"),
            _tr("B", "C"),
            _tr("C", "B"),
            _tr("B", "D"),
        ]
        m = compute_per_prompt_metrics(transitions)
        hist = m["states_visited_histogram"]
        assert hist.get("B", 0) >= 2  # visited at least twice

    def test_cycle_count_from_last_transition(self):
        transitions = [
            _tr("A", "B", cycles=0),
            _tr("B", "A", cycles=1),
            _tr("A", "C", cycles=2),
        ]
        m = compute_per_prompt_metrics(transitions)
        assert m["cycle_count"] == 2

    def test_escalation_count(self):
        transitions = [
            _tr("ERROR", "COMPLEXITY_ESCALATION"),
            _tr("COMPLEXITY_ESCALATION", "DESIGN"),
            _tr("DESIGN", "COMPLETE"),
        ]
        m = compute_per_prompt_metrics(transitions)
        assert m["escalations_triggered"] == 1

    def test_resource_utilization(self):
        transitions = [
            _tr("A", "B", passes=10),
            _tr("B", "C", passes=7),
        ]
        m = compute_per_prompt_metrics(transitions)
        # Used 3 out of 10 passes.
        assert abs(m["resource_utilization"] - 0.3) < 0.01

    def test_path_efficiency_with_cycles(self):
        transitions = [
            _tr("A", "B", cycles=0),
            _tr("B", "C", cycles=0),
            _tr("C", "A", cycles=1),
            _tr("A", "D", cycles=1),
        ]
        m = compute_per_prompt_metrics(transitions)
        # path_efficiency = 1 - (1 / 4) = 0.75
        assert abs(m["path_efficiency"] - 0.75) < 0.01

    def test_code_review_cycles(self):
        transitions = [
            _tr("CODE_WRITTEN", "CODE_REVIEWED"),
            _tr("CODE_REVIEWED", "CODE_WRITTEN"),
            _tr("CODE_WRITTEN", "CODE_REVIEWED"),
            _tr("CODE_REVIEWED", "CODE_EXECUTED"),
        ]
        m = compute_per_prompt_metrics(transitions)
        assert m["code_review_cycles"] == 2

    def test_error_type_distribution(self):
        transitions = [
            _tr("CODE_EXECUTED", "ERROR_CLASSIFICATION", condition="execution_success == false"),
            _tr("ERROR_CLASSIFICATION", "CODE_WRITTEN", condition="error_type in ['SyntaxError']"),
        ]
        m = compute_per_prompt_metrics(transitions)
        assert len(m["error_type_distribution"]) > 0


# ---- Routing quality -------------------------------------------------------


class TestRoutingQuality:
    def test_empty_transitions(self):
        m = compute_routing_quality([])
        assert m["routing_accuracy"] == 0.0
        assert m["misroute_rate"] == 0.0
        assert m["missed_routes"] == 0.0

    def test_perfect_linear_path(self):
        transitions = [
            _tr("A", "B", condition="x == 1"),
            _tr("B", "C", condition="y == 2"),
            _tr("C", "COMPLETE", condition="z == 3"),
        ]
        m = compute_routing_quality(transitions, terminal_states=["COMPLETE"])
        assert m["routing_accuracy"] == 1.0
        assert m["misroute_rate"] == 0.0
        assert m["missed_routes"] == 0.0

    def test_misroute_rate(self):
        transitions = [
            _tr("A", "B", condition="x == 1"),
            _tr("B", "A", condition="always"),  # back to A → misroute
            _tr("A", "COMPLETE", condition="x == 2"),
        ]
        m = compute_routing_quality(transitions, terminal_states=["COMPLETE"])
        # 1 misroute out of 3 transitions
        assert abs(m["misroute_rate"] - 1 / 3) < 0.01

    def test_missed_routes_always_fallback(self):
        transitions = [
            _tr("A", "B", condition="always"),
            _tr("B", "C", condition="always"),
            _tr("C", "COMPLETE", condition="x == 1"),
        ]
        m = compute_routing_quality(transitions, terminal_states=["COMPLETE"])
        # 2 out of 3 used "always"
        assert abs(m["missed_routes"] - 2 / 3) < 0.01

    def test_routing_accuracy_with_backtrack(self):
        transitions = [
            _tr("A", "B", condition="x == 1"),
            _tr("B", "C", condition="x == 2"),
            _tr("C", "B", condition="always"),  # back to B → not toward terminal
            _tr("B", "COMPLETE", condition="x == 3"),
        ]
        m = compute_routing_quality(transitions, terminal_states=["COMPLETE"])
        # 3 toward terminal (A→B, B→C, B→COMPLETE), 1 backtrack (C→B)
        # accuracy = 3/4 = 0.75
        assert abs(m["routing_accuracy"] - 0.75) < 0.01


# ---- Cross-prompt metrics --------------------------------------------------


class TestCrossPromptMetrics:
    def test_empty_list(self):
        m = compute_cross_prompt_metrics([])
        assert m["total_prompts"] == 0

    def test_single_prompt(self):
        pm = {
            "total_transitions": 5,
            "path_efficiency": 0.8,
            "resource_utilization": 0.5,
            "escalations_triggered": 0,
            "cycle_count": 1,
            "initial_complexity": "simple",
        }
        m = compute_cross_prompt_metrics([pm])
        assert m["total_prompts"] == 1
        assert m["mean_path_length"] == 5.0
        assert abs(m["mean_path_efficiency"] - 0.8) < 0.01
        assert m["escalation_rate"] == 0.0

    def test_multiple_prompts(self):
        metrics = [
            {
                "total_transitions": 4,
                "path_efficiency": 1.0,
                "resource_utilization": 0.3,
                "escalations_triggered": 0,
                "cycle_count": 0,
                "initial_complexity": "simple",
            },
            {
                "total_transitions": 10,
                "path_efficiency": 0.7,
                "resource_utilization": 0.8,
                "escalations_triggered": 1,
                "cycle_count": 2,
                "initial_complexity": "moderate",
            },
            {
                "total_transitions": 15,
                "path_efficiency": 0.5,
                "resource_utilization": 0.9,
                "escalations_triggered": 2,
                "cycle_count": 5,
                "initial_complexity": "complex",
            },
        ]
        m = compute_cross_prompt_metrics(metrics)
        assert m["total_prompts"] == 3
        assert abs(m["mean_path_length"] - (4 + 10 + 15) / 3) < 0.01
        assert m["escalation_rate"] == 2 / 3  # 2 out of 3 had escalations

    def test_omission_high_cycles(self):
        metrics = [
            {
                "cycle_count": 5,
                "path_efficiency": 0.2,
                "total_transitions": 10,
                "resource_utilization": 0.8,
                "escalations_triggered": 0,
            },
        ]
        m = compute_cross_prompt_metrics(metrics)
        assert m["omission_error_count"] == 1  # cycle_count > 3

    def test_commission_low_efficiency(self):
        metrics = [
            {
                "path_efficiency": 0.3,
                "cycle_count": 0,
                "total_transitions": 10,
                "resource_utilization": 0.5,
                "escalations_triggered": 0,
            },
        ]
        m = compute_cross_prompt_metrics(metrics)
        assert m["commission_error_count"] == 1  # path_efficiency < 0.5

    def test_mean_path_per_complexity(self):
        metrics = [
            {
                "initial_complexity": "simple",
                "total_transitions": 4,
                "path_efficiency": 1.0,
                "resource_utilization": 0.3,
                "escalations_triggered": 0,
                "cycle_count": 0,
            },
            {
                "initial_complexity": "simple",
                "total_transitions": 6,
                "path_efficiency": 0.9,
                "resource_utilization": 0.4,
                "escalations_triggered": 0,
                "cycle_count": 0,
            },
            {
                "initial_complexity": "complex",
                "total_transitions": 15,
                "path_efficiency": 0.7,
                "resource_utilization": 0.8,
                "escalations_triggered": 1,
                "cycle_count": 2,
            },
        ]
        m = compute_cross_prompt_metrics(metrics)
        per_c = m["mean_path_length_per_complexity"]
        assert abs(per_c["simple"] - 5.0) < 0.01  # (4+6)/2
        assert abs(per_c["complex"] - 15.0) < 0.01
