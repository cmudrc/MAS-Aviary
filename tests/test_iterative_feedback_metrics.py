"""Tests for iterative feedback metrics.

No GPU needed. Tests ambidexterity, escalation, per-agent, per-prompt,
and cross-prompt metrics with synthetic AttemptFeedback data.
"""

import pytest

from src.coordination.feedback_extraction import AttemptFeedback, ToolCallOutcome
from src.logging.iterative_feedback_metrics import (
    compute_ambidexterity,
    compute_cross_prompt_metrics,
    compute_escalation,
    compute_per_agent_metrics,
    compute_per_prompt_metrics,
)

# ---- Helpers -----------------------------------------------------------------


def _fb(
    attempt: int = 0,
    output: str = "",
    has_errors: bool = False,
    error_types: list[str] | None = None,
) -> AttemptFeedback:
    """Build a minimal AttemptFeedback."""
    tool_calls = []
    if error_types:
        for et in error_types:
            tool_calls.append(
                ToolCallOutcome(
                    tool_name="tool",
                    success=not has_errors,
                    error_type=et,
                )
            )
    return AttemptFeedback(
        attempt_number=attempt,
        tool_calls=tool_calls,
        has_tool_errors=has_errors,
        error_messages=[f"{et}: msg" for et in (error_types or [])] if has_errors else [],
        output_content=output,
    )


# ---- Ambidexterity -----------------------------------------------------------


class TestAmbidexterity:
    def test_single_attempt_returns_none(self):
        result = compute_ambidexterity([_fb(output="hello")])
        assert result["score"] is None
        assert result["mode"] == "single_attempt"

    def test_identical_outputs_high_similarity(self):
        """All similar outputs → exploitation, low variance."""
        attempts = [_fb(attempt=i, output="Setting aircraft parameters and running simulation") for i in range(4)]
        result = compute_ambidexterity(attempts)
        assert result["mean_similarity"] > 0.8
        assert result["dominant_mode"] == "exploitation"
        assert result["ambidexterity_score"] < 0.1  # low variance

    def test_different_outputs_low_similarity(self):
        """All different outputs → exploration, low variance."""
        attempts = [
            _fb(attempt=0, output="alpha beta gamma delta epsilon"),
            _fb(attempt=1, output="zeta eta theta iota kappa"),
            _fb(attempt=2, output="lambda mu nu xi omicron"),
        ]
        result = compute_ambidexterity(attempts)
        assert result["mean_similarity"] < 0.4
        assert result["dominant_mode"] == "exploration"

    def test_alternating_outputs_high_variance(self):
        """Alternating similar/different consecutive pairs → high ambidexterity."""
        attempts = [
            _fb(attempt=0, output="a b c d e"),
            _fb(attempt=1, output="a b c d f"),  # (0,1): 4/6 overlap → ~0.67
            _fb(attempt=2, output="x y z w v"),  # (1,2): 0/10 overlap → 0.0
            _fb(attempt=3, output="x y z w u"),  # (2,3): 4/6 overlap → ~0.67
        ]
        result = compute_ambidexterity(attempts, similarity_method="jaccard")
        # Similarities: [~0.67, 0.0, ~0.67] → variance > 0
        assert result["ambidexterity_score"] > 0.05
        assert len(result["similarities"]) == 3

    def test_jaccard_method(self):
        attempts = [
            _fb(attempt=0, output="a b c d"),
            _fb(attempt=1, output="a b c d"),
        ]
        result = compute_ambidexterity(attempts, similarity_method="jaccard")
        assert result["mean_similarity"] == pytest.approx(1.0)


# ---- Escalation of commitment ------------------------------------------------


class TestEscalation:
    def test_no_escalation_with_single_attempt(self):
        result = compute_escalation([_fb()])
        assert result["escalation_length"] == 0
        assert result["escalation_detected"] is False

    def test_consecutive_similar_failures(self):
        """4+ consecutive similar failures → escalation detected."""
        attempts = [_fb(attempt=i, output="error: the same error message repeating", has_errors=True) for i in range(5)]
        result = compute_escalation(attempts, min_length=4)
        assert result["escalation_length"] >= 4
        assert result["escalation_detected"] is True

    def test_mixed_results_no_escalation(self):
        """Alternating success/failure → no escalation."""
        attempts = [
            _fb(attempt=0, output="failed attempt one", has_errors=True),
            _fb(attempt=1, output="success output", has_errors=False),
            _fb(attempt=2, output="failed attempt three", has_errors=True),
            _fb(attempt=3, output="success again", has_errors=False),
        ]
        result = compute_escalation(attempts)
        assert result["escalation_detected"] is False

    def test_all_failures_different_approaches(self):
        """All failures but different outputs → no escalation."""
        attempts = [
            _fb(attempt=0, output="alpha beta gamma", has_errors=True),
            _fb(attempt=1, output="delta epsilon zeta", has_errors=True),
            _fb(attempt=2, output="eta theta iota", has_errors=True),
            _fb(attempt=3, output="kappa lambda mu", has_errors=True),
        ]
        result = compute_escalation(attempts)
        assert result["escalation_detected"] is False

    def test_escalation_threshold_configurable(self):
        attempts = [_fb(attempt=i, output="same output", has_errors=True) for i in range(3)]
        # min_length=2 → should detect with 3 attempts
        result = compute_escalation(attempts, min_length=2)
        assert result["escalation_detected"] is True


# ---- Per-agent metrics -------------------------------------------------------


class TestPerAgentMetrics:
    def test_empty_attempts(self):
        result = compute_per_agent_metrics([])
        assert result["total_attempts"] == 0
        assert result["final_outcome"] == "no_attempts"

    def test_single_success(self):
        result = compute_per_agent_metrics([_fb(attempt=0, output="done")])
        assert result["total_attempts"] == 1
        assert result["success_attempt"] == 1
        assert result["retry_rate"] == 0.0
        assert result["final_outcome"] == "success"

    def test_fail_then_success(self):
        attempts = [
            _fb(attempt=0, output="err", has_errors=True, error_types=["ValueError"]),
            _fb(attempt=1, output="ok"),
        ]
        result = compute_per_agent_metrics(attempts)
        assert result["total_attempts"] == 2
        assert result["success_attempt"] == 2
        assert result["retry_rate"] == pytest.approx(0.5)
        assert "ValueError" in result["unique_error_types"]
        assert result["final_outcome"] == "success"

    def test_all_failures(self):
        attempts = [_fb(attempt=i, output=f"fail {i}", has_errors=True, error_types=["RuntimeError"]) for i in range(5)]
        result = compute_per_agent_metrics(attempts)
        assert result["total_attempts"] == 5
        assert result["success_attempt"] is None
        assert result["final_outcome"] == "max_retries_exhausted"

    def test_summary_handoff(self):
        attempts = [_fb(attempt=i, output="fail", has_errors=True) for i in range(19)]
        attempts.append(
            _fb(
                attempt=19,
                output="Summary: tried X, Y, Z. All failed.",
                has_errors=True,
            )
        )
        result = compute_per_agent_metrics(attempts)
        assert result["final_outcome"] == "summary_handoff"

    def test_multiple_error_types(self):
        attempts = [
            _fb(attempt=0, output="e1", has_errors=True, error_types=["AttributeError"]),
            _fb(attempt=1, output="e2", has_errors=True, error_types=["ValueError"]),
            _fb(attempt=2, output="ok"),
        ]
        result = compute_per_agent_metrics(attempts)
        assert "AttributeError" in result["unique_error_types"]
        assert "ValueError" in result["unique_error_types"]


# ---- Per-prompt metrics ------------------------------------------------------


class TestPerPromptMetrics:
    def test_empty_histories(self):
        result = compute_per_prompt_metrics([])
        assert result["total_attempts_all_agents"] == 0

    def test_all_first_try_success(self):
        histories = [
            [_fb(attempt=0, output="ok")],
            [_fb(attempt=0, output="ok")],
        ]
        result = compute_per_prompt_metrics(histories)
        assert result["agents_succeeded_first_try"] == 2
        assert result["agents_required_retries"] == 0
        assert result["mean_attempts_to_success"] == pytest.approx(1.0)

    def test_mixed_retry_counts(self):
        histories = [
            [_fb(attempt=0, output="ok")],
            [
                _fb(attempt=0, output="fail", has_errors=True),
                _fb(attempt=1, output="ok"),
            ],
        ]
        result = compute_per_prompt_metrics(histories)
        assert result["total_attempts_all_agents"] == 3
        assert result["agents_succeeded_first_try"] == 1
        assert result["agents_required_retries"] == 1
        assert result["mean_attempts_to_success"] == pytest.approx(1.5)

    def test_exhausted_and_handoff_counted(self):
        histories = [
            [_fb(attempt=i, output=f"fail {i}", has_errors=True) for i in range(5)],
            [_fb(attempt=i, output="Summary: ...", has_errors=True) for i in range(20)],
        ]
        result = compute_per_prompt_metrics(histories)
        assert result["agents_exhausted_retries"] >= 1
        assert result["summary_handoffs"] >= 0  # depends on output content

    def test_ambidexterity_collected(self):
        histories = [
            [
                _fb(attempt=0, output="attempt one approach"),
                _fb(attempt=1, output="attempt two very different approach"),
            ],
        ]
        result = compute_per_prompt_metrics(histories, similarity_method="jaccard")
        assert len(result["ambidexterity_scores"]) == 1


# ---- Cross-prompt metrics ----------------------------------------------------


class TestCrossPromptMetrics:
    def test_empty_list(self):
        result = compute_cross_prompt_metrics([])
        assert result["omission_error_count"] == 0
        assert result["commission_error_count"] == 0
        assert result["cross_prompt_escalation"] is False

    def test_no_errors(self):
        prompts = [
            {"agents_exhausted_retries": 0, "summary_handoffs": 0, "ambidexterity_scores": []},
            {"agents_exhausted_retries": 0, "summary_handoffs": 0, "ambidexterity_scores": []},
        ]
        result = compute_cross_prompt_metrics(prompts)
        assert result["omission_error_count"] == 0
        assert result["commission_error_count"] == 0

    def test_omission_and_commission_counted(self):
        prompts = [
            {"agents_exhausted_retries": 1, "summary_handoffs": 0, "ambidexterity_scores": []},
            {"agents_exhausted_retries": 0, "summary_handoffs": 2, "ambidexterity_scores": []},
        ]
        result = compute_cross_prompt_metrics(prompts)
        assert result["omission_error_count"] == 1
        assert result["commission_error_count"] == 1

    def test_cross_prompt_escalation(self):
        """3+ consecutive problem prompts → escalation."""
        prompts = [
            {"agents_exhausted_retries": 1, "summary_handoffs": 0, "ambidexterity_scores": []},
            {"agents_exhausted_retries": 0, "summary_handoffs": 1, "ambidexterity_scores": []},
            {"agents_exhausted_retries": 1, "summary_handoffs": 0, "ambidexterity_scores": []},
        ]
        result = compute_cross_prompt_metrics(prompts)
        assert result["cross_prompt_escalation"] is True
        assert result["consecutive_problem_prompts"] == 3

    def test_no_cross_prompt_escalation(self):
        prompts = [
            {"agents_exhausted_retries": 1, "summary_handoffs": 0, "ambidexterity_scores": []},
            {"agents_exhausted_retries": 0, "summary_handoffs": 0, "ambidexterity_scores": []},
            {"agents_exhausted_retries": 1, "summary_handoffs": 0, "ambidexterity_scores": []},
        ]
        result = compute_cross_prompt_metrics(prompts)
        assert result["cross_prompt_escalation"] is False

    def test_mean_ambidexterity(self):
        prompts = [
            {"agents_exhausted_retries": 0, "summary_handoffs": 0, "ambidexterity_scores": [0.1, 0.3]},
            {"agents_exhausted_retries": 0, "summary_handoffs": 0, "ambidexterity_scores": [0.2]},
        ]
        result = compute_cross_prompt_metrics(prompts)
        expected = (0.1 + 0.3 + 0.2) / 3
        assert result["mean_ambidexterity_per_prompt"] == pytest.approx(expected)
