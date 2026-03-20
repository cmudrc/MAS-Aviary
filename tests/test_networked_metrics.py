"""Tests for networked strategy metric computation.

No GPU needed. Tests per-prompt metrics, cross-prompt metrics,
joint myopia, prediction accuracy, and edge cases.
"""

import time

import pytest

from src.coordination.history import AgentMessage, ToolCallRecord
from src.logging.networked_metrics import (
    _compute_convergence,
    _failure_signature,
    _signature_overlap,
    compute_cross_prompt_metrics,
    compute_networked_metrics,
    compute_prediction_accuracy,
)

# ---- Helpers -----------------------------------------------------------------

def _msg(agent, content, turn, tool_calls=None, error=None):
    return AgentMessage(
        agent_name=agent,
        content=content,
        turn_number=turn,
        timestamp=time.time(),
        tool_calls=tool_calls or [],
        error=error,
    )


def _tc(name, inputs=None, output="ok", duration=0.1, error=None):
    return ToolCallRecord(
        tool_name=name,
        inputs=inputs or {},
        output=output,
        duration_seconds=duration,
        error=error,
    )


# ---- Per-prompt metrics tests ------------------------------------------------

class TestComputeNetworkedMetrics:
    def test_empty_messages(self):
        result = compute_networked_metrics([])
        assert result["total_agents"] == 0
        assert result["duplicate_work_rate"] == 0.0

    def test_basic_metrics(self):
        messages = [
            _msg("agent_1", "did work", 1, [
                _tc("read_blackboard"),
                _tc("write_blackboard", {"key": "task_a", "entry_type": "claim"}),
            ]),
            _msg("agent_2", "also worked", 2, [
                _tc("read_blackboard"),
                _tc("write_blackboard", {"key": "task_b", "entry_type": "result"}),
            ]),
        ]
        result = compute_networked_metrics(
            messages, blackboard_size=4, claim_conflicts=0,
            initial_agents=2, spawned_agents=0,
        )
        assert result["total_agents"] == 2
        assert result["blackboard_size"] == 4
        assert result["total_bb_reads"] == 2
        assert result["total_bb_writes"] == 2

    def test_blackboard_utilization(self):
        messages = [
            _msg("agent_1", "work", 1, [_tc("read_blackboard")]),
            _msg("agent_2", "work", 2, [_tc("echo_tool")]),
            _msg("agent_3", "work", 3, [_tc("read_blackboard")]),
        ]
        result = compute_networked_metrics(messages, initial_agents=3)
        assert result["blackboard_utilization"] == pytest.approx(2 / 3, rel=0.01)

    def test_duplicate_work_rate(self):
        """Two agents working on same subtask."""
        messages = [
            _msg("agent_1", "work", 1, [
                _tc("write_blackboard", {"key": "subtask_A", "entry_type": "claim"}),
            ]),
            _msg("agent_2", "work", 2, [
                _tc("write_blackboard", {"key": "subtask_A", "entry_type": "result"}),
            ]),
            _msg("agent_3", "work", 3, [
                _tc("write_blackboard", {"key": "subtask_B", "entry_type": "result"}),
            ]),
        ]
        result = compute_networked_metrics(messages, initial_agents=3)
        # subtask_A has 2 agents, subtask_B has 1 → 1/2 = 0.5
        assert result["duplicate_work_rate"] == 0.5

    def test_no_duplicate_work(self):
        messages = [
            _msg("agent_1", "work", 1, [
                _tc("write_blackboard", {"key": "task_a", "entry_type": "claim"}),
            ]),
            _msg("agent_2", "work", 2, [
                _tc("write_blackboard", {"key": "task_b", "entry_type": "claim"}),
            ]),
        ]
        result = compute_networked_metrics(messages, initial_agents=2)
        assert result["duplicate_work_rate"] == 0.0

    def test_self_selection_diversity(self):
        messages = [
            _msg("agent_1", "work", 1, [
                _tc("write_blackboard", {"key": "task_a", "entry_type": "claim"}),
            ]),
            _msg("agent_2", "work", 2, [
                _tc("write_blackboard", {"key": "task_b", "entry_type": "claim"}),
            ]),
            _msg("agent_3", "work", 3, [
                _tc("write_blackboard", {"key": "task_c", "entry_type": "result"}),
            ]),
        ]
        result = compute_networked_metrics(messages, initial_agents=3)
        # 3 unique subtasks / 3 total turns = 1.0
        assert result["self_selection_diversity"] == 1.0

    def test_claim_conflicts_passed_through(self):
        result = compute_networked_metrics(
            [_msg("a", "x", 1)],
            claim_conflicts=5,
            initial_agents=1,
        )
        assert result["claim_conflicts"] == 5

    def test_spawn_peer_counted(self):
        messages = [
            _msg("agent_1", "spawning", 1, [_tc("spawn_peer")]),
        ]
        result = compute_networked_metrics(messages, initial_agents=1)
        assert result["peers_spawned"] == 1

    def test_prediction_accuracy(self):
        predictions = [
            {"accuracy_score": 0.8},
            {"accuracy_score": 0.6},
        ]
        result = compute_networked_metrics(
            [_msg("a", "x", 1)],
            initial_agents=1,
            predictions=predictions,
        )
        assert result["prediction_accuracy"] == pytest.approx(0.7, rel=0.01)


# ---- Cross-prompt metrics tests ---------------------------------------------

class TestComputeCrossPromptMetrics:
    def test_empty_data(self):
        result = compute_cross_prompt_metrics([])
        assert result["total_prompts"] == 0

    def test_omission_errors(self):
        data = [
            {"messages": [_msg("a", "x", 1)], "eval_score": 0.3, "redundancy_rate": 0.1},
            {"messages": [_msg("a", "x", 1)], "eval_score": 0.8, "redundancy_rate": 0.1},
            {"messages": [_msg("a", "x", 1)], "eval_score": 0.2, "redundancy_rate": 0.1},
        ]
        result = compute_cross_prompt_metrics(data)
        assert result["omission_errors"] == 2  # scores 0.3, 0.2 are below 0.5

    def test_commission_errors(self):
        """Commission: low eval but no error messages in history."""
        data = [
            {"messages": [_msg("a", "wrong", 1)], "eval_score": 0.3, "redundancy_rate": 0.0},
        ]
        result = compute_cross_prompt_metrics(data)
        assert result["commission_errors"] == 1

    def test_no_commission_if_errors_present(self):
        """If history has errors, it's not a commission error."""
        data = [
            {
                "messages": [_msg("a", "fail", 1, error="something broke")],
                "eval_score": 0.3,
                "redundancy_rate": 0.0,
            },
        ]
        result = compute_cross_prompt_metrics(data)
        assert result["commission_errors"] == 0

    def test_escalation_of_commitment(self):
        """Consecutive failures with similar signatures."""
        msg1 = _msg("a", "geometry calculation failed completely", 1, error="geometry error")
        msg2 = _msg("a", "geometry calculation failed again completely", 1, error="geometry error")
        msg3 = _msg("a", "totally different work succeeded", 1)
        data = [
            {"messages": [msg1], "eval_score": 0.2, "redundancy_rate": 0.1},
            {"messages": [msg2], "eval_score": 0.2, "redundancy_rate": 0.1},
            {"messages": [msg3], "eval_score": 0.8, "redundancy_rate": 0.1},
        ]
        result = compute_cross_prompt_metrics(data)
        assert result["escalation_of_commitment"] >= 1

    def test_ambidexterity_proxy(self):
        """Variance of redundancy rates."""
        data = [
            {"messages": [_msg("a", "x", 1)], "eval_score": 0.8, "redundancy_rate": 0.1},
            {"messages": [_msg("a", "x", 1)], "eval_score": 0.8, "redundancy_rate": 0.9},
        ]
        result = compute_cross_prompt_metrics(data)
        # mean = 0.5, variance = ((0.1-0.5)^2 + (0.9-0.5)^2) / 2 = 0.16
        assert result["ambidexterity_proxy"] == pytest.approx(0.16, rel=0.01)

    def test_prompt_classifications(self):
        data = [
            # joint_myopia: converged + error
            {
                "messages": [
                    _msg("a", "same words here", 1),
                    _msg("b", "same words here", 2),
                ],
                "eval_score": 0.2,
                "redundancy_rate": 0.5,
            },
            # healthy_diversity: diverse + success
            {
                "messages": [
                    _msg("a", "totally unique approach alpha beta", 1),
                    _msg("b", "completely different method gamma delta", 2),
                ],
                "eval_score": 0.9,
                "redundancy_rate": 0.1,
            },
        ]
        result = compute_cross_prompt_metrics(data)
        assert result["prompt_classifications"][0] == "joint_myopia"
        assert result["prompt_classifications"][1] == "healthy_diversity"


# ---- Prediction accuracy tests -----------------------------------------------

class TestPredictionAccuracy:
    def test_identical_text(self):
        score = compute_prediction_accuracy("agent will calculate sum", "agent will calculate sum")
        assert score == 1.0

    def test_no_overlap(self):
        score = compute_prediction_accuracy("alpha beta gamma", "delta epsilon zeta")
        assert score == 0.0

    def test_partial_overlap(self):
        score = compute_prediction_accuracy(
            "agent will calculate the sum",
            "agent will verify the result",
        )
        # Overlap: {"agent", "will", "the"} / union of all words
        assert 0.0 < score < 1.0

    def test_empty_prediction(self):
        assert compute_prediction_accuracy("", "something") == 0.0

    def test_empty_actual(self):
        assert compute_prediction_accuracy("something", "") == 0.0

    def test_both_empty(self):
        assert compute_prediction_accuracy("", "") == 0.0


# ---- Convergence / Joint Myopia tests ----------------------------------------

class TestConvergence:
    def test_identical_content(self):
        messages = [
            _msg("a", "exactly the same content here", 1),
            _msg("b", "exactly the same content here", 2),
        ]
        score = _compute_convergence(messages)
        assert score == 1.0

    def test_different_content(self):
        messages = [
            _msg("a", "alpha beta gamma delta epsilon", 1),
            _msg("b", "zeta eta theta iota kappa", 2),
        ]
        score = _compute_convergence(messages)
        assert score == 0.0

    def test_single_agent(self):
        messages = [_msg("a", "some content", 1)]
        score = _compute_convergence(messages)
        assert score == 0.0  # need at least 2 agents

    def test_empty_messages(self):
        assert _compute_convergence([]) == 0.0


# ---- Failure signature helpers -----------------------------------------------

class TestSignatureHelpers:
    def test_failure_signature_from_error(self):
        msg = _msg("a", "working", 1, error="geometry calculation failed")
        sig = _failure_signature([msg])
        assert "geometry" in sig
        assert "failed" in sig

    def test_signature_overlap_identical(self):
        s = {"a", "b", "c"}
        assert _signature_overlap(s, s) == 1.0

    def test_signature_overlap_disjoint(self):
        assert _signature_overlap({"a", "b"}, {"c", "d"}) == 0.0

    def test_signature_overlap_partial(self):
        overlap = _signature_overlap({"a", "b", "c"}, {"b", "c", "d"})
        assert overlap == pytest.approx(0.5, rel=0.01)
