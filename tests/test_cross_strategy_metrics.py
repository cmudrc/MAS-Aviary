"""Unit tests for cross-strategy metrics."""

import time

from src.coordination.history import AgentMessage
from src.logging.cross_strategy_metrics import (
    compute_coordination_efficiency,
    compute_coordination_overhead,
    compute_cross_strategy_metrics,
    compute_error_amplification,
    compute_message_density,
    compute_redundancy_rate,
)

# ---- Helpers ---------------------------------------------------------------

def _msg(agent: str, content: str, error: str | None = None) -> AgentMessage:
    return AgentMessage(
        agent_name=agent,
        content=content,
        turn_number=0,
        timestamp=time.time(),
        error=error,
    )


# ---- Coordination Overhead -------------------------------------------------

class TestCoordinationOverhead:
    def test_no_overhead(self):
        msgs = [_msg("a", "hello"), _msg("b", "world")]
        assert compute_coordination_overhead(msgs) == 0.0

    def test_with_overhead(self):
        msgs = [_msg("a", "1"), _msg("a", "2"), _msg("a", "3")]
        # 3 turns, 1 unique agent → overhead = 2
        assert compute_coordination_overhead(msgs) == 2.0

    def test_explicit_minimum(self):
        msgs = [_msg("a", "1"), _msg("b", "2"), _msg("c", "3")]
        assert compute_coordination_overhead(msgs, minimum_turns=2) == 1.0

    def test_empty_messages(self):
        assert compute_coordination_overhead([]) == 0.0

    def test_overhead_with_many_agents(self):
        msgs = [_msg(f"a{i}", f"msg{i}") for i in range(5)]
        # 5 turns, 5 unique agents → overhead = 0
        assert compute_coordination_overhead(msgs) == 0.0


# ---- Message Density -------------------------------------------------------

class TestMessageDensity:
    def test_count(self):
        msgs = [_msg("a", "1"), _msg("b", "2"), _msg("c", "3")]
        assert compute_message_density(msgs) == 3.0

    def test_empty(self):
        assert compute_message_density([]) == 0.0


# ---- Redundancy Rate -------------------------------------------------------

class TestRedundancyRate:
    def test_no_redundancy(self):
        msgs = [
            _msg("a", "The quick brown fox jumps over the lazy dog."),
            _msg("b", "Simulation configuration code with parameters and settings."),
        ]
        rate = compute_redundancy_rate(msgs, similarity_method="jaccard")
        assert rate < 0.5

    def test_identical_outputs(self):
        msgs = [
            _msg("a", "Create a box with dimensions 80x60x10"),
            _msg("b", "Create a box with dimensions 80x60x10"),
        ]
        rate = compute_redundancy_rate(msgs, similarity_threshold=0.8, similarity_method="jaccard")
        assert rate == 1.0

    def test_single_message(self):
        assert compute_redundancy_rate([_msg("a", "hello")]) == 0.0

    def test_empty_messages(self):
        assert compute_redundancy_rate([]) == 0.0

    def test_empty_content_ignored(self):
        msgs = [_msg("a", "hello"), _msg("b", ""), _msg("c", "hello")]
        rate = compute_redundancy_rate(msgs, similarity_threshold=0.8, similarity_method="jaccard")
        # msg c is similar to msg a → redundant, but msg b (empty) not counted.
        assert rate > 0.0

    def test_partial_redundancy(self):
        msgs = [
            _msg("a", "Design a box with length 80 width 60 height 10"),
            _msg("b", "Design a box with length 80 width 60 height 10"),  # redundant
            _msg("c", "Execute the simulation code using the tool"),  # unique
        ]
        rate = compute_redundancy_rate(msgs, similarity_threshold=0.8, similarity_method="jaccard")
        # 1 out of 2 (messages after first) is redundant
        assert abs(rate - 0.5) < 0.01


# ---- Coordination Efficiency -----------------------------------------------

class TestCoordinationEfficiency:
    def test_perfect_efficiency(self):
        msgs = [
            _msg("a", "Design plan for aircraft geometry"),
            _msg("b", "Setting aircraft parameters and running simulation"),
        ]
        eff = compute_coordination_efficiency(msgs, similarity_method="jaccard")
        assert eff > 0.5

    def test_all_errors(self):
        msgs = [_msg("a", "SyntaxError: bad code"), _msg("b", "TypeError: wrong type")]
        eff = compute_coordination_efficiency(msgs, similarity_method="jaccard")
        assert eff <= 0.0  # all errors → zero or negative efficiency

    def test_empty(self):
        eff = compute_coordination_efficiency([])
        assert eff == 1.0


# ---- Error Amplification ---------------------------------------------------

class TestErrorAmplification:
    def test_no_errors(self):
        msgs = [_msg("a", "good output"), _msg("b", "also good")]
        assert compute_error_amplification(msgs) == 0

    def test_independent_errors(self):
        msgs = [
            _msg("a", "SyntaxError in code"),
            _msg("b", "TypeError in code"),
        ]
        # Different error types → no amplification.
        assert compute_error_amplification(msgs) == 0

    def test_amplified_error(self):
        msgs = [
            _msg("a", "SyntaxError: invalid syntax on line 5"),
            _msg("b", "The previous SyntaxError caused execution failure"),
        ]
        assert compute_error_amplification(msgs) == 1

    def test_chain_amplification(self):
        msgs = [
            _msg("a", "TypeError in the computation module"),
            _msg("b", "The TypeError propagated to the next stage"),
            _msg("c", "Due to TypeError, the review stage also failed"),
        ]
        assert compute_error_amplification(msgs) == 2

    def test_single_message(self):
        assert compute_error_amplification([_msg("a", "SyntaxError")]) == 0

    def test_empty(self):
        assert compute_error_amplification([]) == 0

    def test_error_in_later_message_only(self):
        msgs = [
            _msg("a", "Good output no errors"),
            _msg("b", "SyntaxError in code"),
        ]
        # No previous error sigs → no amplification.
        assert compute_error_amplification(msgs) == 0


# ---- All-in-one computation ------------------------------------------------

class TestComputeAllMetrics:
    def test_all_metrics_returned(self):
        msgs = [_msg("a", "hello"), _msg("b", "world")]
        m = compute_cross_strategy_metrics(msgs, similarity_method="jaccard")
        assert "coordination_overhead" in m
        assert "message_density" in m
        assert "redundancy_rate" in m
        assert "coordination_efficiency" in m
        assert "error_amplification" in m

    def test_values_consistent(self):
        msgs = [
            _msg("a", "Plan the geometry"),
            _msg("b", "Write the code"),
            _msg("c", "Execute the code"),
        ]
        m = compute_cross_strategy_metrics(msgs, similarity_method="jaccard")
        assert m["message_density"] == 3.0
        assert m["coordination_overhead"] == 0.0
        assert m["error_amplification"] == 0
        assert m["redundancy_rate"] >= 0.0
        assert m["coordination_efficiency"] <= 1.0

    def test_with_explicit_minimum_turns(self):
        msgs = [_msg("a", "1"), _msg("b", "2"), _msg("c", "3")]
        m = compute_cross_strategy_metrics(msgs, minimum_turns=1, similarity_method="jaccard")
        assert m["coordination_overhead"] == 2.0

    def test_empty_messages(self):
        m = compute_cross_strategy_metrics([])
        assert m["message_density"] == 0.0
        assert m["coordination_overhead"] == 0.0
        assert m["coordination_efficiency"] == 1.0
