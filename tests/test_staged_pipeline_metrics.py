"""Unit tests for staged pipeline metrics computation."""


from src.coordination.staged_pipeline_handler import StageResult
from src.logging.staged_pipeline_metrics import (
    compute_cross_prompt_metrics,
    compute_error_propagation,
    compute_per_prompt_metrics,
)

# ---- Helpers ---------------------------------------------------------------

def _sr(
    name: str, index: int, met: bool, *,
    received_failed: bool = False,
    duration: float = 1.0,
    tokens: int = 100,
    tools: list[str] | None = None,
    tools_ok: int = 0,
    tools_err: int = 0,
    output_len: int = 50,
) -> StageResult:
    return StageResult(
        stage_name=name,
        stage_index=index,
        completion_met=met,
        completion_reason="test",
        stage_duration=duration,
        stage_tokens=tokens,
        tools_called=tools or [],
        tools_succeeded=tools_ok,
        tools_failed=tools_err,
        output_length=output_len,
        received_failed_input=received_failed,
    )


# ---- Per-prompt metrics ----------------------------------------------------

class TestPerPromptMetrics:
    def test_empty_results(self):
        m = compute_per_prompt_metrics([])
        assert m["stage_count"] == 0
        assert m["completion_rate"] == 0.0
        assert m["first_failure_stage"] is None
        assert m["propagation_depth"] == 0

    def test_all_stages_met(self):
        results = [
            _sr("s1", 0, True),
            _sr("s2", 1, True),
            _sr("s3", 2, True),
        ]
        m = compute_per_prompt_metrics(results)
        assert m["stage_count"] == 3
        assert m["stages_completed"] == 3
        assert m["completion_rate"] == 1.0
        assert m["first_failure_stage"] is None
        assert m["propagation_depth"] == 0
        assert m["error_propagation_count"] == 0
        assert m["error_recovery_count"] == 0

    def test_one_failure(self):
        results = [
            _sr("s1", 0, True),
            _sr("s2", 1, False),
            _sr("s3", 2, True),
        ]
        m = compute_per_prompt_metrics(results)
        assert m["stages_completed"] == 2
        assert abs(m["completion_rate"] - 2/3) < 0.01
        assert m["first_failure_stage"] == 1

    def test_completion_rate_computed(self):
        results = [
            _sr("s1", 0, True),
            _sr("s2", 1, False),
            _sr("s3", 2, False),
            _sr("s4", 3, True),
        ]
        m = compute_per_prompt_metrics(results)
        assert abs(m["completion_rate"] - 0.5) < 0.01

    def test_error_propagation_count(self):
        results = [
            _sr("s1", 0, True),
            _sr("s2", 1, False),  # first failure
            _sr("s3", 2, False, received_failed=True),  # propagated
            _sr("s4", 3, True, received_failed=True),   # recovered
        ]
        m = compute_per_prompt_metrics(results)
        assert m["error_propagation_count"] == 1
        assert m["error_recovery_count"] == 1

    def test_error_recovery_count(self):
        results = [
            _sr("s1", 0, False),
            _sr("s2", 1, True, received_failed=True),
            _sr("s3", 2, True, received_failed=False),
        ]
        m = compute_per_prompt_metrics(results)
        assert m["error_recovery_count"] == 1

    def test_propagation_depth_consecutive(self):
        results = [
            _sr("s1", 0, True),
            _sr("s2", 1, False),
            _sr("s3", 2, False),
            _sr("s4", 3, False),
        ]
        m = compute_per_prompt_metrics(results)
        assert m["propagation_depth"] == 3  # 3 consecutive failures

    def test_propagation_depth_broken_chain(self):
        results = [
            _sr("s1", 0, False),
            _sr("s2", 1, True),
            _sr("s3", 2, False),
            _sr("s4", 3, False),
        ]
        m = compute_per_prompt_metrics(results)
        assert m["propagation_depth"] == 2  # longest consecutive: s3, s4

    def test_first_failure_stage(self):
        results = [
            _sr("s1", 0, True),
            _sr("s2", 1, True),
            _sr("s3", 2, False),
        ]
        m = compute_per_prompt_metrics(results)
        assert m["first_failure_stage"] == 2

    def test_total_duration_and_tokens(self):
        results = [
            _sr("s1", 0, True, duration=1.5, tokens=200),
            _sr("s2", 1, True, duration=2.5, tokens=300),
        ]
        m = compute_per_prompt_metrics(results)
        assert abs(m["total_duration"] - 4.0) < 0.01
        assert m["total_tokens"] == 500

    def test_per_stage_details(self):
        results = [_sr("s1", 0, True, tools=["tool_a"], tools_ok=1)]
        m = compute_per_prompt_metrics(results)
        assert len(m["per_stage"]) == 1
        ps = m["per_stage"][0]
        assert ps["stage_name"] == "s1"
        assert ps["tools_called"] == ["tool_a"]
        assert ps["tools_succeeded"] == 1

    def test_all_stages_met_propagation_zero(self):
        results = [_sr("s1", 0, True), _sr("s2", 1, True)]
        m = compute_per_prompt_metrics(results)
        assert m["propagation_depth"] == 0
        assert m["first_failure_stage"] is None


# ---- Error propagation analysis --------------------------------------------

class TestErrorPropagation:
    def test_empty_results(self):
        m = compute_error_propagation([])
        assert m["chain_length"] == 0
        assert m["propagation_rate"] == 0.0

    def test_single_success(self):
        m = compute_error_propagation([_sr("s1", 0, True)])
        assert m["chain_length"] == 0

    def test_single_failure(self):
        m = compute_error_propagation([_sr("s1", 0, False)])
        assert m["chain_length"] == 1

    def test_propagation_rate(self):
        results = [
            _sr("s1", 0, False),  # fail
            _sr("s2", 1, False),  # fail after fail → propagated
            _sr("s3", 2, True),   # succeed after fail → recovered
        ]
        m = compute_error_propagation(results)
        # 2 transitions where prev failed: s1→s2 (propagated), s2→s3 (recovered)
        assert abs(m["propagation_rate"] - 0.5) < 0.01
        assert abs(m["recovery_rate"] - 0.5) < 0.01

    def test_recovery_rate(self):
        results = [
            _sr("s1", 0, False),
            _sr("s2", 1, True),   # recovered
        ]
        m = compute_error_propagation(results)
        assert m["recovery_rate"] == 1.0
        assert m["propagation_rate"] == 0.0

    def test_independent_failure_rate(self):
        results = [
            _sr("s1", 0, True),
            _sr("s2", 1, False),  # independent failure (prev succeeded)
            _sr("s3", 2, True),
        ]
        m = compute_error_propagation(results)
        # 1 transition where prev succeeded and current failed out of 1 prev-succeeded transitions
        assert abs(m["independent_failure_rate"] - 1.0) < 0.01

    def test_chain_length(self):
        results = [
            _sr("s1", 0, True),
            _sr("s2", 1, False),
            _sr("s3", 2, False),
            _sr("s4", 3, False),
        ]
        m = compute_error_propagation(results)
        assert m["chain_length"] == 3

    def test_no_failures(self):
        results = [
            _sr("s1", 0, True),
            _sr("s2", 1, True),
            _sr("s3", 2, True),
        ]
        m = compute_error_propagation(results)
        assert m["chain_length"] == 0
        assert m["propagation_rate"] == 0.0
        assert m["recovery_rate"] == 0.0
        assert m["independent_failure_rate"] == 0.0

    def test_all_failures(self):
        results = [
            _sr("s1", 0, False),
            _sr("s2", 1, False),
            _sr("s3", 2, False),
        ]
        m = compute_error_propagation(results)
        assert m["chain_length"] == 3
        assert m["propagation_rate"] == 1.0
        assert m["recovery_rate"] == 0.0


# ---- Cross-prompt metrics --------------------------------------------------

class TestCrossPromptMetrics:
    def test_empty_list(self):
        m = compute_cross_prompt_metrics([])
        assert m["total_prompts"] == 0

    def test_single_prompt_all_met(self):
        pm = {
            "completion_rate": 1.0,
            "propagation_depth": 0,
            "propagation_rate": 0.0,
            "recovery_rate": 0.0,
        }
        m = compute_cross_prompt_metrics([pm])
        assert m["total_prompts"] == 1
        assert m["mean_completion_rate"] == 1.0
        assert m["omission_error_count"] == 0
        assert m["commission_error_count"] == 0

    def test_multiple_prompts(self):
        metrics = [
            {"completion_rate": 1.0, "propagation_depth": 0,
             "propagation_rate": 0.0, "recovery_rate": 0.0},
            {"completion_rate": 0.75, "propagation_depth": 1,
             "propagation_rate": 0.5, "recovery_rate": 0.5},
            {"completion_rate": 0.25, "propagation_depth": 3,
             "propagation_rate": 1.0, "recovery_rate": 0.0},
        ]
        m = compute_cross_prompt_metrics(metrics)
        assert m["total_prompts"] == 3
        assert abs(m["mean_completion_rate"] - (1.0 + 0.75 + 0.25) / 3) < 0.01
        assert abs(m["mean_propagation_depth"] - (0 + 1 + 3) / 3) < 0.01

    def test_omission_low_completion(self):
        metrics = [
            {"completion_rate": 0.25, "propagation_depth": 3,
             "propagation_rate": 0.0, "recovery_rate": 0.0},
        ]
        m = compute_cross_prompt_metrics(metrics)
        assert m["omission_error_count"] == 1

    def test_commission_all_met_but_propagation(self):
        metrics = [
            {"completion_rate": 1.0, "propagation_depth": 2,
             "propagation_rate": 0.0, "recovery_rate": 0.0},
        ]
        m = compute_cross_prompt_metrics(metrics)
        assert m["commission_error_count"] == 1

    def test_mean_error_propagation_rate(self):
        metrics = [
            {"completion_rate": 0.5, "propagation_depth": 1,
             "propagation_rate": 0.8, "recovery_rate": 0.2},
            {"completion_rate": 0.5, "propagation_depth": 1,
             "propagation_rate": 0.4, "recovery_rate": 0.6},
        ]
        m = compute_cross_prompt_metrics(metrics)
        assert abs(m["mean_error_propagation_rate"] - 0.6) < 0.01
        assert abs(m["mean_recovery_rate"] - 0.4) < 0.01
