"""Metrics computation for the Staged Pipeline execution handler.

Computes per-stage, per-prompt aggregate, error propagation analysis,
and cross-prompt metrics from stage result records.
"""

from __future__ import annotations

from typing import Any

from src.coordination.staged_pipeline_handler import StageResult

# ---------------------------------------------------------------------------
# Per-prompt aggregate metrics
# ---------------------------------------------------------------------------


def compute_per_prompt_metrics(
    stage_results: list[StageResult],
) -> dict[str, Any]:
    """Compute per-prompt aggregate metrics from one pipeline execution.

    Args:
        stage_results: List of StageResult from one handler execution.

    Returns:
        Dict with aggregate metric keys.
    """
    if not stage_results:
        return {
            "stage_count": 0,
            "stages_completed": 0,
            "completion_rate": 0.0,
            "total_duration": 0.0,
            "total_tokens": 0,
            "error_propagation_count": 0,
            "error_recovery_count": 0,
            "first_failure_stage": None,
            "propagation_depth": 0,
            "per_stage": [],
        }

    stage_count = len(stage_results)
    stages_completed = sum(1 for r in stage_results if r.completion_met)
    completion_rate = stages_completed / stage_count if stage_count > 0 else 0.0

    total_duration = sum(r.stage_duration for r in stage_results)
    total_tokens = sum(r.stage_tokens for r in stage_results)

    # Error propagation: stages where received_failed_input AND completion NOT met.
    error_propagation_count = sum(1 for r in stage_results if r.received_failed_input and not r.completion_met)

    # Error recovery: stages where received_failed_input AND completion MET.
    error_recovery_count = sum(1 for r in stage_results if r.received_failed_input and r.completion_met)

    # First failure stage.
    first_failure_stage: int | None = None
    for r in stage_results:
        if not r.completion_met:
            first_failure_stage = r.stage_index
            break

    # Propagation depth: longest consecutive NOT MET run starting from
    # first failure.
    propagation_depth = _compute_propagation_depth(stage_results)

    # Per-stage details.
    per_stage = [
        {
            "stage_name": r.stage_name,
            "stage_index": r.stage_index,
            "completion_met": r.completion_met,
            "completion_reason": r.completion_reason,
            "stage_duration": r.stage_duration,
            "stage_tokens": r.stage_tokens,
            "tools_called": r.tools_called,
            "tools_succeeded": r.tools_succeeded,
            "tools_failed": r.tools_failed,
            "output_length": r.output_length,
            "received_failed_input": r.received_failed_input,
        }
        for r in stage_results
    ]

    return {
        "stage_count": stage_count,
        "stages_completed": stages_completed,
        "completion_rate": completion_rate,
        "total_duration": total_duration,
        "total_tokens": total_tokens,
        "error_propagation_count": error_propagation_count,
        "error_recovery_count": error_recovery_count,
        "first_failure_stage": first_failure_stage,
        "propagation_depth": propagation_depth,
        "per_stage": per_stage,
    }


def _compute_propagation_depth(stage_results: list[StageResult]) -> int:
    """Count longest consecutive NOT MET run starting from first failure."""
    max_run = 0
    current_run = 0
    for r in stage_results:
        if not r.completion_met:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


# ---------------------------------------------------------------------------
# Error propagation analysis
# ---------------------------------------------------------------------------


def compute_error_propagation(
    stage_results: list[StageResult],
) -> dict[str, Any]:
    """Track how failures cascade through the pipeline.

    Returns:
        chain_length: longest consecutive failure run
        propagation_rate: P(stage N fails | stage N-1 failed)
        recovery_rate: P(stage N succeeds | stage N-1 failed)
        independent_failure_rate: P(stage N fails | stage N-1 succeeded)
    """
    if len(stage_results) < 2:
        return {
            "chain_length": 0 if not stage_results else (0 if stage_results[0].completion_met else 1),
            "propagation_rate": 0.0,
            "recovery_rate": 0.0,
            "independent_failure_rate": 0.0,
        }

    chain_length = _compute_propagation_depth(stage_results)

    # Count conditional probabilities.
    prev_failed_count = 0  # number of stages where prev failed
    propagated_count = 0  # prev failed AND current failed
    recovered_count = 0  # prev failed AND current succeeded

    prev_succeeded_count = 0  # number of stages where prev succeeded
    independent_fail_count = 0  # prev succeeded AND current failed

    for i in range(1, len(stage_results)):
        prev_met = stage_results[i - 1].completion_met
        curr_met = stage_results[i].completion_met

        if not prev_met:
            prev_failed_count += 1
            if not curr_met:
                propagated_count += 1
            else:
                recovered_count += 1
        else:
            prev_succeeded_count += 1
            if not curr_met:
                independent_fail_count += 1

    propagation_rate = propagated_count / prev_failed_count if prev_failed_count > 0 else 0.0
    recovery_rate = recovered_count / prev_failed_count if prev_failed_count > 0 else 0.0
    independent_failure_rate = independent_fail_count / prev_succeeded_count if prev_succeeded_count > 0 else 0.0

    return {
        "chain_length": chain_length,
        "propagation_rate": propagation_rate,
        "recovery_rate": recovery_rate,
        "independent_failure_rate": independent_failure_rate,
    }


# ---------------------------------------------------------------------------
# Cross-prompt metrics
# ---------------------------------------------------------------------------


def compute_cross_prompt_metrics(
    prompt_metrics_list: list[dict],
) -> dict[str, Any]:
    """Compute cross-prompt aggregate metrics.

    Args:
        prompt_metrics_list: List of per-prompt metric dicts.

    Returns:
        Dict with cross-prompt aggregate metrics.
    """
    if not prompt_metrics_list:
        return {
            "total_prompts": 0,
            "omission_error_count": 0,
            "commission_error_count": 0,
            "mean_completion_rate": 0.0,
            "mean_propagation_depth": 0.0,
            "mean_error_propagation_rate": 0.0,
            "mean_recovery_rate": 0.0,
        }

    total = len(prompt_metrics_list)

    # Omission: prompts where completion_rate < 0.5 (many stages failed).
    omissions = sum(1 for m in prompt_metrics_list if m.get("completion_rate", 1.0) < 0.5)

    # Commission: prompts where completion_rate == 1.0 but we know from
    # external eval that the result is bad. Since we don't have external
    # eval, approximate as prompts where all completed but propagation
    # depth > 0 (some error chaining happened then recovered).
    commissions = sum(
        1 for m in prompt_metrics_list if m.get("completion_rate", 0.0) >= 1.0 and m.get("propagation_depth", 0) > 0
    )

    # Mean completion rate.
    rates = [m.get("completion_rate", 0.0) for m in prompt_metrics_list]
    mean_rate = sum(rates) / total

    # Mean propagation depth.
    depths = [m.get("propagation_depth", 0) for m in prompt_metrics_list]
    mean_depth = sum(depths) / total

    # Mean error propagation rate (from error_propagation analysis).
    prop_rates = [m.get("propagation_rate", 0.0) for m in prompt_metrics_list]
    mean_prop = sum(prop_rates) / total

    # Mean recovery rate.
    rec_rates = [m.get("recovery_rate", 0.0) for m in prompt_metrics_list]
    mean_rec = sum(rec_rates) / total

    return {
        "total_prompts": total,
        "omission_error_count": omissions,
        "commission_error_count": commissions,
        "mean_completion_rate": mean_rate,
        "mean_propagation_depth": mean_depth,
        "mean_error_propagation_rate": mean_prop,
        "mean_recovery_rate": mean_rec,
    }
