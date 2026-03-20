"""Metrics for the Iterative Feedback operational methodology.

Computes ambidexterity, escalation of commitment, per-agent aggregates,
per-prompt aggregates, and cross-prompt metrics from attempt histories
produced by IterativeFeedbackHandler.
"""

from __future__ import annotations

from src.coordination.feedback_extraction import AttemptFeedback
from src.coordination.similarity import compute_similarity

# ---- Ambidexterity -----------------------------------------------------------


def compute_ambidexterity(
    attempts: list[AttemptFeedback],
    similarity_method: str = "tfidf",
) -> dict:
    """Measure exploration/exploitation balance across consecutive attempts.

    Returns a dict with mean_similarity, variance, ambidexterity_score,
    dominant_mode, and the list of pairwise similarities.
    """
    if len(attempts) < 2:
        return {"score": None, "mode": "single_attempt"}

    similarities: list[float] = []
    for i in range(len(attempts) - 1):
        sim = compute_similarity(
            attempts[i].output_content,
            attempts[i + 1].output_content,
            method=similarity_method,
        )
        similarities.append(sim)

    mean_sim = sum(similarities) / len(similarities)
    variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)

    return {
        "mean_similarity": mean_sim,
        "variance": variance,
        "ambidexterity_score": variance,
        "dominant_mode": "exploitation" if mean_sim > 0.6 else "exploration",
        "similarities": similarities,
    }


# ---- Escalation of commitment ------------------------------------------------


def compute_escalation(
    attempts: list[AttemptFeedback],
    similarity_threshold: float = 0.8,
    min_length: int = 4,
    similarity_method: str = "tfidf",
) -> dict:
    """Detect escalation of commitment (repeated similar failing attempts).

    Returns escalation_length (longest run of consecutive similar failures)
    and escalation_detected (True if length >= min_length).
    """
    if len(attempts) < 2:
        return {"escalation_length": 0, "escalation_detected": False}

    max_run = 0
    current_run = 0

    for i in range(len(attempts) - 1):
        both_failed = attempts[i].has_tool_errors and attempts[i + 1].has_tool_errors
        high_sim = (
            compute_similarity(
                attempts[i].output_content,
                attempts[i + 1].output_content,
                method=similarity_method,
            )
            > similarity_threshold
        )

        if both_failed and high_sim:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    escalation_length = max_run + 1 if max_run > 0 else 0
    return {
        "escalation_length": escalation_length,
        "escalation_detected": escalation_length >= min_length,
    }


# ---- Per-agent aggregate metrics ---------------------------------------------


def compute_per_agent_metrics(
    attempts: list[AttemptFeedback],
) -> dict:
    """Compute aggregate metrics for one agent's attempt history.

    Returns total_attempts, success_attempt (1-indexed or None),
    retry_rate, unique_error_types, and final_outcome.
    """
    if not attempts:
        return {
            "total_attempts": 0,
            "success_attempt": None,
            "retry_rate": 0.0,
            "unique_error_types": [],
            "final_outcome": "no_attempts",
        }

    total = len(attempts)
    success_attempt: int | None = None
    error_types: set[str] = set()

    for fb in attempts:
        if not fb.has_tool_errors and success_attempt is None:
            success_attempt = fb.attempt_number + 1  # 1-indexed
        for tc in fb.tool_calls:
            if tc.error_type:
                error_types.add(tc.error_type)
        for err in fb.error_messages:
            # Try to extract error type from message text.
            from src.coordination.feedback_extraction import _extract_error_type

            et = _extract_error_type(err)
            if et:
                error_types.add(et)

    last = attempts[-1]
    if success_attempt is not None:
        final_outcome = "success"
    elif total > 1 and not last.has_tool_errors:
        final_outcome = "success"
        success_attempt = total
    else:
        # Check if the last attempt looks like a summary hand-off.
        if "summary" in last.output_content.lower() or total >= 20:
            final_outcome = "summary_handoff"
        else:
            final_outcome = "max_retries_exhausted"

    retry_rate = (total - 1) / max(total, 1)

    return {
        "total_attempts": total,
        "success_attempt": success_attempt,
        "retry_rate": retry_rate,
        "unique_error_types": sorted(error_types),
        "final_outcome": final_outcome,
    }


# ---- Per-prompt aggregate metrics --------------------------------------------


def compute_per_prompt_metrics(
    all_attempt_histories: list[list[AttemptFeedback]],
    similarity_method: str = "tfidf",
) -> dict:
    """Compute prompt-level aggregates across all agents.

    Args:
        all_attempt_histories: List of per-agent attempt histories
            (from IterativeFeedbackHandler.attempt_histories).
        similarity_method: Method for similarity computation.

    Returns:
        Dict of prompt-level metrics.
    """
    agent_metrics = [compute_per_agent_metrics(h) for h in all_attempt_histories]

    total_attempts = sum(m["total_attempts"] for m in agent_metrics)
    first_try = sum(1 for m in agent_metrics if m["total_attempts"] == 1 and m["final_outcome"] == "success")
    required_retries = sum(1 for m in agent_metrics if m["total_attempts"] > 1)
    exhausted = sum(1 for m in agent_metrics if m["final_outcome"] == "max_retries_exhausted")
    handoffs = sum(1 for m in agent_metrics if m["final_outcome"] == "summary_handoff")
    human_interventions = 0  # TODO: track from handler if needed

    success_attempts = [m["success_attempt"] for m in agent_metrics if m["success_attempt"] is not None]
    mean_attempts = sum(success_attempts) / len(success_attempts) if success_attempts else None

    # Per-agent ambidexterity and escalation.
    ambidexterity_scores = []
    escalation_lengths = []
    for history in all_attempt_histories:
        if len(history) >= 2:
            amb = compute_ambidexterity(history, similarity_method)
            if amb.get("ambidexterity_score") is not None:
                ambidexterity_scores.append(amb["ambidexterity_score"])
            esc = compute_escalation(history, similarity_method=similarity_method)
            escalation_lengths.append(esc["escalation_length"])

    return {
        "total_attempts_all_agents": total_attempts,
        "agents_succeeded_first_try": first_try,
        "agents_required_retries": required_retries,
        "agents_exhausted_retries": exhausted,
        "summary_handoffs": handoffs,
        "human_feedback_interventions": human_interventions,
        "mean_attempts_to_success": mean_attempts,
        "per_agent_metrics": agent_metrics,
        "ambidexterity_scores": ambidexterity_scores,
        "escalation_lengths": escalation_lengths,
    }


# ---- Cross-prompt metrics ----------------------------------------------------


def compute_cross_prompt_metrics(
    prompt_metrics_list: list[dict],
) -> dict:
    """Compute cross-prompt aggregate metrics.

    Args:
        prompt_metrics_list: List of per-prompt metric dicts
            (from compute_per_prompt_metrics).

    Returns:
        Dict with omission/commission counts, cross-prompt escalation,
        and mean ambidexterity.
    """
    if not prompt_metrics_list:
        return {
            "omission_error_count": 0,
            "commission_error_count": 0,
            "cross_prompt_escalation": False,
            "mean_ambidexterity_per_prompt": None,
        }

    omission = sum(1 for m in prompt_metrics_list if m.get("agents_exhausted_retries", 0) > 0)
    commission = sum(1 for m in prompt_metrics_list if m.get("summary_handoffs", 0) > 0)

    # Cross-prompt escalation: consecutive prompts with errors.
    max_consecutive_errors = 0
    current_run = 0
    for m in prompt_metrics_list:
        has_problems = m.get("agents_exhausted_retries", 0) > 0 or m.get("summary_handoffs", 0) > 0
        if has_problems:
            current_run += 1
            max_consecutive_errors = max(max_consecutive_errors, current_run)
        else:
            current_run = 0

    # Mean ambidexterity across prompts.
    all_scores = []
    for m in prompt_metrics_list:
        all_scores.extend(m.get("ambidexterity_scores", []))
    mean_amb = sum(all_scores) / len(all_scores) if all_scores else None

    return {
        "omission_error_count": omission,
        "commission_error_count": commission,
        "cross_prompt_escalation": max_consecutive_errors >= 3,
        "consecutive_problem_prompts": max_consecutive_errors,
        "mean_ambidexterity_per_prompt": mean_amb,
    }
