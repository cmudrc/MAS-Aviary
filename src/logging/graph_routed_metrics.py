"""Metrics computation for the Graph-Routed execution handler.

Computes per-transition, per-prompt aggregate, routing quality, and
cross-prompt metrics from transition history records.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.coordination.graph_routed_handler import TransitionRecord

# ---------------------------------------------------------------------------
# Per-prompt aggregate metrics
# ---------------------------------------------------------------------------


def compute_per_prompt_metrics(
    transitions: list[TransitionRecord],
    initial_complexity: str | None = None,
    final_complexity: str | None = None,
) -> dict[str, Any]:
    """Compute per-prompt aggregate metrics from a transition history.

    Args:
        transitions: List of TransitionRecord from one handler execution.
        initial_complexity: First complexity classification.
        final_complexity: Last complexity classification (after escalations).

    Returns:
        Dict with aggregate metric keys.
    """
    if not transitions:
        return {
            "total_transitions": 0,
            "unique_states_visited": 0,
            "states_visited_histogram": {},
            "cycle_count": 0,
            "escalations_triggered": 0,
            "initial_complexity": initial_complexity,
            "final_complexity": final_complexity,
            "resource_utilization": 0.0,
            "context_utilization": 0.0,
            "path_efficiency": 1.0,
            "error_type_distribution": {},
            "code_review_cycles": 0,
        }

    total = len(transitions)

    # States visited (from_state of each transition + to_state of last).
    visited = [t.from_state for t in transitions]
    visited.append(transitions[-1].to_state)
    unique_visited = set(visited)
    histogram = dict(Counter(visited))

    # Cycle count from the last transition.
    cycle_count = transitions[-1].cycle_count

    # Escalation count: transitions TO an escalation-like state.
    escalations = sum(1 for t in transitions if "ESCALAT" in t.to_state.upper())

    # Resource utilization: passes_used / passes_max.
    # First transition has the max passes_remaining, last has the min.
    if transitions:
        first_passes = transitions[0].passes_remaining
        last_passes = transitions[-1].passes_remaining
        passes_used = first_passes - last_passes
        passes_max = first_passes
        resource_util = passes_used / passes_max if passes_max > 0 else 0.0
    else:
        resource_util = 0.0

    # Context utilization.
    if transitions:
        last_context = transitions[-1].context_used
        # Estimate context budget from resource budgets — use a reasonable default.
        context_budget = 3000  # default
        context_util = last_context / context_budget if context_budget > 0 else 0.0
    else:
        context_util = 0.0

    # Path efficiency: 1 - (cycle_count / total_transitions).
    path_eff = 1.0 - (cycle_count / total) if total > 0 else 1.0

    # Error type distribution — count from_states that are error-related.
    error_types: list[str] = []
    for t in transitions:
        if "ERROR" in t.from_state.upper() or "ERROR" in t.to_state.upper():
            # The condition_matched often contains the error_type info.
            error_types.append(t.condition_matched)
    error_dist = dict(Counter(error_types))

    # Code review cycles: count transitions TO CODE_REVIEWED-like states.
    code_reviews = sum(1 for t in transitions if "REVIEW" in t.to_state.upper() and "OUTPUT" not in t.to_state.upper())

    return {
        "total_transitions": total,
        "unique_states_visited": len(unique_visited),
        "states_visited_histogram": histogram,
        "cycle_count": cycle_count,
        "escalations_triggered": escalations,
        "initial_complexity": initial_complexity,
        "final_complexity": final_complexity,
        "resource_utilization": resource_util,
        "context_utilization": context_util,
        "path_efficiency": path_eff,
        "error_type_distribution": error_dist,
        "code_review_cycles": code_reviews,
    }


# ---------------------------------------------------------------------------
# Routing quality metrics
# ---------------------------------------------------------------------------


def compute_routing_quality(
    transitions: list[TransitionRecord],
    terminal_states: list[str] | None = None,
) -> dict[str, Any]:
    """Compute routing quality metrics from a transition history.

    Args:
        transitions: List of TransitionRecord.
        terminal_states: Names of terminal states for accuracy computation.

    Returns:
        Dict with routing quality metrics.
    """
    terminal = set(terminal_states or ["COMPLETE"])

    if not transitions:
        return {
            "routing_accuracy": 0.0,
            "misroute_rate": 0.0,
            "missed_routes": 0.0,
            "graph_modification_count": 0,
        }

    total = len(transitions)

    # Routing accuracy: transitions leading toward terminal / total.
    # A transition "leads toward terminal" if it doesn't revisit a state.
    visited_before: set[str] = set()
    toward_terminal = 0
    for t in transitions:
        visited_before.add(t.from_state)
        if t.to_state in terminal or t.to_state not in visited_before:
            toward_terminal += 1

    routing_accuracy = toward_terminal / total if total > 0 else 0.0

    # Misroute rate: transitions that go to a previously visited state
    # (excluding terminal states) / total.
    visited_so_far: set[str] = set()
    misroutes = 0
    for t in transitions:
        visited_so_far.add(t.from_state)
        if t.to_state in visited_so_far and t.to_state not in terminal:
            misroutes += 1

    misroute_rate = misroutes / total if total > 0 else 0.0

    # Missed routes: transitions where "always" fallback was used / total.
    always_fallbacks = sum(1 for t in transitions if t.condition_matched == "always")
    missed_routes = always_fallbacks / total if total > 0 else 0.0

    return {
        "routing_accuracy": routing_accuracy,
        "misroute_rate": misroute_rate,
        "missed_routes": missed_routes,
        "graph_modification_count": 0,
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
            "mean_path_length": 0.0,
            "mean_path_efficiency": 0.0,
            "mean_resource_utilization": 0.0,
            "escalation_rate": 0.0,
            "mean_path_length_per_complexity": {},
        }

    total = len(prompt_metrics_list)

    # Omission: prompts with cycle_count > 0 and no terminal reached
    # (heuristic: high cycle count suggests missed good outcomes).
    omissions = sum(1 for m in prompt_metrics_list if m.get("cycle_count", 0) > 3)

    # Commission: prompts where path_efficiency is very low
    # (lots of cycles relative to transitions).
    commissions = sum(1 for m in prompt_metrics_list if m.get("path_efficiency", 1.0) < 0.5)

    # Mean path length.
    path_lengths = [m.get("total_transitions", 0) for m in prompt_metrics_list]
    mean_path = sum(path_lengths) / total if total > 0 else 0.0

    # Mean path efficiency.
    effs = [m.get("path_efficiency", 1.0) for m in prompt_metrics_list]
    mean_eff = sum(effs) / total if total > 0 else 0.0

    # Mean resource utilization.
    utils = [m.get("resource_utilization", 0.0) for m in prompt_metrics_list]
    mean_util = sum(utils) / total if total > 0 else 0.0

    # Escalation rate.
    escalated = sum(1 for m in prompt_metrics_list if m.get("escalations_triggered", 0) > 0)
    escalation_rate = escalated / total if total > 0 else 0.0

    # Mean path length per complexity.
    per_complexity: dict[str, list[int]] = {}
    for m in prompt_metrics_list:
        c = m.get("initial_complexity")
        if c:
            per_complexity.setdefault(c, []).append(m.get("total_transitions", 0))
    mean_per_complexity = {c: sum(v) / len(v) for c, v in per_complexity.items()}

    return {
        "total_prompts": total,
        "omission_error_count": omissions,
        "commission_error_count": commissions,
        "mean_path_length": mean_path,
        "mean_path_efficiency": mean_eff,
        "mean_resource_utilization": mean_util,
        "escalation_rate": escalation_rate,
        "mean_path_length_per_complexity": mean_per_complexity,
    }
