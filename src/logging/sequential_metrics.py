"""Sequential strategy-specific metric computation.

Per-prompt metrics track stage durations, token counts, tool utilization,
tool restriction violations (cross-stage references), interface pass rates,
propagation time, and stage independence scores.

Cross-prompt metrics compute omission/commission errors, escalation of
commitment, ambidexterity proxy, and template comparison.

All metrics are computed from AgentMessage data — no logger or dataclass
changes needed.
"""

from src.coordination.history import AgentMessage


def compute_sequential_metrics(
    messages: list[AgentMessage],
    stage_order: list[str],
    pipeline_template: str = "linear",
    stage_tool_sets: dict[str, list[str]] | None = None,
    interface_results: list[dict] | None = None,
) -> dict:
    """Compute per-prompt sequential metrics from message history.

    Args:
        messages: All AgentMessages from a single prompt/run.
        stage_order: Ordered list of stage names from the pipeline template.
        pipeline_template: Name of the template used (e.g. "linear").
        stage_tool_sets: Mapping of stage name → list of allowed tool names.
            Used to compute tool utilization and cross-stage violations.
        interface_results: List of interface validation results from the
            strategy, each with {stage, valid, interface_output}.

    Returns:
        Dict of sequential-specific metrics.
    """
    if not messages:
        return _empty_sequential_metrics(pipeline_template, len(stage_order))

    if stage_tool_sets is None:
        stage_tool_sets = {}
    if interface_results is None:
        interface_results = []

    # Group messages by stage (agent_name maps to stage name).
    stage_messages: dict[str, list[AgentMessage]] = {}
    for msg in messages:
        stage_messages.setdefault(msg.agent_name, []).append(msg)

    # Per-stage duration: sum of duration_seconds per stage.
    per_stage_duration: dict[str, float] = {}
    for stage_name in stage_order:
        msgs = stage_messages.get(stage_name, [])
        total_dur = sum(
            m.duration_seconds for m in msgs
            if m.duration_seconds is not None
        )
        per_stage_duration[stage_name] = round(total_dur, 4)

    # Per-stage token count: sum of token_count per stage.
    per_stage_tokens: dict[str, int] = {}
    for stage_name in stage_order:
        msgs = stage_messages.get(stage_name, [])
        total_tokens = sum(
            m.token_count for m in msgs if m.token_count is not None
        )
        per_stage_tokens[stage_name] = total_tokens

    # Tool utilization per stage: tools_used / tools_available.
    per_stage_tool_utilization: dict[str, float] = {}
    total_tool_refs = 0
    cross_stage_tool_refs = 0

    for stage_name in stage_order:
        msgs = stage_messages.get(stage_name, [])
        allowed = set(stage_tool_sets.get(stage_name, []))
        is_wildcard = "*" in allowed

        # Collect tools actually used by this stage.
        tools_used: set[str] = set()
        for msg in msgs:
            for tc in msg.tool_calls:
                tools_used.add(tc.tool_name)
                total_tool_refs += 1

                # Cross-stage violation: tool used but not in allowed set.
                if not is_wildcard and tc.tool_name not in allowed:
                    cross_stage_tool_refs += 1

        # Utilization: tools_used / tools_available.
        if is_wildcard:
            # Wildcard means "all tools" — can't compute a ratio meaningfully.
            per_stage_tool_utilization[stage_name] = (
                1.0 if tools_used else 0.0
            )
        elif allowed:
            per_stage_tool_utilization[stage_name] = round(
                len(tools_used & allowed) / len(allowed), 4
            )
        else:
            # No tools allowed — utilization is 0 if no tools used, else violation.
            per_stage_tool_utilization[stage_name] = 0.0

    # Interface pass rate.
    interfaces_checked = len(interface_results)
    interfaces_passed = sum(1 for r in interface_results if r.get("valid"))
    interface_pass_rate = (
        interfaces_passed / interfaces_checked
        if interfaces_checked > 0
        else 0.0
    )

    # Propagation time: first message timestamp to last message timestamp.
    timestamps = [
        m.timestamp for m in messages if m.timestamp is not None
    ]
    propagation_time = (
        max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0
    )

    # Stage independence score: 1 - (cross_stage_refs / total_refs).
    stage_independence = (
        1.0 - (cross_stage_tool_refs / total_tool_refs)
        if total_tool_refs > 0
        else 1.0
    )

    return {
        "stage_count": len(stage_order),
        "pipeline_template": pipeline_template,
        "per_stage_duration": per_stage_duration,
        "per_stage_tokens": per_stage_tokens,
        "per_stage_tool_utilization": per_stage_tool_utilization,
        "tool_restriction_violations": cross_stage_tool_refs,
        "interface_pass_rate": round(interface_pass_rate, 4),
        "propagation_time": round(propagation_time, 4),
        "stage_independence_score": round(stage_independence, 4),
        "total_turns": len(messages),
    }


def compute_cross_prompt_metrics(
    all_runs: list[dict],
) -> dict:
    """Compute cross-prompt metrics accumulated over multiple runs.

    Args:
        all_runs: List of per-run dicts, each with:
            - messages: list[AgentMessage]
            - metrics: dict (output of compute_sequential_metrics)
            - success: bool (whether the task was completed successfully)
            - final_score: float | None (evaluation score, if available)

    Returns:
        Dict of cross-prompt metrics.
    """
    if not all_runs:
        return _empty_cross_prompt_metrics()

    total_prompts = len(all_runs)
    successes = sum(1 for r in all_runs if r.get("success", False))

    # Omission errors: prompts where final_score is below threshold
    # but could have succeeded (proxied by low score).
    omission_threshold = 0.5
    omission_errors = sum(
        1 for r in all_runs
        if r.get("final_score") is not None
        and r["final_score"] < omission_threshold
        and not r.get("success", False)
    )

    # Commission errors: prompts where success=True but final_score is low.
    commission_errors = sum(
        1 for r in all_runs
        if r.get("success", False)
        and r.get("final_score") is not None
        and r["final_score"] < omission_threshold
    )

    # Escalation of commitment: consecutive failed prompts.
    max_consecutive_failures = 0
    current_streak = 0
    for r in all_runs:
        if not r.get("success", False):
            current_streak += 1
            max_consecutive_failures = max(
                max_consecutive_failures, current_streak
            )
        else:
            current_streak = 0

    # Ambidexterity proxy: variance of tool utilization across runs.
    utilizations = []
    for r in all_runs:
        m = r.get("metrics", {})
        util_dict = m.get("per_stage_tool_utilization", {})
        if util_dict:
            utilizations.extend(util_dict.values())

    ambidexterity = 0.0
    if len(utilizations) >= 2:
        mean_u = sum(utilizations) / len(utilizations)
        variance = sum((u - mean_u) ** 2 for u in utilizations) / len(utilizations)
        ambidexterity = round(variance, 4)

    return {
        "total_prompts": total_prompts,
        "success_rate": round(successes / total_prompts, 4) if total_prompts > 0 else 0.0,
        "omission_errors": omission_errors,
        "commission_errors": commission_errors,
        "max_consecutive_failures": max_consecutive_failures,
        "ambidexterity_proxy": ambidexterity,
    }


def compute_template_comparison(
    template_runs: dict[str, list[dict]],
) -> dict:
    """Compare metrics across runs using different pipeline templates.

    Args:
        template_runs: Mapping of template name → list of run dicts.
            Each run dict has: metrics, success, final_score.

    Returns:
        Dict with per-template summary and comparison metrics.
    """
    if not template_runs:
        return {}

    per_template: dict[str, dict] = {}

    for template_name, runs in template_runs.items():
        if not runs:
            continue

        total = len(runs)
        successes = sum(1 for r in runs if r.get("success", False))
        success_rate = successes / total if total > 0 else 0.0

        # Total tokens across all runs for this template.
        total_tokens = 0
        total_propagation = 0.0
        stage_counts = []
        for r in runs:
            m = r.get("metrics", {})
            tokens = m.get("per_stage_tokens", {})
            total_tokens += sum(tokens.values())
            total_propagation += m.get("propagation_time", 0.0)
            stage_counts.append(m.get("stage_count", 0))

        # Template efficiency: success_rate / tokens_per_run.
        avg_tokens = total_tokens / total if total > 0 else 0
        efficiency = success_rate / avg_tokens if avg_tokens > 0 else 0.0

        # Stage bottleneck: stage with highest average duration.
        stage_durations: dict[str, list[float]] = {}
        for r in runs:
            m = r.get("metrics", {})
            for sname, dur in m.get("per_stage_duration", {}).items():
                stage_durations.setdefault(sname, []).append(dur)

        bottleneck = ""
        max_avg_dur = 0.0
        for sname, durs in stage_durations.items():
            avg = sum(durs) / len(durs)
            if avg > max_avg_dur:
                max_avg_dur = avg
                bottleneck = sname

        per_template[template_name] = {
            "runs": total,
            "success_rate": round(success_rate, 4),
            "avg_tokens_per_run": round(avg_tokens, 1),
            "efficiency": round(efficiency, 8),
            "avg_propagation_time": round(
                total_propagation / total, 4
            ) if total > 0 else 0.0,
            "stage_bottleneck": bottleneck,
        }

    return {"per_template": per_template}


def _empty_sequential_metrics(
    pipeline_template: str = "linear",
    stage_count: int = 0,
) -> dict:
    return {
        "stage_count": stage_count,
        "pipeline_template": pipeline_template,
        "per_stage_duration": {},
        "per_stage_tokens": {},
        "per_stage_tool_utilization": {},
        "tool_restriction_violations": 0,
        "interface_pass_rate": 0.0,
        "propagation_time": 0.0,
        "stage_independence_score": 1.0,
        "total_turns": 0,
    }


def _empty_cross_prompt_metrics() -> dict:
    return {
        "total_prompts": 0,
        "success_rate": 0.0,
        "omission_errors": 0,
        "commission_errors": 0,
        "max_consecutive_failures": 0,
        "ambidexterity_proxy": 0.0,
    }
