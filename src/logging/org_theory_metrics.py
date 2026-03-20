"""Organizational theory metrics extracted from AgentMessage lists.

compute_org_theory_metrics(messages, os_name, handler_name, config) -> dict
dispatches to the appropriate subset of metric functions based on OS and
handler, then returns a flat dict of computed values.

Design principles
-----------------
- Work with what is actually present in the batch results (see MISSING_FIELDS.md).
- If a metric CANNOT be computed, set it to None and append an explanatory
  string to the "warnings" list in the returned dict.
- Never crash on missing or unexpected data.
- Use src.coordination.similarity for all cosine/Jaccard similarity.
"""

from __future__ import annotations

import re
from typing import Any

from src.coordination.similarity import compute_similarity

# ---------------------------------------------------------------------------
# Message field accessor (handles AgentMessage objects AND serialised dicts)
# ---------------------------------------------------------------------------

def _get(msg: Any, field: str, default: Any = None) -> Any:
    """Return ``msg.field`` or ``msg[field]``, falling back to *default*."""
    if isinstance(msg, dict):
        return msg.get(field, default)
    return getattr(msg, field, default)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_HIGH_CONVERGENCE_THRESHOLD = 0.6
_DUPLICATE_SIMILARITY_THRESHOLD = 0.8
_ESCALATION_SIMILARITY_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Metadata accessors (Tier 1: read from msg.metadata; Tier 2: tool_calls)
# ---------------------------------------------------------------------------

def _meta(msg: Any, key: str, default: Any = None) -> Any:
    """Read a value from msg.metadata (dict or attr)."""
    md = _get(msg, "metadata", {})
    if isinstance(md, dict):
        return md.get(key, default)
    return getattr(md, key, default)


def _sum_meta_field(messages: list, field: str) -> int | None:
    """Sum a numeric metadata field across messages. None if absent from all."""
    total = 0
    found = False
    for m in messages:
        val = _meta(m, field)
        if val is not None:
            total += val
            found = True
    return total if found else None


def _max_meta_field(messages: list, field: str) -> int | None:
    """Max of a numeric metadata field across messages. None if absent from all."""
    vals = [_meta(m, field) for m in messages]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _count_tool_calls_by_name(messages: list, tool_name: str) -> int:
    """Count tool_calls matching *tool_name* across all messages."""
    count = 0
    for m in messages:
        tcs = _get(m, "tool_calls") or []
        for tc in tcs:
            if hasattr(tc, "tool_name"):
                name = tc.tool_name
            elif isinstance(tc, dict):
                name = tc.get("tool_name", "")
            else:
                name = ""
            if name == tool_name:
                count += 1
    return count


def _contents(messages: list) -> list[str]:
    """Return non-empty content strings from a message list."""
    return [c for c in (_get(m, "content", "") for m in messages) if c and c.strip()]


def _durations(messages: list) -> list[float]:
    return [_get(m, "duration_seconds", 0.0) for m in messages]


def _pairwise_similarities(texts: list[str], method: str = "tfidf") -> list[float]:
    """Return all pairwise cosine similarities for a list of texts."""
    sims: list[float] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sims.append(compute_similarity(texts[i], texts[j], method=method))
    return sims


def _duplicate_work_rate(texts: list[str], threshold: float = _DUPLICATE_SIMILARITY_THRESHOLD) -> float | None:
    """Fraction of texts that have similarity > threshold to at least one other text."""
    if len(texts) < 2:
        return None
    flagged = 0
    for i, a in enumerate(texts):
        for j, b in enumerate(texts):
            if i != j and compute_similarity(a, b) > threshold:
                flagged += 1
                break
    return flagged / len(texts)


def _convergence_classification(score: float | None, eval_success: bool | None) -> str | None:
    """Map (convergence_score, eval_success) to a named classification."""
    if score is None:
        return None
    high = score >= _HIGH_CONVERGENCE_THRESHOLD
    if eval_success is None:
        # Unknown eval — still classify convergence direction
        return "high_convergence" if high else "low_convergence"
    if high and not eval_success:
        return "joint_myopia"
    if high and eval_success:
        return "effective_consensus"
    if not high and not eval_success:
        return "incoherent_team"
    return "healthy_diversity"


# ---------------------------------------------------------------------------
# Orchestrated OS metrics  (Span of Control, Oversight, Info Asymmetry)
# ---------------------------------------------------------------------------

def _orchestrated_os_metrics(
    messages: list,
    config: dict,
    eval_success: bool | None,
    warnings: list[str],
) -> dict:
    """Extract orchestrated-OS metrics from an AgentMessage list."""
    total = len(messages)
    orch_msgs = [m for m in messages if _get(m, "agent_name") == "orchestrator"]
    worker_msgs = [m for m in messages if _get(m, "agent_name") != "orchestrator"]
    unique_workers = set(_get(m, "agent_name") for m in worker_msgs)

    # --- Span of Control ---
    agents_spawned = len(unique_workers)

    orchestrator_turns = len(orch_msgs)
    worker_turns = len(worker_msgs)
    orchestrator_overhead_ratio = orchestrator_turns / total if total else 0.0

    # orchestrator_token_growth: needs token_count (always None)
    warnings.append(
        "orchestrator_token_growth: token_count is always null in current messages"
    )

    # cost_per_agent: mean orchestrator wall-time per spawned worker
    orch_time = sum(_get(m, "duration_seconds", 0.0) for m in orch_msgs)
    cost_per_agent = orch_time / agents_spawned if agents_spawned else None

    # reasoning_iterations: orchestrator turns before the first worker turn
    first_worker_pos = next(
        (i for i, m in enumerate(messages) if _get(m, "agent_name") != "orchestrator"),
        total,
    )
    reasoning_iterations = sum(
        1 for m in messages[:first_worker_pos]
        if _get(m, "agent_name") == "orchestrator"
    )

    # --- Oversight ---
    # authority_holder: "orchestrator" by definition
    authority_holder = "orchestrator"
    # authority_transfers: requires metadata.event (not present)
    warnings.append(
        "authority_transfers: requires metadata.event='authority_transfer', not present in AgentMessage"
    )

    # --- Information Asymmetry ---
    # information_ratio: requires per-message token counts (always None)
    warnings.append(
        "information_ratio: requires token_count on messages, always null in current runs"
    )

    return {
        # Span of Control
        "agents_spawned": agents_spawned,
        "orchestrator_turns": orchestrator_turns,
        "worker_turns": worker_turns,
        "orchestrator_overhead_ratio": round(orchestrator_overhead_ratio, 4),
        "orchestrator_token_growth": None,
        "cost_per_agent": round(cost_per_agent, 3) if cost_per_agent is not None else None,
        "reasoning_iterations": reasoning_iterations,
        # Oversight
        "authority_holder": authority_holder,
        "authority_transfers": None,
        # Information Asymmetry
        "information_ratio": None,
    }


# ---------------------------------------------------------------------------
# Networked OS metrics  (Common Ground, Peer Monitoring, Joint Myopia, …)
# ---------------------------------------------------------------------------

_PEER_NAME_RE = re.compile(r"^agent_\d+$")


def _networked_os_metrics(
    messages: list,
    config: dict,
    eval_success: bool | None,
    warnings: list[str],
) -> dict:
    """Extract networked-OS metrics from an AgentMessage list."""
    # Match peer agents by name (agent_1, agent_2, …) OR by metadata.peer_agent
    # (graph-driven mode uses role names like mission_architect but tags the
    # actual peer in metadata).
    peer_msgs = [
        m for m in messages
        if _PEER_NAME_RE.match(_get(m, "agent_name", ""))
        or (_get(m, "metadata", {}) or {}).get("peer_agent")
    ]
    # Count unique actual peers (prefer metadata.peer_agent over agent_name).
    unique_peers = set()
    for m in peer_msgs:
        meta = _get(m, "metadata", {}) or {}
        unique_peers.add(meta.get("peer_agent") or _get(m, "agent_name"))
    agents_spawned = len(unique_peers)

    peer_texts = _contents(peer_msgs)

    # --- Common Ground (blackboard) ---
    # Tier 1: metadata fields (populated after handler metadata pass-through).
    bb_writes = _sum_meta_field(peer_msgs, "blackboard_writes")
    bb_size_final = _max_meta_field(peer_msgs, "blackboard_size")
    claim_conflicts_total = _sum_meta_field(peer_msgs, "claim_conflicts")

    # Tier 2 fallback: infer from tool_calls when metadata is absent.
    if bb_writes is None:
        # Each write_blackboard call = 1 write.  Each turn also has a
        # structural auto-write (prior context injection) + 1 initial entry.
        explicit_writes = _count_tool_calls_by_name(peer_msgs, "write_blackboard")
        mark_done = _count_tool_calls_by_name(peer_msgs, "mark_task_done")
        structural_writes = len(peer_msgs)  # 1 auto-write per turn
        bb_writes = explicit_writes + mark_done + structural_writes + 1  # +1 initial task entry
        warnings.append(
            "blackboard_writes_total: estimated from tool_calls (Tier 2 fallback)"
        )

    if bb_size_final is None:
        # Estimate board size: unique agent names + 1 task entry.
        bb_size_final = agents_spawned + 1
        warnings.append(
            "blackboard_size_final: estimated from unique agents (Tier 2 fallback)"
        )

    # Blackboard reads from tool_calls (Tier 2 — no metadata for reads yet).
    bb_reads = _count_tool_calls_by_name(peer_msgs, "read_blackboard")
    if bb_reads == 0:
        # Each agent implicitly reads at start of turn.
        bb_reads = len(peer_msgs)
        warnings.append(
            "blackboard_reads_total: estimated from turn count (Tier 2 fallback)"
        )

    # Blackboard utilization: writes / (agents * turns).
    n_turns = len(peer_msgs)
    bb_utilization = (
        round(bb_writes / (agents_spawned * n_turns), 4)
        if agents_spawned and n_turns else None
    )

    # --- Peer Monitoring ---
    if claim_conflicts_total is None:
        claim_conflicts_total = 0
        warnings.append(
            "claim_conflicts: no metadata.claim_conflicts found, defaulting to 0"
        )

    # --- Joint Myopia ---
    convergence_score: float | None = None
    if len(peer_texts) >= 2:
        sims = _pairwise_similarities(peer_texts)
        convergence_score = sum(sims) / len(sims) if sims else None

    convergence_cls = _convergence_classification(convergence_score, eval_success)

    # --- Predictive Knowledge ---
    for metric in ("prediction_count", "prediction_accuracy_mean"):
        warnings.append(
            f"{metric}: requires metadata.prediction_made/accuracy, not present in AgentMessage"
        )

    # --- Self-Organization ---
    # agents_spawned: count unique "agent_N" names ✓
    # self_selection_diversity: needs structured subtask descriptions — not available
    warnings.append(
        "self_selection_diversity: requires structured subtask descriptions, "
        "not derivable from message content alone"
    )

    dup_rate = _duplicate_work_rate(peer_texts)

    return {
        # Common Ground
        "blackboard_utilization": bb_utilization,
        "blackboard_size_final": bb_size_final,
        "blackboard_reads_total": bb_reads,
        "blackboard_writes_total": bb_writes,
        # Peer Monitoring
        "claim_conflicts": claim_conflicts_total,
        # Joint Myopia
        "convergence_score": round(convergence_score, 4) if convergence_score is not None else None,
        "convergence_classification": convergence_cls,
        # Predictive Knowledge
        "prediction_count": None,
        "prediction_accuracy_mean": None,
        # Self-Organization
        "agents_spawned": agents_spawned,
        "self_selection_diversity": None,
        "duplicate_work_rate": round(dup_rate, 4) if dup_rate is not None else None,
    }


# ---------------------------------------------------------------------------
# Sequential OS metrics  (Decomposition, Modularity, Viscosity, Mirroring)
# ---------------------------------------------------------------------------

def _sequential_os_metrics(
    messages: list,
    config: dict,
    warnings: list[str],
) -> dict:
    """Extract sequential-OS metrics from an AgentMessage list."""
    # Preserve stage order from turn sequence
    seen: list[str] = []
    stage_dur: dict[str, float] = {}
    for m in messages:
        name = _get(m, "agent_name", "")
        if name not in seen:
            seen.append(name)
        stage_dur[name] = stage_dur.get(name, 0.0) + _get(m, "duration_seconds", 0.0)

    stage_names = seen
    per_stage_duration = [round(stage_dur[s], 3) for s in stage_names]
    stage_bottleneck = (
        stage_names[per_stage_duration.index(max(per_stage_duration))]
        if stage_names else None
    )

    # Propagation time: last timestamp minus first
    timestamps = [
        _get(m, "timestamp", 0.0) for m in messages
        if _get(m, "timestamp") is not None
    ]
    propagation_time = (
        round(max(timestamps) - min(timestamps), 3) if len(timestamps) > 1 else None
    )

    # Mirroring: pipeline template name from config
    template_used = None
    if isinstance(config, dict):
        template_used = config.get("pipeline_template")
    elif hasattr(config, "pipeline_template"):
        template_used = config.pipeline_template

    # --- Modularity metrics (from tool_calls + allowed_tools config) ---
    # Extract tool names used per stage (excluding final_answer which is
    # a framework tool, not a domain tool).
    stage_tools: dict[str, list[str]] = {}
    for m in messages:
        sname = _get(m, "agent_name", "")
        tcs = _get(m, "tool_calls") or []
        tools_used: list[str] = []
        for tc in tcs:
            tname = (
                tc.tool_name if hasattr(tc, "tool_name")
                else (tc.get("tool_name", tc.get("name", "")) if isinstance(tc, dict) else "")
            )
            if tname and tname != "final_answer":
                tools_used.append(tname)
        stage_tools.setdefault(sname, []).extend(tools_used)

    # Build tool_utilization_per_stage — {stage: [unique tools]}
    tool_utilization: dict[str, list[str]] | None = None
    stage_independence: float | None = None
    tool_violations: int | None = None

    has_any_tools = any(bool(v) for v in stage_tools.values())
    if has_any_tools:
        tool_utilization = {s: sorted(set(stage_tools.get(s, []))) for s in stage_names}

    # allowed_tools per stage from config (set by batch_runner)
    # Format: {"_stage_allowed_tools": {"stage_name": ["tool_a", "tool_b"], ...}}
    stage_allowed: dict[str, list[str]] | None = None
    if isinstance(config, dict):
        stage_allowed = config.get("_stage_allowed_tools")

    if stage_allowed is not None and stage_names:
        # tool_restriction_violations: count stages that used tools not in
        # their allowed list. Stages with allowed_tools=["*"] allow anything.
        violations = 0
        for sname in stage_names:
            allowed = stage_allowed.get(sname)
            if allowed is None:
                continue  # no restriction info for this stage
            if "*" in allowed:
                continue  # wildcard — anything allowed
            used = set(stage_tools.get(sname, []))
            allowed_set = set(allowed)
            if used - allowed_set:
                violations += 1
        tool_violations = violations

        # stage_independence_score: 1 - (violations / total_stages)
        # A stage that uses tools outside its allowed set is "dependent" —
        # it's reaching beyond its boundary.
        stage_independence = round(1.0 - (violations / len(stage_names)), 3)
    elif tool_utilization is not None:
        # No allowed_tools config — compute from tool overlap instead.
        # Cross-stage overlap = fraction of stages sharing tools with another.
        if len(stage_names) > 1:
            all_stage_sets = [set(stage_tools.get(s, [])) for s in stage_names]
            cross_refs = 0
            total_refs = 0
            for i, tools_i in enumerate(all_stage_sets):
                total_refs += len(tools_i)
                for t in tools_i:
                    # Check if any earlier stage also used this tool
                    for j in range(i):
                        if t in all_stage_sets[j]:
                            cross_refs += 1
                            break
            stage_independence = round(1.0 - (cross_refs / total_refs), 3) if total_refs > 0 else 1.0
        else:
            stage_independence = 1.0

    # per_stage_tokens from token_count on messages
    per_stage_tokens: list[int | None] | None = None
    token_values = []
    has_tokens = False
    for s in stage_names:
        stage_msgs = [m for m in messages if _get(m, "agent_name", "") == s]
        total = sum(_get(m, "token_count", 0) or 0 for m in stage_msgs)
        if any(_get(m, "token_count") is not None for m in stage_msgs):
            has_tokens = True
        token_values.append(total if total > 0 else None)
    if has_tokens:
        per_stage_tokens = token_values
    else:
        warnings.append("per_stage_tokens: token_count not available on messages")

    if tool_utilization is None and not has_any_tools:
        warnings.append(
            "tool_utilization_per_stage: no tool_calls found on messages"
        )
    if stage_independence is None:
        warnings.append(
            "stage_independence_score: requires tool_calls or _stage_allowed_tools in config"
        )
    if tool_violations is None and stage_allowed is None:
        warnings.append(
            "tool_restriction_violations: _stage_allowed_tools not in config; "
            "pass stage restrictions to enable"
        )

    return {
        # Decomposition
        "stage_count": len(stage_names),
        "per_stage_duration": per_stage_duration,
        "per_stage_tokens": per_stage_tokens,
        "stage_bottleneck": stage_bottleneck,
        # Modularity
        "tool_utilization_per_stage": tool_utilization,
        "stage_independence_score": stage_independence,
        "tool_restriction_violations": tool_violations,
        # Viscosity
        "propagation_time": propagation_time,
        "per_stage_propagation": per_stage_duration,
        # Mirroring
        "template_used": template_used,
    }


# ---------------------------------------------------------------------------
# Iterative Feedback handler metrics  (Aspiration, Ambidexterity, Escalation)
# ---------------------------------------------------------------------------

def _iterative_feedback_metrics(
    messages: list,
    config: dict,
    warnings: list[str],
) -> dict:
    """Extract iterative-feedback handler metrics from an AgentMessage list.

    Retry information is obtained from the ``is_retry`` attribute on live
    ``AgentMessage`` objects. This field is not serialised to
    batch_summary.json, so when called from batch_runner with live objects,
    retry counts are accurate; when loaded from JSON they default to False
    and all agents appear to have taken one attempt.
    """
    # Build per-agent attempt lists (ordered by turn_number)
    agent_attempts: dict[str, list[str]] = {}
    agent_errors: dict[str, list[bool]] = {}

    for m in sorted(messages, key=lambda x: _get(x, "turn_number", 0)):
        name = _get(m, "agent_name", "")
        content = _get(m, "content", "")
        error = _get(m, "error")
        if name not in agent_attempts:
            agent_attempts[name] = []
            agent_errors[name] = []
        agent_attempts[name].append(content)
        agent_errors[name].append(error is not None)

    # Attempt counts
    attempt_counts = {name: len(v) for name, v in agent_attempts.items()}
    max_attempts_used = max(attempt_counts.values()) if attempt_counts else 0

    # mean_attempts_to_success per agent — only meaningful when retries occurred
    has_retries = any(c > 1 for c in attempt_counts.values())
    mean_attempts = (
        sum(attempt_counts.values()) / len(attempt_counts)
        if attempt_counts else None
    )
    if not has_retries:
        warnings.append(
            "mean_attempts_to_success: no retries detected "
            "(all agents succeeded on first attempt or retries not captured in messages)"
        )

    # Aspiration mode: prefer message metadata, fallback to config.
    aspiration_mode = None
    if messages:
        aspiration_mode = _meta(messages[0], "aspiration_mode")
    if aspiration_mode is None and isinstance(config, dict):
        aspiration_mode = config.get("aspiration_mode")

    # --- Ambidexterity ---
    agent_ambidexterity: list[float] = []
    for name, attempts in agent_attempts.items():
        if len(attempts) < 2:
            continue
        sims = [
            compute_similarity(attempts[i], attempts[i + 1])
            for i in range(len(attempts) - 1)
        ]
        mean_sim = sum(sims) / len(sims)
        variance = sum((s - mean_sim) ** 2 for s in sims) / len(sims)
        agent_ambidexterity.append(variance)

    ambidexterity_score = (
        round(sum(agent_ambidexterity) / len(agent_ambidexterity), 4)
        if agent_ambidexterity else None
    )

    # --- Escalation of Commitment ---
    # Longest consecutive run where output similarity > 0.8 AND the current
    # attempt still fails (has a next attempt, meaning aspiration not met).
    escalation_length = 0
    for name, attempts in agent_attempts.items():
        if len(attempts) < 2:
            continue
        run = 0
        for i in range(len(attempts) - 1):
            sim = compute_similarity(attempts[i], attempts[i + 1])
            # A next attempt exists → the current attempt didn't satisfy aspiration
            if sim > _ESCALATION_SIMILARITY_THRESHOLD:
                run += 1
            else:
                run = 0
            escalation_length = max(escalation_length, run)

    escalation_detected = escalation_length > 3

    # early_stopping_count: requires eval per-attempt, not available
    warnings.append(
        "early_stopping_count: requires per-attempt eval data, set to null"
    )

    return {
        # Aspiration Levels
        "aspiration_mode": aspiration_mode,
        "mean_attempts_to_success": round(mean_attempts, 3) if mean_attempts is not None else None,
        "max_attempts_used": max_attempts_used,
        # Ambidexterity
        "ambidexterity_score": ambidexterity_score,
        # Escalation of Commitment
        "escalation_length": escalation_length,
        "escalation_detected": escalation_detected,
        # False Negative Avoidance
        "early_stopping_count": None,
    }


# ---------------------------------------------------------------------------
# Graph-Routed handler metrics  (Attention, Omission/Commission, Coupled Search)
# ---------------------------------------------------------------------------

def _graph_routed_metrics(
    messages: list,
    config: dict,
    warnings: list[str],
) -> dict:
    """Extract graph-routed handler metrics from an AgentMessage list.

    State transitions are inferred from agent_name changes in the message
    sequence. Graph-routed messages have agent names matching graph role names
    (e.g. "classifier", "coder", "executor").
    """
    # State sequence from agent_names (preserves transitions including revisits)
    state_sequence = [_get(m, "agent_name", "") for m in messages]
    total_transitions = max(len(state_sequence) - 1, 0)

    # Count times each state was visited
    state_hist: dict[str, int] = {}
    for s in state_sequence:
        state_hist[s] = state_hist.get(s, 0) + 1

    # Transitions toward COMPLETE: agent_name == "output_reviewer" heuristic
    terminal_states = {"output_reviewer", "COMPLETE"}
    toward_complete = sum(
        1 for s in state_sequence if s in terminal_states
    )
    routing_accuracy = (
        toward_complete / total_transitions if total_transitions else None
    )

    # Misroute_rate: transitions that returned to a previously-visited state
    misroute_count = 0
    visited: set[str] = set()
    for s in state_sequence:
        if s in visited:
            misroute_count += 1
        visited.add(s)
    misroute_rate = misroute_count / total_transitions if total_transitions else None

    # Missed routes: we cannot detect "always" fallback without graph definition data
    warnings.append(
        "missed_routes: requires access to graph transition records, "
        "set to null (infer from misroute_rate)"
    )

    # --- Structural Distribution of Attention ---
    # Extract resource state from message metadata (exported by GraphRoutedHandler).
    initial_complexity = None
    final_complexity = None
    resource_utilization = None
    context_utilization = None
    complexity_escalations = 0
    complexity_budget = None
    budget_constrained = None

    complexities_seen: list[str] = []
    last_passes_remaining = None
    last_passes_max = None
    last_context_used = None
    last_context_budget = None

    for m in messages:
        meta = _get(m, "metadata", {}) or {}
        cpx = meta.get("complexity")
        if cpx:
            if initial_complexity is None:
                initial_complexity = cpx
            final_complexity = cpx
            if complexities_seen and cpx != complexities_seen[-1]:
                complexity_escalations += 1
            complexities_seen.append(cpx)

        if meta.get("passes_remaining") is not None:
            last_passes_remaining = meta["passes_remaining"]
        if meta.get("passes_max") is not None:
            last_passes_max = meta["passes_max"]
        if meta.get("context_used") is not None:
            last_context_used = meta["context_used"]
        if meta.get("context_budget") is not None:
            last_context_budget = meta["context_budget"]

    if last_passes_max and last_passes_remaining is not None:
        passes_used = last_passes_max - last_passes_remaining
        resource_utilization = round(passes_used / last_passes_max, 4)
    if last_context_budget and last_context_used is not None:
        context_utilization = round(last_context_used / last_context_budget, 4)
    if last_passes_max:
        complexity_budget = last_passes_max
    if last_passes_remaining is not None:
        budget_constrained = last_passes_remaining <= 0

    if initial_complexity is None:
        warnings.append(
            "initial_complexity: 'complexity' not found in message metadata; "
            "ensure GraphRoutedHandler >= v2 is used"
        )

    # Internal representations toggle from config
    mental_model_enabled = None
    if isinstance(config, dict):
        mental_model_enabled = config.get("mental_model_enabled")

    return {
        # Structural Distribution of Attention
        "initial_complexity": initial_complexity,
        "final_complexity": final_complexity,
        "resource_utilization": resource_utilization,
        "context_utilization": context_utilization,
        "complexity_escalations": complexity_escalations if complexities_seen else None,
        # Omission vs Commission
        "total_transitions": total_transitions,
        "misroute_rate": round(misroute_rate, 4) if misroute_rate is not None else None,
        "missed_routes": None,
        "routing_accuracy": round(routing_accuracy, 4) if routing_accuracy is not None else None,
        # Coupled Search
        "complexity_budget": complexity_budget,
        "budget_constrained": budget_constrained,
        # Internal Representations
        "mental_model_enabled": mental_model_enabled,
    }


# ---------------------------------------------------------------------------
# Staged Pipeline handler metrics  (Decomposition, Aspiration, Error Propagation)
# ---------------------------------------------------------------------------

def _staged_pipeline_metrics(
    messages: list,
    config: dict,
    warnings: list[str],
) -> dict:
    """Extract staged-pipeline handler metrics from an AgentMessage list.

    Stage completion is heuristically determined: a stage is considered to
    have 'met criteria' if its output is non-empty and contains no error
    patterns. Error propagation is inferred from the error field.
    """
    # Each message is one pipeline stage execution
    stages = sorted(messages, key=lambda m: _get(m, "turn_number", 0))
    stage_names = [_get(m, "agent_name", "") for m in stages]
    stage_count = len(stages)

    per_stage_duration = [round(_get(m, "duration_seconds", 0.0), 3) for m in stages]

    # Heuristic completion: non-empty output with no error
    _error_pats = re.compile(
        r"(?:Error|Exception|Traceback|FAILED|failed|compilation fail|"
        r"SyntaxError|IndentationError)",
        re.IGNORECASE,
    )

    def _stage_met(m) -> bool:
        content = _get(m, "content", "") or ""
        err = _get(m, "error")
        if err:
            return False
        if not content.strip():
            return False
        if _error_pats.search(content):
            return False
        return True

    stage_results = [_stage_met(m) for m in stages]
    completion_rate = (
        sum(stage_results) / stage_count if stage_count else None
    )

    per_stage_completion = [
        {"stage": name, "met": met, "reason": "heuristic"}
        for name, met in zip(stage_names, stage_results)
    ]

    # Error propagation analysis
    # error_propagation_count: stages that received failed input AND also failed
    error_propagation_count = 0
    prev_failed = False
    first_failure_stage: int | None = None
    propagation_depth = 0
    current_depth = 0

    propagation_rate_numerator = 0
    propagation_rate_denominator = 0
    recovery_rate_numerator = 0
    recovery_rate_denominator = 0

    for i, met in enumerate(stage_results):
        if not met and first_failure_stage is None:
            first_failure_stage = i

        if prev_failed:
            propagation_rate_denominator += 1
            recovery_rate_denominator += 1
            if not met:
                # P(fail | prev failed)
                propagation_rate_numerator += 1
                error_propagation_count += 1
                current_depth += 1
                propagation_depth = max(propagation_depth, current_depth)
            else:
                # P(success | prev failed)
                recovery_rate_numerator += 1
                current_depth = 0
        else:
            current_depth = 0

        prev_failed = not met

    propagation_rate = (
        propagation_rate_numerator / propagation_rate_denominator
        if propagation_rate_denominator else None
    )
    recovery_rate = (
        recovery_rate_numerator / recovery_rate_denominator
        if recovery_rate_denominator else None
    )

    # per_stage_tokens: not available
    warnings.append(
        "per_stage_tokens (staged_pipeline): token_count always null"
    )

    return {
        # Decomposition
        "stage_count": stage_count,
        "per_stage_duration": per_stage_duration,
        "per_stage_tokens": None,
        # Aspiration Levels (completion gates)
        "completion_rate": round(completion_rate, 4) if completion_rate is not None else None,
        "per_stage_completion": per_stage_completion,
        # Error Propagation
        "propagation_rate": round(propagation_rate, 4) if propagation_rate is not None else None,
        "recovery_rate": round(recovery_rate, 4) if recovery_rate is not None else None,
        "propagation_depth": propagation_depth,
        "first_failure_stage": first_failure_stage,
        "error_propagation_count": error_propagation_count,
    }


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def compute_org_theory_metrics(
    messages: list,
    os_name: str,
    handler_name: str,
    config: dict | None = None,
) -> dict:
    """Compute organizational theory metrics from a list of AgentMessages.

    Dispatches to OS-specific and handler-specific metric functions based
    on *os_name* and *handler_name*, then merges the results into a single
    flat dict.

    Args:
        messages:     List of AgentMessage objects (or dicts with the same keys).
        os_name:      Organizational structure name: "orchestrated", "networked",
                      or "sequential".
        handler_name: Execution handler name: "placeholder", "iterative_feedback",
                      "graph_routed", or "staged_pipeline".
        config:       Optional dict (or object with attributes) carrying strategy
                      and handler configuration values.

    Returns:
        dict with all computed metrics plus a "warnings" list describing
        fields that could not be populated.
    """
    cfg = config or {}
    warnings: list[str] = []
    result: dict[str, Any] = {}

    # Determine eval success from config if passed through
    eval_success: bool | None = cfg.get("_eval_success") if isinstance(cfg, dict) else None

    # --- OS-level metrics ---
    if os_name == "orchestrated":
        result.update(_orchestrated_os_metrics(messages, cfg, eval_success, warnings))
    elif os_name == "networked":
        result.update(_networked_os_metrics(messages, cfg, eval_success, warnings))
    elif os_name == "sequential":
        result.update(_sequential_os_metrics(messages, cfg, warnings))
    else:
        warnings.append(f"Unknown os_name {os_name!r}; OS metrics skipped.")

    # --- Handler-level metrics ---
    if handler_name == "iterative_feedback":
        result.update(_iterative_feedback_metrics(messages, cfg, warnings))
    elif handler_name == "graph_routed":
        result.update(_graph_routed_metrics(messages, cfg, warnings))
    elif handler_name == "staged_pipeline":
        result.update(_staged_pipeline_metrics(messages, cfg, warnings))
    elif handler_name == "placeholder":
        pass  # placeholder has no additional metrics
    else:
        warnings.append(f"Unknown handler_name {handler_name!r}; handler metrics skipped.")

    result["warnings"] = warnings
    return result
