"""Networked strategy-specific metric computation.

Per-prompt metrics track blackboard utilization, claiming behavior,
duplicate work, self-selection diversity, and prediction accuracy.
Cross-prompt metrics compute omission/commission errors, escalation
of commitment, ambidexterity proxy, and joint myopia score.

All metrics are computed from AgentMessage data — no logger or
dataclass changes needed.
"""

import re

from src.coordination.history import AgentMessage


def compute_networked_metrics(
    messages: list[AgentMessage],
    blackboard_size: int = 0,
    claim_conflicts: int = 0,
    initial_agents: int = 5,
    spawned_agents: int = 0,
    predictions: list[dict] | None = None,
) -> dict:
    """Compute per-prompt networked metrics from message history.

    Args:
        messages: All AgentMessages from a single prompt/run.
        blackboard_size: Number of entries on the blackboard at prompt end.
        claim_conflicts: Total claim warnings + rejections this prompt.
        initial_agents: Number of initial agents.
        spawned_agents: Number of agents spawned during this prompt.
        predictions: List of prediction records, each with
            {predictor, target, prediction, actual, accuracy_score}.

    Returns:
        Dict of networked-specific metrics.
    """
    if not messages:
        return _empty_networked_metrics()

    total_agents = initial_agents + spawned_agents
    total_turns = len(messages)

    # Blackboard utilization: count turns where read_blackboard was called.
    turns_with_reads = 0
    turns_with_writes = 0
    total_bb_reads = 0
    total_bb_writes = 0
    claims_made = []
    peers_spawned_count = 0
    prediction_count = 0

    # Per-agent work tracking (for duplicate work rate and diversity).
    agent_subtasks: dict[str, set[str]] = {}  # agent -> set of subtask keys
    subtask_agents: dict[str, set[str]] = {}  # subtask -> set of agents

    for msg in messages:
        has_read = False
        has_write = False

        for tc in msg.tool_calls:
            if tc.tool_name == "read_blackboard":
                has_read = True
                total_bb_reads += 1
            elif tc.tool_name == "write_blackboard":
                has_write = True
                total_bb_writes += 1
                # Track subtask claims and results.
                inputs = tc.inputs if isinstance(tc.inputs, dict) else {}
                entry_type = inputs.get("entry_type", "")
                key = inputs.get("key", "")
                if entry_type == "claim":
                    claims_made.append(key)
                if entry_type in ("claim", "result") and key:
                    agent_subtasks.setdefault(msg.agent_name, set()).add(key)
                    subtask_agents.setdefault(key, set()).add(msg.agent_name)
                if entry_type == "prediction":
                    prediction_count += 1
            elif tc.tool_name == "spawn_peer":
                peers_spawned_count += 1

        if has_read:
            turns_with_reads += 1
        if has_write:
            turns_with_writes += 1

    # Blackboard utilization.
    bb_utilization = turns_with_reads / total_turns if total_turns > 0 else 0.0

    # Duplicate work rate: subtasks worked on by >1 agent / total subtasks.
    total_subtasks = len(subtask_agents)
    duplicate_subtasks = sum(
        1 for agents in subtask_agents.values() if len(agents) > 1
    )
    duplicate_work_rate = (
        duplicate_subtasks / total_subtasks if total_subtasks > 0 else 0.0
    )

    # Self-selection diversity: unique subtasks chosen / total agent turns.
    sum(len(subs) for subs in agent_subtasks.values())
    unique_subtasks = len(subtask_agents)
    self_selection_diversity = (
        unique_subtasks / total_turns if total_turns > 0 else 0.0
    )

    # Prediction accuracy.
    prediction_accuracy = 0.0
    if predictions:
        scores = [p.get("accuracy_score", 0.0) for p in predictions]
        prediction_accuracy = sum(scores) / len(scores) if scores else 0.0

    return {
        "total_agents": total_agents,
        "agents_spawned": spawned_agents,
        "blackboard_size": blackboard_size,
        "claim_conflicts": claim_conflicts,
        "duplicate_work_rate": round(duplicate_work_rate, 4),
        "blackboard_utilization": round(bb_utilization, 4),
        "self_selection_diversity": round(self_selection_diversity, 4),
        "prediction_accuracy": round(prediction_accuracy, 4),
        "prediction_count": prediction_count,
        "total_bb_reads": total_bb_reads,
        "total_bb_writes": total_bb_writes,
        "peers_spawned": peers_spawned_count,
        "claims_made": len(claims_made),
    }


def compute_cross_prompt_metrics(
    all_prompt_data: list[dict],
) -> dict:
    """Compute cross-prompt metrics from multiple prompts.

    Args:
        all_prompt_data: List of dicts, each with:
            - messages: list[AgentMessage]
            - eval_score: float (0-1, quality of final output)
            - redundancy_rate: float (from base metrics)

    Returns:
        Dict with omission/commission errors, escalation, ambidexterity,
        and joint myopia classification.
    """
    if not all_prompt_data:
        return _empty_cross_prompt_metrics()

    eval_threshold = 0.5  # below this = failure

    omission_errors = 0
    commission_errors = 0
    redundancy_rates = []
    convergence_scores = []
    prompt_classifications = []
    prev_failure_signatures = []

    for prompt in all_prompt_data:
        messages = prompt.get("messages", [])
        eval_score = prompt.get("eval_score", 0.0)
        redundancy = prompt.get("redundancy_rate", 0.0)
        redundancy_rates.append(redundancy)

        # Omission: eval below threshold.
        if eval_score < eval_threshold:
            omission_errors += 1

        # Commission: bad output that wasn't flagged (eval < threshold
        # but no error messages in history).
        has_errors = any(
            m.error for m in messages if isinstance(m, AgentMessage)
        )
        if eval_score < eval_threshold and not has_errors:
            commission_errors += 1

        # Convergence score for joint myopia.
        convergence = _compute_convergence(messages)
        convergence_scores.append(convergence)

        # Classify this prompt.
        is_error = eval_score < eval_threshold
        is_converged = convergence > 0.5
        if is_converged and is_error:
            classification = "joint_myopia"
        elif is_converged and not is_error:
            classification = "effective_consensus"
        elif not is_converged and is_error:
            classification = "incoherent_team"
        else:
            classification = "healthy_diversity"
        prompt_classifications.append(classification)

        # Track failure signatures for escalation.
        if is_error:
            sig = _failure_signature(messages)
            prev_failure_signatures.append(sig)
        else:
            prev_failure_signatures.append(None)

    # Escalation of commitment: consecutive prompts with similar failures.
    escalation_count = 0
    for i in range(1, len(prev_failure_signatures)):
        if (prev_failure_signatures[i] is not None
                and prev_failure_signatures[i - 1] is not None):
            overlap = _signature_overlap(
                prev_failure_signatures[i],
                prev_failure_signatures[i - 1],
            )
            if overlap > 0.5:
                escalation_count += 1

    # Ambidexterity proxy: variance of redundancy rates.
    mean_red = (
        sum(redundancy_rates) / len(redundancy_rates) if redundancy_rates else 0.0
    )
    variance_red = (
        sum((r - mean_red) ** 2 for r in redundancy_rates) / len(redundancy_rates)
        if redundancy_rates
        else 0.0
    )

    # Mean joint myopia score.
    mean_convergence = (
        sum(convergence_scores) / len(convergence_scores)
        if convergence_scores
        else 0.0
    )

    return {
        "omission_errors": omission_errors,
        "commission_errors": commission_errors,
        "escalation_of_commitment": escalation_count,
        "ambidexterity_proxy": round(variance_red, 4),
        "mean_joint_myopia_score": round(mean_convergence, 4),
        "prompt_classifications": prompt_classifications,
        "total_prompts": len(all_prompt_data),
    }


def compute_prediction_accuracy(
    prediction_text: str, actual_text: str
) -> float:
    """Compute accuracy between a prediction and actual action.

    Uses keyword overlap: |intersection| / |union| of tokenized words.
    Returns a float in [0, 1].
    """
    if not prediction_text or not actual_text:
        return 0.0

    pred_words = set(_tokenize(prediction_text))
    actual_words = set(_tokenize(actual_text))

    if not pred_words and not actual_words:
        return 1.0
    if not pred_words or not actual_words:
        return 0.0

    intersection = pred_words & actual_words
    union = pred_words | actual_words
    return len(intersection) / len(union)


# -- Internal helpers ----------------------------------------------------------

def _compute_convergence(messages: list[AgentMessage]) -> float:
    """Compute mean pairwise keyword overlap of agent reasoning traces.

    This is a simplified version of cosine similarity using word overlap
    (Jaccard). Returns a float in [0, 1] where 1 = identical reasoning.
    """
    if not messages:
        return 0.0

    # Collect content per agent.
    agent_contents: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, AgentMessage):
            continue
        name = msg.agent_name
        agent_contents.setdefault(name, "")
        agent_contents[name] += " " + msg.content

    agents = list(agent_contents.keys())
    if len(agents) < 2:
        return 0.0

    # Pairwise Jaccard similarity.
    similarities = []
    for i in range(len(agents)):
        words_i = set(_tokenize(agent_contents[agents[i]]))
        for j in range(i + 1, len(agents)):
            words_j = set(_tokenize(agent_contents[agents[j]]))
            union = words_i | words_j
            if not union:
                continue
            intersection = words_i & words_j
            similarities.append(len(intersection) / len(union))

    return sum(similarities) / len(similarities) if similarities else 0.0


def _failure_signature(messages: list[AgentMessage]) -> set[str]:
    """Extract a keyword signature from failed messages for escalation detection."""
    keywords = set()
    for msg in messages:
        if not isinstance(msg, AgentMessage):
            continue
        if msg.error:
            keywords.update(_tokenize(msg.error))
        keywords.update(_tokenize(msg.content)[:20])  # first 20 words
    return keywords


def _signature_overlap(sig_a: set[str], sig_b: set[str]) -> float:
    """Jaccard overlap between two failure signatures."""
    if not sig_a and not sig_b:
        return 1.0
    if not sig_a or not sig_b:
        return 0.0
    return len(sig_a & sig_b) / len(sig_a | sig_b)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenization, lowercased."""
    return [w.lower() for w in re.findall(r'\w+', text) if len(w) > 1]


def _empty_networked_metrics() -> dict:
    return {
        "total_agents": 0,
        "agents_spawned": 0,
        "blackboard_size": 0,
        "claim_conflicts": 0,
        "duplicate_work_rate": 0.0,
        "blackboard_utilization": 0.0,
        "self_selection_diversity": 0.0,
        "prediction_accuracy": 0.0,
        "prediction_count": 0,
        "total_bb_reads": 0,
        "total_bb_writes": 0,
        "peers_spawned": 0,
        "claims_made": 0,
    }


def _empty_cross_prompt_metrics() -> dict:
    return {
        "omission_errors": 0,
        "commission_errors": 0,
        "escalation_of_commitment": 0,
        "ambidexterity_proxy": 0.0,
        "mean_joint_myopia_score": 0.0,
        "prompt_classifications": [],
        "total_prompts": 0,
    }
