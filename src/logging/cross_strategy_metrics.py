"""Cross-strategy metrics computable for any OS × handler combination.

These 5 metrics are adapted from the Google multi-agent paper and apply
uniformly across all organizational structure and handler combinations.
"""

from __future__ import annotations

import re
from typing import Any

from src.coordination.history import AgentMessage
from src.coordination.similarity import compute_similarity

# ---------------------------------------------------------------------------
# Common error patterns
# ---------------------------------------------------------------------------

_ERROR_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?:Error|Exception|Traceback|FAILED|error|failed)", re.IGNORECASE),
    re.compile(r"SyntaxError|NameError|TypeError|ValueError|ImportError"),
    re.compile(r"execution.*fail|compile.*fail|tool.*fail", re.IGNORECASE),
]


def _has_error(content: str) -> bool:
    """Check if content contains error patterns."""
    return any(p.search(content) for p in _ERROR_PATTERNS)


def _extract_error_signatures(content: str) -> set[str]:
    """Extract error type signatures from content."""
    sigs: set[str] = set()
    for match in re.finditer(
        r"(SyntaxError|NameError|TypeError|ValueError|ImportError"
        r"|RuntimeError|KeyError|AttributeError|IndexError"
        r"|ZeroDivisionError|FileNotFoundError|ModuleNotFoundError"
        r"|compilation? fail(?:ed|ure)?|execution fail(?:ed|ure)?)",
        content,
        re.IGNORECASE,
    ):
        sigs.add(match.group(0).lower())
    return sigs


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def compute_coordination_overhead(
    messages: list[AgentMessage],
    minimum_turns: int | None = None,
) -> float:
    """Coordination Overhead = total_agent_turns - minimum_possible_turns.

    Args:
        messages: All agent messages from the run.
        minimum_turns: Explicit minimum possible turns. If None, computed
            as the number of unique agents (each needs at least 1 turn).

    Returns:
        Overhead count (float, >= 0).
    """
    total = len(messages)
    if minimum_turns is None:
        unique_agents = len({m.agent_name for m in messages})
        minimum_turns = max(unique_agents, 1)
    return max(0.0, float(total - minimum_turns))


def compute_message_density(
    messages: list[AgentMessage],
) -> float:
    """Message Density = total_agent_messages / 1 (single task).

    Simply returns the total message count for one task run.
    """
    return float(len(messages))


def compute_redundancy_rate(
    messages: list[AgentMessage],
    similarity_threshold: float = 0.8,
    similarity_method: str = "tfidf",
) -> float:
    """Redundancy Rate = fraction of outputs with cosine similarity > threshold.

    For each message, checks if it is similar (> threshold) to any
    previous message. Returns the fraction of redundant messages.

    Args:
        messages: All agent messages from the run.
        similarity_threshold: Minimum similarity to count as redundant.
        similarity_method: "tfidf" or "jaccard".

    Returns:
        Float in [0.0, 1.0].
    """
    if len(messages) <= 1:
        return 0.0

    contents = [m.content for m in messages]
    redundant_count = 0

    for i in range(1, len(contents)):
        current = contents[i]
        if not current.strip():
            continue
        for j in range(i):
            prev = contents[j]
            if not prev.strip():
                continue
            sim = compute_similarity(current, prev, method=similarity_method)
            if sim > similarity_threshold:
                redundant_count += 1
                break  # Only count once per message.

    # Denominator: messages after the first (only they can be redundant).
    return redundant_count / (len(messages) - 1)


def compute_coordination_efficiency(
    messages: list[AgentMessage],
    similarity_threshold: float = 0.8,
    similarity_method: str = "tfidf",
) -> float:
    """Coordination Efficiency = 1.0 - redundancy_rate - error_rate.

    Args:
        messages: All agent messages from the run.
        similarity_threshold: For redundancy detection.
        similarity_method: "tfidf" or "jaccard".

    Returns:
        Float (can be negative if redundancy + errors > 1.0).
    """
    redundancy = compute_redundancy_rate(
        messages,
        similarity_threshold,
        similarity_method,
    )
    error_count = sum(1 for m in messages if _has_error(m.content) or m.error)
    error_rate = error_count / len(messages) if messages else 0.0
    return 1.0 - redundancy - error_rate


def compute_error_amplification(
    messages: list[AgentMessage],
) -> int:
    """Error Amplification = count of agents whose output contains error
    patterns that appeared in a previous agent's output.

    This measures how errors spread through the agent pipeline.

    Returns:
        Count of amplified errors (int, >= 0).
    """
    if len(messages) <= 1:
        return 0

    amplified = 0
    previous_sigs: set[str] = set()

    for msg in messages:
        current_sigs = _extract_error_signatures(msg.content)
        if current_sigs and previous_sigs:
            overlap = current_sigs & previous_sigs
            if overlap:
                amplified += 1
        previous_sigs |= current_sigs

    return amplified


# ---------------------------------------------------------------------------
# All-in-one computation
# ---------------------------------------------------------------------------


def compute_cross_strategy_metrics(
    messages: list[AgentMessage],
    minimum_turns: int | None = None,
    similarity_threshold: float = 0.8,
    similarity_method: str = "tfidf",
) -> dict[str, Any]:
    """Compute all 5 cross-strategy metrics at once.

    Args:
        messages: All agent messages from the run.
        minimum_turns: Explicit minimum possible turns for overhead.
        similarity_threshold: For redundancy detection.
        similarity_method: "tfidf" or "jaccard".

    Returns:
        Dict with all 5 metric keys.
    """
    redundancy = compute_redundancy_rate(
        messages,
        similarity_threshold,
        similarity_method,
    )
    error_count = sum(1 for m in messages if _has_error(m.content) or m.error)
    error_rate = error_count / len(messages) if messages else 0.0

    return {
        "coordination_overhead": compute_coordination_overhead(messages, minimum_turns),
        "message_density": compute_message_density(messages),
        "redundancy_rate": redundancy,
        "coordination_efficiency": 1.0 - redundancy - error_rate,
        "error_amplification": compute_error_amplification(messages),
    }
