"""Aggregate metric computation from coordination history."""

from src.coordination.history import AgentMessage


def compute_metrics(messages: list[AgentMessage]) -> dict:
    """Compute all PRD-defined metrics from a list of AgentMessages.

    Returns a dict with all metric keys, computed from the message history.
    """
    total = len(messages)
    if total == 0:
        return _empty_metrics()

    # Total tool calls
    total_tool_calls = sum(len(m.tool_calls) for m in messages)
    failed_tool_calls = sum(
        1 for m in messages for tc in m.tool_calls if tc.error is not None
    )

    # Retry tracking
    retry_count = sum(1 for m in messages if m.is_retry)
    error_count = sum(1 for m in messages if m.error is not None)

    # Token counts (may be None)
    total_tokens = 0
    has_tokens = False
    for m in messages:
        if m.token_count is not None:
            total_tokens += m.token_count
            has_tokens = True

    # Total duration
    total_duration = sum(m.duration_seconds for m in messages)

    # Redundancy rate: fraction of turns with >80% token overlap with any previous
    redundant = 0
    for i, m in enumerate(messages):
        for j in range(i):
            if _is_redundant(m.content, messages[j].content):
                redundant += 1
                break

    redundancy_rate = redundant / total if total > 0 else 0.0
    error_rate = error_count / total if total > 0 else 0.0
    retry_rate = retry_count / total if total > 0 else 0.0
    tool_error_rate = failed_tool_calls / total_tool_calls if total_tool_calls > 0 else 0.0
    coordination_efficiency = max(0.0, 1.0 - redundancy_rate - error_rate)

    return {
        "total_messages": total,
        "total_duration_seconds": round(total_duration, 4),
        "total_tokens": total_tokens if has_tokens else None,
        "total_tool_calls": total_tool_calls,
        "tool_error_rate": round(tool_error_rate, 4),
        "retry_count": retry_count,
        "retry_rate": round(retry_rate, 4),
        "error_count": error_count,
        "error_rate": round(error_rate, 4),
        "redundancy_rate": round(redundancy_rate, 4),
        "coordination_efficiency": round(coordination_efficiency, 4),
    }


def _is_redundant(a: str, b: str, threshold: float = 0.8) -> bool:
    """Check if two outputs have >threshold token overlap."""
    if not a or not b:
        return False
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b)
    smaller = min(len(tokens_a), len(tokens_b))
    return (overlap / smaller) >= threshold if smaller > 0 else False


def _empty_metrics() -> dict:
    return {
        "total_messages": 0,
        "total_duration_seconds": 0.0,
        "total_tokens": None,
        "total_tool_calls": 0,
        "tool_error_rate": 0.0,
        "retry_count": 0,
        "retry_rate": 0.0,
        "error_count": 0,
        "error_rate": 0.0,
        "redundancy_rate": 0.0,
        "coordination_efficiency": 0.0,
    }
