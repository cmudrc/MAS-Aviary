"""Orchestration-specific metric computation.

Per-prompt metrics are computed from a single prompt's AgentMessage
history. Cross-prompt metrics accumulate over multiple prompts.
All metrics are computed from the standard AgentMessage and
ToolCallRecord data — no logger or dataclass changes needed.
"""

from src.coordination.history import AgentMessage


def compute_orchestration_metrics(
    messages: list[AgentMessage],
    orchestrator_name: str = "orchestrator",
    worker_names: list[str] | None = None,
) -> dict:
    """Compute per-prompt orchestration metrics from message history.

    Args:
        messages: All AgentMessages from a single prompt/run.
        orchestrator_name: Name of the orchestrator agent.
        worker_names: Names of worker agents. If None, any agent
            that is not the orchestrator is considered a worker.

    Returns:
        Dict of orchestration-specific metrics.
    """
    if not messages:
        return _empty_orchestration_metrics()

    orch_msgs = [m for m in messages if m.agent_name == orchestrator_name]
    if worker_names is not None:
        worker_msgs = [m for m in messages if m.agent_name in worker_names]
    else:
        worker_msgs = [m for m in messages if m.agent_name != orchestrator_name]

    total = len(messages)
    orch_turns = len(orch_msgs)
    worker_turns = len(worker_msgs)

    # Orchestrator overhead ratio.
    overhead = orch_turns / total if total > 0 else 0.0

    # Agents spawned — count unique worker agent names.
    agents_spawned = len({m.agent_name for m in worker_msgs})

    # Orchestrator token growth — token counts per orchestrator turn.
    orch_token_growth = [
        m.token_count for m in orch_msgs if m.token_count is not None
    ]

    # Tool distribution — which tools each agent used.
    tool_distribution: dict[str, set[str]] = {}
    for m in worker_msgs:
        for tc in m.tool_calls:
            tool_distribution.setdefault(m.agent_name, set()).add(tc.tool_name)
    # Convert sets to sorted lists for JSON serialization.
    tool_dist_serializable = {
        name: sorted(tools) for name, tools in tool_distribution.items()
    }

    # Information ratio — mean orchestrator context tokens / mean worker context tokens.
    orch_tokens = [m.token_count for m in orch_msgs if m.token_count is not None]
    worker_tokens = [m.token_count for m in worker_msgs if m.token_count is not None]
    mean_orch = sum(orch_tokens) / len(orch_tokens) if orch_tokens else 0.0
    mean_worker = sum(worker_tokens) / len(worker_tokens) if worker_tokens else 0.0
    info_ratio = mean_orch / mean_worker if mean_worker > 0 else 0.0

    return {
        "agents_spawned": agents_spawned,
        "orchestrator_turns": orch_turns,
        "worker_turns": worker_turns,
        "orchestrator_overhead_ratio": round(overhead, 4),
        "orchestrator_token_growth": orch_token_growth,
        "tool_distribution": tool_dist_serializable,
        "information_ratio": round(info_ratio, 4),
    }


def compute_cross_prompt_metrics(
    all_messages: list[list[AgentMessage]],
    orchestrator_name: str = "orchestrator",
) -> dict:
    """Compute cross-prompt metrics accumulated over multiple prompts.

    Args:
        all_messages: List of per-prompt message histories.
        orchestrator_name: Name of the orchestrator agent.

    Returns:
        Dict of cross-prompt metrics including per-agent scores.
    """
    agent_stats: dict[str, dict] = {}

    for prompt_messages in all_messages:
        # Track which agents participated in this prompt.
        prompt_agents: dict[str, dict] = {}

        for msg in prompt_messages:
            if not isinstance(msg, AgentMessage):
                continue
            name = msg.agent_name
            if name not in agent_stats:
                agent_stats[name] = {
                    "total_turns": 0,
                    "retries": 0,
                    "total_tool_calls": 0,
                    "failed_tool_calls": 0,
                    "prompts_participated": 0,
                    "prompts_with_errors": 0,
                }
            stats = agent_stats[name]
            stats["total_turns"] += 1
            if msg.is_retry:
                stats["retries"] += 1
            for tc in msg.tool_calls:
                stats["total_tool_calls"] += 1
                if tc.error:
                    stats["failed_tool_calls"] += 1

            # Track per-prompt errors for this agent.
            if name not in prompt_agents:
                prompt_agents[name] = {"has_error": False}
            if msg.error or any(tc.error for tc in msg.tool_calls):
                prompt_agents[name]["has_error"] = True

        # Update prompt-level stats.
        for name, pstats in prompt_agents.items():
            agent_stats[name]["prompts_participated"] += 1
            if pstats["has_error"]:
                agent_stats[name]["prompts_with_errors"] += 1

    # Compute per-agent rates.
    per_agent = {}
    for name, stats in agent_stats.items():
        total_tc = stats["total_tool_calls"]
        total_turns = stats["total_turns"]
        total_prompts = stats["prompts_participated"]

        tool_error_rate = (
            stats["failed_tool_calls"] / total_tc if total_tc > 0 else 0.0
        )
        retry_rate = stats["retries"] / total_turns if total_turns > 0 else 0.0
        completion_rate = (
            1.0 - (stats["prompts_with_errors"] / total_prompts)
            if total_prompts > 0
            else 0.0
        )
        score = (
            (1 - tool_error_rate) * 0.5
            + (1 - retry_rate) * 0.3
            + completion_rate * 0.2
        )

        per_agent[name] = {
            "tool_error_rate": round(tool_error_rate, 4),
            "retry_rate": round(retry_rate, 4),
            "task_completion_rate": round(completion_rate, 4),
            "authority_score": round(score, 4),
            "prompts_participated": total_prompts,
        }

    return {
        "per_agent": per_agent,
        "total_prompts": len(all_messages),
    }


def _empty_orchestration_metrics() -> dict:
    return {
        "agents_spawned": 0,
        "orchestrator_turns": 0,
        "worker_turns": 0,
        "orchestrator_overhead_ratio": 0.0,
        "orchestrator_token_growth": [],
        "tool_distribution": {},
        "information_ratio": 0.0,
    }
