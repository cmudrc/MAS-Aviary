"""JSON export for run logs and metrics."""

import json
from dataclasses import asdict
from pathlib import Path

from src.coordination.history import AgentMessage


def export_run(messages: list[AgentMessage], metrics: dict, path: str) -> str:
    """Export a full run log as a JSON file.

    Args:
        messages: List of AgentMessages from the run.
        metrics: Computed metrics dict.
        path: Output file path.

    Returns:
        The file path written to.
    """
    data = {
        "history": [asdict(m) for m in messages],
        "metrics": metrics,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return path
