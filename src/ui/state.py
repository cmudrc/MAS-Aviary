"""Dashboard state management — reads from InstrumentationLogger or JSON logs.

The InstrumentationLogger is the shared contract: execution code writes
AgentMessage entries, and this module reads them for display.
"""

import glob
import json
import os
from dataclasses import dataclass, field


@dataclass
class RunState:
    """Snapshot of a coordination run for display."""
    history: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    is_running: bool = False
    task: str = ""
    strategy: str = ""
    error: str | None = None


def load_latest_run(logs_dir: str = "logs/") -> RunState:
    """Load the most recent run JSON from the logs directory."""
    pattern = os.path.join(logs_dir, "*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return RunState()

    return load_run_file(files[0])


def load_run_file(path: str) -> RunState:
    """Load a specific run JSON file into a RunState."""
    try:
        with open(path) as f:
            data = json.load(f)
        return RunState(
            history=data.get("history", []),
            metrics=data.get("metrics", {}),
        )
    except (json.JSONDecodeError, OSError) as e:
        return RunState(error=str(e))


def list_run_files(logs_dir: str = "logs/") -> list[str]:
    """List all run JSON files, newest first."""
    pattern = os.path.join(logs_dir, "*.json")
    return sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)


# Agent color palette — consistent color per agent name.
_AGENT_COLORS = [
    "#FF6B6B",  # red
    "#4ECDC4",  # teal
    "#45B7D1",  # blue
    "#96CEB4",  # green
    "#FFEAA7",  # yellow
    "#DDA0DD",  # plum
    "#98D8C8",  # mint
    "#F7DC6F",  # gold
]

_agent_color_cache: dict[str, str] = {}


def get_agent_color(agent_name: str) -> str:
    """Get a consistent color for an agent name."""
    if agent_name not in _agent_color_cache:
        idx = len(_agent_color_cache) % len(_AGENT_COLORS)
        _agent_color_cache[agent_name] = _AGENT_COLORS[idx]
    return _agent_color_cache[agent_name]
