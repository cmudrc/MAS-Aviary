"""InstrumentationLogger — SHARED CONTRACT between execution and UI.

Any code that runs agents writes AgentMessage entries here.
The Streamlit dashboard reads from here.
Neither knows about the other.
"""

import os
import time

from src.coordination.history import AgentMessage


class InstrumentationLogger:
    """Append-only logger for agent turns. The shared contract for the system."""

    def __init__(self, config=None):
        self._messages: list[AgentMessage] = []
        self._run_id: str = f"run_{int(time.time())}"

        if config is not None:
            log_config = getattr(config, "logging", None) or config.get("logging", {})
            if hasattr(log_config, "output_dir"):
                self._output_dir = log_config.output_dir
            elif isinstance(log_config, dict):
                self._output_dir = log_config.get("output_dir", "logs/")
            else:
                self._output_dir = "logs/"
        else:
            self._output_dir = "logs/"

    def log_turn(self, message: AgentMessage) -> None:
        """Append an agent turn to the log. Must be called for every turn."""
        self._messages.append(message)

    def get_messages(self) -> list[AgentMessage]:
        """Return all logged messages (read-only copy)."""
        return list(self._messages)

    def get_latest(self) -> AgentMessage | None:
        """Return the most recent message, or None if empty."""
        return self._messages[-1] if self._messages else None

    def compute_metrics(self) -> dict:
        """Compute aggregate metrics from the logged history."""
        from src.logging.metrics import compute_metrics

        return compute_metrics(self._messages)

    def export_json(self, path: str | None = None) -> str:
        """Export the full run log as a JSON file. Returns the file path."""
        from src.logging.exporter import export_run

        if path is None:
            os.makedirs(self._output_dir, exist_ok=True)
            path = os.path.join(self._output_dir, f"{self._run_id}.json")
        return export_run(self._messages, self.compute_metrics(), path)

    @property
    def turn_count(self) -> int:
        return len(self._messages)
