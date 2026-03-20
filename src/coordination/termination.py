"""Termination checker for multi-agent coordination runs."""

from src.coordination.history import SharedHistory


class TerminationChecker:
    """Checks multiple termination conditions against the shared history.

    Conditions:
    - Termination keyword found in the most recent message output
    - Maximum total turns reached
    - Maximum consecutive errors reached
    - Stuck detection: same agent produces identical output twice in a row
    """

    def __init__(self, config: dict):
        term = config.get("termination", {})
        self.keyword: str = term.get("keyword", "TASK_COMPLETE")
        self.max_turns: int = term.get("max_turns", 20)
        self.max_consecutive_errors: int = term.get("max_consecutive_errors", 3)

    def should_stop(self, history: SharedHistory) -> bool:
        """Return True if any termination condition is met."""
        return (
            self._keyword_found(history)
            or self._max_turns_reached(history)
            or self._max_consecutive_errors(history)
            or self._stuck_detected(history)
        )

    def check_reason(self, history: SharedHistory) -> str | None:
        """Return the reason for termination, or None if no condition is met."""
        if self._keyword_found(history):
            return f"keyword:{self.keyword}"
        if self._max_turns_reached(history):
            return f"max_turns:{self.max_turns}"
        if self._max_consecutive_errors(history):
            return f"max_consecutive_errors:{self.max_consecutive_errors}"
        if self._stuck_detected(history):
            return "stuck:identical_output"
        return None

    def _keyword_found(self, history: SharedHistory) -> bool:
        if len(history) == 0:
            return False
        last = history.get_recent(1)[0]
        return self.keyword in last.content

    def _max_turns_reached(self, history: SharedHistory) -> bool:
        return len(history) >= self.max_turns

    def _max_consecutive_errors(self, history: SharedHistory) -> bool:
        if len(history) < self.max_consecutive_errors:
            return False
        recent = history.get_recent(self.max_consecutive_errors)
        return all(m.error is not None for m in recent)

    def _stuck_detected(self, history: SharedHistory) -> bool:
        if len(history) < 2:
            return False
        last_two = history.get_recent(2)
        a, b = last_two[0], last_two[1]
        # Don't consider error turns as "stuck" — errors have their own check
        if a.error is not None or b.error is not None:
            return False
        return (a.agent_name == b.agent_name) and (a.content == b.content)
