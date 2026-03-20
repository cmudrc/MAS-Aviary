"""Shared mutable blackboard for peer-based networked coordination.

The blackboard is a key-value store that all peer agents can read and
write. It sits alongside SharedHistory as a separate data structure:
SharedHistory is an append-only log of agent messages; the Blackboard
is a live view of current system state (statuses, claims, results,
gaps, predictions) that agents update as they work.

Entry types:
  - status: agent reports what it's currently doing
  - claim: agent claims a subtask (enforcement depends on claiming mode)
  - result: agent posts completed work
  - gap: agent identifies something nobody is handling
  - prediction: agent predicts what another agent will do next
"""

import time
from dataclasses import dataclass

VALID_ENTRY_TYPES = frozenset({"status", "claim", "result", "gap", "prediction"})


@dataclass
class BlackboardEntry:
    """A single entry on the shared blackboard."""

    key: str
    value: str
    author: str
    entry_type: str
    timestamp: float
    version: int = 1


class Blackboard:
    """Shared mutable state readable and writable by all peer agents.

    Supports three claiming modes:
      - "none": claims are informational only, no enforcement
      - "soft": duplicate claims are allowed with a warning
      - "hard": duplicate claims are rejected (locked)
    """

    def __init__(self, claiming_mode: str = "soft"):
        if claiming_mode not in ("none", "soft", "hard"):
            raise ValueError(
                f"Invalid claiming_mode {claiming_mode!r}. "
                "Must be 'none', 'soft', or 'hard'."
            )
        self._entries: dict[str, BlackboardEntry] = {}
        self._claiming_mode = claiming_mode
        # Track write events for metrics.
        self._write_count: int = 0
        self._claim_conflicts: int = 0

    @property
    def claiming_mode(self) -> str:
        return self._claiming_mode

    @property
    def write_count(self) -> int:
        return self._write_count

    @property
    def claim_conflicts(self) -> int:
        return self._claim_conflicts

    # -- Read operations -------------------------------------------------------

    def read_all(self) -> list[BlackboardEntry]:
        """Return all entries."""
        return list(self._entries.values())

    def read_by_type(self, entry_type: str) -> list[BlackboardEntry]:
        """Return entries filtered by type."""
        return [e for e in self._entries.values() if e.entry_type == entry_type]

    def read_by_author(self, author: str) -> list[BlackboardEntry]:
        """Return entries filtered by author."""
        return [e for e in self._entries.values() if e.author == author]

    def get(self, key: str) -> BlackboardEntry | None:
        """Get a single entry by key."""
        return self._entries.get(key)

    def get_claims(self) -> list[BlackboardEntry]:
        """Return all claim entries."""
        return self.read_by_type("claim")

    def is_claimed(self, key: str) -> bool:
        """Check if a key has an active claim entry."""
        entry = self._entries.get(key)
        return entry is not None and entry.entry_type == "claim"

    # -- Write operations ------------------------------------------------------

    def write(
        self, key: str, value: str, author: str, entry_type: str
    ) -> tuple[BlackboardEntry | None, str | None]:
        """Write a new entry or update an existing one.

        Returns:
            (entry, warning) — entry is None on hard-claim rejection,
            warning is a string if a soft-claim conflict occurred.
        """
        if entry_type not in VALID_ENTRY_TYPES:
            raise ValueError(
                f"Invalid entry_type {entry_type!r}. "
                f"Must be one of {sorted(VALID_ENTRY_TYPES)}."
            )

        self._write_count += 1
        warning = None

        # Claiming logic.
        if entry_type == "claim" and self._claiming_mode != "none":
            existing = self._entries.get(key)
            if existing and existing.entry_type == "claim" and existing.author != author:
                self._claim_conflicts += 1
                if self._claiming_mode == "hard":
                    return None, f"{key} is locked by {existing.author}"
                else:  # soft
                    warning = f"Warning: {key} already claimed by {existing.author}"

        # If key exists and same author, update in place.
        existing = self._entries.get(key)
        if existing and existing.author == author:
            existing.value = value
            existing.entry_type = entry_type
            existing.timestamp = time.time()
            existing.version += 1
            return existing, warning

        # If key exists but different author, create modified key.
        if existing and existing.author != author:
            modified_key = f"{key}_{author}"
            entry = BlackboardEntry(
                key=modified_key,
                value=value,
                author=author,
                entry_type=entry_type,
                timestamp=time.time(),
                version=1,
            )
            self._entries[modified_key] = entry
            return entry, warning

        # New entry.
        entry = BlackboardEntry(
            key=key,
            value=value,
            author=author,
            entry_type=entry_type,
            timestamp=time.time(),
            version=1,
        )
        self._entries[key] = entry
        return entry, warning

    def update(self, key: str, value: str, author: str) -> BlackboardEntry | None:
        """Update an existing entry's value. Only the author can update.

        Returns the updated entry, or None if key not found or wrong author.
        """
        existing = self._entries.get(key)
        if existing is None:
            return None
        if existing.author != author:
            return None
        existing.value = value
        existing.timestamp = time.time()
        existing.version += 1
        self._write_count += 1
        return existing

    def delete(self, key: str, author: str) -> bool:
        """Delete an entry. Only the original author can delete.

        Returns True if deleted, False if not found or wrong author.
        """
        existing = self._entries.get(key)
        if existing is None:
            return False
        if existing.author != author:
            return False
        del self._entries[key]
        return True

    # -- Context rendering -----------------------------------------------------

    def to_context_string(
        self,
        requesting_agent: str,
        config: dict,
    ) -> str:
        """Render blackboard contents as a string for agent context.

        Filtering rules based on config toggles:
          - peer_monitoring_visible=False: strip metric values from entries
          - trans_specialist_knowledge=False: truncate result values to
            first 100 chars and strip reasoning patterns
          - predictive_knowledge=False: exclude prediction entries entirely

        Claims, statuses, and gaps are always visible regardless of toggles.

        Args:
            requesting_agent: name of the agent requesting the view
            config: dict with toggle keys:
                peer_monitoring_visible (bool, default True)
                trans_specialist_knowledge (bool, default True)
                predictive_knowledge (bool, default False)
        """
        peer_monitoring = config.get("peer_monitoring_visible", True)
        trans_specialist = config.get("trans_specialist_knowledge", True)
        predictive = config.get("predictive_knowledge", False)

        lines = []
        for entry in self._entries.values():
            # Filter predictions if disabled.
            if entry.entry_type == "prediction" and not predictive:
                continue

            # Build display value.
            display_value = entry.value

            # Filter trans-specialist knowledge from results.
            if entry.entry_type == "result" and not trans_specialist:
                display_value = _truncate_result(display_value)

            # Filter peer monitoring metrics.
            if not peer_monitoring:
                display_value = _strip_metrics(display_value)

            lines.append(
                f"[{entry.entry_type.upper()}] {entry.key} "
                f"(by {entry.author}, v{entry.version}): "
                f"{display_value}"
            )

        if not lines:
            return "Blackboard is empty."

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)


# -- Helpers -------------------------------------------------------------------

def _truncate_result(value: str, max_chars: int = 400) -> str:
    """Truncate a result value and strip reasoning indicators.

    ~100 tokens ≈ ~400 characters.
    """
    # Strip common reasoning patterns.
    stripped = _strip_reasoning(value)
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars] + "..."


def _strip_reasoning(value: str) -> str:
    """Remove reasoning-indicator patterns from a value string."""
    lines = value.split("\n")
    filtered = []
    skip = False
    for line in lines:
        lower = line.lower().strip()
        # Skip lines that begin with reasoning indicators.
        if lower.startswith(("reasoning:", "because:", "my reasoning:",
                            "explanation:", "rationale:", "thinking:")):
            skip = True
            continue
        # Resume after a blank line following a reasoning block.
        if skip and not lower:
            skip = False
            continue
        if not skip:
            filtered.append(line)
    return "\n".join(filtered)


def _strip_metrics(value: str) -> str:
    """Remove metric/performance data from an entry value."""
    lines = value.split("\n")
    filtered = []
    for line in lines:
        lower = line.lower().strip()
        if any(kw in lower for kw in (
            "error_rate", "retry_count", "success_rate",
            "tool_error", "performance:", "metric:",
        )):
            continue
        filtered.append(line)
    return "\n".join(filtered)
