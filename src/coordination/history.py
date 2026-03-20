"""Shared history and message data structures for multi-agent coordination."""

from dataclasses import dataclass, field


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation within an agent turn."""
    tool_name: str
    inputs: dict
    output: str
    duration_seconds: float
    error: str | None = None


@dataclass
class AgentMessage:
    """A single agent turn in the coordination history."""
    agent_name: str
    content: str
    turn_number: int
    timestamp: float
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    duration_seconds: float = 0.0
    token_count: int | None = None
    is_retry: bool = False
    retry_of_turn: int | None = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)


class SharedHistory:
    """Append-only history of AgentMessage entries shared across coordination strategies."""

    def __init__(self):
        self._messages: list[AgentMessage] = []

    def append(self, message: AgentMessage) -> None:
        """Add a message to the history."""
        self._messages.append(message)

    def get_recent(self, n: int) -> list[AgentMessage]:
        """Get the last N messages."""
        return self._messages[-n:]

    def get_by_agent(self, name: str) -> list[AgentMessage]:
        """Get all messages from a specific agent."""
        return [m for m in self._messages if m.agent_name == name]

    def get_all(self) -> list[AgentMessage]:
        """Return full history as a new list."""
        return list(self._messages)

    def to_context_string(self, max_tokens: int = 3000) -> str:
        """Format history as a string for injection into agent prompts.

        Truncates from oldest if the output exceeds max_tokens (estimated
        as ~4 chars per token). Always includes the most recent messages.
        """
        max_chars = max_tokens * 4
        lines = []
        for m in self._messages:
            line = f"[Turn {m.turn_number}] {m.agent_name}: {m.content}"
            lines.append(line)

        result = "\n".join(lines)
        if len(result) > max_chars:
            # Truncate from the beginning, keeping recent context
            result = result[-max_chars:]
            first_newline = result.find("\n")
            if first_newline != -1:
                result = "...\n" + result[first_newline + 1:]
            else:
                result = "..." + result
        return result

    @property
    def turn_count(self) -> int:
        """Number of messages in history."""
        return len(self._messages)

    def __len__(self) -> int:
        return len(self._messages)
