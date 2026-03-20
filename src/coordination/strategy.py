"""Abstract CoordinationStrategy interface and CoordinationAction dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class CoordinationAction:
    """Describes what the coordinator should do next."""
    action_type: str          # "invoke_agent", "terminate", "error"
    agent_name: str | None    # which agent to run (None if terminating)
    input_context: str        # what to pass to the agent
    metadata: dict = field(default_factory=dict)


@dataclass
class CoordinationResult:
    """Final result of a coordination run."""
    final_output: str
    history: list            # list of AgentMessage
    metrics: dict            # computed metrics dict


class CoordinationStrategy(ABC):
    """Abstract base class for all coordination strategies.

    Implement this interface to define a new way for agents to interact.
    The coordinator calls next_step() in a loop until is_complete() or
    next_step() returns a terminate action.
    """

    @abstractmethod
    def initialize(self, agents: dict, config: dict) -> None:
        """Set up the strategy with available agents and configuration."""

    @abstractmethod
    def next_step(self, history: list, current_state: dict) -> CoordinationAction:
        """Given current history and state, decide what happens next.

        Returns a CoordinationAction specifying which agent to invoke,
        what context to provide, or whether to terminate.
        """

    @abstractmethod
    def is_complete(self, history: list, current_state: dict) -> bool:
        """Determine if the overall task is finished."""
