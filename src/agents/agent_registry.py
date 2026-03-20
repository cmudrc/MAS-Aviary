"""Agent registry — stores and retrieves agents by name."""

from smolagents import ToolCallingAgent


class AgentRegistry:
    """Simple registry that maps agent names to ToolCallingAgent instances."""

    def __init__(self):
        self._agents: dict[str, ToolCallingAgent] = {}

    def register(self, agent: ToolCallingAgent) -> None:
        """Register an agent. Raises ValueError if name already taken."""
        if agent.name in self._agents:
            raise ValueError(f"Agent {agent.name!r} is already registered.")
        self._agents[agent.name] = agent

    def get(self, name: str) -> ToolCallingAgent:
        """Retrieve an agent by name. Raises KeyError if not found."""
        if name not in self._agents:
            raise KeyError(f"Agent {name!r} not found. Registered: {list(self._agents)}")
        return self._agents[name]

    def list_names(self) -> list[str]:
        """Return a list of all registered agent names."""
        return list(self._agents.keys())

    def all(self) -> dict[str, ToolCallingAgent]:
        """Return the full agent dict (name -> agent)."""
        return dict(self._agents)

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents
