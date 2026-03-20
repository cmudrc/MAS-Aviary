"""Graph-Routed coordination strategy.

A directed transition graph defines which agents can follow which. After each
agent turn, routing logic selects the next agent from allowed transitions.

Two sub-modes:
- rule_based: select based on keywords in the output
- llm_routed: ask the LLM to pick the next agent from allowed options
"""

from src.coordination.history import AgentMessage
from src.coordination.strategy import CoordinationAction, CoordinationStrategy


class GraphRoutedStrategy(CoordinationStrategy):
    """Routes agents via a directed transition graph with rule-based or LLM routing."""

    def __init__(self):
        self.transitions: dict[str, list[str]] = {}
        self.routing_mode: str = "rule_based"
        self.routing_rules: dict[str, str] = {}
        self.start_agent: str | None = None
        self._model = None  # set externally for llm_routed mode

    def initialize(self, agents: dict, config: dict) -> None:
        gr_config = config.get("graph_routed", {})
        self.transitions = gr_config.get("transitions", {})
        self.routing_mode = gr_config.get("routing_mode", "rule_based")
        self.routing_rules = gr_config.get("routing_rules", {})

        # Validate all agents in transitions exist
        all_mentioned = set(self.transitions.keys())
        for targets in self.transitions.values():
            all_mentioned.update(targets)
        for name in all_mentioned:
            if name not in agents:
                raise ValueError(
                    f"Graph transition references unknown agent {name!r}. Available: {list(agents.keys())}"
                )

        # Start agent is the first key in transitions
        if self.transitions:
            self.start_agent = next(iter(self.transitions))

    def set_model(self, model) -> None:
        """Set the LLM model for llm_routed mode."""
        self._model = model

    def next_step(self, history: list, current_state: dict) -> CoordinationAction:
        if not history:
            # First step: start with the start agent
            agent_name = self.start_agent
            input_context = current_state.get("task", "")
        else:
            last_msg = history[-1]
            last_agent = last_msg.agent_name if isinstance(last_msg, AgentMessage) else ""
            last_content = last_msg.content if isinstance(last_msg, AgentMessage) else str(last_msg)

            allowed = self.transitions.get(last_agent, [])
            if not allowed:
                return CoordinationAction(
                    action_type="terminate",
                    agent_name=None,
                    input_context="",
                    metadata={"reason": "no_transitions", "from": last_agent},
                )

            agent_name = self._route(last_content, allowed)
            input_context = last_content

        if agent_name is None:
            return CoordinationAction(
                action_type="terminate",
                agent_name=None,
                input_context="",
                metadata={"reason": "no_start_agent"},
            )

        return CoordinationAction(
            action_type="invoke_agent",
            agent_name=agent_name,
            input_context=input_context,
        )

    def _route(self, content: str, allowed: list[str]) -> str:
        """Pick the next agent from allowed options based on routing mode."""
        if self.routing_mode == "rule_based":
            return self._rule_based_route(content, allowed)
        elif self.routing_mode == "llm_routed":
            return self._llm_routed(content, allowed)
        # Fallback to first allowed
        return allowed[0]

    def _rule_based_route(self, content: str, allowed: list[str]) -> str:
        """Match keywords in content against routing rules."""
        content_lower = content.lower()
        for keyword, target in self.routing_rules.items():
            if keyword == "default":
                continue
            if keyword.lower() in content_lower and target in allowed:
                return target
        # Fall back to "default" rule or first allowed
        default = self.routing_rules.get("default")
        if default and default in allowed:
            return default
        return allowed[0]

    def _llm_routed(self, content: str, allowed: list[str]) -> str:
        """Ask the LLM to pick the next agent."""
        if self._model is None:
            # No model available, fall back to rule-based
            return self._rule_based_route(content, allowed)

        prompt = (
            f"Given the following agent output, pick which agent should run next.\n"
            f"Allowed agents: {allowed}\n\n"
            f"Agent output:\n{content}\n\n"
            f"Reply with ONLY the agent name, nothing else."
        )
        try:
            from smolagents.types import ChatMessage

            messages = [ChatMessage(role="user", content=prompt)]
            response = self._model.generate(messages)
            choice = response.content.strip().lower()
            # Match against allowed agents
            for name in allowed:
                if name.lower() in choice:
                    return name
        except Exception:
            pass
        # Fallback
        return self._rule_based_route(content, allowed)

    def is_complete(self, history: list, current_state: dict) -> bool:
        if not history:
            return False
        last = history[-1]
        last_agent = last.agent_name if isinstance(last, AgentMessage) else ""
        return not bool(self.transitions.get(last_agent))
