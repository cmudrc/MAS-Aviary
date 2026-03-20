"""Graph definition dataclasses for the Graph-Routed execution handler.

Defines the structure of a state-machine graph: states, transitions,
and the overall graph definition. Provides loading from YAML/dict
and validation (all targets exist, terminal reachable, no orphans).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphTransition:
    """A single conditional transition from one state to another."""

    condition: str
    target: str


@dataclass
class GraphState:
    """A single state in the graph.

    Attributes:
        name: Unique state identifier (e.g. ``PROMPT_RECEIVED``).
        agent: Agent role to invoke at this state, or ``None`` for
            routing-only states (like ``ERROR_CLASSIFICATION``).
        agent_prompt: Instructions for the agent at this state.
            Can contain ``{variable}`` placeholders filled from the
            state dict.  ``None`` means the agent uses its default
            system prompt.
        description: Human-readable purpose description.
        transitions: Ordered list of conditional transitions.  First
            matching condition wins.
    """

    name: str
    agent: str | None
    description: str
    transitions: list[GraphTransition] = field(default_factory=list)
    agent_prompt: str | None = None


@dataclass
class ResourceBudget:
    """Resource budget for a single complexity level."""

    max_passes: int
    context_budget: int
    reasoning_enabled: bool
    max_code_review_cycles: int
    escalation_threshold: int


@dataclass
class GraphDefinition:
    """Complete graph definition including states, transitions, and budgets.

    Attributes:
        initial_state: Name of the starting state.
        terminal_states: Names of terminal states (execution stops here).
        states: Mapping from state name to ``GraphState``.
        resource_budgets: Mapping from complexity level to ``ResourceBudget``.
    """

    initial_state: str
    terminal_states: list[str]
    states: dict[str, GraphState]
    resource_budgets: dict[str, ResourceBudget] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_transition(data: dict) -> GraphTransition:
    """Parse a single transition dict."""
    return GraphTransition(
        condition=data["condition"],
        target=data["target"],
    )


def _load_state(name: str, data: dict) -> GraphState:
    """Parse a single state dict."""
    transitions = [_load_transition(t) for t in data.get("transitions", [])]
    return GraphState(
        name=name,
        agent=data.get("agent"),
        description=data.get("description", ""),
        transitions=transitions,
        agent_prompt=data.get("agent_prompt"),
    )


def _load_budget(data: dict) -> ResourceBudget:
    """Parse a resource budget dict."""
    return ResourceBudget(
        max_passes=data.get("max_passes", 10),
        context_budget=data.get("context_budget", 3000),
        reasoning_enabled=data.get("reasoning_enabled", True),
        max_code_review_cycles=data.get("max_code_review_cycles", 2),
        escalation_threshold=data.get("escalation_threshold", 3),
    )


def load_graph(data: dict) -> GraphDefinition:
    """Load a ``GraphDefinition`` from a plain dict (parsed YAML/JSON).

    Expected top-level keys: ``initial_state``, ``terminal_states``,
    ``states``, and optionally ``resource_budgets``.

    Raises:
        KeyError: If required keys are missing.
    """
    initial_state = data["initial_state"]
    terminal_states = data["terminal_states"]

    states: dict[str, GraphState] = {}
    for name, state_data in data.get("states", {}).items():
        states[name] = _load_state(name, state_data)

    budgets: dict[str, ResourceBudget] = {}
    for level, budget_data in data.get("resource_budgets", {}).items():
        budgets[level] = _load_budget(budget_data)

    return GraphDefinition(
        initial_state=initial_state,
        terminal_states=terminal_states,
        states=states,
        resource_budgets=budgets,
    )


def load_graph_from_yaml(path: str) -> GraphDefinition:
    """Load a graph definition from a YAML file.

    The YAML file may have the graph under a top-level key (e.g.
    ``aviary_graph:`` or ``graph:``) or directly at the root.
    """
    from src.config.loader import load_yaml

    raw = load_yaml(path)
    # Try common wrapper keys.
    for key in ("aviary_graph", "graph", "custom_graph"):
        if key in raw and isinstance(raw[key], dict):
            return load_graph(raw[key])
    # Assume the root *is* the graph definition.
    return load_graph(raw)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class GraphValidationError(Exception):
    """Raised when a graph definition fails validation."""


def validate_graph(graph: GraphDefinition) -> list[str]:
    """Validate a graph definition and return a list of error messages.

    An empty list means the graph is valid.  Checks performed:

    1. Initial state exists in states.
    2. All terminal states exist in states.
    3. Every transition target references an existing state.
    4. No orphan states (every non-initial state is reachable from
       at least one other state's transitions).
    5. At least one terminal state is reachable from the initial state.
    """
    errors: list[str] = []

    # 1. Initial state exists.
    if graph.initial_state not in graph.states:
        errors.append(
            f"Initial state {graph.initial_state!r} not found in states"
        )

    # 2. Terminal states exist.
    for ts in graph.terminal_states:
        if ts not in graph.states:
            errors.append(f"Terminal state {ts!r} not found in states")

    # 3. All transition targets exist.
    for state in graph.states.values():
        for trans in state.transitions:
            if trans.target not in graph.states:
                errors.append(
                    f"State {state.name!r} has transition to unknown "
                    f"state {trans.target!r}"
                )

    # 4. Orphan detection — every non-initial state must be reachable
    #    from at least one transition.
    referenced: set[str] = {graph.initial_state}
    for state in graph.states.values():
        for trans in state.transitions:
            referenced.add(trans.target)
    for name in graph.states:
        if name not in referenced:
            errors.append(f"State {name!r} is unreachable (orphan)")

    # 5. Terminal reachability via BFS from initial state.
    if graph.initial_state in graph.states:
        visited: set[str] = set()
        queue = [graph.initial_state]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if current in graph.states:
                for trans in graph.states[current].transitions:
                    if trans.target not in visited:
                        queue.append(trans.target)

        terminal_reachable = any(ts in visited for ts in graph.terminal_states)
        if not terminal_reachable:
            errors.append(
                "No terminal state is reachable from the initial state"
            )

    return errors


def validate_graph_strict(graph: GraphDefinition) -> None:
    """Validate a graph and raise ``GraphValidationError`` if invalid."""
    errors = validate_graph(graph)
    if errors:
        raise GraphValidationError(
            f"Graph validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


# ---------------------------------------------------------------------------
# Role resolution
# ---------------------------------------------------------------------------

# Splits identifiers on underscores, hyphens, whitespace, and camelCase
# boundaries (e.g. "codeReviewer" → ["code", "reviewer"]).
_ROLE_SPLIT_RE = re.compile(r"[_\s\-]+|(?<=[a-z])(?=[A-Z])")


def _tokenize(text: str) -> set[str]:
    """Split a name or description into lowercase tokens for fuzzy matching.

    Handles snake_case, camelCase, hyphens, and whitespace.  Drops tokens
    shorter than 3 characters to avoid spurious matches on articles and
    prepositions.

    CamelCase splitting runs *before* lowercasing so that boundaries like
    ``CADCodeGenerator`` → ``[CAD, Code, Generator]`` are preserved.
    """
    # Split on camelCase boundaries and explicit delimiters first, then lower.
    parts = _ROLE_SPLIT_RE.split(text)
    return {p.strip().lower() for p in parts if len(p.strip()) >= 3}


def _common_prefix_len(a: str, b: str) -> int:
    """Return the length of the common prefix between two strings."""
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _role_match_score(role_tokens: set[str], agent_tokens: set[str]) -> float:
    """Score how well role tokens overlap with agent tokens.

    Each role token contributes up to 1.0:
    - Exact match in agent tokens: 1.0
    - Shared prefix >= 4 chars with any agent token: 0.7
      (catches stems like code/coder, execute/executor, review/reviewer)

    Returns the average contribution across role tokens (0.0–1.0).
    """
    if not role_tokens:
        return 0.0

    total = 0.0
    for rt in role_tokens:
        if rt in agent_tokens:
            total += 1.0
            continue
        best_prefix = max(
            (_common_prefix_len(rt, at) for at in agent_tokens),
            default=0,
        )
        if best_prefix >= 4:
            total += 0.7

    return total / len(role_tokens)


def resolve_agent_for_role(
    role: str, agents: dict, state_name: str,
) -> Any:
    """Find the agent matching a role name from the agents dict.

    Resolution order:

    1. **Exact key match** — ``role`` is a key in ``agents``.
    2. **Exact name attribute** — an agent's ``.name`` equals ``role``.
    3. **Substring containment** — ``role`` appears inside the agent's
       name or description (or vice-versa).  Skips orchestrator agents.
    4. **Token overlap scoring** — tokenize role, agent name, and agent
       description; score by exact-token and prefix-stem matches.  Picks
       the highest-scoring agent above zero.  Skips orchestrator agents.

    Raises:
        ValueError: If no agent can be resolved for the role.
    """
    # 1. Exact key match.
    if role in agents:
        return agents[role]

    # 2. Attribute match (agent.name == role).
    for agent in agents.values():
        if getattr(agent, "name", None) == role:
            return agent

    # 3–4: Fuzzy matching (skip orchestrator-type agents).
    role_lower = role.lower().replace("_", " ")
    role_tokens = _tokenize(role)

    best_agent = None
    best_score = 0.0

    for key, agent in agents.items():
        name = getattr(agent, "name", key)

        # Skip orchestrator during fuzzy matching — it's a framework
        # agent, not a task role.
        if name == "orchestrator" or key == "orchestrator":
            continue

        desc = getattr(agent, "description", "") or ""
        name_lower = name.lower().replace("_", " ")
        desc_lower = desc.lower()

        # 3. Substring containment (strong signal → immediate return).
        if role_lower in name_lower or role_lower in desc_lower:
            return agent
        if name_lower in role_lower:
            return agent

        # 4. Token overlap scoring on name + description.
        agent_tokens = _tokenize(name) | _tokenize(desc)
        score = _role_match_score(role_tokens, agent_tokens)
        if score > best_score:
            best_score = score
            best_agent = agent

    # Accept fuzzy match if any token overlap was found.
    if best_agent is not None and best_score > 0:
        return best_agent

    raise ValueError(
        f"No agent available for role {role!r} at state {state_name!r}. "
        f"Available agents: {list(agents.keys())}"
    )
