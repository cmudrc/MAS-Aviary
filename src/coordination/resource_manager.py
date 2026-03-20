"""Resource manager for the Graph-Routed execution handler.

Tracks and decrements resource budgets (passes, context window,
review cycles) based on complexity classification.  Budgets are
set from ``ResourceBudget`` dataclasses and can be upgraded (never
downgraded) on complexity escalation.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.coordination.graph_definition import ResourceBudget


@dataclass
class ResourceState:
    """Current resource state during graph execution.

    Attributes:
        passes_remaining: Decremented each time an agent runs.
        passes_max: Set by complexity budget.
        context_used: Tokens consumed across all agent turns.
        context_budget: Max tokens (complexity-dependent).
        reasoning_enabled: Whether agents get extended reasoning prompts.
        code_review_cycles: How many code review loops have happened.
        max_code_review_cycles: Cap on review loops (complexity-dependent).
        cycle_count: Times graph routed back to a design state.
        escalation_threshold: Cycles before complexity escalation triggers.
    """

    passes_remaining: int = 10
    passes_max: int = 10
    context_used: int = 0
    context_budget: int = 3000
    reasoning_enabled: bool = True
    code_review_cycles: int = 0
    max_code_review_cycles: int = 2
    cycle_count: int = 0
    escalation_threshold: int = 3


# States that count as "design" states for cycle counting.
_DESIGN_STATES = frozenset(
    {
        "QUICK_DESIGN",
        "DESIGN_PLANNED",
        "DESIGN_DECOMPOSED",
    }
)


class ResourceManager:
    """Manages resource budgets during graph-routed execution.

    Args:
        budgets: Mapping from complexity level to ``ResourceBudget``.
        design_states: Set of state names considered "design" states
            for cycle counting.  Defaults to common design states.
    """

    def __init__(
        self,
        budgets: dict[str, ResourceBudget] | None = None,
        design_states: frozenset[str] | None = None,
    ):
        self._budgets = budgets or {}
        self._design_states = design_states or _DESIGN_STATES
        self._state = ResourceState()
        self._complexity: str | None = None
        self._visited_design_states: set[str] = set()

    @property
    def state(self) -> ResourceState:
        """Current resource state (read-only access)."""
        return self._state

    @property
    def complexity(self) -> str | None:
        """Current complexity classification."""
        return self._complexity

    def set_complexity(self, complexity: str) -> None:
        """Set (or upgrade) the complexity level and apply budgets.

        On initial classification, budgets are set directly from the
        budget table.  On escalation (re-classification), budgets are
        only upgraded — never decreased.
        """
        budget = self._budgets.get(complexity)
        if budget is None:
            # No budget for this complexity — use defaults.
            self._complexity = complexity
            return

        if self._complexity is None:
            # First classification — apply directly.
            self._state.passes_remaining = budget.max_passes
            self._state.passes_max = budget.max_passes
            self._state.context_budget = budget.context_budget
            self._state.reasoning_enabled = budget.reasoning_enabled
            self._state.max_code_review_cycles = budget.max_code_review_cycles
            self._state.escalation_threshold = budget.escalation_threshold
        else:
            # Escalation — only increase, never decrease.
            if budget.max_passes > self._state.passes_max:
                additional = budget.max_passes - self._state.passes_max
                self._state.passes_remaining += additional
                self._state.passes_max = budget.max_passes
            if budget.context_budget > self._state.context_budget:
                self._state.context_budget = budget.context_budget
            if budget.max_code_review_cycles > self._state.max_code_review_cycles:
                self._state.max_code_review_cycles = budget.max_code_review_cycles
            if budget.escalation_threshold > self._state.escalation_threshold:
                self._state.escalation_threshold = budget.escalation_threshold
            # Reasoning can only be enabled, not disabled on escalation.
            if budget.reasoning_enabled:
                self._state.reasoning_enabled = True

        self._complexity = complexity

    def consume_pass(self) -> None:
        """Decrement passes_remaining by 1 (agent ran at a state)."""
        self._state.passes_remaining = max(0, self._state.passes_remaining - 1)

    def add_context(self, tokens: int) -> None:
        """Add tokens to the context_used counter."""
        self._state.context_used += tokens

    def increment_code_review(self) -> None:
        """Increment the code review cycle counter."""
        self._state.code_review_cycles += 1

    def increment_cycle(self) -> None:
        """Increment the design-loop cycle counter."""
        self._state.cycle_count += 1

    def reset_cycles_after_escalation(self) -> None:
        """Reset cycle count after a complexity escalation."""
        self._state.cycle_count = 0
        self._visited_design_states.clear()

    def record_state_entry(self, state_name: str) -> bool:
        """Record entry into a state.  Returns True if this is a
        re-entry into a design state (i.e. a cycle).
        """
        if state_name in self._design_states:
            if state_name in self._visited_design_states:
                self.increment_cycle()
                return True
            self._visited_design_states.add(state_name)
        return False

    def should_escalate(self) -> bool:
        """Check whether the cycle count has reached the escalation threshold."""
        return self._state.cycle_count >= self._state.escalation_threshold

    def has_passes(self) -> bool:
        """Check if passes remain."""
        return self._state.passes_remaining > 0

    def to_state_dict(self) -> dict:
        """Export current resource state as a flat dict for condition evaluation."""
        return {
            "passes_remaining": self._state.passes_remaining,
            "passes_max": self._state.passes_max,
            "context_used": self._state.context_used,
            "context_budget": self._state.context_budget,
            "reasoning_enabled": self._state.reasoning_enabled,
            "code_review_cycles": self._state.code_review_cycles,
            "max_code_review_cycles": self._state.max_code_review_cycles,
            "cycle_count": self._state.cycle_count,
            "escalation_threshold": self._state.escalation_threshold,
        }
