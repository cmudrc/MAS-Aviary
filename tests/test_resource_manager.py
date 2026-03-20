"""Unit tests for the resource manager — budget tracking and decrement."""


from src.coordination.graph_definition import ResourceBudget
from src.coordination.resource_manager import ResourceManager

# ---- Helpers ---------------------------------------------------------------

def _budgets() -> dict[str, ResourceBudget]:
    return {
        "simple": ResourceBudget(
            max_passes=6, context_budget=2000, reasoning_enabled=False,
            max_code_review_cycles=1, escalation_threshold=2,
        ),
        "moderate": ResourceBudget(
            max_passes=12, context_budget=3000, reasoning_enabled=True,
            max_code_review_cycles=2, escalation_threshold=3,
        ),
        "complex": ResourceBudget(
            max_passes=20, context_budget=4000, reasoning_enabled=True,
            max_code_review_cycles=3, escalation_threshold=4,
        ),
    }


# ---- Initialization -------------------------------------------------------

class TestInit:
    def test_default_state(self):
        rm = ResourceManager()
        assert rm.state.passes_remaining == 10
        assert rm.state.context_used == 0
        assert rm.complexity is None

    def test_set_simple_complexity(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("simple")
        assert rm.state.passes_remaining == 6
        assert rm.state.passes_max == 6
        assert rm.state.context_budget == 2000
        assert rm.state.reasoning_enabled is False
        assert rm.state.max_code_review_cycles == 1
        assert rm.state.escalation_threshold == 2
        assert rm.complexity == "simple"

    def test_set_moderate_complexity(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("moderate")
        assert rm.state.passes_remaining == 12
        assert rm.state.reasoning_enabled is True

    def test_set_complex_complexity(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("complex")
        assert rm.state.passes_remaining == 20
        assert rm.state.context_budget == 4000
        assert rm.state.max_code_review_cycles == 3

    def test_unknown_complexity_uses_defaults(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("unknown")
        assert rm.complexity == "unknown"
        assert rm.state.passes_remaining == 10  # default unchanged


# ---- Pass consumption ------------------------------------------------------

class TestPassConsumption:
    def test_consume_pass(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("simple")
        rm.consume_pass()
        assert rm.state.passes_remaining == 5

    def test_consume_all_passes(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("simple")
        for _ in range(6):
            rm.consume_pass()
        assert rm.state.passes_remaining == 0

    def test_consume_past_zero(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("simple")
        for _ in range(10):
            rm.consume_pass()
        assert rm.state.passes_remaining == 0

    def test_has_passes(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("simple")
        assert rm.has_passes() is True
        for _ in range(6):
            rm.consume_pass()
        assert rm.has_passes() is False


# ---- Context tracking ------------------------------------------------------

class TestContextTracking:
    def test_add_context(self):
        rm = ResourceManager()
        rm.add_context(100)
        assert rm.state.context_used == 100

    def test_add_context_cumulative(self):
        rm = ResourceManager()
        rm.add_context(100)
        rm.add_context(200)
        assert rm.state.context_used == 300


# ---- Code review cycles ---------------------------------------------------

class TestCodeReviewCycles:
    def test_increment_code_review(self):
        rm = ResourceManager()
        rm.increment_code_review()
        assert rm.state.code_review_cycles == 1

    def test_multiple_increments(self):
        rm = ResourceManager()
        rm.increment_code_review()
        rm.increment_code_review()
        assert rm.state.code_review_cycles == 2


# ---- Cycle counting and escalation ----------------------------------------

class TestCycleTracking:
    def test_record_design_state_first_visit(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("moderate")
        is_cycle = rm.record_state_entry("DESIGN_PLANNED")
        assert is_cycle is False
        assert rm.state.cycle_count == 0

    def test_record_design_state_revisit(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("moderate")
        rm.record_state_entry("DESIGN_PLANNED")
        is_cycle = rm.record_state_entry("DESIGN_PLANNED")
        assert is_cycle is True
        assert rm.state.cycle_count == 1

    def test_different_design_states_no_cycle(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("moderate")
        rm.record_state_entry("QUICK_DESIGN")
        is_cycle = rm.record_state_entry("DESIGN_PLANNED")
        assert is_cycle is False
        assert rm.state.cycle_count == 0

    def test_non_design_state_no_cycle(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("moderate")
        rm.record_state_entry("CODE_WRITTEN")
        is_cycle = rm.record_state_entry("CODE_WRITTEN")
        # CODE_WRITTEN is not a design state, so no cycle tracking.
        assert is_cycle is False
        assert rm.state.cycle_count == 0

    def test_escalation_at_threshold(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("simple")  # threshold=2
        rm.record_state_entry("DESIGN_PLANNED")
        rm.record_state_entry("DESIGN_PLANNED")  # cycle 1
        assert rm.should_escalate() is False
        rm.record_state_entry("DESIGN_PLANNED")  # cycle 2
        assert rm.should_escalate() is True

    def test_reset_cycles_after_escalation(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("simple")
        rm.record_state_entry("DESIGN_PLANNED")
        rm.record_state_entry("DESIGN_PLANNED")  # cycle 1
        rm.record_state_entry("DESIGN_PLANNED")  # cycle 2
        assert rm.should_escalate() is True
        rm.reset_cycles_after_escalation()
        assert rm.state.cycle_count == 0
        assert rm.should_escalate() is False


# ---- Escalation budget adjustment -----------------------------------------

class TestEscalationBudgets:
    def test_upgrade_simple_to_moderate(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("simple")
        assert rm.state.passes_max == 6
        # Simulate some passes used.
        rm.consume_pass()
        rm.consume_pass()
        assert rm.state.passes_remaining == 4

        rm.set_complexity("moderate")
        # passes_max increased from 6 to 12 (+6), so remaining += 6
        assert rm.state.passes_max == 12
        assert rm.state.passes_remaining == 10  # 4 + 6
        assert rm.state.context_budget == 3000
        assert rm.state.reasoning_enabled is True
        assert rm.state.max_code_review_cycles == 2

    def test_upgrade_moderate_to_complex(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("moderate")
        rm.consume_pass()  # 11 remaining
        rm.set_complexity("complex")
        # passes_max 12→20 (+8), remaining 11+8=19
        assert rm.state.passes_max == 20
        assert rm.state.passes_remaining == 19
        assert rm.state.context_budget == 4000

    def test_no_downgrade_on_same_level(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("moderate")
        rm.consume_pass()  # 11 remaining
        rm.set_complexity("moderate")  # re-set same level
        # No change since moderate == current, no upgrade
        assert rm.state.passes_remaining == 11
        assert rm.state.passes_max == 12

    def test_no_downgrade_complex_to_simple(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("complex")
        rm.consume_pass()  # 19 remaining
        rm.set_complexity("simple")
        # Simple has lower budgets — should NOT decrease.
        assert rm.state.passes_max == 20
        assert rm.state.passes_remaining == 19
        assert rm.state.context_budget == 4000
        assert rm.state.reasoning_enabled is True

    def test_reasoning_only_enabled_not_disabled(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("moderate")
        assert rm.state.reasoning_enabled is True
        rm.set_complexity("simple")
        # Simple has reasoning_enabled=False, but we don't downgrade.
        assert rm.state.reasoning_enabled is True


# ---- State dict export -----------------------------------------------------

class TestStateDictExport:
    def test_to_state_dict_keys(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("moderate")
        d = rm.to_state_dict()
        assert "passes_remaining" in d
        assert "passes_max" in d
        assert "context_used" in d
        assert "context_budget" in d
        assert "reasoning_enabled" in d
        assert "code_review_cycles" in d
        assert "max_code_review_cycles" in d
        assert "cycle_count" in d
        assert "escalation_threshold" in d

    def test_to_state_dict_values(self):
        rm = ResourceManager(budgets=_budgets())
        rm.set_complexity("moderate")
        rm.consume_pass()
        rm.add_context(500)
        d = rm.to_state_dict()
        assert d["passes_remaining"] == 11
        assert d["context_used"] == 500
        assert d["context_budget"] == 3000
