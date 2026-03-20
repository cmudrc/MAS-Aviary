"""Unit tests for the condition evaluator — safe expression parser."""

import pytest

from src.coordination.condition_evaluator import (
    ConditionParseError,
    evaluate_condition,
    parse_condition,
)

# ---- Equality tests --------------------------------------------------------

class TestEquality:
    def test_string_equality_match(self):
        result = evaluate_condition("complexity == 'simple'", {"complexity": "simple"})
        assert result.matched is True

    def test_string_equality_no_match(self):
        result = evaluate_condition("complexity == 'simple'", {"complexity": "complex"})
        assert result.matched is False

    def test_string_inequality(self):
        result = evaluate_condition("complexity != 'simple'", {"complexity": "complex"})
        assert result.matched is True

    def test_string_inequality_no_match(self):
        result = evaluate_condition("complexity != 'simple'", {"complexity": "simple"})
        assert result.matched is False

    def test_double_quoted_string(self):
        result = evaluate_condition('complexity == "simple"', {"complexity": "simple"})
        assert result.matched is True

    def test_integer_equality(self):
        result = evaluate_condition("count == 5", {"count": 5})
        assert result.matched is True

    def test_integer_equality_no_match(self):
        result = evaluate_condition("count == 5", {"count": 3})
        assert result.matched is False


# ---- Numeric inequality tests ----------------------------------------------

class TestInequality:
    def test_greater_than_match(self):
        result = evaluate_condition("passes_remaining > 0", {"passes_remaining": 5})
        assert result.matched is True

    def test_greater_than_no_match(self):
        result = evaluate_condition("passes_remaining > 0", {"passes_remaining": 0})
        assert result.matched is False

    def test_less_than_match(self):
        result = evaluate_condition("passes_remaining < 3", {"passes_remaining": 2})
        assert result.matched is True

    def test_less_than_no_match(self):
        result = evaluate_condition("passes_remaining < 3", {"passes_remaining": 5})
        assert result.matched is False

    def test_greater_or_equal_match(self):
        result = evaluate_condition("cycle_count >= 3", {"cycle_count": 3})
        assert result.matched is True

    def test_greater_or_equal_boundary(self):
        result = evaluate_condition("cycle_count >= 3", {"cycle_count": 2})
        assert result.matched is False

    def test_less_or_equal_match(self):
        result = evaluate_condition("passes_remaining <= 0", {"passes_remaining": 0})
        assert result.matched is True

    def test_less_or_equal_no_match(self):
        result = evaluate_condition("passes_remaining <= 0", {"passes_remaining": 1})
        assert result.matched is False

    def test_negative_number(self):
        result = evaluate_condition("value > -1", {"value": 0})
        assert result.matched is True


# ---- Membership tests ------------------------------------------------------

class TestMembership:
    def test_in_list_match(self):
        result = evaluate_condition(
            "error_type in ['SyntaxError', 'NameError']",
            {"error_type": "SyntaxError"},
        )
        assert result.matched is True

    def test_in_list_second_element(self):
        result = evaluate_condition(
            "error_type in ['SyntaxError', 'NameError']",
            {"error_type": "NameError"},
        )
        assert result.matched is True

    def test_in_list_no_match(self):
        result = evaluate_condition(
            "error_type in ['SyntaxError', 'NameError']",
            {"error_type": "AttributeError"},
        )
        assert result.matched is False

    def test_in_empty_list(self):
        result = evaluate_condition("x in []", {"x": "anything"})
        assert result.matched is False

    def test_in_list_with_many_elements(self):
        result = evaluate_condition(
            "error_type in ['OCP_Error', 'StdError', 'GeometryError', 'TopologyError']",
            {"error_type": "GeometryError"},
        )
        assert result.matched is True


# ---- Boolean tests ---------------------------------------------------------

class TestBoolean:
    def test_boolean_true(self):
        result = evaluate_condition(
            "execution_success == true", {"execution_success": True}
        )
        assert result.matched is True

    def test_boolean_false(self):
        result = evaluate_condition(
            "execution_success == false", {"execution_success": False}
        )
        assert result.matched is True

    def test_boolean_mismatch(self):
        result = evaluate_condition(
            "execution_success == true", {"execution_success": False}
        )
        assert result.matched is False

    def test_review_passed_true(self):
        result = evaluate_condition(
            "review_passed == true", {"review_passed": True}
        )
        assert result.matched is True

    def test_review_passed_false(self):
        result = evaluate_condition(
            "review_passed == false", {"review_passed": False}
        )
        assert result.matched is True


# ---- Compound expressions --------------------------------------------------

class TestCompound:
    def test_and_both_true(self):
        result = evaluate_condition(
            "execution_success == true and stl_produced == true",
            {"execution_success": True, "stl_produced": True},
        )
        assert result.matched is True

    def test_and_one_false(self):
        result = evaluate_condition(
            "execution_success == true and stl_produced == true",
            {"execution_success": True, "stl_produced": False},
        )
        assert result.matched is False

    def test_and_both_false(self):
        result = evaluate_condition(
            "execution_success == true and stl_produced == true",
            {"execution_success": False, "stl_produced": False},
        )
        assert result.matched is False

    def test_or_first_true(self):
        result = evaluate_condition(
            "x == 1 or y == 2",
            {"x": 1, "y": 0},
        )
        assert result.matched is True

    def test_or_second_true(self):
        result = evaluate_condition(
            "x == 1 or y == 2",
            {"x": 0, "y": 2},
        )
        assert result.matched is True

    def test_or_neither_true(self):
        result = evaluate_condition(
            "x == 1 or y == 2",
            {"x": 0, "y": 0},
        )
        assert result.matched is False

    def test_triple_and(self):
        result = evaluate_condition(
            "a == 1 and b == 2 and c == 3",
            {"a": 1, "b": 2, "c": 3},
        )
        assert result.matched is True

    def test_mixed_and_or(self):
        """AND has higher precedence: ``a and b or c`` = ``(a and b) or c``."""
        result = evaluate_condition(
            "a == 1 and b == 2 or c == 3",
            {"a": 1, "b": 0, "c": 3},
        )
        assert result.matched is True  # (1==1 and 0==2)=False or (3==3)=True


# ---- Always ----------------------------------------------------------------

class TestAlways:
    def test_always_matches(self):
        result = evaluate_condition("always", {})
        assert result.matched is True

    def test_always_matches_with_state(self):
        result = evaluate_condition("always", {"x": 1, "y": 2})
        assert result.matched is True


# ---- Missing keys ----------------------------------------------------------

class TestMissingKeys:
    def test_missing_key_no_match(self):
        result = evaluate_condition("complexity == 'simple'", {})
        assert result.matched is False

    def test_missing_key_warning(self):
        result = evaluate_condition("complexity == 'simple'", {})
        assert len(result.warnings) > 0
        assert "complexity" in result.warnings[0]

    def test_missing_key_in_membership(self):
        result = evaluate_condition("error_type in ['SyntaxError']", {})
        assert result.matched is False
        assert len(result.warnings) > 0

    def test_missing_key_in_compound(self):
        result = evaluate_condition(
            "a == 1 and b == 2",
            {"b": 2},  # a is missing
        )
        assert result.matched is False


# ---- Invalid expressions ---------------------------------------------------

class TestInvalidExpressions:
    def test_empty_string(self):
        with pytest.raises(ConditionParseError):
            parse_condition("")

    def test_just_operator(self):
        with pytest.raises(ConditionParseError):
            parse_condition("==")

    def test_missing_right_side(self):
        with pytest.raises(ConditionParseError):
            parse_condition("x ==")

    def test_missing_list_close(self):
        with pytest.raises(ConditionParseError):
            parse_condition("x in ['a', 'b'")

    def test_trailing_token(self):
        with pytest.raises(ConditionParseError):
            parse_condition("x == 1 2")


# ---- Edge cases ------------------------------------------------------------

class TestEdgeCases:
    def test_float_comparison(self):
        result = evaluate_condition("score > 0.5", {"score": 0.8})
        assert result.matched is True

    def test_zero_equality(self):
        result = evaluate_condition("count == 0", {"count": 0})
        assert result.matched is True

    def test_string_with_underscore(self):
        result = evaluate_condition(
            "error_type == 'OCP_Error'", {"error_type": "OCP_Error"}
        )
        assert result.matched is True

    def test_review_verdict_passed(self):
        result = evaluate_condition(
            "review_verdict == 'passed'", {"review_verdict": "passed"}
        )
        assert result.matched is True

    def test_review_verdict_minor_issues(self):
        result = evaluate_condition(
            "review_verdict == 'minor_issues'", {"review_verdict": "minor_issues"}
        )
        assert result.matched is True


# ---- Variable references on the right side ---------------------------------

class TestVarRef:
    def test_var_ref_equality(self):
        result = evaluate_condition(
            "cycle_count >= escalation_threshold",
            {"cycle_count": 3, "escalation_threshold": 3},
        )
        assert result.matched is True

    def test_var_ref_inequality(self):
        result = evaluate_condition(
            "cycle_count >= escalation_threshold",
            {"cycle_count": 1, "escalation_threshold": 3},
        )
        assert result.matched is False

    def test_var_ref_missing_right_key(self):
        result = evaluate_condition(
            "cycle_count >= escalation_threshold",
            {"cycle_count": 3},
        )
        assert result.matched is False
        assert len(result.warnings) > 0

    def test_var_ref_in_compound(self):
        result = evaluate_condition(
            "code_review_cycles >= max_code_review_cycles and passes_remaining > 0",
            {"code_review_cycles": 2, "max_code_review_cycles": 2, "passes_remaining": 5},
        )
        assert result.matched is True
