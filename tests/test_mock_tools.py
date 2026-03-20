"""Tests for mock tool definitions."""

import pytest
from smolagents import Tool

from src.tools.mock_tools import (
    CalculatorTool,
    EchoTool,
    StateTool,
    clear_invocation_log,
    create_mock_tool,
    get_invocation_log,
)


@pytest.fixture(autouse=True)
def _clean_log():
    """Clear the invocation log before each test."""
    clear_invocation_log()
    yield
    clear_invocation_log()


# ---- Echo Tool ----------------------------------------------------------------

class TestEchoTool:
    def test_is_smolagents_tool(self):
        assert isinstance(EchoTool(), Tool)

    def test_returns_input_unchanged(self):
        tool = EchoTool()
        assert tool("hello world") == "hello world"

    def test_empty_string(self):
        tool = EchoTool()
        assert tool("") == ""

    def test_preserves_special_chars(self):
        tool = EchoTool()
        assert tool("line1\nline2\ttab") == "line1\nline2\ttab"

    def test_logs_invocation(self):
        tool = EchoTool()
        tool("test")
        log = get_invocation_log()
        assert len(log) == 1
        assert log[0]["tool_name"] == "echo_tool"
        assert log[0]["inputs"] == {"message": "test"}
        assert log[0]["output"] == "test"
        assert "timestamp" in log[0]
        assert "duration_seconds" in log[0]


# ---- Calculator Tool ----------------------------------------------------------

class TestCalculatorTool:
    def test_is_smolagents_tool(self):
        assert isinstance(CalculatorTool(), Tool)

    def test_addition(self):
        tool = CalculatorTool()
        assert tool("2 + 3") == "5"

    def test_multiplication(self):
        tool = CalculatorTool()
        assert tool("6 * 7") == "42"

    def test_complex_expression(self):
        tool = CalculatorTool()
        assert tool("2 + 3 * 4") == "14"

    def test_float_division(self):
        tool = CalculatorTool()
        assert tool("10 / 3") == str(10 / 3)

    def test_floor_division(self):
        tool = CalculatorTool()
        assert tool("10 // 3") == "3"

    def test_exponentiation(self):
        tool = CalculatorTool()
        assert tool("2 ** 10") == "1024"

    def test_negative_numbers(self):
        tool = CalculatorTool()
        assert tool("-5 + 3") == "-2"

    def test_parentheses(self):
        tool = CalculatorTool()
        assert tool("(2 + 3) * 4") == "20"

    def test_invalid_expression_returns_error(self):
        tool = CalculatorTool()
        result = tool("not a number")
        assert result.startswith("Error:")

    def test_rejects_function_calls(self):
        """Calculator must not allow arbitrary function calls like __import__."""
        tool = CalculatorTool()
        result = tool("__import__('os').system('echo pwned')")
        assert result.startswith("Error:")

    def test_logs_invocation(self):
        tool = CalculatorTool()
        tool("1 + 1")
        log = get_invocation_log()
        assert len(log) == 1
        assert log[0]["tool_name"] == "calculator_tool"
        assert log[0]["inputs"] == {"expression": "1 + 1"}
        assert log[0]["output"] == "2"


# ---- State Tool ---------------------------------------------------------------

class TestStateTool:
    def test_is_smolagents_tool(self):
        assert isinstance(StateTool(), Tool)

    def test_increments_each_call(self):
        tool = StateTool()
        assert tool() == "1"
        assert tool() == "2"
        assert tool() == "3"

    def test_independent_instances(self):
        """Two StateTool instances have independent counters."""
        a = StateTool()
        b = StateTool()
        assert a() == "1"
        assert a() == "2"
        assert b() == "1"  # b starts at 0

    def test_reset(self):
        tool = StateTool()
        tool()
        tool()
        tool.reset()
        assert tool() == "1"

    def test_logs_each_invocation(self):
        tool = StateTool()
        tool()
        tool()
        log = get_invocation_log()
        assert len(log) == 2
        assert log[0]["output"] == "1"
        assert log[1]["output"] == "2"


# ---- create_mock_tool factory -------------------------------------------------

class TestCreateMockTool:
    def test_creates_echo(self):
        tool = create_mock_tool("echo_tool")
        assert isinstance(tool, EchoTool)

    def test_creates_calculator(self):
        tool = create_mock_tool("calculator_tool")
        assert isinstance(tool, CalculatorTool)

    def test_creates_state(self):
        tool = create_mock_tool("state_tool")
        assert isinstance(tool, StateTool)

    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown mock tool"):
            create_mock_tool("nonexistent_tool")
