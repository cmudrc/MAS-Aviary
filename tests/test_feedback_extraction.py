"""Tests for feedback extraction from AgentMessage objects.

No GPU needed. Tests extraction of tool call outcomes, error types,
return codes, stdout/stderr from various AgentMessage shapes.
"""

import json

import pytest

from src.coordination.feedback_extraction import (
    AttemptFeedback,
    ToolCallOutcome,
    _extract_error_type,
    extract_feedback,
    format_feedback_for_retry,
)
from src.coordination.history import AgentMessage, ToolCallRecord


def _msg(
    content: str = "",
    tool_calls: list[ToolCallRecord] | None = None,
    error: str | None = None,
) -> AgentMessage:
    """Helper to build a minimal AgentMessage."""
    return AgentMessage(
        agent_name="test_agent",
        content=content,
        turn_number=1,
        timestamp=1000.0,
        tool_calls=tool_calls or [],
        error=error,
    )


def _tc(
    tool_name: str = "my_tool",
    inputs: dict | None = None,
    output: str = "",
    duration: float = 1.0,
    error: str | None = None,
) -> ToolCallRecord:
    """Helper to build a ToolCallRecord."""
    return ToolCallRecord(
        tool_name=tool_name,
        inputs=inputs or {},
        output=output,
        duration_seconds=duration,
        error=error,
    )


# ---- Successful tool calls ---------------------------------------------------

class TestSuccessfulToolCalls:
    def test_single_successful_tool_call(self):
        msg = _msg(
            content="Done",
            tool_calls=[_tc(output=json.dumps({"success": True, "return_code": 0}))],
        )
        fb = extract_feedback(msg, attempt_number=0)
        assert not fb.has_tool_errors
        assert len(fb.tool_calls) == 1
        assert fb.tool_calls[0].success is True
        assert fb.tool_calls[0].return_code == 0
        assert fb.error_messages == []

    def test_multiple_successful_tool_calls(self):
        msg = _msg(
            content="All good",
            tool_calls=[
                _tc(tool_name="tool_a", output=json.dumps({"success": True})),
                _tc(tool_name="tool_b", output=json.dumps({"success": True})),
            ],
        )
        fb = extract_feedback(msg, attempt_number=1)
        assert not fb.has_tool_errors
        assert len(fb.tool_calls) == 2
        assert fb.attempt_number == 1

    def test_tool_with_no_explicit_success_field(self):
        """Tool output without 'success' key and no error → treated as success."""
        msg = _msg(
            content="result",
            tool_calls=[_tc(output="plain text output")],
        )
        fb = extract_feedback(msg)
        assert not fb.has_tool_errors
        assert fb.tool_calls[0].success is True


# ---- Failed tool calls -------------------------------------------------------

class TestFailedToolCalls:
    def test_tool_call_with_error_field(self):
        msg = _msg(
            content="",
            tool_calls=[_tc(error="AttributeError: no attribute 'fillet2D'")],
        )
        fb = extract_feedback(msg)
        assert fb.has_tool_errors
        assert fb.tool_calls[0].success is False
        assert fb.tool_calls[0].error_type == "AttributeError"
        assert len(fb.error_messages) == 1

    def test_tool_call_with_success_false(self):
        msg = _msg(
            content="",
            tool_calls=[_tc(output=json.dumps({
                "success": False,
                "return_code": 1,
                "stderr": "Traceback: ValueError: bad input",
            }))],
        )
        fb = extract_feedback(msg)
        assert fb.has_tool_errors
        assert fb.tool_calls[0].success is False
        assert fb.tool_calls[0].return_code == 1
        assert fb.return_codes == [1]
        assert "ValueError" in (fb.tool_calls[0].error_type or "")

    def test_tool_call_with_nonzero_return_code(self):
        msg = _msg(
            content="",
            tool_calls=[_tc(output=json.dumps({"return_code": 2}))],
        )
        fb = extract_feedback(msg)
        assert fb.has_tool_errors
        assert fb.tool_calls[0].return_code == 2

    def test_mixed_success_and_failure(self):
        msg = _msg(
            content="partial",
            tool_calls=[
                _tc(tool_name="ok_tool", output=json.dumps({"success": True})),
                _tc(tool_name="bad_tool", error="TimeoutError: timed out"),
            ],
        )
        fb = extract_feedback(msg)
        assert fb.has_tool_errors  # at least one failure
        assert fb.tool_calls[0].success is True
        assert fb.tool_calls[1].success is False


# ---- No tool calls -----------------------------------------------------------

class TestNoToolCalls:
    def test_message_with_no_tool_calls(self):
        msg = _msg(content="Just reasoning text")
        fb = extract_feedback(msg)
        assert not fb.has_tool_errors
        assert fb.tool_calls == []
        assert fb.output_content == "Just reasoning text"

    def test_message_with_no_tool_calls_but_error(self):
        msg = _msg(content="", error="Agent crashed")
        fb = extract_feedback(msg)
        assert fb.has_tool_errors
        assert fb.error_messages == ["Agent crashed"]

    def test_empty_message(self):
        msg = _msg(content="")
        fb = extract_feedback(msg)
        assert not fb.has_tool_errors
        assert fb.output_content == ""


# ---- Error type extraction ---------------------------------------------------

class TestErrorTypeExtraction:
    @pytest.mark.parametrize("text,expected", [
        ("AttributeError: 'Workplane' has no attribute 'exportStl'", "AttributeError"),
        ("TimeoutError: connection timed out", "TimeoutError"),
        ("ValueError: invalid literal", "ValueError"),
        ("RuntimeError: CUDA out of memory", "RuntimeError"),
        ("KeyError: 'missing_key'", "KeyError"),
        ("ModuleNotFoundError: No module named 'foo'", "ModuleNotFoundError"),
        ("no error here", None),
        ("", None),
    ])
    def test_extract_error_type(self, text, expected):
        assert _extract_error_type(text) == expected


# ---- Stdout / Stderr extraction -----------------------------------------------

class TestStdoutStderr:
    def test_extracts_stdout_and_stderr(self):
        output = json.dumps({
            "success": True,
            "stdout": "Created results.json",
            "stderr": "Warning: deprecated API",
            "return_code": 0,
        })
        msg = _msg(content="done", tool_calls=[_tc(output=output)])
        fb = extract_feedback(msg)
        assert fb.stdout == "Created results.json"
        assert fb.stderr == "Warning: deprecated API"

    def test_concatenates_multi_tool_stdout(self):
        t1 = _tc(output=json.dumps({"success": True, "stdout": "line1"}))
        t2 = _tc(output=json.dumps({"success": True, "stdout": "line2"}))
        msg = _msg(content="done", tool_calls=[t1, t2])
        fb = extract_feedback(msg)
        assert "line1" in fb.stdout
        assert "line2" in fb.stdout


# ---- Execution time ----------------------------------------------------------

class TestExecutionTime:
    def test_records_execution_time(self):
        msg = _msg(
            content="done",
            tool_calls=[_tc(duration=3.14)],
        )
        fb = extract_feedback(msg)
        assert fb.tool_calls[0].execution_time == 3.14


# ---- Format for retry --------------------------------------------------------

class TestFormatFeedbackForRetry:
    def test_formats_successful_feedback(self):
        fb = AttemptFeedback(
            attempt_number=0,
            tool_calls=[ToolCallOutcome(
                tool_name="calculator", success=True,
                return_code=0, stdout="42",
            )],
            has_tool_errors=False,
            output_content="42",
        )
        text = format_feedback_for_retry(fb, max_retries=20)
        assert "attempt 1 of 20" in text
        assert "calculator" in text
        assert "completed without tool errors" in text

    def test_formats_failed_feedback(self):
        fb = AttemptFeedback(
            attempt_number=2,
            tool_calls=[ToolCallOutcome(
                tool_name="run_simulation",
                success=False,
                return_code=1,
                stderr="AttributeError: no attr",
                error_type="AttributeError",
            )],
            has_tool_errors=True,
            error_messages=["AttributeError: no attr"],
            return_codes=[1],
            output_content="",
        )
        text = format_feedback_for_retry(fb, max_retries=20)
        assert "attempt 3 of 20" in text
        assert "run_simulation" in text
        assert "AttributeError" in text
        assert "different approach" in text

    def test_formats_no_tool_calls_with_error(self):
        fb = AttemptFeedback(
            attempt_number=0,
            has_tool_errors=True,
            error_messages=["Agent crashed"],
            output_content="",
        )
        text = format_feedback_for_retry(fb, max_retries=5)
        assert "Agent crashed" in text


# ---- Graceful handling of missing fields --------------------------------------

class TestGracefulHandling:
    def test_handles_object_without_tool_calls(self):
        """Message-like object without tool_calls attribute."""
        class BareMessage:
            content = "hello"
        fb = extract_feedback(BareMessage())
        assert fb.output_content == "hello"
        assert not fb.has_tool_errors

    def test_handles_object_without_error(self):
        """Message-like object without error attribute."""
        class BareMessage:
            content = "hello"
            tool_calls = []
        fb = extract_feedback(BareMessage())
        assert fb.error_messages == []
