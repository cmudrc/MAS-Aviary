"""Feedback extraction from agent turns for the Iterative Feedback handler.

Extracts structured feedback (tool call outcomes, errors, return codes,
output content) from AgentMessage objects produced by agent.run() calls.
"""

import json
import re
from dataclasses import dataclass, field


@dataclass
class ToolCallOutcome:
    """Structured result of a single tool invocation."""

    tool_name: str
    success: bool
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error_type: str | None = None
    execution_time: float | None = None


@dataclass
class AttemptFeedback:
    """Structured feedback extracted from one agent attempt."""

    attempt_number: int
    tool_calls: list[ToolCallOutcome] = field(default_factory=list)
    has_tool_errors: bool = False
    error_messages: list[str] = field(default_factory=list)
    return_codes: list[int] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    output_content: str = ""


# Common Python exception type patterns.
_ERROR_TYPE_RE = re.compile(
    r"\b([A-Z][a-zA-Z]*(?:Error|Exception|Warning|Fault))\b"
)


def _extract_error_type(text: str) -> str | None:
    """Extract the first recognisable Python error type from a string."""
    m = _ERROR_TYPE_RE.search(text)
    return m.group(1) if m else None


def _parse_tool_output(output: str) -> dict:
    """Try to parse a tool output string as JSON; return raw dict otherwise."""
    try:
        return json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return {}


def extract_feedback(message, attempt_number: int = 0) -> AttemptFeedback:
    """Extract structured feedback from an AgentMessage.

    Args:
        message: An AgentMessage (from src.coordination.history).
        attempt_number: 0-indexed attempt counter for this agent.

    Returns:
        AttemptFeedback populated from the message's tool_calls and content.
    """
    outcomes: list[ToolCallOutcome] = []
    error_messages: list[str] = []
    return_codes: list[int] = []
    all_stdout: list[str] = []
    all_stderr: list[str] = []

    for tc in getattr(message, "tool_calls", []):
        # ToolCallRecord fields: tool_name, inputs, output, duration_seconds, error
        parsed = _parse_tool_output(tc.output)

        # Determine success: explicit "success" key, absence of error, or
        # return_code == 0.  Also check "valid" key — tools like
        # validate_parameters return {"success": true, "valid": false}
        # where "success" means the tool ran but "valid" means the
        # domain-level check failed.  Treat valid=false as a failure so
        # the retry loop keeps the agent working until validation passes.
        if tc.error:
            success = False
        elif isinstance(parsed, dict) and "success" in parsed:
            success = bool(parsed["success"])
            # Override: if the tool succeeded but validation failed,
            # treat as failure so the handler retries.
            if success and "valid" in parsed and not parsed["valid"]:
                success = False
        elif isinstance(parsed, dict) and "valid" in parsed:
            success = bool(parsed["valid"])
        elif isinstance(parsed, dict) and "return_code" in parsed:
            success = parsed["return_code"] == 0
        else:
            success = True

        rc = parsed.get("return_code") if isinstance(parsed, dict) else None
        stdout = parsed.get("stdout", "") if isinstance(parsed, dict) else ""
        stderr = parsed.get("stderr", "") if isinstance(parsed, dict) else ""

        error_type: str | None = None
        if tc.error:
            error_type = _extract_error_type(tc.error)
            error_messages.append(tc.error)
        elif stderr:
            error_type = _extract_error_type(stderr)
            if error_type:
                error_messages.append(stderr)

        if not success and not tc.error and not error_type:
            # Tool returned failure but no explicit error string.
            # Check for validation errors first (validate_parameters).
            if isinstance(parsed, dict) and "valid" in parsed and not parsed["valid"]:
                val_errors = parsed.get("errors", [])
                if val_errors:
                    error_messages.append(
                        f"Validation failed: {'; '.join(str(e) for e in val_errors)}"
                    )
                else:
                    error_messages.append("Validation failed: valid=false")
                error_type = "ValidationError"
            else:
                msg = parsed.get("stderr") or parsed.get("error") or tc.output
                if msg:
                    error_messages.append(str(msg))

        if rc is not None:
            return_codes.append(rc)
        if stdout:
            all_stdout.append(stdout)
        if stderr:
            all_stderr.append(stderr)

        outcomes.append(ToolCallOutcome(
            tool_name=tc.tool_name,
            success=success,
            return_code=rc,
            stdout=stdout,
            stderr=stderr,
            error_type=error_type,
            execution_time=tc.duration_seconds,
        ))

    # Also check top-level message error.
    if getattr(message, "error", None):
        error_messages.append(message.error)

    has_tool_errors = any(not o.success for o in outcomes)
    # If no tool calls were made but the message has an error, flag it.
    if not outcomes and getattr(message, "error", None):
        has_tool_errors = True

    return AttemptFeedback(
        attempt_number=attempt_number,
        tool_calls=outcomes,
        has_tool_errors=has_tool_errors,
        error_messages=error_messages,
        return_codes=return_codes,
        stdout="\n".join(all_stdout),
        stderr="\n".join(all_stderr),
        output_content=getattr(message, "content", ""),
    )


def format_feedback_for_retry(feedback: AttemptFeedback, max_retries: int) -> str:
    """Format feedback into a human-readable string for injection into retry context.

    Args:
        feedback: The AttemptFeedback from the previous attempt.
        max_retries: Total allowed retries (for display).

    Returns:
        A formatted string describing the previous attempt's outcome.
    """
    lines = [
        f"Your previous attempt (attempt {feedback.attempt_number + 1} of "
        f"{max_retries}) produced the following result:",
        "",
    ]

    if feedback.tool_calls:
        for tc in feedback.tool_calls:
            lines.append(f"Tool call: {tc.tool_name}")
            lines.append(f"  Success: {str(tc.success).lower()}")
            if tc.return_code is not None:
                lines.append(f"  Return code: {tc.return_code}")
            if tc.error_type:
                lines.append(f"  Error: {tc.error_type}")
            if tc.stderr:
                lines.append(f"  Stderr: {tc.stderr}")
            lines.append("")
    elif feedback.error_messages:
        for err in feedback.error_messages:
            lines.append(f"Error: {err}")
        lines.append("")

    if feedback.has_tool_errors:
        lines.append("Try a different approach to fix this error.")
    else:
        lines.append("The previous attempt completed without tool errors.")

    return "\n".join(lines)
