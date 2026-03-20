"""Completion criteria evaluation for the Staged Pipeline handler.

Defines CompletionCriteria and CompletionResult dataclasses and the
evaluate_completion() function that checks agent output against per-stage
criteria.  Criteria are **observational only** — they NEVER block pipeline
advancement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CompletionCriteria:
    """Definition of what "done" means for a single pipeline stage."""

    type: str  # "output_contains", "tool_attempted", "any"
    check: str  # specific check within the type
    description: str = ""  # human-readable description
    tool_name: str | None = None  # for tool_attempted type


@dataclass
class CompletionResult:
    """Result of evaluating a completion criteria against agent output."""

    met: bool  # was the criteria satisfied
    reason: str = ""  # why it was/wasn't met
    evidence: str = ""  # what was found (or not found)


# ---------------------------------------------------------------------------
# Default patterns (overridable via config)
# ---------------------------------------------------------------------------

DEFAULT_CODE_BLOCK_PATTERNS: list[str] = [
    "```python",
    "import openmdao",
    "om.Problem",
    "prob.setup",
]

DEFAULT_VERDICT_PATTERNS: list[str] = [
    "ACCEPTABLE",
    "ISSUES",
    "FAILED",
    "TASK_COMPLETE",
    "passed",
    "acceptable",
    "failed",
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_completion(
    criteria: CompletionCriteria,
    content: str,
    tool_calls: list | None = None,
    *,
    code_block_patterns: list[str] | None = None,
    verdict_patterns: list[str] | None = None,
) -> CompletionResult:
    """Evaluate whether agent output meets the stage's completion criteria.

    This function is **observational only** — it produces a report, it does
    NOT block advancement.

    Args:
        criteria: The stage's completion criteria definition.
        content: Agent output text.
        tool_calls: List of ToolCallRecord (or dicts with ``tool_name``).
        code_block_patterns: Override default code-block detection patterns.
        verdict_patterns: Override default verdict-keyword detection patterns.

    Returns:
        CompletionResult with met/reason/evidence.
    """
    if criteria.type == "any" or criteria.check == "always":
        return CompletionResult(
            met=True,
            reason="Stage always completes (any/always criteria)",
            evidence="",
        )

    if criteria.type == "output_contains":
        return _check_output_contains(
            criteria.check,
            content,
            code_block_patterns=code_block_patterns,
            verdict_patterns=verdict_patterns,
        )

    if criteria.type == "tool_attempted":
        return _check_tool_attempted(
            criteria.tool_name,
            tool_calls or [],
        )

    # Unknown criteria type — report but do NOT block.
    return CompletionResult(
        met=False,
        reason=f"Unknown criteria type '{criteria.type}'",
        evidence="",
    )


# ---------------------------------------------------------------------------
# Internal checkers
# ---------------------------------------------------------------------------


def _check_output_contains(
    check: str,
    content: str,
    *,
    code_block_patterns: list[str] | None = None,
    verdict_patterns: list[str] | None = None,
) -> CompletionResult:
    """Evaluate an output_contains check."""

    if check == "non_empty_output":
        stripped = content.strip() if content else ""
        if stripped:
            return CompletionResult(
                met=True,
                reason="Agent produced non-empty output",
                evidence=f"Output length: {len(stripped)} chars",
            )
        return CompletionResult(
            met=False,
            reason="Agent produced empty or whitespace-only output",
            evidence="Output was empty after stripping whitespace",
        )

    if check == "code_block":
        patterns = code_block_patterns or DEFAULT_CODE_BLOCK_PATTERNS
        for pat in patterns:
            if pat in content:
                return CompletionResult(
                    met=True,
                    reason="Python code block found",
                    evidence=f"Matched pattern: {pat!r}",
                )
        # Also check import + def/assignment heuristic.
        if re.search(r"\bimport\b", content) and re.search(r"\b(def |class |\w+\s*=)", content):
            return CompletionResult(
                met=True,
                reason="Python code detected via import + definition pattern",
                evidence="Found 'import' and definition/assignment",
            )
        return CompletionResult(
            met=False,
            reason="No Python code block found",
            evidence=f"Checked {len(patterns)} patterns, none matched",
        )

    if check == "verdict_present":
        patterns = verdict_patterns or DEFAULT_VERDICT_PATTERNS
        for pat in patterns:
            if pat in content:
                return CompletionResult(
                    met=True,
                    reason="Verdict keyword found",
                    evidence=f"Matched keyword: {pat!r}",
                )
        return CompletionResult(
            met=False,
            reason="No verdict keyword found",
            evidence=f"Checked {len(patterns)} verdict patterns, none matched",
        )

    # Unknown check name — report.
    return CompletionResult(
        met=False,
        reason=f"Unknown check '{check}' for type 'output_contains'",
        evidence="",
    )


def _check_tool_attempted(
    expected_tool: str | None,
    tool_calls: list,
) -> CompletionResult:
    """Evaluate a tool_attempted check."""

    if not tool_calls:
        return CompletionResult(
            met=False,
            reason="No tool calls were made",
            evidence="Agent made 0 tool calls",
        )

    # Extract tool names from ToolCallRecord objects or dicts.
    called_names: list[str] = []
    for tc in tool_calls:
        if hasattr(tc, "tool_name"):
            called_names.append(tc.tool_name)
        elif isinstance(tc, dict) and "tool_name" in tc:
            called_names.append(tc["tool_name"])

    if expected_tool is None:
        # Any tool call counts.
        return CompletionResult(
            met=True,
            reason="Tool call was attempted",
            evidence=f"Tools called: {called_names}",
        )

    if expected_tool in called_names:
        return CompletionResult(
            met=True,
            reason=f"Tool '{expected_tool}' was attempted",
            evidence=f"Tools called: {called_names}",
        )

    return CompletionResult(
        met=False,
        reason=f"Expected tool '{expected_tool}' was not called",
        evidence=f"Tools called: {called_names}",
    )


# ---------------------------------------------------------------------------
# Loading from config dict
# ---------------------------------------------------------------------------


def load_completion_criteria(data: dict) -> CompletionCriteria:
    """Create a CompletionCriteria from a YAML-parsed dict.

    Expected keys: type, check, description (optional), tool_name (optional).
    """
    return CompletionCriteria(
        type=data.get("type", "any"),
        check=data.get("check", "always"),
        description=data.get("description", ""),
        tool_name=data.get("tool_name"),
    )
