"""Reliability patterns ported from CMU design-research-agents.

Three patterns for making local models (Qwen3-8B etc.) reliably follow
multi-step tool-calling workflows in smolagents:

1. **Prompt+Validate Retry** — on parse failure inside ``generate()``,
   feed the error back to the model and re-generate.  Retries are
   transparent to smolagents and do NOT consume ``max_steps``.

2. **Strict Tool Schemas** — add ``additionalProperties: false`` to
   tool parameter schemas so the model is less likely to hallucinate
   extra fields.

3. **First-Step Guardrail** — reject ``final_answer`` if the agent
   has not yet produced any tool-call observations.  Uses smolagents'
   built-in ``final_answer_checks`` mechanism (zero monkey-patching).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smolagents.memory import ActionStep


@dataclass
class ReliabilityConfig:
    """Configuration for model reliability patterns."""

    max_retries: int = 3
    """Max retries inside generate() before propagating the error."""

    strict_tool_schemas: bool = True
    """Add ``additionalProperties: false`` to tool parameter schemas."""

    first_step_guardrail: bool = True
    """Reject ``final_answer`` if no tool observations exist yet."""


def _has_tool_observation(memory) -> bool:
    """Check whether the agent's memory contains at least one tool observation."""
    for step in memory.steps:
        if isinstance(step, ActionStep) and step.observations is not None:
            return True
    return False


def first_step_guardrail(answer: Any, memory: Any, agent: Any = None) -> bool:
    """Reject ``final_answer`` when the agent hasn't called any tools yet.

    Compatible with smolagents' ``final_answer_checks`` signature:
    ``(answer, memory, agent=...) -> bool``.  Returning ``False``
    causes smolagents to raise ``AgentError`` and continue the loop.
    """
    return _has_tool_observation(memory)


def make_first_step_guardrail() -> list:
    """Return a list containing the first-step guardrail check.

    Convenience factory for passing to ``ToolCallingAgent(final_answer_checks=...)``.
    """
    return [first_step_guardrail]


def add_strict_properties(tool_schemas: list[dict]) -> list[dict]:
    """Add ``additionalProperties: false`` to each tool's parameter schema.

    Operates in-place on the schema dicts and also returns them for convenience.
    """
    for schema in tool_schemas:
        func = schema.get("function", {})
        params = func.get("parameters", {})
        if params and isinstance(params, dict):
            params["additionalProperties"] = False
    return tool_schemas
