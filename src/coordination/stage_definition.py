"""Stage definitions for the Staged Pipeline handler.

Defines the StageDefinition dataclass and loading/validation functions for
pipeline stage configurations loaded from YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.coordination.completion_criteria import (
    CompletionCriteria,
    load_completion_criteria,
)

# ---------------------------------------------------------------------------
# Valid criteria types (for validation)
# ---------------------------------------------------------------------------

VALID_CRITERIA_TYPES = {"output_contains", "tool_attempted", "any"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StageDefinition:
    """A single stage in the pipeline."""

    name: str
    completion_criteria: CompletionCriteria
    stage_prompt: str = ""


@dataclass
class PipelineDefinition:
    """A complete pipeline — an ordered list of stages."""

    stages: list[StageDefinition]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_stage(data: dict) -> StageDefinition:
    """Create a StageDefinition from a YAML-parsed dict."""
    name = data.get("name", "")
    criteria_data = data.get("completion_criteria", {})
    criteria = load_completion_criteria(criteria_data)
    stage_prompt = data.get("stage_prompt", "")
    return StageDefinition(
        name=name,
        completion_criteria=criteria,
        stage_prompt=stage_prompt,
    )


def load_pipeline(data: dict) -> PipelineDefinition:
    """Create a PipelineDefinition from a YAML-parsed dict.

    Expects a dict with a ``stages`` key containing a list of stage dicts.
    """
    stages_data = data.get("stages", [])
    stages = [load_stage(s) for s in stages_data]
    return PipelineDefinition(stages=stages)


def load_pipeline_from_yaml(path: str | Path) -> PipelineDefinition:
    """Load a PipelineDefinition from a YAML file.

    The YAML file is expected to have a top-level key (e.g. ``aviary_pipeline``
    or ``pipeline``) containing a ``stages`` list, or a direct ``stages``
    list at the top level.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML dict, got {type(raw).__name__}")

    # Try top-level stages first.
    if "stages" in raw:
        return load_pipeline(raw)

    # Try first dict value that has a stages key.
    for key, value in raw.items():
        if isinstance(value, dict) and "stages" in value:
            return load_pipeline(value)

    raise ValueError(f"No 'stages' key found in YAML file: {path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_pipeline(pipeline: PipelineDefinition) -> list[str]:
    """Validate a pipeline definition.

    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []

    if not pipeline.stages:
        errors.append("Pipeline has no stages")
        return errors

    # Check for duplicate stage names.
    names = [s.name for s in pipeline.stages]
    seen: set[str] = set()
    for name in names:
        if name in seen:
            errors.append(f"Duplicate stage name: {name!r}")
        seen.add(name)

    # Check criteria types.
    for stage in pipeline.stages:
        ctype = stage.completion_criteria.type
        if ctype not in VALID_CRITERIA_TYPES:
            errors.append(
                f"Stage {stage.name!r} has invalid criteria type: {ctype!r}"
            )

    return errors


def validate_pipeline_strict(pipeline: PipelineDefinition) -> None:
    """Validate pipeline, raise ValueError if invalid."""
    errors = validate_pipeline(pipeline)
    if errors:
        raise ValueError(
            "Pipeline validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
