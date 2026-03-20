"""Unit tests for stage definitions and pipeline loading."""

import pytest
import yaml

from src.coordination.completion_criteria import CompletionCriteria
from src.coordination.stage_definition import (
    VALID_CRITERIA_TYPES,
    PipelineDefinition,
    StageDefinition,
    load_pipeline,
    load_pipeline_from_yaml,
    load_stage,
    validate_pipeline,
    validate_pipeline_strict,
)

# ---- load_stage ------------------------------------------------------------


class TestLoadStage:
    def test_load_full_stage(self):
        data = {
            "name": "design_planning",
            "completion_criteria": {
                "type": "output_contains",
                "check": "non_empty_output",
                "description": "Agent produced a geometry plan",
            },
            "stage_prompt": "Analyze the task.",
        }
        s = load_stage(data)
        assert s.name == "design_planning"
        assert s.completion_criteria.type == "output_contains"
        assert s.completion_criteria.check == "non_empty_output"
        assert s.stage_prompt == "Analyze the task."

    def test_load_minimal_stage(self):
        s = load_stage({"name": "test"})
        assert s.name == "test"
        assert s.completion_criteria.type == "any"
        assert s.completion_criteria.check == "always"
        assert s.stage_prompt == ""

    def test_load_stage_with_tool_name(self):
        data = {
            "name": "execution",
            "completion_criteria": {
                "type": "tool_attempted",
                "check": "tool_called",
                "tool_name": "run_simulation",
            },
        }
        s = load_stage(data)
        assert s.completion_criteria.tool_name == "run_simulation"


# ---- load_pipeline ---------------------------------------------------------


class TestLoadPipeline:
    def test_load_pipeline_with_stages(self):
        data = {
            "stages": [
                {"name": "stage1", "completion_criteria": {"type": "any", "check": "always"}},
                {"name": "stage2", "completion_criteria": {"type": "any", "check": "always"}},
            ]
        }
        p = load_pipeline(data)
        assert len(p.stages) == 2
        assert p.stages[0].name == "stage1"
        assert p.stages[1].name == "stage2"

    def test_load_empty_pipeline(self):
        p = load_pipeline({"stages": []})
        assert len(p.stages) == 0

    def test_load_pipeline_no_stages_key(self):
        p = load_pipeline({})
        assert len(p.stages) == 0


# ---- load_pipeline_from_yaml -----------------------------------------------


class TestLoadPipelineFromYaml:
    def test_load_pipeline_style(self, tmp_path):
        data = {
            "sample_pipeline": {
                "stages": [
                    {
                        "name": "design_planning",
                        "completion_criteria": {
                            "type": "output_contains",
                            "check": "non_empty_output",
                        },
                    },
                    {
                        "name": "code_writing",
                        "completion_criteria": {
                            "type": "output_contains",
                            "check": "code_block",
                        },
                    },
                    {
                        "name": "code_execution",
                        "completion_criteria": {
                            "type": "tool_attempted",
                            "check": "tool_called",
                            "tool_name": "exec",
                        },
                    },
                    {
                        "name": "output_review",
                        "completion_criteria": {
                            "type": "output_contains",
                            "check": "verdict_present",
                        },
                    },
                ]
            }
        }
        path = tmp_path / "pipeline.yaml"
        path.write_text(yaml.dump(data))
        p = load_pipeline_from_yaml(path)
        assert len(p.stages) == 4
        assert p.stages[0].name == "design_planning"
        assert p.stages[3].name == "output_review"

    def test_load_top_level_stages(self, tmp_path):
        data = {
            "stages": [
                {"name": "s1"},
                {"name": "s2"},
            ]
        }
        path = tmp_path / "pipeline.yaml"
        path.write_text(yaml.dump(data))
        p = load_pipeline_from_yaml(path)
        assert len(p.stages) == 2

    def test_no_stages_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump({"config": {"value": 1}}))
        with pytest.raises(ValueError, match="No 'stages' key"):
            load_pipeline_from_yaml(path)

    def test_non_dict_yaml_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="Expected YAML dict"):
            load_pipeline_from_yaml(path)


# ---- validate_pipeline -----------------------------------------------------


class TestValidatePipeline:
    def test_valid_pipeline(self):
        p = PipelineDefinition(
            stages=[
                StageDefinition(name="s1", completion_criteria=CompletionCriteria(type="any", check="always")),
                StageDefinition(
                    name="s2",
                    completion_criteria=CompletionCriteria(type="output_contains", check="code_block"),
                ),
            ]
        )
        assert validate_pipeline(p) == []

    def test_empty_pipeline(self):
        p = PipelineDefinition(stages=[])
        errors = validate_pipeline(p)
        assert any("no stages" in e.lower() for e in errors)

    def test_duplicate_names(self):
        p = PipelineDefinition(
            stages=[
                StageDefinition(name="dup", completion_criteria=CompletionCriteria(type="any", check="always")),
                StageDefinition(name="dup", completion_criteria=CompletionCriteria(type="any", check="always")),
            ]
        )
        errors = validate_pipeline(p)
        assert any("Duplicate" in e for e in errors)

    def test_invalid_criteria_type(self):
        p = PipelineDefinition(
            stages=[
                StageDefinition(name="s1", completion_criteria=CompletionCriteria(type="magic", check="x")),
            ]
        )
        errors = validate_pipeline(p)
        assert any("invalid criteria type" in e.lower() for e in errors)

    def test_all_valid_criteria_types(self):
        stages = [
            StageDefinition(
                name=f"s_{t}",
                completion_criteria=CompletionCriteria(type=t, check="always"),
            )
            for t in VALID_CRITERIA_TYPES
        ]
        p = PipelineDefinition(stages=stages)
        assert validate_pipeline(p) == []

    def test_validate_strict_raises(self):
        p = PipelineDefinition(stages=[])
        with pytest.raises(ValueError, match="Pipeline validation failed"):
            validate_pipeline_strict(p)

    def test_validate_strict_passes(self):
        p = PipelineDefinition(
            stages=[
                StageDefinition(name="ok", completion_criteria=CompletionCriteria(type="any", check="always")),
            ]
        )
        validate_pipeline_strict(p)  # no exception


# ---- Stage ordering --------------------------------------------------------


class TestStageOrdering:
    def test_stages_maintain_order(self):
        data = {
            "stages": [
                {"name": "first"},
                {"name": "second"},
                {"name": "third"},
                {"name": "fourth"},
            ]
        }
        p = load_pipeline(data)
        assert [s.name for s in p.stages] == ["first", "second", "third", "fourth"]
