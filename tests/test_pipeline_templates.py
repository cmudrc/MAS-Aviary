"""Tests for pipeline template loading and validation.

No GPU needed. Tests all built-in templates, custom templates,
tool resolution, and validation errors.
"""

import pytest

from src.coordination.pipeline_templates import (
    TEMPLATE_NAMES,
    load_template,
    resolve_tools,
)
from src.tools.mock_tools import CalculatorTool, EchoTool, StateTool

# ---- Fixtures ----------------------------------------------------------------


@pytest.fixture
def domain_tools():
    echo = EchoTool()
    calc = CalculatorTool()
    state = StateTool()
    return {echo.name: echo, calc.name: calc, state.name: state}


# ---- Load built-in templates -------------------------------------------------


class TestLoadBuiltinTemplates:
    def test_load_linear(self):
        t = load_template("linear")
        assert t.name == "linear"
        assert len(t.stages) == 3
        names = [s.name for s in t.stages]
        assert names == ["planner", "executor", "reviewer"]

    def test_linear_roles(self):
        t = load_template("linear")
        assert "plan" in t.stages[0].role.lower()
        assert "execute" in t.stages[1].role.lower()
        assert "review" in t.stages[2].role.lower()

    def test_linear_tool_restrictions(self):
        t = load_template("linear")
        assert t.stages[0].allowed_tools == []  # planner: no tools
        assert t.stages[1].allowed_tools == ["*"]  # executor: all
        assert t.stages[2].allowed_tools == []  # reviewer: no tools

    def test_load_v_model(self):
        t = load_template("v_model")
        assert t.name == "v_model"
        assert len(t.stages) == 5
        names = [s.name for s in t.stages]
        assert names == [
            "requirements_analyst",
            "system_designer",
            "detailed_designer",
            "implementer",
            "integration_verifier",
        ]

    def test_v_model_only_implementer_has_tools(self):
        t = load_template("v_model")
        for stage in t.stages:
            if stage.name == "implementer":
                assert stage.allowed_tools == ["*"]
            else:
                assert stage.allowed_tools == []

    def test_load_mbse(self):
        t = load_template("mbse")
        assert t.name == "mbse"
        assert len(t.stages) == 5
        names = [s.name for s in t.stages]
        assert names == [
            "stakeholder_analyst",
            "system_architect",
            "subsystem_designer",
            "implementer",
            "validator",
        ]

    def test_mbse_only_implementer_has_tools(self):
        t = load_template("mbse")
        for stage in t.stages:
            if stage.name == "implementer":
                assert stage.allowed_tools == ["*"]
            else:
                assert stage.allowed_tools == []

    def test_all_templates_have_interface_output(self):
        for name in TEMPLATE_NAMES:
            t = load_template(name)
            for stage in t.stages:
                assert stage.interface_output, f"Stage '{stage.name}' in '{name}' missing interface_output"


# ---- Load custom template ----------------------------------------------------


class TestLoadCustomTemplate:
    def test_custom_from_stages(self):
        stages = [
            {"name": "stage_a", "role": "Do A", "allowed_tools": [], "interface_output": "A output"},
            {"name": "stage_b", "role": "Do B", "allowed_tools": ["*"], "interface_output": "B output"},
        ]
        t = load_template("custom", custom_stages=stages)
        assert t.name == "custom"
        assert len(t.stages) == 2
        assert t.stages[0].name == "stage_a"
        assert t.stages[1].name == "stage_b"

    def test_custom_no_stages_raises(self):
        with pytest.raises(ValueError, match="no custom_stages"):
            load_template("custom")

    def test_custom_empty_stages_raises(self):
        with pytest.raises(ValueError, match="no custom_stages"):
            load_template("custom", custom_stages=[])


# ---- Load from config templates ----------------------------------------------


class TestLoadFromConfig:
    def test_template_from_config(self):
        templates_config = {
            "my_template": {
                "stages": [
                    {"name": "s1", "role": "Role 1", "allowed_tools": [], "interface_output": "Out 1"},
                ]
            }
        }
        t = load_template("my_template", templates_config=templates_config)
        assert t.name == "my_template"
        assert len(t.stages) == 1

    def test_config_template_empty_stages_raises(self):
        templates_config = {"empty": {"stages": []}}
        with pytest.raises(ValueError, match="empty stages"):
            load_template("empty", templates_config=templates_config)


# ---- Validation errors -------------------------------------------------------


class TestValidation:
    def test_unknown_template_raises(self):
        with pytest.raises(ValueError, match="Unknown pipeline template"):
            load_template("nonexistent")

    def test_duplicate_stage_names_raises(self):
        stages = [
            {"name": "stage_a", "role": "R1", "allowed_tools": [], "interface_output": "O1"},
            {"name": "stage_a", "role": "R2", "allowed_tools": [], "interface_output": "O2"},
        ]
        with pytest.raises(ValueError, match="duplicate stage names"):
            load_template("custom", custom_stages=stages)

    def test_empty_stage_name_raises(self):
        stages = [
            {"name": "", "role": "R1", "allowed_tools": [], "interface_output": "O1"},
        ]
        with pytest.raises(ValueError, match="empty name"):
            load_template("custom", custom_stages=stages)


# ---- Tool resolution ---------------------------------------------------------


class TestResolveTools:
    def test_wildcard_resolves_all(self, domain_tools):
        result = resolve_tools(["*"], domain_tools)
        assert len(result) == 3
        names = {t.name for t in result}
        assert names == {"echo_tool", "calculator_tool", "state_tool"}

    def test_specific_tool(self, domain_tools):
        result = resolve_tools(["calculator_tool"], domain_tools)
        assert len(result) == 1
        assert result[0].name == "calculator_tool"

    def test_multiple_tools(self, domain_tools):
        result = resolve_tools(["echo_tool", "state_tool"], domain_tools)
        assert len(result) == 2
        names = {t.name for t in result}
        assert names == {"echo_tool", "state_tool"}

    def test_empty_list_returns_empty(self, domain_tools):
        result = resolve_tools([], domain_tools)
        assert result == []

    def test_unknown_tool_raises(self, domain_tools):
        with pytest.raises(ValueError, match="not found"):
            resolve_tools(["nonexistent_tool"], domain_tools)


# ---- TEMPLATE_NAMES ---------------------------------------------------------


class TestTemplateNames:
    def test_contains_expected(self):
        assert TEMPLATE_NAMES == {"linear", "v_model", "mbse"}
