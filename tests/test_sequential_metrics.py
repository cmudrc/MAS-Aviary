"""Tests for sequential strategy metric computation.

No GPU needed. Tests per-prompt, cross-prompt, and template comparison
metrics using synthetic AgentMessage data.
"""

import time

import pytest

from src.coordination.history import AgentMessage, ToolCallRecord
from src.logging.sequential_metrics import (
    compute_cross_prompt_metrics,
    compute_sequential_metrics,
    compute_template_comparison,
)

# ---- Helpers ----------------------------------------------------------------


def _msg(agent, content, turn, duration=None, token_count=None, tool_calls=None, ts=None):
    return AgentMessage(
        agent_name=agent,
        content=content,
        turn_number=turn,
        timestamp=ts or time.time(),
        duration_seconds=duration,
        token_count=token_count,
        tool_calls=tool_calls or [],
    )


def _tc(tool_name, error=None):
    return ToolCallRecord(
        tool_name=tool_name,
        inputs={"input": "test"},
        output="ok",
        duration_seconds=0.01,
        error=error,
    )


# ---- Per-prompt metrics (empty) ---------------------------------------------


class TestEmptyMessages:
    def test_empty_returns_defaults(self):
        result = compute_sequential_metrics(
            messages=[],
            stage_order=["a", "b"],
            pipeline_template="linear",
        )
        assert result["stage_count"] == 2
        assert result["pipeline_template"] == "linear"
        assert result["total_turns"] == 0
        assert result["tool_restriction_violations"] == 0
        assert result["stage_independence_score"] == 1.0


# ---- Per-prompt metrics (basic) ---------------------------------------------


class TestPerPromptBasic:
    def _linear_msgs(self):
        t = time.time()
        return [
            _msg("planner", "plan output", 1, duration=1.0, token_count=100, ts=t),
            _msg(
                "executor",
                "exec output",
                2,
                duration=2.5,
                token_count=250,
                tool_calls=[_tc("calculator_tool")],
                ts=t + 1.0,
            ),
            _msg("reviewer", "review output", 3, duration=0.5, token_count=80, ts=t + 3.5),
        ]

    def test_stage_count(self):
        msgs = self._linear_msgs()
        result = compute_sequential_metrics(
            msgs,
            ["planner", "executor", "reviewer"],
            "linear",
        )
        assert result["stage_count"] == 3

    def test_pipeline_template(self):
        msgs = self._linear_msgs()
        result = compute_sequential_metrics(
            msgs,
            ["planner", "executor", "reviewer"],
            "linear",
        )
        assert result["pipeline_template"] == "linear"

    def test_per_stage_duration(self):
        msgs = self._linear_msgs()
        result = compute_sequential_metrics(
            msgs,
            ["planner", "executor", "reviewer"],
            "linear",
        )
        assert result["per_stage_duration"]["planner"] == 1.0
        assert result["per_stage_duration"]["executor"] == 2.5
        assert result["per_stage_duration"]["reviewer"] == 0.5

    def test_per_stage_tokens(self):
        msgs = self._linear_msgs()
        result = compute_sequential_metrics(
            msgs,
            ["planner", "executor", "reviewer"],
            "linear",
        )
        assert result["per_stage_tokens"]["planner"] == 100
        assert result["per_stage_tokens"]["executor"] == 250
        assert result["per_stage_tokens"]["reviewer"] == 80

    def test_total_turns(self):
        msgs = self._linear_msgs()
        result = compute_sequential_metrics(
            msgs,
            ["planner", "executor", "reviewer"],
            "linear",
        )
        assert result["total_turns"] == 3

    def test_propagation_time(self):
        msgs = self._linear_msgs()
        result = compute_sequential_metrics(
            msgs,
            ["planner", "executor", "reviewer"],
            "linear",
        )
        assert result["propagation_time"] == pytest.approx(3.5, abs=0.01)


# ---- Tool utilization --------------------------------------------------------


class TestToolUtilization:
    def test_wildcard_used(self):
        msgs = [
            _msg("executor", "out", 1, tool_calls=[_tc("calc")]),
        ]
        result = compute_sequential_metrics(
            msgs,
            ["executor"],
            "linear",
            stage_tool_sets={"executor": ["*"]},
        )
        assert result["per_stage_tool_utilization"]["executor"] == 1.0

    def test_wildcard_not_used(self):
        msgs = [_msg("executor", "out", 1)]
        result = compute_sequential_metrics(
            msgs,
            ["executor"],
            "linear",
            stage_tool_sets={"executor": ["*"]},
        )
        assert result["per_stage_tool_utilization"]["executor"] == 0.0

    def test_specific_tools_partial(self):
        msgs = [
            _msg("executor", "out", 1, tool_calls=[_tc("tool_a")]),
        ]
        result = compute_sequential_metrics(
            msgs,
            ["executor"],
            "linear",
            stage_tool_sets={"executor": ["tool_a", "tool_b"]},
        )
        assert result["per_stage_tool_utilization"]["executor"] == 0.5

    def test_no_tools_allowed(self):
        msgs = [_msg("planner", "out", 1)]
        result = compute_sequential_metrics(
            msgs,
            ["planner"],
            "linear",
            stage_tool_sets={"planner": []},
        )
        assert result["per_stage_tool_utilization"]["planner"] == 0.0


# ---- Cross-stage tool violations --------------------------------------------


class TestToolViolations:
    def test_no_violations(self):
        msgs = [
            _msg("planner", "plan", 1),
            _msg("executor", "exec", 2, tool_calls=[_tc("calculator_tool")]),
        ]
        result = compute_sequential_metrics(
            msgs,
            ["planner", "executor"],
            "linear",
            stage_tool_sets={
                "planner": [],
                "executor": ["calculator_tool"],
            },
        )
        assert result["tool_restriction_violations"] == 0
        assert result["stage_independence_score"] == 1.0

    def test_violation_detected(self):
        msgs = [
            _msg("planner", "plan", 1, tool_calls=[_tc("calculator_tool")]),
        ]
        result = compute_sequential_metrics(
            msgs,
            ["planner"],
            "linear",
            stage_tool_sets={"planner": []},
        )
        assert result["tool_restriction_violations"] == 1
        assert result["stage_independence_score"] == 0.0

    def test_mixed_violations(self):
        msgs = [
            _msg("planner", "plan", 1, tool_calls=[_tc("calculator_tool")]),
            _msg("executor", "exec", 2, tool_calls=[_tc("calculator_tool"), _tc("echo_tool")]),
        ]
        result = compute_sequential_metrics(
            msgs,
            ["planner", "executor"],
            "linear",
            stage_tool_sets={
                "planner": [],
                "executor": ["calculator_tool", "echo_tool"],
            },
        )
        # planner used calculator_tool (1 violation), executor used 2 tools correctly
        assert result["tool_restriction_violations"] == 1
        assert result["stage_independence_score"] == pytest.approx(1.0 - 1 / 3, abs=0.01)


# ---- Interface validation ----------------------------------------------------


class TestInterfaceMetrics:
    def test_all_pass(self):
        msgs = [_msg("a", "out", 1)]
        results = [
            {"stage": "a", "valid": True, "interface_output": "plan"},
            {"stage": "b", "valid": True, "interface_output": "exec"},
        ]
        m = compute_sequential_metrics(
            msgs,
            ["a", "b"],
            interface_results=results,
        )
        assert m["interface_pass_rate"] == 1.0

    def test_partial_pass(self):
        msgs = [_msg("a", "out", 1)]
        results = [
            {"stage": "a", "valid": True, "interface_output": "plan"},
            {"stage": "b", "valid": False, "interface_output": "exec"},
        ]
        m = compute_sequential_metrics(
            msgs,
            ["a", "b"],
            interface_results=results,
        )
        assert m["interface_pass_rate"] == 0.5

    def test_no_results(self):
        msgs = [_msg("a", "out", 1)]
        m = compute_sequential_metrics(
            msgs,
            ["a"],
            interface_results=[],
        )
        assert m["interface_pass_rate"] == 0.0


# ---- Stage independence ------------------------------------------------------


class TestStageIndependence:
    def test_perfect_independence(self):
        msgs = [
            _msg("executor", "out", 1, tool_calls=[_tc("tool_a")]),
        ]
        m = compute_sequential_metrics(
            msgs,
            ["executor"],
            stage_tool_sets={"executor": ["tool_a"]},
        )
        assert m["stage_independence_score"] == 1.0

    def test_no_tool_calls(self):
        msgs = [_msg("planner", "out", 1)]
        m = compute_sequential_metrics(
            msgs,
            ["planner"],
            stage_tool_sets={"planner": []},
        )
        assert m["stage_independence_score"] == 1.0


# ---- Cross-prompt metrics ----------------------------------------------------


class TestCrossPromptMetrics:
    def test_empty_runs(self):
        result = compute_cross_prompt_metrics([])
        assert result["total_prompts"] == 0
        assert result["success_rate"] == 0.0

    def test_all_success(self):
        runs = [
            {"success": True, "final_score": 0.9, "metrics": {}},
            {"success": True, "final_score": 0.8, "metrics": {}},
        ]
        result = compute_cross_prompt_metrics(runs)
        assert result["total_prompts"] == 2
        assert result["success_rate"] == 1.0
        assert result["omission_errors"] == 0

    def test_omission_errors(self):
        runs = [
            {"success": False, "final_score": 0.3, "metrics": {}},
            {"success": True, "final_score": 0.9, "metrics": {}},
        ]
        result = compute_cross_prompt_metrics(runs)
        assert result["omission_errors"] == 1

    def test_commission_errors(self):
        runs = [
            {"success": True, "final_score": 0.2, "metrics": {}},
        ]
        result = compute_cross_prompt_metrics(runs)
        assert result["commission_errors"] == 1

    def test_consecutive_failures(self):
        runs = [
            {"success": False, "metrics": {}},
            {"success": False, "metrics": {}},
            {"success": True, "metrics": {}},
            {"success": False, "metrics": {}},
        ]
        result = compute_cross_prompt_metrics(runs)
        assert result["max_consecutive_failures"] == 2

    def test_ambidexterity_nonzero(self):
        runs = [
            {"success": True, "metrics": {"per_stage_tool_utilization": {"a": 0.0, "b": 1.0}}},
            {"success": True, "metrics": {"per_stage_tool_utilization": {"a": 0.5, "b": 0.5}}},
        ]
        result = compute_cross_prompt_metrics(runs)
        assert result["ambidexterity_proxy"] > 0

    def test_ambidexterity_zero_uniform(self):
        runs = [
            {"success": True, "metrics": {"per_stage_tool_utilization": {"a": 0.5}}},
            {"success": True, "metrics": {"per_stage_tool_utilization": {"a": 0.5}}},
        ]
        result = compute_cross_prompt_metrics(runs)
        assert result["ambidexterity_proxy"] == 0.0


# ---- Template comparison ----------------------------------------------------


class TestTemplateComparison:
    def test_empty(self):
        result = compute_template_comparison({})
        assert result == {}

    def test_single_template(self):
        runs = [
            {
                "success": True,
                "metrics": {
                    "per_stage_tokens": {"a": 100, "b": 200},
                    "per_stage_duration": {"a": 1.0, "b": 2.0},
                    "propagation_time": 3.0,
                    "stage_count": 2,
                },
            },
        ]
        result = compute_template_comparison({"linear": runs})
        assert "linear" in result["per_template"]
        tmpl = result["per_template"]["linear"]
        assert tmpl["runs"] == 1
        assert tmpl["success_rate"] == 1.0
        assert tmpl["avg_tokens_per_run"] == 300.0
        assert tmpl["stage_bottleneck"] == "b"

    def test_two_templates(self):
        runs_linear = [
            {
                "success": True,
                "metrics": {
                    "per_stage_tokens": {"p": 100, "e": 200, "r": 50},
                    "per_stage_duration": {"p": 1.0, "e": 3.0, "r": 0.5},
                    "propagation_time": 4.5,
                    "stage_count": 3,
                },
            },
        ]
        runs_vmodel = [
            {
                "success": False,
                "metrics": {
                    "per_stage_tokens": {"r": 80, "s": 90, "d": 100, "i": 300, "v": 60},
                    "per_stage_duration": {"r": 0.5, "s": 0.8, "d": 1.0, "i": 4.0, "v": 0.3},
                    "propagation_time": 6.6,
                    "stage_count": 5,
                },
            },
        ]
        result = compute_template_comparison(
            {
                "linear": runs_linear,
                "v_model": runs_vmodel,
            }
        )
        assert len(result["per_template"]) == 2
        assert result["per_template"]["linear"]["success_rate"] == 1.0
        assert result["per_template"]["v_model"]["success_rate"] == 0.0
        assert result["per_template"]["v_model"]["stage_bottleneck"] == "i"

    def test_efficiency_zero_tokens(self):
        runs = [
            {
                "success": True,
                "metrics": {
                    "per_stage_tokens": {},
                    "per_stage_duration": {},
                    "propagation_time": 0.0,
                    "stage_count": 0,
                },
            },
        ]
        result = compute_template_comparison({"empty": runs})
        assert result["per_template"]["empty"]["efficiency"] == 0.0
