"""Unit tests for GraphRoutedHandler — the core state machine loop."""

import json

import pytest

from src.coordination.execution_handler import Assignment
from src.coordination.graph_routed_handler import (
    GraphRoutedHandler,
    _extract_complexity,
    _extract_error_type,
    _extract_execution_result,
    _extract_review_result,
)

# ---- Mock agents -----------------------------------------------------------

class _MockAgent:
    """Agent that returns a fixed response."""

    def __init__(self, response: str = "done"):
        self.tools = {"tool": True}
        self._response = response
        self.calls: list[str] = []

    def run(self, ctx: str) -> str:
        self.calls.append(ctx)
        return self._response


class _SequencedAgent:
    """Agent that returns a sequence of responses, cycling through them."""

    def __init__(self, responses: list[str]):
        self.tools = {"tool": True}
        self._responses = responses
        self._index = 0
        self.calls: list[str] = []

    def run(self, ctx: str) -> str:
        self.calls.append(ctx)
        resp = self._responses[self._index % len(self._responses)]
        self._index += 1
        return resp


class _FailOnceAgent:
    """Agent that raises on the first call, then returns normally."""

    def __init__(self, error: str = "SyntaxError: invalid syntax", success: str = "ok"):
        self.tools = {"tool": True}
        self._error = error
        self._success = success
        self._count = 0
        self.calls: list[str] = []

    def run(self, ctx: str) -> str:
        self.calls.append(ctx)
        self._count += 1
        if self._count == 1:
            raise RuntimeError(self._error)
        return self._success


class _AlwaysFailAgent:
    """Agent that always raises."""

    def __init__(self, error: str = "RuntimeError: boom"):
        self.tools = {"tool": True}
        self._error = error
        self.calls: list[str] = []

    def run(self, ctx: str) -> str:
        self.calls.append(ctx)
        raise RuntimeError(self._error)


# ---- Graph data helpers ----------------------------------------------------

def _simple_happy_path_graph() -> dict:
    """Two-state graph: START → DONE.  Simplest possible graph."""
    return {
        "initial_state": "START",
        "terminal_states": ["DONE"],
        "states": {
            "START": {
                "agent": "worker",
                "description": "Do the work",
                "transitions": [
                    {"condition": "always", "target": "DONE"},
                ],
            },
            "DONE": {
                "agent": None,
                "description": "Terminal",
                "transitions": [],
            },
        },
    }


def _classify_then_work_graph() -> dict:
    """Graph with classification → routing → work → done."""
    return {
        "initial_state": "CLASSIFY",
        "terminal_states": ["COMPLETE"],
        "states": {
            "CLASSIFY": {
                "agent": "classifier",
                "agent_prompt": "Classify as simple, moderate, or complex.",
                "description": "Classify complexity",
                "transitions": [
                    {"condition": "complexity == 'simple'", "target": "QUICK_WORK"},
                    {"condition": "complexity == 'moderate'", "target": "DETAILED_WORK"},
                    {"condition": "always", "target": "QUICK_WORK"},
                ],
            },
            "QUICK_WORK": {
                "agent": "worker",
                "description": "Quick path",
                "transitions": [
                    {"condition": "always", "target": "COMPLETE"},
                ],
            },
            "DETAILED_WORK": {
                "agent": "worker",
                "description": "Detailed path",
                "transitions": [
                    {"condition": "always", "target": "COMPLETE"},
                ],
            },
            "COMPLETE": {
                "agent": None,
                "description": "Terminal",
                "transitions": [],
            },
        },
        "resource_budgets": {
            "simple": {
                "max_passes": 6, "context_budget": 2000,
                "reasoning_enabled": False, "max_code_review_cycles": 1,
                "escalation_threshold": 2,
            },
            "moderate": {
                "max_passes": 12, "context_budget": 3000,
                "reasoning_enabled": True, "max_code_review_cycles": 2,
                "escalation_threshold": 3,
            },
        },
    }


def _error_routing_graph() -> dict:
    """Graph with execution → error classification → routing back."""
    return {
        "initial_state": "CODE",
        "terminal_states": ["COMPLETE"],
        "states": {
            "CODE": {
                "agent": "coder",
                "description": "Write code",
                "transitions": [
                    {"condition": "always", "target": "EXECUTE"},
                ],
            },
            "EXECUTE": {
                "agent": "executor",
                "description": "Execute code",
                "transitions": [
                    {"condition": "execution_success == true", "target": "COMPLETE"},
                    {"condition": "execution_success == false", "target": "ERROR_ROUTE"},
                ],
            },
            "ERROR_ROUTE": {
                "agent": None,
                "description": "Route on error type",
                "transitions": [
                    {"condition": "passes_remaining <= 0", "target": "COMPLETE"},
                    {"condition": "error_type in ['SyntaxError', 'NameError']", "target": "CODE"},
                    {"condition": "always", "target": "COMPLETE"},
                ],
            },
            "COMPLETE": {
                "agent": None,
                "description": "Terminal",
                "transitions": [],
            },
        },
        "resource_budgets": {
            "simple": {
                "max_passes": 6, "context_budget": 2000,
                "reasoning_enabled": False, "max_code_review_cycles": 1,
                "escalation_threshold": 2,
            },
        },
    }


def _review_graph() -> dict:
    """Graph with code → review → conditional routing."""
    return {
        "initial_state": "CODE",
        "terminal_states": ["COMPLETE"],
        "states": {
            "CODE": {
                "agent": "coder",
                "description": "Write code",
                "transitions": [
                    {"condition": "always", "target": "REVIEW"},
                ],
            },
            "REVIEW": {
                "agent": "reviewer",
                "description": "Review code",
                "transitions": [
                    {"condition": "review_passed == true", "target": "EXECUTE"},
                    {"condition": "review_passed == false", "target": "CODE"},
                ],
            },
            "EXECUTE": {
                "agent": "executor",
                "description": "Execute code",
                "transitions": [
                    {"condition": "execution_success == true", "target": "OUTPUT_REVIEW"},
                    {"condition": "always", "target": "CODE"},
                ],
            },
            "OUTPUT_REVIEW": {
                "agent": "output_reviewer",
                "description": "Review output",
                "transitions": [
                    {"condition": "review_verdict == 'passed'", "target": "COMPLETE"},
                    {"condition": "review_verdict == 'minor_issues'", "target": "CODE"},
                    {"condition": "review_verdict == 'major_issues'", "target": "CODE"},
                ],
            },
            "COMPLETE": {
                "agent": None,
                "description": "Terminal",
                "transitions": [],
            },
        },
    }


def _escalation_graph() -> dict:
    """Graph with a design cycle and escalation state."""
    return {
        "initial_state": "CLASSIFY",
        "terminal_states": ["COMPLETE"],
        "states": {
            "CLASSIFY": {
                "agent": "classifier",
                "agent_prompt": "Classify complexity.",
                "description": "Classify",
                "transitions": [
                    {"condition": "complexity == 'simple'", "target": "DESIGN"},
                    {"condition": "complexity == 'moderate'", "target": "DESIGN"},
                    {"condition": "always", "target": "DESIGN"},
                ],
            },
            "DESIGN": {
                "agent": "designer",
                "description": "Design",
                "transitions": [
                    {"condition": "always", "target": "EXECUTE"},
                ],
            },
            "EXECUTE": {
                "agent": "executor",
                "description": "Execute",
                "transitions": [
                    {"condition": "execution_success == true", "target": "COMPLETE"},
                    {"condition": "execution_success == false", "target": "ERROR_ROUTE"},
                ],
            },
            "ERROR_ROUTE": {
                "agent": None,
                "description": "Route errors",
                "transitions": [
                    {"condition": "passes_remaining <= 0", "target": "COMPLETE"},
                    {"condition": "cycle_count >= escalation_threshold", "target": "ESCALATE"},
                    {"condition": "always", "target": "DESIGN"},
                ],
            },
            "ESCALATE": {
                "agent": "classifier",
                "agent_prompt": "Re-classify. Previous: {complexity}.",
                "description": "Complexity escalation",
                "transitions": [
                    {"condition": "always", "target": "DESIGN"},
                ],
            },
            "COMPLETE": {
                "agent": None,
                "description": "Terminal",
                "transitions": [],
            },
        },
        "resource_budgets": {
            "simple": {
                "max_passes": 10, "context_budget": 2000,
                "reasoning_enabled": False, "max_code_review_cycles": 1,
                "escalation_threshold": 2,
            },
            "moderate": {
                "max_passes": 15, "context_budget": 3000,
                "reasoning_enabled": True, "max_code_review_cycles": 2,
                "escalation_threshold": 3,
            },
        },
    }


# ---- Helper to create handler from graph data -----------------------------

def _handler_with_graph(graph_data: dict, **extra) -> GraphRoutedHandler:
    cfg = {"_graph_data": graph_data, "max_transitions": 50}
    cfg.update(extra)
    return GraphRoutedHandler(cfg)


# ---- Tests: Happy path -----------------------------------------------------

class TestHappyPath:
    def test_simple_two_state_graph(self):
        handler = _handler_with_graph(_simple_happy_path_graph())
        agents = {"worker": _MockAgent("result")}
        msgs = handler.execute(
            [Assignment(agent_name="worker", task="Do X")],
            agents, logger=None,
        )
        assert len(msgs) == 1
        assert msgs[0].content == "result"
        assert msgs[0].agent_name == "worker"

    def test_classify_simple_route(self):
        handler = _handler_with_graph(_classify_then_work_graph())
        agents = {
            "classifier": _MockAgent("simple"),
            "worker": _MockAgent("quick result"),
        }
        msgs = handler.execute(
            [Assignment(agent_name="classifier", task="Build a box")],
            agents, logger=None,
        )
        # classifier → QUICK_WORK worker → COMPLETE
        assert len(msgs) == 2
        assert msgs[0].content == "simple"  # classifier output
        assert msgs[1].content == "quick result"  # worker output

    def test_classify_moderate_route(self):
        handler = _handler_with_graph(_classify_then_work_graph())
        agents = {
            "classifier": _MockAgent("moderate"),
            "worker": _MockAgent("detailed result"),
        }
        msgs = handler.execute(
            [Assignment(agent_name="classifier", task="Build complex thing")],
            agents, logger=None,
        )
        assert len(msgs) == 2
        assert msgs[1].content == "detailed result"


# ---- Tests: Error routing --------------------------------------------------

class TestErrorRouting:
    def test_execution_failure_routes_to_error_classification(self):
        handler = _handler_with_graph(_error_routing_graph())
        agents = {
            "coder": _MockAgent("code here"),
            "executor": _SequencedAgent([
                "execution failed with SyntaxError",
                "execution success, stl produced",
            ]),
        }
        msgs = handler.execute(
            [Assignment(agent_name="coder", task="Write code")],
            agents, logger=None,
        )
        # coder → executor (fail) → ERROR_ROUTE → CODE → executor (success) → COMPLETE
        assert len(msgs) >= 3  # at least coder + executor fail + more

    def test_agent_exception_sets_error_type(self):
        handler = _handler_with_graph(_error_routing_graph())
        agents = {
            "coder": _SequencedAgent(["code v1", "code v2"]),
            "executor": _FailOnceAgent(
                error="SyntaxError: invalid syntax",
                success="execution success stl produced",
            ),
        }
        msgs = handler.execute(
            [Assignment(agent_name="coder", task="Write code")],
            agents, logger=None,
        )
        # Should have error message with SyntaxError
        error_msgs = [m for m in msgs if m.error]
        assert len(error_msgs) >= 1


# ---- Tests: Review routing -------------------------------------------------

class TestReviewRouting:
    def test_review_passed_routes_to_execute(self):
        handler = _handler_with_graph(_review_graph())
        agents = {
            "coder": _MockAgent("code here"),
            "reviewer": _MockAgent("REVIEW_PASSED"),
            "executor": _MockAgent("execution success, stl produced"),
            "output_reviewer": _MockAgent("PASSED"),
        }
        msgs = handler.execute(
            [Assignment(agent_name="coder", task="Write code")],
            agents, logger=None,
        )
        # code → review (passed) → execute → output_review (passed) → COMPLETE
        assert len(msgs) == 4
        assert msgs[1].content == "REVIEW_PASSED"

    def test_review_failed_routes_back_to_code(self):
        handler = _handler_with_graph(_review_graph(), max_transitions=10)
        # reviewer fails first time, passes second
        agents = {
            "coder": _MockAgent("code here"),
            "reviewer": _SequencedAgent(["REVIEW_FAILED", "REVIEW_PASSED"]),
            "executor": _MockAgent("execution success, stl produced"),
            "output_reviewer": _MockAgent("PASSED"),
        }
        msgs = handler.execute(
            [Assignment(agent_name="coder", task="Write code")],
            agents, logger=None,
        )
        # code → review (fail) → code → review (pass) → execute → output → COMPLETE
        assert len(msgs) >= 5

    def test_output_review_minor_issues(self):
        handler = _handler_with_graph(_review_graph(), max_transitions=15)
        agents = {
            "coder": _MockAgent("code here"),
            "reviewer": _MockAgent("REVIEW_PASSED"),
            "executor": _MockAgent("execution success, stl produced"),
            "output_reviewer": _SequencedAgent(["minor_issues", "PASSED"]),
        }
        msgs = handler.execute(
            [Assignment(agent_name="coder", task="Write code")],
            agents, logger=None,
        )
        # At some point output_reviewer says minor_issues → back to CODE
        assert any(m.content == "minor_issues" for m in msgs)

    def test_output_review_major_issues(self):
        handler = _handler_with_graph(_review_graph(), max_transitions=15)
        agents = {
            "coder": _MockAgent("code here"),
            "reviewer": _MockAgent("REVIEW_PASSED"),
            "executor": _MockAgent("execution success, stl produced"),
            "output_reviewer": _SequencedAgent(["major_issues", "PASSED"]),
        }
        msgs = handler.execute(
            [Assignment(agent_name="coder", task="Write code")],
            agents, logger=None,
        )
        assert any(m.content == "major_issues" for m in msgs)


# ---- Tests: Escalation ----------------------------------------------------

class TestEscalation:
    def test_escalation_triggers_reclassification(self):
        handler = _handler_with_graph(_escalation_graph())
        # classifier says simple first, then moderate on escalation
        # executor always fails → forces design re-entry cycles
        agents = {
            "classifier": _SequencedAgent(["simple", "moderate"]),
            "designer": _MockAgent("design plan"),
            "executor": _SequencedAgent([
                "execution failed",  # fail 1
                "execution failed",  # fail 2
                "execution failed",  # fail 3 (escalation should trigger)
                "execution failed",  # after escalation
                "execution success stl produced",  # finally
            ]),
        }
        handler.execute(
            [Assignment(agent_name="classifier", task="Build thing")],
            agents, logger=None,
        )
        # classifier should have been called at least twice (initial + escalation)
        assert agents["classifier"]._index >= 2

    def test_resource_exhaustion_forces_complete(self):
        graph = _escalation_graph()
        graph["resource_budgets"]["simple"]["max_passes"] = 3
        handler = _handler_with_graph(graph)
        agents = {
            "classifier": _MockAgent("simple"),
            "designer": _MockAgent("design"),
            "executor": _MockAgent("execution failed"),
        }
        msgs = handler.execute(
            [Assignment(agent_name="classifier", task="Task")],
            agents, logger=None,
        )
        # Should terminate due to passes_remaining <= 0
        assert len(msgs) > 0


# ---- Tests: Agent mapping --------------------------------------------------

class TestAgentMapping:
    def test_missing_role_reports_error(self):
        handler = _handler_with_graph(_simple_happy_path_graph())
        agents = {"other_agent": _MockAgent("hi")}  # no "worker"
        msgs = handler.execute(
            [Assignment(agent_name="x", task="Task")],
            agents, logger=None,
        )
        assert len(msgs) == 1
        assert msgs[0].error is not None
        assert "No agent available for role" in msgs[0].error

    def test_agents_resolved_by_key(self):
        handler = _handler_with_graph(_simple_happy_path_graph())
        agents = {"worker": _MockAgent("resolved")}
        msgs = handler.execute(
            [Assignment(agent_name="worker", task="Task")],
            agents, logger=None,
        )
        assert msgs[0].content == "resolved"


# ---- Tests: State dict updates --------------------------------------------

class TestStateDictUpdates:
    def test_complexity_extracted_from_output(self):
        handler = _handler_with_graph(_classify_then_work_graph())
        agents = {
            "classifier": _MockAgent("moderate"),
            "worker": _MockAgent("done"),
        }
        handler.execute(
            [Assignment(agent_name="x", task="Task")],
            agents, logger=None,
        )
        assert handler._state_dict["complexity"] == "moderate"

    def test_review_passed_extracted(self):
        handler = _handler_with_graph(_review_graph(), max_transitions=10)
        agents = {
            "coder": _MockAgent("code"),
            "reviewer": _MockAgent("REVIEW_PASSED"),
            "executor": _MockAgent("execution success stl produced"),
            "output_reviewer": _MockAgent("PASSED"),
        }
        handler.execute(
            [Assignment(agent_name="x", task="Task")],
            agents, logger=None,
        )
        assert handler._state_dict.get("review_passed") is True


# ---- Tests: Internal representations --------------------------------------

class TestInternalRepresentations:
    def test_mental_model_off_by_default(self):
        handler = _handler_with_graph(_simple_happy_path_graph())
        agents = {"worker": _MockAgent("ok")}
        handler.execute(
            [Assignment(agent_name="x", task="Task")],
            agents, logger=None,
        )
        # Worker should only get task context, not workflow context
        ctx = agents["worker"].calls[0]
        assert "WORKFLOW CONTEXT" not in ctx

    def test_mental_model_on(self):
        handler = _handler_with_graph(
            _simple_happy_path_graph(),
            internal_representations={"enabled": True},
        )
        agents = {"worker": _MockAgent("ok")}
        handler.execute(
            [Assignment(agent_name="x", task="Task")],
            agents, logger=None,
        )
        ctx = agents["worker"].calls[0]
        assert "WORKFLOW CONTEXT" in ctx
        assert "passes remaining" in ctx


# ---- Tests: LLM graph mode ------------------------------------------------

class TestLLMGraphMode:
    def test_llm_generated_valid_graph(self):
        """Mock LLM generates a valid graph."""
        graph_json = json.dumps(_simple_happy_path_graph())
        handler = GraphRoutedHandler({
            "graph_mode": "llm_generated",
            "max_transitions": 10,
        })
        agents = {
            "graph_designer": _MockAgent(graph_json),
            "worker": _MockAgent("llm-graph result"),
        }
        msgs = handler.execute(
            [Assignment(agent_name="x", task="Task")],
            agents, logger=None,
        )
        assert len(msgs) >= 1
        # The messages include graph_designer output and worker output
        # But graph_designer isn't in the state machine — it's called during
        # graph generation. So we should just see the worker.
        assert any(m.content == "llm-graph result" for m in msgs)

    def test_llm_generated_invalid_graph_raises(self):
        """Mock LLM generates invalid JSON — should error."""
        handler = GraphRoutedHandler({
            "graph_mode": "llm_generated",
            "max_transitions": 10,
        })
        agents = {
            "graph_designer": _MockAgent("not json"),
        }
        with pytest.raises(ValueError, match="valid JSON"):
            handler.execute(
                [Assignment(agent_name="x", task="Task")],
                agents, logger=None,
            )

    def test_missing_graph_designer_raises(self):
        handler = GraphRoutedHandler({
            "graph_mode": "llm_generated",
            "max_transitions": 10,
        })
        agents = {"worker": _MockAgent("hi")}
        with pytest.raises(ValueError, match="graph_designer"):
            handler.execute(
                [Assignment(agent_name="x", task="Task")],
                agents, logger=None,
            )


# ---- Tests: Max transitions safety valve -----------------------------------

class TestMaxTransitions:
    def test_max_transitions_stops_execution(self):
        handler = _handler_with_graph(
            _simple_happy_path_graph(), max_transitions=0,
        )
        agents = {"worker": _MockAgent("hi")}
        msgs = handler.execute(
            [Assignment(agent_name="x", task="Task")],
            agents, logger=None,
        )
        # With max_transitions=0, the loop body doesn't execute transitions
        # but the agent at the initial state still runs.
        # The agent runs but no transition is followed, so we stop.
        assert len(msgs) <= 1


# ---- Tests: Transition history ---------------------------------------------

class TestTransitionHistory:
    def test_transitions_logged(self):
        handler = _handler_with_graph(_classify_then_work_graph())
        agents = {
            "classifier": _MockAgent("simple"),
            "worker": _MockAgent("done"),
        }
        handler.execute(
            [Assignment(agent_name="x", task="Task")],
            agents, logger=None,
        )
        assert len(handler.transition_history) >= 2
        # First transition: CLASSIFY → QUICK_WORK
        tr0 = handler.transition_history[0]
        assert tr0.from_state == "CLASSIFY"
        assert tr0.to_state == "QUICK_WORK"

    def test_transition_record_fields(self):
        handler = _handler_with_graph(_simple_happy_path_graph())
        agents = {"worker": _MockAgent("ok")}
        handler.execute(
            [Assignment(agent_name="x", task="Task")],
            agents, logger=None,
        )
        assert len(handler.transition_history) >= 1
        tr = handler.transition_history[0]
        assert tr.from_state == "START"
        assert tr.to_state == "DONE"
        assert tr.condition_matched == "always"
        assert tr.agent_invoked == "worker"


# ---- Tests: Extraction helpers ---------------------------------------------

class TestExtractionHelpers:
    def test_extract_complexity_simple(self):
        assert _extract_complexity("The answer is simple.") == "simple"

    def test_extract_complexity_moderate(self):
        assert _extract_complexity("moderate") == "moderate"

    def test_extract_complexity_complex(self):
        assert _extract_complexity("This is complex geometry.") == "complex"

    def test_extract_complexity_none(self):
        assert _extract_complexity("no classification here") is None

    def test_extract_error_type_syntax(self):
        assert _extract_error_type("SyntaxError: bad syntax") == "SyntaxError"

    def test_extract_error_type_attribute(self):
        assert _extract_error_type("AttributeError: no attr") == "AttributeError"

    def test_extract_error_type_type_error(self):
        assert _extract_error_type("TypeError: bad argument") == "TypeError"

    def test_extract_error_type_unknown(self):
        assert _extract_error_type("something weird") == "UnknownError"

    def test_extract_review_passed(self):
        assert _extract_review_result("REVIEW_PASSED")["review_passed"] is True

    def test_extract_review_failed(self):
        assert _extract_review_result("REVIEW_FAILED")["review_passed"] is False

    def test_extract_review_verdict_passed(self):
        assert _extract_review_result("PASSED")["review_verdict"] == "passed"

    def test_extract_review_verdict_minor(self):
        assert _extract_review_result("minor_issues found")["review_verdict"] == "minor_issues"

    def test_extract_review_verdict_major(self):
        assert _extract_review_result("major_issues detected")["review_verdict"] == "major_issues"

    def test_extract_execution_success(self):
        r = _extract_execution_result("execution success, results produced")
        assert r["execution_success"] is True

    def test_extract_execution_failure(self):
        r = _extract_execution_result("execution failed with error")
        assert r["execution_success"] is False


# ---- Tests: Empty assignments ----------------------------------------------

class TestEmptyAssignments:
    def test_empty_assignments(self):
        handler = _handler_with_graph(_simple_happy_path_graph())
        msgs = handler.execute([], {}, logger=None)
        assert msgs == []
