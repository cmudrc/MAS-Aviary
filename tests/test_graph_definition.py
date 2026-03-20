"""Unit tests for graph definition — loading, validation, role resolution."""

import pytest

from src.coordination.graph_definition import (
    GraphDefinition,
    GraphState,
    GraphTransition,
    GraphValidationError,
    load_graph,
    resolve_agent_for_role,
    validate_graph,
    validate_graph_strict,
)

# ---- Helpers ---------------------------------------------------------------


def _minimal_graph_data() -> dict:
    """Return a minimal valid graph dict."""
    return {
        "initial_state": "START",
        "terminal_states": ["DONE"],
        "states": {
            "START": {
                "agent": "worker",
                "description": "Start state",
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


def _sample_graph_data() -> dict:
    """Return a sample graph for testing."""
    return {
        "initial_state": "PROMPT_RECEIVED",
        "terminal_states": ["COMPLETE"],
        "states": {
            "PROMPT_RECEIVED": {
                "agent": "classifier",
                "agent_prompt": "Classify complexity.",
                "description": "Classify prompt",
                "transitions": [
                    {"condition": "complexity == 'simple'", "target": "QUICK_DESIGN"},
                    {"condition": "complexity == 'moderate'", "target": "DESIGN_PLANNED"},
                    {"condition": "complexity == 'complex'", "target": "DESIGN_DECOMPOSED"},
                ],
            },
            "QUICK_DESIGN": {
                "agent": "designer",
                "description": "Brief design",
                "transitions": [
                    {"condition": "always", "target": "CODE_WRITTEN"},
                ],
            },
            "DESIGN_PLANNED": {
                "agent": "designer",
                "description": "Detailed design",
                "transitions": [
                    {"condition": "always", "target": "CODE_WRITTEN"},
                ],
            },
            "DESIGN_DECOMPOSED": {
                "agent": "designer",
                "description": "Component decomposition",
                "transitions": [
                    {"condition": "always", "target": "CODE_WRITTEN"},
                ],
            },
            "CODE_WRITTEN": {
                "agent": "coder",
                "description": "Write code",
                "transitions": [
                    {"condition": "always", "target": "CODE_REVIEWED"},
                ],
            },
            "CODE_REVIEWED": {
                "agent": "code_reviewer",
                "description": "Review code",
                "transitions": [
                    {"condition": "review_passed == true", "target": "CODE_EXECUTED"},
                    {"condition": "review_passed == false", "target": "CODE_WRITTEN"},
                ],
            },
            "CODE_EXECUTED": {
                "agent": "executor",
                "description": "Execute code",
                "transitions": [
                    {"condition": "execution_success == true and results_produced == true", "target": "OUTPUT_REVIEW"},
                    {"condition": "execution_success == false", "target": "ERROR_CLASSIFICATION"},
                ],
            },
            "ERROR_CLASSIFICATION": {
                "agent": None,
                "description": "Route on error type",
                "transitions": [
                    {"condition": "passes_remaining <= 0", "target": "COMPLETE"},
                    {"condition": "cycle_count >= escalation_threshold", "target": "COMPLEXITY_ESCALATION"},
                    {"condition": "error_type in ['SyntaxError', 'NameError']", "target": "CODE_WRITTEN"},
                    {"condition": "always", "target": "DESIGN_PLANNED"},
                ],
            },
            "COMPLEXITY_ESCALATION": {
                "agent": "classifier",
                "description": "Re-classify complexity",
                "transitions": [
                    {"condition": "complexity == 'simple'", "target": "DESIGN_PLANNED"},
                    {"condition": "complexity == 'moderate'", "target": "DESIGN_DECOMPOSED"},
                    {"condition": "complexity == 'complex'", "target": "DESIGN_DECOMPOSED"},
                ],
            },
            "OUTPUT_REVIEW": {
                "agent": "output_reviewer",
                "description": "Review output",
                "transitions": [
                    {"condition": "review_verdict == 'passed'", "target": "COMPLETE"},
                    {"condition": "review_verdict == 'minor_issues'", "target": "CODE_WRITTEN"},
                    {"condition": "review_verdict == 'major_issues'", "target": "DESIGN_PLANNED"},
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
                "max_passes": 6,
                "context_budget": 2000,
                "reasoning_enabled": False,
                "max_code_review_cycles": 1,
                "escalation_threshold": 2,
            },
            "moderate": {
                "max_passes": 12,
                "context_budget": 3000,
                "reasoning_enabled": True,
                "max_code_review_cycles": 2,
                "escalation_threshold": 3,
            },
            "complex": {
                "max_passes": 20,
                "context_budget": 4000,
                "reasoning_enabled": True,
                "max_code_review_cycles": 3,
                "escalation_threshold": 4,
            },
        },
    }


# ---- Loading tests ---------------------------------------------------------


class TestLoadGraph:
    def test_load_minimal_graph(self):
        data = _minimal_graph_data()
        graph = load_graph(data)
        assert graph.initial_state == "START"
        assert graph.terminal_states == ["DONE"]
        assert len(graph.states) == 2
        assert "START" in graph.states
        assert "DONE" in graph.states

    def test_load_state_fields(self):
        data = _minimal_graph_data()
        graph = load_graph(data)
        start = graph.states["START"]
        assert start.name == "START"
        assert start.agent == "worker"
        assert start.description == "Start state"
        assert len(start.transitions) == 1
        assert start.transitions[0].condition == "always"
        assert start.transitions[0].target == "DONE"

    def test_load_null_agent(self):
        data = _minimal_graph_data()
        graph = load_graph(data)
        done = graph.states["DONE"]
        assert done.agent is None

    def test_load_agent_prompt(self):
        data = _minimal_graph_data()
        data["states"]["START"]["agent_prompt"] = "Do something {var}."
        graph = load_graph(data)
        assert graph.states["START"].agent_prompt == "Do something {var}."

    def test_load_agent_prompt_default_none(self):
        data = _minimal_graph_data()
        graph = load_graph(data)
        assert graph.states["START"].agent_prompt is None

    def test_load_sample_graph(self):
        data = _sample_graph_data()
        graph = load_graph(data)
        assert graph.initial_state == "PROMPT_RECEIVED"
        assert "COMPLETE" in graph.terminal_states
        assert len(graph.states) == 11

    def test_load_resource_budgets(self):
        data = _sample_graph_data()
        graph = load_graph(data)
        assert "simple" in graph.resource_budgets
        assert "moderate" in graph.resource_budgets
        assert "complex" in graph.resource_budgets
        simple = graph.resource_budgets["simple"]
        assert simple.max_passes == 6
        assert simple.context_budget == 2000
        assert simple.reasoning_enabled is False
        assert simple.max_code_review_cycles == 1
        assert simple.escalation_threshold == 2

    def test_load_no_resource_budgets(self):
        data = _minimal_graph_data()
        graph = load_graph(data)
        assert graph.resource_budgets == {}

    def test_load_multiple_transitions(self):
        data = _sample_graph_data()
        graph = load_graph(data)
        pr = graph.states["PROMPT_RECEIVED"]
        assert len(pr.transitions) == 3
        assert pr.transitions[0].condition == "complexity == 'simple'"
        assert pr.transitions[0].target == "QUICK_DESIGN"

    def test_load_empty_transitions_for_terminal(self):
        data = _sample_graph_data()
        graph = load_graph(data)
        complete = graph.states["COMPLETE"]
        assert complete.transitions == []


# ---- Validation tests ------------------------------------------------------


class TestValidateGraph:
    def test_valid_minimal_graph(self):
        data = _minimal_graph_data()
        graph = load_graph(data)
        errors = validate_graph(graph)
        assert errors == []

    def test_valid_sample_graph(self):
        data = _sample_graph_data()
        graph = load_graph(data)
        errors = validate_graph(graph)
        assert errors == []

    def test_missing_initial_state(self):
        data = _minimal_graph_data()
        data["initial_state"] = "NONEXISTENT"
        graph = load_graph(data)
        errors = validate_graph(graph)
        assert any("Initial state" in e for e in errors)

    def test_missing_terminal_state(self):
        data = _minimal_graph_data()
        data["terminal_states"] = ["NONEXISTENT"]
        graph = load_graph(data)
        errors = validate_graph(graph)
        assert any("Terminal state" in e for e in errors)

    def test_transition_to_nonexistent_state(self):
        data = _minimal_graph_data()
        data["states"]["START"]["transitions"] = [
            {"condition": "always", "target": "NOWHERE"},
        ]
        graph = load_graph(data)
        errors = validate_graph(graph)
        assert any("unknown state" in e for e in errors)

    def test_orphan_state_detected(self):
        data = _minimal_graph_data()
        data["states"]["ORPHAN"] = {
            "agent": None,
            "description": "Unreachable",
            "transitions": [{"condition": "always", "target": "DONE"}],
        }
        graph = load_graph(data)
        errors = validate_graph(graph)
        assert any("unreachable" in e.lower() for e in errors)

    def test_terminal_unreachable(self):
        """Terminal state exists but isn't reachable from initial."""
        graph = GraphDefinition(
            initial_state="A",
            terminal_states=["C"],
            states={
                "A": GraphState(
                    name="A",
                    agent=None,
                    description="start",
                    transitions=[GraphTransition("always", "B")],
                ),
                "B": GraphState(
                    name="B",
                    agent=None,
                    description="dead end",
                    transitions=[GraphTransition("always", "A")],
                ),
                "C": GraphState(
                    name="C",
                    agent=None,
                    description="terminal",
                    transitions=[],
                ),
            },
        )
        errors = validate_graph(graph)
        assert any("terminal" in e.lower() and "reachable" in e.lower() for e in errors)

    def test_validate_strict_raises(self):
        data = _minimal_graph_data()
        data["initial_state"] = "NONEXISTENT"
        graph = load_graph(data)
        with pytest.raises(GraphValidationError):
            validate_graph_strict(graph)

    def test_validate_strict_ok(self):
        data = _minimal_graph_data()
        graph = load_graph(data)
        validate_graph_strict(graph)  # should not raise


# ---- Role resolution tests -------------------------------------------------


class _MockAgent:
    """Mock agent with name and optional description."""

    def __init__(self, name, description=""):
        self.name = name
        self.description = description


# Backward compat alias used by existing exact/attribute tests.
_MockAgentWithName = _MockAgent


class TestResolveAgentForRole:
    def test_exact_key_match(self):
        agents = {"coder": "agent_object"}
        result = resolve_agent_for_role("coder", agents, "CODE_WRITTEN")
        assert result == "agent_object"

    def test_attribute_name_match(self):
        agent = _MockAgentWithName("coder")
        agents = {"worker_1": agent}
        result = resolve_agent_for_role("coder", agents, "CODE_WRITTEN")
        assert result is agent

    def test_missing_role_raises(self):
        agents = {"planner": "obj"}
        with pytest.raises(ValueError, match="No agent available for role"):
            resolve_agent_for_role("coder", agents, "CODE_WRITTEN")

    def test_missing_role_error_message_includes_state(self):
        agents = {"planner": "obj"}
        with pytest.raises(ValueError, match="CODE_WRITTEN"):
            resolve_agent_for_role("coder", agents, "CODE_WRITTEN")

    def test_exact_key_preferred_over_attribute(self):
        """If both exact key and attribute match, exact key wins."""
        attr_agent = _MockAgentWithName("role_x")
        agents = {"role_x": "exact_match", "other": attr_agent}
        result = resolve_agent_for_role("role_x", agents, "S")
        assert result == "exact_match"

    # -- Fuzzy matching (substring + token overlap) --------------------------

    def test_substring_role_in_agent_name(self):
        """Role 'coder' is a substring of agent name 'simulation_coder'."""
        agent = _MockAgent("simulation_coder", "Writes simulation code")
        agents = {"orchestrator": "orch", "w1": agent}
        result = resolve_agent_for_role("coder", agents, "CODE_WRITTEN")
        assert result is agent

    def test_substring_role_in_description(self):
        """Role 'classifier' appears in agent description."""
        agent = _MockAgent("prompt_analyzer", "A classifier for prompt complexity")
        agents = {"orchestrator": "orch", "w1": agent}
        result = resolve_agent_for_role("classifier", agents, "PROMPT_RECEIVED")
        assert result is agent

    def test_prefix_stem_matching(self):
        """Role 'coder' matches agent 'code_generator' via shared 'code' stem."""
        agent = _MockAgent("code_generator", "Generates simulation Python code")
        agents = {"orchestrator": "orch", "w1": agent}
        result = resolve_agent_for_role("coder", agents, "CODE_WRITTEN")
        assert result is agent

    def test_prefix_stem_executor(self):
        """Role 'executor' matches agent 'code_executor' via 'execut' prefix."""
        agent = _MockAgent("code_executor", "Executes simulation scripts")
        agents = {"orchestrator": "orch", "w1": agent}
        result = resolve_agent_for_role("executor", agents, "CODE_EXECUTED")
        assert result is agent

    def test_prefix_stem_reviewer(self):
        """Role 'code_reviewer' matches agent with 'review' in description."""
        agent = _MockAgent("quality_checker", "Reviews code for correctness")
        agents = {"orchestrator": "orch", "w1": agent}
        result = resolve_agent_for_role("code_reviewer", agents, "CODE_REVIEWED")
        assert result is agent

    def test_orchestrator_excluded_from_fuzzy(self):
        """Orchestrator agent is skipped during fuzzy matching."""
        orch = _MockAgent("orchestrator", "Orchestrates code execution tasks")
        worker = _MockAgent("code_executor", "Executes simulation scripts")
        agents = {"orchestrator": orch, "w1": worker}
        # Both could match "executor" via description, but orchestrator
        # must be skipped — worker should win.
        result = resolve_agent_for_role("executor", agents, "CODE_EXECUTED")
        assert result is worker

    def test_orchestrator_excluded_falls_through(self):
        """When only orchestrator matches, ValueError is raised (not orchestrator)."""
        orch = _MockAgent("orchestrator", "Classifies and executes everything")
        agents = {"orchestrator": orch}
        with pytest.raises(ValueError, match="No agent available"):
            resolve_agent_for_role("classifier", agents, "PROMPT_RECEIVED")

    def test_multi_role_same_agent(self):
        """One agent can serve multiple graph roles via different fuzzy matches."""
        agent = _MockAgent(
            "simulation_worker",
            "Writes simulation code, executes it, and reviews the output",
        )
        agents = {"orchestrator": "orch", "w1": agent}
        # All three roles should resolve to the same agent.
        assert resolve_agent_for_role("coder", agents, "CODE_WRITTEN") is agent
        assert resolve_agent_for_role("executor", agents, "CODE_EXECUTED") is agent
        assert resolve_agent_for_role("output_reviewer", agents, "OUTPUT_REVIEW") is agent

    def test_best_score_wins(self):
        """When multiple agents match, the highest-scoring one wins."""
        generic = _MockAgent("helper", "A general assistant")
        coder = _MockAgent("python_coder", "Expert Python code writer")
        agents = {"orchestrator": "orch", "w1": generic, "w2": coder}
        result = resolve_agent_for_role("coder", agents, "CODE_WRITTEN")
        assert result is coder

    def test_no_fuzzy_match_raises(self):
        """ValueError raised when no fuzzy match is possible."""
        agent = _MockAgent("data_analyst", "Analyzes CSV datasets")
        agents = {"orchestrator": "orch", "w1": agent}
        with pytest.raises(ValueError, match="No agent available"):
            resolve_agent_for_role("classifier", agents, "PROMPT_RECEIVED")
