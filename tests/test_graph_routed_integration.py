"""Integration tests for the Graph-Routed operational methodology.

Fast tests use mock agents (no GPU).
Slow test (@pytest.mark.slow) uses real LLM + mock tools.
"""

import json

import pytest
from smolagents import Model
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
)

from src.coordination.coordinator import Coordinator
from src.coordination.execution_handler import Assignment
from src.coordination.graph_definition import (
    load_graph_from_yaml,
    validate_graph,
)
from src.coordination.graph_routed_handler import GraphRoutedHandler
from src.coordination.strategy import CoordinationAction, CoordinationStrategy
from src.logging.graph_routed_metrics import (
    compute_cross_prompt_metrics,
    compute_per_prompt_metrics,
    compute_routing_quality,
)
from src.logging.logger import InstrumentationLogger

# ---- Model stub (no GPU) ---------------------------------------------------

class _FinalAnswerModel(Model):
    """Returns a final_answer tool call so ToolCallingAgents complete in 1 step."""

    def __init__(self, answer: str = "done"):
        super().__init__(model_id="final-answer-stub")
        self._answer = answer
        self._call_count = 0

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs):
        self._call_count += 1
        tc = ChatMessageToolCall(
            id=f"call_{self._call_count}",
            type="function",
            function=ChatMessageToolCallFunction(
                name="final_answer",
                arguments=json.dumps({"answer": self._answer}),
            ),
        )
        return ChatMessage(role="assistant", content="", tool_calls=[tc])


# ---- Mock agents -----------------------------------------------------------

class _MockAgent:
    """Agent stub returning fixed output."""

    def __init__(self, response: str = "done", tools: dict | None = None):
        self.tools = tools if tools is not None else {"tool": True}
        self._response = response
        self.calls: list[str] = []

    def run(self, ctx: str):
        self.calls.append(ctx)
        return self._response


class _SequencedAgent:
    """Agent returning responses in sequence."""

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
    """Agent that raises on first call, then succeeds."""

    def __init__(self, error: str = "SyntaxError: invalid", success: str = "ok"):
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


# ---- Minimal strategy for coordinator tests --------------------------------

class _SingleAgentStrategy(CoordinationStrategy):
    """Invokes agents once each in order, then terminates."""

    def __init__(self, agent_names: list[str]):
        self._names = agent_names
        self._index = 0
        self._task = ""

    def initialize(self, agents, config):
        self._index = 0

    def next_step(self, history, state):
        if "task" in state:
            self._task = state["task"]
        if self._index >= len(self._names):
            return CoordinationAction(
                action_type="terminate", agent_name=None, input_context="",
            )
        name = self._names[self._index]
        self._index += 1
        ctx = self._task if not history else history[-1].content
        return CoordinationAction(
            action_type="invoke_agent", agent_name=name, input_context=ctx,
        )

    def is_complete(self, history, state):
        return self._index >= len(self._names)


# ---- Handler-level integration tests (fast) --------------------------------

class TestHandlerIntegration:
    """Test the handler with mock agents — no coordinator, no GPU."""

    def test_aviary_graph_loads_and_validates(self):
        """Aviary graph YAML loads and validates without errors."""
        graph = load_graph_from_yaml("config/aviary_graph.yaml")
        errors = validate_graph(graph)
        assert errors == []
        assert len(graph.states) == 11
        assert "TASK_CLASSIFIED" in graph.states
        assert "COMPLETE" in graph.states

    def test_simple_happy_path(self):
        """Simple 2-state graph: classify → work → done."""
        graph_data = {
            "initial_state": "WORK",
            "terminal_states": ["DONE"],
            "states": {
                "WORK": {
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
        handler = GraphRoutedHandler({"_graph_data": graph_data})
        agents = {"worker": _MockAgent("result")}
        msgs = handler.execute(
            [Assignment(agent_name="worker", task="Do something")],
            agents, logger=None,
        )
        assert len(msgs) == 1
        assert msgs[0].content == "result"

    def test_classification_routing(self):
        """Classifier output routes to different design states."""
        graph_data = {
            "initial_state": "CLASSIFY",
            "terminal_states": ["COMPLETE"],
            "states": {
                "CLASSIFY": {
                    "agent": "classifier",
                    "description": "Classify",
                    "transitions": [
                        {"condition": "complexity == 'simple'", "target": "QUICK"},
                        {"condition": "complexity == 'moderate'", "target": "DETAILED"},
                        {"condition": "always", "target": "QUICK"},
                    ],
                },
                "QUICK": {
                    "agent": "worker",
                    "description": "Quick work",
                    "transitions": [{"condition": "always", "target": "COMPLETE"}],
                },
                "DETAILED": {
                    "agent": "worker",
                    "description": "Detailed work",
                    "transitions": [{"condition": "always", "target": "COMPLETE"}],
                },
                "COMPLETE": {
                    "agent": None, "description": "Done", "transitions": [],
                },
            },
        }
        handler = GraphRoutedHandler({"_graph_data": graph_data})
        agents = {
            "classifier": _MockAgent("simple"),
            "worker": _MockAgent("quick result"),
        }
        msgs = handler.execute(
            [Assignment(agent_name="x", task="Build a box")],
            agents, logger=None,
        )
        assert len(msgs) == 2  # classifier + worker
        assert msgs[1].content == "quick result"

    def test_error_routing_with_reroute(self):
        """Execution failure routes through error classification back to code."""
        graph_data = {
            "initial_state": "CODE",
            "terminal_states": ["COMPLETE"],
            "states": {
                "CODE": {
                    "agent": "coder",
                    "description": "Write code",
                    "transitions": [{"condition": "always", "target": "EXECUTE"}],
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
                    "description": "Route on error",
                    "transitions": [
                        {"condition": "passes_remaining <= 0", "target": "COMPLETE"},
                        {"condition": "error_type in ['SyntaxError']", "target": "CODE"},
                        {"condition": "always", "target": "COMPLETE"},
                    ],
                },
                "COMPLETE": {"agent": None, "description": "Done", "transitions": []},
            },
        }
        handler = GraphRoutedHandler({"_graph_data": graph_data, "max_transitions": 20})
        agents = {
            "coder": _SequencedAgent(["code v1", "code v2"]),
            "executor": _SequencedAgent([
                "execution failed SyntaxError",
                "execution success stl produced",
            ]),
        }
        msgs = handler.execute(
            [Assignment(agent_name="x", task="Write code")],
            agents, logger=None,
        )
        # coder → executor (fail) → ERROR_ROUTE → coder → executor (success)
        assert len(msgs) >= 4
        # Last executor should succeed.
        assert any("success" in m.content for m in msgs)

    def test_missing_agent_role_reports_error(self):
        """Handler stops with error when required role is missing."""
        graph_data = {
            "initial_state": "WORK",
            "terminal_states": ["DONE"],
            "states": {
                "WORK": {
                    "agent": "nonexistent_role",
                    "description": "Work",
                    "transitions": [{"condition": "always", "target": "DONE"}],
                },
                "DONE": {"agent": None, "description": "Done", "transitions": []},
            },
        }
        handler = GraphRoutedHandler({"_graph_data": graph_data})
        agents = {"other": _MockAgent("hi")}
        msgs = handler.execute(
            [Assignment(agent_name="x", task="Do it")],
            agents, logger=None,
        )
        assert len(msgs) == 1
        assert msgs[0].error is not None
        assert "No agent available for role" in msgs[0].error

    def test_metrics_from_handler_execution(self):
        """Handler transition history produces valid metrics."""
        graph_data = {
            "initial_state": "CLASSIFY",
            "terminal_states": ["COMPLETE"],
            "states": {
                "CLASSIFY": {
                    "agent": "classifier",
                    "description": "Classify",
                    "transitions": [
                        {"condition": "complexity == 'simple'", "target": "WORK"},
                        {"condition": "always", "target": "WORK"},
                    ],
                },
                "WORK": {
                    "agent": "worker",
                    "description": "Work",
                    "transitions": [{"condition": "always", "target": "COMPLETE"}],
                },
                "COMPLETE": {"agent": None, "description": "Done", "transitions": []},
            },
            "resource_budgets": {
                "simple": {
                    "max_passes": 6, "context_budget": 2000,
                    "reasoning_enabled": False, "max_code_review_cycles": 1,
                    "escalation_threshold": 2,
                },
            },
        }
        handler = GraphRoutedHandler({"_graph_data": graph_data})
        agents = {
            "classifier": _MockAgent("simple"),
            "worker": _MockAgent("done"),
        }
        handler.execute(
            [Assignment(agent_name="x", task="Task")],
            agents, logger=None,
        )
        pm = compute_per_prompt_metrics(handler.transition_history)
        assert pm["total_transitions"] == 2
        assert pm["unique_states_visited"] == 3
        assert pm["cycle_count"] == 0

        rq = compute_routing_quality(
            handler.transition_history, terminal_states=["COMPLETE"],
        )
        assert rq["routing_accuracy"] == 1.0
        assert rq["misroute_rate"] == 0.0

    def test_config_file_loads(self):
        """graph_routed.yaml loads without error."""
        from src.config.loader import load_yaml
        cfg = load_yaml("config/graph_routed.yaml")
        assert "graph_routed" in cfg
        assert cfg["graph_routed"]["graph_mode"] == "predefined"
        assert cfg["graph_routed"]["max_transitions"] == 50

    def test_internal_representations_toggle(self):
        """Mental model context appears when toggle is on."""
        graph_data = {
            "initial_state": "WORK",
            "terminal_states": ["DONE"],
            "states": {
                "WORK": {
                    "agent": "worker",
                    "description": "Work state",
                    "transitions": [{"condition": "always", "target": "DONE"}],
                },
                "DONE": {"agent": None, "description": "Done", "transitions": []},
            },
        }
        handler = GraphRoutedHandler({
            "_graph_data": graph_data,
            "internal_representations": {"enabled": True},
        })
        agents = {"worker": _MockAgent("result")}
        handler.execute(
            [Assignment(agent_name="x", task="Task")],
            agents, logger=None,
        )
        ctx = agents["worker"].calls[0]
        assert "WORKFLOW CONTEXT" in ctx

    def test_max_transitions_safety_valve(self):
        """Execution stops at max_transitions."""
        graph_data = {
            "initial_state": "A",
            "terminal_states": ["DONE"],
            "states": {
                "A": {
                    "agent": "worker",
                    "description": "Loop",
                    "transitions": [
                        {"condition": "passes_remaining <= 0", "target": "DONE"},
                        {"condition": "always", "target": "A"},
                    ],
                },
                "DONE": {"agent": None, "description": "Done", "transitions": []},
            },
        }
        handler = GraphRoutedHandler({
            "_graph_data": graph_data,
            "max_transitions": 5,
        })
        agents = {"worker": _MockAgent("looping")}
        msgs = handler.execute(
            [Assignment(agent_name="x", task="Task")],
            agents, logger=None,
        )
        assert len(msgs) <= 6  # at most 5 transitions + 1 final agent

    def test_llm_graph_mode_with_mock(self):
        """LLM graph mode generates and uses a valid graph."""
        graph_json = json.dumps({
            "initial_state": "START",
            "terminal_states": ["END"],
            "states": {
                "START": {
                    "agent": "doer",
                    "description": "Do it",
                    "transitions": [{"condition": "always", "target": "END"}],
                },
                "END": {"agent": None, "description": "Done", "transitions": []},
            },
        })
        handler = GraphRoutedHandler({
            "graph_mode": "llm_generated",
            "max_transitions": 10,
        })
        agents = {
            "graph_designer": _MockAgent(graph_json),
            "doer": _MockAgent("done by llm graph"),
        }
        msgs = handler.execute(
            [Assignment(agent_name="x", task="Task")],
            agents, logger=None,
        )
        assert any(m.content == "done by llm graph" for m in msgs)


# ---- Coordinator-level integration tests (fast) ----------------------------

class TestCoordinatorWithHandler:
    """Test handler wired through the Coordinator."""

    def test_coordinator_with_graph_routed_handler(self):
        graph_data = {
            "initial_state": "WORK",
            "terminal_states": ["DONE"],
            "states": {
                "WORK": {
                    "agent": "worker",
                    "description": "Work",
                    "transitions": [{"condition": "always", "target": "DONE"}],
                },
                "DONE": {"agent": None, "description": "Done", "transitions": []},
            },
        }
        handler = GraphRoutedHandler({"_graph_data": graph_data})
        strategy = _SingleAgentStrategy(["worker"])
        agents = {"worker": _MockAgent("result")}
        coord = Coordinator(
            agents=agents,
            strategy=strategy,
            config={"termination": {"keyword": "TASK_COMPLETE", "max_turns": 10}},
            execution_handler=handler,
        )
        result = coord.run("do something")
        assert result.final_output == "result"

    def test_coordinator_without_handler_unchanged(self):
        """Without execution_handler, coordinator works as before."""
        strategy = _SingleAgentStrategy(["worker"])
        agents = {"worker": _MockAgent("direct")}
        coord = Coordinator(
            agents=agents,
            strategy=strategy,
            config={"termination": {"keyword": "TASK_COMPLETE", "max_turns": 10}},
        )
        result = coord.run("task")
        assert result.final_output == "direct"

    def test_config_driven_handler_selection(self):
        """Handler is selected from config dict."""
        graph_data = {
            "initial_state": "WORK",
            "terminal_states": ["DONE"],
            "states": {
                "WORK": {
                    "agent": "worker",
                    "description": "Work",
                    "transitions": [{"condition": "always", "target": "DONE"}],
                },
                "DONE": {"agent": None, "description": "Done", "transitions": []},
            },
        }
        handler = GraphRoutedHandler({"_graph_data": graph_data})
        strategy = _SingleAgentStrategy(["worker"])
        agents = {"worker": _MockAgent("result")}
        coord = Coordinator(
            agents=agents,
            strategy=strategy,
            config={
                "execution_handler": "graph_routed",
                "termination": {"keyword": "TASK_COMPLETE", "max_turns": 10},
            },
            execution_handler=handler,
        )
        result = coord.run("task")
        assert result.final_output == "result"


# ---- Cross-prompt metrics end-to-end --------------------------------------

class TestCrossPromptMetrics:
    def test_cross_prompt_from_multiple_runs(self):
        """Metrics computed from multiple handler runs."""
        graph_data = {
            "initial_state": "CLASSIFY",
            "terminal_states": ["COMPLETE"],
            "states": {
                "CLASSIFY": {
                    "agent": "classifier",
                    "description": "Classify",
                    "transitions": [
                        {"condition": "complexity == 'simple'", "target": "WORK"},
                        {"condition": "always", "target": "WORK"},
                    ],
                },
                "WORK": {
                    "agent": "worker",
                    "description": "Work",
                    "transitions": [{"condition": "always", "target": "COMPLETE"}],
                },
                "COMPLETE": {"agent": None, "description": "Done", "transitions": []},
            },
            "resource_budgets": {
                "simple": {
                    "max_passes": 6, "context_budget": 2000,
                    "reasoning_enabled": False, "max_code_review_cycles": 1,
                    "escalation_threshold": 2,
                },
            },
        }

        all_prompt_metrics = []
        for task in ["Build a box", "Build a sphere", "Build a cylinder"]:
            handler = GraphRoutedHandler({"_graph_data": graph_data})
            agents = {
                "classifier": _MockAgent("simple"),
                "worker": _MockAgent("done"),
            }
            handler.execute(
                [Assignment(agent_name="x", task=task)],
                agents, logger=None,
            )
            pm = compute_per_prompt_metrics(
                handler.transition_history,
                initial_complexity="simple",
            )
            all_prompt_metrics.append(pm)

        cross = compute_cross_prompt_metrics(all_prompt_metrics)
        assert cross["total_prompts"] == 3
        assert cross["mean_path_length"] > 0
        assert cross["escalation_rate"] == 0.0
        assert "simple" in cross["mean_path_length_per_complexity"]


# ---- Real LLM integration test (slow) -------------------------------------

@pytest.mark.slow
class TestRealLLMIntegration:
    """Requires GPU. Tests the handler with real model + mock tools."""

    def test_graph_routed_with_real_model(self):
        """Load graph-routed handler + mock tools + real model."""
        from src.config.loader import load_config
        from src.llm.model_loader import load_model
        from src.tools.mock_tools import CalculatorTool, EchoTool

        config = load_config("config/default.yaml")
        model = load_model(config.llm)

        # Build a simple graph: classify → work → done.
        from smolagents import ToolCallingAgent

        tools = [CalculatorTool(), EchoTool()]

        # Create agents for the graph roles.
        classifier = ToolCallingAgent(
            tools=tools, model=model, name="classifier",
            add_base_tools=False, max_steps=3,
        )
        worker = ToolCallingAgent(
            tools=tools, model=model, name="worker",
            add_base_tools=False, max_steps=5,
        )

        graph_data = {
            "initial_state": "CLASSIFY",
            "terminal_states": ["COMPLETE"],
            "states": {
                "CLASSIFY": {
                    "agent": "classifier",
                    "agent_prompt": "Classify this task as simple, moderate, or complex. Output ONLY one word.",
                    "description": "Classify complexity",
                    "transitions": [
                        {"condition": "complexity == 'simple'", "target": "WORK"},
                        {"condition": "complexity == 'moderate'", "target": "WORK"},
                        {"condition": "complexity == 'complex'", "target": "WORK"},
                        {"condition": "always", "target": "WORK"},
                    ],
                },
                "WORK": {
                    "agent": "worker",
                    "agent_prompt": "Calculate 15 * 7 using the calculator_tool.",
                    "description": "Do the work",
                    "transitions": [
                        {"condition": "always", "target": "COMPLETE"},
                    ],
                },
                "COMPLETE": {
                    "agent": None, "description": "Done", "transitions": [],
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
                "complex": {
                    "max_passes": 20, "context_budget": 4000,
                    "reasoning_enabled": True, "max_code_review_cycles": 3,
                    "escalation_threshold": 4,
                },
            },
        }

        handler = GraphRoutedHandler({
            "_graph_data": graph_data,
            "max_transitions": 20,
        })
        logger = InstrumentationLogger()

        agents = {"classifier": classifier, "worker": worker}
        msgs = handler.execute(
            [Assignment(agent_name="classifier", task="Calculate 15 * 7")],
            agents, logger=logger,
        )

        assert len(msgs) >= 2  # at least classifier + worker
        assert len(handler.transition_history) >= 2

        pm = compute_per_prompt_metrics(handler.transition_history)
        assert pm["total_transitions"] >= 2
        rq = compute_routing_quality(
            handler.transition_history, terminal_states=["COMPLETE"],
        )
        assert rq["routing_accuracy"] > 0
