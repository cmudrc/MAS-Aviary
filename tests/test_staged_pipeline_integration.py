"""Integration tests for the Staged Pipeline operational methodology.

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

from src.coordination.completion_criteria import CompletionCriteria
from src.coordination.coordinator import Coordinator
from src.coordination.execution_handler import Assignment
from src.coordination.stage_definition import (
    PipelineDefinition,
    StageDefinition,
)
from src.coordination.staged_pipeline_handler import (
    StagedPipelineHandler,
)
from src.coordination.strategy import CoordinationAction, CoordinationStrategy
from src.logging.staged_pipeline_metrics import (
    compute_cross_prompt_metrics,
    compute_error_propagation,
    compute_per_prompt_metrics,
)

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

    def __init__(self, response: str = "done"):
        self._response = response
        self.calls: list[str] = []
        self.logs = []

    def run(self, ctx: str):
        self.calls.append(ctx)
        return self._response


class _MockAgentWithTools:
    """Agent stub that records tool calls in its logs."""

    def __init__(self, response: str, tool_names: list[str]):
        self._response = response
        self.calls: list[str] = []
        # Simulate smolagents tool call logs.
        self.logs = [_MockStep(tool_names)]

    def run(self, ctx: str):
        self.calls.append(ctx)
        return self._response


class _MockStep:
    def __init__(self, tool_names: list[str]):
        self.tool_calls = [_MockToolCall(n) for n in tool_names]


class _MockToolCall:
    def __init__(self, name: str):
        self.name = name
        self.arguments = {}


class _FailingAgent:
    """Agent that raises on every call."""

    def __init__(self, error: str = "RuntimeError: boom"):
        self._error = error
        self.calls: list[str] = []
        self.logs = []

    def run(self, ctx: str):
        self.calls.append(ctx)
        raise RuntimeError(self._error)


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


def _sample_pipeline() -> PipelineDefinition:
    """Create a 4-stage sample pipeline for integration tests."""
    return PipelineDefinition(stages=[
        StageDefinition(
            name="design_planning",
            completion_criteria=CompletionCriteria(
                type="output_contains", check="non_empty_output",
            ),
            stage_prompt="Produce a design plan.",
        ),
        StageDefinition(
            name="code_writing",
            completion_criteria=CompletionCriteria(
                type="output_contains", check="code_block",
            ),
            stage_prompt="Write Python code for the simulation.",
        ),
        StageDefinition(
            name="code_execution",
            completion_criteria=CompletionCriteria(
                type="tool_attempted", check="tool_called",
                tool_name="run_simulation",
            ),
            stage_prompt="Run the simulation.",
        ),
        StageDefinition(
            name="output_review",
            completion_criteria=CompletionCriteria(
                type="output_contains", check="verdict_present",
            ),
            stage_prompt="Review the result. State ACCEPTABLE, ISSUES, or FAILED.",
        ),
    ])


# ---- Handler-level integration tests (fast) --------------------------------

class TestHandlerIntegration:
    def test_happy_path_all_met(self):
        """4-stage pipeline, all completion criteria met."""
        pipeline = _sample_pipeline()
        handler = StagedPipelineHandler({})
        handler._pipeline = pipeline

        agents = {
            "designer": _MockAgent("Plan: Define aircraft parameters and constraints"),
            "coder": _MockAgent("```python\nimport openmdao.api as om\nprob = om.Problem()\n```"),
            "executor": _MockAgentWithTools("Execution succeeded, results produced", ["run_simulation"]),
            "reviewer": _MockAgent("ACCEPTABLE — simulation completed. TASK_COMPLETE"),
        }
        assignments = [
            Assignment(agent_name="designer", task="Run analysis"),
            Assignment(agent_name="coder", task="Run analysis"),
            Assignment(agent_name="executor", task="Run analysis"),
            Assignment(agent_name="reviewer", task="Run analysis"),
        ]

        msgs = handler.execute(assignments, agents, None)
        results = handler.last_stage_results
        assert all(r.completion_met for r in results)
        assert len(msgs) == 4

    def test_error_propagation_chain(self):
        """Stage 2 fails → stage 3 gets failure notice → stage 4 gets chain."""
        pipeline = _sample_pipeline()
        handler = StagedPipelineHandler({})
        handler._pipeline = pipeline

        agents = {
            "designer": _MockAgent("Good design plan"),
            "coder": _MockAgent("Here is the plan restated"),  # no code → NOT MET
            "executor": _MockAgent("No tool to call"),  # no tool → NOT MET
            "reviewer": _MockAgent("FAILED — execution produced no output"),
        }
        assignments = [
            Assignment(agent_name="designer", task="Run analysis"),
            Assignment(agent_name="coder", task="Run analysis"),
            Assignment(agent_name="executor", task="Run analysis"),
            Assignment(agent_name="reviewer", task="Run analysis"),
        ]

        msgs = handler.execute(assignments, agents, None)
        assert len(msgs) == 4
        results = handler.last_stage_results
        assert results[0].completion_met is True   # non-empty
        assert results[1].completion_met is False   # no code block
        assert results[2].completion_met is False   # no tool called
        assert results[3].completion_met is True    # verdict present

    def test_context_passing_between_stages(self):
        """Each stage receives the previous stage's output."""
        pipeline = PipelineDefinition(stages=[
            StageDefinition(
                name="s1",
                completion_criteria=CompletionCriteria(
                    type="any", check="always"
                ),
                stage_prompt="Prompt1",
            ),
            StageDefinition(
                name="s2",
                completion_criteria=CompletionCriteria(
                    type="any", check="always"
                ),
                stage_prompt="Prompt2",
            ),
        ])
        handler = StagedPipelineHandler({})
        handler._pipeline = pipeline

        a1 = _MockAgent("stage1_output")
        a2 = _MockAgent("stage2_output")
        handler.execute(
            [Assignment(agent_name="a1", task="task"),
             Assignment(agent_name="a2", task="task")],
            {"a1": a1, "a2": a2}, None,
        )
        # Stage 2 should have received stage 1's output.
        assert "stage1_output" in a2.calls[0]

    def test_always_advances(self):
        """Pipeline advances even when completion criteria are not met."""
        pipeline = PipelineDefinition(stages=[
            StageDefinition(
                name="s1",
                completion_criteria=CompletionCriteria(
                    type="output_contains", check="code_block"
                ),
            ),
            StageDefinition(
                name="s2",
                completion_criteria=CompletionCriteria(
                    type="output_contains", check="code_block"
                ),
            ),
            StageDefinition(
                name="s3",
                completion_criteria=CompletionCriteria(
                    type="any", check="always"
                ),
            ),
        ])
        handler = StagedPipelineHandler({})
        handler._pipeline = pipeline

        agents = {
            "a": _MockAgent("no code"),
            "b": _MockAgent("still no code"),
            "c": _MockAgent("final"),
        }
        msgs = handler.execute(
            [Assignment(agent_name="a", task="t"),
             Assignment(agent_name="b", task="t"),
             Assignment(agent_name="c", task="t")],
            agents, None,
        )
        assert len(msgs) == 3  # all ran despite failures

    def test_missing_agent_still_advances(self):
        """Missing agent records error, pipeline continues."""
        pipeline = PipelineDefinition(stages=[
            StageDefinition(name="s1", completion_criteria=CompletionCriteria(type="any", check="always")),
            StageDefinition(name="s2", completion_criteria=CompletionCriteria(type="any", check="always")),
        ])
        handler = StagedPipelineHandler({})
        handler._pipeline = pipeline

        msgs = handler.execute(
            [Assignment(agent_name="missing", task="t"),
             Assignment(agent_name="exists", task="t")],
            {"exists": _MockAgent("ok")}, None,
        )
        assert len(msgs) == 2
        assert msgs[0].error is not None
        assert "ok" in msgs[1].content


# ---- Metrics integration ---------------------------------------------------

class TestMetricsIntegration:
    def test_metrics_from_real_run(self):
        """Per-prompt and error propagation metrics from a handler run."""
        pipeline = _sample_pipeline()
        handler = StagedPipelineHandler({})
        handler._pipeline = pipeline

        agents = {
            "d": _MockAgent("plan"),
            "c": _MockAgent("no code here"),  # NOT MET
            "e": _MockAgent("no tool"),        # NOT MET
            "r": _MockAgent("FAILED"),         # MET
        }
        handler.execute(
            [Assignment(agent_name="d", task="t"),
             Assignment(agent_name="c", task="t"),
             Assignment(agent_name="e", task="t"),
             Assignment(agent_name="r", task="t")],
            agents, None,
        )

        pm = compute_per_prompt_metrics(handler.last_stage_results)
        assert pm["stage_count"] == 4
        assert pm["stages_completed"] == 2  # s1 (non-empty) + s4 (verdict)
        assert abs(pm["completion_rate"] - 0.5) < 0.01

        ep = compute_error_propagation(handler.last_stage_results)
        assert ep["chain_length"] == 2  # s2, s3 consecutive

    def test_cross_prompt_metrics(self):
        """Cross-prompt aggregation from multiple runs."""
        pipeline = PipelineDefinition(stages=[
            StageDefinition(name="s1", completion_criteria=CompletionCriteria(type="any", check="always")),
            StageDefinition(name="s2", completion_criteria=CompletionCriteria(type="any", check="always")),
        ])
        handler = StagedPipelineHandler({})
        handler._pipeline = pipeline

        # Run 1: all met.
        handler.execute(
            [Assignment(agent_name="a", task="t1"),
             Assignment(agent_name="b", task="t1")],
            {"a": _MockAgent("ok"), "b": _MockAgent("ok")}, None,
        )
        pm1 = compute_per_prompt_metrics(handler.last_stage_results)

        # Run 2: different results.
        pipeline2 = PipelineDefinition(stages=[
            StageDefinition(
                name="s1",
                completion_criteria=CompletionCriteria(
                    type="output_contains", check="code_block"
                ),
            ),
            StageDefinition(
                name="s2",
                completion_criteria=CompletionCriteria(
                    type="output_contains", check="code_block"
                ),
            ),
        ])
        handler._pipeline = pipeline2
        handler.execute(
            [Assignment(agent_name="a", task="t2"),
             Assignment(agent_name="b", task="t2")],
            {"a": _MockAgent("no code"), "b": _MockAgent("no code")}, None,
        )
        pm2 = compute_per_prompt_metrics(handler.last_stage_results)

        cm = compute_cross_prompt_metrics([pm1, pm2])
        assert cm["total_prompts"] == 2
        assert cm["mean_completion_rate"] > 0.0


# ---- Coordinator wiring integration ----------------------------------------

class TestCoordinatorWiring:
    def test_coordinator_with_staged_pipeline_handler(self):
        """Coordinator uses StagedPipelineHandler when configured."""
        pipeline = PipelineDefinition(stages=[
            StageDefinition(name="s1", completion_criteria=CompletionCriteria(type="any", check="always")),
        ])
        handler = StagedPipelineHandler({})
        handler._pipeline = pipeline

        strategy = _SingleAgentStrategy(["worker"])
        agents = {"worker": _MockAgent("pipeline result")}

        coord = Coordinator(
            agents=agents,
            strategy=strategy,
            config={"max_turns": 10, "termination_keyword": ""},
            execution_handler=handler,
        )
        result = coord.run("Do something")
        assert "pipeline result" in result.final_output


# ---- Config loading integration --------------------------------------------

class TestConfigLoading:
    def test_staged_pipeline_config_loads(self):
        """staged_pipeline.yaml loads without errors."""
        import yaml
        with open("config/staged_pipeline.yaml") as f:
            cfg = yaml.safe_load(f)
        sp = cfg["staged_pipeline"]
        assert "pipeline" in sp
        assert sp["context_mode"] == "last_only"
        assert sp["include_completion_status"] is True

    def test_handler_from_config(self):
        """StagedPipelineHandler initializes from config dict."""
        import yaml
        with open("config/staged_pipeline.yaml") as f:
            cfg = yaml.safe_load(f)
        handler = StagedPipelineHandler(cfg["staged_pipeline"])
        assert handler._context_mode == "last_only"
        assert handler._include_completion_status is True


# ---- Context mode integration tests ----------------------------------------

class TestContextModeIntegration:
    def test_all_stages_mode_includes_all_history(self):
        """all_stages mode gives the final stage all previous outputs."""
        pipeline = PipelineDefinition(stages=[
            StageDefinition(name="s1", completion_criteria=CompletionCriteria(type="any", check="always")),
            StageDefinition(name="s2", completion_criteria=CompletionCriteria(type="any", check="always")),
            StageDefinition(name="s3", completion_criteria=CompletionCriteria(type="any", check="always")),
        ])
        handler = StagedPipelineHandler({"context_mode": "all_stages"})
        handler._pipeline = pipeline

        a1 = _MockAgent("ALPHA")
        a2 = _MockAgent("BETA")
        a3 = _MockAgent("GAMMA")
        handler.execute(
            [Assignment(agent_name="a1", task="t"),
             Assignment(agent_name="a2", task="t"),
             Assignment(agent_name="a3", task="t")],
            {"a1": a1, "a2": a2, "a3": a3}, None,
        )
        # Stage 3 should see both ALPHA and BETA.
        assert "ALPHA" in a3.calls[0]
        assert "BETA" in a3.calls[0]


# ---- Slow test (real LLM) -------------------------------------------------

@pytest.mark.slow
class TestStagedPipelineRealLLM:
    """End-to-end with real LLM and mock tools. Requires GPU."""

    def test_staged_pipeline_with_real_model(self):
        from smolagents import ToolCallingAgent, TransformersModel

        from src.tools.mock_tools import CalculatorTool, EchoTool

        model = TransformersModel(
            model_id="Qwen/Qwen3-8B",
            max_new_tokens=512,
        )

        tools = [EchoTool(), CalculatorTool()]

        # Create 2-stage pipeline: compute → review.
        pipeline = PipelineDefinition(stages=[
            StageDefinition(
                name="compute",
                completion_criteria=CompletionCriteria(
                    type="output_contains", check="non_empty_output",
                ),
                stage_prompt="Calculate the sum of 15 and 27 using the calculator tool.",
            ),
            StageDefinition(
                name="review",
                completion_criteria=CompletionCriteria(
                    type="output_contains", check="verdict_present",
                ),
                stage_prompt="Review the previous result. If correct, say ACCEPTABLE. If wrong, say FAILED.",
            ),
        ])

        # Build agents for each stage.
        agent1 = ToolCallingAgent(
            tools=tools, model=model,
            name="compute_agent", add_base_tools=False, max_steps=5,
        )
        agent2 = ToolCallingAgent(
            tools=tools, model=model,
            name="review_agent", add_base_tools=False, max_steps=5,
        )

        handler = StagedPipelineHandler({"termination_keyword": ""})
        handler._pipeline = pipeline

        msgs = handler.execute(
            [Assignment(agent_name="compute_agent", task="Calculate the sum of 15 and 27"),
             Assignment(agent_name="review_agent", task="Review the result")],
            {"compute_agent": agent1, "review_agent": agent2},
            logger=None,
        )

        assert len(msgs) == 2
        # At least one stage should have produced output.
        assert any(m.content for m in msgs)

        # Metrics should be computable.
        pm = compute_per_prompt_metrics(handler.last_stage_results)
        assert pm["stage_count"] == 2
        assert pm["completion_rate"] > 0.0
