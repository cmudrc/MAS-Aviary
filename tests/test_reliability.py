"""Tests for reliability patterns — retry loop, strict schemas, first-step guardrail."""

from unittest.mock import MagicMock, patch

import pytest

from src.llm.reliability import (
    ReliabilityConfig,
    add_strict_properties,
    first_step_guardrail,
    make_first_step_guardrail,
)
from src.llm.thinking_model import ThinkingModel

# ---------------------------------------------------------------------------
# ReliabilityConfig
# ---------------------------------------------------------------------------


class TestReliabilityConfig:
    def test_defaults(self):
        cfg = ReliabilityConfig()
        assert cfg.max_retries == 3
        assert cfg.strict_tool_schemas is True
        assert cfg.first_step_guardrail is True

    def test_from_dict(self):
        cfg = ReliabilityConfig(**{"max_retries": 5, "strict_tool_schemas": False})
        assert cfg.max_retries == 5
        assert cfg.strict_tool_schemas is False
        assert cfg.first_step_guardrail is True  # default preserved

    def test_empty_dict(self):
        cfg = ReliabilityConfig(**{})
        assert cfg.max_retries == 3


# ---------------------------------------------------------------------------
# add_strict_properties
# ---------------------------------------------------------------------------


class TestAddStrictProperties:
    def test_adds_additional_properties_false(self):
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "echo_tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                        },
                    },
                },
            },
        ]
        result = add_strict_properties(schemas)
        assert result[0]["function"]["parameters"]["additionalProperties"] is False

    def test_handles_empty_parameters(self):
        schemas = [{"function": {"name": "no_params_tool", "parameters": {}}}]
        # Empty dict is falsy — should NOT add the key.
        result = add_strict_properties(schemas)
        assert "additionalProperties" not in result[0]["function"]["parameters"]

    def test_handles_missing_function_key(self):
        schemas = [{"type": "function"}]
        # Should not crash.
        result = add_strict_properties(schemas)
        assert result == [{"type": "function"}]

    def test_multiple_tools(self):
        schemas = [
            {"function": {"name": "a", "parameters": {"type": "object", "properties": {"x": {"type": "int"}}}}},
            {"function": {"name": "b", "parameters": {"type": "object", "properties": {"y": {"type": "string"}}}}},
        ]
        add_strict_properties(schemas)
        assert schemas[0]["function"]["parameters"]["additionalProperties"] is False
        assert schemas[1]["function"]["parameters"]["additionalProperties"] is False

    def test_in_place_and_return(self):
        schemas = [{"function": {"name": "t", "parameters": {"type": "object", "properties": {"a": {}}}}}]
        result = add_strict_properties(schemas)
        assert result is schemas  # same reference


# ---------------------------------------------------------------------------
# first_step_guardrail
# ---------------------------------------------------------------------------


class TestFirstStepGuardrail:
    def _make_memory(self, has_observations: bool):
        """Create a mock AgentMemory with optional ActionStep observations."""
        from smolagents.memory import ActionStep
        from smolagents.monitoring import Timing

        memory = MagicMock()
        if has_observations:
            step = ActionStep(
                step_number=1,
                timing=Timing(start_time=0.0, end_time=1.0),
                observations="Tool output here",
            )
            memory.steps = [step]
        else:
            step = ActionStep(
                step_number=1,
                timing=Timing(start_time=0.0, end_time=1.0),
                observations=None,
            )
            memory.steps = [step]
        return memory

    def test_rejects_when_no_observations(self):
        memory = self._make_memory(has_observations=False)
        assert first_step_guardrail("done", memory) is False

    def test_accepts_when_observations_exist(self):
        memory = self._make_memory(has_observations=True)
        assert first_step_guardrail("done", memory) is True

    def test_rejects_empty_memory(self):
        memory = MagicMock()
        memory.steps = []
        assert first_step_guardrail("anything", memory) is False

    def test_make_first_step_guardrail_returns_list(self):
        checks = make_first_step_guardrail()
        assert isinstance(checks, list)
        assert len(checks) == 1
        assert checks[0] is first_step_guardrail


# ---------------------------------------------------------------------------
# ThinkingModel.generate() — retry loop
# ---------------------------------------------------------------------------


def _make_model(reliability: ReliabilityConfig | None = None):
    """Create a ThinkingModel without loading any actual weights."""
    with patch("smolagents.TransformersModel.__init__", return_value=None):
        model = ThinkingModel.__new__(ThinkingModel)
        model.tool_name_key = "name"
        model.tool_arguments_key = "arguments"
        model._reliability = reliability or ReliabilityConfig()
        model._thinking_enabled = True
        model.apply_chat_template_kwargs = {}
        model.model_id = "test-model"
    return model


class TestGenerateRetry:
    def test_success_on_first_attempt(self):
        model = _make_model()
        good_content = '{"name": "echo_tool", "arguments": {"message": "hi"}}'
        fake_msg = MagicMock()
        fake_msg.content = good_content
        fake_msg.tool_calls = None

        with patch.object(type(model).__bases__[0], "generate", return_value=fake_msg) as mock_gen:
            result = model.generate([{"role": "user", "content": "test"}])
            assert mock_gen.call_count == 1
            assert result.tool_calls[0].function.name == "echo_tool"

    def test_retry_on_parse_failure_then_succeed(self):
        model = _make_model(ReliabilityConfig(max_retries=2))

        bad_msg = MagicMock()
        bad_msg.content = "Not valid JSON at all"
        bad_msg.tool_calls = None

        good_msg = MagicMock()
        good_msg.content = '{"name": "echo_tool", "arguments": {"message": "hi"}}'
        good_msg.tool_calls = None

        with patch.object(type(model).__bases__[0], "generate", side_effect=[bad_msg, good_msg]) as mock_gen:
            result = model.generate([{"role": "user", "content": "test"}])
            assert mock_gen.call_count == 2
            assert result.tool_calls[0].function.name == "echo_tool"

    def test_retry_exhaustion_returns_raw(self):
        model = _make_model(ReliabilityConfig(max_retries=1))

        bad_msg = MagicMock()
        bad_msg.content = "still not JSON"
        bad_msg.tool_calls = None

        with patch.object(type(model).__bases__[0], "generate", return_value=bad_msg):
            # After exhausting retries, returns raw message for smolagents recovery.
            result = model.generate([{"role": "user", "content": "test"}])
            assert result is bad_msg

    def test_error_feedback_appended_to_messages(self):
        """On retry, the failed output and error feedback should be appended."""
        model = _make_model(ReliabilityConfig(max_retries=1))

        bad_msg = MagicMock()
        bad_msg.content = "bad output"
        bad_msg.tool_calls = None

        good_msg = MagicMock()
        good_msg.content = '{"name": "echo_tool", "arguments": {"message": "ok"}}'
        good_msg.tool_calls = None

        captured_messages = []

        def capture_generate(msgs, **kwargs):
            captured_messages.append(list(msgs))
            if len(captured_messages) == 1:
                return bad_msg
            return good_msg

        with patch.object(type(model).__bases__[0], "generate", side_effect=capture_generate):
            model.generate([{"role": "user", "content": "test"}])

        # Second call should have the original message + assistant + error feedback.
        second_call_msgs = captured_messages[1]
        assert len(second_call_msgs) == 3
        assert second_call_msgs[1]["role"] == "assistant"
        # Content is in block format for smolagents compatibility.
        assert second_call_msgs[1]["content"] == [{"type": "text", "text": "bad output"}]
        assert second_call_msgs[2]["role"] == "user"
        assert "could not be parsed" in second_call_msgs[2]["content"][0]["text"]

    def test_zero_retries_returns_raw_immediately(self):
        model = _make_model(ReliabilityConfig(max_retries=0))

        bad_msg = MagicMock()
        bad_msg.content = "no JSON"
        bad_msg.tool_calls = None

        with patch.object(type(model).__bases__[0], "generate", return_value=bad_msg) as mock_gen:
            result = model.generate([{"role": "user", "content": "test"}])
            assert result is bad_msg
            assert mock_gen.call_count == 1

    def test_does_not_mutate_original_messages(self):
        model = _make_model(ReliabilityConfig(max_retries=1))

        bad_msg = MagicMock()
        bad_msg.content = "bad"
        bad_msg.tool_calls = None

        good_msg = MagicMock()
        good_msg.content = '{"name": "echo_tool", "arguments": {"message": "ok"}}'
        good_msg.tool_calls = None

        original_messages = [{"role": "user", "content": "test"}]
        original_len = len(original_messages)

        with patch.object(type(model).__bases__[0], "generate", side_effect=[bad_msg, good_msg]):
            model.generate(original_messages)

        assert len(original_messages) == original_len


# ---------------------------------------------------------------------------
# ThinkingModel._prepare_completion_kwargs — strict schemas
# ---------------------------------------------------------------------------


class TestStrictSchemas:
    def test_adds_strict_properties_when_enabled(self):
        model = _make_model(ReliabilityConfig(strict_tool_schemas=True))

        base_result = {
            "messages": [],
            "tools": [
                {"function": {"name": "t", "parameters": {"type": "object", "properties": {"x": {"type": "int"}}}}},
            ],
        }

        with patch.object(type(model).__bases__[0], "_prepare_completion_kwargs", return_value=base_result):
            result = model._prepare_completion_kwargs(messages=[])
            assert result["tools"][0]["function"]["parameters"]["additionalProperties"] is False

    def test_skips_when_disabled(self):
        model = _make_model(ReliabilityConfig(strict_tool_schemas=False))

        base_result = {
            "messages": [],
            "tools": [
                {"function": {"name": "t", "parameters": {"type": "object", "properties": {"x": {"type": "int"}}}}},
            ],
        }

        with patch.object(type(model).__bases__[0], "_prepare_completion_kwargs", return_value=base_result):
            result = model._prepare_completion_kwargs(messages=[])
            assert "additionalProperties" not in result["tools"][0]["function"]["parameters"]

    def test_no_tools_key_no_crash(self):
        model = _make_model(ReliabilityConfig(strict_tool_schemas=True))

        base_result = {"messages": []}

        with patch.object(type(model).__bases__[0], "_prepare_completion_kwargs", return_value=base_result):
            result = model._prepare_completion_kwargs(messages=[])
            assert "tools" not in result


# ---------------------------------------------------------------------------
# Integration: model_loader.py
# ---------------------------------------------------------------------------


class TestModelLoaderIntegration:
    def test_reliability_config_from_llm_config(self):
        from src.config.loader import LLMConfig
        from src.llm.reliability import ReliabilityConfig

        config = LLMConfig(reliability={"max_retries": 5, "strict_tool_schemas": False})
        cfg = ReliabilityConfig(**(config.reliability or {}))
        assert cfg.max_retries == 5
        assert cfg.strict_tool_schemas is False

    def test_empty_reliability_uses_defaults(self):
        from src.config.loader import LLMConfig
        from src.llm.reliability import ReliabilityConfig

        config = LLMConfig()
        cfg = ReliabilityConfig(**(config.reliability or {}))
        assert cfg.max_retries == 3
        assert cfg.strict_tool_schemas is True
        assert cfg.first_step_guardrail is True


# ---------------------------------------------------------------------------
# Integration: OrchestratorContext wiring
# ---------------------------------------------------------------------------


class TestOrchestratorContextGuardrail:
    def test_context_accepts_worker_final_answer_checks(self):
        from src.tools.orchestrator_tools import OrchestratorContext

        checks = make_first_step_guardrail()
        ctx = OrchestratorContext(
            available_tools={},
            agents={},
            model=MagicMock(),
            worker_final_answer_checks=checks,
        )
        assert ctx.worker_final_answer_checks == checks
        assert len(ctx.worker_final_answer_checks) == 1

    def test_context_defaults_to_empty(self):
        from src.tools.orchestrator_tools import OrchestratorContext

        ctx = OrchestratorContext(
            available_tools={},
            agents={},
            model=MagicMock(),
        )
        assert ctx.worker_final_answer_checks == []


# ---------------------------------------------------------------------------
# GatedFinalAnswer content validation
# ---------------------------------------------------------------------------


class TestGatedFinalAnswerValidation:
    def _make_context(self, agents=None, assignments=None):
        from src.tools.orchestrator_tools import OrchestratorContext

        ctx = OrchestratorContext(
            available_tools={},
            agents={},
            model=MagicMock(),
        )
        ctx.created_agents = agents or []
        ctx.assignments = assignments or []
        return ctx

    def test_rejects_no_agents(self):
        from src.tools.orchestrator_tools import GatedFinalAnswer

        ctx = self._make_context()
        tool = GatedFinalAnswer(ctx)
        with pytest.raises(ValueError, match="not created any agents"):
            tool.forward("DELEGATION_COMPLETE")

    def test_rejects_no_assignments(self):
        from src.tools.orchestrator_tools import GatedFinalAnswer

        ctx = self._make_context(agents=["worker1"])
        tool = GatedFinalAnswer(ctx)
        with pytest.raises(ValueError, match="not assigned any tasks"):
            tool.forward("DELEGATION_COMPLETE")

    def test_rejects_non_delegation_content(self):
        from src.tools.orchestrator_tools import GatedFinalAnswer

        ctx = self._make_context(
            agents=["worker1"],
            assignments=[{"agent_name": "worker1", "task": "do stuff"}],
        )
        tool = GatedFinalAnswer(ctx)
        with pytest.raises(ValueError, match="DELEGATION_COMPLETE"):
            tool.forward("Here is the solution code: import openmdao...")

    def test_accepts_delegation_complete(self):
        from src.tools.orchestrator_tools import GatedFinalAnswer

        ctx = self._make_context(
            agents=["worker1"],
            assignments=[{"agent_name": "worker1", "task": "do stuff"}],
        )
        tool = GatedFinalAnswer(ctx)
        result = tool.forward("DELEGATION_COMPLETE")
        assert result == "DELEGATION_COMPLETE"

    def test_accepts_delegation_complete_with_extra_text(self):
        from src.tools.orchestrator_tools import GatedFinalAnswer

        ctx = self._make_context(
            agents=["worker1"],
            assignments=[{"agent_name": "worker1", "task": "do stuff"}],
        )
        tool = GatedFinalAnswer(ctx)
        result = tool.forward("DELEGATION_COMPLETE - all tasks assigned")
        assert "DELEGATION_COMPLETE" in result
