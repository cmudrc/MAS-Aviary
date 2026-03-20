"""Tests for ThinkingModel — think-block stripping and robust JSON parsing."""

import json

import pytest

from src.llm.thinking_model import ThinkingModel, _find_tool_call_json, strip_think_blocks

# ---------------------------------------------------------------------------
# strip_think_blocks
# ---------------------------------------------------------------------------

class TestStripThinkBlocks:
    def test_removes_single_block(self):
        text = "<think>reasoning here</think>answer"
        assert strip_think_blocks(text) == "answer"

    def test_removes_multiple_blocks(self):
        text = "<think>a</think>mid<think>b</think>end"
        assert strip_think_blocks(text) == "midend"

    def test_removes_multiline_block(self):
        text = "<think>\nline1\nline2\n{braces}\n</think>\nresult"
        assert strip_think_blocks(text) == "result"

    def test_no_think_blocks_unchanged(self):
        text = '{"name": "final_answer", "arguments": {"answer": "42"}}'
        assert strip_think_blocks(text) == text

    def test_empty_think_block(self):
        text = "<think></think>payload"
        assert strip_think_blocks(text) == "payload"

    def test_nested_angle_brackets_in_think(self):
        text = "<think>if x < 10 and y > 5: pass</think>result"
        assert strip_think_blocks(text) == "result"

    def test_think_with_json_inside(self):
        """Braces inside think blocks must not leak to the parser."""
        text = (
            '<think>{"name": "wrong", "arguments": {}}</think>\n'
            '{"name": "final_answer", "arguments": {"answer": "42"}}'
        )
        cleaned = strip_think_blocks(text)
        assert "<think>" not in cleaned
        obj = json.loads(cleaned)
        assert obj["name"] == "final_answer"


# ---------------------------------------------------------------------------
# _find_tool_call_json — robust parser
# ---------------------------------------------------------------------------

class TestFindToolCallJson:
    def test_clean_json(self):
        text = '{"name": "calculator_tool", "arguments": {"expression": "2+2"}}'
        result = _find_tool_call_json(text)
        assert result["name"] == "calculator_tool"
        assert result["arguments"]["expression"] == "2+2"

    def test_json_with_preamble(self):
        text = 'Some analysis text\n{"name": "echo_tool", "arguments": {"message": "hi"}}'
        result = _find_tool_call_json(text)
        assert result["name"] == "echo_tool"

    def test_json_with_trailing_text(self):
        text = '{"name": "echo_tool", "arguments": {"message": "hi"}}\nMore text here.'
        result = _find_tool_call_json(text)
        assert result["name"] == "echo_tool"

    def test_preamble_with_braces(self):
        """Braces in analysis text before the tool call should not confuse the parser."""
        text = (
            'The expression {x: 1} is interesting.\n'
            '{"name": "calculator_tool", "arguments": {"expression": "2*3*7"}}'
        )
        result = _find_tool_call_json(text)
        assert result["name"] == "calculator_tool"
        assert result["arguments"]["expression"] == "2*3*7"

    def test_tool_call_tags(self):
        """Model wraps JSON in <tool_call> tags."""
        text = '<tool_call>\n{"name": "final_answer", "arguments": {"answer": "42"}}\n</tool_call>'
        result = _find_tool_call_json(text)
        assert result["name"] == "final_answer"

    def test_multiple_json_objects_picks_tool_call(self):
        """When multiple JSON objects exist, pick the one with name+arguments."""
        text = (
            '{"status": "thinking"}\n'
            '{"name": "echo_tool", "arguments": {"message": "done"}}\n'
            '{"extra": true}'
        )
        result = _find_tool_call_json(text)
        assert result["name"] == "echo_tool"
        assert result["arguments"]["message"] == "done"

    def test_nested_json_in_arguments(self):
        text = '{"name": "complex_tool", "arguments": {"data": {"nested": [1, 2, 3]}}}'
        result = _find_tool_call_json(text)
        assert result["name"] == "complex_tool"
        assert result["arguments"]["data"]["nested"] == [1, 2, 3]

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="does not contain any JSON"):
            _find_tool_call_json("No JSON here at all.")

    def test_no_json_braces_only_raises(self):
        """Malformed braces with no valid JSON should raise."""
        with pytest.raises(ValueError, match="does not contain any JSON"):
            _find_tool_call_json("Just some {broken text")

    def test_fallback_to_non_tool_call_json(self):
        """If no object has name+arguments, return the first valid one."""
        text = '{"answer": "42"}'
        result = _find_tool_call_json(text)
        assert result["answer"] == "42"

    def test_real_qwen3_think_then_tool_call(self):
        """Simulate actual Qwen3 output: <think> stripped, then tool_call JSON."""
        raw = (
            "<think>\nOkay, I need to calculate 2*3*7.\n"
            "The expression is straightforward.\n"
            "Let me use the calculator tool.\n</think>\n\n"
            '<tool_call>\n{"name": "calculator_tool", "arguments": {"expression": "2*3*7"}}\n</tool_call>'
        )
        cleaned = strip_think_blocks(raw)
        result = _find_tool_call_json(cleaned)
        assert result["name"] == "calculator_tool"
        assert result["arguments"]["expression"] == "2*3*7"

    def test_real_qwen3_think_with_braces(self):
        """Simulate Qwen3 output where think block contains JSON-like braces."""
        raw = (
            '<think>\nThe JSON should be {"name": "final_answer", "arguments": {"answer": "42"}}\n'
            "But let me check the syntax. The correct format is:\n"
            '{"name": "final_answer", "arguments": {"answer": "42"}}\n'
            "I need to ensure proper formatting.\n</think>\n\n"
            '<tool_call>\n{"name": "final_answer", "arguments": {"answer": "42"}}\n</tool_call>'
        )
        cleaned = strip_think_blocks(raw)
        result = _find_tool_call_json(cleaned)
        assert result["name"] == "final_answer"
        assert result["arguments"]["answer"] == "42"


# ---------------------------------------------------------------------------
# ThinkingModel.parse_tool_calls (integration with ChatMessage)
# ---------------------------------------------------------------------------

class TestParseToolCalls:
    """Test the override via a minimal ThinkingModel (no real GPU needed)."""

    @pytest.fixture()
    def _chat_message_cls(self):
        from smolagents.models import ChatMessage, MessageRole
        return ChatMessage, MessageRole

    def _make_message(self, content: str):
        from smolagents.models import ChatMessage, MessageRole
        return ChatMessage(role=MessageRole.USER, content=content)

    def _make_model(self):
        """Create a ThinkingModel without loading any actual weights."""
        from unittest.mock import patch

        with patch("smolagents.TransformersModel.__init__", return_value=None):
            model = ThinkingModel.__new__(ThinkingModel)
            model.tool_name_key = "name"
            model.tool_arguments_key = "arguments"
        return model

    def test_clean_tool_call(self):
        model = self._make_model()
        msg = self._make_message('{"name": "echo_tool", "arguments": {"message": "hi"}}')
        result = model.parse_tool_calls(msg)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "echo_tool"
        assert result.tool_calls[0].function.arguments == {"message": "hi"}

    def test_think_block_stripped_before_parse(self):
        model = self._make_model()
        raw = (
            '<think>Reasoning with {braces} and "quotes"</think>\n'
            '{"name": "calculator_tool", "arguments": {"expression": "6*7"}}'
        )
        msg = self._make_message(raw)
        result = model.parse_tool_calls(msg)
        assert result.tool_calls[0].function.name == "calculator_tool"
        assert "<think>" not in result.content

    def test_preamble_braces_dont_break_parse(self):
        model = self._make_model()
        text = (
            'Analysis: the set {2, 3, 7} are primes.\n'
            '{"name": "final_answer", "arguments": {"answer": "2, 3, 7"}}'
        )
        msg = self._make_message(text)
        result = model.parse_tool_calls(msg)
        assert result.tool_calls[0].function.name == "final_answer"

    def test_role_set_to_assistant(self):
        model = self._make_model()
        msg = self._make_message('{"name": "echo_tool", "arguments": {"message": "x"}}')
        result = model.parse_tool_calls(msg)
        from smolagents.models import MessageRole
        assert result.role == MessageRole.ASSISTANT

    def test_existing_tool_calls_preserved(self):
        """If tool_calls already set, parse_tool_calls should not re-parse."""
        from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole

        model = self._make_model()
        existing_call = ChatMessageToolCall(
            id="test-id",
            type="function",
            function=ChatMessageToolCallFunction(name="echo_tool", arguments={"message": "hi"}),
        )
        msg = ChatMessage(role=MessageRole.USER, content="anything", tool_calls=[existing_call])
        result = model.parse_tool_calls(msg)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "echo_tool"

    def test_no_content_raises(self):
        model = self._make_model()
        from smolagents.models import ChatMessage, MessageRole
        msg = ChatMessage(role=MessageRole.USER, content=None)
        with pytest.raises(AssertionError, match="no content and no tool calls"):
            model.parse_tool_calls(msg)

    def test_string_arguments_parsed(self):
        """Arguments passed as JSON string should be parsed to dict."""
        model = self._make_model()
        text = '{"name": "echo_tool", "arguments": "{\\"message\\": \\"hi\\"}"}'
        msg = self._make_message(text)
        result = model.parse_tool_calls(msg)
        assert result.tool_calls[0].function.arguments == {"message": "hi"}
