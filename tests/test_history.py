"""Tests for SharedHistory and data structures."""

import time

from src.coordination.history import AgentMessage, SharedHistory, ToolCallRecord


def _msg(agent: str, content: str, turn: int, **kwargs) -> AgentMessage:
    """Helper to build an AgentMessage with minimal boilerplate."""
    return AgentMessage(
        agent_name=agent,
        content=content,
        turn_number=turn,
        timestamp=time.time(),
        **kwargs,
    )


# ---- ToolCallRecord ----------------------------------------------------------

class TestToolCallRecord:
    def test_basic_fields(self):
        rec = ToolCallRecord(
            tool_name="calc", inputs={"expr": "1+1"}, output="2",
            duration_seconds=0.01,
        )
        assert rec.tool_name == "calc"
        assert rec.error is None

    def test_with_error(self):
        rec = ToolCallRecord(
            tool_name="calc", inputs={}, output="",
            duration_seconds=0.0, error="division by zero",
        )
        assert rec.error == "division by zero"


# ---- AgentMessage -------------------------------------------------------------

class TestAgentMessage:
    def test_defaults(self):
        msg = _msg("a", "hello", 1)
        assert msg.is_retry is False
        assert msg.retry_of_turn is None
        assert msg.error is None
        assert msg.tool_calls == []
        assert msg.token_count is None

    def test_retry_fields(self):
        msg = _msg("a", "retry", 3, is_retry=True, retry_of_turn=2)
        assert msg.is_retry is True
        assert msg.retry_of_turn == 2

    def test_with_tool_calls(self):
        rec = ToolCallRecord("echo", {"msg": "hi"}, "hi", 0.001)
        msg = _msg("a", "done", 1, tool_calls=[rec])
        assert len(msg.tool_calls) == 1


# ---- SharedHistory ------------------------------------------------------------

class TestSharedHistory:
    def test_empty(self):
        h = SharedHistory()
        assert len(h) == 0
        assert h.get_all() == []
        assert h.get_recent(5) == []

    def test_append_and_get_all(self):
        h = SharedHistory()
        h.append(_msg("a", "first", 1))
        h.append(_msg("b", "second", 2))
        assert len(h) == 2
        all_msgs = h.get_all()
        assert [m.content for m in all_msgs] == ["first", "second"]

    def test_get_recent(self):
        h = SharedHistory()
        for i in range(10):
            h.append(_msg("agent", f"msg{i}", i))
        recent = h.get_recent(3)
        assert len(recent) == 3
        assert recent[0].content == "msg7"

    def test_get_by_agent(self):
        h = SharedHistory()
        h.append(_msg("a", "a1", 1))
        h.append(_msg("b", "b1", 2))
        h.append(_msg("a", "a2", 3))
        by_a = h.get_by_agent("a")
        assert len(by_a) == 2
        assert by_a[0].content == "a1"
        assert by_a[1].content == "a2"

    def test_get_by_agent_empty(self):
        h = SharedHistory()
        h.append(_msg("a", "hi", 1))
        assert h.get_by_agent("unknown") == []

    def test_to_context_string(self):
        h = SharedHistory()
        h.append(_msg("a", "hello", 1))
        h.append(_msg("b", "world", 2))
        ctx = h.to_context_string()
        assert "[Turn 1] a: hello" in ctx
        assert "[Turn 2] b: world" in ctx

    def test_to_context_string_truncation(self):
        h = SharedHistory()
        for i in range(100):
            h.append(_msg("agent", "x" * 200, i))
        ctx = h.to_context_string(max_tokens=50)
        # 50 tokens ≈ 200 chars — should be truncated
        assert len(ctx) <= 250  # some slack for the "..." prefix
        assert ctx.startswith("...")

    def test_turn_count(self):
        h = SharedHistory()
        assert h.turn_count == 0
        h.append(_msg("a", "hi", 1))
        assert h.turn_count == 1

    def test_get_all_returns_copy(self):
        """Modifying the returned list should not affect the history."""
        h = SharedHistory()
        h.append(_msg("a", "hi", 1))
        all_msgs = h.get_all()
        all_msgs.clear()
        assert len(h) == 1
