"""Tests for logging, metrics, and export."""

import json
import time

import pytest

from src.coordination.history import AgentMessage, ToolCallRecord
from src.logging.exporter import export_run
from src.logging.logger import InstrumentationLogger
from src.logging.metrics import compute_metrics


def _msg(agent: str, content: str, turn: int, **kwargs) -> AgentMessage:
    return AgentMessage(
        agent_name=agent, content=content, turn_number=turn,
        timestamp=time.time(), **kwargs,
    )


# ---- compute_metrics ----------------------------------------------------------

class TestComputeMetrics:
    def test_empty_history(self):
        m = compute_metrics([])
        assert m["total_messages"] == 0
        assert m["total_tokens"] is None
        assert m["coordination_efficiency"] == 0.0

    def test_basic_counts(self):
        msgs = [
            _msg("a", "hello", 1, duration_seconds=1.0),
            _msg("b", "world", 2, duration_seconds=2.0),
        ]
        m = compute_metrics(msgs)
        assert m["total_messages"] == 2
        assert m["total_duration_seconds"] == 3.0
        assert m["error_count"] == 0
        assert m["retry_count"] == 0

    def test_error_metrics(self):
        msgs = [
            _msg("a", "ok", 1, duration_seconds=0.5),
            _msg("a", "", 2, duration_seconds=0.1, error="crash"),
            _msg("a", "ok again", 3, duration_seconds=0.5),
        ]
        m = compute_metrics(msgs)
        assert m["error_count"] == 1
        assert m["error_rate"] == pytest.approx(1 / 3, abs=0.01)

    def test_retry_metrics(self):
        msgs = [
            _msg("a", "", 1, duration_seconds=0.1, error="fail"),
            _msg("a", "ok", 2, duration_seconds=0.5, is_retry=True, retry_of_turn=1),
        ]
        m = compute_metrics(msgs)
        assert m["retry_count"] == 1
        assert m["retry_rate"] == 0.5

    def test_tool_call_metrics(self):
        tc_ok = ToolCallRecord("echo", {"msg": "hi"}, "hi", 0.01)
        tc_fail = ToolCallRecord("calc", {"expr": "1/0"}, "", 0.02, error="div by zero")
        msgs = [
            _msg("a", "done", 1, duration_seconds=0.5, tool_calls=[tc_ok, tc_fail]),
        ]
        m = compute_metrics(msgs)
        assert m["total_tool_calls"] == 2
        assert m["tool_error_rate"] == 0.5

    def test_token_counts(self):
        msgs = [
            _msg("a", "x", 1, duration_seconds=0.1, token_count=100),
            _msg("b", "y", 2, duration_seconds=0.1, token_count=200),
        ]
        m = compute_metrics(msgs)
        assert m["total_tokens"] == 300

    def test_token_counts_none(self):
        msgs = [_msg("a", "x", 1, duration_seconds=0.1)]
        m = compute_metrics(msgs)
        assert m["total_tokens"] is None

    def test_redundancy_detection(self):
        msgs = [
            _msg("a", "the quick brown fox jumps over the lazy dog", 1, duration_seconds=0.1),
            _msg("b", "the quick brown fox jumps over the lazy dog", 2, duration_seconds=0.1),
        ]
        m = compute_metrics(msgs)
        assert m["redundancy_rate"] > 0

    def test_no_redundancy_different_content(self):
        msgs = [
            _msg("a", "alpha beta gamma", 1, duration_seconds=0.1),
            _msg("b", "delta epsilon zeta", 2, duration_seconds=0.1),
        ]
        m = compute_metrics(msgs)
        assert m["redundancy_rate"] == 0.0

    def test_coordination_efficiency(self):
        msgs = [
            _msg("a", "unique output one", 1, duration_seconds=0.1),
            _msg("b", "unique output two", 2, duration_seconds=0.1),
        ]
        m = compute_metrics(msgs)
        assert m["coordination_efficiency"] == 1.0


# ---- InstrumentationLogger ----------------------------------------------------

class TestInstrumentationLogger:
    def test_log_and_retrieve(self):
        logger = InstrumentationLogger()
        msg = _msg("a", "hello", 1, duration_seconds=0.5)
        logger.log_turn(msg)
        assert logger.turn_count == 1
        assert logger.get_latest() is msg
        assert len(logger.get_messages()) == 1

    def test_compute_metrics_integration(self):
        logger = InstrumentationLogger()
        logger.log_turn(_msg("a", "x", 1, duration_seconds=1.0))
        logger.log_turn(_msg("b", "y", 2, duration_seconds=2.0))
        m = logger.compute_metrics()
        assert m["total_messages"] == 2

    def test_empty_logger(self):
        logger = InstrumentationLogger()
        assert logger.turn_count == 0
        assert logger.get_latest() is None


# ---- JSON Export ---------------------------------------------------------------

class TestExporter:
    def test_export_creates_file(self, tmp_path):
        msgs = [_msg("a", "hello", 1, duration_seconds=0.5)]
        metrics = compute_metrics(msgs)
        path = str(tmp_path / "test_run.json")
        result_path = export_run(msgs, metrics, path)
        assert result_path == path

        with open(path) as f:
            data = json.load(f)
        assert "history" in data
        assert "metrics" in data
        assert len(data["history"]) == 1
        assert data["history"][0]["agent_name"] == "a"
        assert data["metrics"]["total_messages"] == 1

    def test_logger_export_json(self, tmp_path):
        logger = InstrumentationLogger({"logging": {"output_dir": str(tmp_path)}})
        logger.log_turn(_msg("a", "test", 1, duration_seconds=0.1))
        path = logger.export_json()
        assert path.startswith(str(tmp_path))

        with open(path) as f:
            data = json.load(f)
        assert data["metrics"]["total_messages"] == 1
