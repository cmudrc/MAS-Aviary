"""Tests for CLI entry point and Coordinator.from_config."""

import json
from unittest.mock import MagicMock, patch

import pytest

from main import main, parse_args
from src.coordination.coordinator import _STRATEGY_MAP, _load_strategy
from src.coordination.strategies.graph_routed import GraphRoutedStrategy
from src.coordination.strategies.sequential import SequentialStrategy

# ---- parse_args --------------------------------------------------------------

class TestParseArgs:
    def test_task_only(self):
        args = parse_args(["Do something"])
        assert args.task == "Do something"
        assert args.config == "config/default.yaml"
        assert args.strategy is None
        assert args.export is None

    def test_with_strategy(self):
        args = parse_args(["--strategy", "sequential", "My task"])
        assert args.strategy == "sequential"
        assert args.task == "My task"

    def test_with_config(self):
        args = parse_args(["--config", "custom.yaml", "Task"])
        assert args.config == "custom.yaml"

    def test_with_export(self):
        args = parse_args(["--export", "output.json", "Task"])
        assert args.export == "output.json"

    def test_all_options(self):
        args = parse_args([
            "--config", "c.yaml",
            "--strategy", "orchestrated",
            "--export", "out.json",
            "Big task",
        ])
        assert args.config == "c.yaml"
        assert args.strategy == "orchestrated"
        assert args.export == "out.json"
        assert args.task == "Big task"

    def test_invalid_strategy_exits(self):
        with pytest.raises(SystemExit):
            parse_args(["--strategy", "invalid", "Task"])


# ---- _load_strategy ----------------------------------------------------------

class TestLoadStrategy:
    def test_load_sequential(self):
        s = _load_strategy("sequential")
        assert isinstance(s, SequentialStrategy)

    def test_load_graph_routed(self):
        s = _load_strategy("graph_routed")
        assert isinstance(s, GraphRoutedStrategy)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            _load_strategy("nonexistent")

    def test_all_strategies_registered(self):
        assert set(_STRATEGY_MAP.keys()) == {"sequential", "graph_routed", "orchestrated", "networked"}


# ---- main() with mocks -------------------------------------------------------

class _MockAgent:
    """Minimal mock agent with a run() method."""

    def __init__(self, name: str, response: str):
        self.name = name
        self._response = response
        self._call_count = 0

    def run(self, task: str) -> str:
        self._call_count += 1
        return f"{self._response} [call {self._call_count}]"


class TestMainCLI:
    @patch("main.Coordinator.from_config")
    @patch("main.load_config")
    def test_main_runs_and_returns_zero(self, mock_load_config, mock_from_config, capsys):
        from src.config.loader import AppConfig
        from src.coordination.strategy import CoordinationResult

        mock_load_config.return_value = AppConfig()

        mock_coordinator = MagicMock()
        mock_coordinator.run.return_value = CoordinationResult(
            final_output="The answer is 42",
            history=[],
            metrics={"total_messages": 1},
        )
        mock_from_config.return_value = mock_coordinator

        result = main(["Test task"])
        assert result == 0

        captured = capsys.readouterr()
        assert "Test task" in captured.out
        assert "The answer is 42" in captured.out
        assert "total_messages" in captured.out

    @patch("main.Coordinator.from_config")
    @patch("main.load_config")
    def test_main_with_strategy_override(self, mock_load_config, mock_from_config):
        from src.config.loader import AppConfig
        from src.coordination.strategy import CoordinationResult

        mock_load_config.return_value = AppConfig()
        mock_coordinator = MagicMock()
        mock_coordinator.run.return_value = CoordinationResult(
            final_output="done", history=[], metrics={},
        )
        mock_from_config.return_value = mock_coordinator

        main(["--strategy", "sequential", "Task"])

        mock_from_config.assert_called_once()
        call_kwargs = mock_from_config.call_args
        assert call_kwargs.kwargs["strategy_override"] == "sequential"

    @patch("main.Coordinator.from_config")
    @patch("main.load_config")
    def test_main_with_export(self, mock_load_config, mock_from_config, tmp_path):
        from src.config.loader import AppConfig
        from src.coordination.strategy import CoordinationResult

        mock_load_config.return_value = AppConfig()
        mock_coordinator = MagicMock()
        mock_coordinator.run.return_value = CoordinationResult(
            final_output="exported", history=[], metrics={"total_messages": 0},
        )
        mock_from_config.return_value = mock_coordinator

        export_path = str(tmp_path / "result.json")
        result = main(["--export", export_path, "Export task"])
        assert result == 0

        with open(export_path) as f:
            data = json.load(f)
        assert "history" in data
        assert "metrics" in data
