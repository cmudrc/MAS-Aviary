"""Shared test fixtures for MAS-Aviary test suite.

Provides reusable mocks and helpers so individual test files
don't need to duplicate model stubs, tool lists, or config builders.
"""

import json

import pytest
import yaml
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    Model,
)

# ---------------------------------------------------------------------------
# 1. mock_model – a FinalAnswerModel that immediately returns final_answer
# ---------------------------------------------------------------------------

class FinalAnswerModel(Model):
    """Model that immediately returns final_answer with configurable output.

    Matches the pattern used across test_coordinator, test_integration, etc.
    Each call increments a counter appended to the answer.
    """

    def __init__(self, answer: str = "dummy output"):
        super().__init__(model_id="final-answer-model")
        self._answer = answer
        self._call_count = 0

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs):
        self._call_count += 1
        answer = f"{self._answer} [call {self._call_count}]"
        tc = ChatMessageToolCall(
            id=f"call_{self._call_count}",
            type="function",
            function=ChatMessageToolCallFunction(
                name="final_answer",
                arguments=json.dumps({"answer": answer}),
            ),
        )
        return ChatMessage(role="assistant", content="", tool_calls=[tc])


@pytest.fixture
def mock_model():
    """Return a FinalAnswerModel that resolves in one step (no GPU needed)."""
    return FinalAnswerModel()


# ---------------------------------------------------------------------------
# 2. mock_mcp_tools – list of mock Aviary MCP tools
# ---------------------------------------------------------------------------

class _MockMCPTool:
    """Lightweight stand-in for a smolagents MCP tool.

    Has the same surface used by agents: name, description, and __call__.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description or f"Mock tool: {name}"
        self.inputs = {}
        self.output_type = "string"
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return f"{self.name} result"


_AVIARY_TOOL_SPECS = [
    ("create_session", "Create a new Aviary analysis session"),
    ("configure_mission", "Configure mission parameters (range, payload, etc.)"),
    ("get_design_space", "Retrieve the current design-space definition"),
    ("set_aircraft_parameters", "Set aircraft design parameters"),
    ("validate_parameters", "Validate current parameter set for consistency"),
    ("run_simulation", "Run the Aviary simulation with current parameters"),
    ("get_results", "Retrieve simulation result summary"),
    ("get_trajectory", "Retrieve trajectory data from a completed run"),
    ("check_constraints", "Check constraint violations on the latest run"),
]


@pytest.fixture
def mock_mcp_tools():
    """Return a list of mock MCP tools matching the Aviary MCP server surface."""
    return [_MockMCPTool(name, desc) for name, desc in _AVIARY_TOOL_SPECS]


# ---------------------------------------------------------------------------
# 3. tmp_config – creates temp YAML config files for testing
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config(tmp_path):
    """Factory fixture: call with optional overrides to get a temp YAML config path.

    Usage in tests::

        def test_something(tmp_config):
            path = tmp_config()                      # defaults
            path = tmp_config(llm={"model_id": "X"}) # override llm section
    """

    def _make(**overrides):
        base = {
            "llm": {
                "model_id": "Qwen/Qwen3-8B",
                "backend": "transformers",
                "device_map": "balanced",
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "reasoning_effort": "medium",
            },
            "mcp": {
                "mode": "mock",
                "servers": [
                    {
                        "url": "http://127.0.0.1:8200/mcp",
                        "transport": "streamable-http",
                    },
                ],
            },
            "logging": {
                "level": "INFO",
                "output_dir": str(tmp_path / "logs"),
                "save_full_history": True,
            },
            "ui": {
                "enabled": False,
            },
        }
        # Merge top-level overrides
        for key, val in overrides.items():
            if isinstance(val, dict) and key in base and isinstance(base[key], dict):
                base[key].update(val)
            else:
                base[key] = val

        path = tmp_path / "test_config.yaml"
        path.write_text(yaml.dump(base, default_flow_style=False))
        return str(path)

    return _make


# ---------------------------------------------------------------------------
# 4. sample_result – representative Aviary simulation result dict
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_result():
    """Return a sample Aviary result dict with typical output values."""
    return {
        "fuel_burn": 12345.6,       # lbm
        "gtow": 175000.0,           # lbm  (gross takeoff weight)
        "wing_mass": 18500.0,       # lbm
        "range": 3500.0,            # nmi
        "payload": 36000.0,         # lbm
        "converged": True,
        "constraints_satisfied": True,
        "objective": 12345.6,
    }
