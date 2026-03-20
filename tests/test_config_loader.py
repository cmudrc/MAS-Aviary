"""Tests for the configuration loader."""

import os

import pytest
import yaml

from src.config.loader import (
    AppConfig,
    LLMConfig,
    LoggingConfig,
    MCPConfig,
    UIConfig,
    load_config,
    load_yaml,
)

# ---- load_config tests -------------------------------------------------------


def test_load_default_config():
    """Loading config/default.yaml produces a fully populated AppConfig."""
    cfg = load_config("config/default.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.llm.model_id == "Qwen/Qwen3-8B"
    assert cfg.llm.backend == "transformers"
    assert cfg.llm.device_map == "balanced"
    assert cfg.llm.max_new_tokens == 1024
    assert cfg.llm.temperature == 0.7
    assert cfg.llm.reasoning_effort == "medium"


def test_load_config_mcp_section():
    """MCP section loads correctly with server list."""
    cfg = load_config("config/default.yaml")
    assert cfg.mcp.mode in ("mock", "real")
    assert len(cfg.mcp.servers) == 1
    assert cfg.mcp.servers[0].url == "http://127.0.0.1:8200/mcp"
    assert cfg.mcp.servers[0].transport == "streamable-http"


def test_load_config_logging_section():
    cfg = load_config("config/default.yaml")
    assert cfg.logging.level == "INFO"
    assert cfg.logging.output_dir == "logs/"
    assert cfg.logging.save_full_history is True


def test_load_config_ui_section():
    cfg = load_config("config/default.yaml")
    assert cfg.ui.enabled is True
    assert cfg.ui.port == 8501


def test_load_config_paths():
    cfg = load_config("config/default.yaml")
    assert cfg.agents_config == "config/agents.yaml"
    assert cfg.coordination_config == "config/coordination.yaml"


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config("config/nonexistent.yaml")


def test_load_config_empty_file(tmp_path):
    """An empty YAML file returns default AppConfig."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    cfg = load_config(str(empty))
    assert isinstance(cfg, AppConfig)
    assert cfg.llm.model_id == "Qwen/Qwen3-8B"


def test_load_config_partial_override(tmp_path):
    """A config that only sets llm.temperature keeps defaults for everything else."""
    partial = tmp_path / "partial.yaml"
    partial.write_text(yaml.dump({"llm": {"temperature": 0.3}}))
    cfg = load_config(str(partial))
    assert cfg.llm.temperature == 0.3
    assert cfg.llm.model_id == "Qwen/Qwen3-8B"  # default preserved
    assert cfg.mcp.mode == "mock"  # default preserved


def test_load_config_ignores_unknown_keys(tmp_path):
    """Unknown top-level or nested keys are silently ignored."""
    noisy = tmp_path / "noisy.yaml"
    noisy.write_text(
        yaml.dump(
            {
                "llm": {"model_id": "test-model", "bogus_key": 42},
                "unknown_section": {"foo": "bar"},
            }
        )
    )
    cfg = load_config(str(noisy))
    assert cfg.llm.model_id == "test-model"


# ---- load_yaml tests ---------------------------------------------------------


def test_load_agents_yaml():
    """agents.yaml loads and contains the expected agent list."""
    data = load_yaml("config/agents.yaml")
    assert "agents" in data
    agents = data["agents"]
    assert len(agents) == 3
    names = [a["name"] for a in agents]
    assert names == ["planner", "executor", "reviewer"]


def test_load_coordination_yaml():
    """coordination.yaml loads and contains strategy settings."""
    data = load_yaml("config/coordination.yaml")
    assert data["strategy"] == "graph_routed"
    assert "termination" in data
    assert data["termination"]["keyword"] == "TASK_COMPLETE"
    assert data["termination"]["max_turns"] == 20


def test_load_yaml_missing_file():
    with pytest.raises(FileNotFoundError, match="YAML file not found"):
        load_yaml("config/nonexistent.yaml")


def test_load_yaml_empty_file(tmp_path):
    """An empty YAML file returns an empty dict."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    data = load_yaml(str(empty))
    assert data == {}


# ---- vLLM backend fields tests ------------------------------------------------


def test_llmconfig_backend_defaults():
    """LLMConfig defaults to transformers backend."""
    cfg = LLMConfig()
    assert cfg.backend == "transformers"
    assert cfg.api_base == "http://localhost:8000/v1"
    assert cfg.api_key == ""


def test_load_config_vllm_backend(tmp_path):
    """A config with backend=vllm loads correctly."""
    vllm_cfg = tmp_path / "vllm.yaml"
    vllm_cfg.write_text(
        yaml.dump(
            {
                "llm": {
                    "model_id": "Qwen/Qwen3-8B",
                    "backend": "vllm",
                    "api_base": "http://myhost:9000/v1",
                    "api_key": "test-key",
                }
            }
        )
    )
    cfg = load_config(str(vllm_cfg))
    assert cfg.llm.backend == "vllm"
    assert cfg.llm.api_base == "http://myhost:9000/v1"
    assert cfg.llm.api_key == "test-key"
    assert cfg.llm.model_id == "Qwen/Qwen3-8B"


def test_load_config_backend_defaults_to_transformers(tmp_path):
    """A config without backend field defaults to transformers."""
    no_backend = tmp_path / "no_backend.yaml"
    no_backend.write_text(yaml.dump({"llm": {"model_id": "test-model", "temperature": 0.5}}))
    cfg = load_config(str(no_backend))
    assert cfg.llm.backend == "transformers"


# ---- dataclass defaults tests -------------------------------------------------


def test_appconfig_defaults():
    """AppConfig() with no arguments uses sane defaults."""
    cfg = AppConfig()
    assert isinstance(cfg.llm, LLMConfig)
    assert isinstance(cfg.mcp, MCPConfig)
    assert isinstance(cfg.logging, LoggingConfig)
    assert isinstance(cfg.ui, UIConfig)
    assert cfg.agents_config == "config/agents.yaml"


# ---- environment variable overrides -------------------------------------------


def test_env_override_mcp_url(tmp_path, monkeypatch):
    """MAS_AVIARY_MCP_URL overrides the first MCP server URL."""
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(
        yaml.dump(
            {
                "mcp": {
                    "mode": "real",
                    "servers": [{"url": "http://original:8600/mcp", "transport": "streamable-http"}],
                }
            }
        )
    )
    monkeypatch.setenv("MAS_AVIARY_MCP_URL", "http://custom-host:9999/mcp")
    cfg = load_config(str(cfg_file))
    assert cfg.mcp.servers[0].url == "http://custom-host:9999/mcp"


def test_env_override_mcp_url_creates_server(tmp_path, monkeypatch):
    """MAS_AVIARY_MCP_URL creates a server entry if none exist."""
    cfg_file = tmp_path / "empty_mcp.yaml"
    cfg_file.write_text(yaml.dump({"mcp": {"mode": "real"}}))
    monkeypatch.setenv("MAS_AVIARY_MCP_URL", "http://new-host:7000/mcp")
    cfg = load_config(str(cfg_file))
    assert len(cfg.mcp.servers) == 1
    assert cfg.mcp.servers[0].url == "http://new-host:7000/mcp"


def test_env_override_mcp_mode(tmp_path, monkeypatch):
    """MAS_AVIARY_MCP_MODE overrides mcp.mode."""
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(yaml.dump({"mcp": {"mode": "real"}}))
    monkeypatch.setenv("MAS_AVIARY_MCP_MODE", "mock")
    cfg = load_config(str(cfg_file))
    assert cfg.mcp.mode == "mock"


def test_env_override_model_id(tmp_path, monkeypatch):
    """MAS_AVIARY_MODEL_ID overrides llm.model_id."""
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(yaml.dump({"llm": {"model_id": "original-model"}}))
    monkeypatch.setenv("MAS_AVIARY_MODEL_ID", "Qwen/Qwen3-4B")
    cfg = load_config(str(cfg_file))
    assert cfg.llm.model_id == "Qwen/Qwen3-4B"


def test_env_override_api_base(tmp_path, monkeypatch):
    """MAS_AVIARY_API_BASE overrides llm.api_base."""
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(yaml.dump({"llm": {"backend": "vllm"}}))
    monkeypatch.setenv("MAS_AVIARY_API_BASE", "http://gpu-box:8080/v1")
    cfg = load_config(str(cfg_file))
    assert cfg.llm.api_base == "http://gpu-box:8080/v1"


def test_env_override_not_set(tmp_path):
    """Without env vars, YAML values are preserved as-is."""
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(
        yaml.dump(
            {
                "mcp": {
                    "mode": "real",
                    "servers": [{"url": "http://yaml-host:8600/mcp"}],
                }
            }
        )
    )
    for var in ("MAS_AVIARY_MCP_URL", "MAS_AVIARY_MCP_MODE", "MAS_AVIARY_MODEL_ID", "MAS_AVIARY_API_BASE"):
        os.environ.pop(var, None)
    cfg = load_config(str(cfg_file))
    assert cfg.mcp.servers[0].url == "http://yaml-host:8600/mcp"
