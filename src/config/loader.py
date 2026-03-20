"""Configuration loader — reads YAML files and returns structured config."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    model_id: str = "Qwen/Qwen3-8B"
    backend: str = "transformers"  # "transformers" or "vllm"
    api_base: str = "http://localhost:8000/v1"
    api_key: str = ""
    device_map: str = "balanced"
    torch_dtype: str = "auto"
    max_new_tokens: int = 1024
    temperature: float = 0.7
    reasoning_effort: str = "medium"
    reliability: dict = field(default_factory=dict)


@dataclass
class MCPServerConfig:
    url: str = ""
    transport: str = "streamable-http"


@dataclass
class MCPConfig:
    mode: str = "mock"
    servers: list[MCPServerConfig] = field(default_factory=list)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    output_dir: str = "logs/"
    save_full_history: bool = True


@dataclass
class UIConfig:
    enabled: bool = True
    port: int = 8501


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    agents_config: str = "config/agents.yaml"
    coordination_config: str = "config/coordination.yaml"
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UIConfig = field(default_factory=UIConfig)


# Map of field names to their nested dataclass types for recursive parsing.
_NESTED_TYPES = {
    "llm": LLMConfig,
    "mcp": MCPConfig,
    "logging": LoggingConfig,
    "ui": UIConfig,
}


def _dict_to_dataclass(cls, data: dict[str, Any]):
    """Recursively convert a dict to a dataclass, ignoring unknown keys."""
    if data is None:
        return cls()
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {}
    for k, v in data.items():
        if k not in field_names:
            continue
        if isinstance(v, dict) and k in _NESTED_TYPES:
            filtered[k] = _dict_to_dataclass(_NESTED_TYPES[k], v)
        elif isinstance(v, list) and k == "servers":
            filtered[k] = [
                _dict_to_dataclass(MCPServerConfig, item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            filtered[k] = v
    return cls(**filtered)


def load_config(path: str | Path) -> AppConfig:
    """Load configuration from a YAML file and return an AppConfig dataclass."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return AppConfig()

    return _dict_to_dataclass(AppConfig, raw)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a raw YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return data if data is not None else {}
