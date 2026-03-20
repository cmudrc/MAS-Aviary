"""Tests for the LLM model loader.

Tests marked @pytest.mark.slow require GPU and model download.
Run with: pytest -m slow tests/test_model_loader.py
"""

from unittest.mock import MagicMock, patch

import pytest

from src.config.loader import LLMConfig
from src.llm.model_loader import load_model


def test_load_model_returns_thinking_model():
    """Smoke test: load a tiny model to verify the pipeline works."""
    config = LLMConfig(
        model_id="HuggingFaceTB/SmolLM-135M-Instruct",
        device_map="cpu",
        torch_dtype="float32",
        max_new_tokens=32,
    )
    model = load_model(config)
    from smolagents import TransformersModel

    from src.llm.thinking_model import ThinkingModel
    assert isinstance(model, ThinkingModel)
    assert isinstance(model, TransformersModel)  # still a TransformersModel subclass


def test_load_model_vllm_backend():
    """vLLM backend creates an OpenAIServerModel."""
    config = LLMConfig(
        model_id="Qwen/Qwen3-8B",
        backend="vllm",
        api_base="http://localhost:8000/v1",
        api_key="test-key",
        max_new_tokens=512,
        temperature=0.5,
    )
    with patch("smolagents.OpenAIServerModel") as MockModel:
        mock_instance = MagicMock()
        MockModel.return_value = mock_instance
        model = load_model(config)
        MockModel.assert_called_once_with(
            model_id="Qwen/Qwen3-8B",
            api_base="http://localhost:8000/v1",
            api_key="test-key",
            temperature=0.5,
            max_tokens=512,
        )
        assert model is mock_instance


def test_load_model_vllm_empty_api_key_defaults_to_EMPTY():
    """When api_key is empty, vLLM backend passes 'EMPTY'."""
    config = LLMConfig(
        model_id="Qwen/Qwen3-8B",
        backend="vllm",
        api_key="",
    )
    with patch("smolagents.OpenAIServerModel") as MockModel:
        MockModel.return_value = MagicMock()
        load_model(config)
        call_kwargs = MockModel.call_args[1]
        assert call_kwargs["api_key"] == "EMPTY"


def test_load_model_default_backend_is_transformers():
    """Without explicit backend, defaults to transformers."""
    config = LLMConfig(
        model_id="HuggingFaceTB/SmolLM-135M-Instruct",
        device_map="cpu",
        torch_dtype="float32",
        max_new_tokens=32,
    )
    # backend defaults to "transformers"
    assert config.backend == "transformers"
    model = load_model(config)
    from src.llm.thinking_model import ThinkingModel
    assert isinstance(model, ThinkingModel)


@pytest.mark.slow
def test_load_gpt_oss_20b():
    """Load gpt-oss-20b on GPU and verify it produces a response.

    Requires: dual 5090 GPUs, ~41 GB VRAM, model download from HuggingFace.
    """
    config = LLMConfig(
        model_id="openai/gpt-oss-20b",
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=64,
    )
    model = load_model(config)
    from smolagents import TransformersModel
    assert isinstance(model, TransformersModel)
