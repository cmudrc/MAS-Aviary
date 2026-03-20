#!/usr/bin/env bash
# Start vLLM server for Qwen3-8B on dual RTX 5090s.
#
# Flags:
#   --tensor-parallel-size 2   — split across both GPUs
#   --enable-auto-tool-choice  — structured tool_calls in responses
#   --tool-call-parser hermes  — Hermes-style tool format (Qwen3 compatible)
#   --max-model-len 8192       — context length cap
#   --enforce-eager             — skip CUDA graph capture (needed for sm_120 / Blackwell)
#
# Usage:
#   bash scripts/start_vllm.sh
#   # Then in another terminal:
#   python -m src.runners.batch_runner --config config/default_mcp.yaml ...

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
TP="${VLLM_TP:-2}"
PORT="${VLLM_PORT:-8000}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

exec vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len "$MAX_LEN" \
    --enforce-eager \
    --host 0.0.0.0 \
    --port "$PORT"
