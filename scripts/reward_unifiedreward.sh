#!/bin/bash
# DeepGen-RL: UnifiedReward Service Launcher
#
# Starts the UnifiedReward scoring service used as a reward signal during RL training.
# The service runs a vLLM server with the UnifiedReward-Think model and exposes
# an OpenAI-compatible API directly.
#
# Prerequisites:
#   conda create -n deepgen_rl_reward python=3.12 -y
#   conda activate deepgen_rl_reward
#   pip install vllm  # follow the vLLM installation guide
#
# Usage:
#   bash scripts/reward_unifiedreward.sh
#
# Environment variables:
#   UNIFIEDREWARD_PORT       - Port for the vLLM API (default: 18087)
#   UNIFIEDREWARD_MODEL      - HuggingFace model name or local path
#                              (default: CodeGoat24/UnifiedReward-Think-qwen3vl-8b)
#   VLLM_PARALLEL_MODE       - Parallelism mode: "dp" or "tp" (default: dp)
#   VLLM_DATA_PARALLEL_SIZE  - Data parallel size when mode=dp (default: 8)
#   VLLM_TENSOR_PARALLEL_SIZE - Tensor parallel size when mode=tp (default: same as dp)
#   VLLM_GPU_MEMORY_UTILIZATION - GPU memory fraction (default: 0.9)

set -euo pipefail

UNIFIEDREWARD_PORT="${UNIFIEDREWARD_PORT:-18087}"
UNIFIEDREWARD_MODEL="${UNIFIEDREWARD_MODEL:-CodeGoat24/UnifiedReward-Think-qwen3vl-8b}"

# Do NOT inherit training PYTHONPATH (it may contain another conda env's site-packages).
# This prevents mixing Python 3.12 (vLLM) with Python 3.11 packages (e.g., flash_attn).
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

# vLLM uses a CUDA memory pool which is not compatible with PyTorch expandable segments.
unset PYTORCH_CUDA_ALLOC_CONF

# Enable vLLM development endpoints required by Sleep Mode (/sleep, /wake_up).
export VLLM_SERVER_DEV_MODE=1

# Configurable parameters via environment variables
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
PARALLEL_MODE="${UNIFIEDREWARD_VLLM_PARALLEL_MODE:-${VLLM_PARALLEL_MODE:-dp}}"  # dp | tp
DP_SIZE="${UNIFIEDREWARD_VLLM_DATA_PARALLEL_SIZE:-${VLLM_DATA_PARALLEL_SIZE:-8}}"
TP_SIZE="${UNIFIEDREWARD_VLLM_TENSOR_PARALLEL_SIZE:-${VLLM_TENSOR_PARALLEL_SIZE:-${DP_SIZE}}}"

if [[ "${PARALLEL_MODE}" == "tp" ]]; then
    PARALLEL_ARGS=(--tensor-parallel-size "${TP_SIZE}")
    PARALLEL_DESC="tp_size=${TP_SIZE}"
else
    PARALLEL_ARGS=(--data-parallel-size "${DP_SIZE}")
    PARALLEL_DESC="dp_size=${DP_SIZE}"
fi

echo "=========================================="
echo "Starting UnifiedReward Service (vLLM)"
echo "=========================================="
echo "  Model:          ${UNIFIEDREWARD_MODEL}"
echo "  API Port:       ${UNIFIEDREWARD_PORT}"
echo "  Parallel Mode:  ${PARALLEL_MODE} (${PARALLEL_DESC})"
echo "  GPU Util:       ${GPU_UTIL}"
echo "=========================================="

set -x

exec vllm serve "${UNIFIEDREWARD_MODEL}" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --served-model-name UnifiedReward \
    --gpu-memory-utilization "${GPU_UTIL}" \
    "${PARALLEL_ARGS[@]}" \
    --pipeline-parallel-size 1 \
    --limit-mm-per-prompt.image 16 \
    --port "${UNIFIEDREWARD_PORT}" \
    --enable-sleep-mode \
    --enable-prefix-caching \
    --disable-log-requests \
    --mm_processor_cache_gb=500
