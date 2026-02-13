#!/bin/bash
# DeepGen-RL: UniGenBench Evaluation Model Launcher
#
# Deploys the UniGenBench-EvalModel via vLLM for evaluation during RL training.
# The model is used to evaluate the quality of generated images against benchmarks.
#
# Prerequisites:
#   pip install vllm
#
# Usage:
#   bash scripts/eval_unigenbench_evalmodel.sh
#
# Environment variables:
#   UNIGENBENCH_PORT       - Port for the vLLM server (default: 8000)
#   UNIGENBENCH_MODEL      - HuggingFace model name or local path
#                            (default: CodeGoat24/UniGenBench-EvalModel-qwen3vl-32b-v1)
#   UNIGENBENCH_MODEL_NAME - Served model name (default: UniGenBench-EvalModel-qwen3vl-32b-v1)
#   DP_SIZE                - Data parallel size (default: 8)
#   GPU_MEMORY_UTILIZATION - GPU memory utilization fraction (default: 0.8)

set -euo pipefail

UNIGENBENCH_PORT="${UNIGENBENCH_PORT:-8000}"
UNIGENBENCH_MODEL="${UNIGENBENCH_MODEL:-CodeGoat24/UniGenBench-EvalModel-qwen3vl-32b-v1}"
UNIGENBENCH_MODEL_NAME="${UNIGENBENCH_MODEL_NAME:-UniGenBench-EvalModel-qwen3vl-32b-v1}"
DP_SIZE="${DP_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"

echo "=========================================="
echo "Starting UniGenBench Evaluation Model"
echo "=========================================="
echo "  Model:       ${UNIGENBENCH_MODEL}"
echo "  Served As:   ${UNIGENBENCH_MODEL_NAME}"
echo "  Port:        ${UNIGENBENCH_PORT}"
echo "  DP Size:     ${DP_SIZE}"
echo "  GPU Mem:     ${GPU_MEMORY_UTILIZATION}"
echo "=========================================="

# Launch vLLM server
exec vllm serve "${UNIGENBENCH_MODEL}" \
    --port "${UNIGENBENCH_PORT}" \
    --served-model-name "${UNIGENBENCH_MODEL_NAME}" \
    --data-parallel-size "${DP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
