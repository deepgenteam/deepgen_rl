#!/bin/bash
# DeepGen-RL Training Script
#
# This script trains DeepGen models using GRPO (Group Relative Policy Optimization)
# with multiple reward functions and multi-dataset support.
#
# Before running:
#   1. Set up the environment (see docs/installation.md)
#   2. Start reward services (see scripts/reward_ocr.sh, scripts/reward_unifiedreward.sh)
#   3. Configure model paths and reward service URLs below
#
# Usage:
#   bash scripts/train.sh

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ========================================
# Model Paths (override via environment variables)
# ========================================
export SD3_5_MODEL_NAME_OR_PATH="${SD3_5_MODEL_NAME_OR_PATH:-/path/to/UniPic2-SD3.5M-Kontext-2B}"
export QWEN2_5_VL_MODEL_NAME_OR_PATH="${QWEN2_5_VL_MODEL_NAME_OR_PATH:-/path/to/Qwen2.5-VL-3B-Instruct}"
export CLIP_MODEL_NAME_OR_PATH="${CLIP_MODEL_NAME_OR_PATH:-/path/to/clip-vit-large-patch14}"

# ========================================
# Dataset Configuration
# ========================================

# Training dataset configuration (YAML file defining datasets and their reward functions)
DATASET_CONFIG="${DATASET_CONFIG:-assets/rl_datasets/deepgen/deepgen_train.yaml}"

# Evaluation dataset configuration
EVAL_DATASET_CONFIG="${EVAL_DATASET_CONFIG:-assets/rl_datasets/deepgen/eval/eval.yaml}"

# SFT auxiliary dataset config (mmengine .py format)
# Used to compute an auxiliary supervised diffusion loss during GRPO training.
SFTAUX_DATASET_CONFIG="${SFTAUX_DATASET_CONFIG:-deepgen_rl/sft/configs/datasets/deepgen/t2i_grpo_moretextdata.py}"

# Model config file path (relative to project root)
MODEL_CONFIG="${MODEL_CONFIG:-configs/models/qwen2_5_vl_7b_stable_diffusion_3_5_medium_hf_dynamic_dpo_fusion.py}"

# Pretrained checkpoint path
# Update this to your actual checkpoint from DeepGen_Image pretraining.
CHECKPOINT="${CHECKPOINT:-/path/to/pretrained_checkpoint.pth}"

# ========================================
# Reward Service Configuration
# ========================================
# Start reward services before training:
#   bash scripts/reward_ocr.sh
#   bash scripts/reward_unifiedreward.sh
#   bash scripts/eval_unigenbench_evalmodel.sh

export OCR_URL="${OCR_URL:-http://localhost:18082}"
export UNIFIEDREWARD_THINK_URL="${UNIFIEDREWARD_THINK_URL:-http://localhost:18087}"
export UNIFIEDREWARD_THINK_WORKERS="${UNIFIEDREWARD_THINK_WORKERS:-256}"
export UNIGENBENCH_API_URL="${UNIGENBENCH_API_URL:-http://localhost:8000}"
export UNIGENBENCH_MODEL_NAME="${UNIGENBENCH_MODEL_NAME:-UniGenBench-EvalModel-qwen3vl-32b-v1}"
export UNIGENBENCH_WORKERS="${UNIGENBENCH_WORKERS:-256}"

# ========================================
# Output Directory
# ========================================
OUTPUT_DIR="${OUTPUT_DIR:-outputs/deepgen_rl_$(date +%Y%m%d_%H%M)}"
mkdir -p "${OUTPUT_DIR}"

# ========================================
# Find Project Root
# ========================================
code_dir=$(dirname "$(realpath "$0")")
while [[ ! -d "${code_dir}/scripts" && "${code_dir}" != "/" ]]; do
  code_dir=$(dirname "${code_dir}")
done
export PYTHONPATH="${code_dir}:${PYTHONPATH:-}"

# ========================================
# Distributed Training Configuration
# ========================================
ALL_NNODES=$(echo "${NODE_IP_LIST:-}" 2>/dev/null | sed 's/,/\n/g' | wc -l)
export NNODES=${NNODES:-${ALL_NNODES:-1}}
export NODE_RANK=${NODE_RANK:-${INDEX:-0}}
export MASTER_ADDR=${MASTER_ADDR:-${CHIEF_IP:-localhost}}
export MASTER_PORT=${MASTER_PORT:-29700}
export MAX_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)
export NGPUS=${NGPUS:-${MAX_GPUS}}
export NUM_PROCESSES=$((NGPUS * NNODES))

echo "Distributed Config: NNODES=${NNODES}, NGPUS=${NGPUS}, NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

# ========================================
# Logging Configuration (Wandb + SwanLab)
# ========================================
SCRIPT_NAME=$(basename "$(realpath "$0")" .sh)
EXP_DATE=$(date +%Y%m%d_%H%M)

export WANDB_PROJECT="${WANDB_PROJECT:-DeepGen-RL}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${SCRIPT_NAME}_${EXP_DATE}}"
export WANDB_NAME="${WANDB_RUN_NAME}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RUN_NAME="${WANDB_RUN_NAME}"

export SWANLAB_MODE="${SWANLAB_MODE:-offline}"
export SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-DeepGen-RL}"
export SWANLAB_EXP_NAME="${SCRIPT_NAME}_${EXP_DATE}"

# Redirect output to log file
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
exec &> >(tee -a "${LOG_DIR}/${NODE_RANK}.log")

# ========================================
# Copy configs to output directory for reproducibility
# ========================================
cp "${MODEL_CONFIG}" "${OUTPUT_DIR}/" 2>/dev/null || true
[[ -n "${DATASET_CONFIG}" ]] && cp "${DATASET_CONFIG}" "${OUTPUT_DIR}/" 2>/dev/null || true
[[ -n "${SFTAUX_DATASET_CONFIG}" ]] && cp "${SFTAUX_DATASET_CONFIG}" "${OUTPUT_DIR}/" 2>/dev/null || true
cp "$(realpath "$0")" "${OUTPUT_DIR}/" 2>/dev/null || true

# ========================================
# Print Configuration Summary
# ========================================
echo "=========================================="
echo "DeepGen-RL GRPO Training"
echo "=========================================="
echo "Model Config:     ${MODEL_CONFIG}"
echo "Checkpoint:       ${CHECKPOINT}"
echo "Dataset Config:   ${DATASET_CONFIG}"
echo "Eval Config:      ${EVAL_DATASET_CONFIG}"
echo "SFT-Aux Config:   ${SFTAUX_DATASET_CONFIG}"
echo "Output Dir:       ${OUTPUT_DIR}"
echo "OCR URL:          ${OCR_URL}"
echo "UnifiedReward:    ${UNIFIEDREWARD_THINK_URL}"
echo "UniGenBench:      ${UNIGENBENCH_API_URL}"
echo "=========================================="

# ========================================
# Run Training
# ========================================
# Most training hyperparameters use sensible defaults defined in deepgen_rl/grpo_deepgen.py.
# Only parameters that differ from defaults are specified here.
# See docs/training.md for a full list of configurable parameters.

torchrun --nproc_per_node=${NGPUS} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m deepgen_rl.grpo_deepgen \
    --model_config "${MODEL_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset_config "${DATASET_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 10 \
    --lr_scheduler_type constant_with_warmup \
    --report_to wandb,swanlab \
    --run_name "${RUN_NAME}" \
    --deepspeed scripts/deepspeed/zero2.json \
    --sftaux_dataset_config "${SFTAUX_DATASET_CONFIG}" \
    --eval_dataset_config "${EVAL_DATASET_CONFIG}"

echo "Training completed!"
echo "Output saved to: ${OUTPUT_DIR}"
