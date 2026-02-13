# Training Guide

This guide covers how to configure and run RL training with DeepGen-RL.

## Overview

DeepGen-RL uses **MR-GRPO** to train DeepGen 1.0. The training pipeline consists of:

1. **Reward services** -- External API services that score generated images
2. **Training script** -- The main `scripts/train.sh` that launches distributed GRPO training

## 1. Start Reward Services

Before training, start the required reward services. Each service runs as an independent HTTP API.

### OCR Reward Service

Evaluates text rendering quality in generated images.

```bash
# In a separate terminal or screen session:
conda activate deepgen_rl_ocr
bash scripts/reward_ocr.sh
```

The service will be available at `http://localhost:18082` by default.

### UnifiedReward Service

Provides a general-purpose reward signal using the UnifiedReward-Think model via vLLM.
The service exposes an OpenAI-compatible API directly (no separate wrapper needed).

```bash
# In a separate terminal or screen session:
conda activate deepgen_rl_reward
bash scripts/reward_unifiedreward.sh

# Optional: customize parallelism and GPU usage
VLLM_PARALLEL_MODE=dp VLLM_DATA_PARALLEL_SIZE=8 bash scripts/reward_unifiedreward.sh
```

The service will be available at `http://localhost:18087` by default.
Set `UNIFIEDREWARD_MODEL` to override the model path (default: `CodeGoat24/UnifiedReward-Think-qwen3vl-8b`).

### UniGenBench Evaluation Model

Deploys the UniGenBench evaluation model via vLLM for periodic evaluation during training.

```bash
# In a separate terminal or screen session:
bash scripts/eval_unigenbench_evalmodel.sh
```

The service will be available at `http://localhost:8000` by default.

## 2. Configure Model Paths

Set the following environment variables to point to your model weights:

```bash
export SD3_5_MODEL_NAME_OR_PATH="/path/to/UniPic2-SD3.5M-Kontext-2B"
export QWEN2_5_VL_MODEL_NAME_OR_PATH="/path/to/Qwen2.5-VL-3B-Instruct"
export CLIP_MODEL_NAME_OR_PATH="/path/to/clip-vit-large-patch14"
```

You also need a pretrained checkpoint from DeepGen\_Image pretraining:

```bash
export CHECKPOINT="/path/to/pretrained_checkpoint.pth"
```

## 3. Configure Reward Service URLs

If reward services are running on a different machine, set the URLs:

```bash
export OCR_URL="http://<reward-host>:18082"
export UNIFIEDREWARD_THINK_URL="http://<reward-host>:18087"
export UNIGENBENCH_API_URL="http://<eval-host>:8000"
```

These default to `http://localhost:<port>` if not set.

## 4. Run Training

```bash
conda activate deepgen_rl
bash scripts/train.sh
```

### Custom Output Directory

```bash
export OUTPUT_DIR="/path/to/output"
bash scripts/train.sh
```

### Multi-Node Training

For multi-node distributed training, set the following on each node:

```bash
export NNODES=4                    # Total number of nodes
export NODE_RANK=0                 # Current node rank (0, 1, 2, ...)
export MASTER_ADDR=10.0.0.1        # Master node IP address
export MASTER_PORT=29700           # Master node port
export NGPUS=8                     # GPUs per node

bash scripts/train.sh
```

## 5. Training Parameters

The training script uses sensible defaults for most parameters (defined in `deepgen_rl/grpo_deepgen.py`). Only a few parameters are explicitly set in `scripts/train.sh`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--num_train_epochs` | 10 | Number of training epochs |
| `--lr_scheduler_type` | `constant_with_warmup` | Learning rate schedule |
| `--deepspeed` | `scripts/deepspeed/zero2.json` | DeepSpeed config |
| `--report_to` | `wandb,swanlab` | Logging backends |

### Key Default Parameters

These are already set to recommended values by default and do not need to be specified unless you want to change them:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning_rate` | 2e-6 | Learning rate |
| `--rollout_n` | 8 | Images per prompt for GRPO |
| `--rollout_micro_batch_size` | 32 | Images per GPU per rollout iteration |
| `--atrain_micro_batch_size` | 32 | Samples per GPU per training step |
| `--atrain_sde_sampler` | `cps_sde` | SDE sampler for training |
| `--atrain_adv_type` | `gdpo` | Advantage computation type |
| `--beta` | 5e-7 | KL penalty coefficient |
| `--atrain_kl_type` | `v-based` | KL divergence type |
| `--num_inference_steps` | 50 | Diffusion inference steps |
| `--image_height` | 512 | Generated image height |
| `--image_width` | 512 | Generated image width |
| `--timestep_fraction` | 0.6 | Fraction of timesteps for training |
| `--clip_range` | 1e-4 | PPO-style clip range |
| `--sftaux_coef` | 0.0001 | SFT auxiliary loss coefficient |
| `--eval_freq` | 10 | Evaluation frequency (steps) |
| `--gradient_checkpointing` | enabled | Memory optimization |

To override any default, add the parameter to the `torchrun` command in `scripts/train.sh`.

### DeepSpeed Configuration

Four DeepSpeed configurations are provided in `scripts/deepspeed/`:

| Config | Description |
|--------|-------------|
| `zero2.json` | ZeRO Stage 2 (recommended, default) |
| `zero3.json` | ZeRO Stage 3 |
| `zero3_offload.json` | ZeRO Stage 3 with CPU offloading (lower VRAM) |
| `zero3_sd3.json` | ZeRO Stage 3 optimized for SD3 models |

### Logging

Training supports both Weights & Biases and SwanLab for experiment tracking. Configure via environment variables:

```bash
# Weights & Biases
export WANDB_API_KEY="your-key"
export WANDB_PROJECT="DeepGen-RL"
export WANDB_MODE="online"   # or "offline"

# SwanLab
export SWANLAB_MODE="online"  # or "offline", "disabled"
export SWANLAB_PROJ_NAME="DeepGen-RL"
```

By default, both are set to `offline` mode.

## 6. Checkpoint Conversion

To convert an RL checkpoint to SFT format (e.g., for inference with the base model):

```bash
python scripts/utils/rlckpt_to_sftckpt.py \
    --rl_checkpoint /path/to/checkpoint-XX \
    --output /path/to/output.pth
```

The converted checkpoint can be loaded with:

```python
from xtuner.model.utils import guess_load_checkpoint
state_dict = guess_load_checkpoint("output.pth")
model.load_state_dict(state_dict, strict=False)
```

## 7. Dataset Configuration

Training and evaluation datasets are configured via YAML files:

- **Training**: `assets/rl_datasets/deepgen/deepgen_train.yaml` -- Defines training datasets and their associated reward functions
- **Evaluation**: `assets/rl_datasets/deepgen/eval/eval.yaml` -- Defines evaluation datasets
- **SFT auxiliary**: `deepgen_rl/sft/configs/datasets/deepgen/t2i_grpo_moretextdata.py` -- MMEngine config for auxiliary supervised loss

Override these paths via environment variables:

```bash
export DATASET_CONFIG="path/to/custom_train.yaml"
export EVAL_DATASET_CONFIG="path/to/custom_eval.yaml"
export SFTAUX_DATASET_CONFIG="path/to/custom_sftaux.py"
```
