# 训练指南

本文档介绍如何配置和运行 DeepGen-RL 的强化学习训练。

## 概述

DeepGen-RL 使用 GRPO（Group Relative Policy Optimization）算法联合训练语言模型和扩散模型。训练流程包括：

1. **奖励服务** -- 对生成图像进行打分的外部 API 服务
2. **训练脚本** -- 主训练脚本 `scripts/train.sh`，启动分布式 GRPO 训练

## 1. 启动奖励服务

训练前需要启动所需的奖励服务。每个服务作为独立的 HTTP API 运行。

### OCR 奖励服务

评估生成图像中的文字渲染质量。

```bash
# 在单独的终端或 screen 会话中：
conda activate deepgen_rl_ocr
bash scripts/reward_ocr.sh
```

默认地址为 `http://localhost:18082`。

### UnifiedReward 服务

通过 vLLM 部署 UnifiedReward-Think 模型，提供通用奖励信号。
服务直接暴露 OpenAI 兼容的 API（无需额外的封装层）。

```bash
# 在单独的终端或 screen 会话中：
conda activate deepgen_rl_reward
bash scripts/reward_unifiedreward.sh

# 可选：自定义并行模式和 GPU 配置
VLLM_PARALLEL_MODE=dp VLLM_DATA_PARALLEL_SIZE=8 bash scripts/reward_unifiedreward.sh
```

默认地址为 `http://localhost:18087`。
可通过 `UNIFIEDREWARD_MODEL` 环境变量覆盖模型路径（默认：`CodeGoat24/UnifiedReward-Think-qwen3vl-8b`）。

### UniGenBench 评估模型

通过 vLLM 部署 UniGenBench 评估模型，用于训练过程中的周期性评估。

```bash
# 在单独的终端或 screen 会话中：
bash scripts/eval_unigenbench_evalmodel.sh
```

默认地址为 `http://localhost:8000`。

## 2. 配置模型路径

设置以下环境变量指向模型权重路径：

```bash
export SD3_5_MODEL_NAME_OR_PATH="/path/to/UniPic2-SD3.5M-Kontext-2B"
export QWEN2_5_VL_MODEL_NAME_OR_PATH="/path/to/Qwen2.5-VL-3B-Instruct"
export CLIP_MODEL_NAME_OR_PATH="/path/to/clip-vit-large-patch14"
```

还需要 DeepGen\_Image 预训练阶段的检查点：

```bash
export CHECKPOINT="/path/to/pretrained_checkpoint.pth"
```

## 3. 配置奖励服务地址

如果奖励服务运行在其他机器上，需要设置对应的 URL：

```bash
export OCR_URL="http://<reward-host>:18082"
export UNIFIEDREWARD_THINK_URL="http://<reward-host>:18087"
export UNIGENBENCH_API_URL="http://<eval-host>:8000"
```

如未设置，默认使用 `http://localhost:<port>`。

## 4. 运行训练

```bash
conda activate deepgen_rl
bash scripts/train.sh
```

### 自定义输出目录

```bash
export OUTPUT_DIR="/path/to/output"
bash scripts/train.sh
```

### 多节点训练

进行多节点分布式训练时，在每个节点上设置以下环境变量：

```bash
export NNODES=4                    # 总节点数
export NODE_RANK=0                 # 当前节点编号（0, 1, 2, ...）
export MASTER_ADDR=10.0.0.1        # 主节点 IP 地址
export MASTER_PORT=29700           # 主节点端口
export NGPUS=8                     # 每节点 GPU 数量

bash scripts/train.sh
```

## 5. 训练参数

训练脚本对大多数参数使用了合理的默认值（定义在 `deepgen_rl/grpo_deepgen.py` 中）。`scripts/train.sh` 中仅显式设置了少数参数：

| 参数 | 值 | 说明 |
|------|------|------|
| `--num_train_epochs` | 10 | 训练轮数 |
| `--lr_scheduler_type` | `constant_with_warmup` | 学习率调度策略 |
| `--deepspeed` | `scripts/deepspeed/zero2.json` | DeepSpeed 配置 |
| `--report_to` | `wandb,swanlab` | 日志记录后端 |

### 主要默认参数

以下参数已设置为推荐的默认值，无需额外指定（如需修改可覆盖）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--learning_rate` | 2e-6 | 学习率 |
| `--rollout_n` | 8 | 每个 prompt 的 GRPO 采样图像数 |
| `--rollout_micro_batch_size` | 32 | 每 GPU 每次 rollout 的图像数 |
| `--atrain_micro_batch_size` | 32 | 每 GPU 每次训练步的样本数 |
| `--atrain_sde_sampler` | `cps_sde` | 训练用 SDE 采样器 |
| `--atrain_adv_type` | `gdpo` | 优势计算类型 |
| `--beta` | 5e-7 | KL 惩罚系数 |
| `--atrain_kl_type` | `v-based` | KL 散度类型 |
| `--num_inference_steps` | 50 | 扩散推理步数 |
| `--image_height` | 512 | 生成图像高度 |
| `--image_width` | 512 | 生成图像宽度 |
| `--timestep_fraction` | 0.6 | 训练使用的时间步比例 |
| `--clip_range` | 1e-4 | PPO 风格裁剪范围 |
| `--sftaux_coef` | 0.0001 | SFT 辅助损失系数 |
| `--eval_freq` | 10 | 评估频率（步数） |
| `--gradient_checkpointing` | 启用 | 内存优化 |

如需覆盖默认值，在 `scripts/train.sh` 的 `torchrun` 命令中添加对应参数即可。

### DeepSpeed 配置

`scripts/deepspeed/` 目录下提供了四种 DeepSpeed 配置：

| 配置文件 | 说明 |
|----------|------|
| `zero2.json` | ZeRO Stage 2（推荐，默认） |
| `zero3.json` | ZeRO Stage 3 |
| `zero3_offload.json` | ZeRO Stage 3 + CPU 卸载（降低显存需求） |
| `zero3_sd3.json` | ZeRO Stage 3 针对 SD3 模型优化 |

### 日志记录

训练支持 Weights & Biases 和 SwanLab 进行实验追踪。通过环境变量配置：

```bash
# Weights & Biases
export WANDB_API_KEY="your-key"
export WANDB_PROJECT="DeepGen-RL"
export WANDB_MODE="online"   # 或 "offline"

# SwanLab
export SWANLAB_MODE="online"  # 或 "offline"、"disabled"
export SWANLAB_PROJ_NAME="DeepGen-RL"
```

默认情况下，两者均为 `offline` 模式。

## 6. 检查点转换

将 RL 检查点转换为 SFT 格式（例如用于基础模型推理）：

```bash
python scripts/utils/rlckpt_to_sftckpt.py \
    --rl_checkpoint /path/to/checkpoint-XX \
    --output /path/to/output.pth
```

转换后的检查点可通过以下方式加载：

```python
from xtuner.model.utils import guess_load_checkpoint
state_dict = guess_load_checkpoint("output.pth")
model.load_state_dict(state_dict, strict=False)
```

## 7. 数据集配置

训练和评估数据集通过 YAML 文件配置：

- **训练数据集**：`assets/rl_datasets/deepgen/deepgen_train.yaml` -- 定义训练数据集及其关联的奖励函数
- **评估数据集**：`assets/rl_datasets/deepgen/eval/eval.yaml` -- 定义评估数据集
- **SFT 辅助数据集**：`deepgen_rl/sft/configs/datasets/deepgen/t2i_grpo_moretextdata.py` -- MMEngine 配置，用于辅助监督损失

通过环境变量覆盖这些路径：

```bash
export DATASET_CONFIG="path/to/custom_train.yaml"
export EVAL_DATASET_CONFIG="path/to/custom_eval.yaml"
export SFTAUX_DATASET_CONFIG="path/to/custom_sftaux.py"
```
