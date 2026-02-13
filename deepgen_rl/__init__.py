# Copyright 2025 Ruihang Li and DeepGen Team @ Shanghai Innovation Institute

"""
DeepGen-RL - Reinforcement Learning for DeepGen Vision-Language Models

This module provides GRPO (Group Relative Policy Optimization) trainer for
DeepGen models (Qwen2.5-VL + SD3 unified architecture).

Usage:
    # For training DeepGen models
    torchrun --nproc_per_node=8 -m deepgen_rl.grpo_deepgen \
        --model_config configs/models/qwen2_5_vl_7b_stable_diffusion_3_5_medium_hf_dynamic_dpo_fusion.py \
        --checkpoint /path/to/deepgen_checkpoint \
        --dataset_config assets/rl_datasets/deepgen/deepgen.yaml
"""

from .trainer import DeepGenGRPOTrainer

__all__ = [
    "DeepGenGRPOTrainer",
]
