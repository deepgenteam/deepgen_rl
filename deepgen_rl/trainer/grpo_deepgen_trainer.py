# Copyright 2025 Ruihang Li and DeepGen Team @ Shanghai Innovation Institute

"""
GRPO Trainer for DeepGen Models (Qwen2p5VLStableDiffusion3HF)

This module implements Group Relative Policy Optimization (GRPO) for DeepGen
vision-language models, supporting text-to-image generation tasks.
"""

import gc
import os
import sys
import math
import time
import json
import re
import random
from collections.abc import Iterable
from contextlib import contextmanager
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import torch.distributed as dist
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from packaging import version
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
    is_tensorboard_available,
)

# SwanLab support
def is_swanlab_available():
    """Check if swanlab is available."""
    try:
        import swanlab
        return True
    except ImportError:
        return False
import transformers
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from deepspeed.runtime.zero import GatheredParameters
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from einops import rearrange

from trl.models import prepare_deepspeed
from trl.trainer.grpo_config import GRPOConfig
from tqdm import tqdm
from PIL import Image

# Phase logger for training process monitoring
from deepgen_rl.trainer.phase_logger import PhaseLogger, set_phase_logger
# Evaluation dataset utilities
from deepgen_rl.utils.eval_dataset import EvalDatasetConfig, EvalPromptDataset
# UniGenBench evaluation
from deepgen_rl.evaluation.unigenbench import (
    UniGenBenchScorer,
    is_unigenbench_enabled,
)
from deepgen_rl.utils.vllm_sleep_mode import is_exclusive_vllm_active, maybe_switch_vllm_server


def is_rank_zero() -> bool:
    """Check if current process is global rank 0."""
    try:
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True
    except Exception:
        return True


def rank0_print(*args, **kwargs):
    """Print only on global rank 0."""
    if is_rank_zero():
        print(*args, **kwargs)


class EMAModuleWrapper:
    """
    Exponential Moving Average (EMA) wrapper for a list of parameters.

    This is adapted from `temp/repo/flow_grpo/flow_grpo/ema.py` with minimal changes.
    It keeps a detached EMA copy of parameters (optionally on CPU to save GPU memory),
    and supports swapping EMA weights into the model temporarily.
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        update_step_interval: int = 1,
        device: torch.device | None = None,
    ):
        parameters = list(parameters)
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters]

        self.temp_stored_parameters = None

        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device

    def get_current_decay(self, optimization_step: int) -> float:
        # Warm up EMA at the beginning to avoid overly stale averages.
        return min((1 + optimization_step) / (10 + optimization_step), self.decay)

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], optimization_step: int):
        parameters = list(parameters)
        one_minus_decay = 1 - self.get_current_decay(optimization_step)

        if (optimization_step + 1) % self.update_step_interval == 0:
            for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
                if parameter.requires_grad:
                    if ema_parameter.device == parameter.device:
                        ema_parameter.add_(one_minus_decay * (parameter - ema_parameter))
                    else:
                        # In-place calculations to save memory.
                        parameter_copy = parameter.detach().to(ema_parameter.device)
                        parameter_copy.sub_(ema_parameter)
                        parameter_copy.mul_(one_minus_decay)
                        ema_parameter.add_(parameter_copy)
                        del parameter_copy

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> None:
        self.device = device
        self.ema_parameters = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.ema_parameters
        ]

    def copy_ema_to(self, parameters: Iterable[torch.nn.Parameter], store_temp: bool = True) -> None:
        # NOTE: `parameters` can be a generator (e.g. module.parameters()).
        # Convert to a list once to avoid consuming the iterator multiple times.
        parameters = list(parameters)

        # Store a temporary copy so we can restore original weights after swap.
        if store_temp:
            self.temp_stored_parameters = [parameter.detach().cpu() for parameter in parameters]
        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            parameter.data.copy_(ema_parameter.to(parameter.device).data)

    def copy_temp_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        if self.temp_stored_parameters is None:
            raise RuntimeError("EMA temp parameters are not initialized. Did you call copy_ema_to(store_temp=True)?")

        parameters = list(parameters)
        for temp_parameter, parameter in zip(self.temp_stored_parameters, parameters, strict=True):
            parameter.data.copy_(temp_parameter.data)
        self.temp_stored_parameters = None

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = self.decay if self.decay else state_dict.get("decay", self.decay)
        self.ema_parameters = state_dict.get("ema_parameters")
        self.to(self.device)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "ema_parameters": self.ema_parameters,
        }


def _remap_ckpt_keys_for_qwen25vl(state_dict: dict) -> dict:
    """
    Normalize checkpoint keys for Qwen2.5-VL compatibility.

    We observed some checkpoints (especially from different Transformers/Qwen2.5-VL
    versions) store LoRA keys under:
      - lmm.model.visual.* / lmm.model.language_model.*
    while the current model expects:
      - lmm.visual.* / lmm.model.*

    Additionally, some checkpoints are wrapped with common prefixes like `module.` or `model.`.
    If not remapped, LoRA (and other) weights will not be loaded correctly, which can
    break KL regularization (ref_model != policy_init) and lead to unstable training.
    """
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict

    new_sd: dict = {}
    changed = 0

    for k, v in state_dict.items():
        nk = k

        # Common wrappers
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]

        # Qwen2.5-VL hierarchy changes across versions
        nk = nk.replace("lmm.model.visual.", "lmm.visual.")
        nk = nk.replace("lmm.model.language_model.", "lmm.model.")

        if nk != k:
            changed += 1
        new_sd[nk] = v

    if changed > 0:
        rank0_print(f"[ckpt_remap] Remapped {changed} checkpoint keys for Qwen2.5-VL compatibility")
    return new_sd


def _is_reporting_to(report_to, backend: str) -> bool:
    """
    Check if a specific logging backend is enabled in report_to.

    Args:
        report_to: The report_to value from TrainingArguments (str or list)
        backend: The backend to check for (e.g., "wandb", "tensorboard")

    Returns:
        True if the backend is enabled
    """
    if report_to is None or report_to == "none":
        return False
    if report_to == "all":
        return True
    if isinstance(report_to, str):
        return report_to == backend
    if isinstance(report_to, (list, tuple)):
        return backend in report_to
    return False


class DiversityLogger:
    """
    A simple logger for SDE sampling diversity analysis.
    Writes logs to {output_dir}/diversity_debug.log.
    Only global rank 0 process writes logs.
    """

    def __init__(self, output_dir: str):
        self._is_rank_zero = self._check_rank_zero()
        self._log_file_path = os.path.join(output_dir, "diversity_debug.log")
        self._file_handle = None

        if self._is_rank_zero:
            self._init_log_file()

    def _check_rank_zero(self) -> bool:
        """Check if current process is global rank 0."""
        try:
            if dist.is_initialized():
                return dist.get_rank() == 0
            return True
        except Exception:
            return True

    def _init_log_file(self) -> None:
        """Initialize the log file for writing."""
        try:
            log_dir = os.path.dirname(self._log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            self._file_handle = open(self._log_file_path, "a", encoding="utf-8")
            self._write_line("=" * 80)
            self._write_line(f"Diversity Logger initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._write_line(f"Log file: {self._log_file_path}")
            self._write_line("=" * 80)
            self._flush()
        except Exception as e:
            rank0_print(f"[WARNING] Failed to initialize diversity log file: {e}")
            self._file_handle = None

    def _write_line(self, line: str) -> None:
        if self._file_handle is not None:
            try:
                self._file_handle.write(line + "\n")
            except Exception:
                pass

    def _flush(self) -> None:
        if self._file_handle is not None:
            try:
                self._file_handle.flush()
            except Exception:
                pass

    def log(self, step: int, **kwargs) -> None:
        """Log diversity statistics."""
        if not self._is_rank_zero or not kwargs:
            return
        datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stats_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        line = f"[{datetime_str}][Step {step}] {stats_str}"
        self._write_line(line)
        self._flush()

    def close(self) -> None:
        """Close the log file."""
        if self._file_handle is not None:
            try:
                self._write_line("=" * 80)
                self._write_line(f"Diversity Logger closed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self._write_line("=" * 80)
                self._flush()
                self._file_handle.close()
            except Exception:
                pass
            finally:
                self._file_handle = None

    def __del__(self):
        self.close()


# mmengine imports for model building
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

if is_wandb_available():
    import wandb


# Type aliases
RewardFunc = Union[str, PreTrainedModel, Callable]
RewardFuncSpec = Union[RewardFunc, List[RewardFunc]]


def sde_step_with_logprob(
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
    next_sample: torch.Tensor,
    eta: float = 1.0,
    debug: bool = False,
    return_sqrt_dt: bool = False,
    sampler_type: str = "flowgrpo_sde",
    clamp_log_prob: bool = False,
    eps: float = 1e-8,
):
    """
    Compute SDE step with log probability for flow matching.

    Reference: flow_grpo/diffusers_patch/sd3_sde_with_logprob.py
    The std_dev_t should vary with timestep t for proper SDE denoising schedule.

    Args:
        scheduler: Flow matching scheduler
        model_output: Model prediction output
        timestep: Current timestep
        sample: Current sample (latent)
        next_sample: Next sample (latent)
        eta: Noise scale factor (noise_level in reference)
        debug: Whether to return debug info
        return_sqrt_dt: Whether to return sqrt_dt for GRPO-Guard RatioNorm
        sampler_type: SDE sampler type, either "flowgrpo_sde" (standard) or "cps_sde" (Coefficients-Preserving)

    Returns:
        Tuple of (prev_sample, log_prob, prev_sample_mean, std_dev_t)
        If return_sqrt_dt=True, also returns sqrt_dt
        If debug=True, also returns debug_info dict
    """
    # bf16 can overflow here when compute prev_sample_mean, convert to fp32
    model_output = model_output.float()
    sample = sample.float()
    next_sample = next_sample.float()

    # Get step indices for sigma lookup
    # Move timestep to CPU for scheduler.index_for_timestep() which compares with CPU tensors
    timestep_cpu = timestep.cpu()
    step_index = [scheduler.index_for_timestep(t) for t in timestep_cpu]
    prev_step_index = [step + 1 for step in step_index]
    sigma = scheduler.sigmas[step_index].view(-1, 1, 1, 1).to(sample.device)
    sigma_prev = scheduler.sigmas[prev_step_index].view(-1, 1, 1, 1).to(sample.device)
    sigma_max = scheduler.sigmas[1].item()
    dt = sigma_prev - sigma

    if sampler_type == "cps_sde":
        # Flow-CPS (Coefficients-Preserving Sampling)
        # Reference: https://arxiv.org/abs/2509.05952, sd3_sde_with_logprob.py
        # std_dev_t = sigma_prev * sin(eta * pi / 2), where eta is noise_level
        std_dev_t = sigma_prev * math.sin(eta * math.pi / 2)
        # pred_original_sample = sample - sigma * model_output (predicted x_0)
        pred_original_sample = sample - sigma * model_output
        # noise_estimate = sample + model_output * (1 - sigma) (predicted x_1)
        noise_estimate = sample + model_output * (1 - sigma)
        # prev_sample_mean = x_0 * (1 - sigma_prev) + x_1 * sqrt(sigma_prev^2 - std_dev_t^2)
        # Note: No clamp to match original FlowCPS exactly
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)
        # For CPS, use simplified log_prob (without variance normalization) as in original FlowCPS
        # This is because the ratio log_prob_new - log_prob_old cancels out constant terms
        actual_std = std_dev_t
        use_simplified_logprob = True
    elif sampler_type == "dance_sde":
        # Dance-SDE
        # Reference: Flow-Factory flow_match_euler_discrete.py
        # pred_original_sample = sample - sigma * model_output (predicted x_0)
        pred_original_sample = sample - sigma * model_output
        # std_dev_t = eta * sqrt(-dt)
        std_dev_t = eta * torch.sqrt(-1 * dt)
        # log_term correction: 0.5 * eta^2 * (sample - x_0 * (1 - sigma)) / sigma^2
        log_term = 0.5 * (eta ** 2) * (sample - pred_original_sample * (1 - sigma)) / (sigma ** 2 + 1e-8)
        # prev_sample_mean = sample + (model_output + log_term) * dt
        prev_sample_mean = sample + (model_output + log_term) * dt
        actual_std = std_dev_t
        use_simplified_logprob = False
    else:
        # Flow-SDE (Standard flow matching SDE)
        # Standard deviation for SDE - varies with timestep t
        # Reference: std_dev_t = torch.sqrt(sigma / (1 - sigma)) * noise_level
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * eta

        # Mean prediction with SDE correction
        # Reference: prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
        prev_sample_mean = sample * (1 + std_dev_t**2 / (2 * sigma) * dt) + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        # For Flow-SDE, the actual std used for sampling/log_prob is std_dev_t * sqrt(-dt)
        actual_std = std_dev_t * torch.sqrt(-1 * dt)
        use_simplified_logprob = False

    # Compute log probability
    diff = next_sample.detach() - prev_sample_mean
    if use_simplified_logprob:
        # CPS uses simplified log_prob from original FlowCPS: log_prob = -((prev_sample - mean)^2)
        # Note: prev_sample (next_sample here) must be detached to match original FlowCPS
        # This works because GRPO only uses log_prob ratios, so constant terms cancel out
        # Reference: flow_grpo/diffusers_patch/sd3_sde_with_logprob.py
        log_prob = -((diff) ** 2)
    else:
        # Full Gaussian log probability for Flow-SDE
        # log p(x_{t-1} | x_t) = -((prev_sample - mean)^2) / (2 * var) - log(std) - log(sqrt(2*pi))
        log_prob = (
            -((diff) ** 2) / (2 * (actual_std ** 2 + eps))
            - torch.log(actual_std + eps)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=sample.device)))
        )
    # Mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    # Optional clamp (Flow-Factory does not clamp by default; keep for safety via flag)
    log_prob_pre_clamp = log_prob.clone()
    if clamp_log_prob:
        log_prob = torch.clamp(log_prob, min=-1e6, max=1e6)

    # Compute sqrt_dt for GRPO-Guard RatioNorm
    sqrt_dt = torch.sqrt(-1 * dt)

    # Debug info for troubleshooting large loss issues
    if debug:
        debug_info = {
            "dt": dt.mean().item() if dt.numel() > 1 else dt.item(),
            "sigma": sigma.mean().item(),
            "std_dev_t": std_dev_t.mean().item(),
            "actual_std": actual_std.mean().item(),
            "diff_mean": diff.mean().item(),
            "diff_std": diff.std().item(),
            "diff_abs_max": diff.abs().max().item(),
            "log_prob_pre_clamp_mean": log_prob_pre_clamp.mean().item(),
            "log_prob_pre_clamp_min": log_prob_pre_clamp.min().item(),
            "log_prob_pre_clamp_max": log_prob_pre_clamp.max().item(),
            "model_output_mean": model_output.mean().item(),
            "model_output_std": model_output.std().item(),
            "sample_mean": sample.mean().item(),
            "sample_std": sample.std().item(),
            "next_sample_mean": next_sample.mean().item(),
            "next_sample_std": next_sample.std().item(),
        }
        if return_sqrt_dt:
            return prev_sample_mean, log_prob, prev_sample_mean, std_dev_t, sqrt_dt, debug_info
        return prev_sample_mean, log_prob, prev_sample_mean, std_dev_t, debug_info

    if return_sqrt_dt:
        return prev_sample_mean, log_prob, prev_sample_mean, std_dev_t, sqrt_dt

    return prev_sample_mean, log_prob, prev_sample_mean, std_dev_t


class DeepGenGRPOTrainer(Trainer):
    @staticmethod
    def _try_parse_flops_value(value) -> Optional[float]:
        """Best-effort parse FLOPs from DeepSpeed FlopsProfiler outputs."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None

        # Common formats: "123.45 GFLOPs", "1.23 TFLOPs", "9.87e12"
        s = value.strip()
        try:
            return float(s)
        except Exception:
            pass

        m = re.search(r"([0-9]*\.?[0-9]+)\s*([KMGTP]?)(?:FLOPs|FLOPS|flops)?", s)
        if not m:
            return None
        num = float(m.group(1))
        unit = m.group(2).upper()
        scale = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15}.get(unit)
        if scale is None:
            return None
        return num * scale
    """
    GRPO Trainer for DeepGen Models (Qwen2p5VLStableDiffusion3HF).

    This trainer uses mmengine to build the model from DeepGen_Image config
    and implements GRPO training for text-to-image generation.
    """

    def __init__(
        self,
        model_config_path: str,
        checkpoint_path: Optional[str] = None,
        reward_funcs: RewardFuncSpec = None,
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
        rollout_n: int = 4,
        rollout_micro_batch_size: int = 4,
        rollout_accumulation_steps: int = 1,
        atrain_micro_batch_size: int = 1,
        atrain_num_actor_update_steps: int = 1,
        log_prob_micro_batch_size: Optional[int] = None,
        beta: float = 0.01,
        cfg_scale: float = 4.5,
        num_inference_steps: int = 10,
        image_height: int = 1024,
        image_width: int = 1024,
        gradient_checkpointing: bool = False,
        log_images_interval: int = 10,
        init_same_noise: bool = False,
        timestep_fraction: float = 1.0,
        clip_range: float = 1e-4,
        sde_eta: float = 1.0,
        use_per_sample_reward_config: bool = False,
        train_sampler: Optional["torch.utils.data.Sampler"] = None,
        # Training algorithm: 'flowgrpo' (default) or 'grpoguard'
        atrain_algorithm: str = "flowgrpo",
        # SDE sampler type: 'flowgrpo_sde' (standard), 'cps_sde', or 'dance_sde'
        atrain_sde_sampler: str = "flowgrpo_sde",
        # Advantage computation type: 'grpo', 'grpo_global_std', 'reinforcepp', 'gdpo'
        atrain_adv_type: str = "grpo",
        # KL divergence type: 'x-based' (latent mean) or 'v-based' (velocity/noise prediction)
        atrain_kl_type: str = "x-based",
        # DiffusionNFT parameters (used when atrain_algorithm='diffusionnft')
        atrain_nft_beta: float = 0.1,
        atrain_nft_adv_clip_range: float = 5.0,
        atrain_nft_off_policy: bool = False,
        # EMA (diffusion transformer only): set to 0 to disable
        ema_diffusion: float = 0.0,
        # Evaluation parameters
        eval_freq: int = 0,
        eval_before_train: bool = False,
        eval_dataset_config: Optional[str] = None,
        eval_inference_mode: str = "ode",
        eval_cfg_scale: Optional[float] = None,
        eval_num_inference_steps: Optional[int] = None,
        eval_sde_eta: Optional[float] = None,
        eval_image_height: Optional[int] = None,
        eval_image_width: Optional[int] = None,
        eval_micro_batch_size: int = 4,
        eval_wandb_num_upload_images: int = 8,
        eval_swanlab_num_upload_images: int = 8,
        # SFT-Aux parameters
        sftaux_dataset_config: Optional[str] = None,
        sftaux_coef: float = 0.1,
        sftaux_every_n_steps: int = 1,
        sftaux_micro_batch_size: Optional[int] = None,
        sftaux_num_workers: Optional[int] = None,
        sftaux_disable_on_error: bool = False,
        # Logging: split grad-norm proxy for GRPO vs SFT-Aux
        log_component_grad_norm: bool = False,
        log_component_grad_norm_every: Optional[int] = None,
        # Logging: approximate TFLOPS (optimization-step level)
        log_tflops: bool = False,
        log_tflops_every: Optional[int] = None,
        log_tflops_warmup_steps: int = 5,
    ):
        """
        Initialize DeepGen GRPO Trainer.

        Args:
            model_config_path: Path to mmengine model config file
            checkpoint_path: Path to pretrained checkpoint (from DeepGen_Image training)
            reward_funcs: Reward function(s) for optimization
            args: Training configuration (GRPOConfig)
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            callbacks: Training callbacks
            optimizers: Optimizer and scheduler tuple
            rollout_n: Number of images to generate per prompt for GRPO
            rollout_micro_batch_size: Number of images per GPU per rollout forward pass
            rollout_accumulation_steps: Number of rollout iterations to reach rollout_global_batch_size
            atrain_micro_batch_size: Number of images per GPU per training forward pass
            atrain_num_actor_update_steps: Number of actor model weight update steps per rollout.
                Each update step samples non-overlapping data from the rollout buffer.
                Inspired by PPO's num_mini_batches and Flow-GRPO's num_inner_epochs.
            log_prob_micro_batch_size: Number of samples per micro-batch for log_prob computation.
                If None, defaults to atrain_micro_batch_size.
            beta: KL divergence penalty coefficient
            cfg_scale: Classifier-free guidance scale
            num_inference_steps: Number of diffusion inference steps
            image_height: Generated image height
            image_width: Generated image width
            gradient_checkpointing: Enable gradient checkpointing to save memory
            log_images_interval: Interval (in steps) for logging training sample images
            init_same_noise: If True, all images in the same group (same prompt) use the same initial noise
            timestep_fraction: Fraction of timesteps to randomly sample for gradient computation
            clip_range: Clipping range for ratio in GRPO loss (PPO-style clipping)
            sde_eta: SDE noise scale factor (higher = more diversity, default 1.0)
            use_per_sample_reward_config: If True, use per-sample reward weights from dataset config.
                Each sample can have a 'reward_config' dict specifying reward function weights.
            train_sampler: Optional sampler for weighted dataset sampling based on sample_weight config.
            grpo_guard: Enable GRPO-Guard for addressing implicit over-optimization in Flow Matching.
                When enabled, uses RatioNorm and Gradient Reweight mechanisms.
                Additional hyperparameters can be set via environment variables:
                - ATRAIN_GRPOGUARD_HIGHCLIP_RANGE: Upper clipping range (default: same as clip_range)
            atrain_sde_sampler: SDE sampler type for training. Options:
                - 'flowgrpo_sde': Standard Flow-GRPO SDE sampling (default)
                - 'cps_sde': Coefficients-Preserving Sampling from FlowCPS (arXiv:2509.05952)
            eval_freq: Evaluation frequency in steps. 0 to disable periodic evaluation.
            eval_before_train: If True, run evaluation before training starts.
            eval_dataset_config: Path to YAML config file for evaluation datasets.
            eval_inference_mode: Inference mode for evaluation: 'sde' or 'ode' (default).
            eval_cfg_scale: CFG scale for evaluation. If None, uses training cfg_scale.
            eval_num_inference_steps: Number of inference steps for evaluation. If None, uses training value.
            eval_sde_eta: SDE noise scale for evaluation. If None, uses training sde_eta.
            eval_image_height: Image height for evaluation. If None, uses training image_height.
            eval_image_width: Image width for evaluation. If None, uses training image_width.
            eval_micro_batch_size: Images per GPU during evaluation inference.
            eval_wandb_num_upload_images: Number of prompts to upload images for to wandb.
                Selects prompts with smallest indices (0, 1, ..., N-1).
            eval_swanlab_num_upload_images: Number of prompts to upload images for to swanlab.
                Selects prompts with smallest indices (0, 1, ..., N-1).
            sftaux_dataset_config: Path to deepgen_sft mmengine dataset config (.py) used for SFT-Aux.
            sftaux_coef: Convex mixing coefficient lambda in [0, 1] for total_loss=(1-lambda)*grpo_loss + lambda*sft_loss.
            sftaux_every_n_steps: Compute SFT-Aux loss every N GRPO steps.
            sftaux_micro_batch_size: Optional override for deepgen_sft train_dataloader.batch_size (SFT-Aux micro batch size per GPU).
            sftaux_num_workers: Optional override for deepgen_sft train_dataloader.num_workers.
            sftaux_disable_on_error: If True, disable SFT-Aux on dataloader build errors instead of raising.
        """
        # Configuration
        if args is None:
            args = GRPOConfig("deepgen-grpo")

        self.model_config_path = model_config_path
        self.checkpoint_path = checkpoint_path
        self.rollout_n = rollout_n
        self.rollout_micro_batch_size = rollout_micro_batch_size
        self.rollout_accumulation_steps = rollout_accumulation_steps
        self.atrain_micro_batch_size = atrain_micro_batch_size
        self.atrain_num_actor_update_steps = atrain_num_actor_update_steps
        self.log_prob_micro_batch_size = log_prob_micro_batch_size if log_prob_micro_batch_size is not None else atrain_micro_batch_size
        self.beta = beta
        self.image_height = image_height
        self.image_width = image_width
        self.log_images_interval = log_images_interval
        self.init_same_noise = init_same_noise
        self.timestep_fraction = timestep_fraction
        self.clip_range = clip_range
        self.sde_eta = sde_eta
        self.cfg_scale = cfg_scale
        self.use_per_sample_reward_config = use_per_sample_reward_config
        self.train_sampler = train_sampler

        # EMA (diffusion transformer only). We initialize the actual EMA wrapper after
        # the parent Trainer sets up accelerator/model wrapping.
        self.ema_diffusion_decay = float(ema_diffusion)
        self.ema_diffusion: Optional[EMAModuleWrapper] = None

        # ============================================================================
        # Training Algorithm Configuration
        # ============================================================================
        self.atrain_algorithm = atrain_algorithm
        # SDE sampler type: 'flowgrpo_sde' (standard), 'cps_sde', or 'dance_sde'
        # Reference for cps_sde: https://arxiv.org/abs/2509.05952 (Flow-CPS)
        self.atrain_sde_sampler = atrain_sde_sampler
        if atrain_sde_sampler not in ["flowgrpo_sde", "cps_sde", "dance_sde"]:
            raise ValueError(f"Invalid atrain_sde_sampler: {atrain_sde_sampler}. Must be 'flowgrpo_sde', 'cps_sde', or 'dance_sde'")
        # Advantage computation type: 'grpo', 'grpo_global_std', 'reinforcepp', 'gdpo'
        self.atrain_adv_type = atrain_adv_type
        if atrain_adv_type not in ["grpo", "grpo_global_std", "reinforcepp", "gdpo"]:
            raise ValueError(f"Invalid atrain_adv_type: {atrain_adv_type}. Must be 'grpo', 'grpo_global_std', 'reinforcepp', or 'gdpo'")
        # KL divergence type
        self.atrain_kl_type = atrain_kl_type
        if atrain_kl_type not in ["x-based", "v-based"]:
            raise ValueError(f"Invalid atrain_kl_type: {atrain_kl_type}. Must be 'x-based' or 'v-based'")
        rank0_print(f"[SDE Sampler] Using {atrain_sde_sampler}")
        rank0_print(f"[Advantage Normalization] adv_type={atrain_adv_type}")
        rank0_print(f"[KL Type] {atrain_kl_type}")
        # GRPO-Guard specific configuration (for addressing implicit over-optimization)
        # Reference: flow_grpo/scripts/train_sd3_GRPO_Guard.py
        self.grpo_guard = (atrain_algorithm == "grpoguard")
        # Read hyperparameters from environment variables with ATRAIN_GRPOGUARD_ prefix
        self.grpo_guard_highclip_range = float(os.environ.get("ATRAIN_GRPOGUARD_HIGHCLIP_RANGE", clip_range))
        if self.grpo_guard:
            rank0_print(f"[GRPO-Guard] Enabled with RatioNorm and Gradient Reweight")
            rank0_print(f"[GRPO-Guard] clip_range={clip_range}, highclip_range={self.grpo_guard_highclip_range}")

        # ============================================================================
        # DiffusionNFT Configuration
        # Reference: https://arxiv.org/abs/2509.16117 (DiffusionNFT)
        # ============================================================================
        self.diffusion_nft = (atrain_algorithm == "diffusionnft")
        # NFT beta: controls the mixture of new and old predictions
        self.nft_beta = float(os.environ.get("ATRAIN_NFT_BETA", atrain_nft_beta))
        # NFT advantage clipping range
        self.nft_adv_clip_range = float(os.environ.get("ATRAIN_NFT_ADV_CLIP_RANGE", atrain_nft_adv_clip_range))
        # NFT off-policy mode: use EMA parameters for old_v_pred computation
        self.nft_off_policy = atrain_nft_off_policy
        if self.diffusion_nft:
            rank0_print(f"[DiffusionNFT] Enabled with v-space loss")
            rank0_print(f"[DiffusionNFT] nft_beta={self.nft_beta}, nft_adv_clip_range={self.nft_adv_clip_range}")
            # ============================================================================
            # Off-policy mode: use EMA parameters for old_v_pred (similar to Flow-Factory)
            # On-policy mode (default): old_v_pred = current model detached
            # ============================================================================
            if self.nft_off_policy:
                if ema_diffusion <= 0:
                    rank0_print(f"[DiffusionNFT] Warning: off_policy=True requires --ema_diffusion > 0. "
                                f"Falling back to on-policy mode.")
                    self.nft_off_policy = False
                else:
                    rank0_print(f"[DiffusionNFT] Off-policy mode enabled: old_v_pred from EMA parameters")
            else:
                rank0_print(f"[DiffusionNFT] On-policy mode: old_v_pred = current model detached")
            # DiffusionNFT uses v-based KL by default
            if atrain_kl_type != "v-based":
                rank0_print(f"[DiffusionNFT] Warning: Switching KL type from '{atrain_kl_type}' to 'v-based' (required for DiffusionNFT)")
                self.atrain_kl_type = "v-based"
            # Recommend GDPO advantage aggregation for DiffusionNFT
            if atrain_adv_type != "gdpo":
                rank0_print(f"[DiffusionNFT] Note: Consider using --atrain_adv_type=gdpo for best results (current: {atrain_adv_type})")
            # ============================================================================
            # NFT requires ODE (deterministic) rollout, not SDE
            # Reference: Official DiffusionNFT uses solver="dpm2" with deterministic=True
            # SDE sampling introduces stochastic noise which breaks NFT's v-space comparison
            # ============================================================================
            if sde_eta > 0.0:
                rank0_print(f"[DiffusionNFT] Warning: Forcing sde_eta from {sde_eta} to 0.0 (NFT requires ODE/deterministic rollout)")
                sde_eta = 0.0
            self.sde_eta = sde_eta
            rank0_print(f"[DiffusionNFT] Using ODE rollout (sde_eta=0.0) as required by NFT algorithm")

        if not self.grpo_guard and not self.diffusion_nft:
            rank0_print(f"[Training Algorithm] Using {atrain_algorithm}")

        # Match Flow-Factory math defaults (can be overridden via env vars)
        # - Flow-Factory GRPO does NOT clamp log_prob_diff before exp
        # - Flow-Factory scheduler does NOT clamp log_prob by default
        self._clamp_log_prob_diff = os.environ.get("ATRAIN_CLAMP_LOGPROB_DIFF", "0") == "1"
        self._clamp_log_prob = os.environ.get("ATRAIN_CLAMP_LOGPROB", "0") == "1"

        # ============================================================================
        # Evaluation Configuration
        # ============================================================================
        self.eval_freq = eval_freq
        self.eval_before_train = eval_before_train
        self.eval_dataset_config_path = eval_dataset_config
        self.eval_inference_mode = eval_inference_mode
        # Use training parameters as defaults if not specified
        self.eval_cfg_scale = eval_cfg_scale if eval_cfg_scale is not None else cfg_scale
        self.eval_num_inference_steps = eval_num_inference_steps if eval_num_inference_steps is not None else num_inference_steps
        self.eval_sde_eta = eval_sde_eta if eval_sde_eta is not None else sde_eta
        self.eval_image_height = eval_image_height if eval_image_height is not None else image_height
        self.eval_image_width = eval_image_width if eval_image_width is not None else image_width
        self.eval_micro_batch_size = eval_micro_batch_size
        self.eval_wandb_num_upload_images = eval_wandb_num_upload_images
        self.eval_swanlab_num_upload_images = eval_swanlab_num_upload_images

        # ============================================================================
        # SFT-Aux Configuration
        # ============================================================================
        self.sftaux_dataset_config = sftaux_dataset_config
        self.sftaux_coef = float(sftaux_coef)
        self.sftaux_every_n_steps = int(sftaux_every_n_steps)
        self.sftaux_micro_batch_size = sftaux_micro_batch_size
        self.sftaux_num_workers = sftaux_num_workers
        self.sftaux_disable_on_error = bool(sftaux_disable_on_error)
        self.sftaux_dataloader = None
        self.sftaux_iter = None
        self._sftaux_build_error: Optional[str] = None

        # ============================================================================
        # Logging: component grad-norm proxy
        # ============================================================================
        # NOTE:
        # We log a proxy metric based on the sum of squared gradient *increments* coming
        # from GRPO backward calls and SFT-Aux backward calls separately within a step.
        # This helps monitor relative gradient scale, but it is NOT the exact L2 norm
        # of the final accumulated gradient vector for each component.
        self.log_component_grad_norm = bool(log_component_grad_norm)
        self.log_component_grad_norm_every = (
            int(log_component_grad_norm_every) if log_component_grad_norm_every is not None else None
        )

        if self.sftaux_coef < 0.0 or self.sftaux_coef > 1.0:
            raise ValueError(f"sftaux_coef must be in [0, 1], got {self.sftaux_coef}")
        if self.sftaux_every_n_steps <= 0:
            raise ValueError(f"sftaux_every_n_steps must be positive, got {self.sftaux_every_n_steps}")
        if self.sftaux_micro_batch_size is not None and int(self.sftaux_micro_batch_size) <= 0:
            raise ValueError(f"sftaux_micro_batch_size must be positive, got {self.sftaux_micro_batch_size}")
        if self.sftaux_num_workers is not None and int(self.sftaux_num_workers) < 0:
            raise ValueError(f"sftaux_num_workers must be >= 0, got {self.sftaux_num_workers}")
        if self.sftaux_coef > 0.0 and not self.sftaux_dataset_config:
            raise ValueError("sftaux_coef > 0 but sftaux_dataset_config is empty")

        # Debugging: write eval upload selection/logs to output_dir for easier inspection
        # Enable via env var: DEBUG_SWANLAB_VIS=1
        self._debug_swanlab_vis = os.environ.get("DEBUG_SWANLAB_VIS", "0") == "1"

        # Initialize evaluation datasets
        self.eval_datasets: List[EvalPromptDataset] = []
        if eval_dataset_config is not None:
            try:
                eval_config = EvalDatasetConfig(eval_dataset_config)
                self.eval_datasets = eval_config.create_datasets()
                rank0_print(f"Loaded {len(self.eval_datasets)} evaluation datasets")
            except Exception as e:
                rank0_print(f"Warning: Failed to load evaluation datasets: {e}")
                self.eval_datasets = []

        # Evaluation is enabled if we have datasets AND (eval_freq > 0 OR eval_before_train)
        self.eval_enabled = len(self.eval_datasets) > 0 and (eval_freq > 0 or eval_before_train)
        if self.eval_enabled:
            rank0_print(f"Evaluation enabled: freq={eval_freq}, before_train={eval_before_train}, mode={eval_inference_mode}")

        # Pre-select fixed random samples for logging upload (consistent across all eval steps)
        # This ensures the same (prompt_id, dup_idx) pairs are uploaded every step for comparison
        # NOTE: Upload indices are now lazily initialized on first evaluate() call,
        # based on actually generated images, and saved to a file for consistency.
        self.eval_upload_indices_wandb = {}   # dataset_name -> set of (prompt_idx, dup_idx)
        self.eval_upload_indices_swanlab = {}  # dataset_name -> set of (prompt_idx, dup_idx)
        self._eval_upload_indices_initialized = False  # Flag to indicate if indices are loaded/initialized

        # Build reward func name to index mapping for per-sample reward config
        self.reward_func_name_to_idx = {}

        # Validate atrain_micro_batch_size
        if atrain_micro_batch_size <= 0:
            raise ValueError(
                f"atrain_micro_batch_size must be positive, got {atrain_micro_batch_size}"
            )

        # Diffusion config
        # num_images_per_prompt controls how many images per prompt in each generation call
        self.diffusion_config = {
            "guidance_scale": cfg_scale,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": rollout_n,  # Generate rollout_n images per prompt
        }

        # Build model using mmengine
        rank0_print(f"Loading model config from: {model_config_path}")
        config = Config.fromfile(model_config_path)
        model = BUILDER.build(config.model)

        # Check if LoRA adapter is present in the model
        self._check_lora_status(model, stage="after_model_build")

        # Load checkpoint if provided
        if checkpoint_path is not None:
            rank0_print(f"Loading checkpoint from: {checkpoint_path}")
            state_dict = guess_load_checkpoint(checkpoint_path)

            state_dict = _remap_ckpt_keys_for_qwen25vl(state_dict)
            ckpt_key_set = set(state_dict.keys())

            # Check if checkpoint contains LoRA weights
            lora_keys_in_ckpt = [k for k in state_dict.keys() if 'lora' in k.lower()]
            rank0_print(f"LoRA keys in checkpoint: {len(lora_keys_in_ckpt)}")
            if lora_keys_in_ckpt:
                rank0_print(f"Sample LoRA keys: {lora_keys_in_ckpt[:5]}")

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                rank0_print(f"Missing keys: {missing}")
            if unexpected:
                rank0_print(f"Unexpected keys: {unexpected}")

            # Check LoRA status after loading checkpoint
            self._check_lora_status(model, stage="after_checkpoint_load")

        # Convert to bfloat16 and set eval mode initially
        model = model.bfloat16()

        # Configure trainable parameters
        self._configure_model_components(model)

        # Sanity check: for "tunable-only" checkpoints, missing keys are expected.
        # What we do want to catch is trainable weights that are NOT present in ckpt at all.
        if checkpoint_path is not None and 'ckpt_key_set' in locals():
            trainable_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
            missing_trainable = [n for n in trainable_param_names if n not in ckpt_key_set]
            if missing_trainable:
                rank0_print(
                    f"[ckpt_sanity] WARNING: {len(missing_trainable)} trainable parameters are not in checkpoint keys. "
                    f"Sample: {missing_trainable[:20]}"
                )
            else:
                rank0_print("[ckpt_sanity] OK: all trainable parameters have matching keys in checkpoint.")

        # Enable gradient checkpointing to save memory
        if gradient_checkpointing:
            rank0_print("Enabling gradient checkpointing for memory optimization...")
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            else:
                # Manually enable for each component
                if hasattr(model, 'lmm') and hasattr(model.lmm, 'gradient_checkpointing_enable'):
                    model.lmm.gradient_checkpointing_enable()
                if hasattr(model, 'transformer') and hasattr(model.transformer, 'enable_gradient_checkpointing'):
                    model.transformer.enable_gradient_checkpointing()
                if hasattr(model, 'connector') and hasattr(model.connector, 'gradient_checkpointing'):
                    model.connector.gradient_checkpointing = True

        # Create reference model (frozen copy for KL computation)
        # Skip if beta=0 (no KL penalty needed)
        self.ref_transformer = None
        self._use_transformer_only_ref = False
        if self.beta > 0:
            # Memory optimization:
            # When the text-embedding path is fully frozen (LMM + connector/meta_queries),
            # and the only trainable module is the diffusion transformer, we can avoid
            # constructing a full reference model (which would duplicate Qwen2.5-VL).
            # Instead, keep only a frozen copy of the transformer as the KL reference.
            if self._can_use_transformer_only_ref(model):
                self.ref_transformer = deepcopy(model.transformer)
                self.ref_transformer.requires_grad_(False)
                self.ref_transformer.eval()
                self._use_transformer_only_ref = True
                self.ref_model = None
                rank0_print("[KL] Using transformer-only reference (skip duplicating LMM/Qwen2.5-VL)")
            else:
                self.ref_model = self._create_reference_model(model_config_path, checkpoint_path)
        else:
            self.ref_model = None
            rank0_print("Skipping reference model creation since beta=0 (no KL penalty)")

        # Setup scheduler for SDE sampling - use model's test_scheduler to match original pipeline
        self.scheduler = model.test_scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # Get tokenizer from model
        self.processing_class = model.tokenizer

        # Reward functions
        self.reward_funcs = reward_funcs if reward_funcs is not None else []

        # Build reward func name to index mapping for per-sample reward config
        for i, (func_name, _, _) in enumerate(self.reward_funcs):
            self.reward_func_name_to_idx[func_name] = i

        # Image transforms
        self._setup_transforms()

        # Logging
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_dir = os.path.join(args.output_dir, "training_samples")
        os.makedirs(self.log_dir, exist_ok=True)
        self._metrics = defaultdict(list)
        # Counter for total actor update steps (for per-update logging)
        self._actor_update_count = 0

        # ------------------------------------------------------------------
        # TFLOPS logging (approximate)
        # ------------------------------------------------------------------
        # NOTE:
        # - We treat "optimization step" as the unit that increments `self.state.global_step`.
        # - With gradient accumulation, `training_step()` can be called multiple times with
        #   the same global_step before a single `optimizer_step()` call.
        # - We measure wall-time across the whole optimizer step (from first micro-step
        #   entering training_step to the end of optimizer_step).
        self.log_tflops = bool(log_tflops)
        self.log_tflops_every = int(log_tflops_every) if log_tflops_every is not None else None
        self.tflops_warmup_steps = int(log_tflops_warmup_steps) if log_tflops_warmup_steps is not None else 0
        self._tflops_step_start_time_s: Optional[float] = None
        self._tflops_step_id: Optional[int] = None
        self._tflops_calibrated_flops_per_opt_step: Optional[float] = None
        self._tflops_profiler = None
        self._tflops_calibration_in_progress: bool = False
        self._tflops_profiler_available: Optional[bool] = None

        # Data collator (identity function)
        def data_collator(features):
            return features

        # Initialize parent Trainer
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Initialize EMA after Trainer has prepared the model wrappers/devices.
        self._init_ema_diffusion()

        self.model_accepts_loss_kwargs = False

        # Initialize PhaseLogger for training process monitoring
        # Only global rank 0 will write to log file
        self.phase_logger = PhaseLogger(output_dir=args.output_dir)
        set_phase_logger(self.phase_logger)

        # Initialize DiversityLogger for SDE sampling diversity analysis
        self.diversity_logger = DiversityLogger(output_dir=args.output_dir)

        # Log trainable and non-trainable parameters on rank 0 before training starts
        self._log_model_parameters(model, args.output_dir)

        # Prepare reference model for distributed training
        # Note: ref_model is fully frozen (no trainable params), so we use prepare_model
        # instead of prepare_deepspeed to avoid empty tensor list error in DeepSpeed optimizer
        # Skip if beta=0 (ref_model is None)
        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        if self.ref_transformer is not None:
            self.ref_transformer = self.accelerator.prepare_model(self.ref_transformer, evaluation_mode=True)

        # Pre-load CLIP model if clip_sim reward is used (avoid loading per step)
        self.clip_model = None
        self.clip_processor = None
        if any(func_name == "clip_sim" for func_name, _, _ in self.reward_funcs):
            from transformers import CLIPModel, CLIPProcessor
            clip_model_path = os.environ.get("CLIP_MODEL_NAME_OR_PATH", "openai/clip-vit-large-patch14")
            rank0_print(f"Pre-loading CLIP model for clip_sim reward from: {clip_model_path}")
            self.clip_model = CLIPModel.from_pretrained(clip_model_path)
            self.clip_model.eval()
            self.clip_model.requires_grad_(False)
            # Prepare CLIP model for distributed training (evaluation mode)
            self.clip_model = self.accelerator.prepare_model(self.clip_model, evaluation_mode=True)
            # Load processor for text/image preprocessing
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
            rank0_print(f"CLIP model loaded successfully on device: {self.accelerator.device}")

        # Setup SFT-Aux dataloader (optional; may disable itself if configured to do so).
        self._setup_sftaux()

    def _get_deepgen_sft_root(self) -> str:
        """Return the absolute path to the sft module under this workspace."""
        return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "sft"))

    def _setup_sftaux(self) -> None:
        """Build deepgen_sft dataloader for SFT-Aux loss (mmengine config)."""
        if self.sftaux_coef <= 0.0:
            # lambda=0 => pure GRPO, no need to build SFT dataloader.
            return

        try:
            deepgen_sft_root = self._get_deepgen_sft_root()
            if not os.path.isdir(deepgen_sft_root):
                raise FileNotFoundError(f"deepgen_sft root not found: {deepgen_sft_root}")

            dataset_cfg_path = str(self.sftaux_dataset_config)
            if not os.path.isabs(dataset_cfg_path):
                # Resolve relative paths against the DeepGen-RL repo root.
                # deepgen_sft_root = {repo_root}/deepgen_rl/sft
                repo_root = os.path.normpath(os.path.join(deepgen_sft_root, "..", ".."))
                dataset_cfg_path = os.path.normpath(os.path.join(repo_root, dataset_cfg_path))
            if not os.path.exists(dataset_cfg_path):
                raise FileNotFoundError(f"SFT-Aux dataset config not found: {dataset_cfg_path}")

            # Load mmengine config from deepgen_sft.
            cfg = Config.fromfile(dataset_cfg_path)
            if not hasattr(cfg, "train_dataloader"):
                raise ValueError(f"SFT-Aux config has no train_dataloader: {dataset_cfg_path}")

            # Apply optional overrides.
            if self.sftaux_micro_batch_size is not None:
                try:
                    cfg.train_dataloader.batch_size = int(self.sftaux_micro_batch_size)
                except Exception:
                    cfg.train_dataloader["batch_size"] = int(self.sftaux_micro_batch_size)
            if self.sftaux_num_workers is not None:
                try:
                    cfg.train_dataloader.num_workers = int(self.sftaux_num_workers)
                except Exception:
                    cfg.train_dataloader["num_workers"] = int(self.sftaux_num_workers)

            # Build dataloader using deepgen_sft runner.
            from deepgen_rl.sft.runners.custom_runner import CustomRunner

            self.sftaux_dataloader = CustomRunner.build_dataloader(cfg.train_dataloader)
            self.sftaux_iter = iter(self.sftaux_dataloader)

            # Extract sampler info for logging
            sampler = self.sftaux_dataloader.sampler
            sampler_type = type(sampler).__name__
            # Try to get sample_weights from sampler (WeightedInfiniteSampler has get_sample_weights method)
            if hasattr(sampler, 'get_sample_weights'):
                sample_weights = sampler.get_sample_weights()
            elif hasattr(sampler, 'sample_weights'):
                sample_weights = sampler.sample_weights
            else:
                sample_weights = None

            log_msg = (
                "[SFT-Aux] Enabled | "
                f"lambda={self.sftaux_coef}, every_n_steps={self.sftaux_every_n_steps}, "
                f"config={dataset_cfg_path}, "
                f"batch_size={getattr(cfg.train_dataloader, 'batch_size', None) or cfg.train_dataloader.get('batch_size')}, "
                f"num_workers={getattr(cfg.train_dataloader, 'num_workers', None) or cfg.train_dataloader.get('num_workers')}, "
                f"sampler={sampler_type}"
            )
            if sample_weights is not None:
                log_msg += f", sample_weights={sample_weights}"
            rank0_print(log_msg)
        except Exception as e:
            self._sftaux_build_error = str(e)
            if self.sftaux_disable_on_error:
                rank0_print(f"[SFT-Aux] WARNING: disabled due to error while building dataloader: {e}")
                self.sftaux_dataloader = None
                self.sftaux_iter = None
                return
            raise

    def _next_sftaux_batch(self) -> Optional[Dict[str, Any]]:
        """Return the next SFT-Aux batch; resets iterator at epoch boundary."""
        if self.sftaux_coef <= 0.0 or self.sftaux_dataloader is None:
            return None
        if self.sftaux_iter is None:
            self.sftaux_iter = iter(self.sftaux_dataloader)
        try:
            return next(self.sftaux_iter)
        except StopIteration:
            self.sftaux_iter = iter(self.sftaux_dataloader)
            return next(self.sftaux_iter)

    def _compute_sftaux_loss(self, model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute SFT diffusion loss from a deepgen_sft batch."""
        if not isinstance(batch, dict) or "data" not in batch:
            raise ValueError(f"Invalid SFT-Aux batch format: expected dict with key 'data', got keys={list(batch.keys()) if isinstance(batch, dict) else type(batch)}")

        data_dict = batch["data"]
        unwrapped = self._get_unwrapped_model(model)
        if not hasattr(unwrapped, "compute_loss"):
            raise AttributeError("Policy model does not implement compute_loss(data_dict) required for SFT-Aux.")

        losses = unwrapped.compute_loss(data_dict)
        if isinstance(losses, dict):
            if "loss_text2image" in losses:
                return losses["loss_text2image"]
            # Fallback: sum all tensor losses.
            total = None
            for v in losses.values():
                if torch.is_tensor(v):
                    total = v if total is None else (total + v)
            if total is None:
                raise ValueError(f"SFT-Aux compute_loss returned dict without tensor losses: {list(losses.keys())}")
            return total
        if torch.is_tensor(losses):
            return losses
        raise ValueError(f"SFT-Aux compute_loss returned unsupported type: {type(losses)}")

    def _check_lora_status(self, model, stage: str = ""):
        """
        Check and log LoRA adapter status in the model.

        Args:
            model: The model to check
            stage: Description of when this check is being performed
        """
        rank0_print(f"\n{'=' * 60}")
        rank0_print(f"LoRA Status Check [{stage}]")
        rank0_print(f"{'=' * 60}")

        # Check if model has LMM component
        if not hasattr(model, 'lmm'):
            rank0_print("Model does not have 'lmm' attribute")
            return

        lmm = model.lmm

        # Check if LMM has LoRA adapters (PEFT style)
        has_adapter = hasattr(lmm, 'peft_config') or hasattr(lmm, 'active_adapter')
        rank0_print(f"LMM has PEFT adapter config: {has_adapter}")

        if hasattr(lmm, 'peft_config'):
            rank0_print(f"PEFT config: {lmm.peft_config}")

        if hasattr(lmm, 'active_adapter'):
            rank0_print(f"Active adapter: {lmm.active_adapter}")

        # Count LoRA parameters in state dict
        lora_params = {}
        total_lora_params = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                lora_params[name] = param.shape
                total_lora_params += param.numel()

        rank0_print(f"Total LoRA parameters in model: {total_lora_params:,}")
        rank0_print(f"Number of LoRA parameter tensors: {len(lora_params)}")

        if lora_params:
            # Print first few LoRA parameter names and their norms
            rank0_print("Sample LoRA parameters (name, shape, norm):")
            for i, (name, shape) in enumerate(list(lora_params.items())[:10]):
                param = dict(model.named_parameters())[name]
                norm = param.data.norm().item()
                rank0_print(f"  {name}: {shape}, norm={norm:.6f}")
        else:
            rank0_print("WARNING: No LoRA parameters found in model!")

        # Check LoRA trainability
        trainable_lora = sum(1 for name, p in model.named_parameters()
                           if 'lora' in name.lower() and p.requires_grad)
        rank0_print(f"Trainable LoRA parameter tensors: {trainable_lora}")

        rank0_print(f"{'=' * 60}\n")

    def _configure_model_components(self, model):
        """
        Configure which model components to train.

        For DeepGen T2I, we train:
        - connector (projector_1, projector_2, projector_3)
        - meta_queries
        - transformer (if not frozen in config)

        We freeze:
        - VAE
        - LMM (language model)
        """
        # Freeze VAE (always frozen)
        if hasattr(model, 'vae') and model.vae is not None:
            model.vae.requires_grad_(False)

        # Freeze LMM if configured (usually frozen)
        if hasattr(model, 'lmm') and model.freeze_lmm:
            model.lmm.requires_grad_(False)

        # The connector and meta_queries: trainable by default, unless freeze_mq is True
        freeze_mq = getattr(model, 'freeze_mq', False)
        if hasattr(model, 'connector'):
            model.connector.requires_grad_(not freeze_mq)
        if hasattr(model, 'projector_1'):
            model.projector_1.requires_grad_(not freeze_mq)
        if hasattr(model, 'projector_2'):
            model.projector_2.requires_grad_(not freeze_mq)
        if hasattr(model, 'projector_3'):
            model.projector_3.requires_grad_(not freeze_mq)
        if hasattr(model, 'meta_queries'):
            model.meta_queries.requires_grad = not freeze_mq

        # Transformer training depends on config
        if hasattr(model, 'transformer'):
            if hasattr(model, 'freeze_transformer') and not model.freeze_transformer:
                model.transformer.requires_grad_(True)
            else:
                model.transformer.requires_grad_(False)

    def _can_use_transformer_only_ref(self, model) -> bool:
        """
        Decide whether we can use a transformer-only reference for KL.

        This is safe when the text-conditioning path is fully frozen, so the only
        learnable module is the diffusion transformer. In that case, reference
        predictions can reuse the same text embeddings and only require a frozen
        transformer snapshot, avoiding duplicating the (large) LMM.
        """
        # LMM must be frozen (no training in Qwen2.5-VL / text path).
        if not getattr(model, "freeze_lmm", False):
            return False
        # Connector/meta_queries/projectors must also be frozen, otherwise the
        # conditioning embeddings would drift during training and a transformer-only
        # reference would be incorrect.
        if not getattr(model, "freeze_mq", False):
            return False

        # Safety check: ensure there are no trainable parameters outside transformer.
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        non_transformer = [n for n in trainable if not (n == "transformer" or n.startswith("transformer."))]
        return len(non_transformer) == 0

    # ============================================================================
    # EMA (Diffusion Transformer Only)
    # ============================================================================

    def _get_unwrapped_model(self, model: Optional[nn.Module] = None) -> nn.Module:
        """Unwrap common wrappers (DDP/Accelerate/DeepSpeed) to access real attributes."""
        if model is None:
            model = self.model
        return model.module if hasattr(model, "module") else model

    def _get_diffusion_transformer_module(self, model: Optional[nn.Module] = None) -> nn.Module:
        """Get the diffusion transformer module (the only target for --ema_diffusion)."""
        unwrapped = self._get_unwrapped_model(model)
        if not hasattr(unwrapped, "transformer") or unwrapped.transformer is None:
            raise AttributeError("Model does not have a valid `transformer` module for diffusion EMA.")
        return unwrapped.transformer

    def _init_ema_diffusion(self) -> None:
        """Initialize EMA for diffusion transformer parameters when --ema_diffusion > 0."""
        if self.ema_diffusion_decay <= 0:
            return

        if self.ema_diffusion_decay >= 1.0:
            rank0_print(
                f"[EMA] Warning: ema_diffusion must be < 1.0, got {self.ema_diffusion_decay}. Disabling EMA."
            )
            self.ema_diffusion = None
            return

        try:
            transformer = self._get_diffusion_transformer_module()
            params = list(transformer.parameters())
            if len(params) == 0:
                rank0_print("[EMA] Warning: diffusion transformer has no parameters. Disabling EMA.")
                self.ema_diffusion = None
                return

            # Store EMA weights on CPU to avoid duplicating the (large) transformer on GPU.
            self.ema_diffusion = EMAModuleWrapper(
                parameters=params,
                decay=self.ema_diffusion_decay,
                update_step_interval=1,
                device=torch.device("cpu"),
            )
            rank0_print(
                f"[EMA] Enabled diffusion EMA (transformer only): decay={self.ema_diffusion_decay}, device=cpu"
            )
        except Exception as e:
            rank0_print(f"[EMA] Warning: failed to initialize diffusion EMA: {e}")
            self.ema_diffusion = None

    def _update_ema_diffusion(self) -> None:
        """Update EMA after a successful optimizer step."""
        if self.ema_diffusion is None:
            return
        try:
            transformer = self._get_diffusion_transformer_module()
            self.ema_diffusion.step(transformer.parameters(), optimization_step=int(self.state.global_step))
        except Exception as e:
            rank0_print(f"[EMA] Warning: diffusion EMA update failed: {e}")

    @contextmanager
    def _swap_in_ema_diffusion_weights(self):
        """Temporarily swap diffusion transformer weights to EMA weights."""
        if self.ema_diffusion is None:
            yield
            return
        transformer = self._get_diffusion_transformer_module()
        self.ema_diffusion.copy_ema_to(transformer.parameters(), store_temp=True)
        try:
            yield
        finally:
            self.ema_diffusion.copy_temp_to(transformer.parameters())

    def optimizer_step(self, *args, **kwargs):
        """
        Hook after the optimizer step to update EMA.

        We intentionally update EMA only once per optimizer step (not per micro-batch),
        because gradients are accumulated inside compute_loss and the optimizer step is
        executed by the parent Trainer.
        """
        result = super().optimizer_step(*args, **kwargs)

        # --------------------------------------------------------------
        # TFLOPS logging: finalize timing + (optional) FLOPs calibration
        # --------------------------------------------------------------
        try:
            if self.log_tflops and self._tflops_step_start_time_s is not None:
                step_end_s = time.perf_counter()
                step_duration_s = max(step_end_s - float(self._tflops_step_start_time_s), 1e-9)

                # Prefer explicit config; fall back to logging_steps.
                every = self.log_tflops_every
                if every is None:
                    every = int(getattr(self.args, "logging_steps", 0) or 0)
                do_log = every > 0 and (int(self.state.global_step) % every == 0)

                # If we are calibrating this step, stop profiler and cache FLOPs.
                if self._tflops_calibration_in_progress and self._tflops_profiler is not None:
                    try:
                        self._tflops_profiler.stop_profile()
                        # DeepSpeed APIs may vary by version; keep this robust.
                        try:
                            raw_flops = self._tflops_profiler.get_total_flops(as_string=False)
                        except TypeError:
                            raw_flops = self._tflops_profiler.get_total_flops()
                        total_flops = self._try_parse_flops_value(raw_flops)
                        if total_flops is not None and total_flops > 0:
                            self._tflops_calibrated_flops_per_opt_step = float(total_flops)
                    except Exception:
                        # Best-effort: do not fail training if profiler API differs.
                        pass
                    finally:
                        try:
                            self._tflops_profiler.end_profile()
                        except Exception:
                            pass
                        self._tflops_profiler = None
                        self._tflops_calibration_in_progress = False

                # Log only on rank 0 to avoid duplicated metrics.
                if do_log and is_rank_zero():
                    self._metrics["perf/opt_step_time_ms"].append(step_duration_s * 1000.0)
                    if self._tflops_calibrated_flops_per_opt_step is not None:
                        tflops = (self._tflops_calibrated_flops_per_opt_step / step_duration_s) / 1e12
                        self._metrics["perf/tflops"].append(float(tflops))
        finally:
            # Always reset step timer for the next optimizer step.
            self._tflops_step_start_time_s = None
            self._tflops_step_id = None

        self._update_ema_diffusion()
        return result

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save model weights.

        - For internal checkpointing (_internal_call=True), we save the raw (non-EMA) weights
          to keep resume behavior consistent with optimizer states.
        - For external/final saving (_internal_call=False), we swap in EMA diffusion weights
          temporarily so the saved model is the EMA version (diffusion transformer only).
        """
        if self.ema_diffusion is not None and not _internal_call:
            with self._swap_in_ema_diffusion_weights():
                return super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    def get_train_dataloader(self) -> DataLoader:
        """
        Get the training dataloader with proper distributed sampling support.

        This method handles two cases:
        1. With custom sampler (WeightedRandomSampler): Wraps it with distributed-aware
           sampling to ensure different GPUs get different data in multi-GPU training.
        2. Without custom sampler: Falls back to parent implementation which uses
           DistributedSampler by default.

        IMPORTANT: In distributed training, each GPU must receive different data samples.
        Without proper distributed sampling, all GPUs would sample the same prompts at
        the same step, which is a serious bug that reduces effective batch diversity.

        Returns:
            DataLoader for training
        """
        from torch.utils.data import DistributedSampler

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        # Check if we're in distributed training mode
        is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        # Use custom sampler if provided, otherwise use default behavior
        if self.train_sampler is not None:
            if is_distributed:
                # In distributed training, we need to ensure each GPU gets different samples.
                # We create a DistributedSampler that wraps the indices, and then use the
                # WeightedRandomSampler weights to influence the sampling within each GPU's subset.
                #
                # Strategy: Use DistributedSampler to split indices across GPUs, then create
                # a per-GPU weighted sampler for the local subset.

                world_size = dist.get_world_size()
                rank = dist.get_rank()

                # Get the weights from the original WeightedRandomSampler
                original_weights = list(self.train_sampler.weights)
                num_samples_total = len(original_weights)

                # Use DistributedSampler to get indices for this rank
                # This sampler will give us a deterministic subset of indices for each epoch
                dist_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,  # Shuffle indices across GPUs
                    drop_last=self.args.dataloader_drop_last,
                )

                # Create a wrapper sampler that combines distributed splitting with weighted sampling
                # Each GPU gets its own subset of indices, then applies weighted sampling within that subset
                from torch.utils.data import Sampler
                import torch

                class DistributedWeightedSampler(Sampler):
                    """
                    A sampler that combines DistributedSampler with WeightedRandomSampler.

                    In each epoch:
                    1. DistributedSampler splits indices across GPUs
                    2. Each GPU applies weighted sampling within its subset
                    """

                    def __init__(self, dataset, weights, num_samples_per_gpu, world_size, rank, replacement=True):
                        self.dataset = dataset
                        self.weights = torch.as_tensor(weights, dtype=torch.float64)
                        self.num_samples_per_gpu = num_samples_per_gpu
                        self.world_size = world_size
                        self.rank = rank
                        self.replacement = replacement
                        self.epoch = 0

                    def set_epoch(self, epoch):
                        """Set epoch for reproducibility across epochs."""
                        self.epoch = epoch

                    def __iter__(self):
                        # Set random seed based on epoch and rank for reproducibility
                        g = torch.Generator()
                        g.manual_seed(self.epoch * self.world_size + self.rank + 42)

                        # Generate weighted random samples for this GPU
                        # Each GPU samples independently with its own seed, ensuring different data
                        indices = torch.multinomial(
                            self.weights,
                            num_samples=self.num_samples_per_gpu,
                            replacement=self.replacement,
                            generator=g
                        ).tolist()

                        return iter(indices)

                    def __len__(self):
                        return self.num_samples_per_gpu

                # Calculate samples per GPU (accounting for batch size and drop_last)
                num_samples_per_gpu = len(train_dataset) // world_size
                if not self.args.dataloader_drop_last:
                    # If not dropping last, some GPUs may get one extra sample
                    if rank < len(train_dataset) % world_size:
                        num_samples_per_gpu += 1

                # Create the distributed weighted sampler
                combined_sampler = DistributedWeightedSampler(
                    train_dataset,
                    weights=original_weights,
                    num_samples_per_gpu=num_samples_per_gpu,
                    world_size=world_size,
                    rank=rank,
                    replacement=True,  # Weighted sampling typically uses replacement
                )

                # Store for epoch setting
                self._distributed_weighted_sampler = combined_sampler

                rank0_print(f"Using DistributedWeightedSampler: "
                            f"num_samples_per_gpu={num_samples_per_gpu}, world_size={world_size}")

                return DataLoader(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    sampler=combined_sampler,
                    collate_fn=data_collator,
                    drop_last=self.args.dataloader_drop_last,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                    persistent_workers=self.args.dataloader_persistent_workers if self.args.dataloader_num_workers > 0 else False,
                )
            else:
                # Single GPU mode: use the sampler directly
                return DataLoader(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    sampler=self.train_sampler,
                    collate_fn=data_collator,
                    drop_last=self.args.dataloader_drop_last,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                    persistent_workers=self.args.dataloader_persistent_workers if self.args.dataloader_num_workers > 0 else False,
                )

        # Fall back to parent implementation (uses DistributedSampler by default)
        return super().get_train_dataloader()

    def _log_model_parameters(self, model, output_dir: str):
        """
        Log trainable and non-trainable parameters to separate log files on rank 0.

        Args:
            model: The model to log parameters for
            output_dir: Output directory for log files
        """
        # Only log on rank 0
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        os.makedirs(output_dir, exist_ok=True)

        trainable_params_file = os.path.join(output_dir, "trainable_parameters.log")
        non_trainable_params_file = os.path.join(output_dir, "non_trainable_parameters.log")

        trainable_params = []
        non_trainable_params = []
        total_trainable = 0
        total_non_trainable = 0

        # Get the underlying model if wrapped
        unwrapped_model = model.module if hasattr(model, 'module') else model

        for name, param in unwrapped_model.named_parameters():
            num_params = param.numel()
            param_info = f"{name}: shape={list(param.shape)}, dtype={param.dtype}, numel={num_params}"
            if param.requires_grad:
                trainable_params.append(param_info)
                total_trainable += num_params
            else:
                non_trainable_params.append(param_info)
                total_non_trainable += num_params

        # Write trainable parameters log
        with open(trainable_params_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("TRAINABLE PARAMETERS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total trainable parameters: {total_trainable:,} ({total_trainable / 1e6:.2f}M)\n")
            f.write(f"Number of trainable parameter groups: {len(trainable_params)}\n")
            f.write("=" * 80 + "\n\n")
            for param_info in trainable_params:
                f.write(param_info + "\n")

        # Write non-trainable parameters log
        with open(non_trainable_params_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("NON-TRAINABLE (FROZEN) PARAMETERS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total non-trainable parameters: {total_non_trainable:,} ({total_non_trainable / 1e6:.2f}M)\n")
            f.write(f"Number of non-trainable parameter groups: {len(non_trainable_params)}\n")
            f.write("=" * 80 + "\n\n")
            for param_info in non_trainable_params:
                f.write(param_info + "\n")

        # Print summary to console
        total_params = total_trainable + total_non_trainable
        rank0_print("\n" + "=" * 80)
        rank0_print("MODEL PARAMETER SUMMARY")
        rank0_print("=" * 80)
        rank0_print(f"Trainable parameters:     {total_trainable:>15,} ({total_trainable / 1e6:>8.2f}M)")
        rank0_print(f"Non-trainable parameters: {total_non_trainable:>15,} ({total_non_trainable / 1e6:>8.2f}M)")
        rank0_print(f"Total parameters:         {total_params:>15,} ({total_params / 1e6:>8.2f}M)")
        rank0_print(f"Trainable ratio:          {100 * total_trainable / total_params:>14.2f}%")
        rank0_print("=" * 80)
        rank0_print(f"Trainable params log: {trainable_params_file}")
        rank0_print(f"Non-trainable params log: {non_trainable_params_file}")
        rank0_print("=" * 80 + "\n")

    def _create_reference_model(self, config_path: str, checkpoint_path: Optional[str]):
        """
        Create a frozen reference model for KL computation.

        Args:
            config_path: Path to model config
            checkpoint_path: Path to checkpoint

        Returns:
            Frozen reference model
        """
        rank0_print("Creating reference model...")
        config = Config.fromfile(config_path)
        ref_model = BUILDER.build(config.model)

        if checkpoint_path is not None:
            state_dict = guess_load_checkpoint(checkpoint_path)
            # Apply the same checkpoint key remapping as the policy model.
            # This is critical to ensure ref_model weights (especially LoRA) are loaded correctly.
            state_dict = _remap_ckpt_keys_for_qwen25vl(state_dict)
            missing, unexpected = ref_model.load_state_dict(state_dict, strict=False)
            # Keep logging lightweight to avoid dumping thousands of keys.
            if missing:
                rank0_print(f"[ref_ckpt] Missing keys: {len(missing)}")
            if unexpected:
                rank0_print(f"[ref_ckpt] Unexpected keys: {len(unexpected)}")

        ref_model = ref_model.bfloat16()

        # Freeze all parameters
        for param in ref_model.parameters():
            param.requires_grad = False

        # Always keep the reference model in eval mode to avoid dropout-induced KL noise.
        ref_model.eval()

        return ref_model

    def _setup_transforms(self):
        """Setup image transformation pipelines."""
        self.output_trsf = T.Compose([
            T.Lambda(lambda x: x.convert("RGB") if hasattr(x, 'convert') else x),
            T.Resize(1024, interpolation=InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(1024),
        ])

    def _set_signature_columns_if_needed(self):
        """Set required dataset columns."""
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "caption"]

    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Skip automatic tensor conversion."""
        return inputs

    def _compute_rewards(self, inputs: List[Dict], images: List[Any]) -> torch.Tensor:
        """
        Compute rewards from all reward functions.

        Supports two modes:
        1. Uniform mode (use_per_sample_reward_config=False): All samples use the same reward weights (1.0 for each)
        2. Per-sample mode (use_per_sample_reward_config=True): Each sample has its own reward weights
           from the 'reward_config' dict in the input.

        Args:
            inputs: Input batch. Each item may contain 'reward_config' dict with reward weights.
            images: Generated images (PIL Images)

        Returns:
            Tuple of (total_rewards, rewards_per_func)
        """
        device = self.accelerator.device
        num_images = len(images)
        num_reward_funcs = len(self.reward_funcs)
        rewards_per_func = torch.zeros(num_images, num_reward_funcs, device=device)

        # Extract captions/prompts
        captions = [ex.get("caption", ex.get("prompt", "")) for ex in inputs]

        # Debug logging for reward computation (only when JONB_DEBUG_OCR is set)
        _debug_ocr = os.environ.get("JONB_DEBUG_OCR", None) is not None
        if _debug_ocr and self.accelerator.is_main_process:
            env_log_dir = os.environ.get("LOG_DIR", ".")
            debug_path = os.path.join(env_log_dir, "debug_ocr_reward.log")
            with open(debug_path, "a") as f:
                f.write(f"\n=== _compute_rewards debug ===")
                f.write(f"\nStep: {self.state.global_step}")
                f.write(f"\nNumber of images: {num_images}")
                f.write(f"\nNumber of captions: {len(captions)}")
                f.write(f"\nCaption samples: {captions[:2]}")
                f.write(f"\nReward functions: {[f[0] for f in self.reward_funcs]}")
                f.write(f"\nrollout_n: {self.rollout_n}")
                f.write(f"\nuse_per_sample_reward_config: {self.use_per_sample_reward_config}")
                f.flush()

        # Optional debug cache for OCR-vLLM raw outputs (only rank 0 writes logs).
        _debug_ocr_vllm = os.environ.get("DEBUG_OCR_VLLM", "0") == "1"
        if _debug_ocr_vllm:
            # Reset per-step cache.
            self._debug_ocr_vllm_last = None

        # Compute raw rewards for each reward function
        for i, (func_name, processor, reward_func) in enumerate(self.reward_funcs):
            reward_start_time = time.perf_counter()

            if func_name == "jpeg_compressibility" or func_name == "jpeg_incompressibility":
                rewards_per_func[:, i] = reward_func(images)
                # Log individual reward timing
                reward_duration_ms = (time.perf_counter() - reward_start_time) * 1000
                self.phase_logger.reward_timing(self.state.global_step, func_name, reward_duration_ms)
            elif func_name in ["pickscore", "hps", "deqa", "image_reward", "aesthetic", "unifiedreward_sglang"]:
                # These functions need prompts repeated for each generation
                expanded_captions = [cap for cap in captions for _ in range(self.rollout_n)]
                result = reward_func(images, expanded_captions)
                if isinstance(result, dict) and "scores" in result:
                    scores = result["scores"]
                elif isinstance(result, (list, tuple)):
                    scores = result
                else:
                    scores = result
                rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32).to(device)
                # Log individual reward timing
                reward_duration_ms = (time.perf_counter() - reward_start_time) * 1000
                self.phase_logger.reward_timing(self.state.global_step, func_name, reward_duration_ms)
            elif func_name == "clip_sim":
                # CLIP similarity: text-image similarity using pre-loaded CLIP model
                # This avoids loading CLIP model every step (major memory leak fix)
                expanded_captions = [cap for cap in captions for _ in range(self.rollout_n)]
                try:
                    if self.clip_model is not None and self.clip_processor is not None:
                        # Use pre-loaded CLIP model (no repeated loading!)
                        text_inputs = self.clip_processor(text=expanded_captions, return_tensors="pt", padding=True, truncation=True, max_length=77)
                        image_inputs = self.clip_processor(images=images, return_tensors="pt")
                        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

                        with torch.no_grad():
                            text_features = self.clip_model.get_text_features(**text_inputs)
                            image_features = self.clip_model.get_image_features(**image_inputs)
                            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            similarity = (text_features * image_features).sum(dim=-1)

                        # Detach to ensure no computation graph is retained
                        rewards_per_func[:, i] = similarity.detach()

                        # Clean up intermediate tensors to prevent memory leak
                        del text_inputs, image_inputs, text_features, image_features, similarity
                        torch.cuda.empty_cache()
                    else:
                        rank0_print("Warning: clip_sim requested but CLIP model not pre-loaded")
                        rewards_per_func[:, i] = torch.zeros(num_images, device=device)
                except Exception as e:
                    rank0_print(f"Warning: clip_sim reward computation failed: {e}")
                    rewards_per_func[:, i] = torch.zeros(num_images, device=device)
                # Log individual reward timing
                reward_duration_ms = (time.perf_counter() - reward_start_time) * 1000
                self.phase_logger.reward_timing(self.state.global_step, func_name, reward_duration_ms)
            elif func_name == "unifiedreward_think":
                # UnifiedReward Think: pairwise comparison within each prompt group
                # This reward function compares images within the same prompt group
                # and computes win rates. Must be called per-group to maintain
                # the property that average win rate = 0.5 within each group.
                #
                # Data layout (with rollout_accumulation_steps):
                #   inputs = [p0, p1, ..., pM, p0, p1, ..., pM, ...] (repeated for each accumulation step)
                #   images = [p0_imgs(rollout_n), p1_imgs, ..., pM_imgs, p0_imgs, p1_imgs, ...]
                # So len(captions) = original_num_prompts * rollout_accumulation_steps
                # And num_images = len(captions) * rollout_n
                #
                # We need to group by (accumulation_step, prompt_idx), each group has rollout_n images.
                num_caption_entries = len(captions)  # = original_prompts * rollout_accumulation_steps
                group_rewards = []

                # Debug logging for unifiedreward_think (all ranks)
                _debug_ur = os.environ.get("DEBUG_UNIFIED_REWARD", "0") == "1"
                _rank = self.accelerator.process_index
                if _debug_ur:
                    env_log_dir = os.environ.get("LOG_DIR", ".")
                    debug_path = os.path.join(env_log_dir, f"debug_unifiedreward_think_rank{_rank}.log")
                    with open(debug_path, "a") as f:
                        f.write(f"\n\n=== UnifiedReward Think Debug (Rank {_rank}) ===")
                        f.write(f"\nStep: {self.state.global_step}")
                        f.write(f"\nnum_images: {num_images}")
                        f.write(f"\nnum_caption_entries: {num_caption_entries}")
                        f.write(f"\nrollout_n: {self.rollout_n}")
                        f.write(f"\nExpected: num_images == num_caption_entries * rollout_n: {num_images == num_caption_entries * self.rollout_n}")
                        f.write(f"\nSample captions[:3]: {captions[:3]}")
                        f.flush()

                for caption_idx in range(num_caption_entries):
                    # Each caption entry corresponds to rollout_n consecutive images
                    start_idx = caption_idx * self.rollout_n
                    end_idx = start_idx + self.rollout_n
                    group_images = images[start_idx:end_idx]
                    group_prompts = [captions[caption_idx]] * self.rollout_n

                    # Compute pairwise win rate within this group
                    group_result = reward_func(group_images, group_prompts)
                    # Detach and clone to CPU to prevent GPU memory accumulation
                    # Each group_result is shape (rollout_n,), storing on CPU saves GPU memory
                    group_result_detached = group_result.detach().cpu()
                    group_rewards.append(group_result_detached)

                    # Debug log for each group (all ranks)
                    if _debug_ur and caption_idx < 3:  # Only log first 3 groups
                        with open(debug_path, "a") as f:
                            f.write(f"\n  Group {caption_idx}: start={start_idx}, end={end_idx}, result_shape={group_result_detached.shape}, mean={group_result_detached.mean().item():.4f}")
                            f.write(f"\n    group_result values: {group_result_detached.tolist()}")
                            f.flush()

                    # Clean up group_result immediately to free GPU memory
                    del group_result, group_result_detached

                # Concatenate results from all groups (on CPU first, then move to GPU)
                # This avoids GPU memory fragmentation from many small allocations
                all_group_rewards_cpu = torch.cat(group_rewards)
                all_group_rewards = all_group_rewards_cpu.to(device)
                rewards_per_func[:, i] = all_group_rewards.detach()

                # Debug log final result (all ranks)
                if _debug_ur:
                    with open(debug_path, "a") as f:
                        f.write(f"\n  Final mean (Rank {_rank}): {all_group_rewards.mean().item():.4f}")
                        f.write(f"\n  All group means: {[g.mean().item() for g in group_rewards[:5]]}")
                        f.flush()

                # Clean up intermediate tensors to prevent memory leak
                del group_rewards, all_group_rewards_cpu, all_group_rewards
                torch.cuda.empty_cache()
                # Log individual reward timing
                reward_duration_ms = (time.perf_counter() - reward_start_time) * 1000
                self.phase_logger.reward_timing(self.state.global_step, func_name, reward_duration_ms)
            elif func_name in ["ocr", "ocr_vllm"]:
                # OCR reward: evaluate text rendering quality
                expanded_captions = [cap for cap in captions for _ in range(self.rollout_n)]

                # Debug log before OCR call (only when JONB_DEBUG_OCR is set)
                if _debug_ocr and self.accelerator.is_main_process:
                    env_log_dir = os.environ.get("LOG_DIR", ".")
                    debug_path = os.path.join(env_log_dir, "debug_ocr_reward.log")
                    with open(debug_path, "a") as f:
                        f.write(f"\n\n--- OCR Reward Computation ---")
                        f.write(f"\nfunc_name: {func_name}")
                        f.write(f"\nExpanded captions count: {len(expanded_captions)}")
                        f.write(f"\nExpanded captions: {expanded_captions}")
                        f.write(f"\nImages count: {len(images)}")
                        f.write(f"\nFirst image size: {images[0].size if images else 'N/A'}")
                        f.flush()

                try:
                    result = reward_func(images, expanded_captions)

                    # Debug log after OCR call (only when JONB_DEBUG_OCR is set)
                    if _debug_ocr and self.accelerator.is_main_process:
                        with open(debug_path, "a") as f:
                            f.write(f"\nOCR result type: {type(result)}")
                            f.write(f"\nOCR result: {result}")
                            f.flush()

                    # Capture OCR-vLLM raw outputs for later step-aligned logging.
                    if func_name == "ocr_vllm" and _debug_ocr_vllm and self.accelerator.is_main_process:
                        try:
                            from ..reward_evaluator import ocr_vllm as _ocr_vllm_mod
                            self._debug_ocr_vllm_last = {
                                "step": int(self.state.global_step),
                                "debug": _ocr_vllm_mod.get_last_debug(),
                            }
                        except Exception:
                            self._debug_ocr_vllm_last = None

                    if isinstance(result, dict) and "scores" in result:
                        scores = result["scores"]
                    elif isinstance(result, list):
                        scores = result
                    else:
                        # Try to convert to scores
                        scores = result

                    # Debug log scores (only when JONB_DEBUG_OCR is set)
                    if _debug_ocr and self.accelerator.is_main_process:
                        with open(debug_path, "a") as f:
                            f.write(f"\nExtracted scores type: {type(scores)}")
                            f.write(f"\nExtracted scores: {scores}")
                            f.flush()

                    if isinstance(scores, torch.Tensor):
                        rewards_per_func[:, i] = scores.to(device=device, dtype=torch.float32)
                    else:
                        rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32).to(device)

                    # Debug log final rewards (only when JONB_DEBUG_OCR is set)
                    if _debug_ocr and self.accelerator.is_main_process:
                        with open(debug_path, "a") as f:
                            f.write(f"\nFinal OCR rewards tensor: {rewards_per_func[:, i]}")
                            f.write(f"\nOCR rewards mean: {rewards_per_func[:, i].mean().item():.4f}")
                            f.write(f"\n--- OCR Reward Computation End ---\n")
                            f.flush()

                except Exception as e:
                    import traceback
                    error_msg = traceback.format_exc()
                    if _debug_ocr and self.accelerator.is_main_process:
                        with open(debug_path, "a") as f:
                            f.write(f"\n!!! OCR Reward ERROR !!!")
                            f.write(f"\nfunc_name: {func_name}")
                            f.write(f"\nError: {str(e)}")
                            f.write(f"\nTraceback: {error_msg}")
                            f.flush()
                    # Set zero rewards on error
                    rewards_per_func[:, i] = torch.zeros(num_images, device=device)
                # Log individual reward timing
                reward_duration_ms = (time.perf_counter() - reward_start_time) * 1000
                self.phase_logger.reward_timing(self.state.global_step, func_name, reward_duration_ms)
            else:
                # Unknown reward function, try generic call
                try:
                    expanded_captions = [cap for cap in captions for _ in range(self.rollout_n)]
                    result = reward_func(images, expanded_captions)
                    if isinstance(result, dict) and "scores" in result:
                        scores = result["scores"]
                    elif isinstance(result, (list, tuple)):
                        scores = list(result)
                    elif isinstance(result, torch.Tensor):
                        scores = result
                    else:
                        scores = result
                    rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32).to(device) if not isinstance(scores, torch.Tensor) else scores.to(device)
                except Exception as e:
                    rank0_print(f"Warning: reward function '{func_name}' failed: {e}")
                    rewards_per_func[:, i] = torch.zeros(num_images, device=device)
                # Log individual reward timing
                reward_duration_ms = (time.perf_counter() - reward_start_time) * 1000
                self.phase_logger.reward_timing(self.state.global_step, func_name, reward_duration_ms)

        # Apply per-sample reward weights if enabled
        if self.use_per_sample_reward_config:
            # Build weight matrix from per-sample reward_config
            # Shape: (num_images, num_reward_funcs)
            weight_matrix = torch.zeros(num_images, num_reward_funcs, device=device)

            for sample_idx, inp in enumerate(inputs):
                reward_config = inp.get("reward_config", {})
                # Expand for rollout_n images per sample
                for img_offset in range(self.rollout_n):
                    img_idx = sample_idx * self.rollout_n + img_offset
                    if img_idx >= num_images:
                        break
                    for func_name, weight in reward_config.items():
                        if func_name in self.reward_func_name_to_idx:
                            func_idx = self.reward_func_name_to_idx[func_name]
                            weight_matrix[img_idx, func_idx] = weight

            # Apply weights to raw rewards
            weighted_rewards = rewards_per_func * weight_matrix
            total_rewards = weighted_rewards.sum(dim=1)

            # Debug log
            if _debug_ocr and self.accelerator.is_main_process:
                env_log_dir = os.environ.get("LOG_DIR", ".")
                debug_path = os.path.join(env_log_dir, "debug_ocr_reward.log")
                with open(debug_path, "a") as f:
                    f.write(f"\n=== Per-sample reward weights ===")
                    f.write(f"\nweight_matrix shape: {weight_matrix.shape}")
                    f.write(f"\nweight_matrix[:4]: {weight_matrix[:4]}")
                    f.write(f"\nweighted_rewards[:4]: {weighted_rewards[:4]}")
                    f.flush()

            # Memory cleanup: delete intermediate tensors used only for weighted computation
            del weight_matrix, weighted_rewards
        else:
            # Uniform weights: sum all reward values
            total_rewards = rewards_per_func.sum(dim=1)

        # Debug log aggregated rewards (only when JONB_DEBUG_OCR is set)
        if _debug_ocr and self.accelerator.is_main_process:
            env_log_dir = os.environ.get("LOG_DIR", ".")
            debug_path = os.path.join(env_log_dir, "debug_ocr_reward.log")
            with open(debug_path, "a") as f:
                f.write(f"\n=== Final Rewards ===")
                f.write(f"\nrewards_per_func shape: {rewards_per_func.shape}")
                f.write(f"\nrewards_per_func: {rewards_per_func}")
                f.write(f"\ntotal_rewards: {total_rewards}")
                f.write(f"\ntotal_rewards mean: {total_rewards.mean().item():.4f}")
                f.write(f"\n=== _compute_rewards End ===\n")
                f.flush()

        # Detach all tensors to prevent computation graph retention and memory leak
        total_rewards = total_rewards.detach()
        rewards_per_func = rewards_per_func.detach()

        # Force CUDA memory cleanup after reward computation
        torch.cuda.empty_cache()

        # Return total rewards and per-func rewards (unweighted for logging)
        return total_rewards, rewards_per_func

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        rewards_per_func: torch.Tensor,
        inputs: List[Dict],
    ) -> torch.Tensor:
        """
        Compute advantages based on atrain_adv_type.

        Supports three modes:
        1. 'grpo' (default): Group-wise normalization
           advantage = (reward - group_mean) / group_std
        2. 'reinforcepp': Batch-wise normalization
           advantage = (reward - batch_mean) / batch_std
        3. 'gdpo': Each reward is group-normalized separately, then weighted sum,
           then batch normalization

        Args:
            rewards: Total rewards tensor, shape (num_images,)
            rewards_per_func: Per-function rewards tensor, shape (num_images, num_reward_funcs)
            inputs: Input batch (used for per-sample reward weights in GDPO mode)

        Returns:
            Advantages tensor, shape (num_images,)
        """
        device = rewards.device
        num_images = rewards.shape[0]

        if self.atrain_adv_type == "reinforcepp":
            # REINFORCE++: Batch-wise normalization
            # Simple global normalization across all samples in the batch
            mean_rewards = rewards.mean()
            std_rewards = rewards.std() + 1e-4
            advantages = (rewards - mean_rewards) / std_rewards

        elif self.atrain_adv_type == "gdpo":
            # GDPO: Each reward is group-normalized separately, then weighted sum, then batch norm
            # Reference: GDPO paper https://arxiv.org/abs/2601.05242
            num_reward_funcs = rewards_per_func.shape[1]

            # Step 1: Group-normalize each reward function separately
            all_reward_advantages = []
            for func_idx in range(num_reward_funcs):
                func_rewards = rewards_per_func[:, func_idx]  # shape (num_images,)

                # Reshape to groups: each group has rollout_n samples
                reshaped_func_rewards = func_rewards.view(-1, self.rollout_n)

                # Group-wise normalization
                group_means = reshaped_func_rewards.mean(dim=1, keepdim=True)  # (num_groups, 1)
                group_stds = reshaped_func_rewards.std(dim=1, keepdim=True) + 1e-4  # (num_groups, 1)
                normalized = (reshaped_func_rewards - group_means) / group_stds
                func_adv = normalized.view(-1)  # flatten back to (num_images,)

                all_reward_advantages.append(func_adv)

            # Step 2: Apply per-sample weights and combine
            if self.use_per_sample_reward_config:
                # Build weight matrix from per-sample reward_config
                weight_matrix = torch.zeros(num_images, num_reward_funcs, device=device)
                for sample_idx, inp in enumerate(inputs):
                    reward_config = inp.get("reward_config", {})
                    for img_offset in range(self.rollout_n):
                        img_idx = sample_idx * self.rollout_n + img_offset
                        if img_idx >= num_images:
                            break
                        for func_name, weight in reward_config.items():
                            if func_name in self.reward_func_name_to_idx:
                                func_idx_w = self.reward_func_name_to_idx[func_name]
                                weight_matrix[img_idx, func_idx_w] = weight

                # Weighted sum of per-reward advantages
                stacked_adv = torch.stack(all_reward_advantages, dim=1)  # (num_images, num_funcs)
                combined_advantages = (stacked_adv * weight_matrix).sum(dim=1)
            else:
                # Uniform weights: simple sum
                combined_advantages = torch.stack(all_reward_advantages, dim=0).sum(dim=0)

            # Step 3: Batch normalization on combined advantages
            bn_mean = combined_advantages.mean()
            bn_std = combined_advantages.std() + 1e-4
            advantages = (combined_advantages - bn_mean) / bn_std

        elif self.atrain_adv_type == "grpo_global_std":
            # GRPO with global std: Group mean + global std
            # Useful when per-group std is too noisy
            reshaped_rewards = rewards.view(-1, self.rollout_n)
            mean_rewards = reshaped_rewards.mean(dim=1).repeat_interleave(self.rollout_n)
            # Global std: compute std across all rewards in the batch
            std_rewards = rewards.std().expand_as(rewards) + 1e-4
            advantages = (rewards - mean_rewards) / std_rewards

        else:  # "grpo" (default)
            # GRPO: Group-wise normalization on aggregated rewards
            # Each group contains rollout_n samples from the same prompt
            reshaped_rewards = rewards.view(-1, self.rollout_n)
            mean_rewards = reshaped_rewards.mean(dim=1).repeat_interleave(self.rollout_n)
            # Per-group std: compute std within each group (default behavior)
            std_rewards = reshaped_rewards.std(dim=1).repeat_interleave(self.rollout_n) + 1e-4

            advantages = (rewards - mean_rewards) / std_rewards

        return advantages

    def _generate_images_with_trajectory(
        self,
        model,
        prompts: List[str],
        cfg_prompts: List[str],
    ):
        """
        Generate images and record trajectory for GRPO.

        When rollout_n > rollout_micro_batch_size, generates images in multiple
        batches to fit GPU memory, then concatenates results.

        Args:
            model: The DeepGen model
            prompts: Text prompts
            cfg_prompts: CFG (classifier-free guidance) prompts

        Returns:
            Tuple of (images, log_probs_trajectory, prev_latents, pred_latents, timesteps,
                      prev_sample_means_traj, sqrt_dts_traj)
            Note: prev_sample_means_traj and sqrt_dts_traj are None when grpo_guard is disabled.
        """
        device = self.accelerator.device
        dtype = torch.bfloat16 if self.args.bf16 else torch.float32
        num_prompts = len(prompts)

        # Calculate how many images to generate per forward pass
        # rollout_micro_batch_size controls GPU memory usage
        images_per_forward = min(self.rollout_micro_batch_size, self.rollout_n * num_prompts)
        total_images = self.rollout_n * num_prompts
        num_forward_passes = math.ceil(total_images / images_per_forward)

        # Expand prompts for all generations: each prompt repeated rollout_n times
        all_expanded_prompts = [p for p in prompts for _ in range(self.rollout_n)]
        all_expanded_cfg_prompts = [p for p in cfg_prompts for _ in range(self.rollout_n)]

        # Accumulate results across forward passes
        all_images = []
        all_log_probs_list = []
        all_prev_latents_list = []
        all_pred_latents_list = []
        all_timesteps_list = []
        # GRPO-Guard: Accumulate prev_sample_means and sqrt_dts
        all_prev_sample_means_list = []
        all_sqrt_dts_list = []

        for fwd_idx in range(num_forward_passes):
            start_idx = fwd_idx * images_per_forward
            end_idx = min(start_idx + images_per_forward, total_images)
            batch_expanded_prompts = all_expanded_prompts[start_idx:end_idx]
            batch_expanded_cfg_prompts = all_expanded_cfg_prompts[start_idx:end_idx]
            expanded_batch_size = len(batch_expanded_prompts)

            # Generate images for this batch
            images, log_probs_traj, prev_latents, pred_latents, timesteps, prev_sample_means, sqrt_dts = \
                self._generate_batch_with_trajectory(
                    model, batch_expanded_prompts, batch_expanded_cfg_prompts,
                    expanded_batch_size, device, dtype, num_prompts=num_prompts
                )

            all_images.extend(images)
            all_log_probs_list.append(log_probs_traj)
            all_prev_latents_list.append(prev_latents)
            all_pred_latents_list.append(pred_latents)
            all_timesteps_list.append(timesteps)
            # GRPO-Guard: Collect prev_sample_means and sqrt_dts
            if self.grpo_guard:
                all_prev_sample_means_list.append(prev_sample_means)
                all_sqrt_dts_list.append(sqrt_dts)

            # Clear CUDA cache after each forward pass to prevent memory accumulation
            torch.cuda.empty_cache()

        # Concatenate all batches
        log_probs_traj = torch.cat(all_log_probs_list, dim=0)
        prev_latents = torch.cat(all_prev_latents_list, dim=0)
        pred_latents = torch.cat(all_pred_latents_list, dim=0)
        timesteps = torch.cat(all_timesteps_list, dim=0)
        # GRPO-Guard: Concatenate prev_sample_means and sqrt_dts
        if self.grpo_guard:
            prev_sample_means_traj = torch.cat(all_prev_sample_means_list, dim=0)
            sqrt_dts_traj = torch.cat(all_sqrt_dts_list, dim=0)
        else:
            prev_sample_means_traj = None
            sqrt_dts_traj = None

        # Debug log
        if dist.is_initialized() and dist.get_rank() == 0:
            env_log_dir = os.environ.get("LOG_DIR", ".")
            debug_path = os.path.join(env_log_dir, "debug_trajectory.log")
            with open(debug_path, "a") as f:
                f.write(f"\n=== _generate_images_with_trajectory ===\n")
                f.write(f"num_prompts: {num_prompts}\n")
                f.write(f"rollout_n: {self.rollout_n}\n")
                f.write(f"rollout_micro_batch_size: {self.rollout_micro_batch_size}\n")
                f.write(f"total_images: {total_images}\n")
                f.write(f"num_forward_passes: {num_forward_passes}\n")
                f.write(f"final log_probs_traj shape: {log_probs_traj.shape}\n")
                if self.grpo_guard:
                    f.write(f"[GRPO-Guard] prev_sample_means_traj shape: {prev_sample_means_traj.shape}\n")
                    f.write(f"[GRPO-Guard] sqrt_dts_traj shape: {sqrt_dts_traj.shape}\n")
                f.flush()

        return all_images, log_probs_traj, prev_latents, pred_latents, timesteps, prev_sample_means_traj, sqrt_dts_traj

    def _generate_batch_with_trajectory(
        self,
        model,
        expanded_prompts: List[str],
        expanded_cfg_prompts: List[str],
        expanded_batch_size: int,
        device,
        dtype,
        num_prompts: int = None,
    ):
        """
        Generate a batch of images and record trajectory for GRPO.

        This is the inner function that generates a single batch of images.
        Called by _generate_images_with_trajectory which handles batching.

        Args:
            model: The DeepGen model
            expanded_prompts: Pre-expanded text prompts (already repeated for rollout_n)
            expanded_cfg_prompts: Pre-expanded CFG prompts
            expanded_batch_size: Number of images to generate in this batch
            device: Device to run on
            dtype: Data type for tensors
            num_prompts: Number of unique prompts (for init_same_noise support)

        Returns:
            Tuple of (images, log_probs_trajectory, prev_latents, pred_latents, timesteps,
                      prev_sample_means_traj, sqrt_dts_traj)
            Note: prev_sample_means_traj and sqrt_dts_traj are None when grpo_guard is disabled.
        """
        # Prepare text embeddings through LMM
        # Get the underlying model if wrapped by DeepSpeed/Accelerator
        unwrapped_model = model.module if hasattr(model, 'module') else model

        # Check if we need unconditional embeddings (only when cfg_scale > 0)
        cfg_scale = self.cfg_scale
        need_cfg = cfg_scale > 0

        if need_cfg:
            # Original path: compute both conditional and unconditional embeddings
            text_inputs = unwrapped_model.prepare_text2image_prompts(expanded_prompts + expanded_cfg_prompts)
            # Use GatheredParameters to collect meta_queries when using DeepSpeed ZeRO-3
            with GatheredParameters([unwrapped_model.meta_queries], modifier_rank=None):
                hidden_states = unwrapped_model.meta_queries[None].expand(2 * expanded_batch_size, unwrapped_model.num_queries, -1).clone()
            inputs = unwrapped_model.prepare_forward_input(query_embeds=hidden_states, **text_inputs)

            with torch.no_grad():
                output = unwrapped_model.llm(**inputs, return_dict=True, output_hidden_states=True)
                # Pass full hidden_states tuple to llm2dit for multi-layer fusion support
                pooled_out, seq_out = unwrapped_model.llm2dit(output.hidden_states)

            # Split into conditional and unconditional
            pooled_cond = pooled_out[:expanded_batch_size]
            pooled_uncond = pooled_out[expanded_batch_size:]
            seq_cond = seq_out[:expanded_batch_size]
            seq_uncond = seq_out[expanded_batch_size:]
        else:
            # cfg_scale=0: only compute conditional embeddings (skip unconditional)
            text_inputs = unwrapped_model.prepare_text2image_prompts(expanded_prompts)
            with GatheredParameters([unwrapped_model.meta_queries], modifier_rank=None):
                hidden_states = unwrapped_model.meta_queries[None].expand(expanded_batch_size, unwrapped_model.num_queries, -1).clone()
            inputs = unwrapped_model.prepare_forward_input(query_embeds=hidden_states, **text_inputs)

            with torch.no_grad():
                output = unwrapped_model.llm(**inputs, return_dict=True, output_hidden_states=True)
                # Pass full hidden_states tuple to llm2dit for multi-layer fusion support
                pooled_cond, seq_cond = unwrapped_model.llm2dit(output.hidden_states)

            # No unconditional embeddings needed
            pooled_uncond = None
            seq_uncond = None

        # Initialize latents
        latent_channels = unwrapped_model.transformer.config.in_channels
        latent_height = self.image_height // 8
        latent_width = self.image_width // 8

        # Generate initial noise
        if self.init_same_noise and num_prompts is not None:
            # When init_same_noise is enabled, all images from the same prompt share the same initial noise
            # Generate one noise per prompt, then repeat for rollout_n
            images_per_prompt = expanded_batch_size // num_prompts
            base_latents = torch.randn(
                (num_prompts, latent_channels, latent_height, latent_width),
                device=device,
                dtype=dtype,
            )
            # Repeat each prompt's noise for all its generated images
            latents = base_latents.repeat_interleave(images_per_prompt, dim=0)
        else:
            # Default: independent random noise for each image
            latents = torch.randn(
                (expanded_batch_size, latent_channels, latent_height, latent_width),
                device=device,
                dtype=dtype,
            )

        # Setup scheduler - use model's test_scheduler to match original pipeline
        scheduler = unwrapped_model.test_scheduler

        # Calculate dynamic shifting mu based on image size (matching original pipeline)
        scheduler_kwargs = {}
        if scheduler.config.get("use_dynamic_shifting", None):
            patch_size = unwrapped_model.transformer.config.patch_size
            image_seq_len = (latent_height // patch_size) * (latent_width // patch_size)

            # Calculate shift using the same formula as original pipeline
            base_seq_len = scheduler.config.get("base_image_seq_len", 256)
            max_seq_len = scheduler.config.get("max_image_seq_len", 4096)
            base_shift = scheduler.config.get("base_shift", 0.5)
            max_shift = scheduler.config.get("max_shift", 1.16)

            m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            b = base_shift - m * base_seq_len
            mu = image_seq_len * m + b
            scheduler_kwargs["mu"] = mu

        scheduler.set_timesteps(self.diffusion_config["num_inference_steps"], device=device, **scheduler_kwargs)

        # Record trajectory for log prob computation
        # Store on CPU to save GPU memory (convert to fp32 since CPU doesn't support bf16)
        all_prev_latents = []  # Will store on CPU as fp32
        all_pred_latents = []  # Will store on CPU as fp32
        all_timesteps = []     # Will store on CPU
        all_log_probs = []     # Will store on CPU as fp32
        # GRPO-Guard: Store prev_sample_mean and sqrt_dt for RatioNorm
        all_prev_sample_means = []  # Will store on CPU as fp32
        all_sqrt_dts = []           # Will store on CPU as fp32

        # NOTE: To exactly match flow_grpo's GRPO-Guard "RatioNorm" behavior,
        # we disable optional log_prob clamping when ATRAIN_GRPOGUARD_PAPER_IMPL=1.
        paper_impl = self.grpo_guard and os.environ.get("ATRAIN_GRPOGUARD_PAPER_IMPL", "0") == "1"
        clamp_log_prob = self._clamp_log_prob and not paper_impl

        # Diffusion sampling loop with SDE
        for i, t in enumerate(scheduler.timesteps):
            # Expand timestep
            timestep = t.expand(expanded_batch_size)

            # Record current state - move to CPU immediately to save GPU memory
            # Convert to fp32 since CPU doesn't support native bf16 operations
            all_prev_latents.append(latents.clone().float().cpu())

            # Get model prediction
            with torch.no_grad():
                # Conditional prediction (always needed)
                model_pred_cond = unwrapped_model.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=seq_cond,
                    pooled_projections=pooled_cond,
                    timestep=timestep,
                    return_dict=False,
                )[0]

                if need_cfg:
                    # CFG enabled: compute unconditional prediction and apply CFG
                    model_pred_uncond = unwrapped_model.transformer(
                        hidden_states=latents,
                        encoder_hidden_states=seq_uncond,
                        pooled_projections=pooled_uncond,
                        timestep=timestep,
                        return_dict=False,
                    )[0]
                    # Apply CFG formula
                    model_pred = model_pred_uncond + cfg_scale * (model_pred_cond - model_pred_uncond)
                else:
                    # cfg_scale=0: use conditional prediction directly
                    model_pred = model_pred_cond

            # SDE step following flow_grpo_step implementation
            sigma = scheduler.sigmas[i]
            sigma_prev = scheduler.sigmas[i + 1] if i < len(scheduler.sigmas) - 1 else torch.tensor(0.0, device=device)
            sigma_max = scheduler.sigmas[1].item()
            dt = sigma_prev - sigma  # negative

            # Compute std_dev_t and prev_sample_mean based on sampler type
            eta = self.sde_eta  # Can be tuned, higher = more diversity

            if self.atrain_sde_sampler == "cps_sde":
                # Flow-CPS (Coefficients-Preserving Sampling)
                # Reference: https://arxiv.org/abs/2509.05952
                std_dev_t = sigma_prev * math.sin(eta * math.pi / 2)
                # pred_original_sample = latents - sigma * model_pred (predicted x_0)
                pred_original_sample = latents - sigma * model_pred
                # noise_estimate = latents + model_pred * (1 - sigma) (predicted x_1)
                noise_estimate = latents + model_pred * (1 - sigma)
                # prev_sample_mean = x_0 * (1 - sigma_prev) + x_1 * sqrt(sigma_prev^2 - std_dev_t^2)
                # Note: No clamp to match original FlowCPS exactly
                prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)
                # For CPS, noise scale is just std_dev_t
                noise_scale = std_dev_t
                # Add SDE noise for exploration
                variance_noise = torch.randn_like(latents)
                latents = prev_sample_mean + noise_scale * variance_noise
            elif self.atrain_sde_sampler == "dance_sde":
                # Dance-SDE
                # Reference: Flow-Factory flow_match_euler_discrete.py
                # pred_original_sample = latents - sigma * model_pred (predicted x_0)
                pred_original_sample = latents - sigma * model_pred
                # std_dev_t = eta * sqrt(-dt)
                std_dev_t = eta * torch.sqrt(-1 * dt)
                # log_term correction: 0.5 * eta^2 * (latents - x_0 * (1 - sigma)) / sigma^2
                log_term = 0.5 * (eta ** 2) * (latents - pred_original_sample * (1 - sigma)) / (sigma ** 2 + 1e-8)
                # prev_sample_mean = latents + (model_pred + log_term) * dt
                prev_sample_mean = latents + (model_pred + log_term) * dt
                noise_scale = std_dev_t
                # Add SDE noise for exploration
                variance_noise = torch.randn_like(latents)
                latents = prev_sample_mean + noise_scale * variance_noise
            else:
                # Flow-SDE (Standard flow matching SDE) - flowgrpo_sde
                sigma_safe = torch.where(sigma == 1, torch.tensor(sigma_max, device=device), sigma)
                std_dev_t = torch.sqrt(sigma / (1 - sigma_safe)) * eta
                # SDE mean prediction with correction terms
                prev_sample_mean = (
                    latents * (1 + std_dev_t**2 / (2 * sigma) * dt)
                    + model_pred * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
                )
                # For Flow-SDE, noise scale is std_dev_t * sqrt(-dt)
                noise_scale = std_dev_t * torch.sqrt(-1 * dt)
                # Add SDE noise for exploration
                variance_noise = torch.randn_like(latents)
                latents = prev_sample_mean + noise_scale * variance_noise

            # Debug logging for diversity analysis (all steps)
            self.diversity_logger.log(
                0,  # step=0 for rollout phase
                sde_sampler=self.atrain_sde_sampler,
                sde_eta=eta,
                sde_step=i,
                sigma=f"{sigma.item():.4f}",
                dt=f"{dt.item():.4f}",
                std_dev_t=f"{std_dev_t.item() if isinstance(std_dev_t, torch.Tensor) else std_dev_t:.4f}",
                noise_scale=f"{noise_scale.item() if isinstance(noise_scale, torch.Tensor) else noise_scale:.4f}",
                variance_noise_std=f"{variance_noise.std().item():.4f}",
                latents_std=f"{latents.std().item():.4f}",
                latents_diff_from_mean=f"{(latents - prev_sample_mean).abs().mean().item():.6f}",
            )

            # DEBUG_CPS_SDE: Write detailed debug log to output folder
            if os.environ.get("DEBUG_CPS_SDE", "0") == "1" and self.atrain_sde_sampler == "cps_sde":
                debug_log_dir = os.path.join(self.args.output_dir, "cps_sde_debug")
                os.makedirs(debug_log_dir, exist_ok=True)
                debug_log_file = os.path.join(debug_log_dir, f"rollout_rank{dist.get_rank() if dist.is_initialized() else 0}.log")
                with open(debug_log_file, "a") as f:
                    f.write(f"=== Rollout Step {i} ===\n")
                    f.write(f"  sigma: {sigma.item():.6f}\n")
                    f.write(f"  sigma_prev: {sigma_prev.item():.6f}\n")
                    f.write(f"  dt: {dt.item():.6f}\n")
                    f.write(f"  eta (sde_eta): {eta:.4f}\n")
                    f.write(f"  std_dev_t = sigma_prev * sin(eta * pi / 2) = {std_dev_t.item() if isinstance(std_dev_t, torch.Tensor) else std_dev_t:.6f}\n")
                    f.write(f"  noise_scale: {noise_scale.item() if isinstance(noise_scale, torch.Tensor) else noise_scale:.6f}\n")
                    f.write(f"  pred_original_sample (x0) mean/std: {pred_original_sample.mean().item():.4f} / {pred_original_sample.std().item():.4f}\n")
                    f.write(f"  noise_estimate (x1) mean/std: {noise_estimate.mean().item():.4f} / {noise_estimate.std().item():.4f}\n")
                    # Compute sigma_prev^2 - std_dev_t^2 for debugging
                    inner_sqrt_val = (sigma_prev**2 - std_dev_t**2).item() if isinstance(sigma_prev, torch.Tensor) else (sigma_prev**2 - std_dev_t**2)
                    f.write(f"  sigma_prev^2 - std_dev_t^2: {inner_sqrt_val:.10f}\n")
                    f.write(f"  prev_sample_mean mean/std: {prev_sample_mean.mean().item():.4f} / {prev_sample_mean.std().item():.4f}\n")
                    f.write(f"  variance_noise std: {variance_noise.std().item():.4f}\n")
                    f.write(f"  latents (after step) mean/std: {latents.mean().item():.4f} / {latents.std().item():.4f}\n")
                    f.write(f"  model_pred mean/std: {model_pred.mean().item():.4f} / {model_pred.std().item():.4f}\n")
                    f.write("\n")

            # Record - move to CPU immediately to save GPU memory
            all_pred_latents.append(latents.clone().float().cpu())
            all_timesteps.append(timestep.cpu())

            # Compute log prob for this step
            # Need to use GPU tensors for computation, then move result to CPU
            if self.grpo_guard:
                # GRPO-Guard: Get prev_sample_mean and sqrt_dt for RatioNorm
                _, log_prob, prev_sample_mean_step, std_dev_t_step, sqrt_dt_step = sde_step_with_logprob(
                    scheduler, model_pred, timestep,
                    all_prev_latents[-1].to(device=device, dtype=dtype), latents,
                    eta=self.sde_eta,
                    return_sqrt_dt=True,
                    sampler_type=self.atrain_sde_sampler,
                    clamp_log_prob=clamp_log_prob,
                )
                all_prev_sample_means.append(prev_sample_mean_step.float().cpu())
                all_sqrt_dts.append(sqrt_dt_step.float().cpu())
            else:
                _, log_prob, _, _ = sde_step_with_logprob(
                    scheduler, model_pred, timestep,
                    all_prev_latents[-1].to(device=device, dtype=dtype), latents,
                    eta=self.sde_eta,
                    sampler_type=self.atrain_sde_sampler,
                    clamp_log_prob=clamp_log_prob,
                )
            all_log_probs.append(log_prob.float().cpu())

        # Diversity analysis after sampling loop
        final_latents_cpu = latents.clone().float().cpu()
        num_samples = final_latents_cpu.shape[0]
        if num_samples > 1:
            # Compute pairwise L2 distance between samples (for diversity measure)
            flat_latents = final_latents_cpu.view(num_samples, -1)
            mean_latent = flat_latents.mean(dim=0, keepdim=True)
            variance_from_mean = ((flat_latents - mean_latent) ** 2).sum(dim=1).sqrt().mean()
            self.diversity_logger.log(
                0,
                final_latents_variance_from_mean=f"{variance_from_mean.item():.4f}",
                final_latents_per_sample_std=f"{final_latents_cpu.std(dim=0).mean().item():.4f}",
                total_log_prob_sum=f"{sum([lp.sum().item() for lp in all_log_probs]):.2f}",
            )

        # Decode final latents to images
        with torch.no_grad():
            images_tensor = unwrapped_model.latents_to_pixels(latents)

        # Convert to PIL images
        images_tensor = rearrange(images_tensor, "b c h w -> b h w c")
        images_tensor = torch.clamp(127.5 * images_tensor + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

        from PIL import Image
        images = [Image.fromarray(img) for img in images_tensor]

        # Stack trajectories on CPU with shape (batch_size, num_steps, ...)
        # This allows efficient per-timestep access during training
        num_steps = len(all_prev_latents)

        # Stack along dim=1 to get (batch_size, num_steps, ...) shape
        # all_prev_latents[i] has shape (batch_size, C, H, W)
        prev_latents = torch.stack(all_prev_latents, dim=1)  # (batch_size, num_steps, C, H, W)
        pred_latents = torch.stack(all_pred_latents, dim=1)  # (batch_size, num_steps, C, H, W)
        # all_timesteps[i] has shape (batch_size,)
        timesteps = torch.stack(all_timesteps, dim=1)  # (batch_size, num_steps)
        # all_log_probs[i] has shape (batch_size,)
        log_probs_traj = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps)

        # GRPO-Guard: Stack prev_sample_mean and sqrt_dt trajectories
        if self.grpo_guard:
            prev_sample_means_traj = torch.stack(all_prev_sample_means, dim=1)  # (batch_size, num_steps, C, H, W)
            sqrt_dts_traj = torch.stack(all_sqrt_dts, dim=1)  # (batch_size, num_steps, 1, 1, 1)
        else:
            prev_sample_means_traj = None
            sqrt_dts_traj = None

        # Clean up intermediate lists to prevent memory leak
        del all_prev_latents, all_pred_latents, all_timesteps, all_log_probs
        if self.grpo_guard:
            del all_prev_sample_means, all_sqrt_dts
        del latents, images_tensor, final_latents_cpu
        del pooled_cond, seq_cond
        if need_cfg:
            del pooled_uncond, seq_uncond
        del text_inputs, hidden_states, inputs, output

        # Force CUDA memory cleanup after rollout
        torch.cuda.empty_cache()

        # Return trajectory data on CPU with shape (batch_size, num_steps, ...)
        # This saves GPU memory and allows per-timestep processing during training
        return images, log_probs_traj, prev_latents, pred_latents, timesteps, prev_sample_means_traj, sqrt_dts_traj

    def _compute_text_embeddings_batched(
        self,
        model,
        prompts: List[str],
        cfg_prompts: List[str],
        micro_batch_size: int = None,
        no_grad: bool = False,
    ):
        """
        Compute text embeddings with micro-batch processing to reduce GPU memory usage.

        This function processes LLM forward pass in micro-batches when the total batch
        size is larger than micro_batch_size.

        Args:
            model: The DeepGen model (unwrapped)
            prompts: Text prompts for conditional generation
            cfg_prompts: CFG prompts for unconditional generation
            micro_batch_size: Number of samples per micro-batch. If None, uses self.atrain_micro_batch_size.
            no_grad: Whether to compute without gradients (for reference model)

        Returns:
            Tuple of (pooled_cond, pooled_uncond, seq_cond, seq_uncond)
            When cfg_scale=0, pooled_uncond and seq_uncond will be None
        """
        if micro_batch_size is None:
            micro_batch_size = self.atrain_micro_batch_size

        batch_size = len(prompts)

        # Check if we need unconditional embeddings (only when cfg_scale > 0)
        need_cfg = self.cfg_scale > 0

        if need_cfg:
            # For LLM, we process both cond and uncond together, so total is 2 * batch_size
            total_prompts = prompts + cfg_prompts
            total_size = len(total_prompts)
        else:
            # cfg_scale=0: only compute conditional embeddings
            total_prompts = prompts
            total_size = len(total_prompts)

        # Get the underlying model if wrapped by DeepSpeed/Accelerator
        unwrapped_model = model.module if hasattr(model, 'module') else model

        # Calculate number of micro-batches
        # Note: micro_batch_size refers to the number of images
        if need_cfg:
            # For LLM we have 2x (cond + uncond), so we use 2 * micro_batch_size
            llm_micro_batch_size = 2 * micro_batch_size
        else:
            llm_micro_batch_size = micro_batch_size
        num_micro_batches = math.ceil(total_size / llm_micro_batch_size)

        def _compute_embeddings():
            if num_micro_batches > 1:
                # Process LLM forward in micro-batches
                pooled_list = []
                seq_list = []

                for mb_idx in range(num_micro_batches):
                    start_idx = mb_idx * llm_micro_batch_size
                    end_idx = min(start_idx + llm_micro_batch_size, total_size)
                    mb_prompts = total_prompts[start_idx:end_idx]
                    mb_size = len(mb_prompts)

                    # Prepare text inputs for this micro-batch
                    mb_text_inputs = unwrapped_model.prepare_text2image_prompts(mb_prompts)

                    # Get meta_queries for this micro-batch
                    with GatheredParameters([unwrapped_model.meta_queries], modifier_rank=None):
                        mb_hidden_states = unwrapped_model.meta_queries[None].expand(
                            mb_size, unwrapped_model.num_queries, -1
                        ).clone()

                    # Prepare forward input
                    embed_weight = unwrapped_model.llm.get_input_embeddings().weight
                    with GatheredParameters([embed_weight], modifier_rank=None):
                        mb_inputs = unwrapped_model.prepare_forward_input(
                            query_embeds=mb_hidden_states, **mb_text_inputs
                        )

                    # LLM forward
                    mb_output = unwrapped_model.llm(**mb_inputs, return_dict=True, output_hidden_states=True)

                    # Convert to DiT format (pass full hidden_states for multi-layer fusion support)
                    mb_pooled, mb_seq = unwrapped_model.llm2dit(mb_output.hidden_states)
                    pooled_list.append(mb_pooled)
                    seq_list.append(mb_seq)

                # Concatenate all micro-batch results
                pooled_out = torch.cat(pooled_list, dim=0)

                # Handle variable sequence lengths by padding to max length
                # seq_list elements have shape (mb_size, seq_len, hidden_dim)
                max_seq_len = max(seq.shape[1] for seq in seq_list)
                padded_seq_list = []
                for seq in seq_list:
                    if seq.shape[1] < max_seq_len:
                        # Pad with zeros on the sequence dimension
                        pad_len = max_seq_len - seq.shape[1]
                        padding = torch.zeros(
                            seq.shape[0], pad_len, seq.shape[2],
                            dtype=seq.dtype, device=seq.device
                        )
                        seq = torch.cat([seq, padding], dim=1)
                    padded_seq_list.append(seq)
                seq_out = torch.cat(padded_seq_list, dim=0)
            else:
                # Original path: process entire batch at once
                text_inputs = unwrapped_model.prepare_text2image_prompts(total_prompts)
                with GatheredParameters([unwrapped_model.meta_queries], modifier_rank=None):
                    hidden_states = unwrapped_model.meta_queries[None].expand(
                        total_size, unwrapped_model.num_queries, -1
                    ).clone()
                embed_weight = unwrapped_model.llm.get_input_embeddings().weight
                with GatheredParameters([embed_weight], modifier_rank=None):
                    inputs_emb = unwrapped_model.prepare_forward_input(query_embeds=hidden_states, **text_inputs)
                output = unwrapped_model.llm(**inputs_emb, return_dict=True, output_hidden_states=True)
                # Pass full hidden_states tuple to llm2dit for multi-layer fusion support
                pooled_out, seq_out = unwrapped_model.llm2dit(output.hidden_states)

            # Split into conditional and unconditional (if needed)
            if need_cfg:
                pooled_cond = pooled_out[:batch_size]
                pooled_uncond = pooled_out[batch_size:]
                seq_cond = seq_out[:batch_size]
                seq_uncond = seq_out[batch_size:]
            else:
                # cfg_scale=0: no unconditional embeddings
                pooled_cond = pooled_out
                pooled_uncond = None
                seq_cond = seq_out
                seq_uncond = None

            return pooled_cond, pooled_uncond, seq_cond, seq_uncond

        if no_grad:
            with torch.no_grad():
                return _compute_embeddings()
        else:
            return _compute_embeddings()

    def _compute_diffusion_loss(
        self,
        model,
        prompts: List[str],
        cfg_prompts: List[str],
        prev_latents: torch.Tensor,
        pred_latents: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_cond: torch.Tensor = None,
        pooled_uncond: torch.Tensor = None,
        seq_cond: torch.Tensor = None,
        seq_uncond: torch.Tensor = None,
        micro_batch_size: int = None,
    ):
        """
        Compute diffusion model predictions for a single timestep.

        Supports micro-batch processing to reduce GPU memory usage. When micro_batch_size
        is specified and smaller than batch_size, the transformer forward pass is split
        into multiple micro-batches.

        Args:
            model: Model to use
            prompts: Text prompts (used only if embeddings not provided)
            cfg_prompts: CFG prompts (used only if embeddings not provided)
            prev_latents: Previous latent states (batch_size, C, H, W)
            pred_latents: Predicted latent states (batch_size, C, H, W)
            timesteps: Timesteps (batch_size,)
            pooled_cond: Pre-computed pooled conditioning (optional)
            pooled_uncond: Pre-computed pooled unconditioning (optional)
            seq_cond: Pre-computed sequence conditioning (optional)
            seq_uncond: Pre-computed sequence unconditioning (optional)
            micro_batch_size: Number of samples per micro-batch for transformer forward.
                             If None, uses self.atrain_micro_batch_size.

        Returns:
            Model prediction tensor
        """
        device = self.accelerator.device
        batch_size = len(prompts)

        # Use provided micro_batch_size or fallback to default
        if micro_batch_size is None:
            micro_batch_size = self.atrain_micro_batch_size

        # Get the underlying model if wrapped by DeepSpeed/Accelerator
        unwrapped_model = model.module if hasattr(model, 'module') else model

        # Compute embeddings only if not provided
        if pooled_cond is None or seq_cond is None:
            text_inputs = unwrapped_model.prepare_text2image_prompts(prompts + cfg_prompts)
            # Use GatheredParameters to collect meta_queries when using DeepSpeed ZeRO-3
            with GatheredParameters([unwrapped_model.meta_queries], modifier_rank=None):
                hidden_states = unwrapped_model.meta_queries[None].expand(2 * batch_size, unwrapped_model.num_queries, -1).clone()
            # Gather embedding layer weights for DeepSpeed ZeRO-3 compatibility
            embed_weight = unwrapped_model.llm.get_input_embeddings().weight
            with GatheredParameters([embed_weight], modifier_rank=None):
                inputs = unwrapped_model.prepare_forward_input(query_embeds=hidden_states, **text_inputs)

            output = unwrapped_model.llm(**inputs, return_dict=True, output_hidden_states=True)
            # Pass full hidden_states tuple to llm2dit for multi-layer fusion support
            pooled_out, seq_out = unwrapped_model.llm2dit(output.hidden_states)

            # Split embeddings
            pooled_cond = pooled_out[:batch_size]
            pooled_uncond = pooled_out[batch_size:]
            seq_cond = seq_out[:batch_size]
            seq_uncond = seq_out[batch_size:]

        # Calculate number of micro-batches for transformer forward
        num_micro_batches = math.ceil(batch_size / micro_batch_size)

        # Process transformer forward in micro-batches if needed
        if num_micro_batches > 1:
            # Collect predictions from all micro-batches
            model_pred_cond_list = []
            model_pred_uncond_list = []

            for mb_idx in range(num_micro_batches):
                start_idx = mb_idx * micro_batch_size
                end_idx = min(start_idx + micro_batch_size, batch_size)

                # Slice inputs for this micro-batch
                mb_prev_latents = prev_latents[start_idx:end_idx]
                mb_timesteps = timesteps[start_idx:end_idx]
                mb_pooled_cond = pooled_cond[start_idx:end_idx]
                mb_pooled_uncond = pooled_uncond[start_idx:end_idx]
                mb_seq_cond = seq_cond[start_idx:end_idx]
                mb_seq_uncond = seq_uncond[start_idx:end_idx]

                # Conditional prediction for micro-batch
                mb_pred_cond = unwrapped_model.transformer(
                    hidden_states=mb_prev_latents,
                    encoder_hidden_states=mb_seq_cond,
                    pooled_projections=mb_pooled_cond,
                    timestep=mb_timesteps,
                    return_dict=False,
                )[0]
                model_pred_cond_list.append(mb_pred_cond)

                # Unconditional prediction for micro-batch
                mb_pred_uncond = unwrapped_model.transformer(
                    hidden_states=mb_prev_latents,
                    encoder_hidden_states=mb_seq_uncond,
                    pooled_projections=mb_pooled_uncond,
                    timestep=mb_timesteps,
                    return_dict=False,
                )[0]
                model_pred_uncond_list.append(mb_pred_uncond)

            # Concatenate all micro-batch predictions
            model_pred_cond = torch.cat(model_pred_cond_list, dim=0)
            model_pred_uncond = torch.cat(model_pred_uncond_list, dim=0)
        else:
            # Original path: process entire batch at once
            model_pred_cond = unwrapped_model.transformer(
                hidden_states=prev_latents,
                encoder_hidden_states=seq_cond,
                pooled_projections=pooled_cond,
                timestep=timesteps,
                return_dict=False,
            )[0]

            model_pred_uncond = unwrapped_model.transformer(
                hidden_states=prev_latents,
                encoder_hidden_states=seq_uncond,
                pooled_projections=pooled_uncond,
                timestep=timesteps,
                return_dict=False,
            )[0]

        # Apply CFG
        cfg_scale = self.diffusion_config["guidance_scale"]
        model_pred = model_pred_uncond + cfg_scale * (model_pred_cond - model_pred_uncond)

        return model_pred

    def _log_step(self, images, prompts_text, advantages):
        """Log training samples."""
        global_step = self.state.global_step

        if self.log_images_interval <= 0 or global_step % self.log_images_interval != 0:
            return

        device_id = str(self.accelerator.device).replace(":", "")
        log_dir = os.path.join(self.log_dir, f"step_{global_step}")
        os.makedirs(log_dir, exist_ok=True)

        text_content = f"Prompt: {prompts_text[0]}"

        log_file = os.path.join(log_dir, f"step_{global_step}_{device_id}.txt")
        if os.path.exists(log_file):
            return

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(text_content)

        # Optional: write OCR-vLLM raw outputs aligned with saved image filenames.
        _debug_ocr_vllm = os.environ.get("DEBUG_OCR_VLLM", "0") == "1"
        debug_file = os.path.join(log_dir, f"ocr_vllm_debug_step_{global_step}_{device_id}.txt")
        debug_text_lines = []
        if _debug_ocr_vllm and self.accelerator.is_main_process:
            try:
                last = getattr(self, "_debug_ocr_vllm_last", None)
                if isinstance(last, dict) and last.get("step") == int(global_step):
                    dbg = last.get("debug") or {}
                    records = dbg.get("records") if isinstance(dbg, dict) else None
                    if isinstance(records, list):
                        # Index by idx for quick lookup
                        by_idx = {}
                        for rec in records:
                            if isinstance(rec, dict) and rec.get("idx") is not None:
                                by_idx[int(rec["idx"])] = rec
                        debug_text_lines.append(f"step: {int(global_step)}")
                        debug_text_lines.append(f"device_id: {device_id}")
                        debug_text_lines.append(f"api_url: {dbg.get('api_url')}")
                        debug_text_lines.append(f"model: {dbg.get('model')}")
                        debug_text_lines.append(f"temperature: {dbg.get('temperature')}")
                        debug_text_lines.append(f"max_tokens: {dbg.get('max_tokens')}")
                        debug_text_lines.append(f"prompt_group: {prompts_text[0] if prompts_text else ''}")
                        debug_text_lines.append("")
                        # Save only the same images that _log_step saves (idx 0..rollout_n-1).
                        for idx in range(min(self.rollout_n, len(images))):
                            advantage = advantages[idx] if idx < len(advantages) else 0.0
                            img_name = f"step_{global_step}_{device_id}_{advantage:.4f}_{idx}.jpg"
                            rec = by_idx.get(idx, {})
                            debug_text_lines.append("=" * 80)
                            debug_text_lines.append(f"idx: {idx}")
                            debug_text_lines.append(f"image_file: {img_name}")
                            debug_text_lines.append(f"image_path: {os.path.join(log_dir, img_name)}")
                            try:
                                adv_f = float(advantage)
                            except Exception:
                                adv_f = advantage
                            debug_text_lines.append(f"advantage: {adv_f}")
                            debug_text_lines.append(f"prompt: {rec.get('prompt', prompts_text[0] if prompts_text else '')}")
                            debug_text_lines.append(f"score: {rec.get('score')}")
                            debug_text_lines.append(f"parsed_json: {rec.get('parsed_json')}")
                            debug_text_lines.append("model_output:")
                            debug_text_lines.append(str(rec.get("model_output")))
                            debug_text_lines.append("")
            except Exception:
                pass

        for idx in range(min(self.rollout_n, len(images))):
            img = images[idx]
            advantage = advantages[idx] if idx < len(advantages) else 0.0
            img.save(os.path.join(log_dir, f"step_{global_step}_{device_id}_{advantage:.4f}_{idx}.jpg"))

        if debug_text_lines and (_debug_ocr_vllm and self.accelerator.is_main_process):
            try:
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(debug_text_lines))
                    f.write("\n")
            except Exception:
                pass

    def _compute_diffusion_loss_single_batch(
        self,
        model,
        prev_latents: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_cond: torch.Tensor,
        pooled_uncond: torch.Tensor,
        seq_cond: torch.Tensor,
        seq_uncond: torch.Tensor,
    ):
        """
        Compute diffusion model predictions for a single micro-batch.

        This method processes a single micro-batch without any internal batching.
        Used by the micro-batch gradient accumulation loop in compute_loss.

        Args:
            model: Model to use
            prev_latents: Previous latent states (micro_batch_size, C, H, W)
            timesteps: Timesteps (micro_batch_size,)
            pooled_cond: Pooled conditioning (micro_batch_size, D)
            pooled_uncond: Pooled unconditioning (micro_batch_size, D), can be None if cfg_scale=0
            seq_cond: Sequence conditioning (micro_batch_size, L, D)
            seq_uncond: Sequence unconditioning (micro_batch_size, L, D), can be None if cfg_scale=0

        Returns:
            Model prediction tensor (micro_batch_size, C, H, W)
        """
        # Get the underlying model if wrapped by DeepSpeed/Accelerator
        unwrapped_model = model.module if hasattr(model, 'module') else model

        # Conditional prediction (always needed)
        model_pred_cond = unwrapped_model.transformer(
            hidden_states=prev_latents,
            encoder_hidden_states=seq_cond,
            pooled_projections=pooled_cond,
            timestep=timesteps,
            return_dict=False,
        )[0]

        # Check if we need CFG
        cfg_scale = self.cfg_scale
        if cfg_scale > 0 and pooled_uncond is not None and seq_uncond is not None:
            # CFG enabled: compute unconditional prediction and apply CFG
            model_pred_uncond = unwrapped_model.transformer(
                hidden_states=prev_latents,
                encoder_hidden_states=seq_uncond,
                pooled_projections=pooled_uncond,
                timestep=timesteps,
                return_dict=False,
            )[0]
            # Apply CFG formula
            model_pred = model_pred_uncond + cfg_scale * (model_pred_cond - model_pred_uncond)
        else:
            # cfg_scale=0: use conditional prediction directly
            model_pred = model_pred_cond

        return model_pred

    def _compute_transformer_pred_single_batch(
        self,
        transformer: nn.Module,
        prev_latents: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_cond: torch.Tensor,
        pooled_uncond: Optional[torch.Tensor],
        seq_cond: torch.Tensor,
        seq_uncond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute transformer prediction (noise/velocity) for a single micro-batch.

        This is used by the transformer-only reference mode for KL computation.
        It reuses the already computed text embeddings (pooled/seq) from the policy
        model, and only runs a frozen reference transformer forward.
        """
        # Handle potential wrappers (DDP/Accelerate)
        unwrapped_transformer = transformer.module if hasattr(transformer, "module") else transformer

        # Conditional prediction (always needed)
        model_pred_cond = unwrapped_transformer(
            hidden_states=prev_latents,
            encoder_hidden_states=seq_cond,
            pooled_projections=pooled_cond,
            timestep=timesteps,
            return_dict=False,
        )[0]

        # Check if we need CFG
        cfg_scale = self.cfg_scale
        if cfg_scale > 0 and pooled_uncond is not None and seq_uncond is not None:
            model_pred_uncond = unwrapped_transformer(
                hidden_states=prev_latents,
                encoder_hidden_states=seq_uncond,
                pooled_projections=pooled_uncond,
                timestep=timesteps,
                return_dict=False,
            )[0]
            model_pred = model_pred_uncond + cfg_scale * (model_pred_cond - model_pred_uncond)
        else:
            model_pred = model_pred_cond

        return model_pred

    def compute_loss(
        self,
        model,
        inputs: List[Dict[str, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ):
        """
        Compute GRPO loss with TRUE micro-batch gradient accumulation.

        This method implements proper gradient accumulation where each micro-batch
        independently computes forward -> loss -> backward, then releases its
        computation graph. This ensures GPU memory usage only depends on
        atrain_micro_batch_size, NOT on rollout_n or total batch size.

        Key design:
        1. Advantage normalization uses only reward scalars (no computation graph)
        2. Text embeddings are pre-computed with no_grad to save memory
        3. Each micro-batch: forward -> loss -> backward -> release graph
        4. Gradients accumulate automatically on model parameters
        5. Return detached loss for logging (training_step skips backward)

        Args:
            model: Policy model
            inputs: Batch of inputs (prompts from DataLoader)
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in batch

        Returns:
            Detached loss tensor (backward already done internally)
        """
        if return_outputs:
            raise ValueError("DeepGenGRPOTrainer does not support returning outputs")

        device = self.accelerator.device
        dtype = torch.bfloat16 if self.args.bf16 else torch.float32

        # ============================================================================
        # SFT-Aux convex mixing
        # total_loss = (1 - lambda) * grpo_loss + lambda * sft_loss
        # ============================================================================
        sft_mix = float(self.sftaux_coef)
        grpo_mix = 1.0 - sft_mix if sft_mix > 0.0 else 1.0

        # GRPO-Guard implementation switch:
        # - paper_impl=True matches flow_grpo/scripts/train_sd3_GRPO_Guard.py "config.rationorm" math/semantics
        # - paper_impl=False matches Flow-Factory GRPOGuardTrainer (reweighted ratio)
        paper_impl = self.grpo_guard and os.environ.get("ATRAIN_GRPOGUARD_PAPER_IMPL", "0") == "1"

        # To keep paper_impl strictly aligned with flow_grpo, we ignore optional clamping knobs.
        clamp_log_prob = self._clamp_log_prob and not paper_impl
        clamp_log_prob_diff = self._clamp_log_prob_diff and not paper_impl

        # Match flow_grpo's config.train.adv_clip_max via env var (default: 5.0).
        adv_clip_max = float(os.environ.get("ATRAIN_ADV_CLIP_MAX", 5.0))

        # Mark training step start
        current_step = self.state.global_step
        self.phase_logger.step_start(current_step)

        # ================================================================
        # Optional: component grad-norm proxy logging (GRPO vs SFT-Aux)
        # ================================================================
        # Implementation detail:
        # - We register parameter grad hooks ONLY on selected steps to control overhead.
        # - During each backward call, we toggle an "active bucket" label ("grpo"/"sft")
        #   and the hooks accumulate sum(grad^2) into the corresponding scalar.
        # - With DeepSpeed ZeRO, grads are sharded; we all_reduce the scalars to get the
        #   global sum of squares across ranks.
        _gradnorm_should_measure = False
        _gradnorm_handles = []
        # Active flags (can overlap). This lets us measure:
        # - GRPO total (policy + beta*KL, after scaling)
        # - GRPO excluding KL (policy only, after scaling)
        # - beta*KL component only (after scaling)
        # - SFT-Aux component only
        _gradnorm_active = {"grpo": False, "grpo_no_kl": False, "kl": False, "sft": False}
        _grpo_grad_sq = None
        _grpo_no_kl_grad_sq = None
        _kl_grad_sq = None
        _sft_grad_sq = None

        _measure_every = self.log_component_grad_norm_every
        if _measure_every is None:
            _measure_every = int(getattr(self.args, "logging_steps", 0) or 0)
        if self.log_component_grad_norm and _measure_every > 0 and (current_step % _measure_every == 0):
            _gradnorm_should_measure = True
            _grpo_grad_sq = torch.zeros((), device=device, dtype=torch.float32)
            _grpo_no_kl_grad_sq = torch.zeros((), device=device, dtype=torch.float32)
            _kl_grad_sq = torch.zeros((), device=device, dtype=torch.float32)
            _sft_grad_sq = torch.zeros((), device=device, dtype=torch.float32)

            def _make_gradnorm_hook():
                def _hook(grad: torch.Tensor):
                    if grad is None:
                        return grad
                    g = grad.detach().float()
                    if _gradnorm_active.get("grpo", False) and _grpo_grad_sq is not None:
                        _grpo_grad_sq.add_(g.pow(2).sum())
                    if _gradnorm_active.get("grpo_no_kl", False) and _grpo_no_kl_grad_sq is not None:
                        _grpo_no_kl_grad_sq.add_(g.pow(2).sum())
                    if _gradnorm_active.get("kl", False) and _kl_grad_sq is not None:
                        _kl_grad_sq.add_(g.pow(2).sum())
                    if _gradnorm_active.get("sft", False) and _sft_grad_sq is not None:
                        _sft_grad_sq.add_(g.pow(2).sum())
                    return grad

                return _hook

            for p in model.parameters():
                if getattr(p, "requires_grad", False):
                    _gradnorm_handles.append(p.register_hook(_make_gradnorm_hook()))

        # ================================================================
        # Phase 1: Rollout - Generate images and record trajectories
        # ================================================================
        self.phase_logger.phase_start(
            current_step, 1, "ROLLOUT",
            rollout_accumulation_steps=self.rollout_accumulation_steps,
            num_prompts=len(inputs)
        )
        phase1_start = time.perf_counter()

        all_images = []
        all_log_probs_traj = []
        all_prev_latents = []
        all_pred_latents = []
        all_timesteps = []
        all_inputs = []
        # GRPO-Guard: Accumulate prev_sample_means and sqrt_dts
        all_prev_sample_means_traj = []
        all_sqrt_dts_traj = []

        # Extract prompts
        prompts_text = [ex.get("prompt", ex.get("caption", "")) for ex in inputs]
        cfg_prompts = [""] * len(prompts_text)  # Empty prompts for CFG

        # Perform multiple rollout iterations to reach rollout_global_batch_size
        for rollout_iter in range(self.rollout_accumulation_steps):
            self.phase_logger.sub_phase(current_step, f"Rollout iteration {rollout_iter + 1}/{self.rollout_accumulation_steps}")
            with torch.no_grad():
                images, log_probs_traj, prev_latents, pred_latents, timesteps, prev_sample_means, sqrt_dts = \
                    self._generate_images_with_trajectory(model, prompts_text, cfg_prompts)

            all_images.extend(images)
            all_log_probs_traj.append(log_probs_traj)
            all_prev_latents.append(prev_latents)
            all_pred_latents.append(pred_latents)
            all_timesteps.append(timesteps)
            all_inputs.extend(inputs)
            # GRPO-Guard: Collect prev_sample_means and sqrt_dts
            if self.grpo_guard:
                all_prev_sample_means_traj.append(prev_sample_means)
                all_sqrt_dts_traj.append(sqrt_dts)

        # Concatenate accumulated trajectory data (all on CPU)
        images = all_images
        log_probs_traj = torch.cat(all_log_probs_traj, dim=0)  # (batch_size, num_steps)
        prev_latents = torch.cat(all_prev_latents, dim=0)      # (batch_size, num_steps, C, H, W)
        pred_latents = torch.cat(all_pred_latents, dim=0)      # (batch_size, num_steps, C, H, W)
        timesteps_traj = torch.cat(all_timesteps, dim=0)       # (batch_size, num_steps)
        inputs = all_inputs
        # GRPO-Guard: Concatenate prev_sample_means and sqrt_dts
        if self.grpo_guard:
            prev_sample_means_traj = torch.cat(all_prev_sample_means_traj, dim=0)  # (batch_size, num_steps, C, H, W)
            sqrt_dts_traj = torch.cat(all_sqrt_dts_traj, dim=0)                    # (batch_size, num_steps, 1, 1, 1)
        else:
            prev_sample_means_traj = None
            sqrt_dts_traj = None

        # Clean up intermediate accumulation lists after concatenation
        del all_images, all_log_probs_traj, all_prev_latents, all_pred_latents, all_timesteps, all_inputs
        if self.grpo_guard:
            del all_prev_sample_means_traj, all_sqrt_dts_traj

        # Re-extract prompts for accumulated inputs
        prompts_text = [ex.get("prompt", ex.get("caption", "")) for ex in inputs]
        cfg_prompts = [""] * len(prompts_text)

        phase1_duration = (time.perf_counter() - phase1_start) * 1000
        self.phase_logger.phase_end(
            current_step, 1, "ROLLOUT",
            duration_ms=phase1_duration,
            num_images=len(images)
        )

        # ================================================================
        # Phase 2: Compute rewards and advantages (scalar operations only)
        # ================================================================
        self.phase_logger.phase_start(
            current_step, 2, "REWARD_COMPUTATION",
            num_images=len(images),
            reward_funcs=[f[0] for f in self.reward_funcs]
        )
        phase2_start = time.perf_counter()

        rewards, rewards_per_func = self._compute_rewards(inputs, images)

        phase2_duration = (time.perf_counter() - phase2_start) * 1000
        self.phase_logger.log_stats(
            current_step,
            mean_reward=rewards.mean().item(),
            **{f"reward/{f[0]}": rewards_per_func[:, i].mean().item() for i, f in enumerate(self.reward_funcs)}
        )
        self.phase_logger.phase_end(current_step, 2, "REWARD_COMPUTATION", duration_ms=phase2_duration)

        # ================================================================
        # Phase 3: Advantage Computation
        # ================================================================
        self.phase_logger.phase_start(current_step, 3, "ADVANTAGE_COMPUTATION")
        phase3_start = time.perf_counter()

        # Compute advantages based on atrain_adv_type (grpo, reinforcepp, or gdpo)
        advantages = self._compute_advantages(
            rewards=rewards,
            rewards_per_func=rewards_per_func,
            inputs=inputs,
        )
        pre_clamp_advantages = advantages.clone()
        advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)

        # Calculate clamp ratio
        clamped_count = (pre_clamp_advantages.abs() > adv_clip_max).sum().item()
        clamp_ratio = clamped_count / len(advantages) if len(advantages) > 0 else 0.0

        phase3_duration = (time.perf_counter() - phase3_start) * 1000
        self.phase_logger.log_stats(
            current_step,
            adv_mean=advantages.mean().item(),
            adv_std=advantages.std().item(),
            adv_min=advantages.min().item(),
            adv_max=advantages.max().item(),
            clamp_ratio=f"{clamp_ratio:.2%}"
        )
        self.phase_logger.phase_end(current_step, 3, "ADVANTAGE_COMPUTATION", duration_ms=phase3_duration)

        # Clean up intermediate variables from advantage computation
        del pre_clamp_advantages

        # Log samples
        self._log_step(images, prompts_text, advantages)

        # ================================================================
        # Phase 3: Prepare for training
        # ================================================================
        # Expand prompts for all generated images
        expanded_prompts = [p for p in prompts_text for _ in range(self.rollout_n)]
        expanded_cfg_prompts = [p for p in cfg_prompts for _ in range(self.rollout_n)]
        batch_size = len(expanded_prompts)  # Total number of images
        num_steps = prev_latents.shape[1]   # Number of diffusion steps

        # Calculate micro-batch parameters
        micro_batch_size = self.atrain_micro_batch_size
        num_micro_batches = math.ceil(batch_size / micro_batch_size)

        # Calculate num_train_timesteps based on timestep_fraction
        # Randomly sample (num_steps * timestep_fraction) timesteps for gradient computation
        num_train_timesteps = max(1, int(num_steps * self.timestep_fraction))
        # Randomly select num_train_timesteps indices from all timesteps
        #
        # IMPORTANT (align with Flow-GRPO / Flow-Factory):
        # Do NOT train on the last denoising step by default. At the final low-noise step,
        # the transition distribution can become extremely sharp, making log-prob / ratio
        # numerically unstable (overflow) especially when CFG is enabled.
        #
        # Flow-GRPO pipeline_with_logprob_fast uses sde_window=(0, len(timesteps)-1) by default,
        # explicitly excluding the last step. Flow-Factory's default train_steps also excludes
        # the last step.
        #
        # If you really want to include the last step (at your own risk), set:
        #   ATRAIN_INCLUDE_LAST_DENOISE_STEP=1
        include_last_step = os.environ.get("ATRAIN_INCLUDE_LAST_DENOISE_STEP", "0") == "1"
        effective_num_steps = num_steps if include_last_step else max(1, num_steps - 1)
        all_step_indices = list(range(effective_num_steps))
        grad_step_indices = sorted(random.sample(all_step_indices, num_train_timesteps))
        num_grad_steps = len(grad_step_indices)

        # Loss scaling factor: ensure mathematical equivalence with full batch processing
        # Total loss = sum over (timesteps × micro_batches) / (num_grad_steps × num_micro_batches)
        loss_scale = 1.0 / (num_grad_steps * num_micro_batches)

        # Debug log: configuration
        if self.accelerator.is_main_process and self.state.global_step % 10 == 0:
            env_log_dir = os.environ.get("LOG_DIR", ".")
            debug_path = os.path.join(env_log_dir, "debug_micro_batch.log")
            with open(debug_path, "a") as f:
                f.write(f"\n=== Step {self.state.global_step} Micro-batch Config ===\n")
                f.write(f"batch_size: {batch_size}, micro_batch_size: {micro_batch_size}\n")
                f.write(f"num_micro_batches: {num_micro_batches}, num_grad_steps: {num_grad_steps}\n")
                f.write(f"loss_scale: {loss_scale}\n")
                f.flush()

        # ================================================================
        # Phase 4: Pre-compute text embeddings (no gradient needed)
        # ================================================================
        # Text embeddings are shared across all timesteps and don't need gradients
        # because we're not training the LLM, only the transformer
        self.phase_logger.phase_start(
            current_step, 4, "TEXT_EMBEDDING_PRECOMPUTATION",
            num_prompts=len(expanded_prompts),
            micro_batch_size=micro_batch_size
        )
        phase4_start = time.perf_counter()

        with torch.no_grad():
            policy_embed_start = time.perf_counter()
            pooled_cond, pooled_uncond, seq_cond, seq_uncond = self._compute_text_embeddings_batched(
                model, expanded_prompts, expanded_cfg_prompts,
                micro_batch_size=micro_batch_size, no_grad=True
            )
            policy_embed_duration = (time.perf_counter() - policy_embed_start) * 1000
            self.phase_logger.sub_phase(current_step, "Policy model text embeddings", duration_ms=policy_embed_duration)

            # Only compute reference model embeddings if beta > 0 AND we are using a full ref_model.
            # When using transformer-only reference, the text-conditioning path is frozen and
            # we can reuse policy embeddings directly for KL reference predictions.
            if self.beta > 0 and self.ref_model is not None:
                ref_embed_start = time.perf_counter()
                ref_pooled_cond, ref_pooled_uncond, ref_seq_cond, ref_seq_uncond = self._compute_text_embeddings_batched(
                    self.ref_model, expanded_prompts, expanded_cfg_prompts,
                    micro_batch_size=micro_batch_size, no_grad=True
                )
                ref_embed_duration = (time.perf_counter() - ref_embed_start) * 1000
                self.phase_logger.sub_phase(current_step, "Reference model text embeddings", duration_ms=ref_embed_duration)
            else:
                # beta=0 OR transformer-only reference: skip reference model embeddings
                ref_pooled_cond = ref_pooled_uncond = ref_seq_cond = ref_seq_uncond = None
                if self.beta > 0 and self.ref_transformer is not None:
                    self.phase_logger.sub_phase(current_step, "Reference model embeddings skipped (transformer-only ref)")
                else:
                    self.phase_logger.sub_phase(current_step, "Reference model embeddings skipped (beta=0)")

        phase4_duration = (time.perf_counter() - phase4_start) * 1000
        self.phase_logger.phase_end(current_step, 4, "TEXT_EMBEDDING_PRECOMPUTATION", duration_ms=phase4_duration)

        # ================================================================
        # Phase 4.5: Pre-compute NFT old_v_pred with EMA parameters (off-policy mode)
        # ================================================================
        # For DiffusionNFT off-policy mode, we need to pre-compute old_v_pred using EMA parameters
        # before the training loop. This is similar to Flow-Factory's sampling_context() approach.
        #
        # old_v_pred_cache: dict mapping (step_idx, sample_idx) -> old_v_pred tensor (on CPU)
        # This cache is used in the training loop to retrieve pre-computed old_v_pred
        nft_old_v_pred_cache = {}
        if self.diffusion_nft and self.nft_off_policy and self.ema_diffusion is not None:
            self.phase_logger.phase_start(
                current_step, 45, "NFT_OLD_V_PRED_PRECOMPUTATION",
                batch_size=batch_size,
                num_timesteps=num_grad_steps,
                mode="off-policy (EMA parameters)"
            )
            phase45_start = time.perf_counter()

            # Swap in EMA weights for old_v_pred computation
            with self._swap_in_ema_diffusion_weights():
                unwrapped_model = self.accelerator.unwrap_model(model)

                # Pre-compute old_v_pred for all timesteps and all samples
                for step_idx in grad_step_indices:
                    self.phase_logger.sub_phase(
                        current_step,
                        f"Pre-computing old_v_pred for timestep {step_idx} ({grad_step_indices.index(step_idx) + 1}/{num_grad_steps})"
                    )

                    # Get trajectory data for this timestep
                    step_prev_latents = prev_latents[:, step_idx].to(device=device, dtype=dtype)  # (batch_size, C, H, W)
                    step_timesteps = timesteps_traj[:, step_idx].to(device=device)  # (batch_size,)

                    # Process in micro-batches to avoid OOM
                    all_old_v_pred_for_step = []
                    for mb_start in range(0, batch_size, micro_batch_size):
                        mb_end = min(mb_start + micro_batch_size, batch_size)
                        mb_prev_latents = step_prev_latents[mb_start:mb_end]
                        mb_timesteps = step_timesteps[mb_start:mb_end]
                        mb_pooled_cond = pooled_cond[mb_start:mb_end]
                        mb_seq_cond = seq_cond[mb_start:mb_end]

                        # Compute old_v_pred using EMA parameters
                        with torch.no_grad():
                            mb_old_v_pred = unwrapped_model.transformer(
                                hidden_states=mb_prev_latents,
                                encoder_hidden_states=mb_seq_cond,
                                pooled_projections=mb_pooled_cond,
                                timestep=mb_timesteps,
                                return_dict=False,
                            )[0]
                        all_old_v_pred_for_step.append(mb_old_v_pred.float().cpu())

                    # Concatenate all micro-batches and store in cache
                    step_old_v_pred = torch.cat(all_old_v_pred_for_step, dim=0)  # (batch_size, C, H, W)
                    nft_old_v_pred_cache[step_idx] = step_old_v_pred

                    # Free GPU memory
                    del step_prev_latents, step_timesteps, all_old_v_pred_for_step
                    torch.cuda.empty_cache()

            phase45_duration = (time.perf_counter() - phase45_start) * 1000
            self.phase_logger.phase_end(current_step, 45, "NFT_OLD_V_PRED_PRECOMPUTATION", duration_ms=phase45_duration)
            rank0_print(f"[DiffusionNFT] Pre-computed old_v_pred for {len(grad_step_indices)} timesteps using EMA parameters")

        # ================================================================
        # Phase 5: TRUE Micro-batch Gradient Accumulation with Multiple Actor Updates
        # ================================================================
        # For each actor update step:
        #   1. Shuffle samples and select non-overlapping subset
        #   For each timestep and each micro-batch:
        #     2. Forward pass (with gradient)
        #     3. Compute loss
        #     4. Backward pass (gradients accumulate)
        #     5. Delete tensors to free GPU memory
        # This ensures GPU memory only depends on micro_batch_size
        #
        # Multiple actor updates per rollout (inspired by PPO/Flow-GRPO num_inner_epochs):
        # - When atrain_num_actor_update_steps > 1, we update the model N times per rollout
        # - Each update step uses a non-overlapping subset of the rollout data
        # - Samples are shuffled before splitting to ensure diverse training

        num_actor_update_steps = self.atrain_num_actor_update_steps
        samples_per_update = batch_size // num_actor_update_steps

        # Recalculate num_micro_batches based on samples_per_update
        num_micro_batches_per_update = math.ceil(samples_per_update / micro_batch_size)

        # Adjust loss_scale for multiple actor updates
        # Total loss = sum over (actor_updates × timesteps × micro_batches)
        loss_scale = 1.0 / (num_actor_update_steps * num_grad_steps * num_micro_batches_per_update)

        self.phase_logger.phase_start(
            current_step, 5, "MICRO_BATCH_GRADIENT_ACCUMULATION",
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            num_micro_batches=num_micro_batches_per_update,
            num_grad_steps=num_grad_steps,
            num_actor_update_steps=num_actor_update_steps,
            samples_per_update=samples_per_update
        )
        phase5_start = time.perf_counter()

        total_loss_for_logging = 0.0
        total_grpo_loss_unscaled_for_logging = 0.0
        total_grpo_loss_scaled_for_logging = 0.0
        sftaux_loss_unscaled_for_logging = 0.0
        sftaux_loss_scaled_for_logging = 0.0
        did_sftaux = False
        all_kl_values = []
        total_forward_backward_count = 0

        # Debug log: configuration with actor updates
        if self.accelerator.is_main_process and self.state.global_step % 10 == 0:
            env_log_dir = os.environ.get("LOG_DIR", ".")
            debug_path = os.path.join(env_log_dir, "debug_micro_batch.log")
            with open(debug_path, "a") as f:
                f.write(f"\n=== Step {self.state.global_step} Actor Update Config ===\n")
                f.write(f"batch_size: {batch_size}, num_actor_update_steps: {num_actor_update_steps}\n")
                f.write(f"samples_per_update: {samples_per_update}, micro_batch_size: {micro_batch_size}\n")
                f.write(f"num_micro_batches_per_update: {num_micro_batches_per_update}, num_grad_steps: {num_grad_steps}\n")
                f.write(f"loss_scale: {loss_scale}\n")
                f.flush()

        # Multiple actor update steps loop
        # Shuffle indices once at the beginning, then split into non-overlapping subsets
        shuffled_indices = torch.randperm(batch_size, device=device)

        for actor_update_idx in range(num_actor_update_steps):
            # Select non-overlapping subset for this actor update step
            update_start = actor_update_idx * samples_per_update
            update_end = min(update_start + samples_per_update, batch_size)
            update_indices = shuffled_indices[update_start:update_end].cpu()

            if num_actor_update_steps > 1:
                self.phase_logger.sub_phase(
                    current_step,
                    f"Actor update {actor_update_idx + 1}/{num_actor_update_steps} (samples {update_start}-{update_end})"
                )

            # Per-update loss tracking (for per-update logging)
            update_grpo_loss = 0.0
            update_kl_values = []

            # Current update's batch size (may be smaller for last update)
            current_update_batch_size = len(update_indices)
            current_num_micro_batches = math.ceil(current_update_batch_size / micro_batch_size)

            for step_idx in grad_step_indices:
                self.phase_logger.sub_phase(
                    current_step,
                    f"Processing timestep {step_idx} ({grad_step_indices.index(step_idx) + 1}/{num_grad_steps})"
                )
                # Get trajectory data for this timestep (still on CPU)
                # Use update_indices to select samples for this actor update step
                step_prev_latents_cpu = prev_latents[update_indices, step_idx]      # (current_update_batch_size, C, H, W)
                step_pred_latents_cpu = pred_latents[update_indices, step_idx]      # (current_update_batch_size, C, H, W)
                step_timesteps_cpu = timesteps_traj[update_indices, step_idx]       # (current_update_batch_size,)
                step_old_log_probs_cpu = log_probs_traj[update_indices, step_idx]   # (current_update_batch_size,)
                # Select advantages for current update indices
                update_advantages = advantages[update_indices.to(device)]  # (current_update_batch_size,)
                # GRPO-Guard: Get prev_sample_mean and sqrt_dt for this timestep
                if self.grpo_guard:
                    step_prev_sample_means_cpu = prev_sample_means_traj[update_indices, step_idx]  # (current_update_batch_size, C, H, W)
                    step_sqrt_dts_cpu = sqrt_dts_traj[update_indices, step_idx]                    # (current_update_batch_size, 1, 1, 1)

                # Select pre-computed embeddings for current update indices
                update_pooled_cond = pooled_cond[update_indices.to(device)]
                update_pooled_uncond = pooled_uncond[update_indices.to(device)] if pooled_uncond is not None else None
                update_seq_cond = seq_cond[update_indices.to(device)]
                update_seq_uncond = seq_uncond[update_indices.to(device)] if seq_uncond is not None else None
                if self.beta > 0 and (self.ref_model is not None or self.ref_transformer is not None):
                    if self.ref_model is not None:
                        update_ref_pooled_cond = ref_pooled_cond[update_indices.to(device)]
                        update_ref_pooled_uncond = ref_pooled_uncond[update_indices.to(device)] if ref_pooled_uncond is not None else None
                        update_ref_seq_cond = ref_seq_cond[update_indices.to(device)]
                        update_ref_seq_uncond = ref_seq_uncond[update_indices.to(device)] if ref_seq_uncond is not None else None

                for mb_idx in range(current_num_micro_batches):
                    # Slice data for this micro-batch
                    start_idx = mb_idx * micro_batch_size
                    end_idx = min(start_idx + micro_batch_size, current_update_batch_size)

                    # Move micro-batch data to GPU
                    mb_prev_latents = step_prev_latents_cpu[start_idx:end_idx].to(device=device, dtype=dtype)
                    mb_pred_latents = step_pred_latents_cpu[start_idx:end_idx].to(device=device, dtype=dtype)
                    mb_timesteps = step_timesteps_cpu[start_idx:end_idx].to(device=device)
                    mb_old_log_probs = step_old_log_probs_cpu[start_idx:end_idx].to(device=device, dtype=dtype)
                    mb_advantages = update_advantages[start_idx:end_idx]  # Already on device
                    # GRPO-Guard: Move prev_sample_mean and sqrt_dt to GPU
                    if self.grpo_guard:
                        mb_old_prev_sample_mean = step_prev_sample_means_cpu[start_idx:end_idx].to(device=device, dtype=dtype)
                        mb_sqrt_dt = step_sqrt_dts_cpu[start_idx:end_idx].to(device=device, dtype=dtype)

                    # Slice pre-computed embeddings for this micro-batch
                    mb_pooled_cond = update_pooled_cond[start_idx:end_idx]
                    mb_pooled_uncond = update_pooled_uncond[start_idx:end_idx] if update_pooled_uncond is not None else None
                    mb_seq_cond = update_seq_cond[start_idx:end_idx]
                    mb_seq_uncond = update_seq_uncond[start_idx:end_idx] if update_seq_uncond is not None else None

                    # ----- Forward pass (WITH gradients) -----
                    mb_model_pred = self._compute_diffusion_loss_single_batch(
                        model, mb_prev_latents, mb_timesteps,
                        mb_pooled_cond, mb_pooled_uncond, mb_seq_cond, mb_seq_uncond
                    )

                    # ----- Reference model forward (only if beta > 0 and reference is available) -----
                    # Two modes:
                    # - Full ref_model: compute reference embeddings and forward through ref_model
                    # - Transformer-only ref: reuse policy embeddings and forward through ref_transformer
                    has_ref = self.beta > 0 and (self.ref_model is not None or self.ref_transformer is not None)
                    mb_ref_pred = None
                    if has_ref:
                        # Compute reference prediction.
                        # - Full ref_model: needs its own text embeddings
                        # - Transformer-only ref: reuse policy embeddings (text path is frozen)
                        if self.ref_model is not None:
                            mb_ref_pooled_cond = update_ref_pooled_cond[start_idx:end_idx]
                            mb_ref_pooled_uncond = update_ref_pooled_uncond[start_idx:end_idx] if update_ref_pooled_uncond is not None else None
                            mb_ref_seq_cond = update_ref_seq_cond[start_idx:end_idx]
                            mb_ref_seq_uncond = update_ref_seq_uncond[start_idx:end_idx] if update_ref_seq_uncond is not None else None

                            with torch.no_grad():
                                mb_ref_pred = self._compute_diffusion_loss_single_batch(
                                    self.ref_model, mb_prev_latents, mb_timesteps,
                                    mb_ref_pooled_cond, mb_ref_pooled_uncond, mb_ref_seq_cond, mb_ref_seq_uncond
                                )
                        else:
                            # Transformer-only reference: reuse policy embeddings.
                            mb_ref_pooled_cond = mb_ref_pooled_uncond = mb_ref_seq_cond = mb_ref_seq_uncond = None
                            with torch.no_grad():
                                mb_ref_pred = self._compute_transformer_pred_single_batch(
                                    self.ref_transformer,
                                    mb_prev_latents,
                                    mb_timesteps,
                                    mb_pooled_cond,
                                    mb_pooled_uncond,
                                    mb_seq_cond,
                                    mb_seq_uncond,
                                )

                    # ----- Compute log probabilities -----
                    # DiffusionNFT does not use log_prob for loss computation
                    # but may still need mean/std for KL loss if enabled
                    if self.diffusion_nft:
                        # For DiffusionNFT, we skip log_prob computation
                        # But we may still need std_policy for KL loss
                        mb_log_prob_policy = torch.zeros(mb_model_pred.shape[0], device=device, dtype=dtype)
                        # Estimate std from scheduler for KL (if needed)
                        # For flow matching: std ~ sqrt(-dt) * sigma_t
                        t_normalized = mb_timesteps.float() / 1000.0
                        mb_std_policy = torch.ones_like(t_normalized).view(-1, 1, 1, 1) * 0.1  # Placeholder
                        mb_mean_policy = mb_model_pred  # Use noise prediction as mean for v-based KL
                    else:
                        # Enable debug logging for first micro-batch of all timesteps
                        should_debug = self.accelerator.is_main_process and mb_idx == 0
                        if should_debug:
                            if self.grpo_guard:
                                # GRPO-Guard: Get prev_sample_mean and std_dev_t for RatioNorm
                                result = sde_step_with_logprob(
                                    self.scheduler, mb_model_pred, mb_timesteps, mb_prev_latents, mb_pred_latents,
                                    eta=self.sde_eta, debug=True, return_sqrt_dt=True, sampler_type=self.atrain_sde_sampler,
                                    clamp_log_prob=clamp_log_prob,
                                )
                                _, mb_log_prob_policy, mb_prev_sample_mean, mb_std_dev_t, mb_new_sqrt_dt, debug_info = result
                                # Also get mb_mean_policy and mb_std_policy for KL computation
                                mb_mean_policy = mb_prev_sample_mean
                                mb_std_policy = mb_std_dev_t
                            else:
                                result = sde_step_with_logprob(
                                    self.scheduler, mb_model_pred, mb_timesteps, mb_prev_latents, mb_pred_latents,
                                    eta=self.sde_eta, debug=True, sampler_type=self.atrain_sde_sampler,
                                    clamp_log_prob=clamp_log_prob,
                                )
                                _, mb_log_prob_policy, mb_mean_policy, mb_std_policy, debug_info = result

                            # Write debug info
                            env_log_dir = os.environ.get("LOG_DIR", ".")
                            debug_path = os.path.join(env_log_dir, "debug_sde_logprob.log")
                            with open(debug_path, "a") as f:
                                f.write(f"\n{'='*80}\n")
                                f.write(f"Step {self.state.global_step} | Timestep {step_idx} | MicroBatch {mb_idx}\n")
                                f.write(f"{'='*80}\n")
                                f.write(f"[Policy Model SDE Debug Info]\n")
                                f.write(f"  sampler_type: {self.atrain_sde_sampler}\n")
                                for k, v in debug_info.items():
                                    f.write(f"  {k}: {v}\n")
                                f.flush()

                            # DEBUG_CPS_SDE: Extra CPS-specific debug logging
                            if os.environ.get("DEBUG_CPS_SDE", "0") == "1" and self.atrain_sde_sampler == "cps_sde":
                                debug_log_dir = os.path.join(self.args.output_dir, "cps_sde_debug")
                                os.makedirs(debug_log_dir, exist_ok=True)
                                debug_log_file = os.path.join(debug_log_dir, f"training_rank{dist.get_rank() if dist.is_initialized() else 0}.log")
                                with open(debug_log_file, "a") as f:
                                    f.write(f"=== Training Step {self.state.global_step}, Timestep {step_idx}, MicroBatch {mb_idx} ===\n")
                                    f.write(f"  atrain_sde_sampler: {self.atrain_sde_sampler}\n")
                                    f.write(f"  sde_eta: {self.sde_eta}\n")
                                    f.write(f"  mb_log_prob_policy: mean={mb_log_prob_policy.mean().item():.6f}, std={mb_log_prob_policy.std().item():.6f}\n")
                                    f.write(f"  mb_log_prob_policy: min={mb_log_prob_policy.min().item():.6f}, max={mb_log_prob_policy.max().item():.6f}\n")
                                    f.write(f"  has_nan: {torch.isnan(mb_log_prob_policy).any().item()}, has_inf: {torch.isinf(mb_log_prob_policy).any().item()}\n")
                                    for k, v in debug_info.items():
                                        f.write(f"  {k}: {v}\n")
                                    f.write("\n")
                        else:
                            if self.grpo_guard:
                                # GRPO-Guard: Get prev_sample_mean and std_dev_t for RatioNorm
                                _, mb_log_prob_policy, mb_prev_sample_mean, mb_std_dev_t, mb_new_sqrt_dt = sde_step_with_logprob(
                                    self.scheduler, mb_model_pred, mb_timesteps, mb_prev_latents, mb_pred_latents,
                                    eta=self.sde_eta, return_sqrt_dt=True, sampler_type=self.atrain_sde_sampler,
                                    clamp_log_prob=clamp_log_prob,
                                )
                                # Also get mb_mean_policy and mb_std_policy for KL computation
                                mb_mean_policy = mb_prev_sample_mean
                                mb_std_policy = mb_std_dev_t
                            else:
                                _, mb_log_prob_policy, mb_mean_policy, mb_std_policy = sde_step_with_logprob(
                                    self.scheduler, mb_model_pred, mb_timesteps, mb_prev_latents, mb_pred_latents,
                                    eta=self.sde_eta, sampler_type=self.atrain_sde_sampler,
                                    clamp_log_prob=clamp_log_prob,
                                )

                    # ----- Compute KL divergence -----
                    if has_ref:
                        if self.atrain_kl_type == "v-based":
                            # v-based KL: compare noise predictions (velocity space)
                            # KL = ((noise_pred_policy - noise_pred_ref)^2) / (2 * std^2)
                            # Match Flow-Factory convention: divide by (2 * std_dev_t^2 + eps)
                            mb_kl = (mb_model_pred.float() - mb_ref_pred.float()) ** 2 / (2 * mb_std_policy ** 2 + 1e-7)
                        else:
                            # x-based KL: compare latent means (default)
                            # KL = ((mean_policy - mean_ref)^2) / (2 * std^2)
                            with torch.no_grad():
                                _, _, mb_mean_ref, _ = sde_step_with_logprob(
                                    self.scheduler, mb_ref_pred, mb_timesteps, mb_prev_latents, mb_pred_latents,
                                    eta=self.sde_eta, sampler_type=self.atrain_sde_sampler,
                                    clamp_log_prob=clamp_log_prob,
                                )
                            # Match Flow-Factory convention: divide by (2 * std_dev_t^2 + eps)
                            mb_kl = (mb_mean_policy - mb_mean_ref) ** 2 / (2 * mb_std_policy ** 2 + 1e-7)
                        mb_kl = mb_kl.mean(dim=tuple(range(1, mb_kl.ndim)))
                        all_kl_values.append(mb_kl.detach())
                        update_kl_values.append(mb_kl.detach())  # Per-update tracking
                        mb_kl_loss = mb_kl.mean()
                    else:
                        mb_kl_loss = torch.tensor(0.0, device=device)

                # ============================================================================
                # DiffusionNFT Loss Computation
                # Reference: https://arxiv.org/abs/2509.16117
                # Key difference: NFT uses v-space (noise prediction) loss with adaptive weighting
                # instead of log_prob-based importance sampling.
                #
                # NFT requires two predictions:
                # 1. old_v_pred: prediction from the "old" policy (rollout-time model)
                # 2. new_v_pred: prediction from the current (training) model
                #
                # Two modes (controlled by --atrain_nft_off_policy):
                # - On-policy mode (default): old_v_pred = current model detached
                # - Off-policy mode: old_v_pred = EMA model predictions (pre-computed in Phase 4.5)
                #
                # This is different from ref_pred (frozen reference model) which is only for KL loss.
                # ============================================================================
                if self.diffusion_nft:
                    # Get timestep in [0, 1] range
                    t_normalized = mb_timesteps.float() / 1000.0
                    t_broadcast = t_normalized.view(-1, *([1] * (mb_model_pred.dim() - 1)))

                    # Use mb_pred_latents as x0 (clean latent estimate) and mb_prev_latents as xt
                    x0 = mb_pred_latents  # Clean latent (prediction target)
                    xt = mb_prev_latents  # Noised latent

                    # new_v_pred is the current model's prediction (with gradients)
                    new_v_pred = mb_model_pred

                    # old_v_pred: depends on on-policy vs off-policy mode
                    if self.nft_off_policy and step_idx in nft_old_v_pred_cache:
                        # Off-policy mode: use pre-computed EMA predictions
                        # Get the global indices for this micro-batch
                        mb_global_indices = update_indices[start_idx:end_idx]
                        old_v_pred = nft_old_v_pred_cache[step_idx][mb_global_indices].to(device=device, dtype=dtype)
                    else:
                        # On-policy mode: use current model prediction but detached
                        # This follows Flow-Factory's default behavior where old_v_pred comes from
                        # the same model at the start of the optimization step.
                        old_v_pred = mb_model_pred.detach()

                    # Clip advantages for NFT
                    nft_adv_clip = self.nft_adv_clip_range
                    mb_adv_clipped = torch.clamp(mb_advantages, -nft_adv_clip, nft_adv_clip)

                    # Normalize advantage to [0, 1] range
                    # r = (adv / adv_clip_max) / 2.0 + 0.5
                    normalized_adv = (mb_adv_clipped / nft_adv_clip) / 2.0 + 0.5
                    r = torch.clamp(normalized_adv, 0, 1).view(-1, *([1] * (new_v_pred.dim() - 1)))

                    # Compute positive and negative predictions (NFT mixing)
                    # positive_pred = beta * new + (1 - beta) * old
                    # negative_pred = (1 + beta) * old - beta * new
                    nft_beta = self.nft_beta
                    positive_pred = nft_beta * new_v_pred + (1 - nft_beta) * old_v_pred
                    negative_pred = (1.0 + nft_beta) * old_v_pred - nft_beta * new_v_pred

                    # Compute x0 predictions from v predictions
                    # x0_pred = xt - t * v_pred
                    x0_pred_positive = xt - t_broadcast * positive_pred
                    x0_pred_negative = xt - t_broadcast * negative_pred

                    # Adaptive weighting for positive loss
                    with torch.no_grad():
                        weight_positive = torch.abs(x0_pred_positive.double() - x0.double()).mean(
                            dim=tuple(range(1, x0.ndim)), keepdim=True
                        ).clamp(min=1e-5)
                    positive_loss = ((x0_pred_positive - x0) ** 2 / weight_positive).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    # Adaptive weighting for negative loss
                    with torch.no_grad():
                        weight_negative = torch.abs(x0_pred_negative.double() - x0.double()).mean(
                            dim=tuple(range(1, x0.ndim)), keepdim=True
                        ).clamp(min=1e-5)
                    negative_loss = ((x0_pred_negative - x0) ** 2 / weight_negative).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    # Combined NFT loss
                    # ori_policy_loss = (r * positive_loss + (1 - r) * negative_loss) / nft_beta
                    ori_policy_loss = (r.squeeze() * positive_loss + (1.0 - r.squeeze()) * negative_loss) / nft_beta
                    mb_policy_loss = (ori_policy_loss * nft_adv_clip).mean()

                    # Total loss with KL penalty
                    mb_loss = (mb_policy_loss + self.beta * mb_kl_loss) * loss_scale

                    # Debug logging for DiffusionNFT
                    if self.accelerator.is_main_process:
                        env_log_dir = os.environ.get("LOG_DIR", ".")
                        debug_path = os.path.join(env_log_dir, "debug_loss_components.log")
                        with open(debug_path, "a") as f:
                            f.write(f"\n{'='*80}\n")
                            f.write(f"Step {self.state.global_step} | Timestep {step_idx} | MicroBatch {mb_idx}\n")
                            f.write(f"{'='*80}\n")
                            old_v_pred_source = "EMA (off-policy)" if (self.nft_off_policy and step_idx in nft_old_v_pred_cache) else "detached (on-policy)"
                            f.write(f"[DiffusionNFT] nft_beta={nft_beta}, nft_adv_clip={nft_adv_clip}, old_v_pred={old_v_pred_source}\n")
                            f.write(f"[DiffusionNFT] t_normalized: mean={t_normalized.mean().item():.6f}\n")
                            f.write(f"[DiffusionNFT] r (normalized adv): mean={r.mean().item():.6f}, "
                                    f"min={r.min().item():.6f}, max={r.max().item():.6f}\n")
                            f.write(f"[DiffusionNFT] positive_loss: mean={positive_loss.mean().item():.6f}\n")
                            f.write(f"[DiffusionNFT] negative_loss: mean={negative_loss.mean().item():.6f}\n")
                            f.write(f"[DiffusionNFT] ori_policy_loss: mean={ori_policy_loss.mean().item():.6f}\n")
                            f.write(f"[DiffusionNFT] mb_policy_loss: {mb_policy_loss.item():.6f}\n")
                            f.write(f"[DiffusionNFT] mb_kl_loss: {mb_kl_loss.item():.6f}\n")
                            f.write(f"[DiffusionNFT] mb_loss: {mb_loss.item():.6f}\n")
                            f.write(f"-" * 40 + "\n")
                            f.flush()

                else:
                    # ----- Compute GRPO loss for this micro-batch -----
                    # Keep the raw Δlogp for exact flow_grpo alignment under paper_impl.
                    raw_log_prob_diff = mb_log_prob_policy - mb_old_log_probs
                    log_prob_diff = raw_log_prob_diff

                    # Debug log: log_prob values before clamp
                    if self.accelerator.is_main_process:
                        env_log_dir = os.environ.get("LOG_DIR", ".")
                        debug_path = os.path.join(env_log_dir, "debug_loss_components.log")
                        with open(debug_path, "a") as f:
                            f.write(f"\n{'='*80}\n")
                            f.write(f"Step {self.state.global_step} | Timestep {step_idx} | MicroBatch {mb_idx}\n")
                            f.write(f"{'='*80}\n")
                            f.write(f"[1] mb_log_prob_policy: mean={mb_log_prob_policy.mean().item():.6f}, "
                                    f"std={mb_log_prob_policy.std().item():.6f}, "
                                    f"min={mb_log_prob_policy.min().item():.6f}, "
                                    f"max={mb_log_prob_policy.max().item():.6f}\n")
                            f.write(f"[2] mb_old_log_probs: mean={mb_old_log_probs.mean().item():.6f}, "
                                    f"std={mb_old_log_probs.std().item():.6f}, "
                                    f"min={mb_old_log_probs.min().item():.6f}, "
                                    f"max={mb_old_log_probs.max().item():.6f}\n")
                            f.write(f"[3] log_prob_diff (before clamp): mean={log_prob_diff.mean().item():.6f}, "
                                    f"std={log_prob_diff.std().item():.6f}, "
                                    f"min={log_prob_diff.min().item():.6f}, "
                                    f"max={log_prob_diff.max().item():.6f}\n")
                            f.flush()

                    # Optional clamp for numerical stability (NOT Flow-Factory default)
                    if clamp_log_prob_diff:
                        log_prob_diff = torch.clamp(log_prob_diff, -10.0, 10.0)

                    # Importance ratio
                    if self.grpo_guard:
                        # Match Flow-Factory's GRPOGuardTrainer ratio:
                        # ratio = exp((Δlogp)*scale_factor + mse/(2*scale_factor))
                        # where scale_factor = sqrt(-dt) * std_dev_t, and mse compares next_latents_mean.
                        #
                        # NOTE: This is intentionally different from the original paper-style RatioNorm/GradReweight
                        # implementation. If you want the paper-style behavior, keep ATRAIN_GRPOGUARD_PAPER_IMPL=1.
                        if paper_impl:
                            # Legacy (paper-style) path kept for backward compatibility
                            # This matches flow_grpo/scripts/train_sd3_GRPO_Guard.py with config.rationorm=True.
                            sigma_t = mb_std_dev_t.mean()
                            ratio_mean_bias = (mb_prev_sample_mean - mb_old_prev_sample_mean).pow(2).mean(
                                dim=tuple(range(1, mb_log_prob_policy.ndim))
                            )
                            ratio_mean_bias = ratio_mean_bias / (2 * (mb_new_sqrt_dt.mean() * sigma_t) ** 2)
                            ratio = torch.exp((raw_log_prob_diff + ratio_mean_bias) * (mb_new_sqrt_dt.mean() * sigma_t))
                        else:
                            # Flow-Factory-like path
                            # scale_factor is per-sample; use mean over broadcast dims to get (B,)
                            scale_factor = (mb_new_sqrt_dt * mb_std_dev_t).view(mb_std_dev_t.shape[0], -1).mean(dim=1)
                            scale_factor = torch.clamp(scale_factor, min=1e-8)

                            # mse between new mean and old mean (both are tensors shaped like latents)
                            mse = (mb_prev_sample_mean - mb_old_prev_sample_mean).flatten(1).pow(2).mean(dim=1)
                            ratio = torch.exp(log_prob_diff * scale_factor + mse / (2.0 * scale_factor))
                    else:
                        ratio = torch.exp(log_prob_diff)

                    # Debug log: ratio and advantage statistics
                    if self.accelerator.is_main_process:
                        with open(debug_path, "a") as f:
                            f.write(f"[4] log_prob_diff (after clamp): mean={log_prob_diff.mean().item():.6f}, "
                                    f"std={log_prob_diff.std().item():.6f}, "
                                    f"min={log_prob_diff.min().item():.6f}, "
                                    f"max={log_prob_diff.max().item():.6f}\n")
                            f.write(f"[5] ratio: mean={ratio.mean().item():.6f}, "
                                    f"std={ratio.std().item():.6f}, "
                                    f"min={ratio.min().item():.6f}, "
                                    f"max={ratio.max().item():.6f}\n")
                            f.write(f"[6] mb_advantages: mean={mb_advantages.mean().item():.6f}, "
                                    f"std={mb_advantages.std().item():.6f}, "
                                    f"min={mb_advantages.min().item():.6f}, "
                                    f"max={mb_advantages.max().item():.6f}\n")
                            if self.grpo_guard:
                                if paper_impl:
                                    f.write(f"[GRPO-Guard paper] sigma_t: {sigma_t.item():.6f}\n")
                                    f.write(f"[GRPO-Guard paper] ratio_mean_bias: mean={ratio_mean_bias.mean().item():.6f}\n")
                                    f.write(f"[GRPO-Guard paper] sqrt_dt: {mb_new_sqrt_dt.mean().item():.6f}\n")
                                else:
                                    # Flow-Factory-like GRPO-Guard
                                    f.write(f"[GRPO-Guard ff] scale_factor: mean={scale_factor.mean().item():.6f}, min={scale_factor.min().item():.6f}, max={scale_factor.max().item():.6f}\n")
                                    f.write(f"[GRPO-Guard ff] mse: mean={mse.mean().item():.6f}\n")
                            f.flush()

                    # Use clip_range parameter for ratio clipping (PPO-style)
                    # Match Flow-Factory semantics:
                    # - if clip_range is scalar c: clamp(ratio, 1-c, 1+c)
                    # - if clip_range is (low, high): clamp(ratio, 1+low, 1+high)
                    unclipped_loss = -mb_advantages * ratio
                    if paper_impl and isinstance(self.clip_range, (tuple, list)):
                        raise ValueError(
                            "ATRAIN_GRPOGUARD_PAPER_IMPL=1 expects scalar clip_range (flow_grpo semantics). "
                            f"Got clip_range={self.clip_range}"
                        )
                    if isinstance(self.clip_range, (tuple, list)) and len(self.clip_range) == 2:
                        low, high = float(self.clip_range[0]), float(self.clip_range[1])
                        ratio_low = 1.0 + low
                        ratio_high = 1.0 + high
                    else:
                        c = float(self.clip_range)
                        ratio_low = 1.0 - c
                        ratio_high = 1.0 + c
                    clipped_loss = -mb_advantages * torch.clamp(ratio, ratio_low, ratio_high)
                    mb_policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                    # GRPO-Guard gradient reweight is only used in the legacy (paper-style) implementation.
                    if self.grpo_guard and os.environ.get("ATRAIN_GRPOGUARD_PAPER_IMPL", "0") == "1":
                        mb_policy_loss = mb_policy_loss / (mb_new_sqrt_dt.mean() ** 2)

                    # Total micro-batch loss with proper scaling
                    mb_loss = (mb_policy_loss + self.beta * mb_kl_loss) * loss_scale

                    # Debug log: loss components
                    if self.accelerator.is_main_process:
                        with open(debug_path, "a") as f:
                            f.write(f"[7] unclipped_loss (-adv * ratio): mean={unclipped_loss.mean().item():.6f}, "
                                    f"min={unclipped_loss.min().item():.6f}, "
                                    f"max={unclipped_loss.max().item():.6f}\n")
                            f.write(f"[8] clipped_loss: mean={clipped_loss.mean().item():.6f}, "
                                    f"min={clipped_loss.min().item():.6f}, "
                                    f"max={clipped_loss.max().item():.6f}\n")
                            f.write(f"[9] mb_policy_loss (mean of max): {mb_policy_loss.item():.6f}\n")
                            f.write(f"[10] mb_kl_loss: {mb_kl_loss.item():.6f}\n")
                            f.write(f"[11] loss_scale: {loss_scale}\n")
                            f.write(f"[12] mb_loss ((policy + beta*kl) * scale): {mb_loss.item():.6f}\n")
                            f.write(f"[13] clip_range: {self.clip_range}, beta: {self.beta}\n")
                            if self.grpo_guard:
                                f.write(f"[GRPO-Guard] highclip_range: {self.grpo_guard_highclip_range}\n")
                            f.write(f"-" * 40 + "\n")
                            f.flush()

                # End of DiffusionNFT / GRPO loss branch
                # ----- Backward pass (gradients accumulate) -----
                mb_loss_scaled = mb_loss * grpo_mix
                if _gradnorm_should_measure:
                    # To log grad-norm for beta*KL (after scaling), we split the GRPO backward
                    # into two passes on measurement steps:
                    #   grad(total) = grad(policy) + grad(beta*KL)
                    # This keeps training gradients identical while enabling component attribution.
                    has_kl = bool(has_ref) and (self.beta > 0)
                    try:
                        if has_kl and mb_kl_loss.requires_grad:
                            mb_policy_loss_scaled = (mb_policy_loss * loss_scale) * grpo_mix
                            mb_beta_kl_loss_scaled = (self.beta * mb_kl_loss * loss_scale) * grpo_mix

                            # Policy part contributes to GRPO total.
                            _gradnorm_active["grpo"] = True
                            _gradnorm_active["grpo_no_kl"] = True
                            _gradnorm_active["kl"] = False
                            _gradnorm_active["sft"] = False
                            self.accelerator.backward(mb_policy_loss_scaled, retain_graph=True)

                            # beta*KL part contributes to both GRPO total and KL component.
                            _gradnorm_active["grpo"] = True
                            _gradnorm_active["grpo_no_kl"] = False
                            _gradnorm_active["kl"] = True
                            _gradnorm_active["sft"] = False
                            self.accelerator.backward(mb_beta_kl_loss_scaled)
                        else:
                            # No KL (beta=0 or no reference): do the usual single backward.
                            _gradnorm_active["grpo"] = True
                            _gradnorm_active["grpo_no_kl"] = True
                            _gradnorm_active["kl"] = False
                            _gradnorm_active["sft"] = False
                            self.accelerator.backward(mb_loss_scaled)
                    finally:
                        _gradnorm_active["grpo"] = False
                        _gradnorm_active["grpo_no_kl"] = False
                        _gradnorm_active["kl"] = False
                        _gradnorm_active["sft"] = False
                else:
                    self.accelerator.backward(mb_loss_scaled)
                total_forward_backward_count += 1

                # Record for logging (detached)
                total_grpo_loss_unscaled_for_logging += mb_loss.detach().item()
                total_grpo_loss_scaled_for_logging += mb_loss_scaled.detach().item()
                total_loss_for_logging += mb_loss_scaled.detach().item()
                update_grpo_loss += mb_loss_scaled.detach().item()  # Per-update tracking

                # ----- Free GPU memory -----
                del mb_prev_latents, mb_pred_latents, mb_timesteps, mb_old_log_probs
                del mb_model_pred
                del mb_policy_loss, mb_kl_loss, mb_loss, mb_loss_scaled, mb_advantages
                # Clean up algorithm-specific variables
                if self.diffusion_nft:
                    # DiffusionNFT-specific cleanup
                    # Note: Variables like old_v_pred, new_v_pred, etc. are local to the NFT branch
                    pass
                else:
                    # GRPO-specific cleanup
                    del mb_log_prob_policy
                    del log_prob_diff, ratio, unclipped_loss, clipped_loss
                # Clean up embeddings slices (they are views but may hold references)
                del mb_pooled_cond, mb_seq_cond
                if mb_pooled_uncond is not None:
                    del mb_pooled_uncond
                if mb_seq_uncond is not None:
                    del mb_seq_uncond
                # Clean up KL-related variables if they exist
                if self.beta > 0 and (self.ref_model is not None or self.ref_transformer is not None):
                    if not self.diffusion_nft:
                        # Only delete these for non-NFT (they may be used differently in NFT)
                        del mb_ref_pred, mb_mean_policy, mb_std_policy, mb_kl
                        # mb_mean_ref only exists for x-based KL
                        if self.atrain_kl_type == "x-based":
                            del mb_mean_ref
                    else:
                        # DiffusionNFT cleanup for KL-related variables
                        if 'mb_ref_pred' in dir():
                            del mb_ref_pred
                        if 'mb_kl' in dir():
                            del mb_kl
                    # Only the full ref_model path creates separate reference embeddings.
                    if self.ref_model is not None:
                        del mb_ref_pooled_cond, mb_ref_seq_cond
                        if mb_ref_pooled_uncond is not None:
                            del mb_ref_pooled_uncond
                        if mb_ref_seq_uncond is not None:
                            del mb_ref_seq_uncond
                # Clean up GRPO-Guard variables
                if self.grpo_guard:
                    del mb_old_prev_sample_mean, mb_sqrt_dt
                    del mb_prev_sample_mean, mb_std_dev_t, mb_new_sqrt_dt

            # Free per-timestep CPU references
            del step_prev_latents_cpu, step_pred_latents_cpu, step_timesteps_cpu, step_old_log_probs_cpu
            # Clean up GRPO-Guard per-timestep CPU references
            if self.grpo_guard:
                del step_prev_sample_means_cpu, step_sqrt_dts_cpu

            # Force CUDA cache cleanup after each timestep to prevent memory accumulation
            torch.cuda.empty_cache()

            # ===== Per-update metrics logging (actor_update/*) =====
            #
            # We keep fine-grained curves for each actor update step, but we DO NOT use W&B's
            # internal `step` for this. Instead:
            # - train/*, eval/* are logged with step=global_step and plotted vs `global_step`
            # - actor_update/* are plotted vs `actor_update/step` (see wandb.define_metric)
            #
            # This avoids desynchronizing train/eval curves while preserving per-update visibility.
            self._actor_update_count += 1
            if num_actor_update_steps > 1 and self.accelerator.is_main_process:
                num_forward_passes_this_update = num_grad_steps * current_num_micro_batches
                update_avg_grpo_loss = (
                    update_grpo_loss / num_forward_passes_this_update
                    if num_forward_passes_this_update > 0
                    else 0.0
                )
                update_avg_kl = torch.cat(update_kl_values).mean().item() if update_kl_values else 0.0

                per_update_metrics = {
                    "global_step": int(current_step),
                    "actor_update/step": int(self._actor_update_count),
                    "actor_update/update_idx": int(actor_update_idx),
                    "actor_update/grpo_loss": float(update_avg_grpo_loss),
                    "actor_update/kl": float(update_avg_kl),
                }

                report_to = getattr(self.args, "report_to", None)

                # W&B: log without overriding global step to avoid changing train/eval x-axis.
                if _is_reporting_to(report_to, "wandb") and is_wandb_available() and wandb.run is not None:
                    try:
                        if not getattr(self, "_wandb_metrics_defined", False):
                            try:
                                wandb.define_metric("global_step")
                                wandb.define_metric("actor_update/step")
                                wandb.define_metric("train/*", step_metric="global_step")
                                wandb.define_metric("eval/*", step_metric="global_step")
                                wandb.define_metric("actor_update/*", step_metric="actor_update/step")
                            finally:
                                self._wandb_metrics_defined = True
                        wandb.log(per_update_metrics)
                    except Exception as e:
                        rank0_print(f"Warning: Failed to log actor_update metrics to wandb: {e}")

                # SwanLab: use actor_update/step as step for fine-grained curves.
                if _is_reporting_to(report_to, "swanlab") and is_swanlab_available():
                    try:
                        import swanlab
                        if swanlab.get_run() is not None:
                            swanlab.log(per_update_metrics, step=int(self._actor_update_count))
                    except Exception as e:
                        rank0_print(f"Warning: Failed to log actor_update metrics to swanlab: {e}")

        # Optional: SFT-Aux loss (one extra batch per step) with convex mixing.
        if sft_mix > 0.0 and (current_step % int(self.sftaux_every_n_steps) == 0):
            sft_start = time.perf_counter()
            try:
                sft_batch = self._next_sftaux_batch()
                if sft_batch is not None:
                    sft_loss = self._compute_sftaux_loss(model, sft_batch)
                    sft_loss_scaled = sft_loss * sft_mix
                    if _gradnorm_should_measure:
                        _gradnorm_active["grpo"] = False
                        _gradnorm_active["kl"] = False
                        _gradnorm_active["sft"] = True
                    self.accelerator.backward(sft_loss_scaled)
                    if _gradnorm_should_measure:
                        _gradnorm_active["grpo"] = False
                        _gradnorm_active["kl"] = False
                        _gradnorm_active["sft"] = False
                    did_sftaux = True
                    sftaux_loss_unscaled_for_logging = float(sft_loss.detach().item())
                    sftaux_loss_scaled_for_logging = float(sft_loss_scaled.detach().item())
                    total_loss_for_logging += sftaux_loss_scaled_for_logging
                    del sft_loss, sft_loss_scaled
                else:
                    sft_batch = None
            finally:
                sft_duration_ms = (time.perf_counter() - sft_start) * 1000
                self.phase_logger.sub_phase(
                    current_step,
                    "SFT-Aux backward",
                    duration_ms=sft_duration_ms,
                    enabled=(sft_mix > 0.0),
                    lambda_coef=f"{sft_mix:.4f}",
                    did_sftaux=did_sftaux,
                )
                try:
                    del sft_batch
                except NameError:
                    pass

        # Phase 5 end logging
        phase5_duration = (time.perf_counter() - phase5_start) * 1000
        avg_kl = torch.cat(all_kl_values).mean().item() if all_kl_values else 0.0
        self.phase_logger.log_stats(
            current_step,
            forward_backward_count=total_forward_backward_count,
            num_actor_update_steps=num_actor_update_steps,
            samples_per_update=samples_per_update,
            avg_kl=f"{avg_kl:.6f}",
            grpo_mix=f"{grpo_mix:.4f}",
            sft_mix=f"{sft_mix:.4f}",
            grpo_loss_scaled=f"{total_grpo_loss_scaled_for_logging:.6f}",
            sft_loss_scaled=f"{sftaux_loss_scaled_for_logging:.6f}",
            accumulated_loss=f"{total_loss_for_logging:.6f}",
        )
        self.phase_logger.phase_end(current_step, 5, "MICRO_BATCH_GRADIENT_ACCUMULATION", duration_ms=phase5_duration)

        # Clean up NFT old_v_pred cache to free memory
        if nft_old_v_pred_cache:
            del nft_old_v_pred_cache
            torch.cuda.empty_cache()

        # ================================================================
        # Phase 6: Optimization (handled by HuggingFace Trainer after training_step)
        # ================================================================
        # Note: The actual optimizer.step() is called by the Trainer base class
        # after training_step returns. We log this phase here for completeness.
        self.phase_logger.phase_start(current_step, 6, "OPTIMIZATION")
        # Optimization will happen after this method returns
        # We'll log the end in training_step

        # ================================================================
        # Phase 7: Logging and cleanup
        # ================================================================
        self.phase_logger.phase_start(current_step, 7, "LOGGING_AND_CLEANUP")
        phase7_start = time.perf_counter()

        # Logging metrics
        mean_reward = self.accelerator.gather_for_metrics(rewards).mean().item()
        self._metrics["reward"].append(mean_reward)
        mean_kl = 0.0
        if all_kl_values:
            all_kl_tensor = torch.cat(all_kl_values)
            mean_kl = self.accelerator.gather_for_metrics(all_kl_tensor).mean().item()
            self._metrics["kl"].append(mean_kl)
        # Record average loss per actor update step (for comparable metrics across different num_actor_update_steps)
        avg_policy_loss = total_loss_for_logging / num_actor_update_steps if num_actor_update_steps > 0 else total_loss_for_logging
        avg_grpo_loss = total_grpo_loss_scaled_for_logging / num_actor_update_steps if num_actor_update_steps > 0 else total_grpo_loss_scaled_for_logging
        self._metrics["policy_loss"].append(avg_policy_loss)
        self._metrics["grpo_loss"].append(avg_grpo_loss)
        # Also record total loss for reference (useful for debugging)
        self._metrics["policy_loss_total"].append(total_loss_for_logging)
        self._metrics["grpo_loss_total"].append(total_grpo_loss_scaled_for_logging)
        if sft_mix > 0.0:
            self._metrics["sftaux_loss"].append(sftaux_loss_scaled_for_logging)

        # Finalize component grad-norm proxy metrics and remove hooks.
        if (
            _gradnorm_should_measure
            and _grpo_grad_sq is not None
            and _grpo_no_kl_grad_sq is not None
            and _sft_grad_sq is not None
            and _kl_grad_sq is not None
        ):
            try:
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(_grpo_grad_sq, op=dist.ReduceOp.SUM)
                    dist.all_reduce(_grpo_no_kl_grad_sq, op=dist.ReduceOp.SUM)
                    dist.all_reduce(_kl_grad_sq, op=dist.ReduceOp.SUM)
                    dist.all_reduce(_sft_grad_sq, op=dist.ReduceOp.SUM)
                grpo_gn = float(torch.sqrt(torch.clamp(_grpo_grad_sq, min=0.0)).detach().item())
                grpo_no_kl_gn = float(torch.sqrt(torch.clamp(_grpo_no_kl_grad_sq, min=0.0)).detach().item())
                kl_gn = float(torch.sqrt(torch.clamp(_kl_grad_sq, min=0.0)).detach().item())
                sft_gn = float(torch.sqrt(torch.clamp(_sft_grad_sq, min=0.0)).detach().item())
                self._metrics["grad_norm/grpo"].append(grpo_gn)
                self._metrics["grad_norm/grpo_no_kl"].append(grpo_no_kl_gn)
                # NOTE: This is the grad norm of the (beta * KL) term AFTER all scaling
                # applied in training (loss_scale and GRPO mix coefficient).
                self._metrics["grad_norm/beta_kl"].append(kl_gn)
                self._metrics["grad_norm/sftaux"].append(sft_gn)
            except Exception:
                # Never fail training for logging-only metrics.
                pass
            finally:
                for h in _gradnorm_handles:
                    try:
                        h.remove()
                    except Exception:
                        pass

        for i, (func_name, _, _) in enumerate(self.reward_funcs):
            # Debug log before gather (all ranks)
            _debug_ur = os.environ.get("DEBUG_UNIFIED_REWARD", "0") == "1"
            if _debug_ur and func_name == "unifiedreward_think":
                _rank = self.accelerator.process_index
                env_log_dir = os.environ.get("LOG_DIR", ".")
                debug_path = os.path.join(env_log_dir, f"debug_unifiedreward_think_rank{_rank}.log")
                local_mean = rewards_per_func[:, i].mean().item()
                gathered = self.accelerator.gather_for_metrics(rewards_per_func[:, i])
                gathered_mean = gathered.mean().item()
                with open(debug_path, "a") as f:
                    f.write(f"\n  [Gather Debug] Rank {_rank}: local_mean={local_mean:.4f}, gathered_mean={gathered_mean:.4f}, gathered_shape={gathered.shape}")
                    f.flush()

            self._metrics[f"reward/{func_name}"].append(
                self.accelerator.gather_for_metrics(rewards_per_func[:, i]).mean().item()
            )

        # Debug log
        if self.accelerator.is_main_process:
            env_log_dir = os.environ.get("LOG_DIR", ".")
            debug_path = os.path.join(env_log_dir, "debug_loss.log")
            with open(debug_path, "a") as f:
                f.write(f"\n=== Step {self.state.global_step} Loss ===\n")
                f.write(f"total_loss: {total_loss_for_logging:.6f}\n")
                f.write(f"grpo_loss_scaled: {total_grpo_loss_scaled_for_logging:.6f} (mix={grpo_mix:.4f})\n")
                if sft_mix > 0.0:
                    f.write(
                        f"sftaux_loss_scaled: {sftaux_loss_scaled_for_logging:.6f} "
                        f"(mix={sft_mix:.4f}, did={did_sftaux})\n"
                    )
                f.flush()

        # Clean up CPU tensors
        del prev_latents, pred_latents, timesteps_traj, log_probs_traj
        # Clean up GRPO-Guard trajectory data
        if self.grpo_guard:
            del prev_sample_means_traj, sqrt_dts_traj
        del pooled_cond, pooled_uncond, seq_cond, seq_uncond
        del ref_pooled_cond, ref_pooled_uncond, ref_seq_cond, ref_seq_uncond

        # Clean up GPU tensors to prevent memory leak
        del rewards, rewards_per_func, advantages
        if all_kl_values:
            del all_kl_values
        # Check if all_kl_tensor was created (only when all_kl_values was non-empty)
        try:
            del all_kl_tensor
        except NameError:
            pass  # all_kl_tensor was not created

        # Clean up images list (PIL images on CPU, but may hold references)
        del images

        # Clean up string lists and other Python objects
        del prompts_text, cfg_prompts, expanded_prompts, expanded_cfg_prompts
        del inputs

        # Force CUDA memory cleanup to prevent memory leak across steps
        torch.cuda.empty_cache()

        phase7_duration = (time.perf_counter() - phase7_start) * 1000
        self.phase_logger.phase_end(current_step, 7, "LOGGING_AND_CLEANUP", duration_ms=phase7_duration)

        # Log step summary
        self.phase_logger.log_stats(
            current_step,
            mean_reward=f"{mean_reward:.4f}",
            mean_kl=f"{mean_kl:.6f}",
            grpo_loss=f"{total_grpo_loss_scaled_for_logging:.6f}",
            sftaux_loss=f"{sftaux_loss_scaled_for_logging:.6f}",
            total_loss=f"{total_loss_for_logging:.6f}",
        )

        # Return a dummy tensor that requires grad but won't trigger backward
        # because training_step will be overridden to skip backward
        # The actual gradients have already been accumulated above
        return torch.tensor(total_loss_for_logging, device=device, requires_grad=False)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None):
        """Log metrics with custom tracking."""
        metrics = {key: sum(val) / len(val) if val else 0.0 for key, val in self._metrics.items()}
        logs = {**logs, **metrics}

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)

        self._metrics.clear()

    # ============================================================================
    # Evaluation Methods
    # ============================================================================

    def _generate_images_ode(
        self,
        model,
        prompts: List[str],
        cfg_prompts: List[str],
        num_inference_steps: int,
        cfg_scale: float,
        height: int,
        width: int,
        seeds: Optional[List[int]] = None,
    ) -> List[Image.Image]:
        """
        Generate images using ODE (deterministic) inference.

        This method performs deterministic ODE sampling without noise injection.
        Uses GatheredParameters for DeepSpeed ZeRO-3 compatibility.

        Args:
            model: The DeepGen model
            prompts: List of text prompts
            cfg_prompts: List of CFG prompts (empty strings for unconditional)
            num_inference_steps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale
            height: Output image height
            width: Output image width
            seeds: Optional list of random seeds for each prompt (for reproducibility)

        Returns:
            List of PIL Images
        """
        import os
        debug_device = os.environ.get('DEBUG_EVAL_DEVICEISSUE', '0') == '1'

        device = self.accelerator.device
        dtype = torch.bfloat16 if self.args.bf16 else torch.float32
        batch_size = len(prompts)

        # Get the underlying model if wrapped by DeepSpeed/Accelerator
        unwrapped_model = model.module if hasattr(model, 'module') else model

        # Debug: Log device information before fixing
        if debug_device:
            log_dir = os.environ.get('LOG_DIR', '/tmp')
            rank = self.accelerator.process_index
            debug_log_path = os.path.join(log_dir, f'device_debug_ode_rank{rank}.log')
            with open(debug_log_path, 'w') as f:
                f.write(f"=== Device Debug Log (ODE) - Rank {rank} ===\n")
                f.write(f"Target device: {device}\n")
                f.write(f"Accelerator device: {self.accelerator.device}\n\n")

                f.write("=== Full Model Devices (BEFORE fix) ===\n")
                cpu_modules = []
                for name, module in unwrapped_model.named_modules():
                    for param_name, param in module.named_parameters(recurse=False):
                        if param.device.type == 'cpu':
                            cpu_modules.append(f"  {name}.{param_name}: {param.device}")
                    for buf_name, buf in module.named_buffers(recurse=False):
                        if buf is not None and buf.device.type == 'cpu':
                            cpu_modules.append(f"  {name}.{buf_name} (buffer): {buf.device}")

                if cpu_modules:
                    f.write(f"Found {len(cpu_modules)} parameters/buffers on CPU:\n")
                    for item in cpu_modules[:50]:  # Limit to first 50
                        f.write(f"{item}\n")
                    if len(cpu_modules) > 50:
                        f.write(f"  ... and {len(cpu_modules) - 50} more\n")
                else:
                    f.write("All parameters/buffers are on GPU\n")
                f.write("\n")

        # Ensure rotary embedding inv_freq buffers are on the correct device
        # This fixes the device mismatch issue with Qwen2.5-VL's rotary_emb.inv_freq
        for name, module in unwrapped_model.named_modules():
            if hasattr(module, 'inv_freq') and module.inv_freq is not None:
                if module.inv_freq.device != device:
                    module.inv_freq = module.inv_freq.to(device)

        # Move ALL parameters and buffers of the ENTIRE model to the correct device
        # This includes llm, projector_1, connector, transformer, etc.
        for name, module in unwrapped_model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.device != device:
                    param.data = param.data.to(device)
            for buf_name, buf in module.named_buffers(recurse=False):
                if buf is not None and buf.device != device:
                    setattr(module, buf_name, buf.to(device))

        # Debug: Log device information after fixing
        if debug_device:
            with open(debug_log_path, 'a') as f:
                f.write("=== Full Model Devices (AFTER fix) ===\n")
                cpu_modules_after = []
                for name, module in unwrapped_model.named_modules():
                    for param_name, param in module.named_parameters(recurse=False):
                        if param.device.type == 'cpu':
                            cpu_modules_after.append(f"  {name}.{param_name}: {param.device}")
                    for buf_name, buf in module.named_buffers(recurse=False):
                        if buf is not None and buf.device.type == 'cpu':
                            cpu_modules_after.append(f"  {name}.{buf_name} (buffer): {buf.device}")

                if cpu_modules_after:
                    f.write(f"Still found {len(cpu_modules_after)} parameters/buffers on CPU:\n")
                    for item in cpu_modules_after[:50]:
                        f.write(f"{item}\n")
                else:
                    f.write("All parameters/buffers are now on GPU\n")
                f.write("\n")

        # IMPORTANT:
        # For evaluation alignment with the original SFT pipeline, we reuse the model's
        # own `generate()` implementation (which internally uses StableDiffusion3Pipeline).
        # The previous custom ODE loop can diverge from the pipeline's internal logic
        # (e.g., scheduler specifics / model-input scaling), leading to noticeable quality gaps.

        # Prepare per-sample generators for reproducibility (Diffusers supports list[Generator])
        generator = None
        if seeds is not None and len(seeds) == batch_size:
            generator = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]

        with torch.no_grad():
            images_tensor = unwrapped_model.generate(
                prompt=prompts,
                cfg_prompt=cfg_prompts,
                pixel_values_src=None,
                cfg_scale=cfg_scale,
                num_steps=num_inference_steps,
                generator=generator,
                height=height,
                width=width,
                progress_bar=False,
            )

        # Convert decoded pixels ([-1, 1] range) to PIL images
        images_tensor = rearrange(images_tensor, "b c h w -> b h w c")
        images_tensor = torch.clamp(127.5 * images_tensor + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
        return [Image.fromarray(img) for img in images_tensor]

    def _generate_images_sde(
        self,
        model,
        prompts: List[str],
        cfg_prompts: List[str],
        num_inference_steps: int,
        cfg_scale: float,
        height: int,
        width: int,
        sde_eta: float = 1.0,
        seeds: Optional[List[int]] = None,
    ) -> List[Image.Image]:
        """
        Generate images using SDE (stochastic) inference.

        This method uses the same SDE sampling process as training rollout,
        with noise injection at each step for diversity.

        Args:
            model: The DeepGen model
            prompts: List of text prompts
            cfg_prompts: List of CFG prompts
            num_inference_steps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale
            height: Output image height
            width: Output image width
            sde_eta: Noise scale factor for SDE
            seeds: Optional list of random seeds for each prompt (for reproducibility)

        Returns:
            List of PIL Images
        """
        import os
        debug_device = os.environ.get('DEBUG_EVAL_DEVICEISSUE', '0') == '1'

        device = self.accelerator.device
        dtype = torch.bfloat16 if self.args.bf16 else torch.float32
        batch_size = len(prompts)

        # Get the underlying model if wrapped by DeepSpeed/Accelerator
        unwrapped_model = model.module if hasattr(model, 'module') else model

        # Debug: Log device information before fixing
        if debug_device:
            log_dir = os.environ.get('LOG_DIR', '/tmp')
            rank = self.accelerator.process_index
            debug_log_path = os.path.join(log_dir, f'device_debug_sde_rank{rank}.log')
            with open(debug_log_path, 'w') as f:
                f.write(f"=== Device Debug Log (SDE) - Rank {rank} ===\n")
                f.write(f"Target device: {device}\n")
                f.write(f"Accelerator device: {self.accelerator.device}\n\n")

                f.write("=== Full Model Devices (BEFORE fix) ===\n")
                cpu_modules = []
                for name, module in unwrapped_model.named_modules():
                    for param_name, param in module.named_parameters(recurse=False):
                        if param.device.type == 'cpu':
                            cpu_modules.append(f"  {name}.{param_name}: {param.device}")
                    for buf_name, buf in module.named_buffers(recurse=False):
                        if buf is not None and buf.device.type == 'cpu':
                            cpu_modules.append(f"  {name}.{buf_name} (buffer): {buf.device}")

                if cpu_modules:
                    f.write(f"Found {len(cpu_modules)} parameters/buffers on CPU:\n")
                    for item in cpu_modules[:50]:  # Limit to first 50
                        f.write(f"{item}\n")
                    if len(cpu_modules) > 50:
                        f.write(f"  ... and {len(cpu_modules) - 50} more\n")
                else:
                    f.write("All parameters/buffers are on GPU\n")
                f.write("\n")

        # Ensure rotary embedding inv_freq buffers are on the correct device
        # This fixes the device mismatch issue with Qwen2.5-VL's rotary_emb.inv_freq
        for name, module in unwrapped_model.named_modules():
            if hasattr(module, 'inv_freq') and module.inv_freq is not None:
                if module.inv_freq.device != device:
                    module.inv_freq = module.inv_freq.to(device)

        # Move ALL parameters and buffers of the ENTIRE model to the correct device
        # This includes llm, projector_1, connector, transformer, etc.
        for name, module in unwrapped_model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.device != device:
                    param.data = param.data.to(device)
            for buf_name, buf in module.named_buffers(recurse=False):
                if buf is not None and buf.device != device:
                    setattr(module, buf_name, buf.to(device))

        # Debug: Log device information after fixing
        if debug_device:
            with open(debug_log_path, 'a') as f:
                f.write("=== Full Model Devices (AFTER fix) ===\n")
                cpu_modules_after = []
                for name, module in unwrapped_model.named_modules():
                    for param_name, param in module.named_parameters(recurse=False):
                        if param.device.type == 'cpu':
                            cpu_modules_after.append(f"  {name}.{param_name}: {param.device}")
                    for buf_name, buf in module.named_buffers(recurse=False):
                        if buf is not None and buf.device.type == 'cpu':
                            cpu_modules_after.append(f"  {name}.{buf_name} (buffer): {buf.device}")

                if cpu_modules_after:
                    f.write(f"Still found {len(cpu_modules_after)} parameters/buffers on CPU:\n")
                    for item in cpu_modules_after[:50]:
                        f.write(f"{item}\n")
                else:
                    f.write("All parameters/buffers are now on GPU\n")
                f.write("\n")

        # Check if we need CFG
        need_cfg = cfg_scale > 0

        # Helper function to prepare LLM inputs manually on the correct device
        # This bypasses model's prepare_forward_input which uses self.device internally
        def prepare_llm_inputs_manual(query_embeds, input_ids, attention_mask, target_device):
            """Manually prepare LLM inputs ensuring all tensors are on the correct device."""
            b, l, _ = query_embeds.shape

            # Ensure all tensors are on the correct device
            input_ids = input_ids.to(target_device)
            attention_mask = attention_mask.to(target_device, dtype=torch.bool)
            query_embeds = query_embeds.to(target_device)

            # Extend input_ids and attention_mask for query tokens
            input_ids_extended = torch.cat([input_ids, input_ids.new_zeros(b, l)], dim=1)
            attention_mask_extended = torch.cat([attention_mask, attention_mask.new_ones(b, l)], dim=1)

            # Get position_ids using the model's method
            position_ids, _ = unwrapped_model.lmm.get_rope_index(
                input_ids=input_ids_extended,
                image_grid_thw=None,
                video_grid_thw=None,
                second_per_grid_ts=None,
                attention_mask=attention_mask_extended,
            )
            position_ids = position_ids.to(target_device)

            # Get input embeddings - move input_ids to embedding device, then move result back
            input_ids_for_embed = input_ids_extended[:, :-l]
            embedding_layer = unwrapped_model.llm.get_input_embeddings()
            # Get the actual weight device (handle PEFT wrapper)
            if hasattr(embedding_layer, 'base_layer'):
                embed_device = embedding_layer.base_layer.weight.device
            else:
                embed_device = embedding_layer.weight.device
            inputs_embeds = embedding_layer(input_ids_for_embed.to(embed_device))
            inputs_embeds = inputs_embeds.to(target_device)

            # Concatenate with query embeddings
            inputs_embeds = torch.cat([inputs_embeds, query_embeds], dim=1)

            return dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask_extended,
                position_ids=position_ids,
            )

        # Compute text embeddings
        if need_cfg:
            text_inputs = unwrapped_model.prepare_text2image_prompts(prompts + cfg_prompts)
            query_embeds = unwrapped_model.meta_queries[None].expand(2 * batch_size, unwrapped_model.num_queries, -1).clone()

            # Manually prepare inputs on the correct device
            inputs = prepare_llm_inputs_manual(
                query_embeds=query_embeds,
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask'],
                target_device=device,
            )

            with torch.no_grad():
                output = unwrapped_model.llm(**inputs, return_dict=True, output_hidden_states=True)
                pooled_out, seq_out = unwrapped_model.llm2dit(output.hidden_states)

            pooled_cond = pooled_out[:batch_size]
            pooled_uncond = pooled_out[batch_size:]
            seq_cond = seq_out[:batch_size]
            seq_uncond = seq_out[batch_size:]
        else:
            text_inputs = unwrapped_model.prepare_text2image_prompts(prompts)
            query_embeds = unwrapped_model.meta_queries[None].expand(batch_size, unwrapped_model.num_queries, -1).clone()

            # Manually prepare inputs on the correct device
            inputs = prepare_llm_inputs_manual(
                query_embeds=query_embeds,
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask'],
                target_device=device,
            )

            with torch.no_grad():
                output = unwrapped_model.llm(**inputs, return_dict=True, output_hidden_states=True)
                pooled_cond, seq_cond = unwrapped_model.llm2dit(output.hidden_states)
            pooled_uncond = seq_uncond = None

        # Initialize latents
        latent_channels = unwrapped_model.transformer.config.in_channels
        latent_height = height // 8
        latent_width = width // 8

        # Generate latents with optional per-sample seeds for reproducibility
        # Also create generators for SDE noise injection
        generators = None
        if seeds is not None and len(seeds) == batch_size:
            # Generate latents with individual seeds for each sample
            latents_list = []
            generators = []
            for i, seed in enumerate(seeds):
                generator = torch.Generator(device=device).manual_seed(seed)
                latent = torch.randn(
                    (1, latent_channels, latent_height, latent_width),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                latents_list.append(latent)
                # Create a new generator with offset seed for SDE noise
                generators.append(torch.Generator(device=device).manual_seed(seed + 1000000))
            latents = torch.cat(latents_list, dim=0)
        else:
            latents = torch.randn(
                (batch_size, latent_channels, latent_height, latent_width),
                device=device,
                dtype=dtype,
            )

        # Setup scheduler - use model's test_scheduler to match original pipeline
        scheduler = unwrapped_model.test_scheduler

        # Calculate dynamic shifting mu based on image size (matching original pipeline)
        scheduler_kwargs = {}
        if scheduler.config.get("use_dynamic_shifting", None):
            patch_size = unwrapped_model.transformer.config.patch_size
            image_seq_len = (latent_height // patch_size) * (latent_width // patch_size)

            # Calculate shift using the same formula as original pipeline
            base_seq_len = scheduler.config.get("base_image_seq_len", 256)
            max_seq_len = scheduler.config.get("max_image_seq_len", 4096)
            base_shift = scheduler.config.get("base_shift", 0.5)
            max_shift = scheduler.config.get("max_shift", 1.16)

            m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            b = base_shift - m * base_seq_len
            mu = image_seq_len * m + b
            scheduler_kwargs["mu"] = mu

        scheduler.set_timesteps(num_inference_steps, device=device, **scheduler_kwargs)

        # SDE sampling loop
        with torch.no_grad():
            for i, t in enumerate(scheduler.timesteps):
                timestep = t.expand(batch_size)

                # Model prediction
                model_pred_cond = unwrapped_model.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=seq_cond,
                    pooled_projections=pooled_cond,
                    timestep=timestep,
                    return_dict=False,
                )[0]

                if need_cfg:
                    model_pred_uncond = unwrapped_model.transformer(
                        hidden_states=latents,
                        encoder_hidden_states=seq_uncond,
                        pooled_projections=pooled_uncond,
                        timestep=timestep,
                        return_dict=False,
                    )[0]
                    model_pred = model_pred_uncond + cfg_scale * (model_pred_cond - model_pred_uncond)
                else:
                    model_pred = model_pred_cond

                # SDE step based on sampler type
                sigma = scheduler.sigmas[i]
                sigma_prev = scheduler.sigmas[i + 1] if i < len(scheduler.sigmas) - 1 else torch.tensor(0.0, device=device)
                sigma_max = scheduler.sigmas[1].item()
                dt = sigma_prev - sigma

                if self.atrain_sde_sampler == "cps_sde":
                    # Flow-CPS (Coefficients-Preserving Sampling)
                    # Reference: https://arxiv.org/abs/2509.05952
                    std_dev_t = sigma_prev * math.sin(sde_eta * math.pi / 2)
                    # pred_original_sample = latents - sigma * model_pred (predicted x_0)
                    pred_original_sample = latents - sigma * model_pred
                    # noise_estimate = latents + model_pred * (1 - sigma) (predicted x_1)
                    noise_estimate = latents + model_pred * (1 - sigma)
                    # prev_sample_mean = x_0 * (1 - sigma_prev) + x_1 * sqrt(sigma_prev^2 - std_dev_t^2)
                    # Note: No clamp to match original FlowCPS exactly
                    prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)
                    # For CPS, noise scale is just std_dev_t
                    noise_scale = std_dev_t
                    # Add SDE noise (use generators if provided for reproducibility)
                    if generators is not None:
                        noise_list = []
                        for gen in generators:
                            noise = torch.randn(
                                (1,) + latents.shape[1:],
                                device=device,
                                dtype=dtype,
                                generator=gen,
                            )
                            noise_list.append(noise)
                        variance_noise = torch.cat(noise_list, dim=0)
                    else:
                        variance_noise = torch.randn_like(latents)
                    latents = prev_sample_mean + noise_scale * variance_noise
                elif self.atrain_sde_sampler == "dance_sde":
                    # Dance-SDE
                    # Reference: Flow-Factory flow_match_euler_discrete.py
                    # pred_original_sample = latents - sigma * model_pred (predicted x_0)
                    pred_original_sample = latents - sigma * model_pred
                    # std_dev_t = sde_eta * sqrt(-dt)
                    std_dev_t = sde_eta * torch.sqrt(-1 * dt)
                    # log_term correction: 0.5 * sde_eta^2 * (latents - x_0 * (1 - sigma)) / sigma^2
                    log_term = 0.5 * (sde_eta ** 2) * (latents - pred_original_sample * (1 - sigma)) / (sigma ** 2 + 1e-8)
                    # prev_sample_mean = latents + (model_pred + log_term) * dt
                    prev_sample_mean = latents + (model_pred + log_term) * dt
                    noise_scale = std_dev_t
                    # Add SDE noise (use generators if provided for reproducibility)
                    if generators is not None:
                        noise_list = []
                        for gen in generators:
                            noise = torch.randn(
                                (1,) + latents.shape[1:],
                                device=device,
                                dtype=dtype,
                                generator=gen,
                            )
                            noise_list.append(noise)
                        variance_noise = torch.cat(noise_list, dim=0)
                    else:
                        variance_noise = torch.randn_like(latents)
                    latents = prev_sample_mean + noise_scale * variance_noise
                else:
                    # Flow-SDE (Standard flow matching SDE) - flowgrpo_sde
                    sigma_safe = torch.where(sigma == 1, torch.tensor(sigma_max, device=device), sigma)
                    std_dev_t = torch.sqrt(sigma / (1 - sigma_safe)) * sde_eta
                    prev_sample_mean = (
                        latents * (1 + std_dev_t**2 / (2 * sigma) * dt)
                        + model_pred * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
                    )
                    # For Flow-SDE, noise scale is std_dev_t * sqrt(-dt)
                    noise_scale = std_dev_t * torch.sqrt(-1 * dt)
                    # Add SDE noise (use generators if provided for reproducibility)
                    if generators is not None:
                        noise_list = []
                        for gen in generators:
                            noise = torch.randn(
                                (1,) + latents.shape[1:],
                                device=device,
                                dtype=dtype,
                                generator=gen,
                            )
                            noise_list.append(noise)
                        variance_noise = torch.cat(noise_list, dim=0)
                    else:
                        variance_noise = torch.randn_like(latents)
                    latents = prev_sample_mean + noise_scale * variance_noise

        # Decode to images
        with torch.no_grad():
            images_tensor = unwrapped_model.latents_to_pixels(latents)

        images_tensor = rearrange(images_tensor, "b c h w -> b h w c")
        images_tensor = torch.clamp(127.5 * images_tensor + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

        images = [Image.fromarray(img) for img in images_tensor]
        return images

    def _get_eval_samples_for_rank(self, dataset: EvalPromptDataset) -> List[Dict[str, Any]]:
        """
        Get the subset of evaluation samples for the current distributed rank.

        Splits the dataset across all GPUs to ensure no overlap and full coverage.
        Each rank gets samples at indices: rank, rank + world_size, rank + 2*world_size, ...

        Args:
            dataset: The evaluation dataset

        Returns:
            List of samples for this rank
        """
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Get samples for this rank using strided indexing
        samples = []
        for i in range(rank, len(dataset), world_size):
            samples.append(dataset[i])

        return samples

    def _init_eval_upload_indices(self, force_reinit: bool = False) -> None:
        """
        [DEPRECATED] This function is now a no-op.

        Upload indices are now lazily initialized during the first evaluate() call,
        based on actually generated images, and saved to a persistent file.

        See: _load_or_create_eval_upload_indices()
        """
        pass

    def _get_eval_upload_indices_path(self) -> str:
        """Get the path to the eval upload indices file."""
        out_dir = getattr(self.args, "output_dir", None) or "."
        return os.path.join(out_dir, "eval_upload_indices.json")

    def _load_eval_upload_indices(self) -> bool:
        """
        Load eval upload indices from file if it exists.

        Returns:
            True if indices were successfully loaded, False otherwise.
        """
        indices_path = self._get_eval_upload_indices_path()
        if not os.path.exists(indices_path):
            if self._debug_swanlab_vis:
                self._debug_swanlab_vis_write(
                    f"[Eval Upload] Indices file not found: {indices_path}\n"
                )
            return False

        try:
            with open(indices_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert lists back to sets of tuples
            self.eval_upload_indices_wandb = {}
            self.eval_upload_indices_swanlab = {}

            for dataset_name, indices in data.get("wandb", {}).items():
                self.eval_upload_indices_wandb[dataset_name] = set(tuple(x) for x in indices)
            for dataset_name, indices in data.get("swanlab", {}).items():
                self.eval_upload_indices_swanlab[dataset_name] = set(tuple(x) for x in indices)

            self._eval_upload_indices_initialized = True

            rank0_print(f"[Eval Upload] Loaded indices from: {indices_path}")
            if self._debug_swanlab_vis:
                self._debug_swanlab_vis_write(
                    f"[Eval Upload] Loaded indices from: {indices_path}\n"
                )
                for dataset_name in self.eval_upload_indices_swanlab:
                    indices = sorted(self.eval_upload_indices_swanlab[dataset_name])
                    self._debug_swanlab_vis_write(
                        f"[Eval Upload] Loaded swanlab indices for '{dataset_name}': count={len(indices)}, "
                        f"indices={indices}\n"
                    )
            return True
        except Exception as e:
            rank0_print(f"[Eval Upload] Warning: failed to load indices from {indices_path}: {e}")
            if self._debug_swanlab_vis:
                self._debug_swanlab_vis_write(
                    f"[Eval Upload] Warning: failed to load indices: {e}\n"
                )
            return False

    def _save_eval_upload_indices(self) -> None:
        """Save eval upload indices to file for persistence across evaluate() calls."""
        indices_path = self._get_eval_upload_indices_path()

        # Convert sets of tuples to lists for JSON serialization
        data = {
            "wandb": {
                dataset_name: sorted(list(indices))
                for dataset_name, indices in self.eval_upload_indices_wandb.items()
            },
            "swanlab": {
                dataset_name: sorted(list(indices))
                for dataset_name, indices in self.eval_upload_indices_swanlab.items()
            },
            "metadata": {
                "num_wandb_images": self.eval_wandb_num_upload_images,
                "num_swanlab_images": self.eval_swanlab_num_upload_images,
            }
        }

        try:
            os.makedirs(os.path.dirname(indices_path), exist_ok=True)
            with open(indices_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            rank0_print(f"[Eval Upload] Saved indices to: {indices_path}")
            if self._debug_swanlab_vis:
                self._debug_swanlab_vis_write(
                    f"[Eval Upload] Saved indices to: {indices_path}\n"
                )
        except Exception as e:
            rank0_print(f"[Eval Upload] Warning: failed to save indices to {indices_path}: {e}")

    def _select_eval_upload_indices_from_generated(
        self,
        dataset_name: str,
        generated_pairs: List[Tuple[int, int]]
    ) -> None:
        """
        Select random upload indices from actually generated image pairs.

        This is called during the first evaluate() on rank 0, after images are generated.
        The selected indices are saved to file for use in subsequent evaluate() calls.

        Args:
            dataset_name: Name of the dataset
            generated_pairs: List of (idx, dup_idx) pairs that were actually generated
        """
        rng = random.Random(42)  # Fixed seed for reproducibility

        # Randomly select for wandb
        num_wandb = min(self.eval_wandb_num_upload_images, len(generated_pairs))
        if num_wandb > 0:
            wandb_selected = set(rng.sample(generated_pairs, num_wandb))
        else:
            wandb_selected = set()
        self.eval_upload_indices_wandb[dataset_name] = wandb_selected

        # Randomly select for swanlab (independent selection with same seed for different count)
        rng2 = random.Random(43)  # Different seed for swanlab to get different selection
        num_swanlab = min(self.eval_swanlab_num_upload_images, len(generated_pairs))
        if num_swanlab > 0:
            swanlab_selected = set(rng2.sample(generated_pairs, num_swanlab))
        else:
            swanlab_selected = set()
        self.eval_upload_indices_swanlab[dataset_name] = swanlab_selected

        rank0_print(
            f"[Eval Upload] Selected indices for '{dataset_name}': "
            f"wandb={num_wandb}/{len(generated_pairs)}, swanlab={num_swanlab}/{len(generated_pairs)}"
        )
        if self._debug_swanlab_vis:
            self._debug_swanlab_vis_write(
                f"[Eval Upload] Selected indices for '{dataset_name}' from {len(generated_pairs)} generated images:\n"
                f"  - wandb: {num_wandb} images, indices={sorted(wandb_selected)}\n"
                f"  - swanlab: {num_swanlab} images, indices={sorted(swanlab_selected)}\n"
            )

    def _debug_swanlab_vis_write(self, msg: str) -> None:
        """
        Write debug logs for eval visualization selection/upload to output_dir.

        Enabled via env var: DEBUG_SWANLAB_VIS=1
        Only writes on global rank 0.
        """
        if not self._debug_swanlab_vis or not is_rank_zero():
            return
        try:
            out_dir = getattr(self.args, "output_dir", None) or "."
            debug_dir = os.path.join(out_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            log_path = os.path.join(debug_dir, "debug_swanlab_vis.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg)
        except Exception:
            # Never crash training due to debug logging
            pass

    def evaluate(self, step: int = 0) -> None:
        """
        Run evaluation on all configured evaluation datasets.

        If --ema_diffusion is enabled, evaluation uses EMA diffusion transformer weights
        temporarily (restored afterwards).
        """
        if not self.eval_enabled or len(self.eval_datasets) == 0:
            return

        if self.ema_diffusion is None:
            return self._evaluate_impl(step=step)

        with self._swap_in_ema_diffusion_weights():
            return self._evaluate_impl(step=step)

    def _evaluate_impl(self, step: int = 0) -> None:
        """
        Run evaluation on all configured evaluation datasets.

        Generates images for all prompts in each evaluation dataset, saves them
        to disk organized by step and dataset name, and optionally uploads
        visualization images to wandb.

        Args:
            step: Current training step (used for output directory naming)
        """
        if not self.eval_enabled or len(self.eval_datasets) == 0:
            return

        model = self.model
        device = self.accelerator.device
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Try to load upload indices from file (only on rank 0)
        # If file doesn't exist, indices will be selected after first image generation
        if rank == 0 and not self._eval_upload_indices_initialized:
            self._load_eval_upload_indices()

        # Log evaluation start
        rank0_print(f"\n{'='*60}")
        rank0_print(f"Starting Evaluation at Step {step}")
        rank0_print(f"{'='*60}")
        rank0_print(f"Inference mode: {self.eval_inference_mode}")
        rank0_print(f"CFG scale: {self.eval_cfg_scale}")
        rank0_print(f"Num steps: {self.eval_num_inference_steps}")
        rank0_print(f"Image size: {self.eval_image_height}x{self.eval_image_width}")
        if self.eval_inference_mode == "sde":
            rank0_print(f"SDE eta: {self.eval_sde_eta}")
        rank0_print(f"Micro batch size: {self.eval_micro_batch_size}")
        rank0_print(f"Number of datasets: {len(self.eval_datasets)}")
        rank0_print(f"Upload indices initialized: {self._eval_upload_indices_initialized}")
        rank0_print(f"{'='*60}")

        # Set model to eval mode
        model.eval()

        # Collect images for logging upload (only from rank 0's assigned prompts)
        # Separate collections for different backends as they may have different limits
        wandb_images_to_upload = {}   # dataset_name -> list of (prompt, image, idx, dup_idx) tuples
        swanlab_images_to_upload = {}  # dataset_name -> list of (prompt, image, idx, dup_idx) tuples

        # Track all generated pairs for each dataset (for first-time index selection)
        all_generated_pairs = {}  # dataset_name -> list of (idx, dup_idx)
        # Also track all generated images for first-time selection
        all_generated_images = {}  # dataset_name -> list of (prompt, img, idx, dup_idx, base_name)

        for dataset in self.eval_datasets:
            dataset_name = dataset.dataset_name

            # Get per-dataset inference parameters, fall back to global defaults
            ds_cfg_scale = dataset.cfg_scale if dataset.cfg_scale is not None else self.eval_cfg_scale
            ds_num_inference_steps = dataset.num_inference_steps if dataset.num_inference_steps is not None else self.eval_num_inference_steps

            # Get samples assigned to this rank
            rank_samples = self._get_eval_samples_for_rank(dataset)
            num_local_samples = len(rank_samples)

            # Log dataset info with per-dataset parameters
            param_info = f"cfg_scale={ds_cfg_scale}, steps={ds_num_inference_steps}"
            rank0_print(f"\nEvaluating dataset '{dataset_name}': {len(dataset)} total prompts, {num_local_samples} for rank {rank} ({param_info})")

            # Create output directory
            output_dir = os.path.join(self.args.output_dir, "eval", f"step_{step}", dataset_name)
            os.makedirs(output_dir, exist_ok=True)

            # Process samples in micro-batches
            num_generated = 0
            progress_bar = tqdm(
                total=num_local_samples,
                desc=f"[Rank {rank}] Generating {dataset_name}",
                disable=rank != 0  # Only show progress on rank 0
            )

            for batch_start in range(0, num_local_samples, self.eval_micro_batch_size):
                batch_end = min(batch_start + self.eval_micro_batch_size, num_local_samples)
                batch_samples = rank_samples[batch_start:batch_end]

                prompts = [s["prompt"] for s in batch_samples]
                indices = [s["index"] for s in batch_samples]
                cfg_prompts = [""] * len(prompts)  # Empty for unconditional

                # Calculate deterministic seeds for each sample based on (index, duplicate_idx)
                # This ensures different duplicates get different seeds, but results are reproducible
                seeds = []
                for s in batch_samples:
                    idx = s["index"]
                    dup_idx = s.get("duplicate_idx", 0)
                    # Combine index and duplicate_idx into a unique seed
                    # Using a large multiplier to avoid collisions
                    seed = idx * 1000 + dup_idx + 42  # 42 is a base offset
                    seeds.append(seed)

                # Generate images based on inference mode
                if self.eval_inference_mode == "ode":
                    images = self._generate_images_ode(
                        model=model,
                        prompts=prompts,
                        cfg_prompts=cfg_prompts,
                        num_inference_steps=ds_num_inference_steps,
                        cfg_scale=ds_cfg_scale,
                        height=self.eval_image_height,
                        width=self.eval_image_width,
                        seeds=seeds,
                    )
                else:  # sde
                    images = self._generate_images_sde(
                        model=model,
                        prompts=prompts,
                        cfg_prompts=cfg_prompts,
                        num_inference_steps=ds_num_inference_steps,
                        cfg_scale=ds_cfg_scale,
                        height=self.eval_image_height,
                        width=self.eval_image_width,
                        sde_eta=self.eval_sde_eta,
                        seeds=seeds,
                    )

                # Save images
                for img, sample, prompt in zip(images, batch_samples, prompts):
                    idx = sample["index"]
                    dup_idx = sample.get("duplicate_idx", 0)
                    # Use 'key' or 'index' from metadata if available (priority: key > index), otherwise use sample index
                    metadata = sample.get("metadata", {})
                    if "key" in metadata:
                        base_name = str(metadata['key'])
                    elif "index" in metadata:
                        base_name = str(metadata['index'])
                    else:
                        base_name = str(idx)
                    # Add duplicate index suffix: xxxx.0.png, xxxx.1.png, etc.
                    filename = f"{base_name}.{dup_idx}.png"
                    img_path = os.path.join(output_dir, filename)
                    img.save(img_path)
                    num_generated += 1

                    # Track generated pair for first-time index selection
                    if not self._eval_upload_indices_initialized:
                        if dataset_name not in all_generated_pairs:
                            all_generated_pairs[dataset_name] = []
                            all_generated_images[dataset_name] = []
                        all_generated_pairs[dataset_name].append((idx, dup_idx))
                        all_generated_images[dataset_name].append((prompt, img, idx, dup_idx, base_name))
                    else:
                        # Indices already initialized, collect directly
                        # Collect for wandb upload (pre-selected random samples)
                        if (idx, dup_idx) in self.eval_upload_indices_wandb.get(dataset_name, set()):
                            if dataset_name not in wandb_images_to_upload:
                                wandb_images_to_upload[dataset_name] = []
                            # Check for duplicates before adding
                            existing_keys = {(it[2], it[3]) for it in wandb_images_to_upload[dataset_name]}
                            if (idx, dup_idx) not in existing_keys:
                                wandb_images_to_upload[dataset_name].append((prompt, img, idx, dup_idx, base_name))
                                if self._debug_swanlab_vis:
                                    self._debug_swanlab_vis_write(
                                        f"[Eval Collect] wandb: dataset='{dataset_name}' idx={idx} dup_idx={dup_idx} "
                                        f"base_name='{base_name}' prompt='{prompt[:50]}...'\n"
                                    )

                        # Collect for swanlab upload (pre-selected random samples, separate selection)
                        if (idx, dup_idx) in self.eval_upload_indices_swanlab.get(dataset_name, set()):
                            if dataset_name not in swanlab_images_to_upload:
                                swanlab_images_to_upload[dataset_name] = []
                            # Check for duplicates before adding
                            existing_keys = {(it[2], it[3]) for it in swanlab_images_to_upload[dataset_name]}
                            if (idx, dup_idx) not in existing_keys:
                                swanlab_images_to_upload[dataset_name].append((prompt, img, idx, dup_idx, base_name))
                                if self._debug_swanlab_vis:
                                    self._debug_swanlab_vis_write(
                                        f"[Eval Collect] swanlab: dataset='{dataset_name}' idx={idx} dup_idx={dup_idx} "
                                        f"base_name='{base_name}' prompt='{prompt[:50]}...'\n"
                                    )

                progress_bar.update(len(batch_samples))

            progress_bar.close()

            rank0_print(f"Generated {num_generated} images for dataset '{dataset_name}'")

            # If this is the first evaluation, select indices from actually generated images
            if not self._eval_upload_indices_initialized and dataset_name in all_generated_pairs:
                rank0_print(f"[Eval Upload] First evaluation: selecting upload indices from {len(all_generated_pairs[dataset_name])} generated images")
                self._select_eval_upload_indices_from_generated(
                    dataset_name,
                    all_generated_pairs[dataset_name]
                )
                # Now collect images based on selected indices
                for prompt, img, idx, dup_idx, base_name in all_generated_images[dataset_name]:
                    if (idx, dup_idx) in self.eval_upload_indices_wandb.get(dataset_name, set()):
                        if dataset_name not in wandb_images_to_upload:
                            wandb_images_to_upload[dataset_name] = []
                        # Check for duplicates before adding
                        existing_keys = {(it[2], it[3]) for it in wandb_images_to_upload[dataset_name]}
                        if (idx, dup_idx) not in existing_keys:
                            wandb_images_to_upload[dataset_name].append((prompt, img, idx, dup_idx, base_name))
                            if self._debug_swanlab_vis:
                                self._debug_swanlab_vis_write(
                                    f"[Eval Collect] wandb: dataset='{dataset_name}' idx={idx} dup_idx={dup_idx} "
                                    f"base_name='{base_name}' prompt='{prompt[:50]}...'\n"
                                )
                    if (idx, dup_idx) in self.eval_upload_indices_swanlab.get(dataset_name, set()):
                        if dataset_name not in swanlab_images_to_upload:
                            swanlab_images_to_upload[dataset_name] = []
                        # Check for duplicates before adding
                        existing_keys = {(it[2], it[3]) for it in swanlab_images_to_upload[dataset_name]}
                        if (idx, dup_idx) not in existing_keys:
                            swanlab_images_to_upload[dataset_name].append((prompt, img, idx, dup_idx, base_name))
                            if self._debug_swanlab_vis:
                                self._debug_swanlab_vis_write(
                                    f"[Eval Collect] swanlab: dataset='{dataset_name}' idx={idx} dup_idx={dup_idx} "
                                    f"base_name='{base_name}' prompt='{prompt[:50]}...'\n"
                                )

            # Log collection summary for this dataset
            wandb_collected = len(wandb_images_to_upload.get(dataset_name, []))
            swanlab_collected = len(swanlab_images_to_upload.get(dataset_name, []))
            wandb_expected = len(self.eval_upload_indices_wandb.get(dataset_name, set()))
            swanlab_expected = len(self.eval_upload_indices_swanlab.get(dataset_name, set()))
            rank0_print(
                f"[Eval Upload] Dataset '{dataset_name}': collected wandb={wandb_collected}/{wandb_expected}, "
                f"swanlab={swanlab_collected}/{swanlab_expected}"
            )
            if self._debug_swanlab_vis:
                # Log detailed selection info
                wandb_indices = self.eval_upload_indices_wandb.get(dataset_name, set())
                swanlab_indices = self.eval_upload_indices_swanlab.get(dataset_name, set())
                self._debug_swanlab_vis_write(
                    f"[Eval Upload Summary] Dataset '{dataset_name}': step={step}\n"
                    f"  - Generated {num_generated} images on this rank\n"
                    f"  - Wandb: collected {wandb_collected}/{wandb_expected}, indices={sorted(wandb_indices)}\n"
                    f"  - Swanlab: collected {swanlab_collected}/{swanlab_expected}, indices={sorted(swanlab_indices)}\n"
                )

        # After processing all datasets, save indices if this was the first evaluation
        if not self._eval_upload_indices_initialized and rank == 0:
            self._save_eval_upload_indices()
            self._eval_upload_indices_initialized = True
            rank0_print("[Eval Upload] Upload indices initialized and saved for subsequent evaluations")

        # Synchronize all ranks before continuing
        if dist.is_initialized():
            dist.barrier()

        # If UniGenBench scoring is enabled and both vLLM servers share the same host,
        # switch to the UniGenBench server before scoring to avoid GPU OOM.
        # Note: We keep all ranks inside evaluation by adding a barrier after scoring
        # and switching back to UnifiedReward before training resumes.
        needs_unigenbench_scoring = any(
            getattr(getattr(ds, "scoring_config", None), "type", None) == "unigenbench"
            for ds in (self.eval_datasets or [])
        )
        use_exclusive_vllm = needs_unigenbench_scoring and is_exclusive_vllm_active()
        if rank == 0 and use_exclusive_vllm:
            maybe_switch_vllm_server("unigenbench")

        # Run scoring evaluations (only on rank 0 to avoid duplicate API calls)
        scoring_metrics = {}
        if rank == 0:
            scoring_metrics = self._run_scoring_evaluations(step, self.eval_datasets)

        # Upload images and log metrics (only on rank 0)
        if rank == 0:
            # Upload evaluation images to configured logging backend(s)
            self._log_eval_images(step, wandb_images_to_upload, swanlab_images_to_upload)
            # Log scoring metrics using the trainer's log mechanism
            if scoring_metrics:
                self._log_eval_metrics(scoring_metrics, step=step)

        # Switch back to UnifiedReward before training resumes.
        if rank == 0 and use_exclusive_vllm:
            maybe_switch_vllm_server("unifiedreward")
        # Ensure all ranks wait for rank 0 scoring/logging and vLLM switching.
        if dist.is_initialized() and use_exclusive_vllm:
            dist.barrier()

        # Set model back to train mode
        model.train()

        # Clean up GPU memory after evaluation to prevent memory leak
        gc.collect()
        torch.cuda.empty_cache()

        rank0_print(f"\n{'='*60}")
        rank0_print(f"Evaluation at Step {step} Complete")
        rank0_print(f"Results saved to: {os.path.join(self.args.output_dir, 'eval', f'step_{step}')}")
        rank0_print(f"{'='*60}\n")

    def _run_scoring_evaluations(
        self,
        step: int,
        datasets: List[EvalPromptDataset],
    ) -> Dict[str, float]:
        """
        Run scoring evaluations for datasets that have scoring configurations.

        Currently supports:
        - unigenbench: UniGenBench evaluation using VLM judge model

        Args:
            step: Current training step
            datasets: List of evaluation datasets

        Returns:
            Dict of metric_name -> value for wandb logging
        """
        all_metrics = {}

        for dataset in datasets:
            # Check if this dataset has scoring configuration
            if not hasattr(dataset, 'scoring_config') or not dataset.scoring_config:
                continue

            scoring_config = dataset.scoring_config
            dataset_name = dataset.dataset_name
            output_dir = os.path.join(self.args.output_dir, "eval", f"step_{step}", dataset_name)

            if scoring_config.type == "unigenbench":
                # Check if UniGenBench API is available
                if not is_unigenbench_enabled():
                    rank0_print(
                        f"[UniGenBench] Skipping scoring for '{dataset_name}': "
                        f"UNIGENBENCH_API_URL not set"
                    )
                    continue

                try:
                    rank0_print(f"\n[UniGenBench] Running scoring for dataset '{dataset_name}'...")

                    # Create scorer with CSV path from the dataset
                    # The CSV file contains testpoints info needed for evaluation
                    # Pass language from scoring config (default: "en")
                    language = getattr(scoring_config, 'language', 'en')
                    scorer = UniGenBenchScorer(csv_path=dataset.file_path, language=language)

                    # Score images
                    stats = scorer.score_images(
                        image_dir=output_dir,
                        num_duplicates=dataset.duplicates,
                        show_progress=True,
                    )

                    if stats.get("success", False):
                        # Print results
                        scorer.print_results(stats)

                        # Format metrics for wandb
                        metrics = scorer.format_wandb_metrics(
                            stats,
                            prefix=f"eval/{dataset_name}",
                        )
                        all_metrics.update(metrics)

                        # Save detailed results to CSV
                        self._save_scoring_results(step, dataset_name, stats)
                    else:
                        rank0_print(f"[UniGenBench] Scoring failed for '{dataset_name}': {stats.get('error', 'Unknown error')}")

                except Exception as e:
                    rank0_print(f"[UniGenBench] Error scoring '{dataset_name}': {e}")
                    import traceback
                    traceback.print_exc()

        return all_metrics

    def _save_scoring_results(
        self,
        step: int,
        dataset_name: str,
        stats: Dict[str, Any],
    ) -> None:
        """
        Save detailed scoring results to CSV file.

        Args:
            step: Current training step
            dataset_name: Name of the dataset
            stats: Statistics from UniGenBenchScorer.score_images()
        """
        import pandas as pd

        output_dir = os.path.join(self.args.output_dir, "eval", f"step_{step}", dataset_name)
        csv_path = os.path.join(output_dir, "unigenbench_scores.csv")

        try:
            results_csv = stats.get("results_csv", [])
            if results_csv:
                df = pd.DataFrame(results_csv)
                df.to_csv(csv_path, index=False)
                rank0_print(f"[UniGenBench] Saved detailed results to {csv_path}")

            # Also save summary statistics
            summary_path = os.path.join(output_dir, "unigenbench_summary.json")
            summary = {
                "overall_accuracy": stats.get("overall_accuracy", 0.0),
                "success_rate": stats.get("success_rate", 0.0),
                "total_correct": stats.get("total_correct", 0),
                "total_count": stats.get("total_count", 0),
                "primary_dims": stats.get("primary_dims", {}),
                "sub_dims": stats.get("sub_dims", {}),
            }
            import json
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            rank0_print(f"[UniGenBench] Saved summary to {summary_path}")

        except Exception as e:
            rank0_print(f"[UniGenBench] Warning: Failed to save results: {e}")

    def _log_metrics_to_trackers(
        self,
        metrics: Dict[str, float],
        step: int,
    ) -> None:
        """
        Log training metrics to the configured logging backend(s).

        This is a lightweight, fault-tolerant logger used inside the training loop.
        It intentionally avoids raising exceptions so logging failures do not crash training.
        """
        if not metrics:
            return

        report_to = getattr(self.args, "report_to", None)
        # Always expose the rollout/optimization step as a metric so W&B charts can use it
        # as the canonical x-axis (instead of relying on W&B internal `_step`).
        if "global_step" not in metrics:
            metrics["global_step"] = step

        # Log to wandb if enabled
        if _is_reporting_to(report_to, "wandb") and is_wandb_available() and wandb.run is not None:
            try:
                # Define W&B metric x-axes once per run.
                # - train/*, eval/*: use global_step (rollout/opt step)
                # - actor_update/*: use actor_update/step (fine-grained curve)
                if not getattr(self, "_wandb_metrics_defined", False):
                    try:
                        wandb.define_metric("global_step")
                        wandb.define_metric("actor_update/step")
                        wandb.define_metric("train/*", step_metric="global_step")
                        wandb.define_metric("eval/*", step_metric="global_step")
                        wandb.define_metric("actor_update/*", step_metric="actor_update/step")
                    finally:
                        # Never crash training due to logging configuration.
                        self._wandb_metrics_defined = True
                wandb.log(metrics, step=step)
            except Exception as e:
                rank0_print(f"Warning: Failed to log metrics to wandb: {e}")

        # Log to tensorboard if enabled
        if _is_reporting_to(report_to, "tensorboard") and is_tensorboard_available():
            try:
                tb_writer = getattr(self, "tb_writer", None)
                if tb_writer is None:
                    for callback in self.callback_handler.callbacks:
                        if hasattr(callback, "tb_writer") and callback.tb_writer is not None:
                            tb_writer = callback.tb_writer
                            break

                if tb_writer is not None:
                    for key, value in metrics.items():
                        tb_writer.add_scalar(key, value, step)
            except Exception as e:
                rank0_print(f"Warning: Failed to log metrics to tensorboard: {e}")

        # Log to swanlab if enabled
        if _is_reporting_to(report_to, "swanlab") and is_swanlab_available():
            try:
                import swanlab
                if swanlab.get_run() is not None:
                    swanlab.log(metrics, step=step)
            except Exception as e:
                rank0_print(f"Warning: Failed to log metrics to swanlab: {e}")

    def _log_eval_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
    ) -> None:
        """
        Log evaluation metrics to the configured logging backend(s).

        This method provides a unified interface for logging metrics regardless of
        which backend (wandb, tensorboard, etc.) is configured.

        Args:
            metrics: Dict of metric_name -> value
            step: Current training step
        """
        if not metrics:
            return

        report_to = getattr(self.args, 'report_to', None)
        logged_to = []
        # Keep eval curves aligned with training rollout/optimization step.
        if "global_step" not in metrics:
            metrics["global_step"] = step

        # Log to wandb if enabled
        if _is_reporting_to(report_to, "wandb") and is_wandb_available() and wandb.run is not None:
            try:
                # Ensure metric definitions exist even if eval happens before train() begins.
                if not getattr(self, "_wandb_metrics_defined", False):
                    try:
                        wandb.define_metric("global_step")
                        wandb.define_metric("actor_update/step")
                        wandb.define_metric("train/*", step_metric="global_step")
                        wandb.define_metric("eval/*", step_metric="global_step")
                        wandb.define_metric("actor_update/*", step_metric="actor_update/step")
                    finally:
                        self._wandb_metrics_defined = True
                wandb.log(metrics, step=step)
                logged_to.append("wandb")
            except Exception as e:
                rank0_print(f"Warning: Failed to log metrics to wandb: {e}")

        # Log to tensorboard if enabled
        if _is_reporting_to(report_to, "tensorboard") and is_tensorboard_available():
            try:
                # Try to get the SummaryWriter from the trainer's tb_writer
                tb_writer = getattr(self, 'tb_writer', None)
                if tb_writer is None:
                    # Try to find it in the callback handler
                    for callback in self.callback_handler.callbacks:
                        if hasattr(callback, 'tb_writer') and callback.tb_writer is not None:
                            tb_writer = callback.tb_writer
                            break

                if tb_writer is not None:
                    for key, value in metrics.items():
                        tb_writer.add_scalar(key, value, step)
                    logged_to.append("tensorboard")
            except Exception as e:
                rank0_print(f"Warning: Failed to log metrics to tensorboard: {e}")

        # Log to swanlab if enabled
        if _is_reporting_to(report_to, "swanlab") and is_swanlab_available():
            try:
                import swanlab
                if swanlab.get_run() is not None:
                    swanlab.log(metrics, step=step)
                    logged_to.append("swanlab")
            except Exception as e:
                rank0_print(f"Warning: Failed to log metrics to swanlab: {e}")

        if logged_to:
            rank0_print(f"Logged {len(metrics)} eval metrics to: {', '.join(logged_to)}")

    def _log_eval_images(
        self,
        step: int,
        wandb_images_by_dataset: Dict[str, List[tuple]],
        swanlab_images_by_dataset: Optional[Dict[str, List[tuple]]] = None,
    ) -> None:
        """
        Log evaluation images to the configured logging backend(s).

        Images are grouped by prompt to enable cross-step comparison of the same
        prompt's generations. Supports wandb, tensorboard, and swanlab backends.

        Selection logic: Images are selected based on prompt index (idx).
        Only prompts with idx < eval_xxx_num_upload_images are uploaded.
        All duplicate images (dup_idx) for selected prompts are included.

        Args:
            step: Current training step
            wandb_images_by_dataset: Dict for wandb/tensorboard, dataset_name -> list of (prompt, image, idx, dup_idx, key)
            swanlab_images_by_dataset: Dict for swanlab, dataset_name -> list of (prompt, image, idx, dup_idx, key)
        """
        if swanlab_images_by_dataset is None:
            swanlab_images_by_dataset = {}

        if not wandb_images_by_dataset and not swanlab_images_by_dataset:
            return

        report_to = getattr(self.args, 'report_to', None)
        logged_to = []
        total_wandb = sum(len(v) for v in wandb_images_by_dataset.values())
        total_swanlab = sum(len(v) for v in swanlab_images_by_dataset.values())
        if self._debug_swanlab_vis:
            self._debug_swanlab_vis_write(
                f"[Eval Upload step={step}] collected: wandb/tensorboard={total_wandb}, swanlab={total_swanlab}\n"
            )

        # Log to wandb if enabled
        if _is_reporting_to(report_to, "wandb") and is_wandb_available() and wandb.run is not None and wandb_images_by_dataset:
            try:
                for dataset_name, image_data in wandb_images_by_dataset.items():
                    # Sort by index and dup_idx to ensure consistent ordering
                    image_data.sort(key=lambda x: (x[2], x[3]) if len(x) > 3 else (x[2], 0))

                    # Create wandb images with captions including key, index, and prompt
                    wandb_images = []
                    for item in image_data:
                        if len(item) >= 5:
                            prompt, img, idx, dup_idx, key = item
                        elif len(item) == 4:
                            prompt, img, idx, dup_idx = item
                            key = str(idx)
                        else:
                            prompt, img, idx = item
                            dup_idx = 0
                            key = str(idx)
                        # Caption format: [key.dup_idx] prompt
                        wandb_images.append(wandb.Image(img, caption=f"[{key}.{dup_idx}] {prompt}"))

                    # Log to wandb with dataset-specific key
                    log_key = f"eval/{dataset_name}"
                    wandb.log({log_key: wandb_images}, step=step)

                logged_to.append("wandb")
            except Exception as e:
                rank0_print(f"Warning: Failed to upload evaluation images to wandb: {e}")

        # Log to tensorboard if enabled (uses wandb image set as they share the same limit)
        if _is_reporting_to(report_to, "tensorboard") and is_tensorboard_available() and wandb_images_by_dataset:
            try:
                import numpy as np

                # Try to get the SummaryWriter from the trainer's tb_writer
                tb_writer = getattr(self, 'tb_writer', None)
                if tb_writer is None:
                    # Try to find it in the callback handler
                    for callback in self.callback_handler.callbacks:
                        if hasattr(callback, 'tb_writer') and callback.tb_writer is not None:
                            tb_writer = callback.tb_writer
                            break

                if tb_writer is not None:
                    for dataset_name, image_data in wandb_images_by_dataset.items():
                        # Sort by index and dup_idx to ensure consistent ordering
                        image_data.sort(key=lambda x: (x[2], x[3]) if len(x) > 3 else (x[2], 0))

                        # Log each image individually with key as tag
                        for item in image_data:
                            if len(item) >= 5:
                                prompt, img, idx, dup_idx, key = item
                            elif len(item) == 4:
                                prompt, img, idx, dup_idx = item
                                key = str(idx)
                            else:
                                prompt, img, idx = item
                                dup_idx = 0
                                key = str(idx)

                            # Convert PIL Image to numpy array for tensorboard
                            img_array = np.array(img)
                            # TensorBoard expects (C, H, W) format
                            if img_array.ndim == 3:
                                img_array = np.transpose(img_array, (2, 0, 1))

                            tag = f"eval/{dataset_name}/img_{key}_{dup_idx}"
                            tb_writer.add_image(tag, img_array, step)

                    logged_to.append("tensorboard")
            except Exception as e:
                rank0_print(f"Warning: Failed to upload evaluation images to tensorboard: {e}")

        # Log to swanlab if enabled (uses separate image set with its own limit)
        if _is_reporting_to(report_to, "swanlab") and is_swanlab_available() and swanlab_images_by_dataset:
            try:
                import swanlab
                if swanlab.get_run() is not None:
                    for dataset_name, image_data in swanlab_images_by_dataset.items():
                        # Deduplicate by (idx, dup_idx) to avoid uploading the same image twice
                        seen_keys = set()
                        unique_data = []
                        for item in image_data:
                            key = (item[2], item[3]) if len(item) > 3 else (item[2], 0)
                            if key not in seen_keys:
                                seen_keys.add(key)
                                unique_data.append(item)
                        if self._debug_swanlab_vis and len(image_data) != len(unique_data):
                            self._debug_swanlab_vis_write(
                                f"[Eval Upload step={step}] swanlab dedup: {len(image_data)} -> {len(unique_data)} for '{dataset_name}'\n"
                            )
                        image_data = unique_data
                        # Sort by index and dup_idx to ensure consistent ordering
                        image_data.sort(key=lambda x: (x[2], x[3]) if len(x) > 3 else (x[2], 0))

                        # Create swanlab images with captions including key, index, and prompt
                        swanlab_images = []
                        for item in image_data:
                            if len(item) >= 5:
                                prompt, img, idx, dup_idx, key = item
                            elif len(item) == 4:
                                prompt, img, idx, dup_idx = item
                                key = str(idx)
                            else:
                                prompt, img, idx = item
                                dup_idx = 0
                                key = str(idx)
                            # Caption format: [key.dup_idx] prompt
                            swanlab_images.append(swanlab.Image(img, caption=f"[{key}.{dup_idx}] {prompt}"))

                        # Log to swanlab with dataset-specific key
                        log_key = f"eval/{dataset_name}"
                        swanlab.log({log_key: swanlab_images}, step=step)
                        if self._debug_swanlab_vis:
                            keys = [(it[4] if len(it) >= 5 else str(it[2]), it[3] if len(it) >= 4 else 0) for it in image_data]
                            self._debug_swanlab_vis_write(
                                f"[Eval Upload step={step}] swanlab dataset='{dataset_name}' "
                                f"count={len(image_data)} keys={sorted(keys)}\n"
                            )

                    logged_to.append("swanlab")
            except Exception as e:
                rank0_print(f"Warning: Failed to upload evaluation images to swanlab: {e}")
                if self._debug_swanlab_vis:
                    self._debug_swanlab_vis_write(
                        f"[Eval Upload step={step}] swanlab upload failed: {repr(e)}\n"
                    )

        if logged_to:
            # Report upload counts per backend
            counts = []
            if "wandb" in logged_to or "tensorboard" in logged_to:
                counts.append(f"wandb/tensorboard: {total_wandb}")
            if "swanlab" in logged_to:
                counts.append(f"swanlab: {total_swanlab}")
            rank0_print(f"Uploaded evaluation images to: {', '.join(logged_to)} ({', '.join(counts)})")
        elif total_wandb > 0 or total_swanlab > 0:
            # No logging backend available, just note that images were generated but not logged
            rank0_print(f"Generated evaluation images but no logging backend configured")

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Override training_step to skip backward call.

        Our compute_loss method already performs backward passes internally for each
        micro-batch during gradient accumulation. The returned loss is detached and
        only used for logging. Therefore, we must NOT call accelerator.backward()
        again in this method.

        This is essential for the micro-batch gradient accumulation to work correctly
        with DeepSpeed ZeRO, which doesn't support multiple backward calls for the
        same parameter in one optimizer step.

        Args:
            model: The model to train
            inputs: Input batch
            num_items_in_batch: Number of items in batch

        Returns:
            Detached loss tensor for logging
        """
        model.train()
        current_step = self.state.global_step

        # --------------------------------------------------------------
        # TFLOPS logging: mark optimizer-step start (first micro-step)
        # --------------------------------------------------------------
        if self.log_tflops:
            # Start timer only once per optimizer step (global_step remains constant
            # across gradient accumulation micro-steps).
            if self._tflops_step_start_time_s is None or self._tflops_step_id != int(current_step):
                self._tflops_step_start_time_s = time.perf_counter()
                self._tflops_step_id = int(current_step)

                # One-time FLOPs calibration (best-effort) after warmup.
                if (
                    self._tflops_calibrated_flops_per_opt_step is None
                    and not self._tflops_calibration_in_progress
                    and int(current_step) >= int(self.tflops_warmup_steps)
                    and is_rank_zero()
                ):
                    # Detect DeepSpeed FLOPs profiler availability lazily.
                    if self._tflops_profiler_available is None:
                        try:
                            from deepspeed.profiling.flops_profiler import FlopsProfiler  # type: ignore
                            self._tflops_profiler_available = True
                        except Exception:
                            self._tflops_profiler_available = False

                    if self._tflops_profiler_available:
                        try:
                            from deepspeed.profiling.flops_profiler import FlopsProfiler  # type: ignore

                            # Use the underlying wrapped module (DS/Accelerator wrapper is still nn.Module).
                            self._tflops_profiler = FlopsProfiler(model)
                            self._tflops_profiler.start_profile()
                            self._tflops_calibration_in_progress = True
                        except Exception:
                            # Best-effort: if profiler fails, disable calibration.
                            self._tflops_profiler = None
                            self._tflops_calibration_in_progress = False

        # Record optimization start time
        optimization_start = time.perf_counter()

        # compute_loss performs all forward and backward passes internally
        # It returns a detached loss tensor for logging purposes only
        loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        # DO NOT call self.accelerator.backward(loss) here!
        # Backward passes have already been completed inside compute_loss
        # for each micro-batch during gradient accumulation.

        # Phase 6 ends after optimizer step (handled by Trainer base class)
        # We mark it here since the actual optimization happens after this method returns
        optimization_duration = (time.perf_counter() - optimization_start) * 1000
        self.phase_logger.phase_end(
            current_step, 6, "OPTIMIZATION",
            duration_ms=optimization_duration,
            lr=self.get_learning_rate() if hasattr(self, 'get_learning_rate') else "N/A"
        )

        # Mark step end
        self.phase_logger.step_end(
            current_step,
            loss=f"{loss.item():.6f}"
        )

        # Run periodic evaluation if enabled
        # Check after step completion: eval_freq > 0 and (step + 1) % eval_freq == 0
        # Note: current_step is the step we just completed (0-indexed before increment)
        # The actual step number after this training_step is (current_step + 1)
        next_step = current_step + 1
        if self.eval_freq > 0 and self.eval_enabled and next_step % self.eval_freq == 0:
            self.evaluate(step=next_step)

        # Return the detached loss for logging
        # Note: loss is already detached (requires_grad=False) from compute_loss
        return loss.detach()

    def train(self, *args, **kwargs):
        """
        Override train method to support evaluation and ensure PhaseLogger is properly closed.

        This wraps the parent Trainer.train() method to:
        1. Run evaluation before training if eval_before_train is True
        2. Handle resource cleanup for the PhaseLogger
        3. Patch DeepSpeed for checkpoint loading compatibility
        """
        import deepspeed.runtime.engine as ds_engine

        # Monkey patch torch.load to use weights_only=False by default
        # This is required for PyTorch 2.6+ which changed the default to weights_only=True
        # DeepSpeed checkpoints contain many custom classes that need to be unpickled
        import torch
        _original_torch_load = torch.load
        def _patched_torch_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _original_torch_load(*args, **kwargs)
        torch.load = _patched_torch_load

        # Patch DeepSpeed's load_module_state_dict to use strict=False
        # This allows loading checkpoints that are missing keys for frozen modules
        original_load_module_state_dict = ds_engine.DeepSpeedEngine.load_module_state_dict

        def patched_load_module_state_dict(engine_self, checkpoint, strict=True, **kwargs):
            """
            Patched version that always uses strict=False to allow missing keys
            from frozen modules that were not saved during training.
            """
            return original_load_module_state_dict(
                engine_self,
                checkpoint,
                strict=False,  # Force strict=False to allow missing keys
                **kwargs  # Pass through all other arguments (e.g., fetch_z3_params, custom_load_fn)
            )

        # Apply the patch before training starts
        ds_engine.DeepSpeedEngine.load_module_state_dict = patched_load_module_state_dict

        try:
            # Run evaluation before training if enabled
            if self.eval_before_train and self.eval_enabled:
                # Initialize logging backends before eval_before_train so metrics can be logged
                # This triggers the callbacks' on_train_begin() which initializes their loggers
                # (e.g., WandbCallback initializes wandb.run, TensorBoardCallback initializes its writer)
                report_to = getattr(self.args, 'report_to', None)
                needs_init = False

                # Check if any logging backend needs initialization
                if _is_reporting_to(report_to, "wandb") and is_wandb_available() and wandb.run is None:
                    needs_init = True
                elif _is_reporting_to(report_to, "tensorboard") and is_tensorboard_available():
                    # TensorBoard callback is typically initialized via on_train_begin
                    needs_init = True
                elif _is_reporting_to(report_to, "swanlab") and is_swanlab_available():
                    # SwanLab callback is typically initialized via on_train_begin
                    import swanlab
                    if swanlab.get_run() is None:
                        needs_init = True

                if needs_init:
                    # Use the callback handler to initialize logging backends properly
                    # We need to set up minimal state for callbacks to work
                    from transformers.trainer_callback import TrainerState, TrainerControl
                    if not hasattr(self, 'state') or self.state is None:
                        self.state = TrainerState()
                    if not hasattr(self, 'control') or self.control is None:
                        self.control = TrainerControl()
                    # Trigger on_train_begin which initializes all configured loggers
                    self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

                rank0_print("\n" + "="*60)
                rank0_print("Running evaluation BEFORE training starts...")
                rank0_print("="*60)
                self.evaluate(step=0)

            return super().train(*args, **kwargs)
        finally:
            # Restore the original method after training
            ds_engine.DeepSpeedEngine.load_module_state_dict = original_load_module_state_dict
            # Ensure PhaseLogger is closed and flushed
            if hasattr(self, 'phase_logger') and self.phase_logger is not None:
                self.phase_logger.close()

