# Copyright 2025 Ruihang Li and DeepGen Team @ Shanghai Innovation Institute

"""
GRPO training script for DeepGen models (Qwen2p5VLStableDiffusion3DPOFusionHF).

This script provides GRPO training for DeepGen T2I models, loading checkpoints
from DeepGen_Image pretrain using mmengine BUILDER.

Usage:
    torchrun --nproc_per_node=8 -m deepgen_rl.grpo_deepgen \
        --model_config configs/models/qwen2_5_vl_7b_stable_diffusion_3_5_medium_hf_dynamic_dpo_fusion.py \
        --checkpoint /path/to/deepgen_checkpoint \
        --reward_funcs jpeg_compressibility \
        --prompts_file prompts.txt

    # Or with dataset config YAML for multi-dataset training:
    torchrun --nproc_per_node=8 -m deepgen_rl.grpo_deepgen \
        --model_config configs/models/qwen2_5_vl_7b_stable_diffusion_3_5_medium_hf_dynamic_dpo_fusion.py \
        --checkpoint /path/to/deepgen_checkpoint \
        --dataset_config assets/rl_datasets/deepgen/deepgen.yaml
"""

import os
import re
import json
import argparse
import yaml
import base64
import itertools
from collections import defaultdict
from io import BytesIO
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import io
import numpy as np
import torch.distributed as dist
from datasets import load_dataset
from trl import GRPOConfig
from transformers import CLIPModel, CLIPProcessor, CLIPImageProcessor
from PIL import Image

from .trainer import DeepGenGRPOTrainer
from .reward_evaluator.reward_evaluator import RewardEvaluatorClient
from .reward_evaluator.ocr_vllm import ocr_vllm
from .utils.vllm_request import evaluate_batch
from .utils.vllm_sleep_mode import maybe_switch_vllm_server


reward_client = RewardEvaluatorClient()


def _truthy_env(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


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


# ============================================================================
# Reward Functions
# ============================================================================

def recon_reward(orig_images, recon_images):
    """
    Compute rewards based on PSNR between original and reconstructed tensor images.
    Images are assumed to have values normalized between 0 and 1, with maximum value exactly 1.0.

    Args:
        orig_images (torch.Tensor): Batch of original images [B, C, H, W], values in [0, 1].
        recon_images (torch.Tensor): Batch of reconstructed images [B, C, H, W], values in [0, 1].

    Returns:
        list: List of PSNR values as rewards for each image pair.
    """
    def calculate_psnr(img1, img2):
        if img1.shape != img2.shape:
            raise ValueError("Image dimensions must match")

        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        perfect_match = mse == 0
        psnr = torch.zeros_like(mse)
        valid = ~perfect_match
        psnr[valid] = -10 * torch.log10(mse[valid])
        psnr[perfect_match] = float('inf')
        return psnr

    orig_images = orig_images.float().clamp(0, 1)
    recon_images = recon_images.float().clamp(0, 1)
    rewards = calculate_psnr(orig_images, recon_images)
    return rewards


def jpeg_incompressibility(images):
    """Compute JPEG file size as reward (higher = less compressible)."""
    buffers = [io.BytesIO() for _ in images]
    for image, buffer in zip(images, buffers):
        image.save(buffer, format="JPEG", quality=95)
    sizes = [buffer.tell() / 1000 for buffer in buffers]
    return torch.tensor(sizes)


def jpeg_compressibility(images):
    """Compute negative JPEG file size as reward (higher = more compressible)."""
    return jpeg_incompressibility(images) * -1.0 / 500


def sim_direction(orig_images, edited_images, original_caption, edited_caption, clip_model, clip_processor):
    """
    Compute sim_direction using HuggingFace CLIP by calculating cosine similarity
    between the difference of image features and the difference of text features.

    Args:
        orig_images: Original image or list of images (PIL.Image)
        edited_images: Edited image or list of images (PIL.Image)
        original_caption: List of original text prompts
        edited_caption: List of edited text prompts
        clip_model: Preloaded HuggingFace CLIP model
        clip_processor: Preloaded HuggingFace CLIP processor

    Returns:
        torch.Tensor: Cosine similarity of image and text feature differences (sim_direction)
    """
    device = clip_model.device
    inputs_orig = clip_processor(images=orig_images, text=original_caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    inputs_edited = clip_processor(images=edited_images, text=edited_caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    inputs_orig = {k: v.to(device) for k, v in inputs_orig.items()}
    inputs_edited = {k: v.to(device) for k, v in inputs_edited.items()}

    with torch.no_grad():
        image_features_orig = clip_model(**inputs_orig)
        image_features_edited = clip_model(**inputs_edited)

    image_features_orig, text_features_orig = image_features_orig.image_embeds, image_features_orig.text_embeds
    image_features_edited, text_features_edited = image_features_edited.image_embeds, image_features_edited.text_embeds

    image_features_orig = image_features_orig / image_features_orig.norm(dim=-1, keepdim=True)
    image_features_edited = image_features_edited / image_features_edited.norm(dim=-1, keepdim=True)
    text_features_orig = text_features_orig / text_features_orig.norm(dim=-1, keepdim=True)
    text_features_edited = text_features_edited / text_features_edited.norm(dim=-1, keepdim=True)

    sim_dir = F.cosine_similarity(
        image_features_edited - image_features_orig,
        text_features_edited - text_features_orig
    )
    return sim_dir


def clip_similarity(orig_images, recon_images, clip_model, clip_processor):
    """
    Compute cosine similarity between images and reference images using CLIP embeddings.

    Args:
        orig_images (list[PIL.Image]): List of original PIL images.
        recon_images (list[PIL.Image]): List of reconstructed PIL images.
        clip_model (CLIPModel): Preloaded CLIP model for computing embeddings.
        clip_processor (CLIPProcessor): Preloaded CLIP processor for image preprocessing.

    Returns:
        torch.Tensor: Cosine similarity scores for each image pair.
    """
    ref_inputs = clip_processor(images=orig_images, return_tensors="pt")
    recon_inputs = clip_processor(images=recon_images, return_tensors="pt")
    ref_inputs = {k: v.to(clip_model.device) for k, v in ref_inputs.items()}
    recon_inputs = {k: v.to(clip_model.device) for k, v in recon_inputs.items()}

    with torch.no_grad():
        ref_embeddings = clip_model.get_image_features(**ref_inputs)
        recon_embeddings = clip_model.get_image_features(**recon_inputs)

    similarity = F.cosine_similarity(ref_embeddings, recon_embeddings, dim=-1)
    return similarity


def pickscore(images, prompts):
    return reward_client.evaluate("pickscore", images, prompts)


def deqa(images, prompts):
    return reward_client.evaluate("deqa", images, prompts)


def gen_eval(images, prompts, meta_files):
    return reward_client.evaluate("gen_eval", images, prompts, meta_files)


def image_reward(images, prompts):
    return reward_client.evaluate("image_reward", images, prompts)


def aesthetic(images, prompts):
    return reward_client.evaluate("aesthetic", images, prompts)


def hps(images, prompts):
    return reward_client.evaluate("hps", images, prompts)


def ocr(images, prompts):
    """OCR reward: evaluate text rendering quality in generated images."""
    return reward_client.evaluate("ocr", images, prompts)


def ocr_vllm_reward(images, prompts):
    """OCR reward via OpenAI-compatible VLM endpoint (returns character accuracy in [0, 1])."""
    return ocr_vllm(images, prompts)


def unifiedreward_sglang(images, prompts):
    """Unified reward using sglang service."""
    return reward_client.evaluate("unifiedreward_sglang", images, prompts)


def editreward(images, prompts):
    """Edit reward for image editing tasks."""
    return reward_client.evaluate("editreward", images, prompts)


# ============================================================================
# UnifiedReward Functions
# ============================================================================

def _extract_normalized_rewards(
    sample_list: List[str],
    *,
    device: str | torch.device = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, List[float]]]:
    """Extract and normalize rewards from UnifiedReward responses."""
    pattern = r"(\w+) Score \(1-5\):\s*([0-5](?:\.\d+)?)"

    all_scores = []
    for response in sample_list:
        matches = re.findall(pattern, response)
        scores = {key: float(value) for key, value in matches}
        if "Coherence" in scores:
            del scores["Coherence"]
        all_scores.append(scores)

    if not all_scores:
        return [], [], {}

    keys = set()
    for score_dict in all_scores:
        keys.update(score_dict.keys())
    keys = sorted(keys)

    dim_scores_raw = {k: [s[k] for s in all_scores if k in s] for k in keys}
    dim_means = {
        k: np.mean(v) if len(v) > 0 else 0.0 for k, v in dim_scores_raw.items()
    }

    alignment_scores = []
    style_scores = []
    log_alignment_scores = []
    log_style_scores = []

    for score_dict in all_scores:
        alignment_score = score_dict.get("Alignment", dim_means.get("Alignment", 0.0))
        style_score = score_dict.get("Style", dim_means.get("Style", 0.0))

        # Detach tensors immediately to prevent computation graph retention
        alignment_scores.append(torch.tensor(alignment_score, device=device).unsqueeze(0).detach())
        style_scores.append(torch.tensor(style_score, device=device).unsqueeze(0).detach())

        log_alignment_scores.append(float(alignment_score))
        log_style_scores.append(float(style_score))

    dim_array = {"Alignment": log_alignment_scores, "Style": log_style_scores}
    return alignment_scores, style_scores, dim_array


def _extract_answer(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    final_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return final_match.group(1).strip() if final_match else None


def _pairwise_win_rate(
    *,
    num_items: int,
    responses: List[dict],
    better_1: str,
    better_2: str,
    device: str | torch.device = "cuda",
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """Compute pairwise win rate from comparison responses.

    Memory optimization: Creates a single tensor instead of a list of small tensors
    to avoid GPU memory fragmentation.
    """
    win_count = {"overall": defaultdict(int)}
    compare_count = {"overall": defaultdict(int)}

    for result in responses:
        idx1 = result.get("first_index")
        idx2 = result.get("second_index")

        # Skip invalid results (missing indices)
        if idx1 is None or idx2 is None:
            continue

        compare_count["overall"][idx1] += 1
        compare_count["overall"][idx2] += 1

        output = result.get("model_output", "")
        final_conclusion = _extract_answer(output)

        if final_conclusion:
            if better_1 in final_conclusion:
                win_count["overall"][idx1] += 1
            elif better_2 in final_conclusion:
                win_count["overall"][idx2] += 1
            else:
                win_count["overall"][idx1] += 0.5
                win_count["overall"][idx2] += 0.5
        else:
            win_count["overall"][idx1] += 0.5
            win_count["overall"][idx2] += 0.5

    # Memory optimization: Create a single tensor on CPU first, then move to GPU
    # This avoids creating many small GPU tensors which causes memory fragmentation
    win_rates_list = [
        round(win_count["overall"][idx] / compare_count["overall"][idx], 3)
        if compare_count["overall"][idx] > 0
        else 0.0
        for idx in range(num_items)
    ]
    # Create single tensor and move to GPU in one operation
    overall_win_rate = torch.tensor(win_rates_list, dtype=torch.float32, device=device).detach()

    # Use plain Python floats for dim_reward to avoid GPU memory usage
    dim_reward = {
        "overall_reward": win_rates_list.copy()  # Reuse the list we already computed
    }
    return overall_win_rate, dim_reward


def _encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string."""
    image = image.convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def unifiedreward(images, prompts, api_url=None):
    """
    UnifiedReward: Extract alignment and style scores from VLM responses.

    Args:
        images: List of PIL Images
        prompts: List of text prompts
        api_url: Optional API URL (defaults to UNIFIEDREWARD_THINK_URL env var)

    Returns:
        torch.Tensor: Combined reward scores (alignment * 0.5 + style * 0.5)
    """
    if api_url is None:
        api_url = os.environ.get("UNIFIEDREWARD_THINK_URL", "http://localhost:18087")

    # If UnifiedReward and UniGenBench share the same host, switch vLLM servers
    # using Sleep Mode to avoid GPU OOM.
    # NOTE: For most training runs, UnifiedReward should stay awake all the time.
    # We only need to sleep reward / wake judge around evaluation. This switch can be
    # disabled for training calls via env var to avoid extra control-plane requests.
    if _truthy_env("DEEPGEN_EXCLUSIVE_VLLM_SWITCH_ON_REWARD_CALLS", default="1"):
        maybe_switch_vllm_server("unifiedreward")

    # Build payload for each image-prompt pair
    payload = []
    for idx, (image, prompt) in enumerate(zip(images, prompts)):
        base64_image = _encode_image_to_base64(image)
        payload.append({
            "images": [base64_image],
            "problem": f"Please evaluate this image based on the prompt: {prompt}",
            "idx": idx,
        })

    # Call VLM service
    # Use UNIFIEDREWARD_WORKERS env var for concurrency control (falls back to VLLM_MAX_WORKERS)
    ur_workers = os.environ.get("UNIFIEDREWARD_WORKERS")
    max_workers = int(ur_workers) if ur_workers else None
    responses = evaluate_batch(payload, api_url=api_url, max_workers=max_workers)

    # Extract scores from responses
    sample_outputs = [r.get("model_output", "") for r in responses]
    alignment_scores, style_scores, dim_array = _extract_normalized_rewards(
        sample_outputs, device="cuda"
    )

    if not alignment_scores:
        # Return default scores if extraction failed
        return torch.zeros(len(images), device="cuda").detach()

    # Combine alignment and style scores
    combined_scores = []
    for align, style in zip(alignment_scores, style_scores):
        combined = (align + style) / 2.0  # Average of alignment and style
        combined_scores.append(combined.squeeze().detach())

    result = torch.stack(combined_scores).detach()

    # Clean up intermediate tensors to prevent memory leak
    del alignment_scores, style_scores, combined_scores, responses, payload
    torch.cuda.empty_cache()

    return result


def unifiedreward_think(images, prompts, api_url=None):
    """
    UnifiedReward Think: Pairwise comparison reward using thinking VLM.

    This reward function compares all pairs of images and computes win rates.

    Args:
        images: List of PIL Images (should be from the same prompt group)
        prompts: List of text prompts (all should be the same for comparison)
        api_url: Optional API URL (defaults to UNIFIEDREWARD_THINK_URL env var)

    Returns:
        torch.Tensor: Win rate scores for each image
    """
    if api_url is None:
        api_url = os.environ.get("UNIFIEDREWARD_THINK_URL", "http://localhost:18087")

    # If UnifiedReward and UniGenBench share the same host, switch vLLM servers
    # using Sleep Mode to avoid GPU OOM.
    # NOTE: For most training runs, UnifiedReward should stay awake all the time.
    # We only need to sleep reward / wake judge around evaluation. This switch can be
    # disabled for training calls via env var to avoid extra control-plane requests.
    if _truthy_env("DEEPGEN_EXCLUSIVE_VLLM_SWITCH_ON_REWARD_CALLS", default="1"):
        maybe_switch_vllm_server("unifiedreward")

    # Debug logging
    _debug_ur = os.environ.get("DEBUG_UNIFIED_REWARD", "0") == "1"
    if _debug_ur:
        env_log_dir = os.environ.get("LOG_DIR", ".")
        debug_path = os.path.join(env_log_dir, "debug_unifiedreward_func.log")
        with open(debug_path, "a") as f:
            f.write(f"\n\n=== unifiedreward_think called ===")
            f.write(f"\nnum_images: {len(images)}")
            f.write(f"\nnum_prompts: {len(prompts)}")
            f.write(f"\nprompts[0][:100]: {prompts[0][:100] if prompts else 'N/A'}")
            f.flush()

    if len(images) < 2:
        # Cannot do pairwise comparison with less than 2 images
        return torch.ones(len(images), device="cuda") * 0.5

    # Encode all images
    encoded_images = [_encode_image_to_base64(img) for img in images]

    # Generate all pairs for comparison
    pairs = list(itertools.combinations(enumerate(encoded_images), 2))
    problem = f"Compare these two images based on the prompt: {prompts[0]}. Which image is better?"

    payload = [
        {
            "images": [img1, img2],
            "problem": problem,
            "first_index": idx1,
            "second_index": idx2,
        }
        for (idx1, img1), (idx2, img2) in pairs
    ]

    # Call VLM service for pairwise comparisons
    # Use UNIFIEDREWARD_THINK_WORKERS env var for concurrency control (falls back to VLLM_MAX_WORKERS)
    ur_think_workers = os.environ.get("UNIFIEDREWARD_THINK_WORKERS")
    max_workers = int(ur_think_workers) if ur_think_workers else None
    responses = evaluate_batch(payload, api_url=api_url, max_workers=max_workers)

    # Debug: check responses
    if _debug_ur:
        success_count = sum(1 for r in responses if r.get("success", False))
        with open(debug_path, "a") as f:
            f.write(f"\nnum_pairs: {len(pairs)}")
            f.write(f"\nnum_responses: {len(responses)}")
            f.write(f"\nsuccess_count: {success_count}")
            # Sample some responses
            for i, r in enumerate(responses[:3]):
                f.write(f"\n  Response {i}: first_index={r.get('first_index')}, second_index={r.get('second_index')}, success={r.get('success')}")
                model_output = r.get("model_output", "")
                answer = _extract_answer(model_output) if model_output else None
                f.write(f", answer={answer[:50] if answer else 'None'}")
            f.flush()

    # Compute win rates
    # Note: _pairwise_win_rate now returns a single tensor instead of list of tensors
    # to avoid GPU memory fragmentation
    overall_win_rate, dim_reward = _pairwise_win_rate(
        num_items=len(images),
        responses=responses,
        better_1="Image 1 is better",
        better_2="Image 2 is better",
        device="cuda",
    )

    # overall_win_rate is already a single tensor, just ensure it's detached
    result = overall_win_rate.detach()

    # Debug: log final result
    if _debug_ur:
        with open(debug_path, "a") as f:
            f.write(f"\nresult: {result.tolist()}")
            f.write(f"\nmean: {result.mean().item():.4f}")
            f.flush()

    # Clean up intermediate tensors to prevent memory leak
    del overall_win_rate, dim_reward, responses, pairs, encoded_images, payload
    torch.cuda.empty_cache()

    # Return detached result
    return result


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r'<answer>.*?</answer>'
    extracted_contents = [
        s[s.find('\nassistant') + len('\nassistant'):].strip() if s.find('\nassistant') != -1 else ""
        for s in completions
    ]
    matches = list(map(lambda x: re.match(pattern, x), extracted_contents))
    return [1.0 if match else 0.0 for match in matches]


# Registry of available reward functions
reward_funcs_registry = {
    "recon": recon_reward,
    "jpeg_compressibility": jpeg_compressibility,
    "jpeg_incompressibility": jpeg_incompressibility,
    "pickscore": pickscore,
    "deqa": deqa,
    "gen_eval": gen_eval,
    "image_reward": image_reward,
    "aesthetic": aesthetic,
    "hps": hps,
    "ocr": ocr,
    "ocr_vllm": ocr_vllm_reward,
    "unifiedreward_sglang": unifiedreward_sglang,
    "unifiedreward": unifiedreward,
    "unifiedreward_think": unifiedreward_think,
    "clip_sim": clip_similarity,
    "sim_direction": sim_direction,
    "format": format_reward,
    "editreward": editreward,
}

# CLIP model path from environment variable with fallback to default
CLIP_MODEL_NAME_OR_PATH = os.environ.get("CLIP_MODEL_NAME_OR_PATH", "openai/clip-vit-large-patch14")

# Lazy-loaded CLIP processor to avoid memory usage at module import time
# This prevents unnecessary memory allocation when CLIP rewards are not used
_clip_processor_cache = None

def _get_clip_processor():
    """Lazy-load CLIP processor to avoid memory allocation at import time."""
    global _clip_processor_cache
    if _clip_processor_cache is None:
        _clip_processor_cache = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME_OR_PATH)
    return _clip_processor_cache

# Note: clip_sim and sim_direction use lazy loading via _get_clip_processor()
# The actual processor is loaded on first use, not at module import
reward_processing_registry = {
    "recon": T.Compose([T.ToTensor()]),
    "jpeg_compressibility": None,
    "jpeg_incompressibility": None,
    "pickscore": None,
    "deqa": None,
    "gen_eval": None,
    "image_reward": None,
    "aesthetic": None,
    "hps": None,
    "ocr": None,
    "ocr_vllm": None,
    "unifiedreward_sglang": None,
    "unifiedreward": None,
    "unifiedreward_think": None,
    "clip_sim": "lazy_clip_processor",  # Marker for lazy loading
    "sim_direction": "lazy_clip_processor",  # Marker for lazy loading
    "format": None,
    "editreward": None,
}


def get_reward_processor(name: str):
    """Get the processor for a reward function, with lazy loading support."""
    processor = reward_processing_registry.get(name)
    if processor == "lazy_clip_processor":
        return _get_clip_processor()
    return processor


# ============================================================================
# Dataset Classes
# ============================================================================

class PromptDataset(Dataset):
    """Dataset for loading prompts from various file formats."""

    def __init__(self, prompts_file: str, reward_config: Optional[Dict[str, float]] = None, dataset_name: str = None):
        """
        Initialize dataset from prompts file.

        Args:
            prompts_file: Path to prompts file (.txt, .jsonl, or .parquet)
            reward_config: Optional dict mapping reward function names to weights.
                          e.g., {"ocr": 0.4, "clip_sim": 0.6}
            dataset_name: Name of the dataset (for logging purposes)
        """
        self.prompts = []
        self.reward_config = reward_config or {}
        self.dataset_name = dataset_name or os.path.basename(prompts_file)

        if prompts_file.endswith(".txt"):
            if not os.path.exists(prompts_file):
                raise FileNotFoundError(f"Prompts file {prompts_file} not found")
            with open(prompts_file, 'r', encoding='utf-8') as f:
                for line in f:
                    prompt = line.strip()
                    if prompt:
                        self.prompts.append({
                            "prompt": prompt,
                            "caption": prompt,
                            "reward_config": self.reward_config,
                            "dataset_name": self.dataset_name,
                        })

        elif prompts_file.endswith(".jsonl"):
            with open(prompts_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    prompt = item.get('prompt', item.get('caption', ''))
                    self.prompts.append({
                        "prompt": prompt,
                        "caption": prompt,
                        "metadata": item,
                        "reward_config": self.reward_config,
                        "dataset_name": self.dataset_name,
                    })

        elif prompts_file.endswith(".parquet"):
            dataset = load_dataset("parquet", data_files={"train": prompts_file})["train"]
            for item in dataset:
                prompt = item.get("prompt", item.get("caption", ""))
                if prompt:
                    self.prompts.append({
                        "prompt": prompt,
                        "caption": prompt,
                        "reward_config": self.reward_config,
                        "dataset_name": self.dataset_name,
                    })
        else:
            raise ValueError("Unsupported file format. Use .txt, .jsonl, or .parquet")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


class MultiDatasetConfig:
    """
    Configuration for multi-dataset training with per-dataset reward functions.

    Parses YAML config like:
    ```yaml
    # Sampling mode: 'sample_proportional' (default) or 'weight_only'
    #   - sample_proportional: weight is a multiplier relative to uniform sampling by sample count
    #     e.g., weight=2.0 means 2x probability compared to uniform sampling
    #   - weight_only: sampling probability is purely based on weight, ignoring dataset size
    #     e.g., all datasets with weight=1.0 have equal probability regardless of size
    sampling_mode: sample_proportional

    datasets:
      - name: blip3o_60k
        path: blip3o_60k.txt
        rewards:
          - unifiedreward_sglang: 0.4
          - clip_sim: 0.6
      - name: qwenimage_textrender
        path: qwenimage_textrender.txt
        sample_weight: 2.0  # 2x sampling probability compared to uniform
        rewards:
          - unifiedreward_sglang: 0.3
          - clip_sim: 0.3
          - ocr: 0.4
    ```

    Sampling Modes:
        - sample_proportional (default):
            Base probability = num_samples / total_samples (uniform by sample count)
            Final probability = base_probability * weight (normalized)
            This means weight is a multiplier: weight=2.0 doubles the sampling rate.

        - weight_only:
            Probability = weight / sum(weights)
            Dataset size is ignored; all datasets with same weight have equal probability.
    """

    # Valid sampling modes
    SAMPLING_MODES = ['sample_proportional', 'weight_only']

    def __init__(self, config_path: str):
        """
        Initialize from YAML config file.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config_dir = os.path.dirname(config_path)

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Parse sampling mode (default: sample_proportional)
        self.sampling_mode = self.config.get('sampling_mode', 'sample_proportional')
        if self.sampling_mode not in self.SAMPLING_MODES:
            raise ValueError(
                f"Invalid sampling_mode '{self.sampling_mode}'. "
                f"Must be one of: {self.SAMPLING_MODES}"
            )

        self.datasets_config = self.config.get('datasets', [])

        # Parse and validate
        self.parsed_datasets = self._parse_datasets()

        # Collect all unique reward functions needed
        self.all_reward_funcs = self._collect_all_reward_funcs()

    def _parse_datasets(self) -> List[Dict[str, Any]]:
        """Parse dataset configurations."""
        parsed = []
        for ds_cfg in self.datasets_config:
            name = ds_cfg.get('name', 'unnamed')
            path = ds_cfg.get('path', '')

            # Handle relative paths
            if not os.path.isabs(path):
                path = os.path.join(self.config_dir, path)

            # Parse rewards list into dict
            rewards = {}
            for reward_item in ds_cfg.get('rewards', []):
                if isinstance(reward_item, dict):
                    rewards.update(reward_item)

            # Validate reward functions exist
            for reward_name in rewards.keys():
                if reward_name not in reward_funcs_registry:
                    raise ValueError(
                        f"Unknown reward function '{reward_name}' in dataset '{name}'. "
                        f"Available: {list(reward_funcs_registry.keys())}"
                    )

            # Parse sample_weight (default: 1.0)
            sample_weight = float(ds_cfg.get('sample_weight', 1.0))
            if sample_weight <= 0:
                raise ValueError(
                    f"sample_weight must be positive, got {sample_weight} for dataset '{name}'"
                )

            parsed.append({
                'name': name,
                'path': path,
                'rewards': rewards,
                'sample_weight': sample_weight,
            })

        return parsed

    def _collect_all_reward_funcs(self) -> List[str]:
        """Collect all unique reward function names from all datasets.

        Returns a sorted list to ensure consistent ordering across all distributed ranks.
        This is critical for distributed training where reward_funcs indices must match.
        """
        all_funcs = set()
        for ds in self.parsed_datasets:
            all_funcs.update(ds['rewards'].keys())
        # Sort to ensure consistent ordering across all ranks in distributed training
        return sorted(all_funcs)

    def create_dataset(self) -> Tuple[ConcatDataset, Optional[WeightedRandomSampler]]:
        """
        Create a combined dataset from all configured datasets with optional weighted sampling.

        Returns:
            Tuple of (ConcatDataset, WeightedRandomSampler or None)
            The sampler is None if all weights are equal (1.0)
        """
        datasets = []
        dataset_lengths = []
        dataset_weights = []

        for ds_cfg in self.parsed_datasets:
            ds = PromptDataset(
                prompts_file=ds_cfg['path'],
                reward_config=ds_cfg['rewards'],
                dataset_name=ds_cfg['name'],
            )
            rank0_print(f"Loaded dataset '{ds_cfg['name']}': {len(ds)} prompts, "
                        f"rewards: {ds_cfg['rewards']}, sample_weight: {ds_cfg['sample_weight']}")
            datasets.append(ds)
            dataset_lengths.append(len(ds))
            dataset_weights.append(ds_cfg['sample_weight'])

        if len(datasets) == 1:
            self._print_dataset_statistics(
                [self.parsed_datasets[0]['name']],
                dataset_lengths,
                dataset_weights,
                use_weighted_sampling=False
            )
            return datasets[0], None

        combined_dataset = ConcatDataset(datasets)

        # Check if all weights are equal (1.0), skip sampler if so
        if all(w == 1.0 for w in dataset_weights):
            self._print_dataset_statistics(
                [ds['name'] for ds in self.parsed_datasets],
                dataset_lengths,
                dataset_weights,
                use_weighted_sampling=False,
                sampling_mode=self.sampling_mode
            )
            return combined_dataset, None

        # Build per-sample weights for WeightedRandomSampler based on sampling_mode
        sample_weights = []

        if self.sampling_mode == 'weight_only':
            # weight_only mode: probability proportional to weight only, ignoring dataset size
            # Each sample's weight = dataset_weight / dataset_size
            # This makes dataset probability = weight / sum(weights)
            for length, weight in zip(dataset_lengths, dataset_weights):
                per_sample_weight = weight / length
                sample_weights.extend([per_sample_weight] * length)
        else:
            # sample_proportional mode (default): weight is a multiplier on uniform sampling
            # Base probability = num_samples / total_samples (uniform)
            # Final probability = base_probability * weight (normalized)
            # This is equivalent to: each sample has weight = dataset_weight (same for all in dataset)
            for length, weight in zip(dataset_lengths, dataset_weights):
                per_sample_weight = weight
                sample_weights.extend([per_sample_weight] * length)

        # Create WeightedRandomSampler with replacement for efficiency
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(combined_dataset),
            replacement=True,
        )

        self._print_dataset_statistics(
            [ds['name'] for ds in self.parsed_datasets],
            dataset_lengths,
            dataset_weights,
            use_weighted_sampling=True,
            sampling_mode=self.sampling_mode
        )

        return combined_dataset, sampler

    def _print_dataset_statistics(
        self,
        names: List[str],
        lengths: List[int],
        weights: List[float],
        use_weighted_sampling: bool,
        sampling_mode: str = 'sample_proportional'
    ):
        """
        Print dataset statistics including sample counts and sampling probabilities.

        Args:
            names: List of dataset names
            lengths: List of dataset sizes
            weights: List of sample_weight values
            use_weighted_sampling: Whether weighted sampling is enabled
            sampling_mode: Sampling mode ('sample_proportional' or 'weight_only')
        """
        total_samples = sum(lengths)
        total_weight = sum(weights)

        rank0_print("\n" + "=" * 70)
        rank0_print("Dataset Statistics Summary")
        rank0_print("=" * 70)
        rank0_print(f"{'Dataset Name':<30} {'Samples':>10} {'Weight':>10} {'Prob (%)':>12}")
        rank0_print("-" * 70)

        for name, length, weight in zip(names, lengths, weights):
            if use_weighted_sampling:
                if sampling_mode == 'weight_only':
                    # weight_only: probability proportional to weight only
                    prob = (weight / total_weight) * 100
                else:
                    # sample_proportional: weight is multiplier on uniform sampling
                    # prob = (length / total_samples) * weight, then normalized
                    weighted_samples = [l * w for l, w in zip(lengths, weights)]
                    total_weighted = sum(weighted_samples)
                    prob = (length * weight / total_weighted) * 100
            else:
                # Uniform sampling: probability proportional to sample count
                prob = (length / total_samples) * 100
            rank0_print(f"{name:<30} {length:>10} {weight:>10.2f} {prob:>11.2f}%")

        rank0_print("-" * 70)
        rank0_print(f"{'Total':<30} {total_samples:>10} {total_weight:>10.2f} {'100.00%':>12}")

        if use_weighted_sampling:
            mode_desc = {
                'sample_proportional': 'sample_proportional (weight = multiplier on uniform sampling)',
                'weight_only': 'weight_only (probability = weight / total_weight)'
            }
            rank0_print(f"Sampling Mode: {mode_desc.get(sampling_mode, sampling_mode)}")
        else:
            rank0_print(f"Weighted Sampling: Disabled (all weights = 1.0)")
        rank0_print("=" * 70 + "\n")

    def get_reward_funcs_list(self) -> List[tuple]:
        """
        Get list of all reward functions needed for training.

        Returns:
            List of (name, processor, func) tuples
        """
        return [
            (name, get_reward_processor(name), reward_funcs_registry[name])
            for name in self.all_reward_funcs
        ]


def get_last_checkpoint(output_dir):
    """Get the last checkpoint from output directory."""
    from transformers.trainer_utils import get_last_checkpoint
    if os.path.isdir(output_dir):
        return get_last_checkpoint(output_dir)
    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GRPO training for DeepGen models")

    # Model arguments
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to mmengine model config file (e.g., configs/models/qwen2_5_vl_7b_stable_diffusion_3_5_medium_hf_dynamic_dpo_fusion.py)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained checkpoint from DeepGen_Image training"
    )

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/deepgen_grpo")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="Learning rate scheduler type (e.g., linear, cosine, constant, constant_with_warmup). Default: constant")
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps. If set to a positive number, overrides num_train_epochs.")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # Rollout (generation) stage arguments
    # Note: all batch sizes are defined in terms of images (not prompts)
    parser.add_argument("--rollout_n", type=int, default=8, help="Number of images to generate per prompt. Default: 8")
    parser.add_argument("--rollout_micro_batch_size", type=int, default=32, help="Number of images per GPU per rollout forward pass (prompt count = rollout_micro_batch_size / rollout_n). Default: 32")
    parser.add_argument("--rollout_global_batch_size", type=int, default=256, help="Total images per rollout step across all GPUs (must be a multiple of rollout_micro_batch_size * world_size). Default: 256")

    # Actor training stage arguments
    # Note: all batch sizes are defined in terms of images (not prompts)
    parser.add_argument("--atrain_micro_batch_size", type=int, default=32, help="Number of images per GPU per forward pass during actor training. Default: 32")
    parser.add_argument("--atrain_global_batch_size", type=int, default=256, help="Total images per optimization step across all GPUs (used to auto-compute gradient_accumulation_steps). Default: 256")
    parser.add_argument("--log_prob_micro_batch_size", type=int, default=None, help="Number of samples per micro-batch for log_prob computation. If None, defaults to atrain_micro_batch_size")

    # GRPO specific arguments
    parser.add_argument("--beta", type=float, default=5e-7, help="KL penalty coefficient. Default: 5e-7")
    parser.add_argument("--cfg_scale", type=float, default=0, help="Classifier-free guidance scale. Default: 0")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion steps. Default: 50")
    parser.add_argument("--image_height", type=int, default=512, help="Generated image height. Default: 512")
    parser.add_argument("--image_width", type=int, default=512, help="Generated image width. Default: 512")
    parser.add_argument("--log_images_interval", type=int, default=1, help="Interval (in steps) for logging training sample images, 0 to disable. Default: 1")

    # Reward arguments
    parser.add_argument(
        "--reward_funcs",
        type=str,
        nargs="+",
        default=["jpeg_compressibility"],
        help="List of reward functions to use (when not using dataset_config)"
    )

    # Data arguments
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="prompts.txt",
        help="Path to prompts file (.txt, .jsonl, or .parquet). Used when dataset_config is not specified."
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Path to dataset configuration YAML file. When specified, overrides prompts_file and reward_funcs."
    )

    # Logging arguments
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="Logging backend(s). Use comma to specify multiple: 'wandb,tensorboard'. "
             "Options: wandb, tensorboard, mlflow, comet_ml, all, none"
    )
    parser.add_argument("--run_name", type=str, default=None)

    # ================================================================
    # Logging: component grad-norm proxy (GRPO vs SFT-Aux)
    # ================================================================
    parser.add_argument(
        "--log_component_grad_norm",
        action="store_true",
        default=True,
        help=(
            "If set, log a proxy grad-norm for GRPO vs SFT-Aux separately. "
            "This metric is computed from per-backward gradient increments and is useful "
            "for monitoring relative gradient scale. Default: on."
        ),
    )
    parser.add_argument(
        "--log_component_grad_norm_every",
        type=int,
        default=None,
        help="Measure/log component grad-norm every N steps. If None, uses --logging_steps.",
    )

    # ================================================================
    # Logging: TFLOPS (approximate)
    # ================================================================
    parser.add_argument(
        "--log_tflops",
        action="store_true",
        default=False,
        help=(
            "If set, log an approximate achieved TFLOPS for optimization steps. "
            "Implementation uses a one-time DeepSpeed FLOPs profiler calibration (if available), "
            "then estimates TFLOPS from wall-time. Default: off."
        ),
    )
    parser.add_argument(
        "--log_tflops_every",
        type=int,
        default=None,
        help="Compute/log TFLOPS every N optimization steps. If None, uses --logging_steps.",
    )
    parser.add_argument(
        "--log_tflops_warmup_steps",
        type=int,
        default=5,
        help="Warm up steps before TFLOPS calibration/estimation starts. Default: 5.",
    )

    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file path")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    # Memory optimization
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing to reduce memory usage at the cost of slower training. Default: on"
    )

    # Gradient clipping
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping. Set to 0 to disable."
    )

    # ============================================================================
    # SFT Auxiliary Loss (SFT-Aux)
    # ============================================================================
    parser.add_argument(
        "--sftaux_dataset_config",
        type=str,
        default="deepgen_rl/sft/configs/datasets/deepgen/t2i_grpo_moretextdata.py",
        help="Path to deepgen_sft mmengine dataset config (.py) used for SFT auxiliary loss."
    )
    parser.add_argument(
        "--sftaux_coef",
        type=float,
        default=0.0001,
        help="Convex mixing coefficient lambda for SFT-Aux: total_loss=(1-lambda)*grpo_loss + lambda*sft_loss. Must be in [0, 1]. Default: 0.0001"
    )
    parser.add_argument(
        "--sftaux_every_n_steps",
        type=int,
        default=1,
        help="Compute SFT auxiliary loss every N GRPO steps. Default: 1 (every step)."
    )
    parser.add_argument(
        "--sftaux_micro_batch_size",
        type=int,
        default=32,
        help="Override deepgen_sft train_dataloader.batch_size for SFT-Aux. This is effectively the SFT-Aux micro batch size per GPU. Default: 32"
    )
    parser.add_argument(
        "--sftaux_num_workers",
        type=int,
        default=None,
        help="Override deepgen_sft train_dataloader.num_workers for SFT-Aux. If None, keep config value."
    )
    parser.add_argument(
        "--sftaux_disable_on_error",
        action="store_true",
        default=False,
        help="If set, disable SFT-Aux when dataloader build fails instead of raising."
    )

    # New GRPO optimization arguments
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=True,
        help="If set, all images in the same group (same prompt) will use the same initial noise. Default: on"
    )
    parser.add_argument(
        "--timestep_fraction",
        type=float,
        default=0.6,
        help="Fraction of timesteps to use for training. Randomly samples (num_inference_steps * timestep_fraction) timesteps for gradient computation. Must be in (0, 1]. Default: 0.6"
    )
    parser.add_argument(
        "--ema_diffusion",
        type=float,
        default=0.0,
        help=(
            "EMA decay for diffusion model (transformer) parameters. "
            "Set to 0 to disable. Typical values: 0.999-0.9999."
        ),
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=1e-4,
        help="Clipping range for ratio in GRPO loss. Ratio will be clipped to [1 - clip_range, 1 + clip_range]."
    )
    parser.add_argument(
        "--sde_eta",
        type=float,
        default=1.0,
        help="SDE noise scale factor for sampling diversity. Higher values increase diversity (default: 1.0)."
    )
    parser.add_argument(
        "--atrain_algorithm",
        type=str,
        default="flowgrpo",
        choices=["flowgrpo", "grpoguard", "diffusionnft"],
        help="Training algorithm for Flow Matching GRPO. "
             "'flowgrpo' (default): Standard Flow-GRPO implementation. "
             "'grpoguard': GRPO-Guard for addressing implicit over-optimization, "
             "uses RatioNorm and Gradient Reweight mechanisms. "
             "'diffusionnft': DiffusionNFT algorithm from arXiv:2509.16117. "
             "Uses v-space (noise prediction) loss with adaptive weighting. "
             "Additional hyperparameters can be set via environment variables: "
             "ATRAIN_GRPOGUARD_HIGHCLIP_RANGE (upper clipping range, default: same as clip_range), "
             "ATRAIN_NFT_BETA (DiffusionNFT beta, default: 0.1)."
    )
    parser.add_argument(
        "--atrain_nft_beta",
        type=float,
        default=0.1,
        help="DiffusionNFT beta parameter controlling the mixture of new and old predictions. "
             "Only used when --atrain_algorithm=diffusionnft. "
             "Higher values give more weight to the new policy. (default: 0.1)"
    )
    parser.add_argument(
        "--atrain_nft_adv_clip_range",
        type=float,
        default=5.0,
        help="Advantage clipping range for DiffusionNFT. "
             "Advantages are clipped to [-adv_clip_range, +adv_clip_range]. "
             "Only used when --atrain_algorithm=diffusionnft. (default: 5.0)"
    )
    parser.add_argument(
        "--atrain_nft_off_policy",
        action="store_true",
        default=False,
        help="Enable off-policy mode for DiffusionNFT. "
             "When enabled, old_v_pred is computed using EMA parameters during rollout "
             "(similar to Flow-Factory's off_policy=True mode). "
             "Requires --ema_diffusion > 0 to be set. "
             "When disabled (default), uses on-policy mode where old_v_pred = current model detached. "
             "Only used when --atrain_algorithm=diffusionnft."
    )
    parser.add_argument(
        "--atrain_sde_sampler",
        type=str,
        default="cps_sde",
        choices=["flowgrpo_sde", "cps_sde", "dance_sde"],
        help="SDE sampler type for training. "
             "'flowgrpo_sde': Standard Flow-GRPO SDE sampling. "
             "'cps_sde' (default): Coefficients-Preserving Sampling from FlowCPS (arXiv:2509.05952). "
             "'dance_sde': Dance-SDE sampling with log correction term."
    )
    parser.add_argument(
        "--atrain_adv_type",
        type=str,
        default="gdpo",
        choices=["grpo", "grpo_global_std", "reinforcepp", "gdpo"],
        help="Advantage computation type. "
             "'grpo': Group-wise normalization - advantage = (reward - group_mean) / group_std. "
             "'grpo_global_std': Group mean + global std - advantage = (reward - group_mean) / global_std. "
             "'reinforcepp': Batch-wise normalization - advantage = (reward - batch_mean) / batch_std. "
             "'gdpo' (default): Each reward is group-normalized separately, then weighted sum, then batch normalization."
    )
    parser.add_argument(
        "--atrain_kl_type",
        type=str,
        default="v-based",
        choices=["x-based", "v-based"],
        help="KL divergence type. "
             "'x-based': Compare latent means (next_latents_mean). "
             "'v-based' (default): Compare noise predictions (velocity space)."
    )
    parser.add_argument(
        "--atrain_num_actor_update_steps",
        type=int,
        default=1,
        help="Number of actor model weight update steps per rollout. "
             "Each update step samples non-overlapping data from the rollout buffer. "
             "When N > 1, the total training samples per epoch = rollout_global_batch_size * N. "
             "Constraint: atrain_global_batch_size * N <= rollout_global_batch_size. "
             "Inspired by PPO's num_mini_batches and Flow-GRPO's num_inner_epochs."
    )

    # Deprecated arguments (will raise error if used)
    parser.add_argument("--num_generations", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help=argparse.SUPPRESS)

    # ============================================================================
    # Evaluation Arguments
    # ============================================================================
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10,
        help="Evaluation frequency (in steps). Set to 0 to disable periodic evaluation. Default: 10"
    )
    parser.add_argument(
        "--eval_before_train",
        action="store_true",
        default=False,
        help="If set, run evaluation before training starts."
    )
    parser.add_argument(
        "--eval_dataset_config",
        type=str,
        default=None,
        help="Path to evaluation dataset configuration YAML file. "
             "Format: datasets: [{name: str, path: str}, ...]"
    )
    parser.add_argument(
        "--eval_inference_mode",
        type=str,
        default="ode",
        choices=["sde", "ode"],
        help="Inference mode for evaluation: 'sde' (stochastic, same as training) or 'ode' (deterministic). Default: ode"
    )
    parser.add_argument(
        "--eval_cfg_scale",
        type=float,
        default=4.0,
        help="CFG scale for evaluation. Default: 4.0"
    )
    parser.add_argument(
        "--eval_num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps for evaluation. Default: 50"
    )
    parser.add_argument(
        "--eval_sde_eta",
        type=float,
        default=None,
        help="SDE noise scale factor for evaluation (only used when eval_inference_mode='sde'). "
             "If not specified, uses training sde_eta."
    )
    parser.add_argument(
        "--eval_image_height",
        type=int,
        default=None,
        help="Image height for evaluation. If not specified, uses training image_height."
    )
    parser.add_argument(
        "--eval_image_width",
        type=int,
        default=None,
        help="Image width for evaluation. If not specified, uses training image_width."
    )
    parser.add_argument(
        "--eval_micro_batch_size",
        type=int,
        default=160,
        help="Number of images per GPU during evaluation inference. Controls GPU memory usage. Default: 160"
    )
    parser.add_argument(
        "--eval_wandb_num_upload_images",
        type=int,
        default=0,
        help="Number of prompts to upload images for to wandb per evaluation step. "
             "Selects prompts with the smallest indices (0, 1, ..., N-1). "
             "All duplicate images for selected prompts are uploaded. Default: 0"
    )
    parser.add_argument(
        "--eval_swanlab_num_upload_images",
        type=int,
        default=32,
        help="Number of prompts to upload images for to swanlab per evaluation step. "
             "Selects prompts with the smallest indices (0, 1, ..., N-1). "
             "All duplicate images for selected prompts are uploaded. Default: 32"
    )

    return parser.parse_args()


def validate_new_args(args):
    """Validate new GRPO optimization arguments."""
    if args.timestep_fraction <= 0 or args.timestep_fraction > 1:
        raise ValueError(
            f"Error: --timestep_fraction must be in (0, 1], got {args.timestep_fraction}"
        )
    if args.clip_range < 0:
        raise ValueError(
            f"Error: --clip_range must be non-negative, got {args.clip_range}"
        )
    if args.ema_diffusion < 0 or args.ema_diffusion >= 1:
        raise ValueError(
            f"Error: --ema_diffusion must be in [0, 1). Use 0 to disable, got {args.ema_diffusion}"
        )
    # SFT-Aux arguments
    if hasattr(args, "sftaux_coef") and (args.sftaux_coef < 0 or args.sftaux_coef > 1):
        raise ValueError(
            f"Error: --sftaux_coef must be in [0, 1], got {args.sftaux_coef}"
        )
    if hasattr(args, "sftaux_every_n_steps") and args.sftaux_every_n_steps <= 0:
        raise ValueError(
            f"Error: --sftaux_every_n_steps must be positive, got {args.sftaux_every_n_steps}"
        )
    if hasattr(args, "sftaux_micro_batch_size") and args.sftaux_micro_batch_size is not None and args.sftaux_micro_batch_size <= 0:
        raise ValueError(
            f"Error: --sftaux_micro_batch_size must be positive, got {args.sftaux_micro_batch_size}"
        )
    if hasattr(args, "sftaux_num_workers") and args.sftaux_num_workers is not None and args.sftaux_num_workers < 0:
        raise ValueError(
            f"Error: --sftaux_num_workers must be >= 0, got {args.sftaux_num_workers}"
        )
    # Component grad-norm logging arguments
    if hasattr(args, "log_component_grad_norm_every") and args.log_component_grad_norm_every is not None:
        if args.log_component_grad_norm_every <= 0:
            raise ValueError(
                f"Error: --log_component_grad_norm_every must be positive, got {args.log_component_grad_norm_every}"
            )

    # TFLOPS logging arguments
    if hasattr(args, "log_tflops_every") and args.log_tflops_every is not None:
        if args.log_tflops_every <= 0:
            raise ValueError(
                f"Error: --log_tflops_every must be positive, got {args.log_tflops_every}"
            )
    if hasattr(args, "log_tflops_warmup_steps") and args.log_tflops_warmup_steps is not None:
        if args.log_tflops_warmup_steps < 0:
            raise ValueError(
                f"Error: --log_tflops_warmup_steps must be >= 0, got {args.log_tflops_warmup_steps}"
            )


def check_deprecated_args(args):
    """Check for deprecated arguments and raise errors with migration guidance."""
    deprecated_mapping = {
        "num_generations": "rollout_n",
        "per_device_train_batch_size": "rollout_micro_batch_size or atrain_micro_batch_size",
        "gradient_accumulation_steps": "atrain_global_batch_size",
    }

    for old_arg, new_arg in deprecated_mapping.items():
        if getattr(args, old_arg, None) is not None:
            raise ValueError(
                f"Error: Parameter '--{old_arg}' has been removed.\n"
                f"Please use '--{new_arg}' instead."
            )


def compute_gradient_accumulation_steps(args, world_size):
    """
    Compute gradient_accumulation_steps from atrain_global_batch_size.

    Args:
        args: Parsed arguments
        world_size: Total number of GPUs (NNODES * NGPUS)

    Returns:
        Computed gradient_accumulation_steps
    """
    if args.atrain_global_batch_size is None:
        # Default: no accumulation
        return 1

    micro_batch = args.atrain_micro_batch_size
    global_batch = args.atrain_global_batch_size

    if global_batch % (micro_batch * world_size) != 0:
        valid_global_batches = [
            micro_batch * world_size * i for i in range(1, 10)
        ]
        raise ValueError(
            f"Error: Cannot evenly divide atrain_global_batch_size by (atrain_micro_batch_size * world_size).\n"
            f"  atrain_global_batch_size = {global_batch} (images)\n"
            f"  atrain_micro_batch_size = {micro_batch} (images per GPU)\n"
            f"  world_size = {world_size}\n"
            f"  Suggested valid atrain_global_batch_size values: {valid_global_batches}"
        )

    return global_batch // (micro_batch * world_size)


def main():
    """Main training function."""
    args = parse_args()

    # Check for deprecated arguments
    check_deprecated_args(args)

    # Validate new arguments
    validate_new_args(args)

    # Get world size for computing gradient_accumulation_steps
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Compute gradient_accumulation_steps from atrain_global_batch_size
    gradient_accumulation_steps = compute_gradient_accumulation_steps(args, world_size)

    # Compute derived parameters (all in terms of images)
    import math

    # Rollout: how many prompts per GPU per rollout step
    # rollout_micro_batch_size is the images per GPU per forward pass
    # When rollout_n > rollout_micro_batch_size, multiple forward passes are used per prompt
    # When rollout_micro_batch_size >= rollout_n, multiple prompts can be processed per forward pass
    # Use ceiling division to ensure at least 1 prompt per batch
    rollout_prompt_batch_size = max(1, args.rollout_micro_batch_size // args.rollout_n)

    # Validate rollout_global_batch_size: must be a multiple of rollout_micro_batch_size * world_size
    rollout_global_batch_size = args.rollout_global_batch_size
    micro_global = args.rollout_micro_batch_size * world_size
    if rollout_global_batch_size % micro_global != 0:
        valid_values = [micro_global * i for i in range(1, 6)]
        raise ValueError(
            f"rollout_global_batch_size ({rollout_global_batch_size}) must be a multiple of "
            f"rollout_micro_batch_size * world_size ({args.rollout_micro_batch_size} * {world_size} = {micro_global})\n"
            f"Suggested valid values: {valid_values}"
        )
    rollout_accumulation_steps = rollout_global_batch_size // micro_global

    # Validate atrain_global_batch_size with atrain_num_actor_update_steps:
    # Total samples used = atrain_global_batch_size * atrain_num_actor_update_steps
    # This must be <= rollout_global_batch_size (can't use more samples than generated)
    atrain_num_actor_update_steps = args.atrain_num_actor_update_steps
    if atrain_num_actor_update_steps < 1:
        raise ValueError(
            f"atrain_num_actor_update_steps ({atrain_num_actor_update_steps}) must be >= 1"
        )

    if args.atrain_global_batch_size is not None:
        total_train_samples = args.atrain_global_batch_size * atrain_num_actor_update_steps
        if total_train_samples > rollout_global_batch_size:
            raise ValueError(
                f"atrain_global_batch_size ({args.atrain_global_batch_size}) * "
                f"atrain_num_actor_update_steps ({atrain_num_actor_update_steps}) = {total_train_samples}\n"
                f"must be <= rollout_global_batch_size ({rollout_global_batch_size})\n"
                f"Cannot train on more images than were generated in the rollout.\n"
                f"Either reduce atrain_global_batch_size, reduce atrain_num_actor_update_steps, "
                f"or increase rollout_global_batch_size."
            )

    # ================================================================
    # Dataset and Reward Setup
    # ================================================================
    use_dataset_config = args.dataset_config is not None

    if use_dataset_config:
        # Load from YAML config
        rank0_print(f"Loading dataset configuration from: {args.dataset_config}")
        dataset_config = MultiDatasetConfig(args.dataset_config)
        train_dataset, train_sampler = dataset_config.create_dataset()
        reward_funcs = dataset_config.get_reward_funcs_list()

        rank0_print(f"Total prompts: {len(train_dataset)}")
        rank0_print(f"All reward functions: {[f[0] for f in reward_funcs]}")
        if train_sampler is not None:
            rank0_print(f"Using weighted sampling based on sample_weight configuration")
    else:
        # Legacy mode: single prompts file with uniform reward functions
        reward_funcs = [
            (func, get_reward_processor(func), reward_funcs_registry[func])
            for func in args.reward_funcs
            if func in reward_funcs_registry
        ]

        if not reward_funcs:
            raise ValueError(f"No valid reward functions found. Available: {list(reward_funcs_registry.keys())}")

        train_dataset = PromptDataset(prompts_file=args.prompts_file)
        train_sampler = None
        rank0_print(f"Loaded {len(train_dataset)} prompts from {args.prompts_file}")

    rank0_print(f"Using reward functions: {[f[0] for f in reward_funcs]}")

    # Print configuration
    if is_rank_zero():
        rank0_print("=" * 60)
        rank0_print("GRPO Training Configuration")
        rank0_print("=" * 60)
        rank0_print("Rollout Stage (image generation):")
        rank0_print(f"  rollout_n: {args.rollout_n} (images per prompt)")
        rank0_print(f"  rollout_micro_batch_size: {args.rollout_micro_batch_size} (images per GPU per step)")
        rank0_print(f"  rollout_global_batch_size: {rollout_global_batch_size} (total images per step)")
        rank0_print(f"  rollout_prompt_batch_size: {rollout_prompt_batch_size} (prompts per GPU, auto-computed)")
        if args.rollout_n > args.rollout_micro_batch_size:
            forward_passes_per_prompt = math.ceil(args.rollout_n / args.rollout_micro_batch_size)
            rank0_print(f"  Note: rollout_n ({args.rollout_n}) > rollout_micro_batch_size ({args.rollout_micro_batch_size}), "
                        f"using {forward_passes_per_prompt} forward passes per prompt")
        rank0_print(f"  rollout_accumulation_steps: {rollout_accumulation_steps} (rollout iterations to reach global batch)")
        rank0_print("Actor Training Stage:")
        rank0_print(f"  atrain_micro_batch_size: {args.atrain_micro_batch_size} (images per GPU per forward)")
        rank0_print(f"  atrain_global_batch_size: {args.atrain_global_batch_size} (total images per optimization step)")
        rank0_print(f"  world_size: {world_size}")
        rank0_print(f"  gradient_accumulation_steps: {gradient_accumulation_steps}")
        rank0_print(f"  atrain_num_actor_update_steps: {atrain_num_actor_update_steps} (actor updates per rollout)")
        if atrain_num_actor_update_steps > 1:
            total_train_samples = (args.atrain_global_batch_size or rollout_global_batch_size) * atrain_num_actor_update_steps
            rank0_print(f"  Note: {atrain_num_actor_update_steps} actor updates per rollout, "
                        f"total samples = {total_train_samples} of {rollout_global_batch_size} generated")
        elif args.atrain_global_batch_size is not None and args.atrain_global_batch_size < rollout_global_batch_size:
            rank0_print(f"  Note: training on {args.atrain_global_batch_size} of {rollout_global_batch_size} generated images")
        rank0_print("Training Algorithm Configuration:")
        rank0_print(f"  atrain_algorithm: {args.atrain_algorithm}")
        rank0_print(f"  atrain_sde_sampler: {args.atrain_sde_sampler}")
        rank0_print(f"  atrain_adv_type: {args.atrain_adv_type}")
        rank0_print(f"  atrain_kl_type: {args.atrain_kl_type}")
        rank0_print("EMA Configuration:")
        rank0_print(f"  ema_diffusion: {args.ema_diffusion} (0=disabled, >0=enabled)")
        if use_dataset_config:
            rank0_print("Dataset Configuration:")
            rank0_print(f"  config_file: {args.dataset_config}")
            rank0_print(f"  multi_dataset_mode: enabled")
        # Evaluation Configuration
        if args.eval_freq > 0 or args.eval_before_train:
            rank0_print("Evaluation Configuration:")
            rank0_print(f"  eval_freq: {args.eval_freq} (steps, 0=disabled)")
            rank0_print(f"  eval_before_train: {args.eval_before_train}")
            rank0_print(f"  eval_dataset_config: {args.eval_dataset_config}")
            rank0_print(f"  eval_inference_mode: {args.eval_inference_mode}")
            rank0_print(f"  eval_cfg_scale: {args.eval_cfg_scale if args.eval_cfg_scale is not None else f'{args.cfg_scale} (from training)'}")
            rank0_print(f"  eval_num_inference_steps: {args.eval_num_inference_steps if args.eval_num_inference_steps is not None else f'{args.num_inference_steps} (from training)'}")
            if args.eval_inference_mode == "sde":
                rank0_print(f"  eval_sde_eta: {args.eval_sde_eta if args.eval_sde_eta is not None else f'{args.sde_eta} (from training)'}")
                rank0_print(f"  atrain_sde_sampler: {args.atrain_sde_sampler}")
            rank0_print(f"  eval_image_size: {args.eval_image_height if args.eval_image_height else args.image_height}x{args.eval_image_width if args.eval_image_width else args.image_width}")
            rank0_print(f"  eval_micro_batch_size: {args.eval_micro_batch_size}")
            rank0_print(f"  eval_wandb_num_upload_images: {args.eval_wandb_num_upload_images}")
            rank0_print(f"  eval_swanlab_num_upload_images: {args.eval_swanlab_num_upload_images}")
        # SFT-Aux Configuration
        rank0_print("SFT-Aux Configuration:")
        rank0_print(f"  sftaux_coef: {args.sftaux_coef} (enabled if > 0)")
        if args.sftaux_coef > 0:
            rank0_print(f"  sftaux_dataset_config: {args.sftaux_dataset_config}")
            rank0_print(f"  sftaux_every_n_steps: {args.sftaux_every_n_steps}")
            rank0_print(f"  sftaux_micro_batch_size: {args.sftaux_micro_batch_size}")
            rank0_print(f"  sftaux_num_workers: {args.sftaux_num_workers}")
            rank0_print(f"  sftaux_disable_on_error: {args.sftaux_disable_on_error}")
        rank0_print("=" * 60)

    # Parse report_to: support comma-separated values for multiple backends
    # e.g., "wandb,tensorboard" -> ["wandb", "tensorboard"]
    report_to = args.report_to
    if isinstance(report_to, str) and "," in report_to:
        report_to = [x.strip() for x in report_to.split(",") if x.strip()]

    # Create training arguments
    # Note: per_device_train_batch_size here is the DataLoader batch size (prompts per GPU)
    # This is used by the Trainer's DataLoader to load prompts
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=rollout_prompt_batch_size,  # Prompts per GPU for DataLoader
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=report_to,
        run_name=args.run_name or f"deepgen_grpo_{'-'.join(args.reward_funcs)}",
        deepspeed=args.deepspeed,
        num_generations=args.rollout_n,
        beta=args.beta,
        max_grad_norm=args.max_grad_norm,
    )

    # Create trainer
    trainer = DeepGenGRPOTrainer(
        model_config_path=args.model_config,
        checkpoint_path=args.checkpoint,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        rollout_n=args.rollout_n,
        rollout_micro_batch_size=args.rollout_micro_batch_size,  # Images per GPU per rollout
        rollout_accumulation_steps=rollout_accumulation_steps,  # Rollout iterations to reach global batch
        atrain_micro_batch_size=args.atrain_micro_batch_size,  # Images per GPU per training forward
        atrain_num_actor_update_steps=atrain_num_actor_update_steps,  # Actor updates per rollout
        log_prob_micro_batch_size=args.log_prob_micro_batch_size,  # Samples per micro-batch for log_prob
        beta=args.beta,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        image_height=args.image_height,
        image_width=args.image_width,
        gradient_checkpointing=args.gradient_checkpointing,
        log_images_interval=args.log_images_interval,
        init_same_noise=args.init_same_noise,
        timestep_fraction=args.timestep_fraction,
        clip_range=args.clip_range,
        sde_eta=args.sde_eta,
        use_per_sample_reward_config=use_dataset_config,  # Enable per-sample reward weights
        train_sampler=train_sampler,  # Weighted sampling based on sample_weight config
        # Training algorithm: flowgrpo (default) or grpoguard
        atrain_algorithm=args.atrain_algorithm,
        # SDE sampler type: flowgrpo_sde (standard), cps_sde, or dance_sde
        atrain_sde_sampler=args.atrain_sde_sampler,
        # Advantage computation type: grpo, grpo_global_std, reinforcepp, or gdpo
        atrain_adv_type=args.atrain_adv_type,
        # KL divergence type: x-based or v-based
        atrain_kl_type=args.atrain_kl_type,
        # DiffusionNFT parameters
        atrain_nft_beta=args.atrain_nft_beta,
        atrain_nft_adv_clip_range=args.atrain_nft_adv_clip_range,
        atrain_nft_off_policy=args.atrain_nft_off_policy,
        # EMA for diffusion model (transformer)
        ema_diffusion=args.ema_diffusion,
        # Evaluation parameters
        eval_freq=args.eval_freq,
        eval_before_train=args.eval_before_train,
        eval_dataset_config=args.eval_dataset_config,
        eval_inference_mode=args.eval_inference_mode,
        eval_cfg_scale=args.eval_cfg_scale,
        eval_num_inference_steps=args.eval_num_inference_steps,
        eval_sde_eta=args.eval_sde_eta,
        eval_image_height=args.eval_image_height,
        eval_image_width=args.eval_image_width,
        eval_micro_batch_size=args.eval_micro_batch_size,
        eval_wandb_num_upload_images=args.eval_wandb_num_upload_images,
        eval_swanlab_num_upload_images=args.eval_swanlab_num_upload_images,
        # SFT-Aux parameters
        sftaux_dataset_config=args.sftaux_dataset_config,
        sftaux_coef=args.sftaux_coef,
        sftaux_every_n_steps=args.sftaux_every_n_steps,
        sftaux_micro_batch_size=args.sftaux_micro_batch_size,
        sftaux_num_workers=args.sftaux_num_workers,
        sftaux_disable_on_error=args.sftaux_disable_on_error,
        # Logging: component grad-norm proxy
        log_component_grad_norm=args.log_component_grad_norm,
        log_component_grad_norm_every=args.log_component_grad_norm_every,
        # Logging: approximate TFLOPS
        log_tflops=args.log_tflops,
        log_tflops_every=args.log_tflops_every,
        log_tflops_warmup_steps=args.log_tflops_warmup_steps,
    )

    # Resume from checkpoint if available
    last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint:
        rank0_print(f"Resuming from checkpoint: {last_checkpoint}")

    # Train
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save final model
    trainer.save_model(args.output_dir)
    rank0_print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
