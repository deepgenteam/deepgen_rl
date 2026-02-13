#!/usr/bin/env python3
"""
Convert RL checkpoint (GRPO trainer saved with DeepSpeed/safetensors) to SFT checkpoint format
that can be loaded by gen_eval.py using guess_load_checkpoint.

RL checkpoint format (DeepSpeed + HuggingFace Trainer):
    - checkpoint-XX/model.safetensors  (merged model weights in safetensors format)
    - checkpoint-XX/global_stepXX/     (optimizer states, not needed for inference)

SFT checkpoint format (mmengine):
    - iter_XXXXX.pth/mp_rank_00_model_states.pt  (DeepSpeed format with 'module' key)
    - Or a single .pth file with state_dict

The converted checkpoint can be loaded by:
    from xtuner.model.utils import guess_load_checkpoint
    state_dict = guess_load_checkpoint(output_path)
    model.load_state_dict(state_dict, strict=False)

Usage:
    python scripts/utils/rlckpt_to_sftckpt.py \
        --rl_checkpoint /path/to/checkpoint-XX \
        --output /path/to/output.pth

    # Or specify the safetensors file directly:
    python scripts/utils/rlckpt_to_sftckpt.py \
        --rl_checkpoint /path/to/checkpoint-XX/model.safetensors \
        --output /path/to/output.pth
"""

import os
import argparse
import torch
from safetensors import safe_open
from collections import OrderedDict


def load_safetensors(path: str) -> OrderedDict:
    """Load state dict from safetensors file."""
    state_dict = OrderedDict()
    with safe_open(path, framework='pt', device='cpu') as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def find_safetensors_file(checkpoint_path: str) -> str:
    """
    Find the safetensors file in the checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory or safetensors file

    Returns:
        Path to the safetensors file
    """
    if os.path.isfile(checkpoint_path):
        if checkpoint_path.endswith('.safetensors'):
            return checkpoint_path
        else:
            raise ValueError(f"Expected .safetensors file, got: {checkpoint_path}")

    if os.path.isdir(checkpoint_path):
        # Look for model.safetensors in the checkpoint directory
        safetensors_path = os.path.join(checkpoint_path, 'model.safetensors')
        if os.path.exists(safetensors_path):
            return safetensors_path

        # Look for any .safetensors file
        for f in os.listdir(checkpoint_path):
            if f.endswith('.safetensors'):
                return os.path.join(checkpoint_path, f)

        raise FileNotFoundError(
            f"No .safetensors file found in {checkpoint_path}. "
            f"Contents: {os.listdir(checkpoint_path)}"
        )

    raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")


def convert_rl_to_sft(rl_checkpoint: str, output_path: str, verbose: bool = True):
    """
    Convert RL checkpoint to SFT format.

    Args:
        rl_checkpoint: Path to RL checkpoint (directory or .safetensors file)
        output_path: Path to save the converted checkpoint (.pth file)
        verbose: Whether to print progress information
    """
    # Find and load the safetensors file
    safetensors_path = find_safetensors_file(rl_checkpoint)
    if verbose:
        print(f"Loading RL checkpoint from: {safetensors_path}")

    state_dict = load_safetensors(safetensors_path)
    if verbose:
        print(f"Loaded {len(state_dict)} parameters")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save as .pth file (simple state_dict format)
    # guess_load_checkpoint will load this directly
    if verbose:
        print(f"Saving converted checkpoint to: {output_path}")

    torch.save(state_dict, output_path)

    if verbose:
        file_size = os.path.getsize(output_path) / (1024 ** 3)
        print(f"Saved checkpoint: {file_size:.2f} GB")
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert RL checkpoint to SFT format for gen_eval.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--rl_checkpoint", "-i",
        type=str,
        required=True,
        help="Path to RL checkpoint directory (e.g., checkpoint-95) or .safetensors file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for converted checkpoint (.pth file)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Ensure output has .pth extension
    if not args.output.endswith('.pth'):
        args.output = args.output + '.pth'

    convert_rl_to_sft(
        rl_checkpoint=args.rl_checkpoint,
        output_path=args.output,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()

