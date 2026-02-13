# Installation

This guide covers the complete setup for DeepGen-RL.

## Prerequisites

- Linux (tested on Ubuntu 20.04+)
- Python 3.11
- CUDA 12.x
- 8+ GPUs with 80GB VRAM each (recommended for full training)

## Step 1: Create Conda Environment

```bash
conda env create -f environment.yml
conda activate deepgen_rl
```

## Step 2: Install Additional Dependencies

After activating the conda environment, install the following packages that are not included in `environment.yml`:

```bash
# MMEngine and xtuner (required for SFT auxiliary training and model loading)
pip install -U openmim
mim install mmengine
pip install xtuner

# Pin specific versions for compatibility
pip install triton==3.1.0
pip install bitsandbytes==0.48.1
pip install transformers==4.51.3
```

## Step 3: Install CLIP

```bash
pip install git+https://github.com/openai/CLIP.git
```

## Step 4: Install Diffusers

Install from source to get the latest features:

```bash
pip install git+https://github.com/huggingface/diffusers.git
```

## Step 5: Install Flash Attention

```bash
pip install flash-attn==2.7.2.post1 --no-build-isolation
```

If building from source fails, you can install a prebuilt wheel:

```bash
# For CUDA 12 + PyTorch 2.7 + Python 3.11:
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# For CUDA 12 + PyTorch 2.6 + Python 3.11:
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

## Step 6: DeepSpeed Patch (Required)

DeepSpeed has a bug that prevents training resumption from checkpoints. You need to apply a one-line patch to the `TorchCheckpointEngine`:

**File to modify:**
```
$(python -c "import deepspeed; import os; print(os.path.join(os.path.dirname(deepspeed.__file__), 'runtime/checkpoint_engine/torch_checkpoint_engine.py'))")
```

**Change in the `load` method:**

```python
# Before (original):
partition = torch.load(path, map_location=map_location)

# After (patched):
partition = torch.load(path, map_location=map_location, weights_only=False)
```

## Step 7: Reward Service Dependencies (Optional)

If you plan to use reward services during training, each service has its own environment. For example, for the OCR reward service:

```bash
cd rewards_services/api_services/ocr_scorer_service
conda create -n deepgen_rl_ocr python=3.10 -y
conda activate deepgen_rl_ocr
pip install -r requirements.txt
```

See `rewards_services/api_services/*/readme.txt` for service-specific setup instructions.

## Verification

After installation, verify the setup:

```bash
conda activate deepgen_rl
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed {deepspeed.__version__}')"
python -c "import mmengine; print(f'MMEngine {mmengine.__version__}')"
python -c "import flash_attn; print(f'Flash Attention {flash_attn.__version__}')"
```

## Next Steps

Proceed to the [Training Guide](training.md) to configure and run RL training.
