"""
Model config for Qwen2.5-VL + SD3 DPO Fusion model.

This model uses multi-layer hidden states fusion from the LLM for better
feature extraction. It concatenates hidden states from 6 layers (every 6th
layer from the last) before projecting to the diffusion model's embedding space.
"""

import torch
from deepgen_rl.models.deepgen.qwen2_5_vl_sd3_hf_dynamic_dpo_fusion import Qwen2p5VLStableDiffusion3DPOFusionHF
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from deepgen_rl.models.deepgen.transformer_sd3_dynamic import SD3Transformer2DModel

# Use __import__ to get the real os module and bypass mmengine's lazy import mechanism
_os = __import__('os')
_getenv = _os.environ.get

# Read from environment variables with default fallback
sd3_5_model_name_or_path = _getenv("SD3_5_MODEL_NAME_OR_PATH", "/apdcephfs_sh3/share_300771694/hunyuan/ruihangli/huggingface/UniPic2-SD3.5M-Kontext-2B")
qwen2_5_vl_model_name_or_path = _getenv("QWEN2_5_VL_MODEL_NAME_OR_PATH", "/apdcephfs_sh3/share_300771694/hunyuan/ruihangli/huggingface/Qwen2.5-VL-3B-Instruct")

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=qwen2_5_vl_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

prompt_template = dict(
    IMG_START_TOKEN='<|vision_start|>',
    IMG_END_TOKEN='<|vision_end|>',
    IMG_CONTEXT_TOKEN='<|image_pad|>',
    IMG_START_TOKEN_FOR_GENERATION=False,
    SYSTEM=('<|im_start|>system\n{system}<|im_end|>\n'),
    INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                 '<|im_start|>assistant\n'),
    SUFFIX='<|im_end|>',
    SUFFIX_AS_EOS=True,
    SEP='\n',
    STOP_WORDS=['<|im_end|>', '<|endoftext|>'],
    GENERATION='Generate an image: {input}',
    CFG='Generate an image.'
)


model = dict(
    type=Qwen2p5VLStableDiffusion3DPOFusionHF,
    num_queries=128,
    num_fusion_layers=6,  # Number of layers to fuse (every 6th layer from last)
    connector=dict(
        hidden_size=2048,
        intermediate_size=11946,
        num_hidden_layers=6,
        _attn_implementation='flash_attention_2',
        num_attention_heads=32,
    ),
    lmm=dict(
        type=Qwen2_5_VLForConditionalGeneration.from_pretrained,
        pretrained_model_name_or_path=qwen2_5_vl_model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ),
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    freeze_lmm=True,
    freeze_mq=True,  # Freeze connector, projector_1/2/3, and meta_queries
    transformer=dict(
        type=SD3Transformer2DModel.from_pretrained,
        pretrained_model_name_or_path=sd3_5_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ),
    test_scheduler=dict(
        type=FlowMatchEulerDiscreteScheduler.from_pretrained,
        pretrained_model_name_or_path=sd3_5_model_name_or_path,
        subfolder="scheduler",
    ),
    train_scheduler=dict(
        type=FlowMatchEulerDiscreteScheduler.from_pretrained,
        pretrained_model_name_or_path=sd3_5_model_name_or_path,
        subfolder="scheduler",
    ),
    vae=dict(
        type=AutoencoderKL.from_pretrained,
        pretrained_model_name_or_path=sd3_5_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ),
    pretrained_pth=None,
    use_activation_checkpointing=False,
    freeze_transformer=False,
    dpo_beta=0.01,
)
