import os
from mmengine.config import read_base
from deepgen_rl.sft.datasets.collate_functions import collate_func_gen_txt_dynamic
from deepgen_rl.sft.datasets.text2image.blip3_o import BLIP3oDataset
from deepgen_rl.sft.datasets.samplers.weighted_infinite_sampler import WeightedInfiniteSampler
from xtuner.dataset import ConcatDataset

with read_base():
    from .processors import image_size, image_process

image_process = 'fix_pixels'

# NOTE: Set the DATA_ROOT environment variable to point to your local data directory.
# e.g., export DATA_ROOT=/path/to/your/data
DATA_ROOT = os.environ.get('DATA_ROOT', '/path/to/your/data')

# General T2I datasets (sample_weight: 1.0)
dataset_blip3o60k = dict(
    type=BLIP3oDataset,
    image_size=image_size,
    data_path=f'{DATA_ROOT}/t2i_grpo_sft/blip3o_60k.json',
    image_process=image_process,
    ceph_folder=None,
    ceph_config=None,
    sample_weight=1.0,
)

dataset_share4oimg = dict(
    type=BLIP3oDataset,
    image_size=image_size,
    data_path=f'{DATA_ROOT}/t2i_grpo_sft/share_4o_img.json',
    image_folder=f'{DATA_ROOT}/raw_data/ShareGPT-4o-Image/t2i',
    image_process=image_process,
    ceph_folder=None,
    ceph_config=None,
    sample_weight=1.0,
)

dataset_echo4oimg = dict(
    type=BLIP3oDataset,
    image_size=image_size,
    data_path=f'{DATA_ROOT}/t2i_grpo_sft/echo-4o-image_t2i.json',
    image_folder=f'{DATA_ROOT}/raw_data',
    image_process=image_process,
    ceph_folder=None,
    ceph_config=None,
    sample_weight=1.0,
)

dataset_open4oimg = dict(
    type=BLIP3oDataset,
    image_size=image_size,
    data_path=f'{DATA_ROOT}/t2i_grpo_sft/OpenGPT-4o-Image.json',
    image_folder=f'{DATA_ROOT}/raw_data/OpenGPT-4o-Image',
    image_process=image_process,
    ceph_folder=None,
    ceph_config=None,
    sample_weight=1.0,
)

dataset_genevalimg = dict(
    type=BLIP3oDataset,
    image_size=image_size,
    data_path=f'{DATA_ROOT}/t2i_grpo_sft/Geneval_t2i.json',
    image_folder=f'{DATA_ROOT}/source_images',
    image_process=image_process,
    ceph_folder=None,
    ceph_config=None,
    sample_weight=1.0,
)

dataset_unigen_banana = dict(
    type=BLIP3oDataset,
    image_size=image_size,
    data_path=f'{DATA_ROOT}/data_zoo/banana/banana_UniGenBench/banana_UniGenBench.json',
    image_process=image_process,
    ceph_folder=None,
    ceph_config=None,
    sample_weight=1.0,
)

# Text rendering datasets (sample_weight: 3.0)
dataset_poster_32k = dict(
    type=BLIP3oDataset,
    image_size=image_size,
    data_path=f'{DATA_ROOT}/data_zoo/poster/poster-z-image-en-32k/poster-z-32k-en.json',
    image_process=image_process,
    ceph_folder=None,
    ceph_config=None,
    sample_weight=3.0,
)

dataset_qwen_text = dict(
    type=BLIP3oDataset,
    image_size=image_size,
    data_path=f'{DATA_ROOT}/data_zoo/sft/qwenimage_textrender.json',
    image_process=image_process,
    ceph_folder=None,
    ceph_config=None,
    sample_weight=3.0,
)

dataset_z_text = dict(
    type=BLIP3oDataset,
    image_size=image_size,
    data_path=f'{DATA_ROOT}/data_zoo/z-distill/text_render_json/text_render.json',
    image_process=image_process,
    ceph_folder=None,
    ceph_config=None,
    sample_weight=3.0,
)

# General T2I datasets (sample_weight: 0.5)
dataset_jimeng_banana = dict(
    type=BLIP3oDataset,
    image_size=image_size,
    data_path=f'{DATA_ROOT}/data_zoo/banana/banana_jimeng/banana_jimeng.json',
    image_process=image_process,
    ceph_folder=None,
    ceph_config=None,
    sample_weight=0.5,
)

dataset_banana_1 = dict(
    type=BLIP3oDataset,
    image_size=image_size,
    data_path=f'{DATA_ROOT}/data_zoo/banana/banana_gpt4o-image-prompts-t2i-10k/banana_gpt4o-image-prompts-t2i-5k_en.json',
    image_process=image_process,
    ceph_folder=None,
    ceph_config=None,
    sample_weight=0.5,
)

dataset_banana_2 = dict(
    type=BLIP3oDataset,
    image_size=image_size,
    data_path=f'{DATA_ROOT}/data_zoo/banana/banana_gpt4o-image-prompts_en_t2i/banana_gpt4o-image-prompts_en_t2i.json',
    image_process=image_process,
    ceph_folder=None,
    ceph_config=None,
    sample_weight=0.5,
)

dataset_banana_3 = dict(
    type=BLIP3oDataset,
    image_size=image_size,
    data_path=f'{DATA_ROOT}/data_zoo/banana/banana_cv.json',
    image_process=image_process,
    ceph_folder=None,
    ceph_config=None,
    sample_weight=0.5,
)

dataset = dict(
    type=ConcatDataset,
    datasets=[
        # General T2I (sample_weight: 1.0)
        dataset_blip3o60k,
        dataset_share4oimg,
        dataset_echo4oimg,
        dataset_open4oimg,
        dataset_genevalimg,
        dataset_unigen_banana,
        # Text rendering (sample_weight: 3.0)
        dataset_poster_32k,
        dataset_qwen_text,
        dataset_z_text,
        # General T2I (sample_weight: 0.5)
        dataset_jimeng_banana,
        dataset_banana_1,
        dataset_banana_2,
        dataset_banana_3,
    ],
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(
        type=WeightedInfiniteSampler,
        shuffle=True,
    ),
    collate_fn=dict(type=collate_func_gen_txt_dynamic),
)
