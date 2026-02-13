# 安装指南

本文档介绍 DeepGen-RL 的完整安装流程。

## 环境要求

- Linux（推荐 Ubuntu 20.04+）
- Python 3.11
- CUDA 12.x
- 8+ 块 80GB 显存 GPU（推荐完整训练配置）

## 第一步：创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate deepgen_rl
```

## 第二步：安装额外依赖

激活 conda 环境后，安装以下未包含在 `environment.yml` 中的依赖包：

```bash
# MMEngine 和 xtuner（SFT 辅助训练和模型加载所需）
pip install -U openmim
mim install mmengine
pip install xtuner

# 指定兼容版本
pip install triton==3.1.0
pip install bitsandbytes==0.48.1
pip install transformers==4.51.3
```

## 第三步：安装 CLIP

```bash
pip install git+https://github.com/openai/CLIP.git
```

## 第四步：安装 Diffusers

从源码安装以获取最新功能：

```bash
pip install git+https://github.com/huggingface/diffusers.git
```

## 第五步：安装 Flash Attention

```bash
pip install flash-attn==2.7.2.post1 --no-build-isolation
```

如果源码编译失败，可以安装预编译的 wheel 包：

```bash
# CUDA 12 + PyTorch 2.7 + Python 3.11：
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# CUDA 12 + PyTorch 2.6 + Python 3.11：
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

## 第六步：DeepSpeed 补丁（必需）

DeepSpeed 存在一个阻止从检查点恢复训练的 bug，需要对 `TorchCheckpointEngine` 进行单行修改：

**需要修改的文件：**
```
$(python -c "import deepspeed; import os; print(os.path.join(os.path.dirname(deepspeed.__file__), 'runtime/checkpoint_engine/torch_checkpoint_engine.py'))")
```

**修改 `load` 方法：**

```python
# 修改前（原始代码）：
partition = torch.load(path, map_location=map_location)

# 修改后（补丁代码）：
partition = torch.load(path, map_location=map_location, weights_only=False)
```

## 第七步：奖励服务依赖（可选）

如果在训练中使用奖励服务，每个服务有独立的环境。以 OCR 奖励服务为例：

```bash
cd rewards_services/api_services/ocr_scorer_service
conda create -n deepgen_rl_ocr python=3.10 -y
conda activate deepgen_rl_ocr
pip install -r requirements.txt
```

各服务的具体安装说明请参考 `rewards_services/api_services/*/readme.txt`。

## 验证安装

安装完成后，验证环境是否正确：

```bash
conda activate deepgen_rl
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed {deepspeed.__version__}')"
python -c "import mmengine; print(f'MMEngine {mmengine.__version__}')"
python -c "import flash_attn; print(f'Flash Attention {flash_attn.__version__}')"
```

## 下一步

请参阅[训练指南](training.md)配置并运行 RL 训练。
