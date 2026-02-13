# cd /inspire/ssd/project/deepgen/liruihang-253108100075/workspace/deepgen/DeepGen-RL/rewards_services/api_services/ocr_scorer_service

# export JONB_DEBUG_OCR=1

# OCR Reward Hyperparameters
export OCR_MIN_CONF=0.2

cp -r /inspire/qb-ilm/project/deepgen/public/homes/ruihangli/projects/deepgen/deepgen_rl/dependencies/paddleocr/.paddleocr /root/

export PATH=/inspire/ssd/project/deepgen/liruihang-253108100075/miniconda3/envs/unirl_ocr/bin:${PATH}
export PYTHONPATH=/inspire/ssd/project/deepgen/liruihang-253108100075/miniconda3/envs/unirl_ocr/lib/python3.10/site-packages:${PYTHONPATH}

python -c 'from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)'

# 检查 PyTorch CUDA
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())"

# 检查 PaddlePaddle
python -c "import paddle; print('Paddle CUDA available:', paddle.is_compiled_with_cuda())"

gunicorn -c gunicorn.conf.py "app:create_app()"

# pkill gunicorn to stop