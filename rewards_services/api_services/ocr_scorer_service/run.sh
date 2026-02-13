# cd /apdcephfs_sh3/share_300771694/hunyuan/ruihangli/workspace/deepgen/DeepGen-RL/rewards_services/api_services/ocr_scorer_service

export LD_LIBRARY_PATH=/usr/lib/python3/dist-packages/torch/lib:$LD_LIBRARY_PATH

source /apdcephfs_sh3/share_300771694/hunyuan/ruihangli/miniconda3/bin/activate unirl_ocr

python -c 'from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)'

gunicorn -c gunicorn.conf.py "app:create_app()"

# pkill gunicorn to stop