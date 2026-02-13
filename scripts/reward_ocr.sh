#!/bin/bash
# DeepGen-RL: OCR Reward Service Launcher
#
# Starts the OCR scoring service used as a reward signal during RL training.
# The service uses PaddleOCR to evaluate text rendering quality in generated images.
#
# Prerequisites:
#   conda create -n deepgen_rl_ocr python=3.10 -y
#   conda activate deepgen_rl_ocr
#   pip install -r rewards_services/api_services/ocr_scorer_service/requirements.txt
#
# Usage:
#   bash scripts/reward_ocr.sh
#
# Environment variables:
#   OCR_PORT          - Port to serve on (default: 18082)
#   CUDA_VISIBLE_DEVICES - GPU devices; set to "" for CPU-only (default: "", CPU mode)

set -euo pipefail

OCR_PORT="${OCR_PORT:-18082}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
SERVICE_DIR="${PROJECT_ROOT}/rewards_services/api_services/ocr_scorer_service"

if [[ ! -d "${SERVICE_DIR}" ]]; then
    echo "ERROR: OCR service directory not found: ${SERVICE_DIR}"
    exit 1
fi

echo "=========================================="
echo "Starting OCR Reward Service"
echo "=========================================="
echo "  Service Dir: ${SERVICE_DIR}"
echo "  Port:        ${OCR_PORT}"
echo "=========================================="

cd "${SERVICE_DIR}"

# Run on CPU by default to avoid competing with training for GPU memory.
# Set CUDA_VISIBLE_DEVICES to a GPU index if GPU acceleration is desired.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

# Pre-warm PaddleOCR to download/cache models before gunicorn forks workers.
python -c "
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False)
print('PaddleOCR pre-warm complete.')
"

# Launch the service via gunicorn
exec gunicorn -c gunicorn.conf.py "app:create_app()"
