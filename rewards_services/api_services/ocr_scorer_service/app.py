import sys
import os
import torch
from PIL import Image
import pickle
import traceback
from io import BytesIO

from flask import Flask, request, Blueprint

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../common')))
from utils import deserialize_images

INFERENCE_FN = None
OCR_IMPL = None  # "old" | "new"

root = Blueprint("root", __name__)

def _resolve_ocr_impl(cli_value: str = None) -> str:
    """
    Resolve which OCR scorer implementation to use.
    Priority: CLI arg > env var > default.
    """
    env_value = os.environ.get("OCR_SCORER_IMPL", None)
    raw = (cli_value or env_value or "old").strip().lower()
    # Accept a few aliases to make usage friendlier.
    if raw in ("old", "ocr_old", "legacy", "v0"):
        return "old"
    if raw in ("new", "ocr", "current", "v1"):
        return "new"
    raise ValueError(f"Unknown OCR_SCORER_IMPL '{raw}'. Expected 'old' or 'new'.")


def _get_scorer_class(impl: str):
    """
    Lazy import to avoid importing both PaddleOCR stacks at module import time.
    """
    if impl == "old":
        from ocr_old import OcrScorer as _OcrScorer
        return _OcrScorer
    if impl == "new":
        from ocr import OcrScorer as _OcrScorer
        return _OcrScorer
    raise ValueError(f"Unknown impl: {impl}")


def create_app(ocr_impl: str = None):
    global INFERENCE_FN
    global OCR_IMPL
    OCR_IMPL = _resolve_ocr_impl(ocr_impl)
    OcrScorer = _get_scorer_class(OCR_IMPL)

    print(f"Loading OCR Scorer model (impl={OCR_IMPL})...")
    use_gpu = torch.cuda.is_available()
    INFERENCE_FN = OcrScorer(use_gpu=use_gpu)
    print("OCR Scorer model loaded.")

    app = Flask(__name__)
    app.register_blueprint(root)
    return app

@root.route("/", methods=["POST"])
def inference():
    # Check if debug mode is enabled via environment variable
    _debug_ocr = os.environ.get("JONB_DEBUG_OCR", None) is not None

    if _debug_ocr:
        print(f"[DEBUG] Received POST request from {request.remote_addr}")
    data = request.get_data()

    try:
        payload = pickle.loads(data)
        images_bytes = payload["images"]
        prompts = payload.get("prompts", [])
        metadata = payload.get("metadata", {})

        images = deserialize_images(images_bytes)
        if _debug_ocr:
            print(f"[DEBUG] Got {len(images)} images for OCR Scorer")
            print(f"[DEBUG] Got {len(prompts)} prompts")
            print(f"[DEBUG] Prompts: {prompts}")
            if images:
                print(f"[DEBUG] First image size: {images[0].size}")
                print(f"[DEBUG] First image mode: {images[0].mode}")

        if not prompts:
            raise ValueError("OCR Scorer requires prompts with target text.")

        if _debug_ocr:
            print(f"[DEBUG] Calling INFERENCE_FN...")
        scores = INFERENCE_FN(images, prompts)
        if _debug_ocr:
            print(f"[DEBUG] INFERENCE_FN returned scores type: {type(scores)}")
            print(f"[DEBUG] INFERENCE_FN returned scores: {scores}")

        response = {"scores": scores}
        if _debug_ocr:
            print(f"[DEBUG] Response dict: {response}")
        response = pickle.dumps(response)
        if _debug_ocr:
            print(f"[DEBUG] Response size: {len(response)} bytes")
        return response, 200

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"[ERROR] Error in OCR Scorer service: {error_msg}")
        response = {"error": error_msg}
        response = pickle.dumps(response)
        return response, 500

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR scorer service")
    parser.add_argument(
        "--ocr-impl",
        default="old",
        choices=["old", "new"],
        help="Select OCR scorer implementation (default: old).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    app = create_app(ocr_impl=args.ocr_impl)
    app.run(host=args.host, port=args.port, debug=args.debug)