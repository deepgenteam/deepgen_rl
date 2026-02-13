# Copyright 2025 Ruihang Li and DeepGen Team @ Shanghai Innovation Institute

import requests
import pickle
from PIL import Image
from typing import List, Dict, Any, Union
import sys
import os
import pickle
from io import BytesIO
import concurrent.futures
import math

# Get reward service address from environment variable, default to localhost
# Each service can have its own URL via <SERVICE_NAME>_URL environment variable
# The URL should include host and port, e.g., "localhost:18082" or "192.168.1.100:8080"
# Falls back to default localhost with predefined ports if not set
REWARD_SERVICE_ADDR = os.environ.get("REWARD_SERVICE_ADDR", "localhost")


def _normalize_url(url: str) -> str:
    """Normalize URL to ensure it has http:// prefix and trailing slash."""
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"
    if not url.endswith("/"):
        url = f"{url}/"
    return url


# Service-specific URLs (can include http:// prefix or just host:port)
# Environment variables:
#   - AESTHETIC_URL: URL for aesthetic scorer (default: localhost:18080)
#   - IMAGE_REWARD_URL: URL for image_reward scorer (default: localhost:18081)
#   - OCR_URL: URL for OCR scorer (default: localhost:18082)
#   - PICKSCORE_URL: URL for pickscore scorer (default: localhost:18083)
#   - DEQA_URL: URL for deqa scorer (default: localhost:18084)
#   - GEN_EVAL_URL: URL for gen_eval scorer (default: localhost:18085)
#   - HPS_URL: URL for hps scorer (default: localhost:18087)
#   - EDITREWARD_URL: URL for editreward scorer (default: localhost:18088)
#   - UNIFIEDREWARD_SGLANG_URL: URL for unifiedreward_sglang scorer (default: localhost:18089)
#
# Concurrency control environment variables (number of parallel workers for each reward):
#   - AESTHETIC_WORKERS: Number of parallel workers for aesthetic scorer (default: 1)
#   - IMAGE_REWARD_WORKERS: Number of parallel workers for image_reward scorer (default: 1)
#   - OCR_WORKERS: Number of parallel workers for OCR scorer (default: 1)
#   - PICKSCORE_WORKERS: Number of parallel workers for pickscore scorer (default: 1)
#   - DEQA_WORKERS: Number of parallel workers for deqa scorer (default: 1)
#   - GEN_EVAL_WORKERS: Number of parallel workers for gen_eval scorer (default: 1)
#   - HPS_WORKERS: Number of parallel workers for hps scorer (default: 1)
#   - EDITREWARD_WORKERS: Number of parallel workers for editreward scorer (default: 1)
#   - UNIFIEDREWARD_SGLANG_WORKERS: Number of parallel workers for unifiedreward_sglang scorer (default: 1)
AESTHETIC_URL = os.environ.get("AESTHETIC_URL", f"{REWARD_SERVICE_ADDR}:18080")
IMAGE_REWARD_URL = os.environ.get("IMAGE_REWARD_URL", f"{REWARD_SERVICE_ADDR}:18081")
OCR_URL = os.environ.get("OCR_URL", f"{REWARD_SERVICE_ADDR}:18082")
PICKSCORE_URL = os.environ.get("PICKSCORE_URL", f"{REWARD_SERVICE_ADDR}:18083")
DEQA_URL = os.environ.get("DEQA_URL", f"{REWARD_SERVICE_ADDR}:18084")
GEN_EVAL_URL = os.environ.get("GEN_EVAL_URL", f"{REWARD_SERVICE_ADDR}:18085")
HPS_URL = os.environ.get("HPS_URL", f"{REWARD_SERVICE_ADDR}:18087")
EDITREWARD_URL = os.environ.get("EDITREWARD_URL", f"{REWARD_SERVICE_ADDR}:18088")
UNIFIEDREWARD_SGLANG_URL = os.environ.get("UNIFIEDREWARD_SGLANG_URL", f"{REWARD_SERVICE_ADDR}:18089")

SCORER_URLS = {
    "aesthetic": _normalize_url(AESTHETIC_URL),
    "image_reward": _normalize_url(IMAGE_REWARD_URL),
    "ocr": _normalize_url(OCR_URL),
    "pickscore": _normalize_url(PICKSCORE_URL),
    "deqa": _normalize_url(DEQA_URL),
    "gen_eval": _normalize_url(GEN_EVAL_URL),
    "hps": _normalize_url(HPS_URL),
    "editreward": _normalize_url(EDITREWARD_URL),
    "unifiedreward_sglang": _normalize_url(UNIFIEDREWARD_SGLANG_URL),
}

# Concurrency settings for each scorer (number of parallel workers)
# Default to 1 (sequential) if not specified
SCORER_WORKERS = {
    "aesthetic": int(os.environ.get("AESTHETIC_WORKERS", "1")),
    "image_reward": int(os.environ.get("IMAGE_REWARD_WORKERS", "1")),
    "ocr": int(os.environ.get("OCR_WORKERS", "1")),
    "pickscore": int(os.environ.get("PICKSCORE_WORKERS", "1")),
    "deqa": int(os.environ.get("DEQA_WORKERS", "1")),
    "gen_eval": int(os.environ.get("GEN_EVAL_WORKERS", "1")),
    "hps": int(os.environ.get("HPS_WORKERS", "1")),
    "editreward": int(os.environ.get("EDITREWARD_WORKERS", "1")),
    "unifiedreward_sglang": int(os.environ.get("UNIFIEDREWARD_SGLANG_WORKERS", "1")),
}

class RewardEvaluatorClient:
    def __init__(self, scorer_urls: Dict[str, str] = SCORER_URLS, scorer_workers: Dict[str, int] = SCORER_WORKERS):
        self.scorer_urls = scorer_urls
        self.scorer_workers = scorer_workers

    def _send_request(self, url: str, payload_bytes: bytes, timeout: int = 600) -> bytes:
        """Send a single request to the reward server and return response content."""
        response = requests.post(url, data=payload_bytes, timeout=timeout)
        response.raise_for_status()
        return response.content

    def _evaluate_batch(self,
                        url: str,
                        images: List[Image.Image],
                        prompts: List[str],
                        metadata: Dict[str, Any],
                        batch_indices: List[int]) -> tuple:
        """Evaluate a batch of images and return (batch_indices, results)."""
        batch_images = [images[i] for i in batch_indices]
        batch_prompts = [prompts[i] for i in batch_indices]
        payload_bytes = create_payload(batch_images, batch_prompts, metadata)
        response_content = self._send_request(url, payload_bytes)
        result = parse_response(response_content)
        return (batch_indices, result)

    def evaluate(self,
                 model_name: str,
                 images: List[Image.Image],
                 prompts: List[str],
                 metadata: Dict[str, Any] = None) -> Union[List[float], Dict[str, Any]]:
        url = self.scorer_urls.get(model_name)
        if not url:
            raise ValueError(f"Reward model '{model_name}' URL not configured.")

        # Get concurrency setting for this model
        max_workers = self.scorer_workers.get(model_name, 1)

        # Debug logging (only when JONB_DEBUG_OCR is set)
        _debug_ocr = os.environ.get("JONB_DEBUG_OCR", None) is not None
        log_dir = os.environ.get("LOG_DIR", ".")
        debug_path = os.path.join(log_dir, "debug_reward_client.log")
        if _debug_ocr:
            try:
                with open(debug_path, "a") as f:
                    f.write(f"\n=== RewardEvaluatorClient.evaluate ===")
                    f.write(f"\nmodel_name: {model_name}")
                    f.write(f"\nurl: {url}")
                    f.write(f"\nnum_images: {len(images)}")
                    f.write(f"\nnum_prompts: {len(prompts)}")
                    f.write(f"\nprompts: {prompts}")
                    f.write(f"\nmax_workers: {max_workers}")
                    if images:
                        f.write(f"\nfirst_image_size: {images[0].size}")
                        f.write(f"\nfirst_image_mode: {images[0].mode}")
                    f.flush()
            except:
                pass

        try:
            # If max_workers <= 1, use sequential processing (original behavior)
            if max_workers <= 1:
                payload_bytes = create_payload(images, prompts, metadata)

                # Debug: log payload size (only when JONB_DEBUG_OCR is set)
                if _debug_ocr:
                    try:
                        with open(debug_path, "a") as f:
                            f.write(f"\npayload_size_bytes: {len(payload_bytes)}")
                            f.write(f"\nprocessing_mode: sequential")
                            f.flush()
                    except:
                        pass

                response = requests.post(url, data=payload_bytes, timeout=600)
                response.raise_for_status()
                result = parse_response(response.content)
            else:
                # Multi-threaded processing: split images into batches
                num_images = len(images)
                batch_size = math.ceil(num_images / max_workers)

                # Create batches of indices
                batches = []
                for i in range(0, num_images, batch_size):
                    end_idx = min(i + batch_size, num_images)
                    batches.append(list(range(i, end_idx)))

                # Adjust number of workers to actual number of batches
                actual_workers = min(max_workers, len(batches))

                if _debug_ocr:
                    try:
                        with open(debug_path, "a") as f:
                            f.write(f"\nprocessing_mode: parallel")
                            f.write(f"\nnum_batches: {len(batches)}")
                            f.write(f"\nbatch_size: {batch_size}")
                            f.write(f"\nactual_workers: {actual_workers}")
                            f.flush()
                    except:
                        pass

                # Process batches in parallel
                all_results = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
                    futures = []
                    for batch_indices in batches:
                        future = executor.submit(
                            self._evaluate_batch,
                            url,
                            images,
                            prompts,
                            metadata,
                            batch_indices
                        )
                        futures.append(future)

                    # Collect results
                    for future in concurrent.futures.as_completed(futures):
                        batch_indices, batch_result = future.result()
                        # Handle different result formats
                        if isinstance(batch_result, dict) and "scores" in batch_result:
                            scores = batch_result["scores"]
                        elif isinstance(batch_result, (list, tuple)):
                            scores = list(batch_result)
                        else:
                            scores = batch_result

                        # Store results with their original indices
                        for idx, score in zip(batch_indices, scores):
                            all_results[idx] = score

                # Reconstruct result in original order
                result = [all_results[i] for i in range(num_images)]

                if _debug_ocr:
                    try:
                        with open(debug_path, "a") as f:
                            f.write(f"\nmerged_results_count: {len(result)}")
                            f.flush()
                    except:
                        pass

            # Debug: log response status (only when JONB_DEBUG_OCR is set)
            if _debug_ocr:
                try:
                    with open(debug_path, "a") as f:
                        f.write(f"\nparsed_result_type: {type(result)}")
                        f.write(f"\nparsed_result: {result}")
                        f.write(f"\n=== RewardEvaluatorClient.evaluate End ===\n")
                        f.flush()
                except:
                    pass

            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(f"Scorer '{model_name}' service returned error: {result['error']}")

            return result

        except requests.exceptions.RequestException as e:
            # Debug: log request error (only when JONB_DEBUG_OCR is set)
            if _debug_ocr:
                try:
                    with open(debug_path, "a") as f:
                        f.write(f"\n!!! Request ERROR: {str(e)} !!!\n")
                        f.flush()
                except:
                    pass
            raise RuntimeError(f"HTTP request to '{model_name}' failed: {e}")
        except Exception as e:
            # Debug: log general error (only when JONB_DEBUG_OCR is set)
            if _debug_ocr:
                try:
                    import traceback
                    with open(debug_path, "a") as f:
                        f.write(f"\n!!! General ERROR: {str(e)} !!!")
                        f.write(f"\n{traceback.format_exc()}\n")
                        f.flush()
                except:
                    pass
            raise RuntimeError(f"Failed to process response from '{model_name}': {e}")

    def evaluate_multiple(self,
                          model_weights: Dict[str, float],
                          images: List[Image.Image],
                          prompts: List[str],
                          metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        all_results = {}
        for model_name, weight in model_weights.items():
            if weight == 0:
                continue
            try:
                if model_name in ["gen_eval"]:
                    specific_metadata = metadata.get(model_name, {})
                    result = self.evaluate(model_name, images, prompts, specific_metadata)
                else:
                    result = self.evaluate(model_name, images, prompts, metadata.get(model_name, {}))
                all_results[model_name] = result
            except Exception as e:
                print(f"Error evaluating model {model_name}: {e}")
                all_results[model_name] = {"error": str(e)}
        return all_results


def serialize_images(images: List[Image.Image]) -> List[bytes]:
    images_bytes = []
    for img in images:
        img_byte_arr = BytesIO()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # img.save(img_byte_arr, format="JPEG", quality=95)
        img.save(img_byte_arr, format="JPEG")
        images_bytes.append(img_byte_arr.getvalue())
    return images_bytes

def deserialize_images(images_bytes: List[bytes]) -> List[Image.Image]:
    images = [Image.open(BytesIO(d)) for d in images_bytes]
    return images

def create_payload(images: List[Image.Image], prompts: List[str], metadata: Dict[str, Any] = None) -> bytes:
    serialized_images = serialize_images(images) if isinstance(images, list) else dict({key: serialize_images(value) for key, value in images.items()})
    payload = {
        "images": serialized_images,
        "prompts": prompts,
        "metadata": metadata if metadata is not None else {}
    }
    return pickle.dumps(payload)

def parse_response(response_content: bytes) -> Union[List[float], Dict[str, Any]]:
    return pickle.loads(response_content)