# Copyright 2025 Ruihang Li and DeepGen Team @ Shanghai Innovation Institute

"""
OCR-vLLM reward for DeepGen GRPO.

This module sends (image, prompt) pairs to an OpenAI-compatible VLM endpoint and asks it to
estimate the character-level accuracy of rendered text in the image, returning a float in [0, 1].
"""

from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from ..utils.vllm_request import evaluate_batch
from ..utils.vllm_sleep_mode import maybe_switch_vllm_server


def _truthy_env(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _debug_enabled() -> bool:
    return _truthy_env("DEBUG_OCR_VLLM", default="0")


def _encode_image_to_base64(image: Image.Image) -> str:
    """Encode a PIL image to a base64 JPEG string (no data: prefix)."""
    image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_system_prompt() -> str:
    """
    Build the system prompt for OCR evaluation.

    NOTE: Comments are in English per repo convention.
    """
    return (
        "You are a strict evaluator for a text-to-image (T2I) image generation model.\n"
        "Your job is to judge how accurately the model rendered the expected text in the image.\n"
        "\n"
        "## Task\n"
        "Given:\n"
        "- A user prompt (the intended instruction for the image generation model)\n"
        "- A generated image\n"
        "\n"
        "Estimate the fraction of characters that are rendered correctly in the image, as a number in [0, 1].\n"
        "\n"
        "## What counts as 'expected text'\n"
        "- Identify the explicit text the prompt is asking to appear (e.g., quoted strings, 'write: ...', 'text says ...', "
        "signboard text, labels, logos with exact wording, or any clearly specified text snippet).\n"
        "- If multiple text snippets are specified, evaluate all of them together by concatenating in a reasonable order.\n"
        "- If the prompt does NOT specify any concrete text to be rendered, set score=0.0.\n"
        "\n"
        "## How to evaluate\n"
        "- First, read the text that actually appears in the image (visual OCR by inspection).\n"
        "- Compare it to the expected text from the prompt.\n"
        "- Compute character-level accuracy:\n"
        "  - Align expected vs recognized text approximately.\n"
        "  - Count a character as correct if it matches exactly (case-sensitive unless prompt implies otherwise).\n"
        "  - Minor spacing/punctuation differences should still be counted as incorrect characters.\n"
        "  - If expected text is empty, score must be 0.0.\n"
        "- Clamp the final score into [0.0, 1.0].\n"
        "\n"
        "## Output format (STRICT)\n"
        "You MUST output ONLY a single JSON object and nothing else.\n"
        "Do NOT include markdown. Do NOT include extra text.\n"
        "The JSON must have this schema:\n"
        "{\n"
        '  "expected_text": <string>,\n'
        '  "recognized_text": <string>,\n'
        '  "score": <number>\n'
        "}\n"
        "\n"
        "## Reliability requirements\n"
        "- If unsure, be conservative (lower score).\n"
    )


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")

_LAST_DEBUG: Optional[Dict[str, Any]] = None


def get_last_debug() -> Optional[Dict[str, Any]]:
    """Get last debug payload for OCR-vLLM call (per-process)."""
    return _LAST_DEBUG


def clear_last_debug() -> None:
    """Clear last debug payload."""
    global _LAST_DEBUG
    _LAST_DEBUG = None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from a model response."""
    if not text:
        return None
    text = text.strip()

    # Fast path: already JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Heuristic: find the first JSON object-like substring.
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    candidate = m.group(0)
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _coerce_score(obj: Dict[str, Any]) -> float:
    """Coerce obj['score'] into a float in [0, 1]."""
    score = obj.get("score", None)
    try:
        score_f = float(score)
    except Exception:
        score_f = 0.0
    if score_f != score_f:  # NaN
        score_f = 0.0
    return float(max(0.0, min(1.0, score_f)))


def ocr_vllm(images: List[Image.Image], prompts: List[str]) -> torch.Tensor:
    """
    OCR reward via vLLM (OpenAI-compatible chat completions).

    Args:
        images: list of PIL images
        prompts: list of prompts (same length as images)

    Returns:
        torch.Tensor of shape (len(images),) with values in [0, 1]
    """
    if len(images) != len(prompts):
        raise ValueError(f"ocr_vllm: len(images)={len(images)} must equal len(prompts)={len(prompts)}")

    api_url = os.environ.get("OCR_VLLM_URL", os.environ.get("UNIFIEDREWARD_THINK_URL", "http://localhost:18087"))
    model = os.environ.get("OCR_VLLM_MODEL", "OCR-VLM")
    max_tokens = int(os.environ.get("OCR_VLLM_MAX_TOKENS", "65536"))
    temperature = float(os.environ.get("OCR_VLLM_TEMPERATURE", "1.0"))

    # Optional: reuse the exclusive switching control-plane if user wants it.
    # This is best-effort and non-fatal inside maybe_switch_vllm_server.
    if _truthy_env("DEEPGEN_EXCLUSIVE_VLLM_SWITCH_ON_REWARD_CALLS", default="0"):
        maybe_switch_vllm_server("unifiedreward")

    system_prompt = _build_system_prompt()

    payload: List[Dict[str, Any]] = []
    for idx, (img, prompt) in enumerate(zip(images, prompts)):
        b64 = _encode_image_to_base64(img)
        problem = (
            "Task: Evaluate how accurately the image renders the expected text specified by the prompt.\n"
            "Steps (brief): infer expected_text from the prompt, read recognized_text from the image, then output JSON.\n"
            f"PROMPT:\n{prompt}\n"
        )
        payload.append(
            {
                "idx": idx,
                "images": [b64],
                "problem": problem,
                "system_prompt": system_prompt,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

    # Concurrency control (optional)
    workers = os.environ.get("OCR_VLLM_WORKERS")
    max_workers = int(workers) if workers else None

    responses = evaluate_batch(payload, api_url=api_url, max_workers=max_workers)

    scores: List[float] = []
    debug_records: List[Dict[str, Any]] = []
    for r in responses:
        idx = None
        if isinstance(r, dict):
            idx = r.get("idx")
        try:
            idx_int = int(idx) if idx is not None else None
        except Exception:
            idx_int = None

        out = (r or {}).get("model_output", "") if isinstance(r, dict) else ""
        obj = _extract_json_object(out)
        if obj is None:
            scores.append(0.0)
            if _debug_enabled():
                debug_records.append(
                    {
                        "idx": idx_int,
                        "prompt": prompts[idx_int] if idx_int is not None and 0 <= idx_int < len(prompts) else None,
                        "model_output": out,
                        "parsed_json": None,
                        "score": 0.0,
                    }
                )
            continue
        s = _coerce_score(obj)
        scores.append(s)
        if _debug_enabled():
            debug_records.append(
                {
                    "idx": idx_int,
                    "prompt": prompts[idx_int] if idx_int is not None and 0 <= idx_int < len(prompts) else None,
                    "model_output": out,
                    "parsed_json": obj,
                    "score": s,
                }
            )

    # Save last debug payload for trainer-side logging (global rank 0).
    if _debug_enabled():
        global _LAST_DEBUG
        _LAST_DEBUG = {
            "api_url": api_url,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "records": debug_records,
        }

    return torch.tensor(scores, dtype=torch.float32)

