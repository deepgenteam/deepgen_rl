# Copyright 2025 Ruihang Li and DeepGen Team @ Shanghai Innovation Institute

"""
Helpers for coordinating vLLM Sleep Mode across multiple OpenAI-compatible vLLM servers.

This module is used to avoid GPU OOM when two vLLM servers (e.g., UnifiedReward and
UniGenBench judge) are deployed on the same machine and share the same GPU set.

When DEEPGEN_EXCLUSIVE_VLLM_SERVERS is enabled and UNIFIEDREWARD_THINK_URL and
UNIGENBENCH_API_URL point to the same host, we will switch the active server by:
  - Putting the other server to sleep (level=1)
  - Waking up the target server

The HTTP endpoints are provided by vLLM Sleep Mode in development mode:
https://docs.vllm.ai/en/latest/features/sleep_mode/#online-serving
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional
from urllib.parse import urlparse

import requests


_LOCK = threading.Lock()
_LAST_TARGET: Optional[str] = None
_DISABLED = False
_WARNED_KEYS: set[str] = set()


def _truthy_env(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _debug_enabled() -> bool:
    return _truthy_env("DEEPGEN_DEBUG_EXCLUSIVE_VLLM", default="0")


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name, None)
    if val is None:
        return default
    try:
        return float(str(val).strip())
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name, None)
    if val is None:
        return default
    try:
        return int(str(val).strip())
    except Exception:
        return default


def _warn_once(key: str, msg: str) -> None:
    global _WARNED_KEYS
    if key in _WARNED_KEYS:
        return
    _WARNED_KEYS.add(key)
    print(msg)


def _is_local_rank0() -> bool:
    """
    Return True if this process is local rank 0 on the current machine.

    We use env vars to avoid importing torch/distributed here.
    If local-rank info is unavailable, default to True (single process).
    """
    for key in ("LOCAL_RANK", "SLURM_LOCALID", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        if key in os.environ:
            try:
                return int(str(os.environ.get(key, "0")).strip()) == 0
            except Exception:
                return True
    return True


def _normalize_base_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    url = str(url).strip()
    if not url:
        return None
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"
    return url.rstrip("/")


def _get_hostname(url: str) -> Optional[str]:
    parsed = urlparse(url)
    return parsed.hostname


def _same_host(url1: str, url2: str) -> bool:
    h1 = _get_hostname(url1)
    h2 = _get_hostname(url2)
    return bool(h1) and bool(h2) and h1 == h2


def is_exclusive_vllm_active() -> bool:
    """
    Return True if exclusive switching is enabled and the two vLLM servers share the same host.

    This function does not perform any network calls.
    """
    if _DISABLED:
        return False
    if not _truthy_env("DEEPGEN_EXCLUSIVE_VLLM_SERVERS", default="0"):
        return False
    unifiedreward_url = _normalize_base_url(os.environ.get("UNIFIEDREWARD_THINK_URL"))
    unigenbench_url = _normalize_base_url(os.environ.get("UNIGENBENCH_API_URL"))
    if not unifiedreward_url or not unigenbench_url:
        return False
    return _same_host(unifiedreward_url, unigenbench_url)


def _post(
    base_url: str,
    path: str,
    *,
    params: Optional[dict] = None,
    timeout_s: Optional[float] = None,
) -> requests.Response:
    full_url = f"{base_url}{path}"
    # NOTE: wake_up can be slow for large models or busy servers. We default to a longer
    # read timeout but keep connect timeout short. Users can override via env vars.
    req_timeout_s = timeout_s if timeout_s is not None else _env_float("DEEPGEN_EXCLUSIVE_VLLM_TIMEOUT_S", 60.0)
    connect_timeout_s = _env_float("DEEPGEN_EXCLUSIVE_VLLM_CONNECT_TIMEOUT_S", 3.0)
    retries = _env_int("DEEPGEN_EXCLUSIVE_VLLM_RETRIES", 3)
    backoff_s = _env_float("DEEPGEN_EXCLUSIVE_VLLM_RETRY_BACKOFF_S", 1.5)

    for attempt in range(max(1, retries)):
        try:
            return requests.post(full_url, params=params, timeout=(connect_timeout_s, req_timeout_s))
        except Exception as e:
            if attempt < max(1, retries) - 1:
                if _debug_enabled():
                    print(f"[exclusive-vllm] request failed (attempt={attempt + 1}/{retries}) url={full_url} err={e}")
                time.sleep(backoff_s * (2**attempt))
                continue
            raise


def _try_sleep_level1(base_url: str) -> bool:
    resp = _post(base_url, "/sleep", params={"level": 1})
    if resp.status_code == 404:
        raise RuntimeError("sleep endpoint not available (missing VLLM_SERVER_DEV_MODE=1 or --enable-sleep-mode)")
    resp.raise_for_status()
    return True


def _try_wake_up(base_url: str) -> bool:
    resp = _post(base_url, "/wake_up")
    if resp.status_code == 404:
        raise RuntimeError("wake_up endpoint not available (missing VLLM_SERVER_DEV_MODE=1 or --enable-sleep-mode)")
    resp.raise_for_status()
    return True


def maybe_switch_vllm_server(target: str) -> None:
    """
    Best-effort exclusive switch between UnifiedReward and UniGenBench vLLM servers.

    This is intentionally non-fatal: failures will disable further switching to
    avoid breaking training.

    Args:
        target: "unifiedreward" or "unigenbench".
    """
    global _LAST_TARGET, _DISABLED

    target = str(target).strip().lower()
    if target not in {"unifiedreward", "unigenbench"}:
        raise ValueError(f"Invalid target '{target}', expected 'unifiedreward' or 'unigenbench'")

    if _DISABLED:
        return

    if not _truthy_env("DEEPGEN_EXCLUSIVE_VLLM_SERVERS", default="0"):
        return

    # Only one process per machine should send sleep/wake control requests.
    # This avoids duplicated control-plane traffic from multiple local ranks.
    if not _is_local_rank0():
        return

    unifiedreward_url = _normalize_base_url(os.environ.get("UNIFIEDREWARD_THINK_URL"))
    unigenbench_url = _normalize_base_url(os.environ.get("UNIGENBENCH_API_URL"))
    if not unifiedreward_url or not unigenbench_url:
        return

    if not _same_host(unifiedreward_url, unigenbench_url):
        return

    with _LOCK:
        if _LAST_TARGET == target:
            return

        try:
            if target == "unifiedreward":
                # Sleep UniGenBench first to free GPU memory, then wake UnifiedReward.
                try:
                    _try_sleep_level1(unigenbench_url)
                except Exception as e:
                    if _debug_enabled():
                        print(f"[exclusive-vllm] sleep unigenbench failed: {e}")
                _try_wake_up(unifiedreward_url)
            else:
                # target == "unigenbench"
                try:
                    _try_sleep_level1(unifiedreward_url)
                except Exception as e:
                    if _debug_enabled():
                        print(f"[exclusive-vllm] sleep unifiedreward failed: {e}")
                _try_wake_up(unigenbench_url)

            _LAST_TARGET = target
            if _debug_enabled():
                print(
                    f"[exclusive-vllm] switched target='{target}' "
                    f"(unifiedreward={unifiedreward_url}, unigenbench={unigenbench_url})"
                )

        except Exception as e:
            # Disable switching after the first hard failure to avoid impacting training.
            _DISABLED = True
            _warn_once(
                "exclusive-vllm-disabled",
                f"[exclusive-vllm] disabling exclusive switching due to error: {e}",
            )


