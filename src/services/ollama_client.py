"""Centralized Ollama client with request queuing, retry, and timeout management.

Prevents concurrent requests from overwhelming the local Ollama instance.
Uses a semaphore to limit concurrency and retry for transient failures.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("PROCWISE_EXTRACTION_MODEL", "BeyondProcwise/AgentNick:extract")

# Max concurrent Ollama requests — match OLLAMA_NUM_PARALLEL (default 2)
_MAX_CONCURRENT = int(os.getenv("OLLAMA_MAX_CONCURRENT", "2"))
_semaphore = threading.Semaphore(_MAX_CONCURRENT)

# Retry and timeout — tuned for queued GPU inference
# With 2 parallel slots, requests queue in Ollama and may take longer
MAX_RETRIES = 3
RETRY_BASE_DELAY = 10  # seconds
RETRY_MAX_DELAY = 30  # seconds
DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))
SEMAPHORE_TIMEOUT = 600  # wait up to 10 min for a slot — Ollama queues internally


def ollama_generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float = 0,
    num_predict: int = 8192,
    num_gpu: int = 99,
    retries: int = MAX_RETRIES,
) -> Optional[str]:
    """Send a generation request to Ollama with queuing and retry.

    Returns the response text, or None on failure.
    """
    model = model or DEFAULT_MODEL
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_gpu": num_gpu,
        },
    }

    for attempt in range(1, retries + 1):
        acquired = _semaphore.acquire(timeout=SEMAPHORE_TIMEOUT)
        if not acquired:
            logger.warning(
                "Ollama semaphore wait exceeded %ds (attempt %d/%d) — proceeding anyway",
                SEMAPHORE_TIMEOUT, attempt, retries,
            )
            # Proceed without semaphore — better than blocking forever

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
            text = (body.get("response") or "").strip()
            if not text:
                # Thinking-capable models (qwen3:30b) may emit JSON inside the
                # `thinking` field when reasoning consumed all generation budget
                # before switching to response output. Salvage anything that
                # looks like JSON from there so extraction doesn't lose data.
                thinking = (body.get("thinking") or "").strip()
                if thinking:
                    import re as _re
                    match = _re.search(r"\{[\s\S]*\}", thinking)
                    if match:
                        text = match.group(0).strip()
            return text
        except requests.exceptions.ReadTimeout:
            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
            logger.warning(
                "Ollama read timeout (attempt %d/%d, model=%s, timeout=%ds) — "
                "retrying in %ds",
                attempt, retries, model, timeout, delay,
            )
            if attempt < retries:
                time.sleep(delay)
        except requests.exceptions.ConnectionError:
            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
            logger.warning(
                "Ollama connection error (attempt %d/%d) — retrying in %ds",
                attempt, retries, delay,
            )
            if attempt < retries:
                time.sleep(delay)
        except Exception as exc:
            logger.exception("Ollama request failed (attempt %d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(RETRY_BASE_DELAY)
        finally:
            if acquired:
                _semaphore.release()

    logger.error("Ollama request failed after %d attempts (model=%s)", retries, model)
    return None


def preload_model(model: Optional[str] = None, timeout: int = 120) -> bool:
    """Preload model into Ollama VRAM with keep_alive."""
    model = model or DEFAULT_MODEL
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": "24h"},
            timeout=timeout,
        )
        response.raise_for_status()
        logger.info("Preloaded Ollama model '%s' with 24h keep_alive", model)
        return True
    except Exception as exc:
        logger.warning("Ollama model preload failed (non-critical): %s", exc)
        return False
