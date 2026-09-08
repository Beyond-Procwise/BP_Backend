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

from src.services import egress

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("PROCWISE_EXTRACTION_MODEL", "BeyondProcwise/AgentNick:unified")

# Ollama Cloud (remote, authenticated) — used for non-critical tasks like
# summarization so the local GPU stays dedicated to AgentNick extraction.
OLLAMA_CLOUD_BASE_URL = os.getenv("OLLAMA_CLOUD_BASE_URL", "https://api.ollama.com")
OLLAMA_CLOUD_API_KEY = os.getenv("OLLAMA_CLOUD_API_KEY")

# ---------------------------------------------------------------------------
# Which OLLAMA_* variable is read by whom
#
# Ollama's *server* is a separate systemd unit and takes its environment from
# /etc/systemd/system/ollama.service.d/, NOT from this project's .env. Setting a
# server variable in .env or via os.environ.setdefault() here therefore does
# nothing at all, silently. That cost real debugging time: .env carried
# OLLAMA_FLASH_ATTENTION="1" while the server's own startup log reported
# `OLLAMA_FLASH_ATTENTION: false`, and OLLAMA_NUM_PARALLEL was setdefault() in
# eight modules with three different values (4, 8, and .env's 8) while the
# server ran with systemd's 2. Thirteen of those lines are now deleted; if you
# are about to add another, this is the note that says not to.
#
#   Server-side, set ONLY in the systemd unit:
#       OLLAMA_NUM_PARALLEL, OLLAMA_KEEP_ALIVE, OLLAMA_MAX_LOADED_MODELS,
#       OLLAMA_FLASH_ATTENTION, OLLAMA_KV_CACHE_TYPE, OLLAMA_GPU_OVERHEAD
#   Client-side, read here from .env and honoured:
#       OLLAMA_MAX_CONCURRENT, OLLAMA_KEEP_ALIVE (per-request; see below),
#       OLLAMA_TIMEOUT, OLLAMA_NUM_GPU_LAYERS, OLLAMA_CLOUD_*
#
# OLLAMA_KEEP_ALIVE is in both lists deliberately: we send it per request, and a
# per-request value overrides the server's. That is why `ollama ps` reports
# "Forever" even though the unit says 5m. The unit's value is dead as long as
# this client sets one.
# ---------------------------------------------------------------------------

# Max concurrent Ollama requests. This is the real client-side limit; keep it
# equal to the server's OLLAMA_NUM_PARALLEL (currently 2 in the systemd unit) so
# we do not queue more work than it will run at once.
_MAX_CONCURRENT = int(os.getenv("OLLAMA_MAX_CONCURRENT", "2"))
_semaphore = threading.Semaphore(_MAX_CONCURRENT)

# Retry and timeout — tuned for queued GPU inference
# With 2 parallel slots, requests queue in Ollama and may take longer
# Ollama keys a loaded model on its load-affecting options, so every distinct num_gpu
# value spawns its OWN runner — a fresh ~20GB load that blocks the caller for minutes at
# 0% GPU. Three different values were in play (Modelfile 25 via an option-less preload,
# 99 here, 999 in BaseAgent.ollama_options), so requests kept naming a configuration that
# was not loaded yet. Live, that timed out every summary and hung /workflows/ask.
# One value, used by every call site AND by the preload, means one instance.
# Any number at or above the model's layer count means "all of them"; llama.cpp clamps.
ALL_GPU_LAYERS = int(os.getenv("OLLAMA_NUM_GPU_LAYERS", "999"))

# A pin, not a preference. Ollama refuses outright when the layout will not fit
# — `500 {"error":"memory layout cannot be allocated with num_gpu = 999"}` — and
# on 2026-09-08 that took out every model call in the product for as long as the
# API held 3.8GB of a 23GB card. When the card refuses, the pin comes off and the
# server picks the split: ten times slower, and an answer.
#
# The refusal sticks for a window rather than being rediscovered per call.
# Ollama keys a loaded model on its load-affecting options, so alternating
# between pinned and unpinned would load a second copy of a 20GB model and block
# every caller while it did.
LAYOUT_RETRY_SECONDS = int(os.getenv("OLLAMA_LAYOUT_RETRY_SECONDS", "600"))
_LAYOUT_REJECTED_UNTIL = 0.0
_layout_lock = threading.Lock()


def is_layout_rejection(text: Any) -> bool:
    """True for the server's own words when it cannot place the layers."""
    body = str(text or "").lower()
    return "memory layout cannot be allocated" in body


def note_layout_rejection(detail: Any = "", now: Optional[float] = None) -> None:
    """Record that the card will not take the whole model just now."""
    global _LAYOUT_REJECTED_UNTIL
    with _layout_lock:
        was_pinned = _LAYOUT_REJECTED_UNTIL <= (now or time.time())
        _LAYOUT_REJECTED_UNTIL = (now or time.time()) + LAYOUT_RETRY_SECONDS
    if was_pinned:
        logger.warning(
            "Ollama refused the GPU layout (%s) — falling back to the server's own "
            "split for %ds. Generation will be several times slower until there is "
            "room for the whole model.", detail, LAYOUT_RETRY_SECONDS,
        )


def clear_layout_rejection() -> None:
    """Forget the refusal. For tests, and for a caller that knows memory freed."""
    global _LAYOUT_REJECTED_UNTIL
    with _layout_lock:
        _LAYOUT_REJECTED_UNTIL = 0.0


def gpu_options(now: Optional[float] = None) -> Dict[str, Any]:
    """``{"num_gpu": ...}``, or nothing at all while the card is refusing.

    Every path to this server reads it from here, so one refusal moves all of
    them at once and only one copy of the model is ever loaded.
    """
    with _layout_lock:
        rejected_until = _LAYOUT_REJECTED_UNTIL
    if (now or time.time()) < rejected_until:
        return {}
    return {"num_gpu": ALL_GPU_LAYERS}


MAX_RETRIES = 3
RETRY_BASE_DELAY = 10  # seconds
RETRY_MAX_DELAY = 30  # seconds
DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))
SEMAPHORE_TIMEOUT = 600  # wait up to 10 min for a slot — Ollama queues internally

# How long Ollama keeps the model resident in VRAM. The server default is short
# (5m), so between sparsely-arriving documents the extraction model gets evicted
# and every next document pays the full cold-load from disk (~60s on network
# storage). Pinning keep_alive on every request keeps the model resident — a
# pure latency win with byte-identical output. "-1" = never unload; a duration
# like "24h" also works. Overridable via OLLAMA_KEEP_ALIVE (.env already sets -1).
# Ollama accepts keep_alive as an int (seconds; -1 = never unload) OR a duration
# string ("24h"), but NOT a numeric string ("-1" → 400 Bad Request). Coerce a
# numeric env value to int so both "-1" and "24h" are valid on the wire.
def _coerce_keep_alive(v: str | int) -> str | int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


KEEP_ALIVE = _coerce_keep_alive(os.getenv("OLLAMA_KEEP_ALIVE", "-1"))


def ollama_generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float = 0,
    num_predict: int = 8192,
    num_gpu: Optional[int] = None,
    retries: int = MAX_RETRIES,
    stop: Optional[list] = None,
    keep_alive: str | int = KEEP_ALIVE,
    think: Optional[bool] = None,
    format: Optional[Any] = None,
) -> Optional[str]:
    """Send a generation request to Ollama with queuing and retry.

    Returns the response text, or None on failure.

    ``stop`` is forwarded as the Ollama ``stop`` option. Useful for
    extraction prompts where the fine-tuned model can drift into
    multi-turn output ("{\"user\":...{\"assistant\":..."); stopping at
    the first ``"\n}\n\n{"`` boundary or a ``{"user":`` literal cuts the
    runaway off after the first valid JSON object.

    ``think`` controls hybrid reasoning models (e.g. AgentNick:unified). When
    left as None it is not sent (the model's default applies — correct for the
    extraction specialist). Pass ``think=False`` for reasoning models so the
    answer lands in ``response`` instead of a separate ``thinking`` field that
    this function does not return — otherwise ``response`` comes back empty.
    """
    model = model or DEFAULT_MODEL
    options: Dict[str, Any] = {
        "temperature": temperature,
        "num_predict": num_predict,
    }
    # The caller may pin explicitly; otherwise the shared state decides, and a
    # card that has just refused the layout is not asked again.
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    else:
        options.update(gpu_options())
    if stop:
        options["stop"] = stop
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": options,
    }
    if think is not None:
        payload["think"] = think
    # Ollama structured outputs: a JSON schema (or the literal "json") passed as
    # ``format`` constrains generation to conforming output via grammar-guided
    # decoding — invalid tokens are masked, so the model cannot emit malformed
    # JSON or abbreviated placeholders.
    if format is not None:
        payload["format"] = format

    for attempt in range(1, retries + 1):
        acquired = _semaphore.acquire(timeout=SEMAPHORE_TIMEOUT)
        if not acquired:
            logger.warning(
                "Ollama semaphore wait exceeded %ds (attempt %d/%d) — aborting to "
                "protect GPU concurrency limit",
                SEMAPHORE_TIMEOUT, attempt, retries,
            )
            return None

        def _send(body: Dict[str, Any]) -> Any:
            return egress.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                purpose=egress.Purpose.MODEL_INFERENCE,
                json=body,
                timeout=timeout,
                # The model daemon is on localhost by design, so the
                # non-global address check does not apply here.
                require_global=False,
                # The retry loop below branches on ReadTimeout vs
                # ConnectionError with different backoffs; it needs the real
                # exception, not None.
                raise_transport_errors=True,
            )

        try:
            response = _send(payload)
            if response is None:
                return None
            # The card will not take the whole model. That is not a transient
            # failure to back off from — it is an answer, and the answer is
            # "ask for less". Retrying here rather than in the loop is
            # deliberate: a caller that asked for one attempt is asking for one
            # real attempt, not one spent discovering how full the card is.
            if getattr(response, "status_code", 0) == 500 and \
                    is_layout_rejection(getattr(response, "text", "")) and \
                    "num_gpu" in payload.get("options", {}):
                note_layout_rejection(payload["options"].get("num_gpu"))
                unpinned = {k: v for k, v in payload["options"].items() if k != "num_gpu"}
                payload = {**payload, "options": unpinned}
                response = _send(payload)
                if response is None:
                    return None
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
        except egress.ReadTimeout:
            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
            logger.warning(
                "Ollama read timeout (attempt %d/%d, model=%s, timeout=%ds) — "
                "retrying in %ds",
                attempt, retries, model, timeout, delay,
            )
            if attempt < retries:
                time.sleep(delay)
        except egress.ConnectionError:
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
    try:
        from services.llm_diagnostics import capture_llm_failure
        capture_llm_failure(
            site="ollama_generate.exhausted",
            prompt=prompt, raw_response="",
            model=model,
            extra={"retries_attempted": retries, "num_predict": num_predict},
        )
    except Exception:
        pass
    return None


def ollama_cloud_generate(
    prompt: str,
    *,
    model: str,
    timeout: int = 120,
    temperature: float = 0,
    num_predict: int = 1024,
    retries: int = 2,
    think: bool = False,
) -> Optional[str]:
    """Send a generation request to the Ollama Cloud API (remote, authenticated).

    Used for non-critical tasks (e.g. deal summarization) so the local GPU
    stays free for AgentNick extraction. Returns the response text, or None on
    failure. Requires OLLAMA_CLOUD_API_KEY in the environment. Unlike the local
    ``ollama_generate``, this does not use the local GPU semaphore — the call is
    remote.

    ``think`` defaults to False: the summary cloud models (e.g. qwen3.5:397b)
    are hybrid reasoning models that, when thinking is enabled, emit their
    answer in a separate ``thinking`` field and leave ``response`` empty — and
    this function deliberately returns only ``response`` (never the raw
    chain-of-thought). Disabling thinking makes the answer land in ``response``.
    """
    api_key = os.getenv("OLLAMA_CLOUD_API_KEY", OLLAMA_CLOUD_API_KEY)
    base = os.getenv("OLLAMA_CLOUD_BASE_URL", OLLAMA_CLOUD_BASE_URL).rstrip("/")
    if not api_key:
        logger.error("ollama_cloud_generate: OLLAMA_CLOUD_API_KEY not set")
        return None
    headers = {"Authorization": f"Bearer {api_key}"}
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }

    for attempt in range(1, retries + 1):
        try:
            response = egress.post(
                f"{base}/api/generate",
                purpose=egress.Purpose.MODEL_INFERENCE,
                json=payload,
                headers=headers,
                timeout=timeout,
                raise_transport_errors=True,
            )
            if response is None:
                return None
            response.raise_for_status()
            body = response.json()
            # Return only the user-facing response. Deliberately do NOT fall back
            # to a thinking/reasoning field — for summaries that would leak raw
            # chain-of-thought.
            text = (body.get("response") or "").strip()
            if text:
                return text
            # Empty response (an occasional model glitch even with think=False).
            # Treat it as transient and retry rather than surfacing an empty
            # summary; return "" only after retries are exhausted.
            logger.warning(
                "Ollama Cloud returned empty response (attempt %d/%d, model=%s)",
                attempt, retries, model,
            )
            if attempt < retries:
                time.sleep(min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY))
                continue
            return text
        except (egress.ReadTimeout, egress.ConnectionError) as exc:
            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
            logger.warning(
                "Ollama Cloud transient error (attempt %d/%d, model=%s): %s — retrying in %ds",
                attempt, retries, model, exc, delay,
            )
            if attempt < retries:
                time.sleep(delay)
        except Exception as exc:
            logger.exception("Ollama Cloud request failed (attempt %d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(RETRY_BASE_DELAY)

    logger.error("Ollama Cloud request failed after %d attempts (model=%s)", retries, model)
    return None


def preload_model(model: Optional[str] = None, timeout: int = 120) -> bool:
    """Preload model into Ollama VRAM with keep_alive."""
    model = model or DEFAULT_MODEL
    try:
        response = egress.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            purpose=egress.Purpose.MODEL_INFERENCE,
            require_global=False,
            # The layer count MUST match what callers ask for, or this pins an instance
            # nothing else can use and the first real request stalls loading another.
            json={
                "model": model,
                "prompt": "",
                "keep_alive": KEEP_ALIVE,
                "options": gpu_options(),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        logger.info("Preloaded Ollama model '%s' with keep_alive=%s", model, KEEP_ALIVE)
        return True
    except Exception as exc:
        logger.warning("Ollama model preload failed (non-critical): %s", exc)
        return False
