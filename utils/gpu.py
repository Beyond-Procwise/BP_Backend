"""GPU configuration utilities for the ProcWise agentic framework.

This module centralises GPU-related environment setup so that all agents
and services can rely on a single, consistent configuration.  The
``configure_gpu`` function is idempotent – it will apply settings only
once and return the detected device (``"cuda"`` or ``"cpu"``).

The module also exposes :func:`load_cross_encoder` which initialises
``sentence_transformers`` cross encoders with a graceful fallback for the
``meta`` tensor initialisation error introduced in newer versions of
PyTorch.  When this happens the model is first constructed on CPU and
then moved to the requested GPU, ensuring that GPU acceleration remains
available without crashing the agent.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Any

# Force HuggingFace libraries to use local cache only - no HTTP calls.
# Must be set before any HF import (sentence_transformers, transformers).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

try:  # ``torch`` is optional at import time for some environments
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

_CONFIGURED: bool = False
_DEVICE: Optional[str] = None
_CROSS_ENCODER_CACHE: dict[tuple[str, str], Any] = {}

logger = logging.getLogger(__name__)

# CUDA OOM surfaces as ``torch.OutOfMemoryError`` on recent PyTorch and as a
# plain ``RuntimeError`` ("CUDA out of memory") on older releases. Catch both.
if torch is not None and hasattr(torch, "OutOfMemoryError"):
    _CUDA_OOM_ERRORS: tuple[type[BaseException], ...] = (torch.OutOfMemoryError, RuntimeError)
else:  # pragma: no cover - torch missing or pre-2.x
    _CUDA_OOM_ERRORS = (RuntimeError,)


def _is_cuda_oom(exc: BaseException) -> bool:
    """True when ``exc`` represents a CUDA out-of-memory condition."""
    if torch is not None and isinstance(exc, getattr(torch, "OutOfMemoryError", ())):
        return True
    return "out of memory" in str(exc).lower()


def _construct_on_cpu(cross_encoder_cls: Any, model_name: str):
    """Construct a cross encoder strictly on CPU.

    ``configure_gpu`` sets the global default device to ``cuda`` via
    ``torch.set_default_device``. Passing ``device="cpu"`` alone is not
    enough: transformers' warmup still allocates scratch tensors on the
    default device and re-triggers a CUDA OOM. We therefore pin the default
    device to CPU for the duration of construction (auto-restored on exit)
    and release any cached CUDA blocks left over from the failed GPU attempt.
    """
    if torch is None:
        return cross_encoder_cls(model_name, device="cpu")
    try:
        torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - no CUDA / driver hiccup
        pass
    with torch.device("cpu"):
        return cross_encoder_cls(model_name, device="cpu")


def configure_gpu() -> str:
    """Configure GPU environment variables and default device.

    Returns
    -------
    str
        The device string (``"cuda"`` or ``"cpu"``) that should be used by
        downstream libraries.
    """
    global _CONFIGURED, _DEVICE
    if _CONFIGURED:
        return _DEVICE or "cpu"

    # Ensure GPU visibility and enablement for libraries that honour these
    # environment variables. Defaults are chosen to utilise the first GPU.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("OLLAMA_USE_GPU", "1")
    os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
    os.environ.setdefault("OMP_NUM_THREADS", "8")

    if torch is not None and torch.cuda.is_available():  # pragma: no cover - hardware dependent
        torch.set_default_device("cuda")
        _DEVICE = "cuda"
    else:
        _DEVICE = "cpu"

    # Many libraries such as ``sentence_transformers`` respect this variable.
    os.environ.setdefault("SENTENCE_TRANSFORMERS_DEFAULT_DEVICE", _DEVICE)
    os.environ.setdefault("PROCWISE_DEVICE", _DEVICE)

    _CONFIGURED = True
    return _DEVICE


def load_cross_encoder(
    model_name: str,
    cross_encoder_cls: Any,
    device: Any | None,
):
    """Initialise a cross encoder on the desired device with GPU fallback.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier.
    cross_encoder_cls:
        The class (typically ``sentence_transformers.CrossEncoder``) used to
        construct the reranker.
    device:
        Preferred device descriptor supplied by the agent.  It can be a
        string (``"cuda"``) or :class:`torch.device` instance.

    Returns
    -------
    Any
        An initialised cross encoder instance.

    Notes
    -----
    Recent PyTorch releases raise ``NotImplementedError`` when moving a
    module containing ``meta`` tensors directly to CUDA.  Some Hugging
    Face models trigger this pathway even though the system has a GPU
    available.  When this happens we retry the initialisation on CPU and
    then move the fully materialised model to the requested device.
    """

    target_device = None if device is None else str(device)
    cache_key = (model_name, (target_device or "cpu"))
    cached = _CROSS_ENCODER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        encoder = cross_encoder_cls(model_name, device=target_device)
        _CROSS_ENCODER_CACHE[cache_key] = encoder
        return encoder
    except _CUDA_OOM_ERRORS as exc:  # pragma: no cover - hardware dependent
        if target_device in (None, "cpu") or not _is_cuda_oom(exc):
            raise
        logger.warning(
            "Cross encoder '%s' could not be loaded on %s due to CUDA OOM "
            "(GPU likely held by Ollama models); falling back to CPU. Error: %s",
            model_name,
            target_device,
            exc,
        )
        encoder = _construct_on_cpu(cross_encoder_cls, model_name)
        # Cache under the CPU key so subsequent calls reuse the CPU reranker
        # instead of re-attempting the GPU load and OOM-ing again.
        _CROSS_ENCODER_CACHE[(model_name, "cpu")] = encoder
        _CROSS_ENCODER_CACHE[cache_key] = encoder
        return encoder
    except NotImplementedError as exc:  # pragma: no cover - hardware dependent
        if "meta tensor" not in str(exc):
            raise
        logger.warning(
            "Cross encoder initialisation failed on device %s due to meta tensor copy; "
            "retrying via CPU fallback.",
            target_device,
        )
        encoder = _construct_on_cpu(cross_encoder_cls, model_name)
        if target_device and target_device != "cpu":
            try:
                encoder.to(target_device)
            except NotImplementedError:
                logger.exception(
                    "Cross encoder could not be moved to device %s after CPU initialisation; "
                    "continuing on CPU.",
                    target_device,
                )
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "Unexpected error moving cross encoder to device %s; continuing on CPU.",
                    target_device,
                )
        _CROSS_ENCODER_CACHE[cache_key] = encoder
        return encoder
