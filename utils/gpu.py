"""GPU configuration utilities for the ProcWise agentic framework.

This module centralises GPU-related environment setup so that all agents
and services can rely on a single, consistent configuration.  The
``configure_gpu`` function is idempotent – it will apply settings only
once and return the detected device (``"cuda"`` or ``"cpu"``).

The module also exposes :func:`load_cross_encoder`, which initialises
``sentence_transformers`` cross encoders with two distinct fallbacks.  They
end in different places, and the difference is the point:

``meta`` tensor initialisation error
    A PyTorch-version quirk, not a capacity problem.  The model is built on
    CPU and then **moved to the requested GPU**, so GPU acceleration is
    retained.

out of memory
    The card is genuinely full — usually because another process legitimately
    owns it; on this host Ollama's runner holds 19.5 GiB of 23 GiB for the
    model every agent depends on.  The model is built on CPU and **stays
    there**, because moving it back is the OOM again.  Reranking is slower;
    slower is not broken.  Before this existed the OOM propagated out of
    ``RAGPipeline`` into the API lifespan's outer ``except Exception``, which
    nulls ``agent_nick``, the orchestrator, the agent registry and eleven other
    pieces of state and then serves requests anyway — so one optional reranker
    cost the whole system a degraded boot, announced only by a CRITICAL line.

Note that the CPU retry must suspend the process-global default device that
``configure_gpu`` installs; see :func:`_construct_on_cpu` for why passing
``device="cpu"`` alone is not enough.
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


#: What "the GPU is full" looks like. ``torch.OutOfMemoryError`` is a
#: ``RuntimeError`` subclass and older PyTorch raised the bare parent with the
#: reason only in the message, so both are recognised. Anything else propagates
#: -- a wrong model name must still be an error, not a silent CPU load.
_OUT_OF_MEMORY_ERRORS: tuple = tuple(
    err for err in (
        getattr(torch, "OutOfMemoryError", None) if torch is not None else None,
        getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if torch is not None else None,
    ) if isinstance(err, type)
) or (RuntimeError,)


def _brief(exc: BaseException) -> str:
    """First sentence of a torch OOM message; they run to a full paragraph."""
    return str(exc).split(".")[0][:120]


def _construct_on_cpu(cross_encoder_cls: Any, model_name: str):
    """Build the encoder on the CPU, with the global default device suspended.

    ``configure_gpu`` calls ``torch.set_default_device("cuda")`` process-wide,
    and that alone is enough to defeat a CPU fallback: transformers resolves a
    CUDA device map from the ambient default and warms its allocator there, so
    ``cross_encoder_cls(model_name, device="cpu")`` **still raises
    OutOfMemoryError**. Verified against BAAI/bge-reranker-large on a full card
    before this helper was written -- passing ``device="cpu"`` on its own is not
    a fallback, it is the same crash one argument later.

    ``torch.device`` as a context manager overrides that default for the
    duration of the construction, which is what makes the retry actually land
    on the CPU.
    """
    if torch is None:  # pragma: no cover - torch is optional at import time
        return cross_encoder_cls(model_name, device="cpu")
    with torch.device("cpu"):
        return cross_encoder_cls(model_name, device="cpu")


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
    except _OUT_OF_MEMORY_ERRORS as exc:
        # The card is full. Almost always because something else legitimately
        # owns it -- on this host Ollama's runner holds 19.5 GiB of 23 GiB for
        # the model every agent depends on, leaving ~1.1 GiB against this
        # reranker's 2.07 GiB.
        #
        # Before this branch existed the OOM propagated out of RAGPipeline into
        # lifespan's outer `except Exception`, which nulls agent_nick, the
        # orchestrator, the agent registry and eleven other pieces of state and
        # serves requests anyway. An optional reranker is not worth a degraded
        # API: reranking on CPU is slower, and slower is not broken.
        logger.warning(
            "Cross encoder %s did not fit on %s (%s); loading on CPU instead. "
            "Reranking will be slower until the GPU has room.",
            model_name, target_device, _brief(exc),
        )
        encoder = _construct_on_cpu(cross_encoder_cls, model_name)
        # Cached under the *requested* device, deliberately: the next caller
        # asking for cuda gets this CPU encoder rather than paying the OOM
        # again. A process restart is what re-tries the GPU.
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
        encoder = cross_encoder_cls(model_name, device="cpu")
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
