"""Qwen2.5-VL-7B-Instruct inference module — dedicated GPU thread.

Provides ``qwen_vl_extract(image, prompt, max_new_tokens) -> str`` for
document-understanding tasks. Used by the schema-coherence judge and the
grounded last-resort judge when EXTRACTION_V3_JUDGE_MODEL=qwen (default).

Thread safety: ALL CUDA operations run on a single dedicated daemon thread
(_GPU_THREAD) that owns the CUDA context. Callers from any thread submit
work via a queue and block until the result is ready. This prevents
CUDNN_STATUS_NOT_INITIALIZED errors that occur when CUDA is used from
multiple threads simultaneously.

Memory budget: Q4 quantization first (~4-5 GiB), falls back to BF16 (~14 GiB).
On the 22 GiB A10G, Q4 leaves ample headroom for KV cache during generation.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)

_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# --------------------------------------------------------------------------
# Dedicated GPU thread — owns the CUDA context exclusively.
# --------------------------------------------------------------------------
_GPU_QUEUE: queue.Queue = queue.Queue()
_GPU_THREAD: threading.Thread | None = None
_GPU_THREAD_LOCK = threading.Lock()


def _gpu_worker():
    """Worker that runs on the dedicated GPU thread. Processes inference tasks."""
    import torch

    log.info("Qwen GPU worker thread started — initializing CUDA context.")
    # Initialize CUDA context on THIS thread, which will own it permanently.
    torch.cuda.init()

    model = None
    processor = None
    load_error = None

    while True:
        try:
            task = _GPU_QUEUE.get(timeout=30)
        except queue.Empty:
            continue

        if task is None:
            # Shutdown signal
            log.info("Qwen GPU worker thread shutting down.")
            break

        image, prompt, max_new_tokens, result_event, result_holder = task

        if load_error is not None:
            result_holder.append(("error", load_error))
            result_event.set()
            continue

        # Load model on first use
        if model is None:
            try:
                model, processor = _load_model_q4_internal()
                log.info("Qwen2.5-VL loaded in Q4 on GPU worker thread.")
            except Exception as e1:
                log.warning("Q4 load failed: %s — trying BF16", e1)
                try:
                    model, processor = _load_model_bf16_internal()
                    log.info("Qwen2.5-VL loaded in BF16 on GPU worker thread.")
                except Exception as e2:
                    load_error = e2
                    log.error("Qwen2.5-VL failed to load: %s", e2)
                    result_holder.append(("error", e2))
                    result_event.set()
                    continue

        # Run inference
        try:
            import torch
            torch.cuda.empty_cache()

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            input_len = inputs["input_ids"].shape[1]
            generated = output_ids[:, input_len:]
            response = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            result_holder.append(("ok", response.strip()))

        except torch.cuda.OutOfMemoryError as oom:
            log.error("Qwen2.5-VL CUDA OOM during inference: %s", oom)
            torch.cuda.empty_cache()
            result_holder.append(("ok", ""))

        except Exception as exc:
            log.exception("Qwen2.5-VL inference error: %s", exc)
            result_holder.append(("ok", ""))

        finally:
            result_event.set()


def _load_model_q4_internal():
    """Load model in Q4 quantization (bitsandbytes NF4, ~4-5 GiB)."""
    import torch
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    processor = AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        _MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def _load_model_bf16_internal():
    """Load model in bfloat16 (SDPA attention, ~14 GiB)."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    log.info("Loading %s in bfloat16 on CUDA (this takes ~60 s)…", _MODEL_ID)
    processor = AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def _ensure_gpu_thread():
    """Start the GPU worker thread if not already running."""
    global _GPU_THREAD
    if _GPU_THREAD is not None and _GPU_THREAD.is_alive():
        return
    with _GPU_THREAD_LOCK:
        if _GPU_THREAD is not None and _GPU_THREAD.is_alive():
            return
        t = threading.Thread(
            target=_gpu_worker,
            name="qwen-vl-gpu-worker",
            daemon=True,
        )
        t.start()
        _GPU_THREAD = t
        log.info("Qwen GPU worker thread launched.")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def qwen_vl_extract(
    image: "Image",
    prompt: str,
    max_new_tokens: int = 512,
) -> str:
    """Run Qwen2.5-VL-7B-Instruct on a document page image with a text prompt.

    Thread-safe: routes inference through a dedicated GPU thread to avoid
    CUDA cuDNN context conflicts when called from multiple worker threads.

    Args:
        image: PIL.Image of the document page (RGB).
        prompt: Instruction text. The model sees both image and prompt.
        max_new_tokens: Token budget for the response.

    Returns:
        Decoded assistant response text (may be empty string on failure).
    """
    _ensure_gpu_thread()

    result_holder: list = []
    result_event = threading.Event()

    _GPU_QUEUE.put((image, prompt, max_new_tokens, result_event, result_holder))

    # Wait up to 3 minutes for inference to complete
    completed = result_event.wait(timeout=180.0)
    if not completed:
        log.error("Qwen2.5-VL inference timed out after 180s")
        return ""

    if not result_holder:
        log.error("Qwen2.5-VL: no result in holder after event set")
        return ""

    status, value = result_holder[0]
    if status == "error":
        log.error("Qwen2.5-VL GPU worker error: %s", value)
        return ""
    return value or ""
