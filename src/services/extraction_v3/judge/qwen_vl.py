"""Qwen2.5-VL-7B-Instruct inference module — lazy-loaded singleton.

Provides ``qwen_vl_extract(image, prompt, max_new_tokens) -> str`` for
document-understanding tasks. Used by the schema-coherence judge and the
grounded last-resort judge when EXTRACTION_V3_JUDGE_MODEL=qwen (default).

Memory budget: ~14 GB BF16. If CUDA OOM occurs, falls back to Q4 via
bitsandbytes. If OOM persists after retry, raises RuntimeError (loud failure
so the pipeline can demote to residual rather than silently skip).

Thread safety: double-checked lock around singleton initialisation.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)

_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
_LOCK = threading.Lock()

# Module-level singletons — None until first call.
_model = None
_processor = None
_load_error: Exception | None = None  # cached failure — don't retry on OOM


def _load_model_bf16():
    """Load model in bfloat16 on CUDA without Flash Attention 2."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    log.info("Loading %s in bfloat16 on CUDA (this takes ~60 s)…", _MODEL_ID)
    processor = AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",      # SDPA; Flash Attention 2 off by default
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def _load_model_q4():
    """Load model in Q4 via bitsandbytes as OOM fallback."""
    import torch
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    log.warning(
        "BF16 OOM — retrying %s with Q4 quantization (bitsandbytes)", _MODEL_ID
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


def _ensure_loaded() -> tuple:
    """Double-checked lock: load model exactly once.

    Returns (model, processor) or raises RuntimeError on OOM after retry.
    """
    global _model, _processor, _load_error

    if _model is not None:
        return _model, _processor  # fast path

    if _load_error is not None:
        raise RuntimeError(f"Qwen2.5-VL failed to load earlier: {_load_error}") from _load_error

    with _LOCK:
        # Re-check inside lock
        if _model is not None:
            return _model, _processor
        if _load_error is not None:
            raise RuntimeError(f"Qwen2.5-VL failed to load: {_load_error}") from _load_error

        try:
            _model, _processor = _load_model_bf16()
            log.info("Qwen2.5-VL loaded successfully in BF16.")
        except Exception as e:
            if "CUDA out of memory" in str(e) or "OutOfMemoryError" in str(e):
                log.warning("BF16 OOM: %s — trying Q4 quantization", e)
                try:
                    _model, _processor = _load_model_q4()
                    log.info("Qwen2.5-VL loaded in Q4 (OOM fallback).")
                except Exception as e2:
                    _load_error = e2
                    raise RuntimeError(
                        f"Qwen2.5-VL OOM even with Q4. Cannot load. Error: {e2}"
                    ) from e2
            else:
                _load_error = e
                raise RuntimeError(f"Qwen2.5-VL load failed: {e}") from e

    return _model, _processor


def qwen_vl_extract(
    image: "Image",
    prompt: str,
    max_new_tokens: int = 512,
) -> str:
    """Run Qwen2.5-VL-7B-Instruct on a document page image with a text prompt.

    Args:
        image: PIL.Image of the document page (RGB).
        prompt: Instruction text. The model sees both image and prompt.
        max_new_tokens: Token budget for the response.

    Returns:
        Decoded assistant response text (may be empty string on failure).

    Raises:
        RuntimeError: if the model failed to load (OOM / missing weights).
    """
    import torch

    model, processor = _ensure_loaded()

    # Build the chat-style messages expected by Qwen2.5-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        # Apply chat template to get the formatted text
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Process inputs — image + text together
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

        # Decode only the newly generated tokens (strip the prompt)
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        response = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response.strip()

    except torch.cuda.OutOfMemoryError as oom:
        log.error("Qwen2.5-VL CUDA OOM during inference: %s", oom)
        # Clear CUDA cache and return empty — the judge returns None when we return ""
        torch.cuda.empty_cache()
        return ""
    except Exception as exc:
        log.exception("Qwen2.5-VL inference error: %s", exc)
        return ""
