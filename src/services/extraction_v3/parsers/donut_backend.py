"""Donut adapter: image → structured text. Fallback for hard scans where
PaddleOCR's PP-Structure returns low confidence.

Donut (Document Understanding Transformer) is OCR-free: it consumes a page
image directly and decodes structured text. Pre-trained `naver-clova-ix/donut-base`
emits Hugging Face-style task-token sequences; we treat the decoder output
as the page's full_text and produce a single Token per non-empty line
(Donut base does not surface fine-grained per-token bboxes from the base
checkpoint).

Loading strategy: the HF cache contains two snapshots for donut-base —
  a959…  — full processor/config files but only pytorch_model.bin (blocked
             by CVE-2025-32434 on torch < 2.6)
  f8ee…  — model.safetensors only (no config)

We load config+processor from a959 and weights from the safetensors blob in
f8ee via safetensors.torch.load_file, then fix the tied lm_head weight.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Literal
import logging
import numpy as np

from src.services.extraction_v3.schemas.parsed_document import (
    ParsedDocument, Page, Token,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HF cache paths — resolved once at module level
# ---------------------------------------------------------------------------
_HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
_DONUT_CACHE = _HF_CACHE / "models--naver-clova-ix--donut-base" / "snapshots"

# Snapshot that holds the full processor + config tree
_SNAP_CONFIG = "a959cf33c20e09215873e338299c900f57047c61"
# Safetensors blob (lives under blobs/, referenced by the f8ee snapshot)
_SAFETENSORS_BLOB = (
    _HF_CACHE
    / "models--naver-clova-ix--donut-base"
    / "blobs"
    / "a489f5f1204286191cd38cfebacaeb876a5f9da99da6a1b4a633bf65508c39b9"
)

_proc = None
_model = None
_lock = threading.Lock()


def _get_donut():
    """Lazy singleton loader — loads model once, keeps it on GPU."""
    global _proc, _model
    if _proc is not None and _model is not None:
        return _proc, _model

    with _lock:
        # Double-checked inside lock
        if _proc is not None and _model is not None:
            return _proc, _model

        import torch
        assert torch.cuda.is_available(), "Donut requires GPU (spec constraint C2)"

        from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel, DonutProcessor
        from safetensors.torch import load_file

        snap_config_path = str(_DONUT_CACHE / _SNAP_CONFIG)
        log.info("Loading Donut processor from %s", snap_config_path)
        proc = DonutProcessor.from_pretrained(snap_config_path)

        log.info("Loading Donut model config from %s", snap_config_path)
        config = VisionEncoderDecoderConfig.from_pretrained(snap_config_path)
        model = VisionEncoderDecoderModel(config)

        log.info("Loading Donut weights from safetensors blob %s", _SAFETENSORS_BLOB)
        state_dict = load_file(str(_SAFETENSORS_BLOB), device="cpu")
        # strict=False: decoder.lm_head.weight is tied to embed_tokens and absent
        # from the safetensors file — this is expected and correct for Donut.
        model.load_state_dict(state_dict, strict=False)

        # Tie lm_head weights explicitly so generation logits are correct
        model.decoder.lm_head.weight = model.decoder.model.decoder.embed_tokens.weight

        model.to("cuda").eval()
        log.info(
            "Donut loaded — VRAM %.2f GB",
            torch.cuda.memory_allocated() / 1e9,
        )

        _proc = proc
        _model = model

    return _proc, _model


def _rasterize(path: Path, dpi: int = 200):
    """Convert a scanned PDF to a list of PIL Images."""
    from pdf2image import convert_from_path
    return convert_from_path(str(path), dpi=dpi)


def parse_with_donut(
    path: Path | str,
    file_format: Literal["pdf-scanned", "image"],
) -> ParsedDocument:
    """Parse a scanned PDF or image file with the Donut model.

    Returns a ParsedDocument whose tokens are derived by splitting the
    decoder output on newlines — the substring guarantee holds because
    full_text is the join of the same parts.

    Args:
        path: Path to the scanned PDF or image file.
        file_format: ``"pdf-scanned"`` or ``"image"``.

    Returns:
        ParsedDocument with parser_backend == "donut".
    """
    p = Path(path)
    proc, model = _get_donut()

    if file_format == "pdf-scanned":
        page_images = _rasterize(p)
    else:
        from PIL import Image
        page_images = [Image.open(p).convert("RGB")]

    import torch

    pages: list[Page] = []
    full_text_parts: list[str] = []
    confidences: list[float] = []

    for pg_idx, img in enumerate(page_images):
        # Donut requires the image to be pixel_values tensors
        pixel_values = proc(img, return_tensors="pt").pixel_values.to("cuda")

        # Use the synthdog task prompt — works with base checkpoint without fine-tuning
        task_prompt = "<s_synthdog>"
        decoder_input_ids = (
            proc.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")
            .input_ids.to("cuda")
        )

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=model.decoder.config.max_position_embeddings,
                pad_token_id=proc.tokenizer.pad_token_id,
                eos_token_id=proc.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[proc.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                output_scores=True,
            )

        seq = proc.batch_decode(outputs.sequences)[0]
        # Strip Donut's special task tokens
        decoded = (
            seq.replace(proc.tokenizer.eos_token, "")
            .replace(proc.tokenizer.pad_token, "")
            .replace(task_prompt, "")
            .strip()
        )

        full_text_parts.append(decoded)

        # Confidence: exp(mean log-prob) over generated token steps → [0, 1]
        if outputs.scores:
            avg_logprob = float(
                torch.stack(
                    [s.softmax(-1).max().log() for s in outputs.scores]
                ).mean()
            )
            confidences.append(float(np.exp(avg_logprob)))

        # Build one Token per non-empty line.
        # Substring guarantee: each tok.text is a line from `decoded`,
        # and full_text = "\n".join(full_text_parts) contains `decoded`,
        # so tok.text ⊆ decoded ⊆ full_text by construction.
        w, h = (
            (float(img.size[0]), float(img.size[1]))
            if hasattr(img, "size")
            else (float(img.shape[1]), float(img.shape[0]))
        )
        page_tokens: list[Token] = [
            Token(
                text=line,
                page=pg_idx,
                # Donut base doesn't expose token-level bboxes; use page bbox
                bbox=(0.0, 0.0, w, h),
            )
            for line in decoded.split("\n")
            if line.strip()
        ]

        pages.append(
            Page(
                index=pg_idx,
                width=w,
                height=h,
                rotation=0,
                regions=[],
                tables=[],
                tokens=page_tokens,
            )
        )

    full_text = "\n".join(full_text_parts)
    overall_conf = (
        float(sum(confidences) / len(confidences)) if confidences else 0.5
    )

    return ParsedDocument(
        source_path=str(p),
        file_format=file_format,
        pages=pages,
        full_text=full_text,
        parser_backend="donut",
        parser_confidence=overall_conf,
    )
