"""Table-Transformer wrapper.

Lazy-loads ``microsoft/table-transformer-structure-recognition``. Tests
monkey-patch ``_get_detector`` to avoid loading the real model.
"""
from __future__ import annotations

from typing import Any

_DETECTOR_CACHE: dict[str, Any] = {"d": None}


def _get_detector():
    if _DETECTOR_CACHE["d"] is not None:
        return _DETECTOR_CACHE["d"]
    # Lazy load; actual impl uses transformers + PIL
    from transformers import (
        DetrImageProcessor,
        TableTransformerForObjectDetection,
    )
    from PIL import Image
    import io as _io

    proc = DetrImageProcessor.from_pretrained(
        "microsoft/table-transformer-structure-recognition"
    )
    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-structure-recognition"
    )

    def _detect(image_bytes: bytes):
        import torch

        img = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
        inputs = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]])
        results = proc.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.7
        )[0]
        return [
            {"bbox": box.tolist(), "score": float(score), "label": int(lbl)}
            for box, score, lbl in zip(
                results["boxes"], results["scores"], results["labels"]
            )
        ]

    _DETECTOR_CACHE["d"] = _detect
    return _detect


def detect_tables(page_image_bytes: bytes, page_num: int) -> list[dict]:
    """Detect table regions in a rendered PDF page image."""
    det = _get_detector()
    return det(page_image_bytes)
