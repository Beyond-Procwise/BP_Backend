"""YOLO-X layout detection wrapper.

Lazy-loads a layout model (unstructuredio/yolo_x_layout). Tests must
monkey-patch ``_get_detector`` — the default raises NotImplementedError so
real loads only happen when configured at service startup.
"""
from __future__ import annotations

from typing import Any

_DETECTOR_CACHE: dict[str, Any] = {"d": None}


def _get_detector():
    if _DETECTOR_CACHE["d"] is not None:
        return _DETECTOR_CACHE["d"]
    # Lazy load unstructuredio/yolo_x_layout
    # Actual impl would instantiate the model; stub raises so tests MUST mock.
    raise NotImplementedError("Configure yolo_x_layout loader at service startup")


def detect_regions(page_image_bytes: bytes, page_num: int) -> list[dict]:
    det = _get_detector()
    return det(page_image_bytes)
