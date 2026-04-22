"""BERT-NER wrapper.

Lazy-loads ``dslim/bert-base-NER`` on first call. Tests monkey-patch
``_get_pipeline`` to avoid loading the 440MB model.
"""
from __future__ import annotations

from typing import Any

_PIPELINE_CACHE: dict[str, Any] = {"p": None}


def _get_pipeline():
    """Lazy-load BERT NER on CPU. Returns a callable str -> list[dict]."""
    if _PIPELINE_CACHE["p"] is not None:
        return _PIPELINE_CACHE["p"]
    from transformers import pipeline  # local import: heavy dep

    pipe = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        grouped_entities=True,
        device=-1,  # CPU
    )
    _PIPELINE_CACHE["p"] = pipe
    return pipe


def run(text: str) -> list[dict]:
    """Return a list of {entity_group, word, start, end, score} spans."""
    if not text or not text.strip():
        return []
    p = _get_pipeline()
    return list(p(text[:4096]))  # cap text length for BERT's 512-token limit
