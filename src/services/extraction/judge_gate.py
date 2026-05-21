"""L3 — grounded-last-resort judge gate for the renovation dispatch.

Fires after L1 (regex) and L2 (NER + tables) have run. For each REQUIRED
schema field where no candidate exists above its confidence threshold AND
the field declares ``judge.grounded_last_resort: true``, the judge re-reads
the parsed document (and optionally the page image via Qwen2.5-VL) and
returns a verbatim-substring value.

Safety contract is enforced by ``call_grounded_last_resort``: the returned
value MUST equal evidence_text AND be a literal substring of
``parsed.full_text``. The substring-grounding gate in ``dispatch.py``
re-validates this, so a misbehaving judge cannot leak hallucinations into
``_raw``.

Only the grounded judge is wired here — not the full
``run_judge_orchestrator``. The acute failure mode in production is
"required field has zero candidates"; tiebreaker (multiple disagreeing
candidates) and coherence are deferred until that base case is closed.
"""
from __future__ import annotations

import logging
from typing import Any

from src.services.extraction.pattern_registry import PatternRegistry
from src.services.extraction.types import Candidate, Span
from src.services.extraction_v3.judge.grounded_last_resort import (
    call_grounded_last_resort,
)

log = logging.getLogger(__name__)


def run_grounded_judge_for_gaps(
    *,
    parsed: Any,
    registry: PatternRegistry,
    existing_candidates: list[Candidate],
    file_path: str | None = None,
) -> list[Candidate]:
    """Call the grounded-last-resort judge for each required field that
    Layers 1+2 left unfilled (or filled below its threshold).

    Returns ONLY new candidates emitted by the judge — renovation
    ``Candidate`` shape. Caller is expected to extend its own candidate
    list with the return value; the substring-grounding gate downstream
    re-validates ``span.text in parsed.full_text``.
    """
    full_text = getattr(parsed, "full_text", "") or ""
    if not full_text.strip():
        return []

    # Best confidence per header field already present.
    best_conf: dict[str, float] = {}
    for c in existing_candidates:
        if c.field.startswith("line_items["):
            # line items are out of scope for the header-field judge gate
            continue
        prev = best_conf.get(c.field, -1.0)
        if c.confidence > prev:
            best_conf[c.field] = c.confidence

    new_candidates: list[Candidate] = []
    judge_calls = 0
    for f in registry.schema.fields:
        if not f.required:
            continue
        judge_spec = getattr(f, "judge", None)
        if judge_spec is None or not getattr(judge_spec, "grounded_last_resort", False):
            continue
        threshold = float(getattr(f, "confidence_threshold", 0.0) or 0.0)
        if best_conf.get(f.name, -1.0) >= threshold and best_conf.get(f.name, -1.0) >= 0:
            # Already covered by L1 or L2; the judge does NOT override
            continue
        try:
            v3_cand = call_grounded_last_resort(
                field=f, doc_full_text=full_text, file_path=file_path,
            )
        except Exception as exc:
            log.warning("grounded judge for field=%r raised %s", f.name, exc)
            continue
        judge_calls += 1
        if v3_cand is None:
            continue
        # v3 Candidate → renovation Candidate. evidence_text is guaranteed
        # to be a substring of full_text by call_grounded_last_resort's
        # safety checks; the dispatch grounding gate re-validates this.
        bbox = tuple(v3_cand.bbox) if v3_cand.bbox else (0.0, 0.0, 0.0, 0.0)
        new_candidates.append(Candidate(
            field=f.name,
            value=v3_cand.value,
            span=Span(
                page=int(v3_cand.page or 1),
                bbox=bbox,  # type: ignore[arg-type]
                text=v3_cand.evidence_text,
            ),
            source="judge",
            pattern_name="grounded_last_resort",
            confidence=float(v3_cand.confidence),
        ))

    if judge_calls:
        log.info(
            "judge_gate: doc_type=%s judge_calls=%d filled=%d",
            registry.doc_type, judge_calls, len(new_candidates),
        )
    return new_candidates
