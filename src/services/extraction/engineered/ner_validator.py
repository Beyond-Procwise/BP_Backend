"""L2 — wraps SpacyNERExtractor as a gap-filler for NER-typed fields.

Only fires for fields that (a) declare ner_type_check != 'none' in the
schema AND (b) have no L1 candidate above threshold yet. Emits Candidate
records in the renovation `extraction.types.Candidate` shape.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

from src.services.extraction.types import Candidate, Span
from src.services.extraction_v3.extractors.spacy_ner import SpacyNERExtractor
from src.services.extraction_v3.yaml_schema.loader import DocSchema

log = logging.getLogger(__name__)


def fill_ner_gaps(
    *,
    parsed: Any,
    schema: DocSchema,
    existing_fields: Iterable[str],
) -> list[Candidate]:
    """For every schema field with ner_type_check != 'none' AND no existing
    candidate, run the existing SpacyNERExtractor and emit any candidates it
    finds. Returns a list of renovation Candidates.

    `existing_fields` is the set of field names already filled by L1 (or any
    earlier tier). We don't want NER to override a regex hit — it only fills
    holes.
    """
    existing = set(existing_fields)
    target_fields = {
        f.name for f in schema.fields
        if f.judge.ner_type_check not in (None, "none") and f.name not in existing
    }
    if not target_fields:
        return []

    # SpacyNERExtractor wants fields to have 'spacy_ner' in extractors. We
    # bypass that check by setting it on a shallow schema copy in memory.
    patched_fields = []
    for f in schema.fields:
        if f.name in target_fields:
            extractors = list(getattr(f, "extractors", []) or [])
            if "spacy_ner" not in extractors:
                extractors.append("spacy_ner")
            patched_fields.append(f.model_copy(update={"extractors": extractors}))
        else:
            patched_fields.append(f)
    patched_schema = schema.model_copy(update={"fields": patched_fields})

    try:
        v3_cands = SpacyNERExtractor().produce_candidates(parsed, patched_schema)
    except Exception as exc:
        log.warning("spacy_ner failed (treated as no candidates): %s", exc)
        return []

    out: list[Candidate] = []
    for vc in v3_cands:
        if vc.field not in target_fields:
            continue
        out.append(Candidate(
            field=vc.field,
            value=vc.value,
            span=Span(page=vc.page, bbox=tuple(vc.bbox), text=vc.evidence_text),
            source="ner",
            pattern_name=None,
            confidence=vc.confidence,
        ))
    return out
