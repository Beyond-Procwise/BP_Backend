"""spaCy NER candidate generator.

For each field whose YAML judge.ner_type_check is non-'none', this extractor
finds spaCy entities of the required type in parsed.full_text and emits each
as a candidate. Acts as both:
  1. a positive evidence source (fields like supplier_name benefit from
     "named-entity-shaped strings in this doc that look like organizations")
  2. an implicit negative signal — if the legacy paths picked up a value
     that's not an ORG, the L3 orchestrator can compare to spaCy's choices.

Substring guarantee: spaCy operates on parsed.full_text directly; entity
spans are substrings by construction (ent.text is doc.text[ent.start_char:ent.end_char]).
"""
from __future__ import annotations

import threading

import spacy

from src.services.extraction_v3.extractors.base import Extractor
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument
from src.services.extraction_v3.yaml_schema.loader import DocSchema
from src.services.extraction_v3.yaml_schema.registry import register_extractor

# Transformer-based model for best accuracy. Falls back to en_core_web_sm if
# the transformer model is not installed (e.g. CPU-only environments).
MODEL_NAME = "en_core_web_trf"

_nlp = None
_lock = threading.Lock()


def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    with _lock:
        if _nlp is None:
            try:
                _nlp = spacy.load(MODEL_NAME)
            except OSError:
                # Transformer model not installed — fall back to small CPU model.
                # In production, install en_core_web_trf for best accuracy.
                _nlp = spacy.load("en_core_web_sm")
    return _nlp


@register_extractor("spacy_ner")
class SpacyNERExtractor(Extractor):
    """Emit one Candidate per named entity that matches a field's ner_type_check.

    Only fields that list "spacy_ner" in their extractors list are considered.
    The spaCy model runs over parsed.full_text once and groups entities by label;
    each entity that matches a field's required NER label becomes one candidate.

    Confidence is fixed at 0.7; the L3 confidence-merger upgrades candidates
    that multiple extractors agree on and downgrades those that stand alone.
    """

    def produce_candidates(self, parsed: ParsedDocument, schema: DocSchema) -> list[Candidate]:
        # Collect only fields that opted into spacy_ner
        active = [f for f in schema.fields if "spacy_ner" in f.extractors]
        if not active:
            return []

        # Filter further to fields that actually have a NER type check
        active = [f for f in active if f.judge.ner_type_check != "none"]
        if not active:
            return []

        # Run NER over the full document text once
        nlp = _get_nlp()
        doc = nlp(parsed.full_text)

        # Group entities by spaCy label for O(1) lookup per field
        ents_by_label: dict[str, list] = {}
        for ent in doc.ents:
            ents_by_label.setdefault(ent.label_, []).append(ent)

        candidates: list[Candidate] = []
        for field in active:
            # YAML uses spaCy's literal label names (ORG, PERSON, GPE, LOC, …)
            spacy_label = field.judge.ner_type_check
            for ent in ents_by_label.get(spacy_label, []):
                ent_text = ent.text.strip()
                if not ent_text:
                    continue
                # Substring guarantee: entity text must appear in full_text.
                # spaCy guarantees this by construction (ent.text is a slice of
                # the input string), but the strip() could in theory break it
                # for purely-whitespace-surrounded entities. Guard defensively.
                if ent_text not in parsed.full_text:
                    continue

                bbox = _find_bbox_for_text(parsed, ent_text)
                page_idx, b = bbox if bbox else (0, (0.0, 0.0, 0.0, 0.0))

                candidates.append(Candidate(
                    field=field.name,
                    value=ent_text,
                    page=page_idx,
                    bbox=b,
                    evidence_text=ent_text,
                    model="spacy_ner",
                    # Base confidence; L3 merges with other models' agreement.
                    confidence=0.7,
                ))

        return candidates


def _find_bbox_for_text(parsed: ParsedDocument, text: str):
    """Return (page_index, bbox) of the first token that contains `text`.

    Searches case-insensitively so short entities like "Ltd" match even if
    the token was extracted with mixed case from the PDF.
    Returns None when no matching token is found (caller falls back to page 0,
    zero-bbox — the candidate is still valid, just without precise location).
    """
    text_lower = text.lower().strip()
    for page in parsed.pages:
        for tok in page.tokens:
            if text_lower in tok.text.lower():
                return (page.index, tok.bbox)
    return None
