"""Semantic-anchor extractor: maps label-shaped tokens to schema fields by
embedding-cosine similarity, then emits a candidate whose value is the next
non-label token in proximity to the label.

This catches synonymy that canonical_labels misses — e.g. 'Sold By:' →
supplier_name; 'Account Number:' → buyer_id; 'Amt Due' → invoice_total_incl_tax.

Substring guarantee preserved: candidate.evidence_text comes directly from a
parsed Token, so it's a substring of parsed.full_text.
"""
from __future__ import annotations
import threading
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from src.services.extraction_v3.extractors.base import Extractor
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument, Token
from src.services.extraction_v3.yaml_schema.loader import DocSchema, FieldSpec
from src.services.extraction_v3.yaml_schema.registry import register_extractor

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MIN_COSINE = 0.55  # below this, semantic match is too weak
RIGHT_OF_LABEL_MAX_DX = 250
BELOW_LABEL_MAX_DY = 50

_model = None
_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            _model = SentenceTransformer(MODEL_NAME, device="cuda")
    return _model


def _is_label_shaped(text: str) -> bool:
    """Heuristic for "this token looks like a label, not a value"."""
    s = text.strip()
    if len(s) < 2 or len(s) > 60:
        return False
    if s.endswith(":") or s.endswith(": "):
        return True
    # all-caps multi-word with no digits
    words = s.replace(":", "").split()
    if len(words) >= 1 and all(w.isupper() and w.isalpha() for w in words) and len(s) >= 3:
        return True
    return False


def _next_value_token(tokens: list[Token], label_tok: Token) -> Token | None:
    lx0, ly0, lx1, ly1 = label_tok.bbox
    cands = []
    for t in tokens:
        if t is label_tok:
            continue
        if _is_label_shaped(t.text):
            continue  # don't pick another label as the value
        tx0, ty0, tx1, ty1 = t.bbox
        same_line = abs(ty0 - ly0) < (ly1 - ly0) * 1.2
        dx = tx0 - lx1
        if same_line and 0 < dx < RIGHT_OF_LABEL_MAX_DX:
            cands.append((dx, t))
        else:
            dy = ty0 - ly1
            if 0 < dy < BELOW_LABEL_MAX_DY and abs(tx0 - lx0) < 100:
                cands.append((dy + 1000, t))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    return cands[0][1]


@register_extractor("sbert_anchor")
class SbertAnchorExtractor(Extractor):

    def produce_candidates(self, parsed: ParsedDocument, schema: DocSchema) -> list[Candidate]:
        # Identify which fields opted in to sbert_anchor
        active_fields = [f for f in schema.fields if "sbert_anchor" in f.extractors]
        if not active_fields:
            return []

        # Pre-compute canonical-label embeddings (mean over each field's labels)
        model = _get_model()
        field_to_emb: dict[str, np.ndarray] = {}
        for f in active_fields:
            if not f.canonical_labels:
                continue
            label_embs = model.encode(f.canonical_labels, normalize_embeddings=True)
            field_to_emb[f.name] = np.mean(label_embs, axis=0)

        # Find label-shaped tokens
        all_label_tokens = []
        for page in parsed.pages:
            for t in page.tokens:
                if _is_label_shaped(t.text):
                    all_label_tokens.append(t)
        if not all_label_tokens or not field_to_emb:
            return []

        # Embed label tokens in one batch
        label_texts = [t.text.rstrip(":").strip() for t in all_label_tokens]
        label_embs = model.encode(label_texts, normalize_embeddings=True)

        candidates = []
        for label_tok, label_emb in zip(all_label_tokens, label_embs):
            best_field, best_score = None, 0.0
            for fname, fanc in field_to_emb.items():
                score = float(np.dot(label_emb, fanc))
                if score > best_score:
                    best_score, best_field = score, fname
            if best_score < MIN_COSINE or best_field is None:
                continue
            value_tok = _next_value_token(parsed.pages[label_tok.page].tokens, label_tok)
            if value_tok is None:
                continue
            value = value_tok.text.strip()
            if not value or value not in parsed.full_text:
                continue
            candidates.append(Candidate(
                field=best_field,
                value=value,
                page=value_tok.page,
                bbox=value_tok.bbox,
                evidence_text=value,
                model="sbert_anchor",
                confidence=min(0.9, best_score),
            ))
        return candidates
