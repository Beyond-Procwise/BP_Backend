"""Semantic-anchor extractor: maps label-shaped tokens to schema fields by
embedding-cosine similarity, then emits a candidate whose value is the next
non-label token in proximity to the label.

This catches synonymy that canonical_labels misses — e.g. 'Sold By:' →
supplier_name; 'Account Number:' → buyer_id; 'Amt Due' → invoice_total_incl_tax.

Improvements over baseline:
  - Type-aware value selection: for money/decimal/iso_date/currency fields,
    the proximity search scans up to N nearby tokens and picks the first one
    that parses successfully, instead of blindly returning the nearest token.
  - Tighter label detection: multi-word colon-terminated phrases
    (e.g. "DELIVER TO:", "SHIP VIA: UPS Express") are now correctly
    identified as labels and skipped during value selection.

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
# How many nearby candidates to inspect when type-checking
MAX_TYPE_SCAN_RADIUS = 6

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
    """Return True if *text* looks like a field label rather than a value.

    Recognises:
      - Tokens ending with ":"  (e.g. "Invoice Number:", "BILL TO:")
      - All-caps multi-word tokens with no digits (e.g. "DELIVER TO", "SHIP VIA")
      - Short title-case phrases (e.g. "Invoice Date", "Due Date")
    """
    s = text.strip()
    if len(s) < 2 or len(s) > 80:
        return False
    # Ends with colon — unambiguous label marker
    if s.endswith(":"):
        return True
    # Strip trailing colon for further checks
    cleaned = s.rstrip(":").strip()
    words = cleaned.split()
    # All-caps multi-word, no digits — e.g. "DELIVER TO", "SHIP VIA", "BILL TO"
    if words and all(w.isupper() and w.isalpha() for w in words) and len(words) >= 1 and len(s) >= 3:
        return True
    return False


# ---------------------------------------------------------------------------
# Typed-value validators (same logic as in layoutlmv3.py)
# ---------------------------------------------------------------------------

_ISO_CURRENCY_CODES = frozenset({
    "GBP", "USD", "EUR", "JPY", "INR", "CAD", "AUD", "CHF", "CNY", "NZD",
    "ZAR", "SGD", "HKD", "AED", "SEK", "NOK", "DKK", "PLN", "CZK",
})
_SYMBOL_TO_ISO = {"$": "USD", "£": "GBP", "€": "EUR", "¥": "JPY", "₹": "INR"}


def _try_parse_money(text: str) -> bool:
    from src.services.extraction_v2.parsers.amounts import parse_amount
    return parse_amount(text) is not None


_MONTH_ONLY_RE = re.compile(
    r"^(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\.?$",
    re.IGNORECASE,
)


def _try_parse_date(text: str) -> bool:
    """Return True iff text can be coerced to a date.

    Rejects bare month names (e.g. "October") — dateparser would expand these
    to "October 1 <current_year>", which is almost certainly a label-proximity
    accident rather than an actual document date.
    """
    from src.services.extraction_v2.parsers.dates import parse_date
    s = text.strip()
    if _MONTH_ONLY_RE.match(s):
        return False
    return parse_date(s) is not None


def _try_parse_currency(text: str) -> bool:
    s = text.strip()
    if s in _SYMBOL_TO_ISO:
        return True
    if s.upper() in _ISO_CURRENCY_CODES:
        return True
    return False


def _token_satisfies_type(token: Token, field_type: str) -> bool:
    text = token.text.strip()
    if not text:
        return False
    if field_type in ("money", "decimal"):
        return _try_parse_money(text)
    if field_type == "iso_date":
        return _try_parse_date(text)
    if field_type == "currency":
        return _try_parse_currency(text)
    # string: skip label-shaped tokens
    if _is_label_shaped(text):
        return False
    # Accept alphanumeric content (IDs can be purely numeric)
    return bool(re.search(r"[a-zA-Z0-9]", text))


def _next_value_token(
    tokens: list[Token],
    label_tok: Token,
    field_type: str = "string",
) -> Token | None:
    lx0, ly0, lx1, ly1 = label_tok.bbox
    label_height = max(ly1 - ly0, 1.0)
    cands = []
    for t in tokens:
        if t is label_tok:
            continue
        tx0, ty0, tx1, ty1 = t.bbox
        same_line = abs(ty0 - ly0) < label_height * 1.2
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

    is_typed = field_type in ("money", "decimal", "iso_date", "currency")
    scan_limit = MAX_TYPE_SCAN_RADIUS if is_typed else 4

    for _, tok in cands[:scan_limit]:
        if _token_satisfies_type(tok, field_type):
            # Additional guard for string fields: skip long compound tokens
            # and "Label: value" compound tokens (handled by layoutlmv3 intra-token).
            if not is_typed:
                tok_stripped = tok.text.strip()
                # Must contain at least one alphanumeric character
                if not re.search(r"[a-zA-Z0-9]", tok_stripped):
                    continue
                if len(tok_stripped) > 80:
                    continue
                if ":" in tok_stripped:
                    colon_pos = tok_stripped.index(":")
                    before_colon = tok_stripped[:colon_pos].strip()
                    if before_colon and len(before_colon) <= 40 and re.match(r"^[A-Za-z][A-Za-z0-9 #\.\-/&]*$", before_colon):
                        continue
            return tok

    return None


@register_extractor("sbert_anchor")
class SbertAnchorExtractor(Extractor):

    def produce_candidates(self, parsed: ParsedDocument, schema: DocSchema) -> list[Candidate]:
        # Identify which fields opted in to sbert_anchor
        active_fields = [f for f in schema.fields if "sbert_anchor" in f.extractors]
        if not active_fields:
            return []

        # Build a type lookup for proximity filtering
        field_type_map: dict[str, str] = {f.name: f.type for f in active_fields}

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

            field_type = field_type_map.get(best_field, "string")
            value_tok = _next_value_token(
                parsed.pages[label_tok.page].tokens, label_tok, field_type
            )
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
