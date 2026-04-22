import re
from src.services.structural_extractor.parsing.model import ParsedDocument, BBox
from src.services.structural_extractor.discovery.schema import FieldType, type_of, fields_for
from src.services.structural_extractor.discovery.type_entities import find_candidates
from src.services.structural_extractor.discovery.proximity import inferred_label
from src.services.structural_extractor.types import ExtractedValue


# Patterns that identify non-ID tokens (SWIFT / IBAN / postcode / currency)
# which should never be returned as a document ID.
SWIFT_RE = re.compile(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?$")
IBAN_PREFIX_RE = re.compile(r"^[A-Z]{2}\d{2}$")  # e.g., "GB29"
UK_POSTCODE_OUTCODE_RE = re.compile(r"^[A-Z]{1,2}\d[A-Z0-9]?$")  # e.g., "RH13", "M1"
UK_POSTCODE_INCODE_RE = re.compile(r"^\d[A-Z]{2}$")  # e.g., "5QH", "7HY"


# Field-key conflict table: when we're looking for field X, penalize candidates
# whose inferred label prominently mentions a different field's keyword.
CONFLICT_KEYWORDS: dict[str, set[str]] = {
    "invoice": {"po", "purchase", "swift", "iban", "bic", "sort", "account"},
    "po": {"invoice", "swift", "iban", "bic", "sort", "account"},
    "order": {"invoice", "swift", "iban", "bic"},
    "quote": {"invoice", "po", "purchase"},
}

# Synonym labels that count as a positive match even when the field_key word
# itself is absent. Doc-type-specific: an Invoice's "No." / "Number" labels
# almost always identify the invoice number.
FIELD_SYNONYMS: dict[tuple[str, str], set[str]] = {
    ("invoice", "Invoice"): {"no", "no.", "#", "number", "num"},
    ("po", "Purchase_Order"): {"no", "no.", "#", "number", "num"},
    ("quote", "Quote"): {"no", "no.", "#", "number", "num"},
}

# "Reference" in the label flips semantic: an invoice that "references PO X"
# — the ID attached is a PO, not an invoice. Treat as conflict for invoice.
REFERENCE_POISON = {"references", "reference", "ref", "ref:"}


def _is_rejectable_id_format(text: str) -> bool:
    """Return True for tokens that match recognized non-ID formats."""
    t = text.strip()
    if SWIFT_RE.match(t):
        return True
    if IBAN_PREFIX_RE.match(t):
        return True
    if UK_POSTCODE_OUTCODE_RE.match(t) and len(t) <= 4:
        return True
    if UK_POSTCODE_INCODE_RE.match(t):
        return True
    return False


def _label_words(label: str) -> list[str]:
    words = []
    for raw in label.lower().split():
        norm = raw.strip(".,:;()[]{}\"'-/")
        if norm:
            words.append(norm)
    return words


def _word_in_label(word: str, label_words: list[str]) -> bool:
    target = word.lower().strip()
    return target in label_words


def _nearby_label(cand, all_tokens) -> str:
    """Find a label near the candidate — same line OR one line above (left-aligned).
    Falls back to inferred_label if the candidate is not BBox-anchored.
    """
    if not cand.tokens:
        return ""
    first = cand.tokens[0]
    anchor = first.anchor
    if not isinstance(anchor, BBox):
        return inferred_label(cand, all_tokens)
    same_line = inferred_label(cand, all_tokens)
    # Also collect tokens immediately above (within 25pt vertically) AND horizontally overlapping
    above: list = []
    cand_cy = (anchor.y0 + anchor.y1) / 2
    for t in all_tokens:
        if not isinstance(t.anchor, BBox) or t.anchor.page != anchor.page:
            continue
        t_cy = (t.anchor.y0 + t.anchor.y1) / 2
        # Token center must be above candidate center, and within 30pt
        if t_cy >= cand_cy:
            continue
        if cand_cy - t_cy > 30:
            continue
        # Horizontal overlap with candidate
        if t.anchor.x1 < anchor.x0 - 10 or t.anchor.x0 > anchor.x1 + 80:
            continue
        above.append(t)
    above.sort(key=lambda t: (t.anchor.y0, t.anchor.x0))
    above_text = " ".join(t.text for t in above[-5:])
    return (same_line + " " + above_text).strip()


def _score_candidate(field_key: str, doc_type: str, cand, label: str) -> int:
    """Score a candidate for a given field_key. Higher = better.

    Positive signals:
      +10 per direct label word-match for a field keyword ("invoice", "po")
      + 7 for a field-synonym match ("no", "#", "number") when doc_type aligns
    Negative signals:
      -100 if candidate's OWN text matches a rejectable format (SWIFT/IBAN/postcode)
      - 15 per conflict-keyword match in label (e.g. "swift" label for po_id)
      -  5 if label contains a "references"/"ref" poison word AND the
          field_key is "invoice" (the phrase "this invoice references PO" is
          about the PO, not the invoice)
      - 50 if candidate text STARTS WITH a conflicting prefix ("PO" for
          invoice_id; "INV" for po_id)
    """
    label_words = _label_words(label)
    score = 0
    words = [w for w in field_key.split() if w]
    primary = words[0] if words else ""

    # "This invoice references: X" — the "invoice" word in this context
    # does NOT mean X is an invoice. Neutralize invoice/po keywords when
    # a reference-poison word is present.
    has_reference_poison = any(p in label_words for p in REFERENCE_POISON)

    for w in words:
        if _word_in_label(w, label_words):
            # If the field is "invoice" and "references" is in the label,
            # this is actually a PO-reference phrase, not an invoice label.
            if w == "invoice" and has_reference_poison:
                continue
            score += 10

    synonyms = FIELD_SYNONYMS.get((primary, doc_type), set())
    for syn in synonyms:
        if _word_in_label(syn, label_words):
            score += 7
            break

    # Self-label: the candidate's own text starts with the field keyword
    # (e.g. "PO438295" for po_id, "INV600254" for invoice_id). This is a
    # strong structural signal — many invoices omit an explicit label and
    # just print "PO12345" or "INV-0001".
    txt_upper = cand.text.upper()
    primary_upper = primary.upper()
    if primary_upper == "INVOICE":
        if txt_upper.startswith("INV") and any(c.isdigit() for c in txt_upper):
            score += 8
    elif primary_upper == "PO":
        if txt_upper.startswith("PO") and any(c.isdigit() for c in txt_upper):
            score += 8

    # Conflict penalty from label
    for conflict_word in CONFLICT_KEYWORDS.get(primary, set()):
        if _word_in_label(conflict_word, label_words):
            # Same neutralization: "invoice" doesn't count as a conflict
            # when the phrase is "this invoice references X".
            if conflict_word == "invoice" and has_reference_poison:
                continue
            score -= 15

    # Reference poison: "this invoice references PO X" — flips semantics
    if primary == "invoice":
        for poison in REFERENCE_POISON:
            if _word_in_label(poison, label_words):
                score -= 5
                break

    # Reject tokens that LOOK like SWIFT/IBAN/postcode entirely
    if _is_rejectable_id_format(cand.text):
        score -= 100

    # Cross-keyword bleed in the candidate text itself
    if primary == "invoice" and (txt_upper.startswith("PO") and any(c.isdigit() for c in txt_upper)):
        score -= 50
    if primary == "po" and txt_upper.startswith("INV"):
        score -= 50

    return score


# Filename-hint patterns: ID values that are commonly embedded in
# filenames but omitted from the document body. Each entry is
# (field_key, compiled regex, value-builder).
# The regex is matched against the filename (case-insensitive); the
# value-builder takes the match object and returns the normalized ID
# value (preserving any required prefix).
#
# Invoice IDs are deliberately NOT listed: the invoice body always
# carries its own invoice number, and filenames are often derivative
# (e.g. 'AQUARIUS INV-25-050 for PO508084 .pdf' has INV-25-050 as the
# subject but it's also in the body — body wins).
_FILENAME_HINT_PATTERNS: list[tuple[str, "re.Pattern", "callable"]] = [
    # PO number: 'PO12345', 'PO-12345', 'for PO12345' — preserve the PO prefix.
    ("po", re.compile(r"\bPO[- ]?(\d{4,})\b", re.IGNORECASE), lambda m: "PO" + m.group(1)),
    # Quote number: 'QUT-1234', 'Q1234', 'QUOTE-123'.
    ("quote", re.compile(r"\bQUT[- ]?(\d{3,})\b", re.IGNORECASE), lambda m: "QUT-" + m.group(1)),
    ("quote", re.compile(r"\bQUOTE[- ]?(\d{3,})\b", re.IGNORECASE), lambda m: "QUOTE-" + m.group(1)),
]


def _filename_hint(field_key: str, filename: str) -> str | None:
    """Scan the filename for an ID token matching field_key's pattern.
    Returns the normalized ID value, or None if no match.
    """
    if not filename:
        return None
    primary = field_key.split()[0] if field_key else ""
    for key, pattern, builder in _FILENAME_HINT_PATTERNS:
        if key != primary:
            continue
        m = pattern.search(filename)
        if m:
            return builder(m)
    return None


def extract_ids(doc: ParsedDocument, doc_type: str) -> dict[str, ExtractedValue]:
    out: dict[str, ExtractedValue] = {}
    id_fields = [f for f in fields_for(doc_type) if type_of(doc_type, f) == FieldType.ID]
    cands = find_candidates(doc, FieldType.ID)
    for field in id_fields:
        field_key = field.replace("_id", "").replace("_", " ").lower()
        scored: list = []
        for c in cands:
            lbl = _nearby_label(c, doc.tokens)
            score = _score_candidate(field_key, doc_type, c, lbl)
            if score > 0:
                scored.append((score, c))
        if scored:
            scored.sort(key=lambda s: -s[0])
            best = scored[0][1]
            out[field] = ExtractedValue(
                value=best.text, provenance="extracted",
                anchor_text=best.text, anchor_ref=best.tokens[0].anchor,
                source="structural", confidence=1.0, attempt=1,
            )

    # Filename-hint fallback: for fields still unresolved after body
    # discovery, try to pull the ID out of the document filename. This
    # is the "...for PO508084 .pdf" convention where the PO reference
    # lives only in the filename and not in the invoice body.
    # Only applied to fields whose primary keyword has a registered
    # filename pattern (po / quote), never to invoice_id.
    for field in id_fields:
        if field in out:
            continue
        field_key = field.replace("_id", "").replace("_", " ").lower()
        hint_val = _filename_hint(field_key, getattr(doc, "filename", "") or "")
        if hint_val:
            out[field] = ExtractedValue(
                value=hint_val, provenance="extracted",
                anchor_text=hint_val, anchor_ref=None,
                source="filename_hint", confidence=0.9, attempt=1,
            )
    return out
