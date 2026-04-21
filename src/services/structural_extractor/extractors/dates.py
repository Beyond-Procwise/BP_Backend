import re
from datetime import date as _date
from dateutil import parser as date_parser
from src.services.structural_extractor.parsing.model import ParsedDocument, BBox
from src.services.structural_extractor.discovery.schema import FieldType, type_of, fields_for
from src.services.structural_extractor.discovery.type_entities import find_candidates
from src.services.structural_extractor.discovery.proximity import inferred_label
from src.services.structural_extractor.types import ExtractedValue

UK_POSTCODE_RE = re.compile(r"^[A-Z]{1,2}[0-9][A-Z0-9]?$")

# A candidate's text must contain AT LEAST one of these to be a "real" date:
#   (a) a 4-digit year, OR
#   (b) a month name or abbreviation (Jan, February, etc.)
MONTH_NAMES = {
    "jan", "january", "feb", "february", "mar", "march", "apr", "april",
    "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept",
    "september", "oct", "october", "nov", "november", "dec", "december",
}
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
NUMERIC_DATE_RE = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")

# Synonym labels per (field_key, doc_type) — labels that unambiguously
# identify the field even when the canonical keyword is absent.
FIELD_LABEL_SYNONYMS: dict[tuple[str, str], set[str]] = {
    # Invoices: a bare "Date:" label usually IS the invoice date.
    ("invoice", "Invoice"): {"date"},
    # POs: "PO Date" → order_date; "Order Date" obviously too.
    ("order", "Purchase_Order"): {"po date", "po"},
    # Quotes: bare "Date:" is the quote date.
    ("quote", "Quote"): {"date"},
}


def detect_locale(doc: ParsedDocument) -> str:
    for t in doc.tokens:
        m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", t.text)
        if m:
            d1 = int(m.group(1))
            if d1 > 12:
                return "dmy"
    for t in doc.tokens:
        if UK_POSTCODE_RE.match(t.text):
            return "dmy"
    if any(t.text.upper() == "USD" for t in doc.tokens):
        return "mdy"
    return "ambiguous"


def _candidate_is_real_date(text: str) -> bool:
    """A DATE candidate must have a 4-digit year OR a month name OR
    match a numeric date pattern (dd/mm/yyyy, dd-mm-yy). A bare number
    like '14' or '04' yields dateutil-defaulted today's date — reject it.
    """
    txt = text.strip().lower()
    if YEAR_RE.search(txt):
        return True
    # Token-level month check (reject substrings inside unrelated words)
    for tok in txt.replace(",", " ").split():
        tok_clean = tok.strip(".")
        if tok_clean in MONTH_NAMES:
            return True
    if NUMERIC_DATE_RE.match(text.strip()):
        return True
    return False


def _label_words(label: str) -> list[str]:
    words = []
    for raw in label.lower().split():
        norm = raw.strip(".,:;()[]{}\"'-/")
        if norm:
            words.append(norm)
    return words


def _word_in(word: str, words: list[str]) -> bool:
    return word.lower().strip() in words


def _phrase_in(phrase: str, label: str) -> bool:
    return phrase.lower() in label.lower()


def _nearby_label(cand, all_tokens) -> str:
    """Find a label near the candidate — same line OR one line above.

    The above-line search includes tokens that share the candidate's
    vertical band and are on the immediately-above y-line. Horizontal
    overlap is loosened to 80pt before / 80pt after, because right-aligned
    metadata blocks commonly place the label ('Invoice No. 2025-290')
    on one line and the value ('1 April 2020') on the next, with the
    value slightly indented or mis-aligned.
    """
    if not cand.tokens:
        return ""
    first = cand.tokens[0]
    last = cand.tokens[-1]
    anchor = first.anchor
    last_anchor = last.anchor
    if not isinstance(anchor, BBox):
        return inferred_label(cand, all_tokens)
    same_line = inferred_label(cand, all_tokens)
    above: list = []
    cand_cy = (anchor.y0 + anchor.y1) / 2
    cand_x0 = anchor.x0
    cand_x1 = last_anchor.x1 if isinstance(last_anchor, BBox) else anchor.x1
    for t in all_tokens:
        if not isinstance(t.anchor, BBox) or t.anchor.page != anchor.page:
            continue
        t_cy = (t.anchor.y0 + t.anchor.y1) / 2
        if t_cy >= cand_cy:
            continue
        if cand_cy - t_cy > 30:
            continue
        # Loosened horizontal overlap: token must not be FAR left or FAR
        # right of the candidate block.
        if t.anchor.x1 < cand_x0 - 80 or t.anchor.x0 > cand_x1 + 80:
            continue
        above.append(t)
    above.sort(key=lambda t: (t.anchor.y0, t.anchor.x0))
    above_text = " ".join(t.text for t in above[-6:])
    return (same_line + " " + above_text).strip()


def _score_label(field_key: str, doc_type: str, label: str) -> int:
    """Score a candidate by label match. Uses word-boundary matching and
    doc-type-aware synonym labels.
    """
    words = _label_words(label)
    field_words = [w for w in field_key.split() if w]
    primary = field_words[0] if field_words else ""

    score = 0
    for w in field_words:
        if _word_in(w, words):
            score += 10

    # Doc-type synonyms (bare "Date" accepted for invoice_date, etc.)
    synonyms = FIELD_LABEL_SYNONYMS.get((primary, doc_type), set())
    for syn in synonyms:
        if " " in syn:
            if _phrase_in(syn, label):
                score += 8
                break
        else:
            if _word_in(syn, words):
                score += 6
                break

    return score


def extract_dates(doc: ParsedDocument, doc_type: str) -> dict[str, ExtractedValue]:
    out: dict[str, ExtractedValue] = {}
    date_fields = [
        f for f in fields_for(doc_type)
        if type_of(doc_type, f) == FieldType.DATE and f != "due_date"
    ]
    locale = detect_locale(doc)
    dayfirst = locale == "dmy"
    cands = find_candidates(doc, FieldType.DATE)
    today = _date.today()
    for field in date_fields:
        field_key = field.replace("_date", "").replace("_", " ").lower()
        scored: list = []
        for c in cands:
            # Hard gate: the candidate text must look like a real date
            # (contains a year, month name, or numeric dd/mm/yyyy form).
            # Prevents 'NET 14' or 'Account 04' from being parsed into
            # today's date via dateutil defaults.
            if not _candidate_is_real_date(c.text):
                continue
            lbl = _nearby_label(c, doc.tokens)
            label_score = _score_label(field_key, doc_type, lbl)
            if label_score <= 0:
                continue
            try:
                dt = date_parser.parse(c.text, dayfirst=dayfirst, fuzzy=False)
            except Exception:
                continue
            if not (1980 <= dt.year <= 2100):
                continue
            # Belt-and-braces: if the candidate text lacks a year token
            # and the parsed year equals the current year, that's dateutil
            # filling in today's year — reject.
            if not YEAR_RE.search(c.text):
                if dt.year == today.year:
                    continue
            scored.append((label_score, c, dt))
        if scored:
            # Prefer highest label score, then prefer candidates whose
            # text contains an explicit 4-digit year, then longest tokens.
            def _rank(s):
                score, cand, dt = s
                has_year = bool(YEAR_RE.search(cand.text))
                return (-score, -int(has_year), -len(cand.tokens))
            scored.sort(key=_rank)
            _, best, parsed = scored[0]
            out[field] = ExtractedValue(
                value=parsed.strftime("%Y-%m-%d"), provenance="extracted",
                anchor_text=best.text, anchor_ref=best.tokens[0].anchor,
                source="structural", confidence=1.0, attempt=1,
            )
    return out
