import re
from dateutil import parser as date_parser
from src.services.structural_extractor.parsing.model import ParsedDocument, BBox
from src.services.structural_extractor.discovery.schema import FieldType, type_of, fields_for
from src.services.structural_extractor.discovery.type_entities import find_candidates
from src.services.structural_extractor.discovery.proximity import inferred_label
from src.services.structural_extractor.types import ExtractedValue

UK_POSTCODE_RE = re.compile(r"^[A-Z]{1,2}[0-9][A-Z0-9]?$")


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


def _nearby_label(cand, all_tokens) -> str:
    """Find a label near the candidate — same line OR one line above (left-aligned)."""
    if not cand.tokens:
        return ""
    first = cand.tokens[0]
    anchor = first.anchor
    if not isinstance(anchor, BBox):
        return inferred_label(cand, all_tokens)
    same_line = inferred_label(cand, all_tokens)
    above: list = []
    cand_cy = (anchor.y0 + anchor.y1) / 2
    for t in all_tokens:
        if not isinstance(t.anchor, BBox) or t.anchor.page != anchor.page:
            continue
        t_cy = (t.anchor.y0 + t.anchor.y1) / 2
        if t_cy >= cand_cy:
            continue
        if cand_cy - t_cy > 30:
            continue
        if t.anchor.x1 < anchor.x0 - 10 or t.anchor.x0 > anchor.x1 + 80:
            continue
        above.append(t)
    above.sort(key=lambda t: (t.anchor.y0, t.anchor.x0))
    above_text = " ".join(t.text for t in above[-5:])
    return (same_line + " " + above_text).strip()


def extract_dates(doc: ParsedDocument, doc_type: str) -> dict[str, ExtractedValue]:
    out: dict[str, ExtractedValue] = {}
    date_fields = [
        f for f in fields_for(doc_type)
        if type_of(doc_type, f) == FieldType.DATE and f != "due_date"
    ]
    locale = detect_locale(doc)
    dayfirst = locale == "dmy"
    cands = find_candidates(doc, FieldType.DATE)
    for field in date_fields:
        field_key = field.replace("_date", "").replace("_", " ").lower()
        scored: list = []
        for c in cands:
            lbl = _nearby_label(c, doc.tokens).lower()
            overlap = sum(1 for w in field_key.split() if w and w in lbl)
            if overlap > 0:
                try:
                    dt = date_parser.parse(c.text, dayfirst=dayfirst, fuzzy=False)
                    if 1980 <= dt.year <= 2100:
                        scored.append((overlap, c, dt))
                except Exception:
                    continue
        if scored:
            # Prefer: (a) highest label overlap, (b) candidate that contains an explicit
            # 4-digit year token (more complete dates), (c) most tokens.
            def _rank(s):
                overlap, cand, dt = s
                has_year = any(len(t.text) == 4 and t.text.isdigit() for t in cand.tokens)
                return (-overlap, -int(has_year), -len(cand.tokens))
            scored.sort(key=_rank)
            _, best, parsed = scored[0]
            out[field] = ExtractedValue(
                value=parsed.strftime("%Y-%m-%d"), provenance="extracted",
                anchor_text=best.text, anchor_ref=best.tokens[0].anchor,
                source="structural", confidence=1.0, attempt=1,
            )
    return out
