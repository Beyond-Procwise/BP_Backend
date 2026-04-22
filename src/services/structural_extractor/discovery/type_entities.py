import re
from dataclasses import dataclass
from dateutil import parser as date_parser

from src.services.structural_extractor.discovery.schema import FieldType
from src.services.structural_extractor.parsing.model import (
    Token, ParsedDocument, CellRef, ColumnRef, NodeRef
)


ISO_4217 = {"USD", "GBP", "EUR", "AUD", "CAD", "JPY", "CHF", "INR", "CNY", "HKD", "SGD", "NZD"}
CURRENCY_SYMBOLS = {"£": "GBP", "$": "USD", "€": "EUR", "¥": "JPY", "₹": "INR"}
UK_POSTCODE_RE = re.compile(r"^[A-Z]{1,2}[0-9][A-Z0-9]?$")
US_ZIP_RE = re.compile(r"^\d{5}(-\d{4})?$")


@dataclass
class Candidate:
    text: str
    tokens: list[Token]
    parsed_value: object


def find_candidates(doc: ParsedDocument, ftype: FieldType) -> list[Candidate]:
    if ftype == FieldType.DATE:
        return _find_date(doc)
    if ftype == FieldType.MONEY:
        return _find_money(doc)
    if ftype == FieldType.PERCENT:
        return _find_percent(doc)
    if ftype == FieldType.CURRENCY_CODE:
        return _find_currency(doc)
    if ftype == FieldType.ID:
        return _find_id(doc)
    if ftype == FieldType.ORG:
        return _find_org(doc)
    if ftype == FieldType.ADDRESS:
        return _find_address(doc)
    if ftype == FieldType.TEXT:
        return [Candidate(t.text, [t], t.text) for t in doc.tokens]
    return []


def _find_date(doc: ParsedDocument) -> list[Candidate]:
    cands: list[Candidate] = []
    tokens = doc.tokens
    n = len(tokens)
    for i in range(n):
        for window in range(1, 5):
            if i + window > n:
                break
            segment = tokens[i:i + window]
            text = " ".join(t.text for t in segment)
            try:
                dt = date_parser.parse(text, fuzzy=False)
                if 1980 <= dt.year <= 2100:
                    cands.append(Candidate(text=text, tokens=segment, parsed_value=dt))
            except Exception:
                pass
    return cands


def _find_money(doc: ParsedDocument) -> list[Candidate]:
    cands: list[Candidate] = []
    for t in doc.tokens:
        has_symbol = any(s in t.text for s in CURRENCY_SYMBOLS)
        clean = t.text.replace(",", "").replace(" ", "")
        for s in CURRENCY_SYMBOLS:
            clean = clean.replace(s, "")
        try:
            val = float(clean)
        except ValueError:
            continue
        # Structured-table anchors (XLSX CellRef, CSV ColumnRef) carry
        # their own typing via cell context — a numeric cell in a column
        # labeled 'Unit Price' or 'Total' is money regardless of whether
        # it was stored with a decimal or a currency symbol. For PDFs
        # and DOCX (BBox / NodeRef) we still require a currency symbol
        # or a decimal point to avoid treating free-floating integers
        # (e.g. postal numbers, invoice number prefixes) as money.
        is_structured = isinstance(t.anchor, (CellRef, ColumnRef))
        if has_symbol or "." in t.text or is_structured:
            if 0 <= val < 1e10:
                cands.append(Candidate(text=t.text, tokens=[t], parsed_value=val))
    return cands


_PERCENT_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _find_percent(doc: ParsedDocument) -> list[Candidate]:
    cands: list[Candidate] = []
    for t in doc.tokens:
        if "%" not in t.text:
            continue
        # Try direct clean-up first ('20%', '(20%)', '20 %').
        clean = t.text.replace("%", "").replace("(", "").replace(")", "").strip()
        val: float | None = None
        try:
            val = float(clean)
        except ValueError:
            # Fallback: extract the first numeric fragment before '%' —
            # handles XLSX cells like 'Tax (10%)' where label and percent
            # co-exist in a single cell.
            m = _PERCENT_NUM_RE.search(t.text)
            if m:
                try:
                    val = float(m.group(1))
                except ValueError:
                    val = None
        if val is not None and 0 <= val <= 100:
            cands.append(Candidate(text=t.text, tokens=[t], parsed_value=val))
    return cands


def _find_currency(doc: ParsedDocument) -> list[Candidate]:
    cands: list[Candidate] = []
    for t in doc.tokens:
        tt = t.text.strip().upper()
        if tt in ISO_4217:
            cands.append(Candidate(text=t.text, tokens=[t], parsed_value=tt))
        else:
            for sym, code in CURRENCY_SYMBOLS.items():
                if sym in t.text:
                    cands.append(Candidate(text=sym, tokens=[t], parsed_value=code))
                    break
    return cands


def _find_id(doc: ParsedDocument) -> list[Candidate]:
    cands: list[Candidate] = []
    for t in doc.tokens:
        s = t.text.strip().rstrip(":,.")
        if len(s) >= 3 and any(c.isdigit() for c in s) and not any(x in s for x in "£$€¥%"):
            cands.append(Candidate(text=s, tokens=[t], parsed_value=s))
    return cands


def _same_anchor_group(a, b) -> bool:
    """Return True if two tokens share the same logical block/anchor —
    i.e. their walk-back shouldn't cross a boundary.

    - BBox: same page AND same y-row (within 6pt) — a PDF line.
    - NodeRef paragraph: same paragraph_index.
    - NodeRef table_cell: same (table_index, row, col).
    - CellRef: same (sheet, row, col).
    - ColumnRef: same row.
    """
    from src.services.structural_extractor.parsing.model import BBox as _BBox
    if type(a) is not type(b):
        return False
    if isinstance(a, _BBox):
        if a.page != b.page:
            return False
        ay = (a.y0 + a.y1) / 2
        by = (b.y0 + b.y1) / 2
        return abs(ay - by) <= 6
    if isinstance(a, NodeRef):
        if a.kind != b.kind:
            return False
        if a.kind == "paragraph":
            return a.paragraph_index == b.paragraph_index
        return (a.table_index == b.table_index and a.row == b.row and a.col == b.col)
    if isinstance(a, CellRef):
        return a.sheet == b.sheet and a.row == b.row and a.col == b.col
    if isinstance(a, ColumnRef):
        return a.row == b.row and a.col == b.col
    return False


def _find_org(doc: ParsedDocument) -> list[Candidate]:
    cands: list[Candidate] = []
    suffixes = {"Ltd", "Ltd.", "LLC", "Inc", "Inc.", "Limited", "plc", "Corp", "Company", "GmbH", "AG", "SA"}
    suffix_ci = {s.upper() for s in suffixes} | {s.upper().rstrip(".") for s in suffixes}
    tokens = doc.tokens
    n = len(tokens)
    for i in range(n):
        stripped = tokens[i].text.rstrip(",.")
        # Case-insensitive suffix match — DOCX preserves ALL-CAPS ('LTD',
        # 'INC') in letterhead paragraphs, which otherwise fail the strict
        # titlecase suffix list.
        if stripped in suffixes or stripped.upper() in suffix_ci:
            start = i
            # Walk back while tokens are (a) capitalised AND (b) share the
            # same anchor group (same paragraph for DOCX, same PDF line for
            # BBox, same cell for XLSX). Without this gate the walk-back
            # jumps across paragraph boundaries and yields spans like
            # 'PO No: PO-77821 Bill To: Assurity Ltd'.
            while start > 0 and tokens[start - 1].text and tokens[start - 1].text[0].isupper():
                if not _same_anchor_group(tokens[start - 1].anchor, tokens[i].anchor):
                    break
                start -= 1
            span = tokens[start:i + 1]
            text = " ".join(t.text for t in span)
            cands.append(Candidate(text=text, tokens=span, parsed_value=text))

    # Structured-anchor formats (XLSX / CSV) often store an entire org
    # name in ONE cell ('WidgetCo Ltd', 'MEGAMART SUPPLIES INC'). Detect
    # such single-cell ORG tokens by checking (a) contains a corporate
    # suffix as its last word, or (b) all-uppercase multi-word cell with
    # ≥2 words. Gate on structured anchors so PDF/DOCX word-level
    # tokens (which the logic above already handles) aren't double-counted.
    for t in tokens:
        if not isinstance(t.anchor, (CellRef, ColumnRef)):
            continue
        txt = t.text.strip()
        if not txt:
            continue
        words = txt.split()
        if len(words) < 2:
            continue
        last_ci = words[-1].rstrip(",.").upper()
        if last_ci in suffix_ci:
            cands.append(Candidate(text=txt, tokens=[t], parsed_value=txt))
            continue
        # All-uppercase multi-word letterhead cell
        alpha = "".join(ch for ch in txt if ch.isalpha())
        if alpha and alpha.isupper() and len(words) >= 2:
            cands.append(Candidate(text=txt, tokens=[t], parsed_value=txt))
    return cands


def _find_address(doc: ParsedDocument) -> list[Candidate]:
    """Anchor on a postcode, take the preceding 2-4 lines as the address block."""
    cands: list[Candidate] = []
    lines: dict[tuple, list[Token]] = {}
    for t in doc.tokens:
        key = (getattr(t.anchor, "page", 0), t.line_no if t.line_no is not None else t.block_no or t.order)
        lines.setdefault(key, []).append(t)
    line_items = sorted(lines.items())
    for idx, (_, line_toks) in enumerate(line_items):
        for tok in line_toks:
            clean = tok.text.rstrip(",.")
            if UK_POSTCODE_RE.match(clean) or US_ZIP_RE.match(clean):
                start = max(0, idx - 3)
                block = [t for _, ln in line_items[start:idx + 1] for t in ln]
                text = " ".join(t.text for t in block)
                cands.append(Candidate(text=text, tokens=block, parsed_value=text))
                break
    return cands
