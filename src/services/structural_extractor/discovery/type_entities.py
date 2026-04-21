import re
from dataclasses import dataclass
from dateutil import parser as date_parser

from src.services.structural_extractor.discovery.schema import FieldType
from src.services.structural_extractor.parsing.model import Token, ParsedDocument


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
            if has_symbol or (val >= 0 and val < 1e10 and "." in t.text):
                cands.append(Candidate(text=t.text, tokens=[t], parsed_value=val))
        except ValueError:
            continue
    return cands


def _find_percent(doc: ParsedDocument) -> list[Candidate]:
    cands: list[Candidate] = []
    for t in doc.tokens:
        if "%" in t.text:
            clean = t.text.replace("%", "").replace("(", "").replace(")", "").strip()
            try:
                val = float(clean)
                if 0 <= val <= 100:
                    cands.append(Candidate(text=t.text, tokens=[t], parsed_value=val))
            except ValueError:
                continue
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


def _find_org(doc: ParsedDocument) -> list[Candidate]:
    cands: list[Candidate] = []
    suffixes = {"Ltd", "Ltd.", "LLC", "Inc", "Inc.", "Limited", "plc", "Corp", "Company", "GmbH", "AG", "SA"}
    tokens = doc.tokens
    n = len(tokens)
    for i in range(n):
        if tokens[i].text.rstrip(",.") in suffixes:
            start = i
            while start > 0 and tokens[start - 1].text and tokens[start - 1].text[0].isupper():
                start -= 1
            span = tokens[start:i + 1]
            text = " ".join(t.text for t in span)
            cands.append(Candidate(text=text, tokens=span, parsed_value=text))
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
