"""quote_template_reader: a seller's own quote in, a PROPOSED template out. It saves nothing.

The seller uploads one of their standard quotes; this reads its FORMAT -- which lines it
has, what each is measured in, which terms it states, its column and section headings --
and proposes a template. The seller reviews it and saves it through
PUT /spendiq/quote-templates/{key}, the same as a template they wrote by hand.

A template is structure and nothing else (see templates.py), so no price, cost, discount
or total leaves here. The model is not trusted with that: it is given no field to answer
a figure in, and every string it does return is checked in code --

  * a line, a heading, a unit or an assumption is kept only if it is on the document;
  * a money figure anywhere in the answer is taken out (an assumption that states one
    is left out whole);
  * a term is kept only if the document has words for it;
  * how many lines carried a price is counted here, from the document's own rows.

The instruction is a governed row in proc.bp_prompt (`sales_quote_template_read`). A
missing row fails the reading; no text held in code stands in for it.
"""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.services.agentnick_driver import generate as ollama_generate
from src.services.sell_side import templates
from src.services.sell_side.builder import LINE_TAGS, TERM_KEYS, clean_html
from src.services.sell_side.quote_agent import AgentUnavailable, _default_busy, _default_prompt

PROMPT_NAME = "sales_quote_template_read"
MAX_BYTES = 15 * 1024 * 1024
MAX_DOCUMENT_CHARS = 20000
MAX_OCR_PAGES = 10
MAX_HEADINGS = 20
CONVERT_SECONDS = 60

_TEXT_TYPES = (".txt",)
_TABLE_TYPES = (".pdf", ".docx", ".xlsx", ".csv")
_OLD_OFFICE = {".doc": "docx", ".xls": "xlsx"}
_IMAGE_TYPES = (".png", ".jpg", ".jpeg")
SUPPORTED = _TEXT_TYPES + _TABLE_TYPES + tuple(_OLD_OFFICE) + _IMAGE_TYPES

# What the model may say. Nothing here holds a figure: there is no field for a price, a
# cost, a quantity or a total, so a reading cannot carry one even when the model tries.
SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "category": {"type": "string"},
        "columns": {"type": "array", "maxItems": 30, "items": {"type": "string"}},
        "sections": {"type": "array", "maxItems": 30, "items": {"type": "string"}},
        "lines": {"type": "array", "maxItems": 40, "items": {
            "type": "object",
            "properties": {"description": {"type": "string"},
                           "unit": {"type": "string"},
                           "tag": {"type": "string", "enum": list(LINE_TAGS)}},
            "required": ["description", "unit", "tag"]}},
        "termFields": {"type": "array", "items": {"type": "string", "enum": list(TERM_KEYS)}},
        "assumptions": {"type": "array", "maxItems": 30, "items": {"type": "string"}},
    },
    "required": ["name", "category", "columns", "sections", "lines", "termFields", "assumptions"],
}

# A money figure: with a currency sign or code, with pence, or with thousands separators.
# A bare "10" (a quantity, "12 months") is not one.
MONEY = re.compile(
    r"(?:[£$€¥]|\b(?:GBP|USD|EUR|INR)\b)\s?\d[\d,]*(?:\.\d+)?(?:\s?[kKmM]\b)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:GBP|USD|EUR|INR)\b"
    r"|\b\d[\d,]*\.\d{2}\b"
    r"|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")
_NUMBER = re.compile(r"^[£$€¥]?\s?-?\d[\d,]*(?:\.\d+)?%?$")

# The words a document uses for each term. A term the model names is kept only if one of
# these is on the document: "it has payment terms" must be something the page says.
_TERM_WORDS = {
    "payment": r"\bpayment|\bpayable|\bnet\s?\d+\b|\bdays? (?:from|of|after) (?:the )?(?:date of )?invoice"
               r"|\bdue (?:on|within|in) \d",
    "start": r"\bstart|\bcommenc|\bbegin|\beffective date|\bgo[- ]live",
    "lengthMonths": r"\b\d+\s*months?\b|\bterm of\b|\bduration|\bcontract term|\bminimum term"
                    r"|\bsubscription term|\binitial term",
    "renewal": r"\brenew",
    "sla": r"\bservice level|\bsla\b|\buptime|\bavailability|\bresponse time",
}


class Unsupported(ValueError):
    """Not a file this reads (type or size). HTTP 400."""


class Unreadable(ValueError):
    """A file of the right type with nothing usable in it. HTTP 422."""


# --- the document as text ----------------------------------------------------------

def _cell(region: Any) -> str:
    words = " ".join(t.text for t in (region.tokens or []) if t.text)
    return re.sub(r"\s+", " ", words or (region.label or "")).strip()


def _rows(doc: Any) -> List[List[str]]:
    out = []
    for table in doc.tables:
        for row in table.rows:
            cells = [_cell(r) for r in row]
            if any(cells):
                out.append(cells)
    return out


def _row_text(cells: List[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _ocr_image(data: bytes) -> str:
    import io

    import pytesseract
    from PIL import Image

    try:
        # --psm 4 reads a table row by row; the default reads it column by column, which
        # parts each item from its price and so from the count of lines that had one.
        return pytesseract.image_to_string(Image.open(io.BytesIO(data)), config="--psm 4")
    except Exception as exc:
        raise Unreadable(f"I could not read text from this image ({type(exc).__name__}).") from None


def _ocr_pdf(data: bytes) -> str:
    import fitz

    parts = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in list(doc)[:MAX_OCR_PAGES]:
            parts.append(_ocr_image(page.get_pixmap(dpi=200).tobytes("png")))
    return "\n".join(parts)


def _convert(data: bytes, ext: str) -> bytes:
    """An old Word or Excel file, as its newer kind, through LibreOffice."""
    target = _OLD_OFFICE[ext]
    with tempfile.TemporaryDirectory(prefix="quote-read-") as tmp:
        src = os.path.join(tmp, "quote" + ext)
        with open(src, "wb") as fh:
            fh.write(data)
        try:
            subprocess.run(
                ["soffice", f"-env:UserInstallation=file://{tmp}/profile", "--headless",
                 "--convert-to", target, "--outdir", tmp, src],
                check=True, capture_output=True, timeout=CONVERT_SECONDS)
            with open(os.path.join(tmp, "quote." + target), "rb") as fh:
                return fh.read()
        except (OSError, subprocess.SubprocessError):
            raise Unreadable(f"I could not open this {ext} file; try saving it as "
                             f".{target} or PDF.") from None


def _parsed(data: bytes, filename: str, ext: str) -> Tuple[str, List[List[str]]]:
    """(the document's prose, its table rows)."""
    from src.services.structural_extractor import parsing

    if ext in _TEXT_TYPES:
        try:
            return data.decode("utf-8-sig"), []
        except UnicodeDecodeError:
            return data.decode("latin-1"), []
    if ext in _IMAGE_TYPES:
        return _ocr_image(data), []
    if ext in _OLD_OFFICE:
        data, ext = _convert(data, ext), "." + _OLD_OFFICE[ext]
        filename = os.path.splitext(filename)[0] + ext
    try:
        doc = parsing.parse(data, filename)
    except Exception:
        raise Unreadable(f"I could not open this {ext} file. It may be damaged or "
                         "password-protected.") from None
    rows = _rows(doc)
    if doc.source_format in ("csv", "xlsx"):
        return "", rows                     # their text IS their cells
    prose = doc.full_text or ""
    if doc.source_format == "pdf" and len(prose.strip()) < 40:
        prose = _ocr_pdf(data)              # a scan: no text layer to read
    return prose, rows


def document_text(data: bytes, filename: str) -> str:
    """The document as the model reads it: its prose, then each table row as
    `| cell | cell |`. Raises Unsupported or Unreadable."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in SUPPORTED:
        raise Unsupported("I can read PDF, Word, Excel, CSV, text and image (PNG or JPG) "
                          f"files, not “{ext or 'a file with no extension'}”.")
    if len(data) > MAX_BYTES:
        raise Unsupported("The file is over 15 MB.")
    prose, rows = _parsed(data, filename, ext)
    parts = [prose.strip()] if prose.strip() else []
    parts += [_row_text(r) for r in rows]
    text = "\n".join(parts).strip()
    if not text:
        raise Unreadable("The document has no text I could read.")
    return text


# --- checking what the model said ---------------------------------------------------

def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split())


def _on(doc: str, phrase: str) -> bool:
    p = _norm(phrase)
    return bool(p) and f" {p} " in f" {doc} "


def _unpriced(s: Any) -> str:
    """The text with every money figure taken out, and the joins it leaves tidied."""
    s = MONEY.sub(" ", str(s or ""))
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"^[\s\-–—:@,/|]+|[\s\-–—:@,/|(]+$", "", s).strip()


def _slug(text: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40].strip("_")
    if not key or not key[0].isalpha():
        key = ("line_" + key).strip("_")
    return key if len(key) >= 2 else "line"


def _priced(cells: List[str], description: str) -> bool:
    """Does this row put a price on it: a money-shaped cell, or two numbers (a quantity
    and a price) besides the description."""
    others = [c for c in cells if c and _norm(c) != _norm(description)]
    if any(MONEY.search(c) for c in others):
        return True
    return sum(1 for c in others if _NUMBER.match(c.strip())) >= 2


def _price_rows(text: str) -> List[List[str]]:
    rows = []
    for raw in text.splitlines():
        raw = raw.strip()
        if raw.startswith("|"):
            rows.append([c.strip() for c in raw.strip("|").split("|")])
        elif raw:
            rows.append([c.strip() for c in re.split(r"\t|\s{2,}", raw) if c.strip()])
    return rows


def _headings(found: Any, doc: str) -> List[str]:
    out: List[str] = []
    for h in found or []:
        h = _unpriced(h)[:120]
        if h and _on(doc, h) and h not in out:
            out.append(h)
    return out[:MAX_HEADINGS]


def _file_name(filename: str) -> str:
    base = re.sub(r"[_\-]+", " ", os.path.splitext(os.path.basename(filename or ""))[0]).strip()
    base = _unpriced(base) or "Quote"
    return (base[:1].upper() + base[1:])[:120]


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many.format(n=n)


def _shape(said: Dict[str, Any], text: str, filename: str) -> Dict[str, Any]:
    doc = _norm(text)
    rows = _price_rows(text)
    notes: List[str] = []

    lines, keys, not_on, no_unit, priced = [], set(), [], 0, 0
    for item in said.get("lines") or []:
        if not isinstance(item, dict):
            continue
        description = _unpriced(item.get("description"))[:300]
        if not description:
            continue
        if not _on(doc, description):
            not_on.append(description[:60])
            continue
        unit = _unpriced(item.get("unit"))[:40]
        if not unit or not _on(doc, unit):
            unit, no_unit = "unit", no_unit + 1
        key, n = _slug(description), 2
        while key in keys:
            key, n = f"{_slug(description)[:36]}_{n}", n + 1
        keys.add(key)
        tag = item.get("tag") if item.get("tag") in LINE_TAGS else "service"
        lines.append({"key": key, "tag": tag, "description": description, "unit": unit})
        if any(_priced(r, description) for r in rows if any(_on(_norm(c), description) for c in r)):
            priced += 1

    if not lines:
        raise Unreadable("I could not find a pricing table in this document: no line items "
                         "were found. Try a quote that lists its items in a table.")
    if len(lines) > templates.MAX_TEMPLATE_LINES:
        notes.append(f"The document has {len(lines)} lines; a template holds "
                     f"{templates.MAX_TEMPLATE_LINES}, so the first "
                     f"{templates.MAX_TEMPLATE_LINES} are kept.")
        kept = lines[:templates.MAX_TEMPLATE_LINES]
        priced = min(priced, len(kept))
        lines = kept
    if not_on:
        notes.append("Left out because I could not find "
                     + _plural(len(not_on), "it", "them") + " on the document: "
                     + "; ".join(f"“{d}”" for d in not_on[:5]) + ".")
    if no_unit:
        notes.append(_plural(no_unit, "One line had no unit I could find; it is “unit”.",
                             "{n} lines had no unit I could find; they are “unit”."))
    if priced:
        notes.append(_plural(priced, "The price on one line was left out: a template carries no prices.",
                             "The prices on {n} lines were left out: a template carries no prices."))

    terms = [f for f in TERM_KEYS if f in (said.get("termFields") or [])
             and re.search(_TERM_WORDS[f], text, re.I)]

    kept_assumptions, priced_assumptions = [], 0
    for sentence in said.get("assumptions") or []:
        sentence = re.sub(r"\s+", " ", str(sentence or "")).strip()
        if not sentence or not _on(doc, sentence):
            continue
        if MONEY.search(sentence):
            priced_assumptions += 1
            continue
        kept_assumptions.append(sentence[:500])
    if priced_assumptions:
        notes.append(_plural(priced_assumptions,
                             "One assumption stated a figure and was left out.",
                             "{n} assumptions stated a figure and were left out."))
    assumptions_html = (clean_html("<ul>" + "".join(f"<li>{html.escape(s)}</li>"
                                                    for s in kept_assumptions) + "</ul>",
                                   "assumptionsHtml") if kept_assumptions else None)

    name = _unpriced(said.get("name"))[:120]
    if not name or not _on(doc, name):
        name = _file_name(filename)

    body = {"category": _unpriced(said.get("category"))[:300] or None, "lines": lines,
            "termFields": terms, "assumptionsHtml": assumptions_html}
    cleaned = templates.clean(body)          # the save route's own check, as it will run
    return {"name": name, "category": cleaned["category"], "lines": cleaned["lines"],
            "termFields": cleaned["termFields"],
            "columns": _headings(said.get("columns"), doc),
            "sections": _headings(said.get("sections"), doc),
            "pricesLeftOut": priced, "notes": [_unpriced(n) or n for n in notes],
            "assumptionsHtml": cleaned["assumptionsHtml"], "sample": False}


def read(data: bytes, filename: str, *,
         call: Optional[Callable[..., Optional[str]]] = None,
         prompt: Optional[Callable[[], str]] = None,
         busy: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """The proposed template for this document. Raises Unsupported (wrong type or size),
    Unreadable (nothing to propose from) or AgentUnavailable (no prompt, model busy or
    silent). Writes nothing."""
    from src.services.agent_policy.extractor import _fill

    text = document_text(data, filename)
    truncated = len(text) > MAX_DOCUMENT_CHARS
    wording = (prompt or (lambda: _default_prompt(PROMPT_NAME)))()
    if (busy or (_default_busy if call is None else (lambda: False)))():
        raise AgentUnavailable("the model is busy with other work; try again shortly")
    raw = (call or ollama_generate)(
        _fill(wording, {"filename": os.path.basename(filename or ""),
                        "document": text[:MAX_DOCUMENT_CHARS]}),
        format=SCHEMA, think=False, temperature=0, num_predict=3072, use_load_options=True)
    if not raw:
        raise AgentUnavailable("the model did not answer")
    try:
        said = json.loads(raw)
        if not isinstance(said, dict):
            raise ValueError("not an object")
    except ValueError:
        raise AgentUnavailable("the model's answer could not be read") from None
    out = _shape(said, text[:MAX_DOCUMENT_CHARS], filename)
    if truncated:
        out["notes"].append("The document is long; only its first part was read.")
    return out
