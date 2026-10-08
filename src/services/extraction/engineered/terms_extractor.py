"""Commercial terms -- payment terms, validity, lead time, surcharges -- as the document states them.

A quote's "Commercial terms" table is not a list of priced items. Read by the line-item
extractor its rows became amountless "lines" (the freight quotes' "Fuel surcharge -fixed 4.0%"),
and once summary/label rows were filtered out they were not captured at all. This reads such a
table as what it is: label / value pairs, kept verbatim -- nothing is interpreted, so a value
like "variable" or "DTI-linked, capped at 3.0%" is stored exactly as printed.

A table counts as terms when its title row names terms ("Commercial terms", "Terms &
conditions") or when one of its row labels is a known term ("Payment terms", "Validity", "Fuel
surcharge"...), and every row has at most a label and a value. A line-item table (many columns)
or an address block never qualifies. Works from the parser's markdown full_text, so documents
already stored can be read without parsing them again.
"""
from __future__ import annotations

import html
import re

#: Labels that name a commercial term. Matched as whole words at the start of a row's label.
_TERM_WORDS = (
    "payment terms", "payment", "terms", "validity", "price validity", "quote validity",
    "lead time", "delivery", "invoicing", "warranty", "incoterms", "surcharge", "fuel surcharge",
    "retention", "liquidated", "minimum order", "cancellation", "price review",
    "indexation", "escalation", "insurance", "currency", "contract term", "contract sum",
    "contract value", "change control", "key personnel", "termination", "notice period",
    "price basis",
)
_TERM_LABEL_RE = re.compile(
    r"^\s*(?:" + "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in sorted(_TERM_WORDS, key=len, reverse=True))
    + r")\b", re.I)
_TITLE_RE = re.compile(r"\bterms\b", re.I)
_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}")
#: Where a term's name ends and its stated value begins: "Fuel surcharge -fixed 4.0%",
#: "Fuel surcharge — 4.2%", "Payment: 30 days", "Surcharge (fuel)".
_KEY_END_RE = re.compile(r"\s+[-–—]|\s*[:(]|\s+\d")


def term_key(label: str) -> str:
    """The term a label names, without the value some documents write into the label."""
    s = re.sub(r"\s+", " ", str(label or "")).strip()
    m = _KEY_END_RE.search(s)
    return (s[:m.start()] if m else s).strip().lower()


def _cells(line: str) -> list[str]:
    parts = line.strip().strip("|").split("|")
    return [html.unescape(re.sub(r"\s+", " ", p).strip().strip("*").strip()) for p in parts]


def _tables(markdown: str) -> list[list[list[str]]]:
    tables, cur = [], []
    for line in (markdown or "").splitlines():
        if line.lstrip().startswith("|"):
            if not _SEPARATOR_RE.match(line.strip()):
                cur.append(_cells(line))
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    return tables


def _distinct(row: list[str]) -> list[str]:
    out: list[str] = []
    for c in row:
        if c and c not in out:
            out.append(c)
    return out


def terms_from_rows(rows: list[list[str]]) -> list[dict[str, str]]:
    """Label / value pairs from one table's rows, or [] when it is not a terms table."""
    distinct = [_distinct(r) for r in rows]
    if not distinct or any(len(d) > 2 for d in distinct):
        return []                                   # a line-item table, not terms
    title = distinct[0]
    titled = len(title) == 1 and bool(_TITLE_RE.search(title[0]))
    pairs = [d for d in (distinct[1:] if titled else distinct) if len(d) == 2]
    if not pairs:
        return []
    if not titled:
        # No "terms" title: keep only rows that are themselves terms. A reference block
        # ("PO reference | Payment terms") or a reviewer's callout ("Expenses regression")
        # is not a term of the deal.
        pairs = [p for p in pairs if _TERM_LABEL_RE.match(p[0])]
    return [{"label": label, "value": value} for label, value in pairs]


def terms_from_markdown(markdown: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for rows in _tables(markdown):
        out.extend(terms_from_rows(rows))
    return out


def terms_from_parsed(parsed) -> list[dict[str, str]]:
    """Terms from a ParsedDocument: its tables first, its markdown text when no table yields any."""
    out: list[dict[str, str]] = []
    for page in getattr(parsed, "pages", None) or []:
        for tbl in getattr(page, "tables", None) or []:
            rows = [[(c.text or "").strip() for c in row] for row in (tbl.rows or [])]
            out.extend(terms_from_rows([[re.sub(r"\s+", " ", c) for c in r] for r in rows]))
    return out or terms_from_markdown(getattr(parsed, "full_text", "") or "")
