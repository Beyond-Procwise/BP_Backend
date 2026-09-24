"""What a deal's triage depends on, as one hash (final review F1).

The scheduler re-triages a deal when this hash (or the tolerance fingerprint) differs
from the one stored for its last successful triage. It covers every field the checks
read, so promotion, deal assignment, a duplicate flag, or a document leaving the deal
all change it -- none of which a _trgt timestamp reliably records.

Order-independent: documents, lines and duplicate flags are sorted before hashing, so
the same content loaded in a different order hashes the same. Pure.
"""
from __future__ import annotations

import hashlib
import json

from .model import Doc, DocumentSet, DuplicateFlag, Line

_DOC_FIELDS = ("supplier_id", "currency", "doc_date", "net", "tax", "gross",
               "payment_terms", "po_id", "quote_ref", "confidence", "fx_to_gbp")
_LINE_FIELDS = ("line_ref", "item_id", "description", "quantity", "uom", "unit_price",
                "line_amount", "po_id", "delivery_date")


def _s(value):
    return None if value is None else str(value)


def _line(l: Line) -> list:
    return [_s(getattr(l, f)) for f in _LINE_FIELDS]


def _key(row: list) -> list:
    # None sorts before any string, and never compares against one.
    return [(v is not None, v or "") for v in row]


def _doc(d: Doc) -> list:
    lines = sorted((_line(l) for l in d.lines), key=_key)
    return [d.doc_type, d.doc_id, *[_s(getattr(d, f)) for f in _DOC_FIELDS], lines]


def _dup(f: DuplicateFlag) -> list:
    return [_s(f.invoice_id), _s(f.earlier_invoice_id), _s(f.amount)]


def deal_content_hash(ds: DocumentSet) -> str:
    docs = sorted((_doc(d) for d in (*ds.quotes, *ds.pos, *ds.invoices)),
                  key=lambda row: (row[0], row[1], json.dumps(row)))
    dups = sorted((_dup(f) for f in ds.duplicates), key=_key)
    blob = json.dumps({"deal_id": ds.deal_id, "docs": docs, "duplicates": dups},
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
