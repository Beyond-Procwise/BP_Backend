"""Layout fingerprinting.

A fingerprint is a stable hex hash that identifies a document's
structural layout — independently of the specific values it carries.

Two docs from the same vendor using the same template should yield the
same fingerprint, even if their dates/IDs/amounts differ. The pipeline
uses the fingerprint to look up cached vendor-specific locator hints
(see template_store) so that subsequent docs from a known vendor
commit faster and more accurately.

Features hashed:
    - source_format (pdf/docx/xlsx/csv)
    - sorted set of "label tokens" — short text fragments that look
      like field labels (alphabetic, ≤ 4 words, no digits)
    - per-table column counts (sorted)
    - count of tables
"""
from __future__ import annotations

import hashlib
import re

from src.services.structural_extractor.parsing.model import ParsedDocument


__all__ = ["compute_fingerprint"]


# Recognise the LABEL portion of either:
#   - "Invoice No: 12345"  (label + value, same line)
#   - "Invoice No"          (whole line is the label, value follows on next line)
# In both cases we capture only the label part — the value is dropped.
_LABEL_HEAD_RE = re.compile(
    r"^\s*"
    r"([A-Za-z][A-Za-z\s/&\-]{1,40}?)"         # leading alphabetic phrase
    r"\s*([:#]|$)"                             # terminated by ':' / '#' / EOL
)


def _label_set(text: str) -> set[str]:
    """Pull out the LABEL words seen in the document.

    For each line, capture only the leading alphabetic phrase up to the
    first ':' or end of line — this is the label. Values are erased.
    Two docs from the same template will share the same label set even
    if their values differ.
    """
    labels = set()
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if len(line) > 80:
            continue
        m = _LABEL_HEAD_RE.match(line)
        if not m:
            continue
        token = m.group(1).strip().lower()
        if len(token) < 3:
            continue
        labels.add(token)
    return labels


def compute_fingerprint(doc: ParsedDocument) -> str:
    """Return a stable hex fingerprint for `doc`'s structural layout."""
    parts: list[str] = []

    # 1. Source format dimension
    parts.append(f"fmt={doc.source_format}")

    # 2. Page/sheet count
    parts.append(f"pages={doc.pages_or_sheets}")

    # 3. Label set (stable across runs because we sort)
    labels = sorted(_label_set(doc.full_text or ""))
    parts.append("labels=" + "|".join(labels))

    # 4. Table shape: count of tables + sorted column counts
    table_shapes = sorted(
        len(t.rows[0]) if t.rows else 0
        for t in (doc.tables or [])
    )
    parts.append(f"tables={len(doc.tables or [])}:cols={table_shapes}")

    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]
