"""Primary-key normalization for extracted document IDs.

Without normalization, the same physical document can produce different
``invoice_id`` values across runs depending on which extraction path
populated it:

  - Structural extractor:  ``148769``     (digits-only)
  - Filename rescue:       ``INV148769``  (with leading prefix)

Each persists a SEPARATE row in ``proc.bp_invoice``, fragmenting the
record. :func:`normalize_pk` collapses these into a single canonical
form by stripping a recognised type-prefix when the digits-only form is
itself a plausible identifier (4+ digits) and the prefix is one of the
known document-type tokens.

Behaviour:

  - ``"INV148769"`` → ``"148769"``       (type-prefix stripped)
  - ``"PO526689"``  → ``"526689"``
  - ``"QUT-25-032"``→ ``"QUT-25-032"``   (mixed digits/separators kept)
  - ``"148769"``    → ``"148769"``       (already canonical)
  - ``"DHA-2025-143"`` → ``"DHA-2025-143"`` (vendor-token, not a known prefix)

The known prefixes are ``INV``, ``INVOICE``, ``PO``, ``ORDER``, ``QUOTE``,
``QUT``, ``Q`` — case-insensitive. Other prefixes are left intact so we
don't accidentally strip vendor-meaningful tokens.
"""
from __future__ import annotations

import re

__all__ = ["normalize_pk"]


# Recognised type-prefixes per doc_type. Order matters: longest first so
# "INVOICE" matches before "INV".
_PREFIXES = {
    "Invoice": ("INVOICE", "INV"),
    "Purchase_Order": ("ORDER", "PO"),
    "Quote": ("QUOTE", "QUT", "Q"),
    "Contract": ("CONTRACT",),
}

_DIGITS_RE = re.compile(r"^\d+$")


def normalize_pk(value: str, doc_type: str) -> str:
    """Return a canonical primary-key string for ``value``.

    Trims whitespace and strips a recognised type-prefix when:
      - the prefix matches one of the doc-type's known tokens
      - the remainder (after optionally stripping a separator) is
        purely digits and ≥ 4 characters

    Otherwise returns the trimmed value unchanged. Empty / None input
    returns an empty string.
    """
    if not value:
        return ""
    trimmed = str(value).strip()
    if not trimmed:
        return ""
    prefixes = _PREFIXES.get(doc_type, ())
    upper = trimmed.upper()
    for prefix in prefixes:
        if upper.startswith(prefix):
            tail = trimmed[len(prefix):]
            # Allow an optional separator between prefix and digits
            if tail and tail[0] in "-_ /":
                tail = tail[1:]
            if _DIGITS_RE.match(tail) and len(tail) >= 4:
                return tail
    return trimmed
