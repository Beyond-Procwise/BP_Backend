"""Shared vendor-key derivation.

The proposer/telemetry (producers) and context_layer (consumer) MUST derive the
same vendor key from a file path, or an approved hint scoped to a vendor would
never match at extraction time. This is the single source of truth; it is the
logic that previously lived in telemetry_service._vendor_hint.
"""
from __future__ import annotations

import os
import re

# Leading supplier token of the filename, up to an INV/PO/QUT/QUOTE marker —
# a coarse layout/vendor pattern key (e.g. "NEXASPARK INV4759276 ..." -> "NEXASPARK",
# "GOMEZ, GOOD ETC QUT104683 ..." -> "GOMEZ, GOOD ETC").
_VENDOR = re.compile(r"^([A-Za-z][A-Za-z&,\.\s]{1,40}?)\s+(?:INV|PO|QUT|QTE|QUOTE)", re.I)


def vendor_key(file_path: str | None) -> str | None:
    """Normalized vendor token from a document file path/name, or None."""
    base = os.path.basename(file_path or "")
    m = _VENDOR.match(base)
    if m:
        return m.group(1).strip().rstrip(",").strip()
    return base.split()[0] if base.split() else None
