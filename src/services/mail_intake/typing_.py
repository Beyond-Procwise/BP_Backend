"""Determine content type from the bytes, never from the declared type or the
file extension.

libmagic alone is not sufficient here, and the reason matters. An OOXML file —
.docx, .xlsx, .pptx — is a ZIP archive, and libmagic correctly reports it as
application/zip. Two wrong things follow if that verdict is taken at face
value: every Word document heads for the archive quarantine, and a genuine ZIP
renamed to .pdf looks no different from a legitimate workbook. Container
inspection after magic resolves both, by asking what is actually inside the
archive.

Spec §6 stage 6.
"""
from __future__ import annotations

import io
import zipfile
from typing import Dict, FrozenSet, Optional

try:
    import magic  # type: ignore
except Exception:  # pragma: no cover - dependency is pinned; guard for import order
    magic = None  # type: ignore

UNKNOWN = "application/octet-stream"

_OOXML = {
    "word/": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xl/": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "ppt/": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

# Declared types that may disagree with the detected type without it being an
# evasion signal. Senders mislabel constantly; the alias set keeps ordinary
# sloppiness out of the review queue while a genuine mismatch still quarantines.
ALIASES: Dict[str, FrozenSet[str]] = {
    "application/pdf": frozenset({"application/pdf"}),
    "application/octet-stream": frozenset(
        {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/msword",
            "application/vnd.ms-excel",
            "text/csv",
            "text/plain",
            "image/png",
            "image/jpeg",
            "image/tiff",
        }
    ),
    "application/vnd.ms-excel": frozenset(
        {
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/csv",
            "text/plain",
        }
    ),
    "text/csv": frozenset({"text/csv", "text/plain"}),
    "text/plain": frozenset({"text/plain", "text/csv"}),
    "image/jpeg": frozenset({"image/jpeg"}),
    "image/png": frozenset({"image/png"}),
}


def _inspect_zip_container(content: bytes) -> str:
    """Distinguish an OOXML document from a genuine archive by its contents."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = zf.namelist()
    except Exception:
        return "application/zip"
    if "[Content_Types].xml" not in names:
        return "application/zip"
    for prefix, mime in _OOXML.items():
        if any(name.startswith(prefix) for name in names):
            return mime
    return "application/zip"


def detect(content: bytes) -> str:
    """Return the MIME type implied by the bytes themselves."""
    if not content:
        return UNKNOWN
    if magic is None:  # pragma: no cover - guarded for import-order safety only
        return UNKNOWN
    try:
        detected = magic.from_buffer(content, mime=True) or UNKNOWN
    except Exception:
        return UNKNOWN
    if detected == "application/zip":
        return _inspect_zip_container(content)
    return detected


def is_permitted_alias(
    declared: Optional[str], detected: str, filename: Optional[str] = None
) -> bool:
    """True when a declared/detected disagreement is sloppiness, not evasion."""
    declared_norm = (declared or UNKNOWN).split(";")[0].strip().lower()
    detected_norm = (detected or UNKNOWN).split(";")[0].strip().lower()
    if declared_norm == detected_norm:
        return True
    return detected_norm in ALIASES.get(declared_norm, frozenset())
