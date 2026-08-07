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


def _has_unsafe_entry_names(names) -> bool:
    """Absolute paths or parent-directory segments in member names.

    No genuine Office file carries these (verified across every real workbook
    in this repo). Their presence is a deliberate construction, and a container
    holding one should never be granted a document verdict.
    """
    for name in names:
        if name.startswith("/") or name.startswith("\\"):
            return True
        if ".." in name.replace("\\", "/").split("/"):
            return True
    return False


def _inspect_zip_container(content: bytes) -> str:
    """Distinguish an OOXML document from a genuine archive by its contents.

    The test is deliberately strict, because the verdict decides which gate
    rule applies downstream: archives go to human review, documents do not.
    A container that merely *contains* a document-shaped entry is not a
    document — a zip holding `[Content_Types].xml`, `word/document.xml` and an
    arbitrary payload was previously typed as a Word document and passed a gate
    that would have quarantined the identical payload as a plain .zip.

    So an OOXML verdict requires the package to be well formed: the content
    types part, the mandatory OPC relationships part, a recognised content
    directory, and no member names that traverse outside the package. Every
    real workbook in this repo satisfies all four; a hand-built shell does not.
    Anything short of that is reported as the archive it actually is, and the
    safety gate holds it for a person.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = zf.namelist()
    except Exception:
        return "application/zip"

    lookup = set(names)
    if "[Content_Types].xml" not in lookup:
        return "application/zip"
    # _rels/.rels is mandatory in the Open Packaging Conventions; a package
    # without it is not a valid Office document whatever else it contains.
    if "_rels/.rels" not in lookup:
        return "application/zip"
    if _has_unsafe_entry_names(names):
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
    # Both bare ZIPs and OOXML files (which are ZIP-based) need container
    # inspection to validate well-formedness and check for path traversal.
    if detected == "application/zip" or detected in _OOXML.values():
        return _inspect_zip_container(content)
    return detected


def is_permitted_alias(
    declared: Optional[str], detected: str, filename: Optional[str] = None
) -> bool:
    """True when a declared/detected disagreement is sloppiness, not evasion.

    `filename` is accepted for call-site symmetry and is deliberately IGNORED.
    The extension comes from the sender and is exactly as untrustworthy as the
    declared type; letting it influence the verdict would reintroduce the thing
    this module exists to remove.
    """
    declared_norm = (declared or UNKNOWN).split(";")[0].strip().lower()
    detected_norm = (detected or UNKNOWN).split(";")[0].strip().lower()
    if declared_norm == detected_norm:
        return True
    return detected_norm in ALIASES.get(declared_norm, frozenset())
