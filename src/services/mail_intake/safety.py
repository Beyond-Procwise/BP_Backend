# src/services/mail_intake/safety.py
"""The stage-7 gate. Fails closed on anything it cannot clear with certainty.

Three rules hold everywhere in this module:

* **Never execute, never auto-open, never auto-expand.** Encryption and macro
  detection read container structure only. A macro-bearing workbook is
  identified by the presence of a vbaProject stream, not by opening the
  workbook; an encrypted PDF by its encryption dictionary, not by decrypting
  it. An archive is quarantined for a human decision and is never expanded.
* **Order matters.** The most specific and most dangerous verdicts are
  returned first, so a macro-bearing archive-shaped file reports MACRO rather
  than the vaguer ARCHIVE.
* **Detail carries no content.** The detail dict is written to the database and
  read by humans; it holds structural facts only. Spec §9.

Spec §6 stage 7.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.services.mail_intake.states import QuarantineReason
from src.services.mail_intake.typing_ import UNKNOWN, is_permitted_alias

DEFAULT_SIZE_CEILING_BYTES = 25 * 1024 * 1024

_EXECUTABLE_MIMES = frozenset(
    {
        "application/x-executable",
        "application/x-dosexec",
        "application/x-sharedlib",
        "application/x-mach-binary",
        "application/x-msdownload",
        "application/vnd.microsoft.portable-executable",
        "text/x-shellscript",
        "text/x-python",
        "text/x-perl",
        "application/x-bat",
    }
)

_OOXML_MIMES = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
)

_ARCHIVE_MIMES = frozenset(
    {
        "application/zip",
        "application/x-rar",
        "application/vnd.rar",
        "application/x-7z-compressed",
        "application/x-tar",
        "application/gzip",
        "application/x-bzip2",
        "application/x-xz",
    }
)

# OOXML macro streams, and the legacy OLE equivalents.
_OOXML_MACRO_NAMES = ("vbaproject.bin", "vbadata.xml")

# Top-level entries a genuine Office file uses. Surveyed against real workbooks
# in this repo, which use exactly {[Content_Types].xml, _rels, docProps, xl};
# the rest are the documented equivalents for Word, PowerPoint and custom parts.
#
# This exists because type detection alone is foolable. A zip carrying a valid
# minimal document skeleton PLUS arbitrary extra members is typed as a document,
# not an archive — so it passes a gate that would have quarantined the same
# payload had it arrived as a plain .zip. Verified: a container holding
# [Content_Types].xml, word/document.xml and payload.exe types cleanly as a Word
# document. The design sends archives to human review precisely because their
# contents are unknown; a document skeleton wrapped around unknown members is
# that same unknown, wearing a better label.
_OOXML_TOP_LEVEL = frozenset(
    {
        "[content_types].xml",
        "_rels",
        "docprops",
        "customxml",
        "customui",
        "word",
        "xl",
        "ppt",
    }
)
_OLE_MACRO_MARKERS = (b"VBA", b"_VBA_PROJECT", b"Macros")

_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# An encrypted Office file is an OLE container holding a stream named
# "EncryptedPackage". OLE stores stream names as UTF-16LE, so the name is
# derived rather than hand-typed — a hand-written \x00-interleaved literal is
# easy to get subtly wrong (one missing trailing byte still matches as a prefix,
# so the mistake reads as correct and never fails a test).
_ENCRYPTED_PACKAGE = "EncryptedPackage".encode("utf-16-le")


@dataclass(frozen=True)
class SafetyVerdict:
    ok: bool
    reason: Optional[QuarantineReason] = None
    detail: Dict[str, Any] = field(default_factory=dict)


def _zip_names(content: bytes) -> Optional[List[str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            return zf.namelist()
    except Exception:
        return None


def _has_ooxml_macro(names: List[str]) -> bool:
    lowered = [n.lower() for n in names]
    return any(
        name.rsplit("/", 1)[-1] in _OOXML_MACRO_NAMES for name in lowered
    )


def _foreign_ooxml_members(names: List[str]) -> List[str]:
    """Top-level members of an OOXML container that no genuine Office file
    would carry.

    Returned only for counting. The literal names are never written to
    `detail` — a filename chosen by an untrusted sender is exactly the kind
    of content this module must not persist. See `evaluate`.
    """
    foreign = []
    for name in names:
        top = name.split("/", 1)[0].strip().lower()
        if top and top not in _OOXML_TOP_LEVEL:
            foreign.append(top)
    return sorted(set(foreign))


def _is_encrypted_ooxml(content: bytes) -> bool:
    """An encrypted Office file is an OLE container holding EncryptedPackage.

    Detected by signature and stream name, without decrypting anything.
    """
    if not content.startswith(_OLE_SIGNATURE):
        return False
    return _ENCRYPTED_PACKAGE in content[:8192]


def _is_encrypted_pdf(content: bytes) -> bool:
    try:
        import fitz
    except Exception:  # pragma: no cover - dependency is pinned
        return True  # fail closed: cannot verify means cannot clear
    try:
        with fitz.open(stream=content, filetype="pdf") as doc:
            return bool(doc.needs_pass)
    except Exception:
        return True  # unreadable PDF is not a cleared PDF


def _has_ole_macro(content: bytes) -> bool:
    if not content.startswith(_OLE_SIGNATURE):
        return False
    head = content[:65536]
    return any(marker in head for marker in _OLE_MACRO_MARKERS)


def evaluate(
    content: bytes,
    *,
    declared_mime: Optional[str],
    detected_mime: str,
    filename: Optional[str],
    size_ceiling_bytes: int = DEFAULT_SIZE_CEILING_BYTES,
) -> SafetyVerdict:
    """Clear the artifact for storage, or quarantine it with a reason."""
    size = len(content)

    if size > size_ceiling_bytes:
        return SafetyVerdict(
            False,
            QuarantineReason.OVERSIZE,
            {"size_bytes": size, "ceiling_bytes": size_ceiling_bytes},
        )

    if detected_mime in _EXECUTABLE_MIMES:
        return SafetyVerdict(
            False,
            QuarantineReason.EXECUTABLE,
            {"detected_mime": detected_mime, "declared_mime": declared_mime},
        )

    if detected_mime == UNKNOWN:
        return SafetyVerdict(
            False,
            QuarantineReason.UNKNOWN_TYPE,
            {"declared_mime": declared_mime},
        )

    names = _zip_names(content) if detected_mime.startswith("application/") else None
    if names is not None and _has_ooxml_macro(names):
        return SafetyVerdict(
            False, QuarantineReason.MACRO, {"container": "ooxml"}
        )
    if _has_ole_macro(content):
        return SafetyVerdict(
            False, QuarantineReason.MACRO, {"container": "ole"}
        )

    if names is not None and detected_mime in _OOXML_MIMES:
        foreign = _foreign_ooxml_members(names)
        if foreign:
            # A document skeleton wrapped around unknown members is still an
            # archive of unknown content, and archives go to a human. Only the
            # count is recorded — the member names themselves are sender-
            # supplied content and never belong in a database-persisted detail.
            return SafetyVerdict(
                False,
                QuarantineReason.ARCHIVE,
                {"container": "ooxml", "foreign_members": len(foreign)},
            )

    if _is_encrypted_ooxml(content):
        return SafetyVerdict(
            False, QuarantineReason.ENCRYPTED, {"container": "ooxml"}
        )
    if detected_mime == "application/pdf" and _is_encrypted_pdf(content):
        return SafetyVerdict(
            False, QuarantineReason.ENCRYPTED, {"container": "pdf"}
        )

    if not is_permitted_alias(declared_mime, detected_mime, filename):
        return SafetyVerdict(
            False,
            QuarantineReason.TYPE_MISMATCH,
            {"declared_mime": declared_mime, "detected_mime": detected_mime},
        )

    if detected_mime in _ARCHIVE_MIMES:
        # Queued for a human decision. Never expanded.
        return SafetyVerdict(
            False, QuarantineReason.ARCHIVE, {"detected_mime": detected_mime}
        )

    return SafetyVerdict(True, None, {"detected_mime": detected_mime})
