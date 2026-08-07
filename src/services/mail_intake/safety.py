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
from src.services.mail_intake.typing_ import UNKNOWN, detect, is_permitted_alias

DEFAULT_SIZE_CEILING_BYTES = 25 * 1024 * 1024

# Bounds on member classification. Every number here caps what hostile input can
# make us do. The member limit sits well above the largest real workbook measured
# in this repo (42 members), so legitimate files are never truncated — and
# exceeding it fails closed rather than quietly inspecting a prefix.
_MEMBER_SNIFF_BYTES = 512
_MEMBER_SNIFF_LIMIT = 512
_MEMBER_MAX_BYTES = 8 * 1024 * 1024

# Detected types a document may legitimately be. The tail of evaluate() clears
# only these; anything else libmagic can name — an MSI installer, a .lnk command
# launcher, a CAB, a disk image — quarantines. Allow-listing is the difference
# between "not known to be dangerous" and "known to be acceptable", and only the
# second is a safe basis for handing bytes onward.
_PERMITTED_DOCUMENT_MIMES = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/msword",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
        "application/rtf",
        "text/csv",
        "text/plain",
        "text/html",
        "text/xml",
        "image/png",
        "image/jpeg",
        "image/tiff",
        "image/gif",
        "image/bmp",
        "image/webp",
    }
)

# Determined by running libmagic against real samples, not from memory. Two
# entries in an earlier version of this set were simply wrong — `text/x-python`
# and `application/x-bat` are not strings libmagic emits, so a Python script and
# a Windows batch file both sailed through while the set looked complete.
#
# KNOWN LIMIT, deliberately not papered over: VBScript, PowerShell, and Python
# without a shebang all sniff as `text/plain`, because that is what they are.
# Content sniffing cannot separate those from a legitimate text attachment, and
# adding `text/plain` here would quarantine every covering note. The cases that
# matter most — real binaries, shell scripts, batch files — are caught. A
# text-shaped script smuggled inside a container is inert unless someone
# extracts and runs it, which is a different control from this one.
_EXECUTABLE_MIMES = frozenset(
    {
        # Native binaries
        "application/x-executable",
        "application/x-sharedlib",          # real ELF binaries report this
        "application/x-dosexec",
        "application/x-mach-binary",
        "application/x-msdownload",
        "application/vnd.microsoft.portable-executable",
        "application/x-java-applet",        # compiled .class
        # Scripts
        "text/x-shellscript",               # sh, bash
        "text/x-script.python",             # NOT text/x-python
        "text/x-msdos-batch",               # .bat and .cmd; NOT application/x-bat
        "text/x-perl",
        "text/x-ruby",
        "text/x-php",
        "application/javascript",
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


def _inspect_members(content: bytes) -> Dict[str, Any]:
    """Classify EVERY member of a container by what it is. Names decide nothing.

    Two earlier versions of this check were defeated by the same mistake in
    different clothes. The first quarantined on the mere presence of a member
    outside the known Office namespaces, and wrongly held a real spreadsheet
    carrying LibreOffice's inert `[trash]/*.dat` padding. The second sniffed
    content — but still used the member's top-level NAME to decide which
    members were worth sniffing, so a payload at `word/payload.exe` was never
    examined at all. Renaming one path segment defeated the entire control.

    So no name is consulted here. Every member is sniffed, wherever it sits.
    Legitimate Office parts are XML, text and images; none of those are
    executables or archives, so inspecting all of them costs no false
    positives — measured across every real workbook in this repo, whose members
    sniff exclusively as text/xml, text/plain and text/html.

    On "never auto-expand": this reads a bounded prefix — at most
    _MEMBER_SNIFF_BYTES from each of at most _MEMBER_SNIFF_LIMIT members, and
    nothing at all from a member declaring an implausible size. That is
    classification, the same thing detect() does for the outer file, not
    extraction: nothing is written, executed, or handed onward. Reading a
    bounded length also bounds decompression, so a member that lies about its
    size cannot become a decompression bomb.

    KNOWN LIMIT, stated rather than implied: a payload placed beyond the first
    _MEMBER_SNIFF_BYTES of a member will not be recognised. Sniffing further
    only moves the boundary, since a payload can sit at any offset. The detail
    therefore records how much was examined, so a reviewer can tell a
    fully-classified member from a partially-classified one rather than reading
    silence as safety. What makes the residue tolerable is that such a payload
    cannot execute here — extraction parses the container as a document and
    ignores foreign parts — and the bytes are retained for audit either way.

    Returns a summary safe to persist and log: counts and byte-derived content
    kinds. Member NAMES never appear — they are sender-controlled strings and
    would carry attacker-chosen text into the record a human reads.
    """
    summary: Dict[str, Any] = {
        "member_count": 0,
        "inspected_members": 0,
        "sniffed_prefix_bytes": _MEMBER_SNIFF_BYTES,
        "flagged_kinds": [],
        "archive_kinds": [],
        "oversized_members": 0,
        "truncated": False,
        "unreadable": False,
    }

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            summary["member_count"] = len(infos)
            if len(infos) > _MEMBER_SNIFF_LIMIT:
                # More members than we are willing to inspect is precisely the
                # "cannot clear with certainty" condition, so it fails closed
                # rather than inspecting a prefix and reporting silence.
                summary["truncated"] = True
                return summary

            executables, archives = set(), set()
            for info in infos:
                summary["inspected_members"] += 1
                if info.file_size > _MEMBER_MAX_BYTES:
                    summary["oversized_members"] += 1
                    continue
                with zf.open(info) as handle:
                    head = handle.read(_MEMBER_SNIFF_BYTES)
                kind = detect(head)
                if kind in _EXECUTABLE_MIMES:
                    executables.add(kind)
                elif kind in _ARCHIVE_MIMES:
                    # A container nested inside a container is still unknown
                    # content, and unknown content goes to a person.
                    archives.add(kind)
    except Exception:
        summary["unreadable"] = True
        return summary

    summary["flagged_kinds"] = sorted(executables)
    summary["archive_kinds"] = sorted(archives)
    return summary


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

    # `declared_mime` is the sender's Content-Type header and never appears in
    # a detail dict. The header survives MIME parsing largely intact — a value
    # like `application/"><img src=x onerror=y>` comes through — and the detail
    # is persisted to quarantine_detail AND interpolated into a log line, so
    # recording it verbatim would carry attacker-chosen text straight into the
    # record a human reads. What a reviewer actually needs is the derived fact.
    declared_agrees = is_permitted_alias(declared_mime, detected_mime, filename)

    if detected_mime in _EXECUTABLE_MIMES:
        return SafetyVerdict(
            False,
            QuarantineReason.EXECUTABLE,
            {"detected_mime": detected_mime, "declared_agrees": declared_agrees},
        )

    if detected_mime == UNKNOWN:
        return SafetyVerdict(
            False,
            QuarantineReason.UNKNOWN_TYPE,
            {"declared_agrees": declared_agrees},
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
        summary = _inspect_members(content)
        if summary["flagged_kinds"]:
            # Executable content smuggled inside a well-formed document.
            # Reported as EXECUTABLE rather than ARCHIVE because a reviewer in
            # the action centre needs to see what was found, not merely that
            # something was.
            return SafetyVerdict(
                False, QuarantineReason.EXECUTABLE, dict(summary, container="ooxml")
            )
        if (
            summary["archive_kinds"]
            or summary["oversized_members"]
            or summary["truncated"]
            or summary["unreadable"]
        ):
            # Each of these is a "cannot clear with certainty": a container
            # nested inside a container, a member too large to inspect, more
            # members than we are willing to inspect, or a container we could
            # not read at all.
            return SafetyVerdict(
                False, QuarantineReason.ARCHIVE, dict(summary, container="ooxml")
            )

    if _is_encrypted_ooxml(content):
        return SafetyVerdict(
            False, QuarantineReason.ENCRYPTED, {"container": "ooxml"}
        )
    if detected_mime == "application/pdf" and _is_encrypted_pdf(content):
        return SafetyVerdict(
            False, QuarantineReason.ENCRYPTED, {"container": "pdf"}
        )

    if not declared_agrees:
        return SafetyVerdict(
            False,
            QuarantineReason.TYPE_MISMATCH,
            {"detected_mime": detected_mime, "declared_agrees": False},
        )

    if detected_mime in _ARCHIVE_MIMES:
        # Queued for a human decision. Never expanded.
        return SafetyVerdict(
            False, QuarantineReason.ARCHIVE, {"detected_mime": detected_mime}
        )

    # The tail allow-lists rather than default-allows. Reaching here having
    # merely dodged every named hazard is not the same as being a document:
    # an MSI installer, a .lnk command launcher, a CAB or a disk image are all
    # types libmagic names happily, none appear in the hazard lists above, and
    # each would previously have cleared whenever the sender declared its true
    # type. "Not known to be dangerous" is not a safe basis for handing bytes
    # to the extractor; "known to be acceptable" is.
    if detected_mime not in _PERMITTED_DOCUMENT_MIMES:
        return SafetyVerdict(
            False,
            QuarantineReason.UNKNOWN_TYPE,
            {"detected_mime": detected_mime, "reason_detail": "not_a_permitted_document"},
        )

    return SafetyVerdict(True, None, {"detected_mime": detected_mime})
