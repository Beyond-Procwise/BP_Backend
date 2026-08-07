# tests/services/mail_intake/test_safety.py
"""The gate. Everything here fails closed: an unsure verdict quarantines.

Nothing in this module may open, execute or expand its input. The macro test
in particular must pass without the workbook ever being handed to openpyxl.
"""
import pathlib

import pytest

from src.services.mail_intake import safety, typing_
from src.services.mail_intake.states import QuarantineReason

FIXTURES = pathlib.Path("tests/fixtures/mail_intake")


def _verdict(name, declared, filename=None, ceiling=safety.DEFAULT_SIZE_CEILING_BYTES):
    content = (FIXTURES / name).read_bytes()
    return safety.evaluate(
        content,
        declared_mime=declared,
        detected_mime=typing_.detect(content),
        filename=filename or name,
        size_ceiling_bytes=ceiling,
    )


def test_clean_pdf_passes():
    assert _verdict("clean.pdf", "application/pdf").ok


def test_password_protected_pdf_quarantines():
    verdict = _verdict("encrypted.pdf", "application/pdf")
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.ENCRYPTED


def test_macro_workbook_quarantines():
    verdict = _verdict(
        "macro_workbook.xlsm",
        "application/vnd.ms-excel.sheet.macroEnabled.12",
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.MACRO


def test_password_protected_workbook_quarantines():
    verdict = _verdict(
        "encrypted_workbook.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.ENCRYPTED


def test_document_skeleton_wrapped_around_a_payload_quarantines():
    """Type detection alone is foolable, and this is the case that fools it.

    The container holds a genuine minimal Word document AND an executable, so
    it types as a Word document rather than an archive — passing a gate that
    would have quarantined the identical payload arriving as a plain .zip.
    """
    verdict = _verdict(
        "smuggling_document.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert not verdict.ok
    # EXECUTABLE, not ARCHIVE: a reviewer in the action centre needs to see
    # what was found, not merely that something was.
    assert verdict.reason is QuarantineReason.EXECUTABLE
    assert verdict.detail["flagged_kinds"], "the reviewer must learn what was found"


def test_detail_names_the_content_kind_but_never_the_senders_filename():
    """The audit record has to be useful AND safe.

    A bare count tells a reviewer nothing actionable. The member's NAME is
    sender-controlled text and would carry attacker-chosen strings into the
    record a human reads. Byte-derived content kinds are both.
    """
    verdict = _verdict(
        "smuggling_document.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    blob = repr(verdict.detail)
    assert "payload.exe" not in blob
    assert "payload" not in blob
    assert any("exec" in k or "dosexec" in k for k in verdict.detail["flagged_kinds"])


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"#!/bin/sh\nrm -rf /\n", id="shell"),
        pytest.param(b"#!/usr/bin/env python3\nimport os\n", id="python"),
        pytest.param(b"@echo off\r\ndel /f /q C:\\*\r\n", id="batch"),
        pytest.param(b"#!/usr/bin/perl\nprint 1;\n", id="perl"),
        pytest.param(b"<?php system($_GET['c']); ?>\n", id="php"),
        pytest.param(b"MZ\x90\x00" + b"\x00" * 256, id="pe-binary"),
    ],
)
def test_smuggled_script_types_are_caught(payload):
    """The executable MIME set must match what libmagic actually emits.

    An earlier version listed `text/x-python` and `application/x-bat`. Neither
    is a string libmagic produces, so a Python script and a batch file both
    passed while the set looked complete. Each case here is one libmagic
    verdict, pinned.
    """
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("_rels/.rels", "<Relationships/>")
        zf.writestr("word/document.xml", "<document/>")
        zf.writestr("extra/thing", payload)
    content = buf.getvalue()

    verdict = safety.evaluate(
        content,
        declared_mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        detected_mime=typing_.detect(content),
        filename="quote.docx",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.EXECUTABLE
    assert verdict.detail["flagged_kinds"]


def test_real_elf_binary_is_caught():
    """Synthetic stubs can sniff differently from real binaries, so use one."""
    import io
    import pathlib
    import zipfile

    elf = pathlib.Path("/bin/ls").read_bytes()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("_rels/.rels", "<Relationships/>")
        zf.writestr("word/document.xml", "<document/>")
        zf.writestr("extra/tool", elf)
    content = buf.getvalue()

    verdict = safety.evaluate(
        content,
        declared_mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        detected_mime=typing_.detect(content),
        filename="quote.docx",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.EXECUTABLE


def test_real_libreoffice_artifact_is_not_quarantined():
    """Regression for a REAL false positive.

    A genuine spreadsheet in this repo carries [trash]/*.dat — inert LibreOffice
    padding. The first version of this rule quarantined it. One wrongly held
    file in twelve is enough to make a review queue untrustworthy.
    """
    verdict = _verdict(
        "libreoffice_trash.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    assert verdict.ok, f"real workbook wrongly quarantined as {verdict.reason}"


def test_member_declaring_an_implausible_size_quarantines_without_reading_it():
    """A member inside a document declaring tens of megabytes is its own signal,
    and reading it is exactly what a zip bomb is for."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("_rels/.rels", "<Relationships/>")
        zf.writestr("xl/workbook.xml", "<workbook/>")
        zf.writestr("bulk/pad.bin", b"\x00" * (9 * 1024 * 1024))
    content = buf.getvalue()
    verdict = safety.evaluate(
        content,
        declared_mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        detected_mime=typing_.detect(content),
        filename="prices.xlsx",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.ARCHIVE
    assert verdict.detail["oversized_members"] == 1


def test_ordinary_workbook_is_not_falsely_flagged():
    """The false-positive guard for both Office heuristics.

    A macro or encryption detector that fires on ordinary workbooks would fill
    the review queue with legitimate price lists, and a queue of false
    positives is one people stop reading.
    """
    verdict = _verdict(
        "clean_workbook.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    assert verdict.ok, f"clean workbook wrongly quarantined as {verdict.reason}"


def test_zip_wearing_a_pdf_name_quarantines_on_mismatch():
    verdict = _verdict("zip_named_quote.pdf", "application/pdf")
    assert not verdict.ok
    # Both readings are defensible; the gate must report the more specific one.
    assert verdict.reason in (
        QuarantineReason.TYPE_MISMATCH,
        QuarantineReason.ARCHIVE,
    )


def test_archive_is_never_expanded():
    content = (FIXTURES / "zip_named_quote.pdf").read_bytes()
    verdict = safety.evaluate(
        content,
        declared_mime="application/zip",
        detected_mime="application/zip",
        filename="quotes.zip",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.ARCHIVE


def test_oversize_quarantines():
    verdict = _verdict("clean.pdf", "application/pdf", ceiling=10)
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.OVERSIZE


def test_executable_quarantines_regardless_of_extension():
    verdict = safety.evaluate(
        b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64,
        declared_mime="application/pdf",
        detected_mime="application/x-executable",
        filename="quote.pdf",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.EXECUTABLE


def test_shell_script_quarantines():
    verdict = safety.evaluate(
        b"#!/bin/sh\necho hi\n",
        declared_mime="text/plain",
        detected_mime="text/x-shellscript",
        filename="notes.txt",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.EXECUTABLE


def test_unknown_type_quarantines():
    verdict = safety.evaluate(
        b"\x00\x01\x02\x03",
        declared_mime="application/pdf",
        detected_mime="application/octet-stream",
        filename="mystery.pdf",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.UNKNOWN_TYPE


def test_benign_mislabel_passes():
    content = (FIXTURES / "clean.pdf").read_bytes()
    verdict = safety.evaluate(
        content,
        declared_mime="application/octet-stream",
        detected_mime="application/pdf",
        filename="quote.pdf",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert verdict.ok


def test_detail_never_carries_sender_controlled_text():
    """Aimed at a DATA-DEPENDENT detail path, deliberately.

    An earlier version of this test asserted against the encrypted-PDF path,
    whose detail is the hardcoded literal {"container": "pdf"} — an assertion
    that could never fail whatever the gate did, and which therefore missed a
    real leak. The detail is persisted to quarantine_detail and interpolated
    into a log line, so anything sender-controlled reaching it lands in the
    record a human reads.
    """
    content = (FIXTURES / "clean.pdf").read_bytes()
    marker = "MARKER-INJECTED-BY-SENDER"
    verdict = safety.evaluate(
        content,
        declared_mime=f'application/{marker}"><img src=x onerror=y>',
        detected_mime=typing_.detect(content),
        filename=f"{marker}.pdf",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert not verdict.ok  # declared/detected disagree, so this quarantines
    assert marker not in repr(verdict.detail)
    assert "onerror" not in repr(verdict.detail)


def test_payload_under_an_office_prefix_is_still_caught():
    """The Critical finding: naming decided whether classification happened.

    An earlier version selected candidates by top-level path segment, so a
    payload placed under `word/` or `xl/` was never sniffed at all — renaming
    one path segment defeated the whole control.
    """
    import io
    import pathlib
    import zipfile

    elf = pathlib.Path("/bin/ls").read_bytes()
    for location in ("word/payload.exe", "xl/media/payload.exe", "docProps/x.exe"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("_rels/.rels", "<Relationships/>")
            zf.writestr("word/document.xml", "<document/>")
            zf.writestr(location, elf)
        content = buf.getvalue()
        verdict = safety.evaluate(
            content,
            declared_mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            detected_mime=typing_.detect(content),
            filename="quote.docx",
            size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
        )
        assert not verdict.ok, location
        assert verdict.reason is QuarantineReason.EXECUTABLE, location


def test_padding_past_the_inspection_limit_fails_closed():
    """Exceeding the inspection limit must quarantine, not report silence.

    An earlier version truncated the member list and recorded nothing, so
    padding a container with decoys and putting the payload last produced a
    clean verdict.
    """
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("_rels/.rels", "<Relationships/>")
        zf.writestr("word/document.xml", "<document/>")
        for i in range(safety._MEMBER_SNIFF_LIMIT + 1):
            zf.writestr(f"pad/{i:04d}.dat", b"inert")
    content = buf.getvalue()
    verdict = safety.evaluate(
        content,
        declared_mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        detected_mime=typing_.detect(content),
        filename="quote.docx",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.ARCHIVE
    assert verdict.detail["truncated"] is True


def test_archive_nested_inside_a_document_is_caught():
    """"Archives go to a human" must not be defeated by wrapping."""
    import io
    import zipfile

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("evil.txt", "payload")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("_rels/.rels", "<Relationships/>")
        zf.writestr("word/document.xml", "<document/>")
        zf.writestr("word/inner.zip", inner.getvalue())
    content = buf.getvalue()
    verdict = safety.evaluate(
        content,
        declared_mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        detected_mime=typing_.detect(content),
        filename="quote.docx",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.ARCHIVE
    assert verdict.detail["archive_kinds"]


@pytest.mark.parametrize(
    "mime",
    [
        "application/x-msi",
        "application/vnd.ms-cab-compressed",
        "application/x-ms-shortcut",
        "application/x-iso9660-image",
    ],
)
def test_named_but_unexpected_types_do_not_clear(mime):
    """The tail allow-lists. Dodging every named hazard is not being a document.

    An MSI is an installer and a .lnk is a command launcher; both were cleared
    by the previous default-allow tail whenever the sender declared the true
    type, because neither appears in the hazard lists.
    """
    verdict = safety.evaluate(
        b"arbitrary bytes",
        declared_mime=mime,
        detected_mime=mime,
        filename="quote.docx",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.UNKNOWN_TYPE
