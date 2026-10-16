"""MIME walking, including the two cases that decide whether a real quote is
seen at all: attachments buried in a forwarded chain, and the inline/CID
distinction that separates a signature logo from a document."""
import pathlib

import pytest

from src.services.mail_intake.mime_walk import (
    MalformedMime,
    drop_cid_referenced,
    parse,
)

FIXTURES = pathlib.Path("tests/fixtures/mail_intake")


def _load(name):
    return (FIXTURES / name).read_bytes()


def test_simple_message_yields_one_attachment_and_the_body():
    bundle = parse(_load("simple_pdf.eml"))
    assert bundle.body_text.startswith("Our quote is attached")
    assert len(bundle.parts) == 1
    part = bundle.parts[0]
    assert part.filename == "quote-QT-4471.pdf"
    assert part.declared_mime == "application/pdf"
    assert part.content.startswith(b"%PDF-1.4")
    assert part.nesting_depth == 0
    assert part.disposition == "attachment"


def test_nested_quote_two_levels_down_is_recovered_with_its_depth():
    bundle = parse(_load("forwarded_two_levels.eml"))
    pdfs = [p for p in bundle.parts if p.declared_mime == "application/pdf"]
    assert len(pdfs) == 1
    assert pdfs[0].filename == "quote-QT-9002.pdf"
    assert pdfs[0].nesting_depth == 2


def _forwarded(inner: bytes, times: int) -> bytes:
    """Wrap a message in `times` forwarding layers, as a mail client does."""
    for n in range(times):
        inner = (
            f"From: buyer{n}@procure.test\r\n"
            f"Subject: Fwd: quote (layer {n + 1})\r\n"
            'Content-Type: multipart/mixed; boundary="F{n}"\r\n\r\n'
            f"--F{n}\r\nContent-Type: text/plain\r\n\r\nSee below.\r\n"
            f"--F{n}\r\nContent-Type: message/rfc822\r\n\r\n"
        ).replace("{n}", str(n)).encode() + inner + f"\r\n--F{n}--\r\n".encode()
    return inner


def test_a_quote_forwarded_four_times_is_recovered():
    """Forwarding a quote round a procurement team four times is ordinary; the
    old default depth of 3 dropped the PDF and reported nothing."""
    bundle = parse(_forwarded(_load("simple_pdf.eml"), 4))
    pdfs = [p for p in bundle.parts if p.declared_mime == "application/pdf"]
    assert [p.filename for p in pdfs] == ["quote-QT-4471.pdf"]
    assert pdfs[0].nesting_depth == 4


def test_a_tree_deeper_than_max_depth_is_rejected_not_truncated():
    """Below the limit there is a PDF. Returning the bundle without it would be
    the partial ingestion the module forbids, with nothing to investigate."""
    # Precondition: the fixture really has the PDF two levels down.
    full = parse(_load("forwarded_two_levels.eml"))
    assert [p.nesting_depth for p in full.parts if p.declared_mime == "application/pdf"] == [2]

    with pytest.raises(MalformedMime, match="max_depth"):
        parse(_load("forwarded_two_levels.eml"), max_depth=1)


def test_cid_referenced_inline_image_is_dropped():
    bundle = parse(_load("cid_signature_logo.eml"))
    assert any(p.declared_mime == "image/png" for p in bundle.parts)
    kept = drop_cid_referenced(bundle)
    assert not any(p.declared_mime == "image/png" for p in kept)


def test_inline_part_not_referenced_by_cid_is_retained():
    bundle = parse(_load("inline_non_cid_document.eml"))
    kept = drop_cid_referenced(bundle)
    assert [p.filename for p in kept] == ["scan.pdf"]


def test_body_and_attachment_arrive_in_the_same_bundle():
    # Acceptance test 17: validity window in the note, pricing in the file.
    bundle = parse(_load("body_validity_window.eml"))
    assert "valid until 30 September 2026" in bundle.body_text
    assert [p.filename for p in bundle.parts] == ["quote-QT-5510.pdf"]


def test_malformed_mime_raises_rather_than_returning_partial_results():
    with pytest.raises(MalformedMime):
        parse(_load("malformed.eml"))


def test_headers_are_preserved_for_provenance():
    bundle = parse(_load("simple_pdf.eml"))
    assert bundle.headers["from"] == ["Copperleaf Sales <quotes@copperleaf.test>"]
    assert bundle.headers["message-id"] == ["<quote-qt-4471@copperleaf.test>"]


def test_excessive_nesting_is_rejected_before_the_parser_runs():
    """A depth limit applied after parsing is not a DoS defence.

    BytesParser builds the whole nested tree in one eager call, so the cost is
    already paid by the time max_depth is consulted. This must be rejected on a
    raw scan instead — and fast enough that the rejection itself is not the DoS.
    """
    import time

    hostile = (
        b"From: attacker@example.invalid\r\n"
        b"Subject: Quote\r\n"
        b'Content-Type: multipart/mixed; boundary="X"\r\n\r\n'
        + b'--X\r\nContent-Type: message/rfc822\r\n\r\n' * 200
        + b"--X--\r\n"
    )
    started = time.monotonic()
    with pytest.raises(MalformedMime, match="nested messages"):
        parse(hostile)
    assert time.monotonic() - started < 1.0


def test_oversize_raw_message_is_rejected():
    with pytest.raises(MalformedMime, match="exceeds"):
        parse(b"From: a@b.test\r\n\r\n" + b"x" * (25 * 1024 * 1024 + 1))


def test_cosmetic_defect_does_not_reject_recoverable_mail():
    """A stray mbox 'From ' envelope line is a normal artifact of mailing-list
    and migration tooling. The mail decodes perfectly; quarantining it would
    lose a real supplier quote for a blemish that cost nothing."""
    recoverable = b"From attacker Mon Aug  5 09:00:00 2026\r\n" + _load("simple_pdf.eml")
    bundle = parse(recoverable)
    assert [p.filename for p in bundle.parts] == ["quote-QT-4471.pdf"]
    assert bundle.body_text.startswith("Our quote is attached")


def test_structural_corruption_nested_inside_a_forward_is_rejected():
    """The failure mode a top-level-only defect check misses entirely.

    The outer message is well formed, so its own defect list is empty. Without
    a recursive check this returns a "successful" bundle holding the covering
    note and no attachments — silently dropping the quote, which is exactly the
    partial ingestion the design forbids.
    """
    from email import policy as _policy
    from email.message import EmailMessage
    from email.parser import BytesParser as _BytesParser

    # Attaching the malformed bytes directly (maintype="message",
    # subtype="rfc822") routes through the generic binary content manager,
    # which base64-encodes the payload — that is not how message/rfc822 is
    # supposed to travel (RFC 2046 §5.2.1 requires 7bit/8bit/binary), and on
    # reparse it no longer resembles a nested message at all. Parsing the
    # malformed bytes into an EmailMessage first and attaching *that* routes
    # through set_message_content, which embeds it as literal 8bit text and
    # genuinely preserves the corrupted structure one level down.
    corrupted = _BytesParser(policy=_policy.default).parsebytes(
        (FIXTURES / "malformed.eml").read_bytes()
    )

    outer = EmailMessage()
    outer["From"] = "Buyer <buyer@acme.test>"
    outer["To"] = "intake@acme.procureiq.io"
    outer["Subject"] = "Fwd: Quote"
    outer["Message-ID"] = "<fwd-corrupt@acme.test>"
    outer.set_content("Passing this on.")
    outer.add_attachment(
        corrupted,
        subtype="rfc822",
        filename="forwarded.eml",
    )
    with pytest.raises(MalformedMime):
        parse(outer.as_bytes())
