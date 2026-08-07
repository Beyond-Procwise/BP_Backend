"""Regenerate the committed .eml fixtures. Run from the repo root:

    ./venv/bin/python tests/fixtures/mail_intake/build_fixtures.py

The .eml files are the fixtures — this script exists so they are reproducible,
not so tests can call it. Tests read the files.
"""
import pathlib
from email.message import EmailMessage

HERE = pathlib.Path(__file__).parent


def _base(subject: str, sender: str = "quotes@copperleaf.test") -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"Copperleaf Sales <{sender}>"
    msg["To"] = "intake@acme.procureiq.io"
    msg["Date"] = "Tue, 05 Aug 2026 09:14:02 +0100"
    msg["Message-ID"] = f"<{subject.replace(' ', '-').lower()}@copperleaf.test>"
    return msg


def simple_pdf() -> bytes:
    """Acceptance test 1: one message, one PDF attachment."""
    msg = _base("Quote QT-4471")
    msg.set_content("Our quote is attached. Valid for 30 days from today.")
    msg.add_attachment(
        b"%PDF-1.4\n% minimal fixture body\n",
        maintype="application",
        subtype="pdf",
        filename="quote-QT-4471.pdf",
    )
    return msg.as_bytes()


def forwarded_two_levels() -> bytes:
    """Acceptance test 2: the quote is two nested messages down."""
    inner = _base("Quote QT-9002")
    inner.set_content("Pricing attached.")
    inner.add_attachment(
        b"%PDF-1.4\n% nested fixture body\n",
        maintype="application",
        subtype="pdf",
        filename="quote-QT-9002.pdf",
    )

    middle = _base("Fwd: Quote QT-9002", sender="buyer@acme.test")
    middle.set_content("Passing this on.")
    middle.add_attachment(inner, subtype="rfc822")

    outer = _base("Fwd: Fwd: Quote QT-9002", sender="manager@acme.test")
    outer.set_content("For the file.")
    outer.add_attachment(middle, subtype="rfc822")
    return outer.as_bytes()


def cid_signature_logo() -> bytes:
    """Acceptance test 3: an inline logo referenced by CID is not a document."""
    msg = _base("Re: pricing")
    msg.set_content("Confirming as discussed.")
    msg.add_alternative(
        '<html><body><p>Confirming as discussed.</p>'
        '<img src="cid:logo-9931@copperleaf.test"></body></html>',
        subtype="html",
    )
    html_part = msg.get_payload()[-1]
    html_part.add_related(
        b"\x89PNG\r\n\x1a\n fake logo bytes",
        maintype="image",
        subtype="png",
        cid="<logo-9931@copperleaf.test>",
        filename="logo.png",
        disposition="inline",
    )
    return msg.as_bytes()


def inline_non_cid_document() -> bytes:
    """Acceptance test 4: a real document that arrived inline and unreferenced.

    Senders do misconfigure this, and a quote occasionally arrives this way.
    """
    msg = _base("Scanned quote")
    msg.set_content("See below.")
    msg.add_alternative(
        "<html><body><p>See below.</p></body></html>", subtype="html"
    )
    html_part = msg.get_payload()[-1]
    html_part.add_related(
        b"%PDF-1.4\n% inline but unreferenced\n",
        maintype="application",
        subtype="pdf",
        filename="scan.pdf",
        disposition="inline",
    )
    return msg.as_bytes()


def body_carries_validity_window() -> bytes:
    """Acceptance test 17: the covering note holds terms the attachment lacks."""
    msg = _base("Quote QT-5510 - valid to 30 September")
    msg.set_content(
        "Pricing attached. This quote is valid until 30 September 2026 and "
        "excludes delivery to offshore sites. Payment terms 45 days net."
    )
    msg.add_attachment(
        b"%PDF-1.4\n% pricing only, no terms\n",
        maintype="application",
        subtype="pdf",
        filename="quote-QT-5510.pdf",
    )
    return msg.as_bytes()


def malformed() -> bytes:
    """Acceptance test 14: MIME that cannot be parsed into parts.

    Declares a multipart content type and a boundary, then supplies a body with
    no boundary markers at all, so there are no parts to walk.
    """
    return (
        b"From: broken@copperleaf.test\r\n"
        b"To: intake@acme.procureiq.io\r\n"
        b"Subject: Quote\r\n"
        b'Content-Type: multipart/mixed; boundary="----X"\r\n'
        b"MIME-Version: 1.0\r\n"
        b"\r\n"
        b"there is no boundary in this body at all\r\n"
    )


FIXTURES = {
    "simple_pdf.eml": simple_pdf,
    "forwarded_two_levels.eml": forwarded_two_levels,
    "cid_signature_logo.eml": cid_signature_logo,
    "inline_non_cid_document.eml": inline_non_cid_document,
    "body_validity_window.eml": body_carries_validity_window,
    "malformed.eml": malformed,
}

if __name__ == "__main__":
    for name, builder in FIXTURES.items():
        (HERE / name).write_bytes(builder())
        print("wrote", name)
