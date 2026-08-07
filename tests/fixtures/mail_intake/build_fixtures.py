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


def _encrypted_pdf() -> bytes:
    """Acceptance test 7: a password-protected PDF."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Confidential pricing", fontsize=11)
    return doc.write(
        encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user"
    )


def _macro_workbook() -> bytes:
    """Acceptance test 8: a macro-bearing workbook (.xlsm shape)."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("xl/workbook.xml", "<workbook/>")
        zf.writestr("xl/vbaProject.bin", b"\xd0\xcf\x11\xe0 fake vba")
    return buf.getvalue()


def _encrypted_ooxml() -> bytes:
    """A password-protected Office file, as a byte signature.

    A genuinely encrypted workbook cannot be produced here — neither
    msoffcrypto nor oletools is installed. That is acceptable because the
    detector is itself a byte-signature check: an encrypted OOXML file is an
    OLE container carrying a stream named "EncryptedPackage". This fixture
    reproduces exactly the structure the detector inspects, and the companion
    fixture below proves an ordinary workbook does not trip it.
    """
    ole_signature = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    return (
        ole_signature
        + b"\x00" * 512
        + "EncryptedPackage".encode("utf-16-le")
        + b"\x00" * 64
    )


def _clean_workbook() -> bytes:
    """An ordinary .xlsx: no macros, no encryption, no foreign members.

    Member layout mirrors real workbooks surveyed in this repo, which carry
    exactly [Content_Types].xml, _rels, docProps and xl. Using the realistic
    shape matters: a stripped-down fixture would pass the foreign-member check
    without ever exercising it.
    """
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("_rels/.rels", "<Relationships/>")
        zf.writestr("docProps/app.xml", "<Properties/>")
        zf.writestr("docProps/core.xml", "<coreProperties/>")
        zf.writestr("xl/workbook.xml", "<workbook/>")
        zf.writestr("xl/worksheets/sheet1.xml", "<worksheet/>")
    return buf.getvalue()


def _smuggling_workbook() -> bytes:
    """A valid document skeleton wrapped around an unrelated payload.

    Type detection alone calls this a Word document, because it genuinely
    contains one — so it passes a gate that would have quarantined the same
    payload had it arrived as a plain .zip.
    """
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("_rels/.rels", "<Relationships/>")
        zf.writestr("word/document.xml", "<document/>")
        zf.writestr("payload.exe", b"MZ\x90\x00" + b"\x00" * 256)
        zf.writestr("secret/notes.txt", "not part of any Office format")
    return buf.getvalue()


def _zip_wearing_a_pdf_name() -> bytes:
    """Acceptance test 6: ZIP payload, .pdf extension."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("payload.txt", "definitely not a pdf")
    return buf.getvalue()


def _clean_pdf() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Unit price 12.50 GBP", fontsize=11)
    return doc.write()


BINARY_FIXTURES = {
    "encrypted.pdf": _encrypted_pdf,
    "encrypted_workbook.xlsx": _encrypted_ooxml,
    "macro_workbook.xlsm": _macro_workbook,
    "clean_workbook.xlsx": _clean_workbook,
    "smuggling_document.docx": _smuggling_workbook,
    "zip_named_quote.pdf": _zip_wearing_a_pdf_name,
    "clean.pdf": _clean_pdf,
}

if __name__ == "__main__":
    for name, builder in FIXTURES.items():
        (HERE / name).write_bytes(builder())
        print("wrote", name)
    for name, builder in BINARY_FIXTURES.items():
        (HERE / name).write_bytes(builder())
        print("wrote", name)
