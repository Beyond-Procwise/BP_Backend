from src.services.structural_extractor.exceptions import (
    FormatParseError, PDFParseError, DocxParseError, XlsxParseError, CsvParseError,
    DerivationError, UnresolvedFieldError
)


def test_hierarchy():
    assert issubclass(PDFParseError, FormatParseError)
    assert issubclass(DocxParseError, FormatParseError)
    assert issubclass(XlsxParseError, FormatParseError)
    assert issubclass(CsvParseError, FormatParseError)


def test_unresolved_field_error_has_fields():
    e = UnresolvedFieldError("invoice_date", 3)
    assert e.field_name == "invoice_date"
    assert e.attempt == 3
