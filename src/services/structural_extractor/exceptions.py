class FormatParseError(Exception):
    pass


class PDFParseError(FormatParseError):
    pass


class DocxParseError(FormatParseError):
    pass


class XlsxParseError(FormatParseError):
    pass


class CsvParseError(FormatParseError):
    pass


class DerivationError(Exception):
    pass


class UnresolvedFieldError(Exception):
    def __init__(self, field_name: str, attempt: int):
        self.field_name = field_name
        self.attempt = attempt
        super().__init__(f"{field_name} unresolved at attempt {attempt}")
