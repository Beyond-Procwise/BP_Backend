from src.services.structural_extractor.parsing.dispatcher import detect_format
from src.services.structural_extractor.parsing.pdf_parser import parse_pdf
from src.services.structural_extractor.parsing.docx_parser import parse_docx
from src.services.structural_extractor.parsing.xlsx_parser import parse_xlsx
from src.services.structural_extractor.parsing.csv_parser import parse_csv
from src.services.structural_extractor.parsing.model import ParsedDocument

_PARSERS = {
    "pdf": parse_pdf, "docx": parse_docx, "xlsx": parse_xlsx, "csv": parse_csv,
}


def parse(file_bytes: bytes, filename: str) -> ParsedDocument:
    fmt = detect_format(file_bytes, filename)
    return _PARSERS[fmt](file_bytes, filename)
