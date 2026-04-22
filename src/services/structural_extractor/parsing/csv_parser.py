import csv
from io import StringIO
from src.services.structural_extractor.parsing.model import (
    ColumnRef, Token, Region, ParsedDocument, Table
)
from src.services.structural_extractor.exceptions import CsvParseError


def _is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", "").replace("£", "").replace("$", "").replace("€", ""))
        return True
    except ValueError:
        return False


def parse_csv(file_bytes: bytes, filename: str) -> ParsedDocument:
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("latin-1")
        except Exception as exc:
            raise CsvParseError(f"CSV decode failed for {filename}: {exc}") from exc

    try:
        reader = list(csv.reader(StringIO(text)))
    except Exception as exc:
        raise CsvParseError(f"CSV parse failed for {filename}: {exc}") from exc

    if not reader:
        raise CsvParseError(f"CSV {filename} is empty")

    first_row = reader[0]
    header_is_labels = all(not _is_numeric(c.strip()) for c in first_row if c.strip())
    header_row: list[str] = first_row if header_is_labels else [f"col_{i}" for i in range(len(first_row))]
    data_rows = reader[1:] if header_is_labels else reader

    tokens: list[Token] = []
    regions: list[Region] = []
    table_rows: list[list[Region]] = []
    full_text_parts: list[str] = []
    order = 0

    header_regions = [Region(tokens=[], kind="column", label=name) for name in header_row]
    table_rows.append(header_regions)

    for r_idx, row in enumerate(data_rows):
        row_regions: list[Region] = []
        for c_idx, cell in enumerate(row):
            col_name = header_row[c_idx] if c_idx < len(header_row) else f"col_{c_idx}"
            tok = Token(
                text=cell,
                anchor=ColumnRef(row=r_idx, col=c_idx, column_name=col_name),
                order=order,
            )
            order += 1
            tokens.append(tok)
            row_regions.append(Region(tokens=[tok], kind="cell"))
            full_text_parts.append(cell)
        table_rows.append(row_regions)
        regions.extend(row_regions)

    tables = [Table(rows=table_rows, header_row_index=0 if header_is_labels else None)]

    return ParsedDocument(
        source_format="csv",
        filename=filename,
        tokens=tokens,
        regions=regions,
        tables=tables,
        pages_or_sheets=1,
        full_text="\n".join(full_text_parts),
        raw_bytes=file_bytes,
    )
