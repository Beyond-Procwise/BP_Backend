from dataclasses import dataclass
from typing import Literal, Optional, Union


@dataclass(frozen=True)
class BBox:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class CellRef:
    sheet: str
    row: int
    col: int
    merged_range: Optional[str] = None


@dataclass(frozen=True)
class ColumnRef:
    row: int
    col: int
    column_name: str


@dataclass(frozen=True)
class NodeRef:
    kind: Literal["paragraph", "table_cell"]
    paragraph_index: Optional[int] = None
    table_index: Optional[int] = None
    row: Optional[int] = None
    col: Optional[int] = None


AnchorRef = Union[BBox, CellRef, ColumnRef, NodeRef]


@dataclass(frozen=True)
class Token:
    text: str
    anchor: AnchorRef
    block_no: Optional[int] = None
    line_no: Optional[int] = None
    order: int = 0


@dataclass
class Region:
    tokens: list[Token]
    kind: Literal["paragraph", "cell", "block", "row", "column"]
    label: Optional[str] = None


@dataclass
class Table:
    rows: list[list[Region]]
    header_row_index: Optional[int] = None
    source_anchor: Optional[AnchorRef] = None


@dataclass
class ParsedDocument:
    source_format: Literal["pdf", "docx", "xlsx", "csv"]
    filename: str
    tokens: list[Token]
    regions: list[Region]
    tables: list[Table]
    pages_or_sheets: int
    full_text: str
    raw_bytes: bytes
