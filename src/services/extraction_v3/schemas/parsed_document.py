from typing import Literal
from pydantic import BaseModel, Field, field_validator

BBox = tuple[float, float, float, float]

class Token(BaseModel):
    text: str
    page: int
    bbox: BBox
    font_size: float | None = None
    is_bold: bool = False

class Cell(BaseModel):
    page: int
    bbox: BBox
    text: str
    row_index: int
    col_index: int
    row_span: int = 1
    col_span: int = 1

class Table(BaseModel):
    page: int
    bbox: BBox
    rows: list[list[Cell]]
    header_row_index: int | None = None

class Region(BaseModel):
    page: int
    bbox: BBox
    role: Literal["header", "footer", "body", "address-block", "table", "logo", "signature"]
    text: str

class Page(BaseModel):
    index: int
    width: float
    height: float
    rotation: int
    regions: list[Region]
    tables: list[Table]
    tokens: list[Token]

    @field_validator("rotation")
    @classmethod
    def _rotation_multiple_of_90(cls, v: int) -> int:
        if v not in (0, 90, 180, 270):
            raise ValueError(f"rotation must be 0/90/180/270, got {v}")
        return v

class ParsedDocument(BaseModel):
    source_path: str
    file_format: Literal["pdf-native", "pdf-scanned", "docx", "image"]
    pages: list[Page]
    full_text: str
    parser_backend: str
    parser_confidence: float = Field(ge=0.0, le=1.0)
