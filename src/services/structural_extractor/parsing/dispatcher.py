import os
from typing import Literal

FormatName = Literal["pdf", "docx", "xlsx", "csv"]


def detect_format(file_bytes: bytes, filename: str) -> FormatName:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf" or file_bytes[:5] == b"%PDF-":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext == ".xlsx":
        return "xlsx"
    if ext == ".csv":
        return "csv"
    raise ValueError(f"Cannot detect format for filename={filename!r}, magic={file_bytes[:8]!r}")
