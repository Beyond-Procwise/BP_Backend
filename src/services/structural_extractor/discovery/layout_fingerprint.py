import hashlib

from src.services.structural_extractor.parsing.model import (
    ParsedDocument, BBox, CellRef, ColumnRef, NodeRef
)


def layout_signature(doc: ParsedDocument) -> str:
    buckets: list[str] = []
    if doc.source_format == "pdf":
        for t in doc.tokens:
            if isinstance(t.anchor, BBox):
                bx = int(t.anchor.x0 / 60)
                by = int(t.anchor.y0 / 80)
                buckets.append(f"p{t.anchor.page}:{bx},{by}")
    elif doc.source_format == "xlsx":
        for t in doc.tokens:
            if isinstance(t.anchor, CellRef):
                buckets.append(f"s:{t.anchor.sheet}:r{t.anchor.row}:c{t.anchor.col}")
    elif doc.source_format == "csv":
        for tbl in doc.tables:
            if tbl.header_row_index is not None:
                header = tbl.rows[tbl.header_row_index]
                buckets.extend(r.label or "" for r in header)
    elif doc.source_format == "docx":
        for t in doc.tokens:
            if isinstance(t.anchor, NodeRef):
                buckets.append(f"{t.anchor.kind}:{t.anchor.paragraph_index or t.anchor.table_index}")
    joined = "|".join(sorted(set(buckets)))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]
