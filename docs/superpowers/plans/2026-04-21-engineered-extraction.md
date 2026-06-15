# Engineered Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LLM-primary extraction pipeline with an engineered-first extractor across PDF / DOCX / XLSX / CSV, with type-driven anchor discovery, pattern learning, layered NLU fallback, deterministic derivation of schema-required fields, and full provenance tracking. 100% field-level accuracy on the golden corpus before rollout.

**Architecture:** Four layers — format adapters (produce unified `ParsedDocument`) → type-driven anchor discovery (no hardcoded label vocab) → retry loop with progressive NLU escalation → pattern learning (extends `proc.bp_extraction_patterns`). A Derivation Registry fills schema columns not directly present in the source via deterministic rules (e.g. `due_date = invoice_date + 90 days`). Every stored value has provenance in `proc.bp_extraction_provenance`. Spec: `docs/superpowers/specs/2026-04-21-engineered-extraction-design.md`.

**Tech Stack:** Python 3.12, PyMuPDF, pdfplumber, python-docx, openpyxl, pandas, transformers (BERT-NER, Table-Transformer, Layout-YOLO on CPU), psycopg2, pytest, Ollama (AgentNick LLM fallback on GPU).

**Precondition:** Work in a dedicated git branch (`feat/structural-extractor`) off `cb6add4` (last-good commit). All uncommitted changes are currently in `stash@{0}` — leave them stashed.

## Execution order (IMPORTANT — overrides naive top-to-bottom reading)

The plan is organized by phase for readability. But execution order must respect dependencies flagged in review:

1. **Phases 1-3 (Tasks 1-15)** — execute in order.
2. **Task 40e (ranking formula)** must execute BEFORE Tasks 16-19 use it. Move its execution into Phase 3 between Task 15 and Task 16. Tasks 16-19's ad-hoc scoring stays as written (fallback), but 40e replaces it during ranking calls.
3. **Task 21 is replaced by Tasks 51a/51b/51c.** Task 21 becomes "skipped — superseded by 51a/51b/51c". Its stub `...` branches in `_xlsx_line_items`/`_docx_line_items`/`_csv_line_items` are NOT written in Task 20 — only `_pdf_line_items` is.
4. **Task 40d (`persist_pattern`) must execute BEFORE Task 38i** (which calls it). Pull Task 40d earlier: after Task 40a/b/c are done, execute 40d, THEN begin Phase 9 (Task 38a onward).
5. **Task 56 (golden-set fixtures) must execute BEFORE Task 46** (which loads them). Task 56 is the FIRST task in Phase 14 (before 45, 46, 47, 48). Rename its placement accordingly — it's listed under "Phase 15" only because it was added in the plan-review round; conceptually it belongs to Phase 14.
6. **Provenance wiring (new Task 57, see below)** executes before Task 43.

The dependency-respecting execution order is therefore:
```
Tasks 1-15 → Task 40e → Tasks 16-19 → Task 20 → Tasks 51a/b/c → Task 22 →
Tasks 23-31 (derivation — Phase 5) → Tasks 52a/b/c (validation split) →
Tasks 33-36 (NLU) → Tasks 53a/b/c (LLM split) →
Task 38a-g, h → Tasks 39, 40a, 40b, 40c, 40d → Task 38i →
Task 41 → Tasks 42a/b/c → Task 43 → Task 57 (provenance wiring) →
Task 49 (incoterm/delivery) → Task 50 (warm) →
Tasks 54a/b/c, 55a/b/c (derivation/lookup split, retrofitted) →
Task 44 (orchestrator) → Task 56 → Task 46 → Task 47 → Task 48
```


---

## Phase 1 — Foundations & data model

### Task 1: Create module skeleton with empty files

**Files:**
- Create: `src/services/structural_extractor/__init__.py`
- Create: `src/services/structural_extractor/parsing/__init__.py`
- Create: `src/services/structural_extractor/parsing/model.py`
- Create: `src/services/structural_extractor/discovery/__init__.py`
- Create: `src/services/structural_extractor/extractors/__init__.py`
- Create: `src/services/structural_extractor/nlu/__init__.py`
- Create: `tests/structural_extractor/__init__.py`
- Create: `tests/structural_extractor/conftest.py` (empty)

- [ ] **Step 1: Create directory tree and empty `__init__.py` files**

```bash
mkdir -p src/services/structural_extractor/{parsing,discovery,extractors,nlu}
mkdir -p tests/structural_extractor
touch src/services/structural_extractor/__init__.py
touch src/services/structural_extractor/parsing/__init__.py
touch src/services/structural_extractor/discovery/__init__.py
touch src/services/structural_extractor/extractors/__init__.py
touch src/services/structural_extractor/nlu/__init__.py
touch tests/structural_extractor/__init__.py
touch tests/structural_extractor/conftest.py
```

- [ ] **Step 2: Commit the skeleton**

```bash
git add src/services/structural_extractor tests/structural_extractor
git commit -m "feat(extraction): structural_extractor module skeleton"
```

---

### Task 2: Data model — AnchorRef union and format-specific types

**Files:**
- Modify: `src/services/structural_extractor/parsing/model.py`
- Test: `tests/structural_extractor/test_model.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/structural_extractor/test_model.py
import pytest
from src.services.structural_extractor.parsing.model import (
    BBox, CellRef, ColumnRef, NodeRef, Token, Region, Table, ParsedDocument
)

def test_bbox_frozen():
    b = BBox(page=1, x0=0.0, y0=0.0, x1=10.0, y1=10.0)
    with pytest.raises(AttributeError):
        b.x0 = 1.0  # frozen

def test_cellref_requires_sheet_row_col():
    c = CellRef(sheet="Sheet1", row=1, col=2)
    assert c.sheet == "Sheet1" and c.row == 1 and c.col == 2 and c.merged_range is None

def test_columnref_defaults():
    c = ColumnRef(row=0, col=0, column_name="Invoice No")
    assert c.row == 0 and c.col == 0 and c.column_name == "Invoice No"

def test_noderef_paragraph_vs_table():
    p = NodeRef(kind="paragraph", paragraph_index=3)
    t = NodeRef(kind="table_cell", table_index=0, row=1, col=2)
    assert p.kind == "paragraph" and t.kind == "table_cell"

def test_token_has_anchor():
    t = Token(text="Invoice", anchor=BBox(1, 0, 0, 50, 20), order=0)
    assert t.text == "Invoice"
    assert isinstance(t.anchor, BBox)

def test_parsed_document_shape():
    d = ParsedDocument(
        source_format="pdf", filename="x.pdf", tokens=[], regions=[],
        tables=[], pages_or_sheets=1, full_text="", raw_bytes=b"",
    )
    assert d.source_format == "pdf"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/structural_extractor/test_model.py -v`
Expected: FAIL (module/classes don't exist)

- [ ] **Step 3: Implement `parsing/model.py`**

```python
# src/services/structural_extractor/parsing/model.py
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `.venv/bin/pytest tests/structural_extractor/test_model.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/services/structural_extractor/parsing/model.py tests/structural_extractor/test_model.py
git commit -m "feat(extraction): ParsedDocument and AnchorRef types"
```

---

### Task 3: `ExtractedValue` dataclass with provenance

**Files:**
- Create: `src/services/structural_extractor/types.py`
- Test: `tests/structural_extractor/test_types.py`

- [ ] **Step 1: Write failing test**

```python
# tests/structural_extractor/test_types.py
from src.services.structural_extractor.types import ExtractedValue
from src.services.structural_extractor.parsing.model import BBox

def test_extracted_value_extracted_provenance():
    ev = ExtractedValue(
        value=8333.0, provenance="extracted",
        anchor_text="Subtotal £8,333",
        anchor_ref=BBox(1, 100, 200, 300, 220),
        source="structural", confidence=1.0, attempt=1,
    )
    assert ev.provenance == "extracted"
    assert ev.derivation_trace is None

def test_extracted_value_derived_provenance():
    ev = ExtractedValue(
        value="2019-11-20", provenance="derived",
        derivation_trace={"rule_id": "due_date_default", "inputs": {"invoice_date": "2019-08-22"}},
        source="derivation_registry", confidence=1.0, attempt=1,
    )
    assert ev.provenance == "derived"
    assert ev.anchor_ref is None
    assert ev.derivation_trace["rule_id"] == "due_date_default"
```

- [ ] **Step 2: Run — fails (module doesn't exist)**

- [ ] **Step 3: Implement**

```python
# src/services/structural_extractor/types.py
from dataclasses import dataclass
from typing import Any, Literal, Optional
from src.services.structural_extractor.parsing.model import AnchorRef

@dataclass
class ExtractedValue:
    value: Any
    provenance: Literal["extracted", "derived", "inferred", "lookup"]
    anchor_text: Optional[str] = None
    anchor_ref: Optional[AnchorRef] = None
    derivation_trace: Optional[dict] = None
    confidence: float = 1.0
    source: Literal[
        "structural", "pattern_cached", "nlu_ner", "nlu_table", "nlu_layout",
        "llm_fallback", "derivation_registry", "lookup_api", "lookup_db"
    ] = "structural"
    attempt: int = 1

@dataclass
class ExtractionResult:
    header: dict[str, ExtractedValue]
    line_items: list[dict[str, ExtractedValue]]
    parsed_text: str
    unresolved_fields: list[str]
    attempts: int
    pattern_id_used: Optional[int] = None
    layout_signature: str = ""
    process_monitor_id: Optional[int] = None
    doc_type: str = ""
```

- [ ] **Step 4: Run — pass; Step 5: Commit**

```bash
git add src/services/structural_extractor/types.py tests/structural_extractor/test_types.py
git commit -m "feat(extraction): ExtractedValue and ExtractionResult types"
```

---

### Task 4: Format exceptions hierarchy

**Files:**
- Create: `src/services/structural_extractor/exceptions.py`
- Test: `tests/structural_extractor/test_exceptions.py`

- [ ] **Step 1: Failing test**

```python
# tests/structural_extractor/test_exceptions.py
from src.services.structural_extractor.exceptions import (
    FormatParseError, PDFParseError, DocxParseError, XlsxParseError, CsvParseError
)

def test_hierarchy():
    assert issubclass(PDFParseError, FormatParseError)
    assert issubclass(DocxParseError, FormatParseError)
    assert issubclass(XlsxParseError, FormatParseError)
    assert issubclass(CsvParseError, FormatParseError)
```

- [ ] **Step 2: Run — fail; Step 3: Implement**

```python
# src/services/structural_extractor/exceptions.py
class FormatParseError(Exception): pass
class PDFParseError(FormatParseError): pass
class DocxParseError(FormatParseError): pass
class XlsxParseError(FormatParseError): pass
class CsvParseError(FormatParseError): pass
class DerivationError(Exception): pass
class UnresolvedFieldError(Exception):
    def __init__(self, field_name: str, attempt: int):
        self.field_name = field_name
        self.attempt = attempt
        super().__init__(f"{field_name} unresolved at attempt {attempt}")
```

- [ ] **Step 4: Pass; Step 5: Commit**

```bash
git add src/services/structural_extractor/exceptions.py tests/structural_extractor/test_exceptions.py
git commit -m "feat(extraction): format parse exceptions"
```

---

## Phase 2 — Format adapters (Layer 1)

### Task 5: Format detection dispatcher

**Files:**
- Create: `src/services/structural_extractor/parsing/__init__.py` (flesh out)
- Create: `src/services/structural_extractor/parsing/dispatcher.py`
- Test: `tests/structural_extractor/test_dispatcher.py`

- [ ] **Step 1: Failing test**

```python
# tests/structural_extractor/test_dispatcher.py
import pytest
from src.services.structural_extractor.parsing.dispatcher import detect_format

def test_detect_pdf_by_ext():
    assert detect_format(b"%PDF-1.4...", "inv.pdf") == "pdf"

def test_detect_docx_by_magic():
    # zip header PK\x03\x04 — DOCX is a zip
    assert detect_format(b"PK\x03\x04something", "inv.docx") == "docx"

def test_detect_xlsx_by_ext():
    assert detect_format(b"PK\x03\x04", "inv.xlsx") == "xlsx"

def test_detect_csv_by_ext():
    assert detect_format(b"a,b,c\n1,2,3\n", "inv.csv") == "csv"

def test_reject_unknown():
    with pytest.raises(ValueError, match="Cannot detect format"):
        detect_format(b"xxx", "inv.bin")
```

- [ ] **Step 2: Run — fail; Step 3: Implement**

```python
# src/services/structural_extractor/parsing/dispatcher.py
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
    # magic-byte fallback for zip-based Office docs w/o extension
    if file_bytes[:4] == b"PK\x03\x04" and ext == ".docx":
        return "docx"
    raise ValueError(f"Cannot detect format for filename={filename!r}, magic={file_bytes[:8]!r}")
```

- [ ] **Step 4: Pass; Step 5: Commit**

```bash
git add src/services/structural_extractor/parsing/dispatcher.py tests/structural_extractor/test_dispatcher.py
git commit -m "feat(extraction): format dispatcher"
```

---

### Task 6: PDF parser adapter

**Files:**
- Create: `src/services/structural_extractor/parsing/pdf_parser.py`
- Test: `tests/structural_extractor/test_pdf_parser.py`
- Fixture: `tests/structural_extractor/fixtures/docs/sample_invoice.pdf` (copy from `/tmp/src/CITY OF NEWPORT INV600254 for PO502004 .pdf`)

- [ ] **Step 1: Copy test fixture**

```bash
mkdir -p tests/structural_extractor/fixtures/docs
cp "/tmp/src/CITY OF NEWPORT INV600254 for PO502004 .pdf" "tests/structural_extractor/fixtures/docs/INV600254.pdf"
```

- [ ] **Step 2: Failing test**

```python
# tests/structural_extractor/test_pdf_parser.py
from pathlib import Path
from src.services.structural_extractor.parsing.pdf_parser import parse_pdf
from src.services.structural_extractor.parsing.model import BBox, Token

FIXTURE = Path(__file__).parent / "fixtures/docs/INV600254.pdf"

def test_parse_pdf_produces_tokens():
    doc = parse_pdf(FIXTURE.read_bytes(), "INV600254.pdf")
    assert doc.source_format == "pdf"
    assert doc.pages_or_sheets == 1
    assert len(doc.tokens) > 20  # real invoice has many tokens
    # Every token has BBox anchor
    for t in doc.tokens:
        assert isinstance(t.anchor, BBox)
        assert t.anchor.page == 1

def test_parse_pdf_full_text_contains_key_tokens():
    doc = parse_pdf(FIXTURE.read_bytes(), "INV600254.pdf")
    # Key tokens from the source PDF
    assert "INV600254" in doc.full_text
    assert "8,333" in doc.full_text or "8333" in doc.full_text
    assert "City of Newport" in doc.full_text
```

- [ ] **Step 3: Run — fail; Step 4: Implement**

```python
# src/services/structural_extractor/parsing/pdf_parser.py
from io import BytesIO
import logging
import fitz  # PyMuPDF
from src.services.structural_extractor.parsing.model import (
    BBox, Token, Region, ParsedDocument, Table
)
from src.services.structural_extractor.exceptions import PDFParseError

log = logging.getLogger(__name__)

def parse_pdf(file_bytes: bytes, filename: str) -> ParsedDocument:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFParseError(f"PyMuPDF failed to open {filename}: {exc}") from exc

    tokens: list[Token] = []
    regions: list[Region] = []
    full_text_parts: list[str] = []
    order = 0

    for page_num, page in enumerate(doc, start=1):
        # words: list of (x0, y0, x1, y1, text, block, line, word)
        words = page.get_text("words")
        page_regions_by_block: dict[int, list[Token]] = {}
        for w in words:
            x0, y0, x1, y1, text, block, line, _ = w
            if not text.strip():
                continue
            tok = Token(
                text=text.strip(),
                anchor=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                block_no=block,
                line_no=line,
                order=order,
            )
            order += 1
            tokens.append(tok)
            page_regions_by_block.setdefault(block, []).append(tok)

        for block_no, block_tokens in page_regions_by_block.items():
            regions.append(Region(tokens=block_tokens, kind="block"))

        full_text_parts.append(page.get_text("text"))

    doc.close()

    return ParsedDocument(
        source_format="pdf",
        filename=filename,
        tokens=tokens,
        regions=regions,
        tables=[],  # Table extraction via pdfplumber in a later task
        pages_or_sheets=len(doc),
        full_text="\n".join(full_text_parts),
        raw_bytes=file_bytes,
    )
```

- [ ] **Step 5: Run — pass**

Run: `.venv/bin/pytest tests/structural_extractor/test_pdf_parser.py -v`

- [ ] **Step 6: Commit**

```bash
git add src/services/structural_extractor/parsing/pdf_parser.py tests/structural_extractor/test_pdf_parser.py tests/structural_extractor/fixtures/docs/INV600254.pdf
git commit -m "feat(extraction): PDF parser adapter using PyMuPDF"
```

---

### Task 7: Add pdfplumber table extraction to PDF parser

**Files:**
- Modify: `src/services/structural_extractor/parsing/pdf_parser.py`
- Test: `tests/structural_extractor/test_pdf_parser.py`

- [ ] **Step 1: Failing test — tables populated when present**

```python
def test_parse_pdf_populates_tables_when_pdfplumber_finds_them():
    doc = parse_pdf(FIXTURE.read_bytes(), "INV600254.pdf")
    # `tables` is always a list (never None); may be empty if pdfplumber can't
    # find real tables. The interesting assertion is: a Table, if present, has
    # non-empty rows — we do NOT accept `[Table(rows=[])]` as a valid result.
    assert isinstance(doc.tables, list)
    for tbl in doc.tables:
        assert len(tbl.rows) > 0
        assert any(any(r.tokens for r in row) for row in tbl.rows), \
            "Degenerate tables (all cells empty) must have been filtered out"
```

- [ ] **Step 2: Implement table extraction**

Add to `pdf_parser.py`:
```python
import pdfplumber

def _extract_tables(file_bytes: bytes) -> list[Table]:
    tables: list[Table] = []
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                for t in (page.extract_tables() or []):
                    # Degenerate tables: skip (empty rows/cols)
                    if not t or not any(any(cell for cell in row) for row in t):
                        continue
                    rows: list[list[Region]] = []
                    for row_data in t:
                        row_regions = [
                            Region(
                                tokens=[Token(text=str(cell or ""), anchor=BBox(page_idx, 0, 0, 0, 0), order=0)],
                                kind="cell",
                            )
                            for cell in row_data
                        ]
                        rows.append(row_regions)
                    tables.append(Table(rows=rows, header_row_index=0 if rows else None))
    except Exception:
        log.debug("pdfplumber table extraction failed", exc_info=True)
    return tables
```

Wire into `parse_pdf`:
```python
return ParsedDocument(
    ...
    tables=_extract_tables(file_bytes),
    ...
)
```

- [ ] **Step 3: Pass; Step 4: Commit**

```bash
git commit -am "feat(extraction): add pdfplumber tables to PDF adapter"
```

---

### Task 8: DOCX parser adapter

**Files:**
- Create: `src/services/structural_extractor/parsing/docx_parser.py`
- Test: `tests/structural_extractor/test_docx_parser.py`
- Fixture: generate a simple invoice DOCX programmatically in the test.

- [ ] **Step 1: Create a test fixture generator + failing test**

```python
# tests/structural_extractor/test_docx_parser.py
import io
from docx import Document
from src.services.structural_extractor.parsing.docx_parser import parse_docx
from src.services.structural_extractor.parsing.model import NodeRef, Token

def _build_sample_docx() -> bytes:
    d = Document()
    d.add_paragraph("Invoice No: INV-001")
    d.add_paragraph("Invoice Date: 01/07/2022")
    t = d.add_table(rows=2, cols=3)
    t.rows[0].cells[0].text = "Description"
    t.rows[0].cells[1].text = "Qty"
    t.rows[0].cells[2].text = "Amount"
    t.rows[1].cells[0].text = "Widget"
    t.rows[1].cells[1].text = "10"
    t.rows[1].cells[2].text = "100.00"
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()

def test_parse_docx_paragraphs_have_noderef():
    doc = parse_docx(_build_sample_docx(), "test.docx")
    assert doc.source_format == "docx"
    para_tokens = [t for t in doc.tokens if isinstance(t.anchor, NodeRef) and t.anchor.kind == "paragraph"]
    assert len(para_tokens) >= 2
    assert any(t.text == "INV-001" for t in para_tokens)

def test_parse_docx_table_has_cell_noderef():
    doc = parse_docx(_build_sample_docx(), "test.docx")
    assert len(doc.tables) == 1
    # "Widget" should be a token anchored to a table cell
    cell_tokens = [t for t in doc.tokens if isinstance(t.anchor, NodeRef) and t.anchor.kind == "table_cell"]
    assert any(t.text == "Widget" for t in cell_tokens)
```

- [ ] **Step 2: Fail; Step 3: Implement**

```python
# src/services/structural_extractor/parsing/docx_parser.py
from io import BytesIO
from docx import Document
from src.services.structural_extractor.parsing.model import (
    NodeRef, Token, Region, ParsedDocument, Table
)
from src.services.structural_extractor.exceptions import DocxParseError

def parse_docx(file_bytes: bytes, filename: str) -> ParsedDocument:
    try:
        d = Document(BytesIO(file_bytes))
    except Exception as exc:
        raise DocxParseError(f"python-docx failed for {filename}: {exc}") from exc

    tokens: list[Token] = []
    regions: list[Region] = []
    tables: list[Table] = []
    full_text_parts: list[str] = []
    order = 0

    for p_idx, para in enumerate(d.paragraphs):
        para_tokens: list[Token] = []
        for word in para.text.split():
            tok = Token(
                text=word,
                anchor=NodeRef(kind="paragraph", paragraph_index=p_idx),
                order=order,
            )
            order += 1
            tokens.append(tok)
            para_tokens.append(tok)
        if para_tokens:
            regions.append(Region(tokens=para_tokens, kind="paragraph"))
            full_text_parts.append(para.text)

    for t_idx, tbl in enumerate(d.tables):
        rows: list[list[Region]] = []
        for r_idx, row in enumerate(tbl.rows):
            row_regions: list[Region] = []
            for c_idx, cell in enumerate(row.cells):
                cell_tokens: list[Token] = []
                for word in cell.text.split():
                    tok = Token(
                        text=word,
                        anchor=NodeRef(
                            kind="table_cell", table_index=t_idx, row=r_idx, col=c_idx
                        ),
                        order=order,
                    )
                    order += 1
                    tokens.append(tok)
                    cell_tokens.append(tok)
                row_regions.append(Region(tokens=cell_tokens, kind="cell"))
            rows.append(row_regions)
            full_text_parts.append(" | ".join(c.text for c in row.cells))
        tables.append(Table(rows=rows, header_row_index=0 if rows else None))

    return ParsedDocument(
        source_format="docx",
        filename=filename,
        tokens=tokens,
        regions=regions,
        tables=tables,
        pages_or_sheets=1,
        full_text="\n".join(full_text_parts),
        raw_bytes=file_bytes,
    )
```

- [ ] **Step 4: Pass; Step 5: Commit**

```bash
git add src/services/structural_extractor/parsing/docx_parser.py tests/structural_extractor/test_docx_parser.py
git commit -m "feat(extraction): DOCX parser adapter"
```

---

### Task 9: XLSX parser adapter

**Files:**
- Create: `src/services/structural_extractor/parsing/xlsx_parser.py`
- Test: `tests/structural_extractor/test_xlsx_parser.py`

- [ ] **Step 1: Failing test**

```python
# tests/structural_extractor/test_xlsx_parser.py
import io
import openpyxl
from src.services.structural_extractor.parsing.xlsx_parser import parse_xlsx
from src.services.structural_extractor.parsing.model import CellRef

def _build_sample_xlsx() -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Invoice"
    ws["A1"] = "Invoice No"
    ws["B1"] = "INV-001"
    ws["A2"] = "Date"
    ws["B2"] = "01/07/2022"
    ws["A4"] = "Description"
    ws["B4"] = "Qty"
    ws["C4"] = "Price"
    ws["A5"] = "Widget"
    ws["B5"] = 10
    ws["C5"] = 99.99
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

def test_parse_xlsx_cellrefs():
    doc = parse_xlsx(_build_sample_xlsx(), "inv.xlsx")
    assert doc.source_format == "xlsx"
    assert doc.pages_or_sheets == 1
    # INV-001 should be a token at B1
    t = next(t for t in doc.tokens if t.text == "INV-001")
    assert isinstance(t.anchor, CellRef)
    assert t.anchor.sheet == "Invoice"
    assert t.anchor.row == 1 and t.anchor.col == 2

def test_parse_xlsx_table_detected():
    doc = parse_xlsx(_build_sample_xlsx(), "inv.xlsx")
    assert len(doc.tables) >= 1
```

- [ ] **Step 2: Fail; Step 3: Implement**

```python
# src/services/structural_extractor/parsing/xlsx_parser.py
from io import BytesIO
import openpyxl
from src.services.structural_extractor.parsing.model import (
    CellRef, Token, Region, ParsedDocument, Table
)
from src.services.structural_extractor.exceptions import XlsxParseError

def parse_xlsx(file_bytes: bytes, filename: str) -> ParsedDocument:
    try:
        wb = openpyxl.load_workbook(BytesIO(file_bytes), data_only=True)
    except Exception as exc:
        raise XlsxParseError(f"openpyxl failed for {filename}: {exc}") from exc

    tokens: list[Token] = []
    regions: list[Region] = []
    tables: list[Table] = []
    full_text_parts: list[str] = []
    order = 0

    for sheet_idx, ws in enumerate(wb.worksheets):
        merged_lookup = {}
        for merged in ws.merged_cells.ranges:
            for r in range(merged.min_row, merged.max_row + 1):
                for c in range(merged.min_col, merged.max_col + 1):
                    merged_lookup[(r, c)] = str(merged)

        sheet_regions: list[list[Region]] = []
        non_empty_count = 0
        for row_cells in ws.iter_rows():
            row_regions: list[Region] = []
            for cell in row_cells:
                if cell.value is None:
                    row_regions.append(Region(tokens=[], kind="cell"))
                    continue
                text = str(cell.value)
                non_empty_count += 1
                merged = merged_lookup.get((cell.row, cell.column))
                tok = Token(
                    text=text,
                    anchor=CellRef(
                        sheet=ws.title,
                        row=cell.row,
                        col=cell.column,
                        merged_range=merged,
                    ),
                    order=order,
                )
                order += 1
                tokens.append(tok)
                row_regions.append(Region(tokens=[tok], kind="cell"))
                full_text_parts.append(text)
            sheet_regions.append(row_regions)

        if non_empty_count >= 4:
            # Detect header row = first row where all non-empty cells are text
            header_idx = None
            for r_idx, row in enumerate(sheet_regions):
                non_empty = [r for r in row if r.tokens]
                if non_empty and all(
                    not r.tokens[0].text.replace(".", "").replace("-", "").isdigit()
                    for r in non_empty
                ):
                    header_idx = r_idx
                    break
            tables.append(Table(
                rows=sheet_regions, header_row_index=header_idx,
            ))

        regions.extend([r for row in sheet_regions for r in row])

    return ParsedDocument(
        source_format="xlsx",
        filename=filename,
        tokens=tokens,
        regions=regions,
        tables=tables,
        pages_or_sheets=len(wb.worksheets),
        full_text="\n".join(full_text_parts),
        raw_bytes=file_bytes,
    )
```

- [ ] **Step 4: Pass; Step 5: Commit**

```bash
git add src/services/structural_extractor/parsing/xlsx_parser.py tests/structural_extractor/test_xlsx_parser.py
git commit -m "feat(extraction): XLSX parser adapter"
```

---

### Task 10: CSV parser adapter

**Files:**
- Create: `src/services/structural_extractor/parsing/csv_parser.py`
- Test: `tests/structural_extractor/test_csv_parser.py`

- [ ] **Step 1: Failing test**

```python
# tests/structural_extractor/test_csv_parser.py
from src.services.structural_extractor.parsing.csv_parser import parse_csv
from src.services.structural_extractor.parsing.model import ColumnRef

def test_parse_csv_with_header():
    data = b"invoice_id,amount,currency\nINV-001,100.0,GBP\nINV-002,200.0,USD\n"
    doc = parse_csv(data, "test.csv")
    assert doc.source_format == "csv"
    # Tokens in data rows have ColumnRef with column_name populated
    inv_token = next(t for t in doc.tokens if t.text == "INV-001")
    assert isinstance(inv_token.anchor, ColumnRef)
    assert inv_token.anchor.column_name == "invoice_id"

def test_parse_csv_without_header():
    data = b"INV-001,100.0,GBP\nINV-002,200.0,USD\n"
    doc = parse_csv(data, "test.csv")
    assert len(doc.tables) == 1
```

- [ ] **Step 2: Fail; Step 3: Implement**

```python
# src/services/structural_extractor/parsing/csv_parser.py
import csv
from io import StringIO
from src.services.structural_extractor.parsing.model import (
    ColumnRef, Token, Region, ParsedDocument, Table
)
from src.services.structural_extractor.exceptions import CsvParseError

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

    # Header detection: if every cell in row 0 is non-numeric, treat as header
    first_row = reader[0]
    header_is_labels = all(
        not _is_numeric(c.strip()) for c in first_row if c.strip()
    )
    header_row: list[str] = first_row if header_is_labels else [f"col_{i}" for i in range(len(first_row))]
    data_rows = reader[1:] if header_is_labels else reader

    tokens: list[Token] = []
    regions: list[Region] = []
    table_rows: list[list[Region]] = []
    full_text_parts: list[str] = []
    order = 0

    # Header row as Region (kind=column per column)
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

def _is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", "").replace("£", "").replace("$", "").replace("€", ""))
        return True
    except ValueError:
        return False
```

- [ ] **Step 4: Pass; Step 5: Commit**

```bash
git add src/services/structural_extractor/parsing/csv_parser.py tests/structural_extractor/test_csv_parser.py
git commit -m "feat(extraction): CSV parser adapter"
```

---

### Task 11: Wire parsers into `parsing/__init__.py`

**Files:**
- Modify: `src/services/structural_extractor/parsing/__init__.py`
- Test: `tests/structural_extractor/test_parsing_dispatch.py`

- [ ] **Step 1: Failing test**

```python
# tests/structural_extractor/test_parsing_dispatch.py
from pathlib import Path
from src.services.structural_extractor.parsing import parse

FIX = Path(__file__).parent / "fixtures/docs"

def test_dispatch_to_pdf():
    doc = parse((FIX / "INV600254.pdf").read_bytes(), "INV600254.pdf")
    assert doc.source_format == "pdf"

def test_dispatch_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        parse(b"binary", "x.bin")
```

- [ ] **Step 2: Implement**

```python
# src/services/structural_extractor/parsing/__init__.py
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
```

- [ ] **Step 3: Pass; Step 4: Commit**

```bash
git commit -am "feat(extraction): wire format dispatcher to parsers"
```

---

## Phase 3 — Type-driven discovery (Layer 2)

### Task 12: Schema types registry

**Files:**
- Create: `src/services/structural_extractor/discovery/schema.py`
- Test: `tests/structural_extractor/test_schema.py`

- [ ] **Step 1: Failing test**

```python
# tests/structural_extractor/test_schema.py
from src.services.structural_extractor.discovery.schema import (
    FieldType, FIELD_TYPES, fields_for, type_of,
)

def test_invoice_field_types():
    assert type_of("Invoice", "invoice_date") == FieldType.DATE
    assert type_of("Invoice", "invoice_total_incl_tax") == FieldType.MONEY
    assert type_of("Invoice", "supplier_id") == FieldType.ORG
    assert type_of("Invoice", "currency") == FieldType.CURRENCY_CODE

def test_po_fields_contain_total():
    fs = fields_for("Purchase_Order")
    assert "total_amount" in fs
    assert "supplier_name" in fs
```

- [ ] **Step 2: Fail; Step 3: Implement**

```python
# src/services/structural_extractor/discovery/schema.py
from enum import Enum

class FieldType(str, Enum):
    ID = "id"
    DATE = "date"
    MONEY = "money"
    PERCENT = "percent"
    CURRENCY_CODE = "currency"
    ORG = "org"
    ADDRESS = "address"
    TEXT = "text"
    INTEGER = "integer"

FIELD_TYPES: dict[tuple[str, str], FieldType] = {
    # Invoice
    ("Invoice", "invoice_id"):            FieldType.ID,
    ("Invoice", "po_id"):                 FieldType.ID,
    ("Invoice", "supplier_id"):           FieldType.ORG,
    ("Invoice", "buyer_id"):              FieldType.ORG,
    ("Invoice", "invoice_date"):          FieldType.DATE,
    ("Invoice", "due_date"):              FieldType.DATE,
    ("Invoice", "invoice_amount"):        FieldType.MONEY,
    ("Invoice", "tax_amount"):            FieldType.MONEY,
    ("Invoice", "tax_percent"):           FieldType.PERCENT,
    ("Invoice", "invoice_total_incl_tax"): FieldType.MONEY,
    ("Invoice", "currency"):              FieldType.CURRENCY_CODE,
    ("Invoice", "payment_terms"):         FieldType.TEXT,
    # Purchase_Order
    ("Purchase_Order", "po_id"):                FieldType.ID,
    ("Purchase_Order", "supplier_name"):        FieldType.ORG,
    ("Purchase_Order", "supplier_id"):          FieldType.ORG,
    ("Purchase_Order", "buyer_id"):             FieldType.ORG,
    ("Purchase_Order", "order_date"):           FieldType.DATE,
    ("Purchase_Order", "expected_delivery_date"): FieldType.DATE,
    ("Purchase_Order", "total_amount"):         FieldType.MONEY,
    ("Purchase_Order", "tax_amount"):           FieldType.MONEY,
    ("Purchase_Order", "tax_percent"):          FieldType.PERCENT,
    ("Purchase_Order", "total_amount_incl_tax"): FieldType.MONEY,
    ("Purchase_Order", "currency"):             FieldType.CURRENCY_CODE,
    ("Purchase_Order", "payment_terms"):        FieldType.TEXT,
    ("Purchase_Order", "incoterm"):             FieldType.TEXT,
    # Quote
    ("Quote", "quote_id"):               FieldType.ID,
    ("Quote", "supplier_id"):            FieldType.ORG,
    ("Quote", "buyer_id"):               FieldType.ORG,
    ("Quote", "quote_date"):             FieldType.DATE,
    ("Quote", "validity_date"):          FieldType.DATE,
    ("Quote", "total_amount"):           FieldType.MONEY,
    ("Quote", "tax_amount"):             FieldType.MONEY,
    ("Quote", "tax_percent"):            FieldType.PERCENT,
    ("Quote", "total_amount_incl_tax"):  FieldType.MONEY,
    ("Quote", "currency"):               FieldType.CURRENCY_CODE,
}

def fields_for(doc_type: str) -> list[str]:
    return [fname for (dt, fname) in FIELD_TYPES if dt == doc_type]

def type_of(doc_type: str, field_name: str) -> FieldType | None:
    return FIELD_TYPES.get((doc_type, field_name))
```

- [ ] **Step 4: Pass; Step 5: Commit**

```bash
git commit -am "feat(extraction): schema field type registry"
```

---

### Task 13: Typed-entity detectors

**Files:**
- Create: `src/services/structural_extractor/discovery/type_entities.py`
- Test: `tests/structural_extractor/test_type_entities.py`

- [ ] **Step 1: Failing test**

```python
# tests/structural_extractor/test_type_entities.py
from src.services.structural_extractor.discovery.schema import FieldType
from src.services.structural_extractor.discovery.type_entities import find_candidates
from src.services.structural_extractor.parsing.model import Token, BBox, ParsedDocument

def _tok(text, order=0):
    return Token(text=text, anchor=BBox(1, 0, 0, 0, 0), order=order)

def _doc(token_texts):
    toks = [_tok(t, i) for i, t in enumerate(token_texts)]
    return ParsedDocument(
        source_format="pdf", filename="", tokens=toks, regions=[], tables=[],
        pages_or_sheets=1, full_text=" ".join(token_texts), raw_bytes=b"",
    )

def test_date_candidates():
    d = _doc(["Invoice", "Date:", "01/07/2022", "Random"])
    cands = find_candidates(d, FieldType.DATE)
    assert any(c.text == "01/07/2022" for c in cands)

def test_money_candidates():
    d = _doc(["Subtotal", "£8,333.00", "Tax", "20%", "Total", "£9,999.60"])
    cands = find_candidates(d, FieldType.MONEY)
    texts = {c.text for c in cands}
    assert "£8,333.00" in texts and "£9,999.60" in texts

def test_percent_candidate():
    d = _doc(["Tax", "(20%)"])
    cands = find_candidates(d, FieldType.PERCENT)
    assert any("20" in c.text for c in cands)

def test_currency_code_candidate():
    d = _doc(["Total", "9999.60", "GBP"])
    cands = find_candidates(d, FieldType.CURRENCY_CODE)
    assert any(c.text == "GBP" for c in cands)
```

- [ ] **Step 2: Fail; Step 3: Implement**

```python
# src/services/structural_extractor/discovery/type_entities.py
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
from dateutil import parser as date_parser
from src.services.structural_extractor.discovery.schema import FieldType
from src.services.structural_extractor.parsing.model import Token, ParsedDocument

ISO_4217 = {
    "USD", "GBP", "EUR", "AUD", "CAD", "JPY", "CHF", "INR", "CNY", "HKD", "SGD", "NZD",
}
CURRENCY_SYMBOLS = {"£": "GBP", "$": "USD", "€": "EUR", "¥": "JPY", "₹": "INR"}

@dataclass
class Candidate:
    text: str
    tokens: list[Token]
    parsed_value: object  # type-specific

def find_candidates(doc: ParsedDocument, ftype: FieldType) -> list[Candidate]:
    if ftype == FieldType.DATE:
        return _find_date(doc)
    if ftype == FieldType.MONEY:
        return _find_money(doc)
    if ftype == FieldType.PERCENT:
        return _find_percent(doc)
    if ftype == FieldType.CURRENCY_CODE:
        return _find_currency(doc)
    if ftype == FieldType.ID:
        return _find_id(doc)
    if ftype == FieldType.ORG:
        return _find_org(doc)
    if ftype == FieldType.ADDRESS:
        return _find_address(doc)
    if ftype == FieldType.TEXT:
        return [Candidate(t.text, [t], t.text) for t in doc.tokens]
    return []

def _find_date(doc: ParsedDocument) -> list[Candidate]:
    cands: list[Candidate] = []
    tokens = doc.tokens
    n = len(tokens)
    # Try token window of size 1..4
    for i in range(n):
        for window in range(1, 5):
            if i + window > n:
                break
            segment = tokens[i:i+window]
            text = " ".join(t.text for t in segment)
            try:
                dt = date_parser.parse(text, fuzzy=False)
                # Sanity: year between 1980 and 2100
                if 1980 <= dt.year <= 2100:
                    cands.append(Candidate(text=text, tokens=segment, parsed_value=dt))
            except Exception:
                pass
    return cands

def _find_money(doc: ParsedDocument) -> list[Candidate]:
    cands: list[Candidate] = []
    for t in doc.tokens:
        stripped = t.text.replace(",", "").replace(" ", "")
        # Must contain a currency symbol OR be numeric adjacent-to-symbol context
        has_symbol = any(s in t.text for s in CURRENCY_SYMBOLS)
        clean = stripped
        for s in CURRENCY_SYMBOLS:
            clean = clean.replace(s, "")
        try:
            val = float(clean)
            if has_symbol or (val >= 0 and val < 1e10 and "." in t.text):
                cands.append(Candidate(text=t.text, tokens=[t], parsed_value=val))
        except ValueError:
            continue
    return cands

def _find_percent(doc: ParsedDocument) -> list[Candidate]:
    cands: list[Candidate] = []
    for t in doc.tokens:
        if "%" in t.text:
            clean = t.text.replace("%", "").replace("(", "").replace(")", "").strip()
            try:
                val = float(clean)
                if 0 <= val <= 100:
                    cands.append(Candidate(text=t.text, tokens=[t], parsed_value=val))
            except ValueError:
                continue
    return cands

def _find_address(doc: ParsedDocument) -> list[Candidate]:
    """ADDRESS candidate: a contiguous run of ≥ 2 lines where at least one line
    matches a postcode pattern (UK format `[A-Z]{1,2}[0-9]...`, US ZIP `\\d{5}`).
    Returns the full run as a single candidate."""
    import re
    UK_PC = re.compile(r"^[A-Z]{1,2}[0-9][A-Z0-9]?$")
    US_ZIP = re.compile(r"^\d{5}(-\d{4})?$")
    cands: list[Candidate] = []
    # Group tokens into lines via (page, line_no) or fallback to block_no
    lines: dict[tuple, list[Token]] = {}
    for t in doc.tokens:
        key = (getattr(t.anchor, "page", 0), t.line_no or t.block_no or t.order)
        lines.setdefault(key, []).append(t)
    line_items = sorted(lines.items())
    for idx, (_, line_toks) in enumerate(line_items):
        for tok in line_toks:
            if UK_PC.match(tok.text) or US_ZIP.match(tok.text.rstrip(",")):
                # Claim the 2-4 lines before as the address block
                start = max(0, idx - 3)
                block = [t for _, ln in line_items[start:idx+1] for t in ln]
                text = " ".join(t.text for t in block)
                cands.append(Candidate(text=text, tokens=block, parsed_value=text))
                break
    return cands

def _find_currency(doc: ParsedDocument) -> list[Candidate]:
    cands: list[Candidate] = []
    for t in doc.tokens:
        tt = t.text.strip().upper()
        if tt in ISO_4217:
            cands.append(Candidate(text=t.text, tokens=[t], parsed_value=tt))
        else:
            for sym, code in CURRENCY_SYMBOLS.items():
                if sym in t.text:
                    cands.append(Candidate(text=sym, tokens=[t], parsed_value=code))
                    break
    return cands

def _find_id(doc: ParsedDocument) -> list[Candidate]:
    cands: list[Candidate] = []
    for t in doc.tokens:
        s = t.text.strip().rstrip(":,.")
        if len(s) >= 3 and any(c.isdigit() for c in s) and not any(x in s for x in "£$€¥%"):
            cands.append(Candidate(text=s, tokens=[t], parsed_value=s))
    return cands

def _find_org(doc: ParsedDocument) -> list[Candidate]:
    """Heuristic ORG detection: capitalized runs ending in Ltd/LLC/Inc/etc.
    NLU NER augments this in Layer 3 retries."""
    cands: list[Candidate] = []
    suffixes = {"Ltd", "Ltd.", "LLC", "Inc", "Inc.", "Limited", "plc", "Corp", "Company", "GmbH", "AG", "SA"}
    tokens = doc.tokens
    n = len(tokens)
    for i in range(n):
        if tokens[i].text.rstrip(",.") in suffixes:
            # Walk back to find the capitalized run
            start = i
            while start > 0 and tokens[start-1].text and tokens[start-1].text[0].isupper():
                start -= 1
            span = tokens[start:i+1]
            text = " ".join(t.text for t in span)
            cands.append(Candidate(text=text, tokens=span, parsed_value=text))
    return cands
```

- [ ] **Step 4: Pass; Step 5: Commit**

```bash
git commit -am "feat(extraction): typed entity detectors"
```

---

### Task 14: Proximity inference + scoring helpers

**Files:**
- Create: `src/services/structural_extractor/discovery/proximity.py`
- Test: `tests/structural_extractor/test_proximity.py`

- [ ] **Step 1: Failing test**

```python
# tests/structural_extractor/test_proximity.py
from src.services.structural_extractor.discovery.proximity import inferred_label, arithmetic_fit
from src.services.structural_extractor.discovery.type_entities import Candidate
from src.services.structural_extractor.parsing.model import Token, BBox

def _tok(text, x, y, order):
    return Token(text=text, anchor=BBox(1, x, y, x+50, y+10), order=order)

def test_inferred_label_from_same_line_left():
    # "Invoice Date: 01/07/2022" — label = "Invoice Date:"
    tokens = [
        _tok("Invoice", 0, 100, 0),
        _tok("Date:", 60, 100, 1),
        _tok("01/07/2022", 120, 100, 2),
    ]
    cand = Candidate(text="01/07/2022", tokens=[tokens[2]], parsed_value=None)
    label = inferred_label(cand, tokens)
    assert "Date" in label

def test_arithmetic_fit_reconciles():
    # subtotal=8333, tax=1666.60, total=9999.60 → fit=1.0
    assert arithmetic_fit(8333.0, 1666.60, 9999.60) == 1.0
    assert arithmetic_fit(8333.0, 1666.60, 10000.0) == 0.0
```

- [ ] **Step 2: Fail; Step 3: Implement**

```python
# src/services/structural_extractor/discovery/proximity.py
from typing import Iterable
from src.services.structural_extractor.discovery.type_entities import Candidate
from src.services.structural_extractor.parsing.model import Token, BBox, CellRef, ColumnRef, NodeRef

def inferred_label(cand: Candidate, all_tokens: list[Token], max_lookback: int = 5) -> str:
    """Return the inferred label (tokens preceding the candidate on the same line)."""
    if not cand.tokens:
        return ""
    first = cand.tokens[0]
    anchor = first.anchor
    label_tokens: list[Token] = []
    if isinstance(anchor, BBox):
        # Same-line-left: tokens at similar y with smaller x
        line_y = (anchor.y0 + anchor.y1) / 2
        for t in all_tokens:
            if t.order >= first.order:
                break
            if not isinstance(t.anchor, BBox) or t.anchor.page != anchor.page:
                continue
            t_y = (t.anchor.y0 + t.anchor.y1) / 2
            if abs(t_y - line_y) <= 6 and t.anchor.x0 < anchor.x0:
                label_tokens.append(t)
        label_tokens = label_tokens[-max_lookback:]
    elif isinstance(anchor, CellRef):
        # Cell to the left on same sheet/row
        for t in all_tokens:
            if not isinstance(t.anchor, CellRef):
                continue
            if t.anchor.sheet == anchor.sheet and t.anchor.row == anchor.row and t.anchor.col < anchor.col:
                label_tokens.append(t)
        label_tokens = label_tokens[-1:] if label_tokens else []
    elif isinstance(anchor, ColumnRef):
        return anchor.column_name or ""
    elif isinstance(anchor, NodeRef):
        # Preceding tokens in same paragraph / cell
        for t in all_tokens:
            if t.order >= first.order:
                break
            if isinstance(t.anchor, NodeRef):
                if (anchor.kind == "paragraph" and t.anchor.kind == "paragraph"
                    and t.anchor.paragraph_index == anchor.paragraph_index):
                    label_tokens.append(t)
                elif (anchor.kind == "table_cell" and t.anchor.kind == "table_cell"
                      and t.anchor.table_index == anchor.table_index
                      and t.anchor.row == anchor.row and t.anchor.col == anchor.col):
                    label_tokens.append(t)
        label_tokens = label_tokens[-max_lookback:]
    return " ".join(t.text for t in label_tokens)

def arithmetic_fit(subtotal: float, tax: float, total: float, tol: float = 0.01) -> float:
    return 1.0 if abs(subtotal + tax - total) <= tol else 0.0
```

- [ ] **Step 4: Pass; Step 5: Commit**

```bash
git commit -am "feat(extraction): label-proximity inference + arithmetic fit"
```

---

### Task 15: Layout fingerprinting

**Files:**
- Create: `src/services/structural_extractor/discovery/layout_fingerprint.py`
- Test: `tests/structural_extractor/test_layout_fingerprint.py`

- [ ] **Step 1: Failing test**

```python
# tests/structural_extractor/test_layout_fingerprint.py
from src.services.structural_extractor.discovery.layout_fingerprint import layout_signature
from src.services.structural_extractor.parsing.model import (
    ParsedDocument, Token, BBox, Table, Region
)

def _pdf_doc(tokens):
    return ParsedDocument(
        source_format="pdf", filename="", tokens=tokens, regions=[], tables=[],
        pages_or_sheets=1, full_text="", raw_bytes=b"",
    )

def test_same_layout_same_signature():
    toks1 = [Token(text="A", anchor=BBox(1, 10, 10, 20, 20), order=0)]
    toks2 = [Token(text="B", anchor=BBox(1, 10, 10, 20, 20), order=0)]
    s1 = layout_signature(_pdf_doc(toks1))
    s2 = layout_signature(_pdf_doc(toks2))
    # Same layout (positions), different text → same signature
    assert s1 == s2

def test_different_layout_different_signature():
    toks1 = [Token(text="A", anchor=BBox(1, 10, 10, 20, 20), order=0)]
    toks2 = [Token(text="A", anchor=BBox(1, 500, 500, 510, 510), order=0)]
    assert layout_signature(_pdf_doc(toks1)) != layout_signature(_pdf_doc(toks2))
```

- [ ] **Step 2: Fail; Step 3: Implement**

```python
# src/services/structural_extractor/discovery/layout_fingerprint.py
import hashlib
from src.services.structural_extractor.parsing.model import (
    ParsedDocument, BBox, CellRef, ColumnRef, NodeRef
)

def layout_signature(doc: ParsedDocument) -> str:
    """16-hex-char deterministic hash of the document's position signature.
    Based on positions, NOT text content, so same-layout-different-vendor-name
    produces the same signature."""
    buckets: list[str] = []
    if doc.source_format == "pdf":
        # Coarse grid: 10x10 buckets per page
        for t in doc.tokens:
            if isinstance(t.anchor, BBox):
                bx = int(t.anchor.x0 / 60)  # assuming ~600pt page width
                by = int(t.anchor.y0 / 80)
                buckets.append(f"p{t.anchor.page}:{bx},{by}")
    elif doc.source_format == "xlsx":
        for t in doc.tokens:
            if isinstance(t.anchor, CellRef):
                buckets.append(f"s:{t.anchor.sheet}:r{t.anchor.row}:c{t.anchor.col}")
    elif doc.source_format == "csv":
        # header signature: column names
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
```

- [ ] **Step 4: Pass; Step 5: Commit**

```bash
git commit -am "feat(extraction): layout fingerprinting"
```

---

## Phase 4 — Per-field extractors

### Task 16: ID extractor

**Files:**
- Create: `src/services/structural_extractor/extractors/ids.py`
- Test: `tests/structural_extractor/test_extractor_ids.py`

- [ ] **Step 1: Failing test**

```python
# tests/structural_extractor/test_extractor_ids.py
from src.services.structural_extractor.extractors.ids import extract_ids
from src.services.structural_extractor.parsing.pdf_parser import parse_pdf
from pathlib import Path

FIX = Path(__file__).parent / "fixtures/docs/INV600254.pdf"

def test_extract_invoice_id_from_real_pdf():
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    out = extract_ids(doc, doc_type="Invoice")
    assert "invoice_id" in out
    assert out["invoice_id"].value == "INV600254"
    assert out["invoice_id"].provenance == "extracted"
    assert out["invoice_id"].anchor_ref is not None
```

- [ ] **Step 2: Fail; Step 3: Implement**

```python
# src/services/structural_extractor/extractors/ids.py
from src.services.structural_extractor.parsing.model import ParsedDocument
from src.services.structural_extractor.discovery.schema import FieldType, type_of, fields_for
from src.services.structural_extractor.discovery.type_entities import find_candidates
from src.services.structural_extractor.discovery.proximity import inferred_label
from src.services.structural_extractor.types import ExtractedValue

def extract_ids(doc: ParsedDocument, doc_type: str) -> dict[str, ExtractedValue]:
    out: dict[str, ExtractedValue] = {}
    id_fields = [f for f in fields_for(doc_type) if type_of(doc_type, f) == FieldType.ID]
    cands = find_candidates(doc, FieldType.ID)
    for field in id_fields:
        # Score candidates by label-similarity to field name
        # (e.g. invoice_id's label should contain "invoice")
        field_key = field.replace("_id", "").replace("_", " ").lower()
        scored: list[tuple[float, object]] = []
        for c in cands:
            lbl = inferred_label(c, doc.tokens).lower()
            # Simple token overlap score
            overlap = sum(1 for w in field_key.split() if w in lbl)
            if overlap > 0:
                scored.append((overlap, c))
        if scored:
            scored.sort(key=lambda s: -s[0])
            best = scored[0][1]
            out[field] = ExtractedValue(
                value=best.text,
                provenance="extracted",
                anchor_text=best.text,
                anchor_ref=best.tokens[0].anchor,
                source="structural",
                confidence=1.0,
                attempt=1,
            )
    return out
```

- [ ] **Step 4: Pass; Step 5: Commit**

```bash
git commit -am "feat(extraction): ID extractor"
```

---

### Task 17: Date extractor with locale detection

**Files:**
- Create: `src/services/structural_extractor/extractors/dates.py`
- Test: `tests/structural_extractor/test_extractor_dates.py`

- [ ] **Step 1: Failing test**

```python
# tests/structural_extractor/test_extractor_dates.py
from src.services.structural_extractor.extractors.dates import extract_dates, detect_locale
from src.services.structural_extractor.parsing.pdf_parser import parse_pdf
from pathlib import Path

FIX = Path(__file__).parent / "fixtures/docs"

def test_detect_locale_uk_from_postcode():
    # Synthetic: doc mentions RH13 5QH
    from src.services.structural_extractor.parsing.model import Token, BBox, ParsedDocument
    toks = [
        Token(text="RH13", anchor=BBox(1, 0, 0, 0, 0), order=0),
        Token(text="5QH", anchor=BBox(1, 0, 0, 0, 0), order=1),
    ]
    d = ParsedDocument(source_format="pdf", filename="", tokens=toks, regions=[], tables=[],
                       pages_or_sheets=1, full_text="RH13 5QH", raw_bytes=b"")
    assert detect_locale(d) == "dmy"  # UK → day-first

def test_extract_invoice_date_from_newport():
    doc = parse_pdf((FIX / "INV600254.pdf").read_bytes(), "INV600254.pdf")
    out = extract_dates(doc, doc_type="Invoice")
    assert "invoice_date" in out
    assert out["invoice_date"].value == "2019-08-22"
    assert out["invoice_date"].provenance == "extracted"
```

- [ ] **Step 2: Fail; Step 3: Implement**

```python
# src/services/structural_extractor/extractors/dates.py
import re
from dateutil import parser as date_parser
from src.services.structural_extractor.parsing.model import ParsedDocument
from src.services.structural_extractor.discovery.schema import FieldType, type_of, fields_for
from src.services.structural_extractor.discovery.type_entities import find_candidates
from src.services.structural_extractor.discovery.proximity import inferred_label
from src.services.structural_extractor.types import ExtractedValue

UK_POSTCODE_RE = re.compile(r"^[A-Z]{1,2}[0-9][A-Z0-9]?$")

def detect_locale(doc: ParsedDocument) -> str:
    """Returns 'dmy' (day-first), 'mdy' (month-first), or 'ambiguous'."""
    # 1. Any numeric-slash date with day > 12 → DMY
    for t in doc.tokens:
        m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", t.text)
        if m:
            d1 = int(m.group(1))
            if d1 > 12:
                return "dmy"
    # 2. Any UK postcode token → DMY
    for t in doc.tokens:
        if UK_POSTCODE_RE.match(t.text):
            return "dmy"
    # 3. Any ISO currency USD → MDY
    if any(t.text.upper() == "USD" for t in doc.tokens):
        return "mdy"
    # Default: ambiguous (month-name dates parse unambiguously regardless)
    return "ambiguous"

def extract_dates(doc: ParsedDocument, doc_type: str) -> dict[str, ExtractedValue]:
    out: dict[str, ExtractedValue] = {}
    date_fields = [
        f for f in fields_for(doc_type)
        if type_of(doc_type, f) == FieldType.DATE and f != "due_date"
        # due_date is derived in the Derivation Registry, not directly extracted
    ]
    locale = detect_locale(doc)
    dayfirst = locale == "dmy"
    cands = find_candidates(doc, FieldType.DATE)

    for field in date_fields:
        field_key = field.replace("_date", "").replace("_", " ").lower()
        scored: list[tuple[int, object, object]] = []
        for c in cands:
            lbl = inferred_label(c, doc.tokens).lower()
            overlap = sum(1 for w in field_key.split() if w and w in lbl)
            if overlap > 0:
                # Re-parse with locale
                try:
                    dt = date_parser.parse(c.text, dayfirst=dayfirst, fuzzy=False)
                    if 1980 <= dt.year <= 2100:
                        scored.append((overlap, c, dt))
                except Exception:
                    continue
        if scored:
            scored.sort(key=lambda s: -s[0])
            _, best, parsed = scored[0]
            out[field] = ExtractedValue(
                value=parsed.strftime("%Y-%m-%d"),
                provenance="extracted",
                anchor_text=best.text,
                anchor_ref=best.tokens[0].anchor,
                source="structural",
                confidence=1.0,
                attempt=1,
            )
    return out
```

- [ ] **Step 4: Pass; Step 5: Commit**

```bash
git commit -am "feat(extraction): date extractor with locale detection"
```

---

### Task 18: Parties extractor

**Files:**
- Create: `src/services/structural_extractor/extractors/parties.py`
- Test: `tests/structural_extractor/test_extractor_parties.py`

- [ ] **Step 1: Failing test**

```python
# tests/structural_extractor/test_extractor_parties.py
from src.services.structural_extractor.extractors.parties import extract_parties
from src.services.structural_extractor.parsing.pdf_parser import parse_pdf
from pathlib import Path

FIX = Path(__file__).parent / "fixtures/docs"

def test_extract_newport_parties():
    doc = parse_pdf((FIX / "INV600254.pdf").read_bytes(), "INV600254.pdf")
    out = extract_parties(doc, "Invoice")
    # Source has "City of Newport" (supplier, letterhead position) and "Assurity Ltd" (buyer, Bill-To)
    assert "supplier_id" in out
    assert "Newport" in out["supplier_id"].value
    assert "buyer_id" in out
    assert "Assurity" in out["buyer_id"].value
    # Suffix preserved
    assert "Ltd" in out["buyer_id"].value
```

- [ ] **Step 2: Fail; Step 3: Implement**

```python
# src/services/structural_extractor/extractors/parties.py
from src.services.structural_extractor.parsing.model import ParsedDocument, Token, BBox
from src.services.structural_extractor.types import ExtractedValue
from src.services.structural_extractor.discovery.type_entities import find_candidates
from src.services.structural_extractor.discovery.schema import FieldType
from src.services.structural_extractor.discovery.proximity import inferred_label

BUYER_ANCHORS = {"bill to", "billed to", "invoice to", "ship to", "sold to", "customer", "invoice for"}
SUPPLIER_ANCHORS = {"from", "remit to", "payable to", "vendor", "supplier"}

def extract_parties(doc: ParsedDocument, doc_type: str) -> dict[str, ExtractedValue]:
    out: dict[str, ExtractedValue] = {}
    org_cands = find_candidates(doc, FieldType.ORG)
    if not org_cands:
        return out

    # For each ORG candidate, check its inferred-label to classify
    for c in org_cands:
        lbl = inferred_label(c, doc.tokens).lower().strip(":;,.")
        if any(a in lbl for a in BUYER_ANCHORS) and "buyer_id" not in out:
            out["buyer_id"] = ExtractedValue(
                value=c.text, provenance="extracted", anchor_text=c.text,
                anchor_ref=c.tokens[0].anchor, source="structural",
                confidence=1.0, attempt=1,
            )
        elif any(a in lbl for a in SUPPLIER_ANCHORS) and "supplier_id" not in out:
            out["supplier_id"] = ExtractedValue(
                value=c.text, provenance="extracted", anchor_text=c.text,
                anchor_ref=c.tokens[0].anchor, source="structural",
                confidence=1.0, attempt=1,
            )

    # Fallback: if no supplier anchor found but there's only one "header-area" ORG,
    # use the top-most ORG by Y-coordinate (letterhead heuristic)
    if "supplier_id" not in out and org_cands:
        def _topmost_y(c):
            t = c.tokens[0]
            return t.anchor.y0 if isinstance(t.anchor, BBox) else 1e9
        top_org = min(org_cands, key=_topmost_y)
        if top_org.text != out.get("buyer_id", ExtractedValue(value="", provenance="extracted")).value:
            out["supplier_id"] = ExtractedValue(
                value=top_org.text, provenance="extracted", anchor_text=top_org.text,
                anchor_ref=top_org.tokens[0].anchor, source="structural",
                confidence=0.8, attempt=1,  # letterhead heuristic = lower confidence
            )

    return out
```

- [ ] **Step 4: Pass; Step 5: Commit**

```bash
git commit -am "feat(extraction): parties extractor with letterhead fallback"
```

---

### Task 19: Amounts extractor with arithmetic reconciliation

**Files:**
- Create: `src/services/structural_extractor/extractors/amounts.py`
- Test: `tests/structural_extractor/test_extractor_amounts.py`

- [ ] **Step 1: Failing test**

```python
# tests/structural_extractor/test_extractor_amounts.py
from src.services.structural_extractor.extractors.amounts import extract_amounts
from src.services.structural_extractor.parsing.pdf_parser import parse_pdf
from pathlib import Path

FIX = Path(__file__).parent / "fixtures/docs/INV600254.pdf"

def test_extract_newport_amounts():
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    out = extract_amounts(doc, "Invoice")
    # Source: Subtotal £8,333, Tax (20%) £1,666.60, Total £9,999.60
    assert abs(out["invoice_amount"].value - 8333.0) < 0.01
    assert abs(out["tax_amount"].value - 1666.60) < 0.01
    assert abs(out["invoice_total_incl_tax"].value - 9999.60) < 0.01
    assert out["currency"].value == "GBP"
```

- [ ] **Step 2: Fail; Step 3: Implement**

```python
# src/services/structural_extractor/extractors/amounts.py
from itertools import combinations
from src.services.structural_extractor.parsing.model import ParsedDocument
from src.services.structural_extractor.discovery.schema import FieldType, type_of, fields_for
from src.services.structural_extractor.discovery.type_entities import find_candidates, CURRENCY_SYMBOLS
from src.services.structural_extractor.discovery.proximity import inferred_label, arithmetic_fit
from src.services.structural_extractor.types import ExtractedValue

_SUBTOTAL_FIELD = {"Invoice": "invoice_amount", "Purchase_Order": "total_amount", "Quote": "total_amount"}
_TOTAL_FIELD    = {"Invoice": "invoice_total_incl_tax", "Purchase_Order": "total_amount_incl_tax", "Quote": "total_amount_incl_tax"}

def extract_amounts(doc: ParsedDocument, doc_type: str) -> dict[str, ExtractedValue]:
    out: dict[str, ExtractedValue] = {}
    money_cands = find_candidates(doc, FieldType.MONEY)
    if not money_cands:
        return out

    # Truncate to top-40 (position-prior-ranked; for now, by doc order)
    money_cands = money_cands[:40]

    subtotal_field = _SUBTOTAL_FIELD.get(doc_type)
    total_field = _TOTAL_FIELD.get(doc_type)

    # Enumerate (subtotal, tax, total) triples that reconcile
    best_triple = None
    for i, sub_c in enumerate(money_cands):
        sub_v = sub_c.parsed_value
        for j, tax_c in enumerate(money_cands):
            if i == j: continue
            tax_v = tax_c.parsed_value
            for k, tot_c in enumerate(money_cands):
                if k in (i, j): continue
                tot_v = tot_c.parsed_value
                if arithmetic_fit(sub_v, tax_v, tot_v) == 1.0 and tot_v > sub_v > 0:
                    # Tie-break: prefer the triple where labels are most label-like
                    sub_lbl = inferred_label(sub_c, doc.tokens).lower()
                    tax_lbl = inferred_label(tax_c, doc.tokens).lower()
                    tot_lbl = inferred_label(tot_c, doc.tokens).lower()
                    score = (
                        ("sub" in sub_lbl or "net" in sub_lbl)
                        + ("tax" in tax_lbl or "vat" in tax_lbl)
                        + ("total" in tot_lbl or "amount due" in tot_lbl or "balance" in tot_lbl)
                    )
                    if best_triple is None or score > best_triple[0]:
                        best_triple = (score, sub_c, tax_c, tot_c)

    if best_triple:
        _, sub_c, tax_c, tot_c = best_triple
        if subtotal_field:
            out[subtotal_field] = _mk(sub_c)
        out["tax_amount"] = _mk(tax_c)
        if total_field:
            out[total_field] = _mk(tot_c)
        if sub_c.parsed_value > 0:
            pct = round(tax_c.parsed_value / sub_c.parsed_value * 100, 2)
            out["tax_percent"] = ExtractedValue(
                value=pct, provenance="derived",
                derivation_trace={"rule_id": "tax_pct_from_amounts",
                                  "inputs": {"tax_amount": tax_c.parsed_value,
                                             "subtotal": sub_c.parsed_value}},
                source="derivation_registry", confidence=1.0, attempt=1,
            )

    # Currency
    curr_cands = find_candidates(doc, FieldType.CURRENCY_CODE)
    if curr_cands:
        by_count: dict[str, int] = {}
        for c in curr_cands:
            by_count[c.parsed_value] = by_count.get(c.parsed_value, 0) + 1
        best_curr = max(by_count.items(), key=lambda kv: kv[1])[0]
        out["currency"] = ExtractedValue(
            value=best_curr, provenance="extracted",
            anchor_text=curr_cands[0].text, anchor_ref=curr_cands[0].tokens[0].anchor,
            source="structural", confidence=1.0, attempt=1,
        )

    return out

def _mk(c):
    return ExtractedValue(
        value=c.parsed_value, provenance="extracted",
        anchor_text=c.text, anchor_ref=c.tokens[0].anchor,
        source="structural", confidence=1.0, attempt=1,
    )
```

- [ ] **Step 4: Pass; Step 5: Commit**

```bash
git commit -am "feat(extraction): amounts extractor with arithmetic reconciliation"
```

---

### Task 20: Line items extractor (PDF spatial) — type-discovery based, no hardcoded vocab

**Files:**
- Create: `src/services/structural_extractor/extractors/line_items.py`
- Test: `tests/structural_extractor/test_extractor_line_items.py`

**Design note** — per spec Principle #2 and the reviewer's objection to an earlier draft, this extractor does NOT use a hardcoded set of header keywords. It identifies the header row by Layer-2 discovery output (columns are positions where ≥ 2 `MONEY`/`INTEGER`/`TEXT` candidates line up vertically across the first 3-5 rows below a `TEXT`-only row). Header labels are learned per-document, not matched against a vocabulary.

- [ ] **Step 1: Fetch Duncan PO fixture from S3 (not `/tmp`) and commit**

```bash
aws s3 cp "s3://procwisemvp/documents/po/DUNCAN PO526800 for QUT128300.pdf" \
    tests/structural_extractor/fixtures/docs/DUNCAN_PO526800.pdf
```

All fixtures must be committed to the repo. Do NOT depend on `/tmp/src/` which isn't guaranteed to exist.

- [ ] **Step 2: Failing test**

```python
# tests/structural_extractor/test_extractor_line_items.py
from src.services.structural_extractor.extractors.line_items import extract_line_items
from src.services.structural_extractor.parsing.pdf_parser import parse_pdf
from pathlib import Path

FIX = Path(__file__).parent / "fixtures/docs"

def test_duncan_po_has_3_fellowes_chairs():
    doc = parse_pdf((FIX / "DUNCAN_PO526800.pdf").read_bytes(), "DUNCAN.pdf")
    items = extract_line_items(doc, "Purchase_Order")
    assert len(items) == 3
    # All three should be "Fellowes Office Chair" variants
    for item in items:
        desc = item["item_description"].value
        assert "Fellowes" in desc
        assert abs(item["line_total"].value - 799.90) < 0.01
        assert item["quantity"].value == 10
```

- [ ] **Step 2: Fail; Step 3: Implement**

```python
# src/services/structural_extractor/extractors/line_items.py
from src.services.structural_extractor.parsing.model import (
    ParsedDocument, Token, BBox, CellRef, ColumnRef, NodeRef
)
from src.services.structural_extractor.types import ExtractedValue

# No hardcoded label vocabulary. Headers are discovered by finding a row whose
# tokens are all of TEXT type (Layer-2 candidates), followed by rows where the
# columns (by X-midpoint) have consistent typed entities: at least one MONEY
# column (unit_price/total), at least one INTEGER column (qty), one TEXT column
# (description).
MIN_COLUMNS = 2  # a table must have ≥ 2 columns to be considered

# The implementation below DOES NOT reference any hardcoded keyword sets.
# Instead it calls discovery.type_entities to classify each row's tokens
# by type (MONEY / INTEGER / TEXT) and finds the header as the first row
# whose tokens are all TEXT and whose X-positions align with a subsequent
# row that has ≥1 MONEY and ≥1 INTEGER/MONEY column.

def extract_line_items(doc: ParsedDocument, doc_type: str) -> list[dict[str, ExtractedValue]]:
    if doc.source_format == "pdf":
        return _pdf_line_items(doc)
    if doc.source_format == "xlsx":
        return _xlsx_line_items(doc)
    if doc.source_format == "docx":
        return _docx_line_items(doc)
    if doc.source_format == "csv":
        return _csv_line_items(doc)
    return []

def _pdf_line_items(doc: ParsedDocument) -> list[dict[str, ExtractedValue]]:
    """Type-driven line-item discovery. No hardcoded label vocabulary.

    Algorithm:
    1. Group tokens into lines by (page, y-bucket).
    2. For each line, classify each token into one of: TEXT, INTEGER, MONEY,
       PERCENT, DATE by calling discovery.type_entities detectors.
    3. A HEADER ROW is the first line whose non-empty tokens are all TEXT AND
       at least 2 of the subsequent 5 lines show a consistent column structure:
       the same X-midpoints have MONEY/INTEGER tokens.
    4. A STOP ROW is the first line after the header that contains ≥ 1 MONEY
       token at a column position NOT present in the header (i.e. a summary
       row — subtotal/tax/total that sit on a different layout).
    5. Columns are inferred from header tokens' X-midpoints; each data row's
       tokens are assigned to the nearest column.
    6. Column ROLES (description / qty / unit_price / line_total) are assigned
       by the TYPE signature of the column's data:
       - Column with majority TEXT tokens → description
       - Leftmost INTEGER / small-MONEY column → quantity
       - MONEY column where values × qty column ≈ another MONEY column → (qty × unit_price = line_total)
       Role assignment is done by the arithmetic-fit heuristic in Task 40e's
       ranker, NOT by matching header label text.
    """
    from src.services.structural_extractor.discovery.type_entities import find_candidates
    from src.services.structural_extractor.discovery.schema import FieldType

    lines: dict[tuple[int, int], list[Token]] = {}
    for t in doc.tokens:
        if isinstance(t.anchor, BBox):
            y_bucket = int(t.anchor.y0 / 4)
            lines.setdefault((t.anchor.page, y_bucket), []).append(t)

    ordered_keys = sorted(lines.keys())

    # Classify each line's tokens by type
    money_tokens = {id(c.tokens[0]) for c in find_candidates(doc, FieldType.MONEY)}
    percent_tokens = {id(c.tokens[0]) for c in find_candidates(doc, FieldType.PERCENT)}
    date_tokens = set()  # not relevant here

    def _classify(tok: Token) -> str:
        if id(tok) in money_tokens: return "MONEY"
        if id(tok) in percent_tokens: return "PERCENT"
        try:
            int(tok.text.replace(",", ""))
            return "INTEGER"
        except ValueError:
            pass
        return "TEXT"

    # Find header row: first line where ALL non-empty tokens are TEXT AND
    # the next 2-5 rows contain ≥ 1 MONEY-classified token at a consistent X.
    header_key = None
    for idx, k in enumerate(ordered_keys):
        line_tokens = [t for t in lines[k] if t.text.strip()]
        if not line_tokens: continue
        types = {_classify(t) for t in line_tokens}
        if types != {"TEXT"}: continue
        # Check next 3 rows for MONEY-bearing consistent column
        next_rows = [lines[kk] for kk in ordered_keys[idx+1:idx+4]]
        if any(any(_classify(t) == "MONEY" for t in row) for row in next_rows):
            header_key = k
            break

    if header_key is None:
        return []

    # Stop at first row that has ≥ 2 MONEY tokens and ≥ 1 TEXT token like
    # "subtotal"/"total" position — structurally, a summary row. We identify
    # this by MONEY-column count > data-row MONEY-column count.
    # (Simplified: stop at first row whose MONEY count exceeds header's column count)
    header_toks = sorted([t for t in lines[header_key] if t.text.strip()], key=lambda t: t.anchor.x0)
    columns: list[tuple[int, float]] = [(i, (t.anchor.x0 + t.anchor.x1) / 2) for i, t in enumerate(header_toks)]
    n_cols = len(columns)
    stop_key = None
    for k in ordered_keys:
        if k <= header_key: continue
        row_types = [_classify(t) for t in lines[k]]
        money_count = row_types.count("MONEY")
        if money_count > n_cols:
            stop_key = k
            break

    # Collect rows between header and stop
    items: list[dict[str, ExtractedValue]] = []
    row_tokens: dict[int, list[Token]] = {}
    for k in ordered_keys:
        if k <= header_key: continue
        if stop_key and k >= stop_key: break
        if k[0] != header_key[0]: continue
        row_tokens.setdefault(k[1], []).extend(lines[k])

    seen = set()
    line_no = 1
    for y_bucket in sorted(row_tokens.keys()):
        toks = row_tokens[y_bucket]
        if not toks: continue
        # Assign each token to nearest column by x-midpoint
        cells: dict[int, list[Token]] = {col_idx: [] for col_idx, _ in columns}
        for t in toks:
            mid = (t.anchor.x0 + t.anchor.x1) / 2
            best = min(columns, key=lambda c: abs(c[1] - mid))
            cells[best[0]].append(t)

        # Assign column roles by TYPE signature (no header-text matching)
        col_types: dict[int, str] = {}
        for col_idx, col_toks in cells.items():
            if not col_toks:
                col_types[col_idx] = "EMPTY"; continue
            types_here = [_classify(t) for t in col_toks]
            col_types[col_idx] = max(set(types_here), key=types_here.count)  # majority

        # Description = widest TEXT column. Qty = INTEGER column.
        # Unit price / line total = the two MONEY columns; which is which is
        # decided by arithmetic fit (qty × unit_price ≈ line_total).
        text_cols = [c for c, t in col_types.items() if t == "TEXT"]
        int_cols = [c for c, t in col_types.items() if t == "INTEGER"]
        money_cols = [c for c, t in col_types.items() if t == "MONEY"]

        desc_col = max(text_cols, key=lambda c: sum(len(t.text) for t in cells[c]), default=None)
        qty_col = int_cols[0] if int_cols else None

        desc = " ".join(t.text for t in cells[desc_col]) if desc_col is not None else ""
        qty_val = _parse_num(cells[qty_col]) if qty_col is not None else None

        # Assign the two MONEY columns via arithmetic fit
        price_col, total_col = None, None
        if len(money_cols) >= 2 and qty_val:
            a = _parse_num(cells[money_cols[0]])
            b = _parse_num(cells[money_cols[1]])
            if a is not None and b is not None:
                # qty * a ≈ b?
                if abs(qty_val * a - b) < 0.01:
                    price_col, total_col = money_cols[0], money_cols[1]
                elif abs(qty_val * b - a) < 0.01:
                    price_col, total_col = money_cols[1], money_cols[0]
                else:
                    # Fallback: leftmost = price, rightmost = total
                    price_col, total_col = money_cols[0], money_cols[-1]
        elif len(money_cols) == 1:
            total_col = money_cols[0]

        qty_tok = cells.get(qty_col) if qty_col is not None else None
        price_tok = cells.get(price_col) if price_col is not None else None
        total_tok = cells.get(total_col) if total_col is not None else None
        price_val = _parse_num(price_tok)
        total_val = _parse_num(total_tok)

        if desc and (qty_val is not None or total_val is not None):
            key = (desc, qty_val, price_val, total_val)
            if key in seen: continue
            seen.add(key)
            item: dict[str, ExtractedValue] = {
                "line_no": ExtractedValue(
                    value=line_no, provenance="derived",
                    derivation_trace={"rule_id": "line_no_monotonic", "inputs": {}},
                    source="derivation_registry", confidence=1.0, attempt=1,
                ),
                "item_description": ExtractedValue(
                    value=desc, provenance="extracted", anchor_text=desc,
                    anchor_ref=cells["description"][0].anchor if cells.get("description") else None,
                    source="structural", confidence=1.0, attempt=1,
                ),
            }
            if qty_val is not None:
                item["quantity"] = _ev(qty_val, qty_tok[0])
            if price_val is not None:
                item["unit_price"] = _ev(price_val, price_tok[0])
            if total_val is not None:
                item["line_total"] = _ev(total_val, total_tok[0])
            items.append(item)
            line_no += 1
    return items

def _parse_num(toks):
    if not toks: return None
    txt = "".join(t.text for t in toks).replace(",", "").replace("£", "").replace("$", "").replace("€", "")
    try: return float(txt)
    except ValueError: return None

def _ev(value, tok):
    return ExtractedValue(
        value=value, provenance="extracted", anchor_text=str(value),
        anchor_ref=tok.anchor, source="structural", confidence=1.0, attempt=1,
    )

def _xlsx_line_items(doc): ...  # implement similarly; table.header_row_index identifies header row
def _docx_line_items(doc): ...  # walk doc.tables, rows below header_row_index
def _csv_line_items(doc): ...   # every row after header is a line item
```

(Format-specific branches `_xlsx_line_items`, `_docx_line_items`, `_csv_line_items` implemented in Task 21.)

- [ ] **Step 4: Run test — expect to pass for Duncan PO (PDF spatial)**
- [ ] **Step 5: Commit**

```bash
git commit -am "feat(extraction): PDF spatial line-item extraction with dedup"
```

---

### Task 21: ⚠️ SUPERSEDED — see Tasks 51a/51b/51c

> **DO NOT EXECUTE THIS TASK.** The per-format line-item branches originally drafted here have been split into Tasks 51a (XLSX), 51b (DOCX), 51c (CSV) in Phase 15 after the plan review found the combined task was under-specified. Task 20 writes only `_pdf_line_items`; the other three branches are added by Tasks 51a/b/c.
>
> The content below is retained for historical context only. Skip to Task 22.

### Task 21 (historical, do not implement): Line items — XLSX/DOCX/CSV dispatch

**Files:**
- Modify: `src/services/structural_extractor/extractors/line_items.py`
- Test: extend `test_extractor_line_items.py`

- [ ] **Step 1: Failing tests for each format**

```python
def test_xlsx_line_items():
    from tests.structural_extractor.test_xlsx_parser import _build_sample_xlsx
    from src.services.structural_extractor.parsing.xlsx_parser import parse_xlsx
    doc = parse_xlsx(_build_sample_xlsx(), "x.xlsx")
    items = extract_line_items(doc, "Invoice")
    assert len(items) >= 1
    assert any("Widget" in i["item_description"].value for i in items)

def test_csv_line_items():
    from src.services.structural_extractor.parsing.csv_parser import parse_csv
    data = b"description,qty,unit_price,total\nWidget,10,99.99,999.90\n"
    doc = parse_csv(data, "x.csv")
    items = extract_line_items(doc, "Invoice")
    assert len(items) == 1
    assert items[0]["item_description"].value == "Widget"
    assert items[0]["quantity"].value == 10

def test_docx_line_items():
    from tests.structural_extractor.test_docx_parser import _build_sample_docx
    from src.services.structural_extractor.parsing.docx_parser import parse_docx
    doc = parse_docx(_build_sample_docx(), "x.docx")
    items = extract_line_items(doc, "Invoice")
    assert len(items) == 1
    assert items[0]["item_description"].value == "Widget"
```

- [ ] **Step 2: Implement the 3 branches; Step 3: Pass; Step 4: Commit**

```bash
git commit -am "feat(extraction): XLSX/DOCX/CSV line-item extractors"
```

---

### Task 22: Payment terms extractor

**Files:**
- Create: `src/services/structural_extractor/extractors/payment_terms.py`
- Test: `tests/structural_extractor/test_extractor_payment_terms.py`

- [ ] **Step 1-5: failing test → implement → pass → commit**

```python
# test: the INV600254 invoice mentions "Payment must be made within 30 days"
# extractor returns the phrase verbatim with provenance="extracted"
```

```python
# payment_terms.py
# Anchor tokens: {"Payment Terms", "Payment Due", "Terms"} OR standalone "Net 14", "Net 30", "Net 60", "Net 90"
# Value: the run of tokens after the anchor up to next section-break / 30-char cap
```

```bash
git commit -am "feat(extraction): payment terms extractor"
```

---

## Phase 5 — Derivation Registry

### Task 23: Derivation rule base class + registry

**Files:**
- Create: `src/services/structural_extractor/derivation.py`
- Test: `tests/structural_extractor/test_derivation.py`

- [ ] **Steps 1-5**: Build `DerivationRule` dataclass, registry decorator, topological resolver.

```python
# derivation.py sketch
@dataclass
class DerivationRule:
    rule_id: str
    target_field: str
    inputs: list[str]
    compute: Callable[[dict], Any]

REGISTRY: list[DerivationRule] = []

def rule(rule_id, target_field, inputs):
    def _decorator(fn):
        REGISTRY.append(DerivationRule(rule_id, target_field, inputs, fn))
        return fn
    return _decorator

def resolve_all(header: dict, doc_type: str) -> dict:
    """Runs rules in topological order until no more can fire."""
    ...
```

```bash
git commit -am "feat(extraction): derivation registry framework"
```

---

### Task 24: `due_date` derivation rules

**Files:**
- Modify: `src/services/structural_extractor/derivation.py`
- Test: extend `test_derivation.py`

- [ ] **Steps 1-5**: Implement `due_date_from_terms` (parse "Net N" / "within N days" from payment_terms) and `due_date_default` (invoice_date + 90 days).

```python
@rule("due_date_from_terms", "due_date", ["invoice_date", "payment_terms"])
def _due_from_terms(inputs): ...

@rule("due_date_default", "due_date", ["invoice_date"])
def _due_default(inputs):
    from datetime import timedelta
    return inputs["invoice_date"] + timedelta(days=90)
```

Tests verify:
- "Net 30" → +30 days
- "within 14 days" → +14 days
- No payment_terms → +90 days
- Missing invoice_date → no rule fires (due_date remains unresolved)

```bash
git commit -am "feat(extraction): due_date derivation rules"
```

---

### Task 25: Amount math inversion rules

**Files:**
- Modify: `src/services/structural_extractor/derivation.py`
- Test: extend `test_derivation.py`

- [ ] **Steps 1-5**: Implement:
- `subtotal_from_total_tax`
- `tax_amount_from_pct`
- `total_from_subtotal_tax`
- `tax_pct_from_amounts`

Each with input-availability preconditions + unit tests.

```bash
git commit -am "feat(extraction): amount math inversion rules"
```

---

### Task 26: Currency inference

**Files:** same
- [ ] **Steps 1-5**: `currency_from_symbol` rule (£→GBP etc.) when `currency` field is absent.

```bash
git commit -am "feat(extraction): currency-from-symbol inference"
```

---

### Task 27: Exchange-rate lookup + USD conversion

**Files:**
- Modify: `derivation.py`
- Leverage: existing `services/extraction_validator.py` has a 1h-cached live-rate fetch — extract it to `derivation.py` rather than duplicate.

- [ ] **Steps 1-5**: `xrate_lookup` (open.er-api.com, 1h memcache, 24h fallback) + `convert_to_usd`.

```bash
git commit -am "feat(extraction): FX rate lookup and USD conversion rules"
```

---

### Task 28: Supplier / Buyer lookup + auto-create

**Files:**
- Modify: `derivation.py`
- Test: requires a DB fixture (use test Postgres or mock).

- [ ] **Steps 1-5**: `supplier_id_from_lookup` — SELECT from `proc.bp_supplier` by normalized name; if not found, generate `SUP-{name}` + INSERT new row. Same for `buyer_id`.

Use pytest fixture `db_conn` that spins up a transaction (auto-rollback) so tests don't pollute production.

```bash
git commit -am "feat(extraction): supplier/buyer id lookup with auto-create"
```

---

### Task 29: Country / region inference

**Files:** same.
- [ ] **Steps 1-5**: `country_from_postcode` (UK regex `^[A-Z]{1,2}[0-9]`, US ZIP, EU postal codes), `region_from_address` (parse state/county).

```bash
git commit -am "feat(extraction): country/region inference from address"
```

---

### Task 30: Status defaults + ai_flag_required

**Files:** same.
- [ ] **Steps 1-5**: `invoice_status_default` ("Issued"), `po_status_default` ("Open"), `ai_flag_compute` (Y if any validation warning else N).

```bash
git commit -am "feat(extraction): status defaults + ai_flag rule"
```

---

### Task 31: Line-item derivations

**Files:**
- Modify: `derivation.py` (add line-item scope)
- Test: extend `test_derivation.py`

- [ ] **Steps 1-5**: `line_total_from_qty_price`, `unit_price_from_qty_total`, `quantity_from_price_total` applied per line item.

```bash
git commit -am "feat(extraction): line-item math derivations"
```

---

## Phase 6 — Validation

### Task 32: Validation module

**Files:**
- Create: `src/services/structural_extractor/validation.py`
- Test: `tests/structural_extractor/test_validation.py`

- [ ] **Steps 1-5**: Implement
- `verify_anchors(result)` — every extracted value's anchor text matches its source token
- `verify_math(header, line_items)` — `subtotal + tax = total`, `Σ line_totals = subtotal`, qty*price=line_total per line
- `verify_cross_field(header)` — `invoice_date ≤ due_date`, supplier ≠ buyer, currency consistent

Returns a `ValidationReport(passed: bool, failures: list[str])`.

```bash
git commit -am "feat(extraction): validation (anchor + math + cross-field)"
```

---

## Phase 7 — NLU models (Layer 3)

### Task 33: Thread-safe model registry

**Files:**
- Create: `src/services/structural_extractor/nlu/_registry.py`
- Test: `tests/structural_extractor/test_nlu_registry.py`

- [ ] **Steps 1-5**: Double-checked-locking singleton loader.

```python
class ModelRegistry:
    _instances = {}
    _locks = defaultdict(threading.Lock)
    @classmethod
    def get(cls, name): ...
    @classmethod
    def warm(cls): ...  # eager-load all NLU models at startup
```

Test: mock `_load` and call `get` from 4 threads, assert one call.

```bash
git commit -am "feat(extraction): thread-safe NLU model registry"
```

---

### Task 34: BERT-NER wrapper

**Files:**
- Create: `src/services/structural_extractor/nlu/ner.py`
- Test: `tests/structural_extractor/test_nlu_ner.py`

- [ ] **Steps 1-5**: `NER.run(text) -> list[NERSpan]` with `dslim/bert-base-NER` on CPU. Test: simple sentence, assert ORG entity present.

```bash
git commit -am "feat(extraction): BERT-NER wrapper"
```

---

### Task 35: Table-Transformer wrapper

**Files:**
- Create: `src/services/structural_extractor/nlu/table_transformer.py`
- Test: `tests/structural_extractor/test_nlu_table.py`

- [ ] **Steps 1-5**: Render PDF page at 150 DPI, run `microsoft/table-transformer-structure-recognition`, return bbox-defined table regions. Cap at 3 pages.

```bash
git commit -am "feat(extraction): table-transformer wrapper (PDF page-image based)"
```

---

### Task 36: Layout-YOLO wrapper

**Files:**
- Create: `src/services/structural_extractor/nlu/layout.py`
- Test: `tests/structural_extractor/test_nlu_layout.py`

- [ ] **Steps 1-5**: Wrap `unstructuredio/yolo_x_layout`, return labelled regions (title, text, table, figure, list). Cap at 3 pages.

```bash
git commit -am "feat(extraction): yolo layout wrapper"
```

---

## Phase 8 — LLM fallback

### Task 37: Strict grounded LLM call

**Files:**
- Create: `src/services/structural_extractor/llm_fallback.py`
- Test: `tests/structural_extractor/test_llm_fallback.py`

- [ ] **Steps 1-5**: Implement `extract_fields_with_llm(doc_text, fields_needed, prior_attempts) -> dict[str, str]`.

Prompt template:
```
You are an extraction engine. Below is a procurement document.
Extract ONLY these fields: {fields_needed}.
Return JSON: {"field_name": {"value": "<exact substring>", "anchor": "<5-10 words around the value>"}}.
If a field is not present in the document, return null.
Never invent or calculate values.

Document:
{doc_text}

Prior attempts (fields already found):
{prior_attempts}
```

After calling AgentNick, **verify every returned value is a substring of doc_text** before accepting. Drop values that fail substring check.

Test uses mock Ollama client.

```bash
git commit -am "feat(extraction): strict grounded LLM fallback"
```

---

## Phase 9 — Retry loop (decomposed — reviewer flagged single-task compression as a sub-project)

### Task 38a: RetryState and AttemptOutput dataclasses

**Files:**
- Create: `src/services/structural_extractor/retry/__init__.py` (empty)
- Create: `src/services/structural_extractor/retry/state.py`
- Test: `tests/structural_extractor/test_retry_state.py`

- [ ] **Step 1: Failing test — dataclass shape**

```python
from src.services.structural_extractor.retry.state import (
    AttemptOutput, RetryState
)

def test_attempt_output_fields():
    o = AttemptOutput(attempt=1, source="structural", extracted={}, line_items=None,
                      validation_failures=[], residual_unresolved=["invoice_date"],
                      latency_ms=100)
    assert o.attempt == 1

def test_retry_state_tracks_accepted():
    s = RetryState(doc=None, doc_type="Invoice", target_fields={"invoice_id"},
                   attempts=[], accepted_header={}, accepted_line_items=None,
                   unresolved={"invoice_id"})
    assert "invoice_id" in s.unresolved
```

- [ ] **Steps 2-5**: Implement dataclasses per spec lines 266-289, run tests, commit.

```bash
git commit -am "feat(extraction): RetryState + AttemptOutput dataclasses"
```

---

### Task 38b: Attempt 1 — structural + pattern-cached

**Files:**
- Create: `src/services/structural_extractor/retry/attempts.py`
- Test: `tests/structural_extractor/test_retry_attempt1.py`

- [ ] **Steps 1-5**: Implement `run_attempt_1(state) -> AttemptOutput` that:
1. Calls `extractors.extract_all(doc, doc_type)` (aggregated call across ids/dates/parties/amounts/line_items/payment_terms).
2. Checks pattern-store hit by `layout_signature` → if hit, boosts candidates at cached positions.
3. Runs Derivation Registry to fill derivable fields.
4. Returns AttemptOutput with validated fields, residual unresolved.

Test: on a doc where structural succeeds, assert `attempts == 1` and all required fields resolved.

```bash
git commit -am "feat(extraction): retry attempt 1 — structural + pattern-cached"
```

---

### Task 38c: Attempt 2 — BERT-NER augmentation

**Files:** append to `retry/attempts.py`; test in `test_retry_attempt2.py`

- [ ] **Steps 1-5**: Implement `run_attempt_2(state)`. Runs BERT-NER on `doc.full_text`, emits ORG/DATE/MONEY spans that feed back into Layer 2 discovery as new candidates (alongside structural). Re-runs extractors. Commits.

```bash
git commit -am "feat(extraction): retry attempt 2 — BERT-NER augmentation"
```

---

### Task 38d: Attempt 3 — Table-Transformer for line items

**Files:** same; test `test_retry_attempt3.py`

- [ ] **Steps 1-5**: Run Table-Transformer on page images, emit bbox-defined table regions that override spatial line-item detection. Commit.

```bash
git commit -am "feat(extraction): retry attempt 3 — Table-Transformer"
```

---

### Task 38e: Attempt 4 — Layout-YOLO region assist

**Files:** same; test `test_retry_attempt4.py`

- [ ] **Steps 1-5**: Layout-YOLO labels regions; assist Parties extractor (ORG regions), Line Items (table regions). Commit.

```bash
git commit -am "feat(extraction): retry attempt 4 — Layout-YOLO"
```

---

### Task 38f: Attempts 5-10 — LLM arbiter with strict grounding + prompt variation

**Files:** same; test `test_retry_attempt_llm.py`

- [ ] **Steps 1-5**: Implement `run_attempt_llm(state, attempt_no)`. Calls `llm_fallback.extract_fields_with_llm(doc_text, unresolved_fields, prior_attempts)`. Applies substring verification. On attempts 6-10, vary prompt (temperature micro-jitter, reworded instructions, add prior-failure examples). Commit.

```bash
git commit -am "feat(extraction): retry attempts 5-10 — LLM arbiter"
```

---

### Task 38g: Attempt merge function + conflict resolution

**Files:**
- Create: `src/services/structural_extractor/retry/merge.py`
- Test: `tests/structural_extractor/test_retry_merge.py`

- [ ] **Steps 1-5**: Implement `merge_attempt_into_state(state, attempt_output) -> RetryState` per spec lines 292-296 (identical math → keep earlier; arithmetic-tied disagreement → replace group; unrelated disagreement → re-extract before commit). Test cases for each conflict type.

```bash
git commit -am "feat(extraction): cross-attempt conflict resolution"
```

---

### Task 38h: Field-level residual tracking

**Files:** modify `retry/state.py`; test `test_retry_field_residual.py`

- [ ] **Steps 1-5**: Implement `RetryState.residual_fields() -> set[str]` which returns `target_fields - accepted_header.keys() - (line_items accepted)`. Ensure attempts 2+ only work on residuals. Test: attempt 1 resolves 3 of 4 fields; attempt 2's input is exactly the 1 remaining field.

```bash
git commit -am "feat(extraction): field-level residual tracking"
```

---

### Task 38i: Retry driver + review-queue escalation

**Files:**
- Create: `src/services/structural_extractor/retry/driver.py`
- Test: `tests/structural_extractor/test_retry_driver.py`

- [ ] **Steps 1-5**: Implement `run_retry_loop(doc, doc_type, max_attempts=10) -> ExtractionResult`. Loops 1..10 calling the appropriate attempt runner, merging via Task 38g, checking residuals. If residuals remain after 10, calls `review_queue.park_in_review_queue(state)` and returns `ExtractionResult` with `unresolved_fields` populated.

Test: happy path (attempt 1 succeeds), medium path (attempts 1-3 needed), unhappy path (10 attempts, residuals remain → queue).

```bash
git commit -am "feat(extraction): retry driver + review-queue escalation"
```

---

## Phase 10 — Pattern learning (Layer 4)

### Task 39: DB migration — extend bp_extraction_patterns + add type_priors

**Files:**
- Create: `scripts/migrations/2026-04-21-engineered-extraction.sql`
- Test: migration runs cleanly and is reversible.

- [ ] **Step 1: Write migration**

```sql
-- 2026-04-21-engineered-extraction.sql
ALTER TABLE proc.bp_extraction_patterns
    ADD COLUMN IF NOT EXISTS anchor_patterns JSONB;

CREATE TABLE IF NOT EXISTS proc.bp_extraction_type_priors (
    doc_type     TEXT PRIMARY KEY,
    priors       JSONB NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS proc.extraction_review_queue (
    id                   BIGSERIAL PRIMARY KEY,
    process_monitor_id   INT REFERENCES proc.process_monitor(id) ON DELETE CASCADE,
    file_path            TEXT NOT NULL,
    doc_type             TEXT NOT NULL,
    partial_header       JSONB,
    partial_line_items   JSONB,
    failed_fields        TEXT[],
    parsed_text          TEXT,
    attempt_count        INT NOT NULL DEFAULT 0,
    last_attempt_at      TIMESTAMPTZ,
    signals_json         JSONB,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at          TIMESTAMPTZ,
    resolved_by          TEXT
);
CREATE INDEX IF NOT EXISTS idx_eq_unresolved ON proc.extraction_review_queue (resolved_at) WHERE resolved_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_eq_doc_type ON proc.extraction_review_queue (doc_type);

CREATE TABLE IF NOT EXISTS proc.bp_extraction_provenance (
    id                  BIGSERIAL PRIMARY KEY,
    parent_table        TEXT NOT NULL,
    parent_pk           TEXT NOT NULL,
    field_name          TEXT NOT NULL,
    source              TEXT NOT NULL,
    anchor_ref          JSONB,
    derivation_trace    JSONB,
    confidence          NUMERIC(3,2),
    attempt             INT,
    extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_prov_parent ON proc.bp_extraction_provenance (parent_table, parent_pk);
```

- [ ] **Step 2: Run against dev DB + commit**

```bash
export $(grep -v '^#' .env | grep -E '^DB_' | xargs) && PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f scripts/migrations/2026-04-21-engineered-extraction.sql
git add scripts/migrations/2026-04-21-engineered-extraction.sql
git commit -m "feat(db): migration — anchor_patterns + type_priors + review_queue + provenance"
```

---

### Task 40a: Pattern-store extension — read/write anchor_patterns

**Files:**
- Create: `src/services/structural_extractor/pattern_store.py`
- Test: `tests/structural_extractor/test_pattern_store.py`

- [ ] **Step 1: Failing test**

```python
def test_save_and_get_pattern_anchors(db_conn):
    from src.services.structural_extractor.pattern_store import (
        get_pattern_anchors, save_pattern_anchors
    )
    save_pattern_anchors(db_conn, "pdf", "Invoice", "Acme Corp", "abc123",
                         {"invoice_id": {"pdf": {"page": 1, "x0": 420, "y0": 48}}})
    anchors = get_pattern_anchors(db_conn, "pdf", "Invoice", "Acme Corp", "abc123")
    assert anchors["invoice_id"]["pdf"]["page"] == 1
```

- [ ] **Steps 2-5**: Implement wrapper around existing `ExtractionPatternStore`, add the two new methods. Test covers INSERT-new, UPDATE-existing, and cache-miss-returns-None. Commit.

```bash
git commit -am "feat(extraction): pattern-store anchor persistence"
```

---

### Task 40b: Pattern trust levels (none / learning / trusted)

**Files:**
- Modify: `pattern_store.py`
- Test: extend `test_pattern_store.py`

- [ ] **Steps 1-5**: Implement `get_trust_level(success_count) -> "none"|"learning"|"trusted"` per spec lines 516-522. Demotion logic: on validation failure with a trusted pattern, reset success_count to 2. Test covers all three tiers + demotion.

```bash
git commit -am "feat(extraction): pattern trust levels with demotion"
```

---

### Task 40c: Type priors table writes + running-mean update

**Files:**
- Modify: `pattern_store.py`
- Test: `test_pattern_store.py`

- [ ] **Steps 1-5**: Implement `update_type_priors(doc_type, field_name, position)` using incremental running-mean formula: `new_mean = old_mean + (new_sample - old_mean) / n`. Persists to `proc.bp_extraction_type_priors`. Test verifies mean converges with repeated samples.

```bash
git commit -am "feat(extraction): type priors with running-mean updates"
```

---

### Task 40d: Pattern write-back on successful extraction

**Files:**
- Create: `src/services/structural_extractor/retry/learn.py`
- Test: `tests/structural_extractor/test_retry_learn.py`

- [ ] **Steps 1-5**: `persist_pattern(result, doc)` — called after `run_retry_loop` when `unresolved_fields` is empty and validation passed. Writes/updates `bp_extraction_patterns` with new anchor positions; increments `success_count`; calls `update_type_priors` for each field. Hooked into `run_retry_loop` in Task 38i.

```bash
git commit -am "feat(extraction): pattern learning write-back on success"
```

---

### Task 40e: Candidate ranking formula (Layer 2 Step 2)

**Files:**
- Create: `src/services/structural_extractor/discovery/ranking.py`
- Test: `tests/structural_extractor/test_ranking.py`

- [ ] **Step 1: Failing test**

```python
def test_score_weights_sum_to_one():
    from src.services.structural_extractor.discovery.ranking import WEIGHTS
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6

def test_tie_breaker_earlier_position_wins():
    from src.services.structural_extractor.discovery.ranking import rank_candidates
    # Two candidates with identical scores, different positions
    # Earlier one wins
    ...
```

- [ ] **Steps 2-5**: Implement the 5-signal weighted-sum formula (spec lines 322-330): `0.40 × pattern_hit + 0.25 × positional_prior + 0.20 × arithmetic_fit + 0.10 × uniqueness + 0.05 × label_semantic_similarity`. Tie-breaker: earlier doc position. Call from per-field extractors (refactor Tasks 16-19 to use this ranker instead of their ad-hoc overlap scoring).

```bash
git commit -am "feat(extraction): weighted-sum candidate ranking formula"
```

---

## Phase 11 — Provenance

### Task 41: Provenance writer

**Files:**
- Create: `src/services/structural_extractor/provenance.py`
- Test: `tests/structural_extractor/test_provenance.py`

- [ ] **Steps 1-5**: `write_provenance(db_conn, parent_table, parent_pk, extracted_values)` — one INSERT per field, with anchor_ref JSON-serialized or derivation_trace.

```bash
git commit -am "feat(extraction): provenance writer"
```

---

## Phase 12 — Review queue

### Task 42a: Review-queue writer

**Files:**
- Create: `src/services/structural_extractor/review_queue.py`
- Test: `tests/structural_extractor/test_review_queue.py`

- [ ] **Steps 1-5**: Implement `park_in_review_queue(db_conn, result, process_monitor_id)` — INSERTs into `proc.extraction_review_queue`; UPDATEs `proc.process_monitor.status = 'Extraction_InReview'`. Test covers both writes within one transaction.

```bash
git commit -am "feat(extraction): review queue writer"
```

---

### Task 42b: Watcher recovery-sweep & poll-query carve-out

**Files:**
- Modify: `src/services/process_monitor_watcher.py` lines 109, 618, 634 (from spec)
- Test: `tests/structural_extractor/test_watcher_integration.py`

- [ ] **Steps 1-5**: Update recovery sweep to leave `Extraction_InReview` rows alone; verify poll and LISTEN predicates unchanged. Test: seed a row with `Extraction_InReview`, restart-simulate, assert status unchanged.

```bash
git commit -am "feat(extraction): watcher carve-out for Extraction_InReview"
```

---

### Task 42c: Review-queue resolution trigger

**Files:**
- Modify: `scripts/migrations/2026-04-21-engineered-extraction.sql` (add trigger)
- Test: `tests/structural_extractor/test_review_queue_resolution.py`

- [ ] **Step 1: Failing test — resolving a queue entry flips process_monitor back to Completed**

```python
def test_resolving_queue_entry_resets_process_monitor(db_conn):
    # Insert a Process_Monitor row in Extraction_InReview + a queue entry;
    # UPDATE queue.resolved_at = NOW();
    # Assert process_monitor.status is now 'Completed'
    ...
```

- [ ] **Step 2: Add trigger to migration**

```sql
CREATE OR REPLACE FUNCTION proc.fn_resolve_extraction_review() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.resolved_at IS NOT NULL AND OLD.resolved_at IS NULL THEN
        UPDATE proc.process_monitor
           SET status='Completed', start_ts=NULL, end_ts=NULL
         WHERE id = NEW.process_monitor_id;
    END IF;
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_resolve_extraction_review
AFTER UPDATE ON proc.extraction_review_queue
FOR EACH ROW EXECUTE FUNCTION proc.fn_resolve_extraction_review();
```

- [ ] **Steps 3-5**: Re-run migration, test passes, commit.

```bash
git commit -am "feat(db): resolution trigger for extraction_review_queue"
```

---

## Phase 13 — Orchestrator integration

### Task 43: Public API — `extract()`

**Files:**
- Modify: `src/services/structural_extractor/__init__.py`
- Test: `tests/structural_extractor/test_public_api.py`

- [ ] **Steps 1-5**: Wire everything — `extract(file_bytes, filename, doc_type)` runs parse → retry → return `ExtractionResult`.

```python
# __init__.py
from src.services.structural_extractor.parsing import parse
from src.services.structural_extractor.retry import run_retry_loop
from src.services.structural_extractor.types import ExtractionResult

def extract(file_bytes: bytes, filename: str, doc_type: str) -> ExtractionResult:
    doc = parse(file_bytes, filename)
    return run_retry_loop(doc, doc_type, max_attempts=10)
```

```bash
git commit -am "feat(extraction): public extract() API"
```

---

### Task 44: Orchestrator integration (feature-flagged)

**Files:**
- Modify: `src/services/agent_nick_orchestrator.py` (method `_dispatch_extraction` around line 361)
- Test: `tests/structural_extractor/test_orchestrator_integration.py`

- [ ] **Steps 1-5**: Wrap call behind `USE_STRUCTURAL_EXTRACTOR` env var; preserve `_legacy_dispatch` path for rollback. Implement `_park_in_review_queue` helper.

```bash
git commit -am "feat(extraction): orchestrator integration behind feature flag"
```

---

## Phase 14 — Testing & rollout

### Task 45: Golden-set fixtures

**Files:**
- Copy PDFs from S3 + `/tmp/src/`:
  - 20 PDFs: 10 Aquarius + 10 CityOfNewport + 4 from audit
  - 3 multi-page + 3 edge-case PDFs
- Synthesize DOCX (6), XLSX (4), CSV (2) samples
- Create: `tests/structural_extractor/fixtures/ground_truth.yaml` with hand-labeled expected values per doc

- [ ] **Steps 1-5**: Populate fixtures and ground_truth.yaml. Commit.

```bash
git commit -am "test(extraction): golden set fixtures and ground truth"
```

---

### Task 46: End-to-end golden-set test

**Files:**
- Create: `tests/structural_extractor/test_full_extraction.py`

- [ ] **Steps 1-5**: Loops over every doc in `fixtures/docs/`, loads ground-truth, runs `extract()`, asserts every GROUND-TRUTH field matches; unresolved fields only count as failures if they appear in the ground truth.

```python
@pytest.mark.parametrize("doc_file,ground_truth", _load_golden_set())
def test_golden_doc_extraction(doc_file, ground_truth):
    result = extract(doc_file.read_bytes(), doc_file.name, ground_truth["doc_type"])
    # Every field the ground truth says SHOULD be present must be present and match.
    for field, expected in ground_truth["header"].items():
        if expected is None:
            # Ground truth says NULL — value must either not be in result.header
            # OR be explicitly NULL. (It's legitimately absent.)
            if field in result.header:
                assert result.header[field].value is None, f"{doc_file.name}.{field}: expected NULL, got {result.header[field].value}"
        else:
            assert field in result.header, f"{doc_file.name}.{field} missing"
            assert result.header[field].value == expected, f"{doc_file.name}.{field}"
    # Unresolved only counts if the field was in ground truth and non-null.
    gt_required = {f for f, v in ground_truth["header"].items() if v is not None}
    unresolved_required = set(result.unresolved_fields) & gt_required
    assert not unresolved_required, f"{doc_file.name}: required fields unresolved: {unresolved_required}"
```

```bash
git commit -am "test(extraction): full golden-set end-to-end test"
```

---

### Task 47: Regression test against existing DB invoices

**Files:**
- Create: `tests/structural_extractor/test_regression_existing_docs.py`

- [ ] **Steps 1-5**: Query the 20 morning-batch invoices from `proc.bp_invoice`, re-extract them with the new pipeline, compare field-by-field, assert only `value_not_anchored`-flagged fields change (in a good direction).

```bash
git commit -am "test(extraction): regression test against existing DB rows"
```

---

### Task 48: CI gate + flag rollout

**Files:**
- Modify: `.github/workflows/ci.yml` (if exists) or `pytest.ini`
- Modify: `.env`

- [ ] **Steps 1-5**: Gate merge on `test_full_extraction` passing. Default `USE_STRUCTURAL_EXTRACTOR=true` in dev, `false` in staging/prod until post-merge validation.

```bash
git commit -am "ci(extraction): gate merge on golden-set + flag rollout config"
```

---

## Phase 15 — Coverage gaps identified during plan review (added after first review round)

### Task 49: Incoterm / delivery address extractor (PO-specific)

**Files:**
- Create: `src/services/structural_extractor/extractors/delivery.py`
- Test: `tests/structural_extractor/test_extractor_delivery.py`

- [ ] **Steps 1-5**: Extract `incoterm` (EXW/DDP/FOB/etc — 3-letter uppercase token + optional location), `delivery_address_line1`, `delivery_address_line2`, `delivery_city`, `postal_code`. Same discovery approach as Parties (anchor on `{"Ship To", "Delivery", "Incoterm", "Delivery Terms"}`). Output has `CellRef`/`BBox`/`NodeRef` anchor.

```bash
git commit -am "feat(extraction): incoterm + delivery address extractor"
```

---

### Task 50: ModelRegistry warm-up at procwise startup

**Files:**
- Modify: `src/api/main.py` (startup event)
- Test: `tests/structural_extractor/test_model_warmup.py`

- [ ] **Steps 1-5**: Call `ModelRegistry.warm()` in the FastAPI startup handler, behind `USE_STRUCTURAL_EXTRACTOR` flag. Test: mock `_load`, assert called once per model during startup; subsequent `get()` calls don't reload.

```bash
git commit -am "feat(extraction): warm NLU registry at service startup"
```

---

### Task 51: Split Task 21 into per-format line-item tasks (reviewer objection)

Previous Task 21 combined XLSX+DOCX+CSV line-item extraction into one task. Split now into:

**Task 51a: XLSX line-item extractor** — test with multi-sheet + merged-cells fixture, implement `_xlsx_line_items`, commit.

**Task 51b: DOCX line-item extractor** — test with native-tables fixture + paragraph-layout fallback, implement `_docx_line_items`, commit.

**Task 51c: CSV line-item extractor** — test with header + headerless CSV, implement `_csv_line_items`, commit.

(Task 21's stub `...` branches should be removed; these three replace it.)

---

### Task 52: Split Task 32 validation into three

**Task 52a: Anchor verification** — for every `ExtractedValue` with `provenance='extracted'`, assert the `anchor_ref`'s source Token(s) have `.text` matching the value's raw form. Commit.

**Task 52b: Math verification** — implement the arithmetic invariants (line_total = qty × price; Σ line_totals = subtotal; subtotal + tax = total). Commit.

**Task 52c: Cross-field verification** — `invoice_date ≤ due_date` (when both present), supplier ≠ buyer at normalized-org level, currency consistent across all amounts. Commit.

---

### Task 53: Split Task 37 LLM fallback into three

**Task 53a: Grounded prompt + JSON parse** — prompt template, JSON parsing, handle malformed output. Commit.

**Task 53b: Substring verification layer** — every LLM-returned value must be substring of doc text; drop ungrounded values. Commit.

**Task 53c: Prompt variation across attempts 6-10** — temperature jitter, reworded instructions, accumulated do-not-invent examples from prior failures. Commit.

---

### Task 54: Split Task 23 derivation framework into three

**Task 54a: DerivationRule dataclass + registry decorator** — Commit.

**Task 54b: Topological resolver** — detects cycles, resolves in order, handles missing inputs gracefully. Commit.

**Task 54c: Unit tests covering cycle detection, partial resolution, priority of extracted-over-derived values.** Commit.

---

### Task 55: Split Task 28 supplier/buyer lookup into three

**Task 55a: Name normalization helper** — strip whitespace/punctuation, lowercase, expand common abbreviations. Commit.

**Task 55b: Lookup-only mode** — `supplier_id_from_lookup(name)` → returns `None` on miss. Commit.

**Task 55c: Auto-create mode** — on miss, INSERT new `bp_supplier` row + return generated id. Transactional test fixture. Commit.

---

### Task 57: Wire provenance writer into extract() API

**Files:**
- Modify: `src/services/structural_extractor/__init__.py` (the `extract()` function from Task 43)
- Modify: `src/services/structural_extractor/retry/driver.py` (Task 38i)
- Test: `tests/structural_extractor/test_provenance_wiring.py`

- [ ] **Step 1: Failing test**

```python
def test_extract_writes_provenance(db_conn, sample_pdf_bytes):
    from src.services.structural_extractor import extract
    from src.services.structural_extractor.provenance import count_rows_for
    result = extract(sample_pdf_bytes, "INV600254.pdf", "Invoice", db_conn=db_conn)
    # One provenance row per extracted / derived field
    n = count_rows_for(db_conn, parent_table="bp_invoice", parent_pk=result.header["invoice_id"].value)
    assert n >= 6  # at least 6 non-NULL non-SYSTEM fields
```

- [ ] **Steps 2-5**: After `run_retry_loop` returns, call `provenance.write_provenance(db_conn, "bp_invoice", pk, result.header)` and same for line items into `bp_invoice_line_items`. Pass `db_conn` down from `extract()` → `run_retry_loop` → provenance writer. Commit.

```bash
git commit -am "feat(extraction): wire provenance writer into extract() pipeline"
```

---

### Task 56: Golden-set fixture acquisition & ground-truth authoring (real work, days not minutes)

**Files:**
- Create: `tests/structural_extractor/fixtures/docs/*.pdf/.docx/.xlsx/.csv`
- Create: `tests/structural_extractor/fixtures/ground_truth.yaml`

- [ ] **Step 1: Fetch PDFs from S3 to the repo fixtures dir** (32 total as listed in spec Testing section).
- [ ] **Step 2: Synthesize 6 DOCX / 4 XLSX / 2 CSV fixtures** using Python scripts committed under `tests/structural_extractor/fixtures/generators/`. Reviewable + reproducible.
- [ ] **Step 3: Hand-author ground_truth.yaml** with expected values per doc (requires viewing each source).
- [ ] **Step 4: Commit all fixtures + ground truth + generator scripts** in a dedicated commit (large but reviewable).
- [ ] **Step 5: Second-engineer review of ground truth** via PR comments (correctness matters — a wrong ground-truth makes tests lie).

**This task is explicitly days of work, not minutes.** It is a precondition for Task 46 (end-to-end test). Surface this estimate to the user during execution.

```bash
git commit -am "test(extraction): golden-set fixtures and ground truth (32 docs)"
```

---

## Summary

- **64 tasks** across 15 phases (up from original 48 after reviewer feedback).
- **File count**: ~25 new source files, ~25 test files, 1 SQL migration.
- **Commit cadence**: every task produces at least one commit; average 2-3 commits per task with TDD cycle.
- **Integration gate**: full golden-set test (Task 46) must pass before feature flag is flipped in production.
- **Rollback**: feature flag `USE_STRUCTURAL_EXTRACTOR=false` restores legacy path; no DB changes need to be rolled back (new tables remain, just not populated).

## Skills to use during execution

- `superpowers:test-driven-development` — every task is a Red→Green→Refactor cycle.
- `superpowers:verification-before-completion` — before marking any task done, verify the command outputs match expected.
- `superpowers:systematic-debugging` — any test failure escalates through the systematic-debugging phases.
