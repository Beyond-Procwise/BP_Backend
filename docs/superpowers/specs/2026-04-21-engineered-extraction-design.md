# Universal Engineered Extraction — Multi-Format, Pattern-Learning, LLM-Fallback

**Date**: 2026-04-21
**Status**: Draft v2 (pending review)
**Owner**: ProcWise extraction team
**Author**: AgentNick pair-programming session

## Problem statement

Procurement documents arrive in **four file formats** (PDF, DOCX, Excel, CSV) and **multiple document types** (Invoice, Purchase Order, Quote, Contract). Each vendor uses a different layout, label vocabulary, date format, and column arrangement. The extraction pipeline must convert any of these into a uniform structured record with **100% accuracy across every field**, regardless of format or layout. The system must never assume a predefined set of labels or layouts — it must *learn* patterns from the documents themselves.

Today's pipeline relies on an LLM (finetuned AgentNick on Qwen2.5-7B-Instruct + QLoRA adapter) as the primary extractor. Audit of four production PDFs on 2026-04-21 revealed:

- **Line-item hallucination** (critical): Duncan PO526800's three "Fellowes Office Chair" line items were extracted as a single "Staedtler Ballpoint Pen" row — content invented, not present in source. INV600254's line description became "Monthly Fee for Marketing Services" (not in source) with amount £1,879 instead of the source's £8,333.
- **Date errors**: INV600254 invoice_date empty (source: "August 22, 2019"); INV-2025-290 invoice_date off by 9 days (source: "1 April 2020" → extracted 2020-04-10).
- **Name truncation**: Duncan PO supplier_name became "Duncan" instead of "Duncan LLC".
- **Line-item duplication**: DHA-2025-143's single row appeared twice.

Root cause: the LLM, despite procurement finetuning, regurgitates training-data patterns when a new document is structurally similar but has different content. It cannot separate "this document's values" from "values from similar training documents". The fix is to anchor every extracted value back to a specific source location (pixel bbox, cell coordinate, column index) and let the LLM only fill residual gaps.

## Goal

Extract **every entity** (dates, names, IDs, amounts, line items, payment terms, addresses, currency) from every procurement document with **100% accuracy — no deviation**, across **four file formats** (PDF, DOCX, Excel, CSV) and **three primary doc types** (Invoice, Purchase Order, Quote; Contract handled later). No field may be persisted to the database unless the value is anchored to a specific source location and, where arithmetic applies, mathematically reconciled.

## Non-goals

- OCR quality improvements for scanned PDFs (the extractor consumes whatever text OCR produces; scanned docs degrade to the OCR'd-text path).
- Training a custom extraction / NER model from scratch (weeks of R&D — deferred; we use pre-trained HF models off the shelf).
- DB schema changes beyond (a) one new review-queue table and (b) extending the existing `proc.bp_extraction_patterns` with per-field anchor position columns.
- Changes to the QLoRA adapter or Modelfile (LLM stays as-is; it becomes a graceful fallback for unresolved fields).
- Changes to unrelated agents (opportunity, supplier interaction, etc.).
- Handling image-only documents (JPG/PNG of scanned receipts) — these require OCR-first and are treated as PDFs after the OCR path.

## Guiding principles

Every value written to `proc.bp_invoice`, `proc.bp_purchase_order`, `proc.bp_quote`, `proc.bp_contracts`, or the corresponding line-item tables must satisfy all of:

1. **Source-location-anchored, not substring-anchored**. Every value carries an `AnchorRef` pointing to a specific location in the source document: a `BBox` for PDF, a `CellRef` (sheet + row + column) for Excel, a `ColumnRef` (row + column-index) for CSV, or a `NodeRef` (paragraph index + table/row/cell) for DOCX. Verification is: "the value's normalized form equals the normalized form of the source location's text" — a local equality check, not a global substring scan. This prevents spurious matches on unrelated digits (phone numbers, IDs, years, line numbers). Dates: the anchor must point to a human-format date token (e.g. "August 22, 2019" or "01/07/2022" or the XLSX cell containing a date); the YYYY-MM-DD normalized value is only for storage, never required to appear in source.
2. **Extraction first, derivation second, NULL only as last resort; no hardcoded label vocabulary**. A value comes from one of four provenance categories, in priority order:
   - (a) **extracted** — has a direct anchor to a source location in the document;
   - (b) **derived** — computed from other extracted values via a deterministic, tested rule registered in the Derivation Registry (Section: Derivation Rules). Example: `due_date = invoice_date + payment_terms_days` where `payment_terms_days` defaults to 90 when no explicit payment terms are found. Derivation itself is 100% accurate because the rule is deterministic and its inputs are already extraction-validated.
   - (c) **inferred** — computed from contextual signals (e.g. `country` from UK postcode pattern in the address; `currency` from `£` symbol; `supplier_id` generated from `supplier_name`). Inference rules are also in the Derivation Registry.
   - (d) **lookup** — retrieved from another DB table (e.g. `buyer_id` looked up from supplier master by company name; `exchange_rate_to_usd` from live FX API).
   
   A value is written as NULL **only if** none of (a)-(d) can produce a value from the available context — and the field is not required by the schema.
   
   Every stored value carries its provenance (`extracted` | `derived` | `inferred` | `lookup`) plus the inputs that produced it, for audit trail.
   
   The extractor does NOT ship with a fixed set of labels like `{"Invoice No", "INV No", "Bill No"}`; labels are **discovered per document** via type-driven NLU (see Type-Driven Anchor Discovery) and **remembered per vendor+layout** (see Pattern Learning). The only hardcoded knowledge is the **schema field type** for each target field (e.g. `invoice_date: DATE`, `invoice_total_incl_tax: MONEY`, `supplier_id: ORG`) plus the Derivation Registry's rule set.
3. **Math-reconciled**. Where arithmetic exists, the values must reconcile within £0.01:
   - `quantity × unit_price = line_total` for each line.
   - `Σ line_totals = subtotal` (header).
   - `subtotal + tax_amount = total_amount_incl_tax`.
   - `tax_amount ≈ subtotal × (tax_percent / 100)`.
4. **Cross-field-validated**. `invoice_date ≤ due_date` (only when both present); `order_date ≤ expected_delivery_date`; supplier's organization-name token-set disjoint from buyer's; currency consistent across all amounts in the same document.
5. **Attempt-level atomicity**. Values produced within one retry attempt form an atomic group. A later attempt can **supersede** an earlier attempt's group entirely (all-or-nothing for the fields it re-extracts), but fields from different attempts can only coexist if they have **no arithmetic relationship** (e.g., attempt 1's `invoice_id` + attempt 3's `line_items` is fine; attempt 1's `subtotal` + attempt 3's `line_items` requires re-validating the arithmetic and fails to a new attempt if math disagrees).
6. **Fail loud, not silent**. If the above fail for a required field after the retry loop exhausts fast attempts (10 retries), the record is parked in `proc.extraction_review_queue` with partial values + parsed text + failed field names — never persisted as `Extracted` with silent wrong values.
7. **Format-agnostic pipeline**. Every post-parsing stage of the pipeline (anchor discovery, validation, retry, LLM fallback, pattern learning) operates on a unified `ParsedDocument` abstraction. Format-specific logic lives only in the four parser adapters (Layer 1); everything downstream treats all formats uniformly.

## Architecture — four layers

```
  Layer 1: Ingestion adapters            (format-specific)
       │                                  PDF / DOCX / XLSX / CSV  →  ParsedDocument
       ▼
  Layer 2: Type-driven anchor discovery  (format-agnostic, no hardcoded labels)
       │                                  + Pattern-store lookup (fast path when layout is known)
       ▼
  Layer 3: Retry loop                    (layered NLU → LLM arbiter)
       │
       ▼
  Layer 4: Pattern learning              (persist anchor positions for future runs)
       │
       ▼
                  ExtractionResult → orchestrator → DB
```

### Module layout

```
src/services/structural_extractor/
    __init__.py                          — public API: extract(file_bytes, filename, doc_type)
    parsing/                             — LAYER 1: format adapters, each produces ParsedDocument
        __init__.py                      — detect_format() + dispatch
        pdf_parser.py                    — PyMuPDF word bboxes + pdfplumber tables
        docx_parser.py                   — python-docx: paragraphs, tables, runs
        xlsx_parser.py                   — openpyxl: cells, merged ranges, multi-sheet
        csv_parser.py                    — pandas / csv: header row + data rows
        model.py                         — ParsedDocument, Token, Region, Table, AnchorRef types
    discovery/                           — LAYER 2: type-driven anchor discovery
        schema.py                        — schema field types per doc_type (the ONLY hardcoded knowledge)
        type_entities.py                 — emit typed entities (DATE, MONEY, ORG, ID, TEXT) from tokens
        proximity.py                     — label-value proximity inference (no hardcoded labels)
        layout_fingerprint.py            — deterministic hash of layout features for pattern lookup
    extractors/                          — per-field extractors (format-agnostic, work on ParsedDocument)
        ids.py                           — invoice_id, po_id, quote_id, contract_id
        dates.py                         — all dates with per-doc locale detection
        parties.py                       — supplier, buyer, addresses
        amounts.py                       — subtotal, tax, total, tax_pct, currency
        line_items.py                    — format-dispatched table parsing (PDF spatial vs XLSX cells vs DOCX rows vs CSV rows)
        payment_terms.py                 — extracts literal payment-terms text only; due_date derivation lives in derivation.py
    derivation.py                        — Derivation Registry: deterministic rules to fill derivable fields (due_date, tax_pct, currency-from-symbol, etc.)
    provenance.py                        — Writes per-field provenance rows to proc.bp_extraction_provenance
    nlu/                                 — LAYER 3 escalation: pre-trained models
        _registry.py                     — thread-safe lazy singleton, CPU placement
        ner.py                           — BERT-NER (dslim/bert-base-NER) for ORG/PERSON/DATE/MONEY/MISC
        table_transformer.py             — microsoft/table-transformer-structure-recognition
        layout.py                        — unstructuredio/yolo_x_layout for field regions
    llm_fallback.py                      — strict grounded LLM call (AgentNick) with anchor verification
    retry.py                             — progressive retry loop (Section: Retry Strategy)
    validation.py                        — math reconciliation, cross-field checks, anchor verification
    pattern_store.py                     — LAYER 4: extends ExtractionPatternStore with per-field anchors
    review_queue.py                      — persistence to proc.extraction_review_queue

tests/structural_extractor/
    fixtures/
        docs/                            — golden docs across all 4 formats (see Testing section)
        ground_truth.yaml                — hand-labeled expected values per doc
    test_parsing_pdf.py
    test_parsing_docx.py
    test_parsing_xlsx.py
    test_parsing_csv.py
    test_discovery.py
    test_extractors/                     — per-field unit tests
    test_pattern_store.py
    test_retry.py
    test_full_extraction.py              — end-to-end per-format on golden set, asserts 100% field match
```

### Unified data model (Layer 1 output)

The four parser adapters all produce the same `ParsedDocument` shape. Format-specific concepts are absorbed into a discriminated `AnchorRef`:

```python
# Format-specific anchor references (one per ingestion format)
@dataclass(frozen=True)
class BBox:                          # PDF
    page: int                        # 1-indexed
    x0: float; y0: float; x1: float; y1: float

@dataclass(frozen=True)
class CellRef:                       # XLSX
    sheet: str
    row: int                         # 1-indexed, per openpyxl
    col: int                         # 1-indexed
    merged_range: Optional[str] = None   # e.g. "A1:C1" if cell is part of merged range

@dataclass(frozen=True)
class ColumnRef:                     # CSV
    row: int                         # 0-indexed; -1 = header row
    col: int                         # 0-indexed
    column_name: str                 # header text, empty string if no header

@dataclass(frozen=True)
class NodeRef:                       # DOCX
    kind: Literal["paragraph", "table_cell"]
    paragraph_index: Optional[int] = None    # for kind="paragraph"
    table_index: Optional[int] = None        # for kind="table_cell"
    row: Optional[int] = None                # for kind="table_cell"
    col: Optional[int] = None                # for kind="table_cell"

AnchorRef = Union[BBox, CellRef, ColumnRef, NodeRef]

# Unified token: every atomic unit of text across all formats
@dataclass(frozen=True)
class Token:
    text: str
    anchor: AnchorRef                # where in the source file this token came from
    # format-specific metadata
    block_no: Optional[int] = None   # PDF: block group; DOCX: paragraph group; else None
    line_no: Optional[int] = None    # PDF/DOCX: visual line; XLSX/CSV: row
    order: int = 0                   # canonical linearization order for full_text construction

@dataclass
class Region:
    """A rectangular / cell-range grouping of Tokens. Abstraction for 'block', 'cell', 'paragraph'."""
    tokens: list[Token]
    kind: Literal["paragraph", "cell", "block", "row", "column"]
    label: Optional[str] = None      # e.g. column name for CSV/XLSX

@dataclass
class Table:
    """Native table when the format provides one (DOCX tables, XLSX sheets, CSV rows)."""
    rows: list[list[Region]]         # rows x cells
    header_row_index: Optional[int] = None
    source_anchor: AnchorRef

@dataclass
class ParsedDocument:
    source_format: Literal["pdf", "docx", "xlsx", "csv"]
    filename: str
    tokens: list[Token]              # ALL tokens, ordered (linearizable to full_text)
    regions: list[Region]            # paragraphs / cells / blocks
    tables: list[Table]              # natively-structured tables
    pages_or_sheets: int             # PDF: page count; XLSX: sheet count; DOCX: 1; CSV: 1
    full_text: str                   # canonical linearized text (for NLU/LLM)
    raw_bytes: bytes                 # original file (for table-transformer which needs page images)

# Extracted value — carries either an anchor (for extracted values) or a derivation trace
@dataclass
class ExtractedValue:
    value: Any                       # typed per field (str | float | date | int)
    provenance: Literal["extracted", "derived", "inferred", "lookup"]
    # For provenance='extracted':
    anchor_text: Optional[str] = None                # exact source tokens (joined)
    anchor_ref: Optional[AnchorRef] = None           # source-location reference
    # For provenance in ('derived','inferred','lookup'):
    derivation_trace: Optional[dict] = None          # {rule_id, inputs: {field_name: {value, provenance}}}
    # Extraction metadata (always populated):
    confidence: float = 1.0          # 1.0 = structurally certain; lower = NLU/LLM-derived / soft rule
    source: Literal[
        "structural", "pattern_cached", "nlu_ner", "nlu_table", "nlu_layout",
        "llm_fallback", "derivation_registry", "lookup_api", "lookup_db"
    ] = "structural"
    attempt: int = 1                 # retry attempt number that produced this value

@dataclass
class ExtractionResult:
    header: dict[str, ExtractedValue]
    line_items: list[dict[str, ExtractedValue]]
    parsed_text: str                 # = doc.full_text at the time of extraction
    unresolved_fields: list[str]
    attempts: int
    pattern_id_used: Optional[int]   # if a cached pattern was used
    layout_signature: str            # for post-run pattern persistence
```

### Public API

```python
from src.services.structural_extractor import extract

result = extract(
    file_bytes=doc_bytes,
    filename="CITY OF NEWPORT INV600254 for PO502004.pdf",   # also used as layout-signature hint
    doc_type="Invoice",              # Invoice | Purchase_Order | Quote | Contract
)
# Works identically for .pdf, .docx, .xlsx, .csv — the adapter is chosen from filename+magic-bytes.
# result.header["invoice_amount"].value == 8333.0
# result.header["invoice_amount"].anchor_ref  # BBox(page=1, x0=..., ...) OR CellRef OR ColumnRef OR NodeRef
# result.unresolved_fields == []  (if fully reconciled)
```

## Layer 1 — Multi-format ingestion (per-adapter notes)

Each adapter normalizes to the `ParsedDocument` shape. The orchestrator detects format from file extension + magic bytes and dispatches to the right adapter.

### PDF adapter (`pdf_parser.py`)

- PyMuPDF `page.get_text("words")` → list of `(x0, y0, x1, y1, text, block, line, word)`.
- Each word becomes a `Token` with `anchor=BBox(page, x0, y0, x1, y1)`.
- Regions: visually-close tokens are grouped into `Region(kind="block")` by PyMuPDF's block numbers.
- Tables: pdfplumber `page.extract_tables()` runs in parallel; when it produces a non-degenerate table, it's recorded as a `Table` with cell regions. If pdfplumber fails (as observed in the audit), the table slot stays empty — spatial line-item parsing still runs on the tokens.
- `full_text`: tokens linearized by `(page, y, x)` order.

### DOCX adapter (`docx_parser.py`)

- `python-docx` walks the document body.
- Each paragraph → `Region(kind="paragraph")` with tokens from word-level splits on whitespace; each token gets `anchor=NodeRef(kind="paragraph", paragraph_index=i)`.
- Each table → `Table` with cells; each cell's tokens get `anchor=NodeRef(kind="table_cell", table_index=..., row=..., col=...)`.
- `full_text`: paragraphs and table rows concatenated in document order.

### XLSX adapter (`xlsx_parser.py`)

- `openpyxl` walks each sheet's used range.
- Each cell's text content → one or more `Token`s with `anchor=CellRef(sheet, row, col, merged_range=...)`.
- A sheet's used range becomes a single `Table` when ≥ 2 rows × ≥ 2 cols have non-empty cells. Header-row detection: first row where every cell is text (no pure numbers or dates) is marked as header.
- Multi-sheet: each sheet contributes its own `Table`; the document overall has `pages_or_sheets = workbook.sheet_count`.
- `full_text`: cells linearized sheet-by-sheet in row-major order.

### CSV adapter (`csv_parser.py`)

- `pandas.read_csv` with `dtype=str` to preserve textual forms; fallback to the stdlib `csv` module on quirky delimiters.
- Header row → `Region(kind="column", label=col_name)` per column.
- Each cell → `Token` with `anchor=ColumnRef(row, col, column_name)`.
- Always exactly one `Table` with all rows.
- `full_text`: rows joined with newlines, cells with tabs.

## Layer 2 — Type-driven anchor discovery (replaces hardcoded label vocabulary)

The ONLY hardcoded procurement knowledge is the **schema type** of each target field. No label tokens are hardcoded.

```python
# src/services/structural_extractor/discovery/schema.py
FIELD_TYPES: dict[tuple[str, str], FieldType] = {
    ("Invoice", "invoice_id"):            ID,
    ("Invoice", "po_id"):                 ID,
    ("Invoice", "supplier_id"):           ORG,
    ("Invoice", "buyer_id"):              ORG,
    ("Invoice", "invoice_date"):          DATE,
    ("Invoice", "due_date"):              DATE,
    ("Invoice", "invoice_amount"):        MONEY,
    ("Invoice", "tax_amount"):            MONEY,
    ("Invoice", "tax_percent"):           PERCENT,
    ("Invoice", "invoice_total_incl_tax"): MONEY,
    ("Invoice", "currency"):              CURRENCY_CODE,
    ("Invoice", "payment_terms"):         TEXT,
    # ... same pattern for Purchase_Order, Quote, Contract
}

class FieldType(Enum):
    ID          = "id"           # alphanumeric identifier
    DATE        = "date"
    MONEY       = "money"        # monetary amount
    PERCENT     = "percent"
    CURRENCY_CODE = "currency"   # ISO 4217 code
    ORG         = "org"          # organization name
    ADDRESS     = "address"
    TEXT        = "text"         # free text
    INTEGER     = "integer"
```

### Discovery algorithm (per field)

For each target field with known type:

1. **Emit type-matching candidates from the document.** A typed-entity detector scans the `ParsedDocument` tokens/regions and proposes candidates:
   - `DATE`: any token (or contiguous run of ≤ 4 tokens) that `dateutil.parser.parse` accepts, OR a CellRef with XLSX `number_format` marked as a date.
   - `MONEY`: any token matching a currency symbol + numeric pattern, OR a numeric CellRef with XLSX `number_format` marked as currency.
   - `PERCENT`: numeric token immediately followed by `%`, OR CellRef with percent format.
   - `ORG`: (a) BERT-NER `ORG` spans when NLU tier is active, (b) contiguous capitalized word runs ending in `Ltd/LLC/Inc/Limited/plc/Corp/GmbH/Company`.
   - `ID`: alphanumeric tokens of length ≥ 3 containing at least one digit, without currency/date/percent markers.
   - `CURRENCY_CODE`: 3-letter uppercase tokens matching ISO 4217 set, OR any currency symbol mapped to its code.
   - `TEXT`: any token/region.

   No label matching at this step. We're just finding "things that could be a date", "things that could be money", etc.

2. **Rank candidates by label-proximity inference.** For each candidate, look at nearby tokens (left and above; DOCX: preceding runs in the same paragraph or cell above; XLSX: the cell to the left or the cell above; CSV: the column header). These nearby tokens are the **inferred label**. We do not match the inferred label against any hardcoded list. Instead, we score the candidate using a weighted sum:

   ```
   score(candidate) = 0.40 * pattern_hit
                    + 0.25 * positional_prior
                    + 0.20 * arithmetic_fit
                    + 0.10 * uniqueness
                    + 0.05 * label_semantic_similarity
   ```

   Each sub-signal is normalized to [0, 1]:
   - **pattern_hit** — 1.0 if the candidate's `AnchorRef` matches (within tolerance: ±20 px for PDF bbox, same cell for XLSX, same column for CSV, same paragraph/cell for DOCX) a cached pattern with `success_count ≥ 3`; 0.5 for `success_count` 1-2; 0 if no pattern.
   - **positional_prior** — cosine similarity between the candidate's normalized (x, y) or (row, col) position and the doc_type's `type_priors` histogram mean for this field. Zero for cold-start (no priors yet).
   - **arithmetic_fit** — for `MONEY` candidates only: 1.0 if selecting this candidate makes a math invariant reconcile (`subtotal + tax_amount = total` within £0.01); 0 otherwise. For non-MONEY fields: always 0.
   - **uniqueness** — `1 / (number_of_same_type_candidates_in_doc)`. A field with a sole candidate of its type gets 1.0.
   - **label_semantic_similarity** — cosine similarity between the inferred label's embedding (reuse `BAAI/bge-large-en-v1.5` already loaded by procwise) and the target field name's embedding (e.g. "invoice amount"). **Computed once and cached per-document**; not a hardcoded label list.

   Tie-breaker when two candidates score within 0.05 of each other: earlier document position wins (top-to-bottom, left-to-right).

3. **Pick the highest-scoring candidate.** If the top two candidates' scores are both within 0.05 of each other AND neither exceeds 0.7, leave the field unresolved for this attempt (escalate to NLU/LLM retry).

4. **Verify the anchor.** Walk back to the `AnchorRef` of the chosen candidate; confirm the token at that location matches the value's raw form.

5. **If unresolved after structural discovery**, retry escalates into NLU tier (BERT-NER refines ORG/DATE/MONEY candidate sets; Table-Transformer improves line-item row discovery; Layout-YOLO improves field-region grouping). See Retry Strategy below.

### Field-specific notes

- **ID fields**: after ranking, the chosen candidate must satisfy a type sanity check (length, digit-present, not a common word). If it fails, the field is unresolved, not kept.
- **Dates**: locale detection per document (unchanged from v1): month-name → unambiguous; `day > 12` token somewhere → `dayfirst=True`; UK address markers → `dayfirst=True`; `CURRENCY=USD` → `dayfirst=False`; else unresolved → LLM arbiter. `invoice_date`, `order_date`, `quote_date`, `validity_date`, `expected_delivery_date` must be directly extracted via a date anchor. `due_date` is not extracted here — it is produced downstream by the Derivation Registry (`due_date_from_terms` → `due_date_default`, see Derivation Rules).
- **Parties**: candidate selection. Step 1 emits ORG candidates. Step 2 ranks by: (a) which cluster of lines does this ORG appear in — clusters separated by blank lines / section breaks tend to be Bill-To / From / Payable-To blocks; (b) pattern-store history; (c) NLU confirmation in later retries. Suffix preservation is enforced at step 4: the chosen anchor must cover the full ORG span including `Ltd`/`LLC`/`Inc`/etc., not a truncated substring.
- **Amounts**: step 2's arithmetic-fit signal is dominant. The extractor enumerates MONEY candidates in the document and tests combinations against the math-reconciliation invariant (`subtotal + tax = total`) to resolve subtotal/tax/total simultaneously. This is an O(n³) search over MONEY candidates. **Hard cap**: before enumeration, candidates are ranked by their position-prior score and the list is truncated to the top 40 entries; enumeration operates only on that prefix. In practice n ≤ 20 for typical invoices; the truncation protects against pathological docs where phone numbers, IDs, or line numbers slip through as MONEY-typed. With n=40 the worst case is 64,000 combinations (~30 ms); with early termination on the first combination that reconciles within ±£0.01, the typical case is ~200 combinations.
- **Currency**: highest-priority candidate is an ISO 4217 token (`GBP`, `USD`, `EUR`); fallback to the most frequent currency symbol → code mapping. Must be consistent across all amounts.
- **Line items**: format-dispatched because the table concept is format-native:
  - **PDF**: spatial table detection (header-row anchor via column-header-like tokens → crop to `subtotal` Y → row-cluster → column-assign by X-midpoint → dedup identical rows within 0.5×line-height → monotonic line_no).
  - **XLSX**: native cell grid; rows BELOW the detected header row and ABOVE the subtotal row are line items. Columns are cell-column-indexed.
  - **DOCX**: native tables from `docx.tables`; rows below the header row.
  - **CSV**: every data row is a line item; columns are CSV columns.
  - **Common validation (all formats)**: `Σ line_totals = subtotal (±£0.01)` AND `qty × unit_price = line_total (±£0.01)` for each row. Subtotal used MUST come from the same attempt's header extraction (Atomicity, Principle #5).
- **Payment terms**: free-text. Any token run between the inferred label (one of: the nearest word-to-the-left that is not itself a value) and the end-of-section. Used as input to the Derivation Registry for computing `due_date` (see Derivation Rules).

## Derivation & inference rules (replaces "no derivation" from earlier drafts)

The **Derivation Registry** (`src/services/structural_extractor/derivation.py`) holds every deterministic rule that fills a DB column when the value is not directly in the source but IS computable from already-extracted values. Each rule:

- Is a pure function `(inputs) -> value` (no side effects except lookup-tier rules that call external APIs / DB).
- Declares its inputs; runs only when all inputs are already satisfied by extraction/derivation (topological ordering of the rule DAG).
- Produces a value with `source="derived"` (or `"inferred"` / `"lookup"`) plus a `derivation_trace` dict recording the rule id, the input field names, and the input values.
- Has unit tests in `tests/structural_extractor/test_derivation.py`.

### Rule set v1 (header fields)

| Target field | Rule id | Inputs | Logic | Fallback |
|---|---|---|---|---|
| `due_date` | `due_date_from_terms` | `invoice_date`, `payment_terms` | Parse "Net N" or "within N days" from `payment_terms` text → return `invoice_date + N days` | If no N parseable: `invoice_date + 90 days` |
| `due_date` | `due_date_default` | `invoice_date` (no `payment_terms`) | `invoice_date + 90 days` | — |
| `tax_percent` | `tax_pct_from_amounts` | `tax_amount`, `invoice_amount` (or `total_amount` for POs) | `round(tax_amount / subtotal * 100, 2)` | — |
| `tax_amount` | `tax_amount_from_pct` | `tax_percent`, `invoice_amount` | `round(subtotal * tax_percent / 100, 2)` | — |
| `invoice_total_incl_tax` | `total_from_subtotal_tax` | `invoice_amount`, `tax_amount` | `round(subtotal + tax_amount, 2)` | — |
| `invoice_amount` | `subtotal_from_total_tax` | `invoice_total_incl_tax`, `tax_amount` | `round(total - tax_amount, 2)` | — |
| `currency` | `currency_from_symbol` | any `MONEY` candidate with currency symbol | `£→GBP, $→USD, €→EUR, ¥→JPY, A$→AUD, C$→CAD, ₹→INR` | — |
| `exchange_rate_to_usd` | `xrate_lookup` | `currency`, (date) | Fetch from `open.er-api.com/v6/latest/{ccy}`, cache 1h | If API down: use last-known cached rate within 24h; else NULL + flag `requires_review` |
| `converted_amount_usd` | `convert_to_usd` | `invoice_total_incl_tax` (or `total_amount`), `exchange_rate_to_usd` | `round(amount * xrate, 2)` | — |
| `supplier_id` | `supplier_id_from_lookup` | `supplier_name` | SELECT from `proc.bp_supplier` WHERE normalized(supplier_name) matches; else... | Generate: `SUP-` + supplier_name stripped of spaces/punct, UPPER. INSERT new `bp_supplier` row. |
| `buyer_id` | `buyer_id_from_lookup` | buyer org name (from Bill-To) | SELECT from `proc.bp_supplier` (buyers are also in supplier master) | Else: INSERT new buyer row; return generated id |
| `country` | `country_from_postcode` | address tokens | UK postcode pattern (e.g. `RH13 5QH`) → `United Kingdom`; US ZIP → `United States`; EU postcodes per country | NULL |
| `region` | `region_from_address` | address tokens, `country` | Parse county/state/province from address; UK: "West Sussex"; US: state abbreviation | NULL |
| `delivery_region` (PO) | same as `region` | PO delivery_address_line1+line2 | same logic | — |
| `ship_to_country` (PO) | same as `country` | PO delivery_address | same logic | — |
| `invoice_status` | `invoice_status_default` | (none) | `"Issued"` for newly-extracted invoices; further updates via workflow | — |
| `po_status` | `po_status_default` | (none) | `"Open"` for newly-extracted POs | — |
| `ai_flag_required` | `ai_flag_compute` | line-item sum vs header total reconciliation result | `"Y"` if any validation warning present; else `"N"` | — |
| `tax_amount`, `total_amount` (line-items) | `line_tax_total_from_header_ratio` | header `tax_percent`, per-line `line_total` | Apply header `tax_percent` proportionally to each line | Only applies when document has single tax rate |

### Rule set v1 (line-item fields)

| Target field | Rule id | Inputs | Logic |
|---|---|---|---|
| `line_total` / `line_amount` | `line_total_from_qty_price` | `quantity`, `unit_price` | `round(quantity * unit_price, 2)` |
| `unit_price` | `unit_price_from_qty_total` | `quantity`, `line_total`, `quantity > 0` | `round(line_total / quantity, 2)` |
| `quantity` | `quantity_from_price_total` | `unit_price`, `line_total`, `unit_price > 0` | `round(line_total / unit_price, 0)` — integer for invoice, decimal for PO per schema |
| `line_number` / `line_no` | `line_no_monotonic` | row order in source | Sequential 1..N in order of Y-position (PDF) or row index (XLSX/CSV/DOCX) |

### Rule execution order

Rules form a DAG. On each extraction attempt, after Layer 2 discovery, the engine:
1. Builds the set of already-extracted fields.
2. Resolves derivable fields in topological order — e.g. `tax_amount` before `invoice_total_incl_tax` (which depends on it), `exchange_rate_to_usd` before `converted_amount_usd`.
3. Produces an `ExtractedValue` for each resolved field with `source="derived"`, `derivation_trace={rule_id, inputs: {field: value}}`.
4. Unresolvable fields remain NULL (unless they're required — then retry escalates).

### Audit trail — where does provenance live?

Every `ExtractedValue` carries `source` (enum) and either `anchor_ref` (for extracted) or `derivation_trace` (for derived/inferred/lookup). At persistence time, a new sibling table records the full trace:

```sql
CREATE TABLE proc.bp_extraction_provenance (
    id                  BIGSERIAL PRIMARY KEY,
    parent_table        TEXT NOT NULL,      -- 'bp_invoice' | 'bp_purchase_order' | ...
    parent_pk           TEXT NOT NULL,      -- invoice_id / po_id / quote_id
    field_name          TEXT NOT NULL,
    source              TEXT NOT NULL,      -- 'extracted' | 'derived' | 'inferred' | 'lookup'
    anchor_ref          JSONB,              -- populated for source='extracted'
    derivation_trace    JSONB,              -- populated for source in ('derived','inferred','lookup')
    confidence          NUMERIC(3,2),
    attempt             INT,
    extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_prov_parent ON proc.bp_extraction_provenance (parent_table, parent_pk);
```

Every extraction writes one provenance row per populated field. Auditors can answer "why does `INV600254.due_date = 2019-11-20`?" by querying this table and seeing `source='derived', derivation_trace={rule_id: 'due_date_from_terms', inputs: {invoice_date: 2019-08-22, payment_terms: "within 30 days"}}`.

## Column coverage matrix — which `bp_*` columns get filled, how

Every column in every target table has a defined source. The matrix below is authoritative; a column not listed here is intentionally SYSTEM (auto-filled by DB) or WORKFLOW (managed by the orchestrator after extraction).

### `proc.bp_invoice` (28 columns)

| Column | Source category | Rule |
|---|---|---|
| `invoice_id` | extracted | ID anchor discovery |
| `po_id` | extracted | ID anchor discovery |
| `supplier_id` | derived→lookup | `supplier_id_from_lookup` (on `supplier_name` from Parties extractor) |
| `buyer_id` | derived→lookup | `buyer_id_from_lookup` (on Bill-To org name) |
| `requisition_id` | WORKFLOW | Populated by the requisition-to-invoice matching workflow (not this extractor) |
| `requested_by` | WORKFLOW | same |
| `requested_date` | WORKFLOW | same |
| `invoice_date` | extracted | Date anchor discovery |
| `due_date` | derived | `due_date_from_terms` or `due_date_default` |
| `invoice_paid_date` | WORKFLOW | Populated when invoice is actually paid |
| `payment_terms` | extracted | Payment-terms extractor |
| `currency` | extracted OR derived | ISO code if printed; else `currency_from_symbol` |
| `invoice_amount` | extracted OR derived | Subtotal anchor discovery; else `subtotal_from_total_tax` |
| `tax_percent` | extracted OR derived | Direct anchor; else `tax_pct_from_amounts` |
| `tax_amount` | extracted OR derived | Direct anchor; else `tax_amount_from_pct` |
| `invoice_total_incl_tax` | extracted OR derived | Direct anchor; else `total_from_subtotal_tax` |
| `exchange_rate_to_usd` | lookup | `xrate_lookup` (live API with cache) |
| `converted_amount_usd` | derived | `convert_to_usd` |
| `country` | inferred | `country_from_postcode` (from Parties address) |
| `region` | inferred | `region_from_address` |
| `invoice_status` | derived | `invoice_status_default` = `"Issued"` |
| `ai_flag_required` | derived | `ai_flag_compute` |
| `trigger_type` | WORKFLOW | `"ManualUpload"` or `"EmailIngest"` based on upload source |
| `trigger_context_description` | WORKFLOW | Filename + upload timestamp |
| `created_date` | SYSTEM | `NOW()` |
| `created_by` | SYSTEM | `"AgentNick"` or the authenticated user |
| `last_modified_by` | SYSTEM | same as created_by on insert |
| `last_modified_date` | SYSTEM | same as created_date on insert |

### `proc.bp_invoice_line_items` (20 columns)

| Column | Source | Rule |
|---|---|---|
| `invoice_line_id` | SYSTEM | `{invoice_id}-{line_no}` generated |
| `invoice_id` | SYSTEM | FK from parent |
| `line_no` | derived | `line_no_monotonic` |
| `item_id` | extracted OR derived | Extracted from item SKU if present; else `ITM-{supplier_id}-{slugified_description}` |
| `item_description` | extracted | Line-item extractor, description column |
| `quantity` | extracted OR derived | Direct cell / spatial column; else `quantity_from_price_total` |
| `unit_of_measure` | extracted | Direct cell; else NULL |
| `unit_price` | extracted OR derived | Direct cell; else `unit_price_from_qty_total` |
| `line_amount` | extracted OR derived | Direct cell; else `line_total_from_qty_price` |
| `tax_percent` | derived | Inherited from header `tax_percent` |
| `tax_amount` | derived | `line_tax_total_from_header_ratio` |
| `total_amount_incl_tax` | derived | `line_amount + tax_amount` |
| `po_id` | derived | Inherited from header `po_id` |
| `delivery_date` | extracted | If per-line delivery dates are present; else NULL |
| `country` | inferred | Inherited from header `country` |
| `region` | inferred | Inherited from header `region` |
| `created_date` / `created_by` / `last_modified_date` / `last_modified_by` | SYSTEM | `NOW()` / `"AgentNick"` |

### `proc.bp_purchase_order` (35 columns)

Mirror structure. Key differences:
- `supplier_name` (text) instead of `supplier_id` at header (per existing schema quirk) — extracted directly; `supplier_id` is derived via `supplier_id_from_lookup`.
- `order_date` extracted (date anchor).
- `expected_delivery_date` extracted if present, else NULL (no safe derivation rule).
- `ship_to_country` / `delivery_region` / `delivery_city` / `postal_code` / `delivery_address_line1` / `delivery_address_line2` from Parties address extractor (Bill-To typically, or explicit Ship-To anchor).
- `incoterm` / `incoterm_responsibility` extracted if present, else NULL.
- `contract_id` WORKFLOW (matched later).
- `po_status` derived = `"Open"`.
- `total_amount_incl_tax` / `tax_amount` / `tax_percent` / `total_amount` — same derivation as invoice amounts.
- `default_currency` / `currency` — both set to extracted/derived `currency`.

### `proc.bp_po_line_items` (18 columns)

Same pattern as invoice line items. `unit_of_measue` [sic, schema typo] extracted as `unit_of_measure`; the DB column retains the typo.

### `proc.bp_quote` (23 columns)

| Extracted | Derived | Inferred | Lookup | WORKFLOW | SYSTEM |
|---|---|---|---|---|---|
| quote_id, quote_date, validity_date, supplier_address, buyer_address, total_amount, tax_amount, tax_percent, total_amount_incl_tax, currency | ai_flag_required, converted_amount_usd | country, region | supplier_id, buyer_id, exchange_rate_to_usd | po_id (matched after quote accepted) | created_date/by, last_modified_date/by |

### `proc.bp_quote_line_items` (17 columns)

Same pattern as invoice line items; no `po_id` inheritance.

### `proc.bp_contracts` (27 columns)

Deferred — not part of this PR. Contracts have legal-text extraction needs (clause parsing) that are out of scope. Columns listed in Non-goals.

## Retry strategy (progressive layered NLU)

The retry loop **never gives up** but each attempt layers in a new source of signal. The accumulated context from previous attempts is passed forward; attempts 2+ have strictly more information than attempt 1.

| Attempt | Added signal | What it sees |
|---|---|---|
| 1 | Structural extractor only (PyMuPDF word boxes + anchor lookup) | Raw PDF structure |
| 2 | + BERT NER (`dslim/bert-base-NER`) on parsed text | PERSON / ORG / LOC / MISC spans as supplementary signal for parties and addresses |
| 3 | + Table-Transformer (`microsoft/table-transformer-structure-recognition`) on rendered page images | Learned table-detection for line items when anchor-based row detection failed |
| 4 | + Layout-aware (`unstructuredio/yolo_x_layout`) | Field/region detection for complex layouts (multi-column, boxed) |
| 5 | + Strict grounded LLM call (AgentNick), passed all prior attempt outputs + parsed text + failed fields | LLM acts as arbiter with full context |
| 6–10 | Same as 5 with prompt variation, temperature 0.0 → 0.05 micro-jitter, and accumulated "do-not-invent" examples from prior failures | Progressive prompt refinement |
| 11+ | Doc escalates to `proc.extraction_review_queue` with all accumulated signals; pipeline continues | Human / later-automated review |

### Key properties of the retry loop

- **Monotone context growth**: each attempt receives all prior-attempt outputs as additional input. A value extracted in attempt 1 with anchor bbox for "Subtotal £8,333" carries forward as known-good context to attempts 2+.
- **Field-level retry**, not document-level: if attempt 3 reconciles 18 out of 20 fields, attempts 4+ only work on the 2 remaining fields.
- **BBox anchoring is always the final gate**: any value returned by any model must point to a `Word` in `doc.words` whose `.text` matches the value's raw form. An NER model's output without a valid source-word mapping is dropped.
- **Math verification is always the final gate** — scoped to the attempt that produced the numbers. Values from different attempts cannot be combined in a math check without the combined group being re-extracted atomically by a later attempt (see Guiding Principles #5).

### Attempt context schema

The state passed between attempts is a `RetryState` dataclass:

```python
@dataclass
class AttemptOutput:
    attempt: int
    source: str                                  # "structural" | "nlu_ner" | "nlu_table" | "nlu_layout" | "llm"
    extracted: dict[str, ExtractedValue]          # field_name -> value (header fields)
    line_items: list[dict[str, ExtractedValue]]   # None if the attempt didn't address line items
    validation_failures: list[str]                # field names that failed reconciliation this attempt
    residual_unresolved: list[str]                # fields still needed after this attempt
    latency_ms: int

@dataclass
class RetryState:
    doc: ParsedDocument
    doc_type: str
    target_fields: set[str]                       # all required fields for doc_type
    attempts: list[AttemptOutput]                 # oldest → newest
    accepted_header: dict[str, ExtractedValue]    # final winners by field (bbox-anchored + math-reconciled)
    accepted_line_items: list[dict[str, ExtractedValue]] | None
    unresolved: set[str]                          # target_fields − accepted_header.keys() − (line_items if accepted)
```

Each attempt receives `RetryState` and returns an `AttemptOutput`. A post-attempt merge function decides which values in the new `AttemptOutput.extracted` replace values in `accepted_header`, subject to the atomicity rule.

### Conflict resolution between attempts

When an attempt produces a value for field F that differs from the currently-`accepted` value for F:
- If both values have identical arithmetic consequences (e.g., both amounts pass the math check), **keep the earlier-attempt value** (older is more deterministic — structural beats learned beats LLM).
- If they differ materially AND are tied together by arithmetic (e.g., a new `subtotal` + new `line_items` group), the **entire new group** replaces the old only if the new group passes its own math check internally. If the new group passes but disagrees with an unrelated accepted value (e.g., a new subtotal that disagrees with an unchanged `tax_amount + total_incl_tax`), re-extract the unrelated field in the NEXT attempt before committing.
- Every commit to `accepted_header` / `accepted_line_items` is logged with `(attempt_no, source, reason)` for post-mortem.

## DB changes

### New table

```sql
CREATE TABLE proc.extraction_review_queue (
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
    signals_json         JSONB,                -- accumulated outputs from structural/ner/table/layout/llm
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at          TIMESTAMPTZ,
    resolved_by          TEXT
);
CREATE INDEX idx_eq_unresolved ON proc.extraction_review_queue (resolved_at) WHERE resolved_at IS NULL;
CREATE INDEX idx_eq_doc_type ON proc.extraction_review_queue (doc_type);
```

### New status value and watcher integration

`proc.process_monitor.status` gets a new valid value: `'Extraction_InReview'` — set when retry attempts ≥ 10 and field(s) still unreconciled. This status is not terminal until manually resolved.

**Required changes to `src/services/process_monitor_watcher.py`** (so the watcher doesn't silently re-extract parked docs):
- Recovery sweep (currently line 618: `UPDATE ... SET status='Completed' WHERE status='Extracting'`) must be expanded to leave `'Extraction_InReview'` alone. Only `'Extracting'` rows get reset.
- Poll query (currently line 634: `WHERE status IN ('Completed', 'Running')`) unchanged — `'Extraction_InReview'` is explicitly NOT re-picked.
- LISTEN trigger (line 109: `IF NEW.status IN ('Completed', 'Running')`) unchanged.
- Add a resolution path: when an operator updates an `extraction_review_queue` row's `resolved_at`, a trigger (to be added in migration) resets the underlying `process_monitor` row to `'Completed'` so the watcher reprocesses it.

### Extension of `proc.bp_extraction_patterns`

The existing `ExtractionPatternStore` (`src/services/extraction_pattern_store.py`) operates on the table below; we **add** two JSONB columns, we don't replace the table:

```sql
-- Existing columns (unchanged):
--   pattern_id SERIAL PK
--   file_type TEXT            -- 'pdf' | 'docx' | 'xlsx' | 'csv'
--   doc_type TEXT             -- 'Invoice' | 'Purchase_Order' | 'Quote' | 'Contract'
--   supplier_name TEXT
--   layout_signature TEXT     -- existing fingerprint
--   column_mapping JSONB      -- existing: maps columns to schema fields (XLSX/CSV-oriented)
--   extraction_hints TEXT
--   success_count INT
--   last_used TIMESTAMP
--   created_date TIMESTAMP

-- NEW column on existing table — one addition only, for per-vendor anchor positions:
ALTER TABLE proc.bp_extraction_patterns
    ADD COLUMN anchor_patterns JSONB;    -- per-field anchor positions (see schema below)

-- NEW table — cross-vendor type priors (one row per doc_type, not per vendor):
CREATE TABLE proc.bp_extraction_type_priors (
    doc_type     TEXT PRIMARY KEY,
    priors       JSONB NOT NULL,          -- {field_name: {mean_x, mean_y, mean_row, mean_col, variance, n_samples}}
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

Rationale for splitting `type_priors` into its own table: priors are per-doc_type aggregates, not per-vendor facts. Storing them per-pattern row would duplicate data and force every pattern write to touch every other row's priors. The separate table has 4 rows max (one per doc_type) and is updated atomically after each successful extraction with an incremental running-mean update.

`anchor_patterns` JSONB schema (per field):
```jsonc
{
  "invoice_id": {
    "format_variants": {
      "pdf":  {"page": 1, "region": "top_right", "inferred_label_tokens": ["invoice", "no"], "anchor_sample": {"x0": 420.5, "y0": 48.2, "x1": 502.3, "y1": 59.8}},
      "xlsx": {"sheet_name_pattern": "Sheet1", "cell": "B3", "inferred_label_cell": "A3"},
      "docx": {"paragraph_index": 0, "table_index": null, "inferred_label_paragraph": "-"},
      "csv":  {"column_name_pattern": "Invoice Number", "column_index": 0}
    },
    "success_count": 12,
    "last_confirmed_at": "2026-04-21T14:52:16Z"
  },
  "invoice_amount": { ... similar ... },
  "line_items": {
    "format_variants": {
      "pdf": {"page": 1, "table_region": {"x0": 50, "y0": 300, "x1": 550, "y1": 500}, "columns": [{"label": "Description", "x_mid": 130}, {"label": "Qty", "x_mid": 310}, {"label": "Unit Price", "x_mid": 410}, {"label": "Total", "x_mid": 510}]},
      "xlsx": {"header_row": 7, "columns": [{"label": "Item", "col_index": 1}, {"label": "Qty", "col_index": 3}, ...]},
      ...
    },
    ...
  }
}
```

`type_priors` JSONB schema (per doc_type, not per-pattern):
```jsonc
{
  "Invoice": {
    "invoice_id_typical_region": {"pdf": "top_right_page1", "xlsx": "cell_B3"},
    "subtotal_typical_region":   {"pdf": "bottom_right_page1"},
    ...
  }
}
```
`type_priors` is updated incrementally (weighted moving average of positional fingerprints) each time a pattern confirms successfully. Cold-start: a new vendor with a never-seen layout starts with no pattern row but uses accumulated `type_priors` as a soft guide; if extraction succeeds, a new pattern row is written.

## Pattern learning (Layer 4)

On every successful `ExtractionResult` (zero unresolved fields, math reconciled, cross-validated), the pattern store is updated:

1. Compute `layout_signature` for this document: a hash of (supplier_id, number_of_pages_or_sheets, header_column_labels_sorted, page1_region_bbox_distribution_histogram). The hash is deterministic and short (16 hex chars).
2. Look up `(file_type, doc_type, supplier_name, layout_signature)` in `proc.bp_extraction_patterns`.
3. If hit → increment `success_count`, update `last_used = NOW()`, merge new `anchor_patterns` into existing (same-field anchors: average the bbox/cell positions; different-field anchors: add).
4. If miss → INSERT new row with this pattern.

Pattern reuse on **next** document from same vendor+layout:
1. Compute `layout_signature` for incoming doc.
2. Lookup `(file_type, doc_type, supplier_name, layout_signature)` — if hit, the `anchor_patterns` are used as **positional priors** in Layer 2's Step 2 ranking. A candidate at the cached position gets a heavy boost.
3. Extraction completes in far fewer attempts (often attempt 1 is sufficient once the layout is cached).
4. Still runs full validation — pattern reuse is a speedup, not an accuracy shortcut. If the cached anchors don't validate, the extractor re-discovers and **overwrites** the stale pattern.

### Pattern trust levels (prevents poisoning from a single bad extraction)

| `success_count` | Trust level | How the pattern is used in Layer 2 Step 2 |
|---|---|---|
| 0 (no pattern row exists) | none | Fresh discovery; `pattern_hit` signal = 0 for this field. |
| 1-2 | learning | `pattern_hit` signal = 0.5. The cached position is a soft prior but NOT sufficient to win on its own. The extractor still independently discovers and compares; if discovery disagrees with the cached pattern, discovery wins and the pattern is marked invalid (reset `success_count` to 0). |
| ≥ 3 | trusted | `pattern_hit` signal = 1.0. The cached anchor is the primary candidate; other candidates are only considered if the cached one fails anchor verification or math reconciliation. |

A pattern transitions from `learning` to `trusted` only after 3 consecutive validations (no failures in between). Any validation failure with a `trusted` pattern demotes it back to `learning` (success_count = 2) and logs the event for observability.

## Integration points

### `src/services/agent_nick_orchestrator.py`

`_dispatch_extraction` becomes:

```python
def _dispatch_extraction(self, file_path: str, doc_type: str):
    file_bytes = self._download(file_path)
    if not os.getenv("USE_STRUCTURAL_EXTRACTOR", "true").lower() in ("true", "1", "yes"):
        return self._legacy_dispatch(file_path, doc_type)  # unchanged path

    from services.structural_extractor import extract
    result = extract(file_bytes, filename=os.path.basename(file_path), doc_type=doc_type)
    if result.unresolved_fields:
        self._park_in_review_queue(result, file_path)  # status = Extraction_InReview
    return {
        "header":      {k: v.value for k, v in result.header.items() if v.value is not None},
        "line_items":  [{k: v.value for k, v in item.items()} for item in result.line_items],
        "_source_text": result.parsed_text,
    }

def _park_in_review_queue(self, result, file_path: str):
    """Write accumulated signals to proc.extraction_review_queue and set
    process_monitor.status='Extraction_InReview'. Called when
    result.unresolved_fields is non-empty after 10 retry attempts."""
    with self._db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO proc.extraction_review_queue "
            "(process_monitor_id, file_path, doc_type, partial_header, "
            " partial_line_items, failed_fields, parsed_text, attempt_count, "
            " last_attempt_at, signals_json) "
            "VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, NOW(), %s::jsonb)",
            (
                result.process_monitor_id, file_path, result.doc_type,
                json.dumps({k: v.value for k, v in result.header.items()}),
                json.dumps(result.line_items_as_serializable()),
                result.unresolved_fields,
                result.parsed_text,
                result.attempts,
                json.dumps(result.signals_summary()),
            ),
        )
        cur.execute(
            "UPDATE proc.process_monitor SET status='Extraction_InReview' WHERE id=%s",
            (result.process_monitor_id,),
        )
```

The legacy LLM-first path stays behind the feature flag for rollback.

## Concurrency & resource management

**Model lifecycle**. NLU models (`BERT-NER`, `Table-Transformer`, `Layout-YOLO`) are wrapped in lazy-loaded thread-safe singletons under `src/services/structural_extractor/nlu/_registry.py`:

```python
class ModelRegistry:
    _instances: dict[str, Any] = {}
    _locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

    @classmethod
    def get(cls, name: str):
        if name in cls._instances:
            return cls._instances[name]
        with cls._locks[name]:
            if name in cls._instances:   # double-checked locking
                return cls._instances[name]
            cls._instances[name] = cls._load(name)
            return cls._instances[name]
```

**Warm-up**. Procwise service startup calls `ModelRegistry.warm()` which loads all three NLU models synchronously before the first extraction request — prevents per-first-doc cold-start latency. Adds ~8s to service startup; acceptable.

**Device placement**. AgentNick LLM stays on GPU. NLU models are loaded on **CPU** (not GPU) to avoid VRAM contention:
- `dslim/bert-base-NER`: 438 MB on disk; ~1.3 GB RAM at inference (tokenizer + batch overhead). CPU inference: 200-400 ms per document.
- `microsoft/table-transformer-structure-recognition`: 113 MB on disk; ~900 MB RAM at inference (single-page image). PDF rendering at 150 DPI (not 300) bounds per-page image to ~3 MB. Multi-page invoices capped at first 3 pages for table detection. CPU inference: 2-4 s per page.
- `unstructuredio/yolo_x_layout`: 276 MB on disk; ~1.1 GB RAM at inference. CPU inference: 1-2 s per page.

**Total steady-state RAM budget for NLU stack**: ~3.5 GB. System has 62 GB; procwise uses ~8 GB baseline. Headroom is ample.

**No GPU impact**. NLU models never touch CUDA; AgentNick GPU utilization remains unchanged.

**Thread pool sizing**. Structural extractor is synchronous per document (no internal threading). The existing process_monitor_watcher `ThreadPoolExecutor(max_workers=4)` remains the concurrency boundary. The module is thread-safe by construction (no module-level mutable state except the ModelRegistry which has locks).

### Feature flag rollout

- **Dev**: `USE_STRUCTURAL_EXTRACTOR=true` from day 1.
- **Staging/prod**: flag stays `false` until all 24 golden docs achieve 100% field-level match in CI. Then flip to `true`. The `legacy_dispatch` code stays in the codebase for one release cycle as emergency rollback.

## Testing

### Golden set — multi-format coverage

Minimum 32 documents across all 4 formats, chosen to exercise distinct layouts:

**PDF (20 docs)**:
- 4 from today's audit: INV600254, DHA-2025-143, 2025-290, PO526800.
- 10 from the 09:00 batch (5× AQUARIUS invoices, 5× CITY OF NEWPORT invoices; these also seed `type_priors` since same supplier = same layout).
- 3 multi-page PDFs (a quote with 2 pages, a contract with 3+ pages, an invoice with continuation pages).
- 3 layout-diverse PDFs (minimalist invoice with no explicit labels; boxed/columnar layout; OCR'd scan).

**DOCX (6 docs)**:
- 2 invoices from different vendors (to verify no hardcoded layout assumption).
- 2 POs (one with native table, one with paragraph-style layout).
- 2 quotes.

**XLSX (4 docs)**:
- 2 invoices (one with multi-sheet — separate summary + line-items sheets; one single-sheet).
- 1 PO with header block + line-items block on the same sheet, separated by blank rows.
- 1 quote in a complex template with merged cells.

**CSV (2 docs)**:
- 1 flat line-items CSV (typical exported bulk-order data).
- 1 header-then-rows CSV.

Ground truth authored as `tests/structural_extractor/fixtures/ground_truth.yaml`:
```yaml
INV600254:
  doc_type: Invoice
  source_format: pdf
  header:
    invoice_id: INV600254
    po_id: PO502004
    supplier_id: City of Newport          # full name, never truncated
    buyer_id: Assurity Ltd
    invoice_date: 2019-08-22
    invoice_amount: 8333.00
    tax_amount: 1666.60
    invoice_total_incl_tax: 9999.60
    currency: GBP
    payment_terms: "within 30 days"
  line_items:
    - line_no: 1
      item_description: "Bespoke Marketing Services (1 Month) 3-5 Posts Per Week"
      quantity: 1
      unit_price: 8333.00
      line_amount: 8333.00

acme_invoice_01.xlsx:
  doc_type: Invoice
  source_format: xlsx
  header: {...}
  line_items: [...]
# ... per-doc entries for all 32 files, covering each doc_type × format cell
```

### Test gates

`test_full_extraction.py` asserts, for every doc:
- Every ground-truth field is present in `result.header` or `result.line_items`.
- Every value matches exactly (strings: case-sensitive equal; numbers: within £0.01; dates: exact YYYY-MM-DD).
- `result.unresolved_fields` is empty.
- Every value in `result.header` has a valid `anchor_ref` (BBox / CellRef / ColumnRef / NodeRef) that points to a source location whose normalized text matches the value's raw form.

CI blocks the PR merge unless all 32 docs pass across all 4 formats.

### Additional test coverage (closing gaps)

- **Multi-page PDFs**: 3 multi-page docs as above. Assert extractor correctly handles headers on page 1 with line items on pages 2-3.
- **Multi-sheet XLSX**: the dual-sheet invoice tests that the extractor identifies the correct sheet for line items and correctly disambiguates `Sheet1.B3` vs `Sheet2.B3`.
- **Merged XLSX cells**: the complex quote with merged cells; assert values anchored to the merged range, not the individual cell.
- **DOCX native tables**: assert line items extracted via `NodeRef(kind="table_cell")` match ground truth without needing spatial inference.
- **CSV with & without header**: both variants; assert column-label inference when no header is present (extractor treats the first row as data + infers column types from content).
- **Scanned / OCR PDFs**: the OCR'd doc in golden set. When text comes from OCR (no reliable bbox), the extractor MUST degrade cleanly: the PDF adapter detects OCR text and sets `token.anchor.x0 = token.anchor.x1 = 0` as a sentinel; anchor verification then uses text-equality (token.text == value.anchor_text). Test asserts extraction succeeds (not fails silently) OR `unresolved_fields` is populated.
- **Anchor absent**: A minimalist invoice without `"Invoice No"`-style labels. Assert structural step returns a candidate set (from type detection) and retry escalates to NLU/LLM when no proximity label is discoverable.
- **Corrupted file / empty parsed_text**: Test fixtures with truncated bytes per format. Each adapter raises a specific `FormatParseError` subclass (`PDFParseError`, `DocxParseError`, `XlsxParseError`, `CsvParseError`); caller marks record `Extraction_Failed` (not `InReview`).
- **Review-queue write assertion**: Test that when retry exhausts, the corresponding row is written to `proc.extraction_review_queue` with correct `failed_fields`, `partial_header`, `partial_line_items`, and `process_monitor.status` transitions to `'Extraction_InReview'`.
- **Feature flag legacy path**: With `USE_STRUCTURAL_EXTRACTOR=false`, assert orchestrator calls unchanged legacy path and new module isn't imported (check `sys.modules` absence of `structural_extractor`).
- **Concurrency**: Parallel extraction of 4 docs across threads using the same `ModelRegistry`; assert each model loads exactly once and all 4 complete without race condition.
- **Anchor ambiguity**: A date token `"04/05/2020"` with no other month-name dates in the doc. Assert it remains unresolved after attempt 1 and escalates to LLM arbiter by attempt 5.
- **Pattern learning — first vs Nth doc**: Process two documents from the same vendor+layout. Assert:
  - First doc: `result.pattern_id_used is None` (no cached pattern).
  - First doc: after completion, a new row appears in `proc.bp_extraction_patterns`.
  - Second doc: `result.pattern_id_used == <first doc's pattern_id>`, `result.attempts == 1`, and the cached anchor was confirmed.
  - If the second doc is structurally different from the first (edited template), assert the cached pattern is rejected and a new pattern row is written.
- **Pattern learning — bad cache recovery**: Inject a deliberately-wrong cached pattern for a known doc. Assert extractor detects mismatch (math doesn't reconcile), overwrites the stale pattern, succeeds on next attempt.

### Regression tests

- Run against the 20 PDF invoices already in DB from the morning batch; confirm no row changes values except where a prior discrepancy flagged "value_not_anchored" — in which case the new extraction must produce a value that IS anchored.
- Benchmark: each doc's end-to-end extraction must complete in ≤ 30s on the A10G (structural + pattern-cached: sub-second typical; the 30s budget is for NLU fallback if attempt 2–4 are needed). Fail the test if p95 > 30s.
- Benchmark: second extraction of same vendor+layout must complete in ≤ 2s (pattern-cached fast path).

## Observability

Every extraction emits a structured log line:
```json
{
  "process_monitor_id": 1076,
  "doc_type": "Invoice",
  "attempts": 1,
  "sources_used": ["structural"],
  "unresolved_fields": [],
  "latency_ms": 812
}
```
A retry with NLU escalation:
```json
{
  "process_monitor_id": 1077,
  "doc_type": "Invoice",
  "attempts": 3,
  "sources_used": ["structural", "nlu_ner", "table_transformer"],
  "unresolved_fields": [],
  "latency_ms": 14308,
  "escalation_reason": "line_items sum mismatch in attempt 1; recovered by table_transformer"
}
```

Existing `proc.bp_discrepancy_data` table continues to receive discrepancy rows for post-extraction validation (redundant for structurally-extracted fields, still useful as a second-order check).

## Rollback

If production issues emerge after flag flip:
1. Set `USE_STRUCTURAL_EXTRACTOR=false` in `.env`, restart procwise.
2. The legacy LLM-first path resumes immediately; no DB changes required to roll back (new `Extraction_InReview` status values remain in history but aren't produced by legacy path).
3. Reprocess any `Extraction_InReview` records via the legacy path by resetting status to `'Completed'`.

## Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Type-driven discovery fails on docs with multiple same-type candidates (e.g., 5 dates, 3 totals) | Medium | Medium | Arithmetic-fit scoring (for MONEY) + label-proximity + positional priors disambiguate. If still tied, field remains unresolved → NLU + LLM escalation. |
| Line-item table parsing fails on unusual layouts | Medium | Medium | Per-format specialization: XLSX/DOCX/CSV use native structure; PDF spatial falls back to Table-Transformer (attempt 3). LLM arbiter last resort. |
| NLU models cost RAM and conflict with running AgentNick | Low | Low | NLU models load on CPU only (total ~3.5 GB RAM budget). AgentNick stays on GPU. No contention. |
| Retry loop gets stuck on a pathological doc for long periods | Low | Low | 10-attempt cap, then park in review_queue. Pipeline never blocks on one doc. |
| Ground truth has human error — test fails for a doc we mis-labeled | Medium | Low | Ground truth committed as a PR, reviewed by second engineer, corrected, re-reviewed. |
| Pattern store poisoning — a bad extraction writes a wrong pattern, corrupting future extractions | Medium | Medium | Patterns only written on extractions that passed ALL validation gates (math + cross-field + anchor verification). Stale/wrong cached patterns are detected by validation failure on next use and overwritten. `success_count` threshold before a pattern is trusted as primary (`≥ 3` successful uses). |
| DOCX/XLSX/CSV parsers mis-parse edge cases (empty cells, merged cells, multi-line paragraphs) | Medium | Medium | Each adapter has its own test suite on edge-case fixtures. Adapter errors raise `FormatParseError` → record marked `Extraction_Failed`, not `InReview`. |
| Overfitting to 32 golden docs; fails on production variety | Medium | High | Extractor uses type-driven discovery (not pattern-matching). Starts with zero hardcoded labels. Learns from production via pattern store. After rollout, monitor `Extraction_InReview` rate — if > 5% of docs land there, re-open design. |

## Open questions (to resolve during review)

1. **BERT NER** choice: `dslim/bert-base-NER` is general-purpose (PERSON/ORG/LOC/MISC). Should we also try a procurement-specific NER, or is general-purpose sufficient as supplementary signal? *Recommendation: start with general-purpose; escalate if error analysis shows NER isn't helping.*
2. **Review-queue resolution UI**: admin page to view/edit parked records, or direct DB access? *Recommendation: DB access for now; revisit if > 10 records/week land in queue.*
3. **Pattern-store migration** for existing `column_mapping` data: existing rows lack the new `anchor_patterns` and `type_priors` columns. Migration leaves them NULL and the extractor treats rows with NULL `anchor_patterns` as "vendor known, layout anchors not yet learned" — new pattern data is populated on next successful extraction. OK?

## Success criteria

- All 32 golden docs (across PDF, DOCX, XLSX, CSV) produce 100% field-level match in CI — no unresolved required fields, no math reconciliation failures, every extracted value anchor-verifiable, every derived value traceable to its rule+inputs.
- **Column coverage ≥ 90%**: across all 32 golden docs, ≥ 90% of `bp_*` columns (per the Column Coverage Matrix) are populated (non-NULL) after extraction + derivation. Columns that are WORKFLOW or SYSTEM are excluded from the denominator. Tracked per doc_type.
- Legacy-path fallback works correctly when flag is off (regression test passes).
- `proc.extraction_review_queue` exists and is populated only for docs with unresolved required fields after 10 retries.
- `proc.bp_extraction_patterns` populated with at least one pattern per vendor in the golden set; second-extraction-same-vendor test confirms pattern reuse.
- `proc.bp_extraction_provenance` populated: for every DB row written by the extractor, every non-SYSTEM non-WORKFLOW column has a provenance row with either `anchor_ref` (extracted) or `derivation_trace` (derived/inferred/lookup).
- Derivation rules unit-tested: every rule in the Derivation Registry has ≥ 1 direct unit test and ≥ 1 golden-doc integration test asserting the derived value.
- Metrics dashboard shows per-attempt source usage: target ≥ 70% of docs resolved at attempt 1 (structural or pattern-cached), ≥ 90% by attempt 3 (NLU tier), < 5% reach `Extraction_InReview`.

## Appendix A: what is hardcoded vs discovered vs learned vs derived

**Hardcoded (stable across every document, every vendor, every future change)**:
- Schema field types (`invoice_date: DATE`, `invoice_total_incl_tax: MONEY`, etc.) — derived from the DB schema itself.
- Procurement math invariants (`subtotal + tax = total`, `qty × price = line_total`) — fundamental arithmetic.
- Type detectors (how to recognize that a token is a DATE, MONEY, ORG, etc.) — general NLP, not procurement-specific.
- **The Derivation Registry**: the set of deterministic rules that compute derivable fields (`due_date = invoice_date + payment_terms_days`, currency-from-symbol mapping, postcode-to-country mapping, etc.). The rules are stable; the *values* they produce differ per document.

**Discovered per document**:
- What label text precedes each value in *this* document's layout.
- Which tokens/cells correspond to which schema field.
- The document's date locale (DMY vs MDY).
- The currency (when explicit).

**Learned per vendor/layout (persisted in `proc.bp_extraction_patterns`)**:
- Anchor positions (bbox / cell / column index) per field.
- Inferred label patterns per field (stored for observability; not used as hard constraint).
- Position priors (where each field tends to live for this vendor's layout) — aggregated cross-vendor in `proc.bp_extraction_type_priors`.

**Derived at extraction time (via the Derivation Registry)**:
- `due_date` from `invoice_date + payment_terms_days` (default 90).
- Missing amount/total from math inversion when 2 of 3 are extracted.
- `currency` from symbol when ISO code is absent.
- `converted_amount_usd` from amount × live FX rate.
- `supplier_id` / `buyer_id` via lookup/generate on `bp_supplier` master.
- `country` / `region` from postcode + address tokens.
- `invoice_status` = `"Issued"` / `po_status` = `"Open"` defaults.
- Line-item fills: `line_total`, `unit_price`, `quantity` when any two of the three are present.

The system ships with **zero label vocabulary** but does ship with a **finite, deterministic, tested Derivation Registry**. Every row written to the DB has complete provenance in `proc.bp_extraction_provenance`.
