# Data Extraction Redesign — Design

**Date:** 2026-05-06
**Author:** Muthu
**Status:** Draft, awaiting spec review
**Scope:** All four document types — Invoice, Purchase Order, Quote, Contract
**Cutover:** `agent_nick_orchestrator.py` preserved as a feature-flagged fallback; not deleted

## 1. Problem

The current extraction stack — `agent_nick_orchestrator.py` (≈2,900 LOC, 8 distinct LLM extraction call sites) sitting on top of `extraction_v2/` (≈6,000 LOC, deterministic, isolated) — does not produce reliable output. Concrete evidence drawn from `artifacts/log_monitor/ISSUES.md`:

- 60% of recent invoices closed with `lines=0` despite non-zero header totals (I-19, I-26).
- 40% of recent documents needed filename-fallback because the LLM emitted a wrong / hallucinated supplier (I-18: `'SUP-AuariusMarketing'`, `'INVOICE NUMBER: 4759275'`, `'UrbEdge Facilities Management Ltd'` for an Office Clean doc).
- The fine-tuned Ollama model leaks training-set names across documents (I-35, I-38: `'Eleanor Price'`, `'Jane Doe'`, `'Laura Stevens'` written into `requested_by` of unrelated invoices).
- Three TECHWORLD invoices land in the DB with `invoice_amount = 675` while `line_items_sum = 6,750` — a 10× scale mismatch that today's confidence score (0.86) does not reflect (I-39).
- Multi-column tables flatten into column-major order, losing row adjacency (I-37 — WADE, DUNCAN, AQUARIUS).
- Items-after-totals layouts (I-34 — UrbEdge PO 502004) produce empty line items even though every line is in the parsed text.

The root cause is not any single component. It is the **shape** of the pipeline: regex-based locators and LLM-based extraction race each other; no single source of truth exists for "what fields does an Invoice have"; the LLM is asked to extract from raw text and is the arbiter of its own output; and provenance (which model produced which value, citing what evidence) is not recorded, so failures cannot be diagnosed without re-running.

This document specifies a re-engineered pipeline that:

- Uses **deep learning, classical machine learning, NLP and NLU** for primary extraction. No regex is allowed in the primary extraction path.
- Uses **the LLM only as a layered judge** — never as a free-form extractor.
- Runs **fully locally** on GPU (no cloud OCR, no Textract).
- Carries **per-field cited evidence** through to the database for audit.
- Is **schema-driven**: every field is declared once, and the runtime configures the extractors and judge from that declaration.
- Targets **all four document types** simultaneously.

## 2. Constraints (locked from brainstorming)

These are decisions that the design must respect; they are not open for re-litigation in the spec review.

| # | Constraint | Source |
|---|-----------|--------|
| C1 | Inputs: native PDF, scanned PDF, DOCX, image (PNG / JPEG). | User brief 2026-05-06 |
| C2 | All processing local on GPU. No cloud OCR, no Textract. | User brief 2026-05-06 |
| C3 | No regex as the primary extraction mechanism. Regex may exist *only* inside post-extraction sanity checks (e.g. ISO-date format check after a value is already extracted), not as a way to find a value in the document. | User brief 2026-05-06; existing memory `feedback_no_regex_fixes.md` |
| C4 | LLM only as a layered judge: tiebreaker on disagreement, grounded last-resort with mandatory citation, schema-coherence final pass. The judge never sees a raw document and asks "extract everything." | User decision (Q1 = D) |
| C5 | Schema source-of-truth is hybrid: `proc.bp_*` tables remain authoritative for storage shape; per-doc-type YAML files carry extraction metadata. A startup consistency check fails the service on drift. | User decision (Q2 = D) |
| C6 | Phase 1 covers all four document types: Invoice, Purchase Order, Quote, Contract. | User decision |
| C7 | `agent_nick_orchestrator.py` is preserved in the codebase. It is moved behind a per-doc-category feature flag and remains available as an emergency fallback. It is not the live path once the new pipeline ships. | User decision |
| C8 | 100% accuracy is the primary success metric. Speed, throughput, cost are secondary. If a value cannot be extracted with cited evidence, it is left NULL and the document is enqueued for review (existing memory `feedback_no_fabrication_null_when_absent.md`). | Existing memory + user brief |
| C9 | Every committed value must carry a verifiable evidence span (page, bbox, source-text substring) that downstream consumers can audit. | Derived from C4 + accuracy mandate |

## 3. Architecture overview

Three layers, one direction of flow, one schema.

```
PDF / scanned PDF / DOCX / image
            │
            ▼
   ┌──────────────────────────────────────────────────┐
   │  L1  Universal Document Parser (DL-based)        │
   │       routes by file type → ParsedDocument       │
   │       (pages → regions → tables → tokens+bbox)   │
   └────────────────────────┬─────────────────────────┘
                            ▼
   ┌──────────────────────────────────────────────────┐
   │  L2  Multi-Model Candidate Generation            │
   │       LayoutLMv3, Table Transformer, sBERT,      │
   │       spaCy NER, extractive QA, vendor template  │
   │       each emits (field, value, evidence, conf)  │
   └────────────────────────┬─────────────────────────┘
                            ▼
   ┌──────────────────────────────────────────────────┐
   │  L3  Schema Bind + Invariants + Layered Judge    │
   │       typed coercion, business invariants,       │
   │       LLM-as-judge (tiebreaker /                 │
   │       grounded-last-resort / coherence)          │
   └────────────────────────┬─────────────────────────┘
                            ▼
   proc.bp_*          proc.bp_extraction_provenance
   (committed)        (one row per committed field)
                            │
                            ▼
   residuals → proc.extraction_review_queue
```

The boundary between layers is a frozen Pydantic schema. A change to `ParsedDocument`, `Candidate`, or `ExtractionResult` is a versioned breaking change; layers do not reach into each other's internals.

## 4. Layer 1 — Universal Document Parser

One Python interface, four backends, auto-routed by file type and a "is this a scanned PDF?" classifier.

| Input | Backend | Notes |
|-------|---------|-------|
| Native PDF (selectable text) | **Docling** (IBM, fully local, runs DL layout models) | Produces structured tree: pages → regions → tables → cells → tokens with bboxes, font + reading-order metadata. Handles rotated pages and multi-column natively. |
| DOCX | **Docling** (built-in DOCX support) | Same `ParsedDocument` schema. Preserves table cells and headings. |
| Image (PNG/JPEG) | **PaddleOCR PP-Structure** (GPU) | Layout detection + OCR + table-structure recognition in one pass. Multilingual. |
| Scanned PDF (image-only PDF) | **PaddleOCR PP-Structure** per page | Detected via `pdfplumber` page text count = 0; page rasterized at 300 DPI then fed to PP-Structure. |
| Hard cases (PP-Structure low conf) | **Donut** (image → JSON, OCR-free encoder-decoder) | Fallback when layout extractor fragments severely. Used for the page only, not the whole doc. |

Output is a single `ParsedDocument` Pydantic schema regardless of input format:

```python
class Token(BaseModel):
    text: str
    page: int
    bbox: tuple[float, float, float, float]   # (x0, y0, x1, y1) in page coords
    font_size: float | None
    is_bold: bool

class Cell(BaseModel):
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    row_index: int
    col_index: int
    row_span: int = 1
    col_span: int = 1

class Table(BaseModel):
    page: int
    bbox: tuple[float, float, float, float]
    rows: list[list[Cell]]              # rows[r][c]
    header_row_index: int | None

class Region(BaseModel):
    page: int
    bbox: tuple[float, float, float, float]
    role: Literal["header", "footer", "body", "address-block", "table", "logo", "signature"]
    text: str

class Page(BaseModel):
    index: int
    width: float
    height: float
    rotation: int                       # 0, 90, 180, 270
    regions: list[Region]
    tables: list[Table]
    tokens: list[Token]

class ParsedDocument(BaseModel):
    source_path: str
    file_format: Literal["pdf-native", "pdf-scanned", "docx", "image"]
    pages: list[Page]
    full_text: str                      # reading-order concatenation, useful for QA model
    parser_backend: str                 # "docling" / "paddleocr" / "donut"
    parser_confidence: float            # backend-reported overall confidence
```

Layer 1 deletes the parsing portions of `intelligent_extractor.py`, `pdf_table_recovery.py`, and the OCR fallbacks scattered across services. It does **not** delete any business logic.

A "is this a scanned PDF?" classifier is a one-line check: if `pdfplumber` extracts ≤ 5 characters per page on average, treat as scanned. No DL model needed for that decision.

## 5. Layer 2 — Multi-model candidate generation

Each Layer-2 component emits zero or more `Candidate` records per schema field. Multiple components may emit candidates for the same field — agreement / disagreement is resolved in Layer 3.

```python
class Candidate(BaseModel):
    field: str                              # "invoice_id", "supplier_name", "line[3].amount", ...
    value: str                              # raw string before type coercion
    page: int
    bbox: tuple[float, float, float, float]
    evidence_text: str                      # the literal substring the value came from
    model: Literal["layoutlmv3", "table_transformer",
                   "sbert_anchor", "spacy_ner", "qa_roberta",
                   "vendor_template"]
    confidence: float                       # in [0, 1], model-reported
```

### 5.1 LayoutLMv3 (primary header-field extractor)

Hugging Face token-classification model, fine-tuned per doc-type on the user's existing labeled corpus (see §11). For every token in the `ParsedDocument`, predicts a label like `B-INVOICE_ID`, `I-SUPPLIER_NAME`, `B-LINE_DESC`, `O`, etc. (BIO scheme). Spans are aggregated into field candidates.

Inputs to the model: token text, token bbox, page image (rasterized). LayoutLMv3 is multimodal — it uses text, 2-D position, and visual features.

Replaces: regex-based locators in `structural_extractor`, label-anchored regex in `extraction_v2/locator/strategies/`, the `_llm_extract` calls in `direct_extraction_service.py`.

### 5.2 Table Transformer (primary line-item extractor)

Microsoft Table Transformer (`microsoft/table-transformer-structure-recognition-v1.1-all`) detects table structure end-to-end on rasterized pages — rows, columns, cells, header rows, projected merges. Pairs cells into line-item rows mapped to the schema's line-item field set (`description`, `quantity`, `unit_price`, `amount`, `tax_amount`, etc.).

Replaces: `pdf_table_recovery.py`, `line_recovery.py`'s heuristic sweeps. Both are deleted from the live path.

For the items-after-totals case (I-34): Table Transformer finds the table by structure, not by position relative to totals, so this layout class works without special-casing.

For the multi-column-flatten case (I-37): the rasterized-page input is 2-D, so column boundaries are visible to the model — column-major flattening cannot happen by construction.

### 5.3 Sentence-Transformers (semantic label-anchor mapping)

`sentence-transformers/all-mpnet-base-v2`. For ambiguous label text near a value (e.g. "Sold To:", "Bill From:", "Account Manager:"), embeds the label and computes cosine similarity to canonical field-label embeddings declared in the YAML schema. Picks the field whose canonical label is closest.

Replaces: label-pattern regex tables. Used as a tiebreaker for LayoutLM when two adjacent labels could both plausibly attach to a value.

### 5.4 spaCy NER (free-text validator)

A custom procurement-tuned spaCy pipeline (built on `en_core_web_trf`) validates that free-text candidates are *plausibly* the right entity type. Examples:

- `supplier_name` candidate must contain at least one `ORG` entity.
- `requested_by` candidate must contain at least one `PERSON` entity.
- Address candidates must contain at least one `GPE` or `LOC` entity.

A candidate that fails the NER type check is downgraded, not deleted, and Layer 3's judge sees the failure as a signal.

This is what catches the "INVOICE NUMBER: 4759275" supplier-name regression observed in I-18 — that string contains zero `ORG` entities so its confidence is downgraded before the judge ever sees it.

### 5.5 Extractive QA (residual gap-filler)

A RoBERTa-based extractive QA model fine-tuned on SQuAD2 + a small procurement-domain Q-A set. Given the `ParsedDocument.full_text` and a question, it returns either an answer span with confidence or "no-answer." Used **only** for required fields where neither LayoutLMv3 nor Table Transformer produced a candidate.

The QA model's output is treated as a candidate exactly like any other, with `model = "qa_roberta"`. It does not bypass the judge. Crucially: it returns a span, not free-form text — so by construction the value already exists in the source.

Replaces: `_llm_fill_gaps`, `_llm_extract_single_service` from `agent_nick_orchestrator.py`.

### 5.6 Vendor template hint (deterministic locator, kept from `extraction_v2`)

`extraction_v2/template_store_pg.py`, `template_service.py`, and `fingerprint.py` are kept and integrated as one Layer-2 candidate source. When the document's layout fingerprint matches a learned template, the template's stored field locations are emitted as candidates with `model = "vendor_template"` and a high prior confidence.

Vendor templates are **learned**, not authored — when an analyst corrects a doc in the review queue, the corrections are written back into the template store. This is the existing `vendor_onboarding_api` (commit `829c94c`).

## 6. Layer 3 — Schema bind + invariants + layered judge

### 6.1 Schema YAML format

Per doc type, one YAML file under `extraction_schemas/`:

```yaml
# extraction_schemas/invoice.yaml
doc_type: invoice
db_table: proc.bp_invoice
db_lines_table: proc.bp_invoice_line_items

fields:
  - name: invoice_id
    type: string
    required: true
    db_column: invoice_id
    canonical_labels: ["Invoice Number", "Invoice No", "Invoice #", "Inv No", "Document Number"]
    extractors: [layoutlmv3, vendor_template, qa_roberta]
    judge:
      tiebreaker: enabled
      grounded_last_resort: enabled
      ner_type_check: none

  - name: supplier_name
    type: string
    required: true
    db_column: supplier_name
    canonical_labels: ["From", "Bill From", "Vendor", "Supplier", "Sold By"]
    extractors: [layoutlmv3, sbert_anchor, vendor_template, qa_roberta]
    judge:
      ner_type_check: ORG       # spaCy must find an ORG entity in the candidate
      grounded_last_resort: enabled

  - name: invoice_date
    type: iso_date
    required: true
    db_column: invoice_date
    canonical_labels: ["Invoice Date", "Date", "Date of Issue", "Issued"]
    extractors: [layoutlmv3, qa_roberta]
    judge:
      grounded_last_resort: enabled
    invariants: [date_sanity]

  - name: invoice_amount
    type: money
    required: true
    db_column: invoice_amount
    canonical_labels: ["Subtotal", "Net Amount", "Amount Before Tax"]
    extractors: [layoutlmv3, qa_roberta]
    invariants: [subtotal_closure]

  # ... 20+ more fields

line_items:
  primary_extractor: table_transformer
  fallback_extractor: layoutlmv3   # token-classified line spans, when no table detected
  fields:
    - name: description
      type: string
      required: true
    - name: quantity
      type: decimal
      required: false
    - name: unit_price
      type: money
      required: false
    - name: amount
      type: money
      required: true
  invariants: [line_arithmetic, line_sum_closure]

document_invariants: [tax_closure, grand_total_closure, currency_consistency]
```

A single startup hook reads the YAML for every doc type and verifies:
- Every `db_column` exists in the named table with a compatible type.
- Every named extractor / invariant exists in the runtime registry.
- No required field is missing a `canonical_labels` list.

If any check fails, the service refuses to start with a precise error message naming the field and the inconsistency. This is the C5 "fail fast on drift" mechanism.

### 6.2 Type binding

Existing `extraction_v2/parsers/` typed parsers (`Money`, `IsoDate`, `Address`, `Postcode`, etc.) handle string-to-typed coercion. A coercion failure does **not** silently null the field — it produces a `bind_error` signal that Layer 3 routes to the judge as a tiebreaker between conflicting candidates, and to the review queue if no candidate coerces.

### 6.3 Invariants

Kept from `extraction_v2/invariants.py` unchanged. These are arithmetic and business logic, not regex. The 10 existing invariants:

1. `LineArithmetic` — `quantity × unit_price ≈ amount` per line.
2. `LineSumClosure` — `sum(line.amount) ≈ subtotal`.
3. `SubtotalClosure` — `subtotal ≈ sum(line.amount)`.
4. `TaxClosure` — `tax_amount ≈ subtotal × tax_percent / 100`.
5. `GrandTotalClosure` — `subtotal + tax_amount + adjustments ≈ total`.
6. `CurrencyConsistency` — every monetary field uses the same currency.
7. `DateSanity` — `invoice_date ≤ due_date`, no future invoice dates ≥ +1 day, no historical dates more than 5 years prior.
8. `VendorIdentity` — extracted supplier resolves to an existing `proc.bp_supplier` row, or is auto-created with a flag.
9. `QuantitySign` — line quantities non-negative unless the doc is a credit note.
10. `RoundOffBucket` — closure-mismatch ≤ £0.05 is logged but not failed; > £0.05 demotes confidence.

A new 11th invariant addresses I-39 (TECHWORLD scale mismatch):

11. `ScaleMismatch` — `|line_sum / invoice_amount| > 9` ⇒ critical demotion. Catches 10× decimal-point misreads.

CRITICAL invariant failures force the offending field(s) to residual; WARNING failures demote confidence below the commit threshold.

### 6.4 Layered LLM judge — contract

The judge is the only LLM call site in the entire pipeline. Local Ollama, model `BeyondProcwise/AgentNick:judge` (a fine-tune dedicated to judging — see §11). Three invocation types, all on the same model, all with strict input/output contracts.

#### 6.4.1 Tiebreaker

Fires only when ≥ 2 Layer-2 candidates for the same field disagree (different normalized values).

```
Input contract (JSON):
{
  "field": "supplier_name",
  "field_type": "string",
  "candidates": [
    {"value": "Aquarius Marketing Ltd", "model": "layoutlmv3",  "confidence": 0.81, "evidence": "...", "page": 1, "bbox": [...]},
    {"value": "AuariusMarketing",        "model": "qa_roberta",  "confidence": 0.62, "evidence": "...", "page": 1, "bbox": [...]}
  ],
  "context_text": "<surrounding 200 chars from each candidate's page>"
}

Output contract (JSON):
{
  "chosen_candidate_index": 0,           # or null
  "rationale": "<one sentence>"
}
```

The orchestrator post-validates that `chosen_candidate_index` is a valid index or null. The judge is not allowed to invent a new value.

#### 6.4.2 Grounded last-resort

Fires only for required fields where Layer 2 produced **zero** candidates.

```
Input contract (JSON):
{
  "field": "invoice_id",
  "field_type": "string",
  "field_canonical_labels": ["Invoice Number", "Invoice No", "Invoice #"],
  "doc_full_text": "<entire ParsedDocument.full_text>",
  "constraints": {
    "must_be_verbatim_substring_of_doc_full_text": true,
    "max_length": 64
  }
}

Output contract (JSON):
{
  "value": "INV148769",                  # or null
  "evidence_text": "INV148769",          # must equal value, must be a substring of doc_full_text
  "rationale": "<one sentence>"
}
```

The orchestrator post-validates that `value == evidence_text` AND `evidence_text in doc_full_text`. If either fails, the candidate is rejected and the field stays NULL → review queue.

This is the structural anti-hallucination mechanism. The model cannot invent "Eleanor Price" for a TECHWORLD invoice because "Eleanor Price" is not a substring of the doc. Even if the fine-tune leaks training-set tokens, the substring check kills the leak.

#### 6.4.3 Schema-coherence final pass

One call per document, after all fields are bound and invariants run.

```
Input contract (JSON):
{
  "doc_type": "invoice",
  "extracted_record": {
    "invoice_id": "INV148769",
    "supplier_name": "PERRY",
    "invoice_date": "2025-10-12",
    "invoice_amount": "1500.00",
    "currency": "GBP",
    "line_items": [...]
  },
  "invariant_results": [
    {"name": "subtotal_closure", "passed": true, "delta": 0.00},
    {"name": "tax_closure",       "passed": true, "delta": 0.00}
  ]
}

Output contract (JSON):
{
  "verdict": "coherent" | "incoherent",
  "issues": [
    {"field": "<name>", "issue": "<one sentence>"}
  ]
}
```

`incoherent` verdict demotes the document confidence below the KG-sync threshold and enqueues for review. The judge cannot mutate the record.

This catches the I-38 "Eleanor Price written into requested_by of unrelated invoices" class of failure — the schema-coherence pass sees a TECHWORLD invoice with an Eleanor-Price-shaped `requested_by` and flags it incoherent. (The grounded substring check from §6.4.2 already prevents Layer 2 from producing such a value, but the coherence pass is the second line of defence for cross-field consistency.)

#### 6.4.4 Cost ceiling

A per-document hard ceiling: at most `(N_disagreement_fields × 1) + (N_required_missing × 1) + 1` judge calls. For a clean doc the judge fires exactly once (the schema-coherence pass). For a doc with 2 disagreements and 1 missing required field, it fires 4 times. There is no retry-on-low-confidence loop — that pattern is responsible for the legacy stack's runaway latency.

### 6.5 Provenance — `proc.bp_extraction_provenance_v3`

**Note (clarification 2026-05-08):** the original spec named this table `proc.bp_extraction_provenance` but that name is already taken by the legacy AgentNick path (see `src/services/extraction_v2/provenance.py`). Per C7 (AgentNick preserved), the V3 provenance table is a NEW, separate table named `_v3` so both can coexist. The legacy table's schema (`parent_table` / `parent_pk` / `field_name` / `source`) differs from V3's bbox-and-evidence schema below.

A new table. One row per committed field (header or line item).

```sql
CREATE TABLE proc.bp_extraction_provenance_v3 (
    provenance_id    BIGSERIAL PRIMARY KEY,
    doc_type         TEXT NOT NULL,           -- 'invoice' | 'purchase_order' | 'quote' | 'contract'
    doc_pk           TEXT NOT NULL,           -- e.g. invoice_id
    field_path       TEXT NOT NULL,           -- 'invoice_id' or 'line_items[3].amount'
    value            TEXT NOT NULL,
    page             INT NOT NULL,
    bbox_x0          REAL NOT NULL,
    bbox_y0          REAL NOT NULL,
    bbox_x1          REAL NOT NULL,
    bbox_y1          REAL NOT NULL,
    evidence_text    TEXT NOT NULL,
    model            TEXT NOT NULL,
    model_confidence REAL NOT NULL,
    judge_actions    JSONB,                   -- ['tiebreaker', 'coherence_ok']
    final_confidence REAL NOT NULL,
    extracted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pipeline_version TEXT NOT NULL,
    UNIQUE (doc_type, doc_pk, field_path, extracted_at)
);

CREATE INDEX idx_provenance_v3_doc ON proc.bp_extraction_provenance_v3 (doc_type, doc_pk);
```

Every committed write to `proc.bp_invoice / _line_items / etc.` is paired in the same transaction with provenance rows. If the provenance write fails, the data write rolls back. This makes 100% of committed values auditable.

### 6.6 Review queue

Existing `proc.extraction_review_queue` is kept. Reasons added:

- `unsupported_layout` — Layer 1 parser confidence below threshold.
- `required_field_missing_no_grounding` — Layer 2 produced no candidates and grounded-last-resort returned null.
- `invariant_critical_failed` — one of the 11 invariants returned CRITICAL.
- `judge_incoherent` — schema-coherence pass returned `incoherent`.
- `bind_error_no_resolution` — type coercion failed for all candidates.

A document with any required residual is enqueued; a document with only optional residuals commits and is **not** enqueued.

## 7. End-to-end flow for one document

1. `process_monitor_watcher` receives `process_monitor_ready` NOTIFY for record N.
2. Watcher reads doc path, dispatches to **Pipeline V3** (new).
3. **Layer 1**: parser is selected by file format and scanned-classifier. `ParsedDocument` is produced.
4. **Layer 2**: each extractor named in the YAML for this doc type runs in parallel, producing `Candidate` records. Vendor template hints are looked up by fingerprint. Empty results from any extractor are not errors.
5. **Layer 3 — bind**: candidates are grouped by field. For each field, the highest-confidence candidate that passes type coercion is provisional. If multiple candidates disagree → **judge tiebreaker**. If no candidate exists for a required field → **judge grounded last-resort**. If no value commits and the field is required, the field is residual.
6. **Layer 3 — invariants**: all 11 invariants run. CRITICAL failures move fields to residual. WARNING failures demote confidence.
7. **Layer 3 — coherence**: the assembled record is shown to the **judge schema-coherence pass**. Incoherent verdict demotes the whole document.
8. **Persist**: in a single transaction, write `proc.bp_*` rows AND `proc.bp_extraction_provenance` rows. Update `process_monitor` row status. If any required residual exists, enqueue to `proc.extraction_review_queue` with a reason code.
9. **KG sync**: existing `procurement_kg_builder` runs only when document confidence ≥ 0.70 AND no required residuals. (Same threshold as today, applied to the new confidence score.)

## 8. Cutover / coexistence with `agent_nick_orchestrator.py`

`agent_nick_orchestrator.py` is kept in the repo. It is reachable only through a feature flag.

```python
# .env
EXTRACTION_PIPELINE_INVOICE=v3        # 'v3' | 'agentnick'
EXTRACTION_PIPELINE_PURCHASE_ORDER=v3
EXTRACTION_PIPELINE_QUOTE=v3
EXTRACTION_PIPELINE_CONTRACT=v3
```

`process_monitor_watcher` reads the per-category flag and dispatches accordingly. Default in production is `v3` for all four. If a regression is observed for one category, ops flips that one category back to `agentnick` without touching the others. No code change, no redeploy.

The flag is per-category, not per-document, because mixing pipelines per-document creates incomparable provenance and is a debugging nightmare.

`agent_nick_orchestrator.py`'s 8 LLM extraction call sites stay where they are — they are simply not exercised when the flag is `v3`. They are not deleted because:
- The user explicitly chose to preserve them.
- They are the rollback path. Deleting them removes that escape hatch.
- They contain accumulated tribal knowledge that is hard to re-derive — keeping them is cheap insurance.

A follow-up project (out of scope here) can decide whether to delete after some weeks of stable v3 operation.

## 9. Testing strategy

Three tiers.

### 9.1 Unit tests
- One per extractor: `tests/extraction_v3/test_layoutlmv3_invoice.py`, `test_table_transformer.py`, `test_qa_roberta.py`, etc. Each loads a fixture `ParsedDocument` and asserts `Candidate` records.
- One per invariant.
- One per judge invocation type with a hand-crafted prompt and a recorded expected output. Judge tests use a frozen Ollama model snapshot.

### 9.2 Integration tests
- 30 fixture documents per doc type (mix of clean / multi-column / scanned / DOCX / hard-vendor) ⇒ 120 documents total. Each fixture has hand-labeled ground truth.
- For each fixture, the pipeline runs end-to-end and the result is diffed against ground truth field-by-field. Pass criterion: 100% of fields match for the "clean" subset; 95%+ for the "hard" subset.

### 9.3 Live-shadow comparison
For two weeks pre-cutover, every newly-arrived production document is run through **both** `agentnick` and `v3`. Disagreements are written to `proc.extraction_pipeline_diff` for analyst review. v3 is not the live path until shadow mode shows ≥ 99% field-level agreement on documents where agentnick was correct, AND ≥ 90% of agentnick's known wrong outputs are corrected by v3.

Shadow comparison consumes 2× compute for two weeks but is the cheapest way to discover regressions before they land.

## 10. Performance budget

Per document, on the existing GPU:

| Stage | Budget |
|-------|--------|
| Layer 1 parse (Docling, native PDF, ~5 pages) | < 3 s |
| Layer 1 parse (PaddleOCR, scanned PDF, ~5 pages) | < 8 s |
| Layer 2 LayoutLMv3 (one pass) | < 2 s |
| Layer 2 Table Transformer | < 2 s |
| Layer 2 QA, sBERT, spaCy, vendor template | < 1 s combined |
| Layer 3 bind + invariants | < 0.1 s |
| Layer 3 judge calls (worst case 4 calls) | < 8 s |
| **End-to-end p95** | **< 25 s** |

For comparison, today's CPU-bound legacy path is 5–14 minutes per document (I-17). The 20–30× speedup is a side-effect of staying on GPU and avoiding the LLM-as-extractor retry loops.

## 11. Training-data strategy

The single highest-leverage one-time investment is fine-tuning LayoutLMv3 on the user's existing labeled data.

**Source:** `proc.bp_invoice / _line_items / etc.` already contain ~thousands of extracted records. Joined with their parsed text from S3 canonical (and the document images), this is a labeled corpus of (document, field, value) triples.

**Quality filter:** only records that downstream-validated successfully (KG-synced + no review-queue entry + manual confirmation if available) are used as training data. Documents the legacy stack got wrong are explicitly excluded — we don't want to fine-tune on the failure modes we are trying to eliminate.

**Cleaning:** the I-38 fabrications (`'Jane Doe'`, `'Eleanor Price'`, `'Laura Stevens'` written into `requested_by` of unrelated invoices) need to be cleaned from the corpus before training. A pre-training audit pass equivalent to `/tmp/hallucination_audit.py` runs against the corpus and removes records where any field's substantive tokens don't appear in the source text.

**Splits:** 80/10/10 train/val/test, stratified by vendor and document category, so a held-out vendor exists in test for generalization measurement.

The judge model fine-tune is a separate, smaller task: a few hundred (input-contract, output-contract) examples per invocation type, focused on teaching it the strict output format. Started after Layer 2 is stable so judge training can use real Layer 2 outputs.

## 12. What this design does not include

- **Vision beyond Layer 1**: no end-to-end vision-language extraction via hosted multimodal APIs. Out of scope per C2 (local only) and C4 (LLM as judge only).
- **Active learning loop**: Phase 1 retrains LayoutLMv3 quarterly on a fresh dump. A continuous-learning loop where review-queue corrections trigger automatic retraining is a Phase 2 item.
- **Multi-language support**: the judge prompts and YAML schemas are English. Multilingual is post-Phase-1.
- **Email-body extraction**: emails are routed by `email_watcher` to the same pipeline only after attachments are extracted. Body-only invoices are not yet supported and continue through `agentnick`.
- **`agent_nick_orchestrator.py` deletion**: explicitly out of scope per C7. Possible follow-up.
- **`proc.bp_extraction_provenance` as a queryable provenance UI**: schema and writes only; no read-side surface in Phase 1.

## 13. Success criteria for Phase 1 launch

The new pipeline is the live path for all four document types when, measured over the live-shadow comparison window:

1. Field-level agreement with legacy on documents the legacy stack got right: ≥ 99%.
2. Recovery rate on documents the legacy stack got wrong (per `ISSUES.md` audit): ≥ 90%.
3. Zero regression on the 11 invariants on the integration test set.
4. End-to-end p95 latency: < 25 s per document.
5. Per-document judge call count: p95 ≤ 3, p99 ≤ 4.
6. Provenance row written for 100% of committed fields.
7. No hallucinated value committed across 1,000-document audit (defined as: a committed value whose `evidence_text` is not a substring of `ParsedDocument.full_text`).

Failure of any criterion blocks cutover for that doc type. The flag stays on `agentnick` for that category until the gap is closed.
