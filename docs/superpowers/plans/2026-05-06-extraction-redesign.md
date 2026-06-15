# Extraction Redesign — Phase 1 Plan 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an end-to-end Pipeline V3 — schema-driven, deterministic-first, LLM-as-judge — that can extract Invoice documents from PDF / scanned PDF / DOCX / image into `proc.bp_invoice` + `proc.bp_invoice_line_items` with provenance, behind a per-category feature flag (off by default). Subsequent plans add fine-tuning, the other three doc types, and live-shadow cutover.

**Architecture:** Three layers behind one Pydantic boundary. Layer 1 = Universal Document Parser (Docling for native PDF / DOCX, PaddleOCR PP-Structure for scanned / image, Donut as PP-Structure low-confidence fallback) producing one `ParsedDocument` schema. Layer 2 = multi-model candidate generation (LayoutLMv3 + Table Transformer + sentence-transformers + spaCy NER + extractive QA + vendor template), each candidate carrying cited evidence. Layer 3 = type-bind + 11 invariants + layered LLM judge (tiebreaker on disagreement, grounded last-resort with mandatory substring citation, schema-coherence final pass).

**Tech Stack:** Python 3.12, PyTorch (CUDA), `docling`, `paddleocr` + `paddlepaddle-gpu`, `donut-python` (`naver-clova-ix/donut-base`), `transformers` + `microsoft/layoutlmv3-base`, `microsoft/table-transformer-structure-recognition-v1.1-all`, `sentence-transformers/all-mpnet-base-v2`, `spacy` + `en_core_web_trf`, RoBERTa SQuAD2 (`deepset/roberta-base-squad2`), local Ollama (`BeyondProcwise/AgentNick:judge` — fine-tune deferred to Plan 2; pre-trained model used here), PostgreSQL, pytest.

**Spec:** `docs/superpowers/specs/2026-05-06-extraction-redesign-design.md` (commit `cff2bef`). All nine constraints C1–C9 in §2 are non-negotiable.

**Out of scope for THIS plan:**
- LayoutLMv3 fine-tune (Plan 2 — uses pre-trained `microsoft/layoutlmv3-base` here for the smoke path; accuracy iteration begins after this plan ships)
- PO / Quote / Contract YAML schemas + fixtures (Plan 3 — a stub YAML that satisfies the startup consistency check is included here, but no extractor wiring or fixtures for those types)
- Live-shadow comparison + cutover decision (Plan 4)
- `proc.bp_extraction_provenance_v3` query/UI surface (writes only this plan)

**Iteration mandate (per project memory `project_extraction_redesign_iteration_mandate.md`):** This is project-defining. The previous extraction stacks failed by stopping iteration before accuracy was reached — that mistake is not to be repeated.

- When any task fails an accuracy or zero-hallucination assertion, **stop and fix it in-plan first** before deferring to a future plan. Investigate root cause (parser? extractor config? judge prompt? bind logic? invariant gap?) and remedy. Only escalate to "Plan 2 fine-tune" if the failure genuinely requires labeled-corpus work that cannot be solved by configuration / prompt / additional invariants.
- Once this plan ships green, immediately run live extraction tests against the integration fixture set + the live `proc.process_monitor` queue, audit accuracy and zero-hallucination, and **continue iterating** (corpus, fine-tuning, judge-prompt refinement, additional invariants, additional vendor templates, additional fixtures) until all seven §13 success criteria pass.
- Do **not** declare success on plan completion alone. Plan completion is the start of accuracy iteration, not the end of work.
- Hallucinations (a committed value whose `evidence_text` is not a substring of `ParsedDocument.full_text`) are P0 — fix immediately, do not ship a release that has any.

**Approach this as a senior Python / ML architect would.** No shortcuts. No regex band-aids. No "good enough for now" thresholds. If a layout class breaks an extractor, research the failure mode (papers, model cards, HuggingFace forums), pick the right model / config / fine-tune, and verify. Speed of iteration matters; correctness of iteration matters more.

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `scripts/migrations/2026-05-08-extraction-provenance-v3.sql` | DB migration creating `proc.bp_extraction_provenance_v3` (note: `proc.bp_extraction_provenance` is already taken by the legacy AgentNick path via `src/services/extraction_v2/provenance.py` — this is a NEW, V3-only table) |
| `src/services/extraction_v3/__init__.py` | Module marker, re-export public API |
| `src/services/extraction_v3/schemas/parsed_document.py` | Pydantic: `ParsedDocument`, `Page`, `Region`, `Token`, `Cell`, `Table` |
| `src/services/extraction_v3/schemas/candidate.py` | Pydantic: `Candidate` (the L1→L2→L3 boundary record) |
| `src/services/extraction_v3/schemas/result.py` | Pydantic: `ExtractionResult`, `ResidualReason`, `JudgeAction` |
| `src/services/extraction_v3/yaml_schema/loader.py` | Load `extraction_schemas/*.yaml`, validate against DB, fail-fast on drift |
| `src/services/extraction_v3/yaml_schema/registry.py` | Extractor + invariant runtime registry; `register_extractor`, `register_invariant` |
| `src/services/extraction_v3/parsers/router.py` | Pick parser backend by `(file_format, is_scanned)` |
| `src/services/extraction_v3/parsers/scanned_classifier.py` | Detect "is this PDF actually a scan?" (≤ 5 chars/page heuristic) |
| `src/services/extraction_v3/parsers/docling_backend.py` | Docling → `ParsedDocument` for native PDF + DOCX |
| `src/services/extraction_v3/parsers/paddleocr_backend.py` | PaddleOCR PP-Structure → `ParsedDocument` for scanned PDF + images |
| `src/services/extraction_v3/parsers/donut_backend.py` | Donut → `ParsedDocument` (per-page fallback when PP-Structure low conf) |
| `src/services/extraction_v3/extractors/base.py` | `Extractor` ABC: `produce_candidates(parsed: ParsedDocument, schema: DocSchema) -> list[Candidate]` |
| `src/services/extraction_v3/extractors/layoutlmv3.py` | Token-classification → `Candidate` per labeled span |
| `src/services/extraction_v3/extractors/table_transformer.py` | Detect tables on rasterized pages → line-item `Candidate` rows |
| `src/services/extraction_v3/extractors/sbert_anchor.py` | Semantic label-anchor → field mapping |
| `src/services/extraction_v3/extractors/spacy_ner.py` | NER type-check; demotes candidates that fail (does not delete them) |
| `src/services/extraction_v3/extractors/qa_roberta.py` | Extractive QA gap-filler for required fields with no candidate |
| `src/services/extraction_v3/extractors/vendor_template.py` | Wraps `extraction_v2/template_store_pg.py` as a Layer-2 extractor |
| `src/services/extraction_v3/binding/type_binder.py` | Coerce `Candidate.value` → typed Python value via `extraction_v2/parsers` |
| `src/services/extraction_v3/binding/invariants_runner.py` | Run all 11 invariants in order, collect results |
| `src/services/extraction_v3/binding/scale_mismatch.py` | New invariant: `\|line_sum / invoice_amount\| > 9` ⇒ CRITICAL |
| `src/services/extraction_v3/judge/contracts.py` | Frozen JSON schemas for tiebreaker / grounded / coherence |
| `src/services/extraction_v3/judge/orchestrator.py` | Decide which judge invocations to run; enforce per-doc cost ceiling |
| `src/services/extraction_v3/judge/tiebreaker.py` | Tiebreaker invocation; post-validate `chosen_candidate_index` |
| `src/services/extraction_v3/judge/grounded_last_resort.py` | Grounded last-resort; post-validate substring of `doc_full_text` |
| `src/services/extraction_v3/judge/schema_coherence.py` | Schema-coherence pass; treat verdict as advisory demotion |
| `src/services/extraction_v3/pipeline.py` | `PipelineV3.run(doc_path, doc_type) -> ExtractionResult` — wires L1→L2→L3 |
| `src/services/extraction_v3/persistence.py` | Single-tx write to `proc.bp_*` + `proc.bp_extraction_provenance_v3` |
| `src/services/extraction_v3/dispatch.py` | Per-category feature flag gate; route to v3 or `agent_nick_orchestrator` |
| `extraction_schemas/invoice.yaml` | Full invoice schema (header fields + line items) |
| `extraction_schemas/purchase_order.yaml` | Stub for startup consistency check (Plan 3 fills it) |
| `extraction_schemas/quote.yaml` | Stub for startup consistency check (Plan 3 fills it) |
| `extraction_schemas/contract.yaml` | Stub for startup consistency check (Plan 3 fills it) |
| `tests/extraction_v3/conftest.py` | pytest fixtures: model loaders (session-scoped), fixture invoice docs |
| `tests/extraction_v3/fixtures/invoices/INV-001.pdf` (5 fixtures) | Hand-picked invoice PDFs covering clean / multi-column / scanned / DOCX / vendor-quirky |
| `tests/extraction_v3/fixtures/invoices/INV-001.expected.json` | Ground-truth field-by-field expected output |
| `tests/extraction_v3/test_parsed_document_schema.py` | Pydantic validation tests for L1 boundary |
| `tests/extraction_v3/test_yaml_schema_loader.py` | YAML loader rejects drift |
| `tests/extraction_v3/test_scanned_classifier.py` | Native vs scanned detection |
| `tests/extraction_v3/test_docling_backend.py` | Docling produces `ParsedDocument` |
| `tests/extraction_v3/test_paddleocr_backend.py` | PaddleOCR produces `ParsedDocument` |
| `tests/extraction_v3/test_donut_backend.py` | Donut produces `ParsedDocument` |
| `tests/extraction_v3/test_layoutlmv3_extractor.py` | LayoutLMv3 produces candidates with bbox + evidence |
| `tests/extraction_v3/test_table_transformer.py` | Table Transformer produces line-item candidates |
| `tests/extraction_v3/test_sbert_anchor.py` | sBERT picks correct field for ambiguous label |
| `tests/extraction_v3/test_spacy_ner.py` | NER demotes garbage supplier candidate |
| `tests/extraction_v3/test_qa_roberta.py` | QA gap-fills missing field |
| `tests/extraction_v3/test_vendor_template_extractor.py` | Template hint emits candidate when fingerprint matches |
| `tests/extraction_v3/test_type_binder.py` | Money / IsoDate / Address coercion |
| `tests/extraction_v3/test_scale_mismatch_invariant.py` | New invariant catches 10× decimal misread |
| `tests/extraction_v3/test_judge_tiebreaker.py` | Tiebreaker post-validation |
| `tests/extraction_v3/test_judge_grounded_last_resort.py` | Grounded judge rejects non-substring values |
| `tests/extraction_v3/test_judge_schema_coherence.py` | Coherence judge demotes incoherent records |
| `tests/extraction_v3/test_persistence_provenance.py` | Provenance row written for every committed field |
| `tests/extraction_v3/test_dispatch_feature_flag.py` | Per-category flag routes correctly |
| `tests/extraction_v3/test_pipeline_e2e.py` | End-to-end: PDF → DB row + provenance |

### Modified files

| File | Change |
|------|--------|
| `src/services/process_monitor_watcher.py` | Replace direct call to `agent_nick_orchestrator` with `extraction_v3.dispatch.dispatch(...)` |
| `src/api/main.py` | Lifespan startup: load YAML schemas + warm L1/L2 models on GPU |
| `pyproject.toml` (or `requirements.txt`) | Add `docling`, `paddleocr`, `paddlepaddle-gpu`, `donut-python`, `sentence-transformers`, `spacy[cuda]`, `transformers` (already present, pin version) |
| `procwise.service` or `.env` | Add `EXTRACTION_PIPELINE_INVOICE`, `_PURCHASE_ORDER`, `_QUOTE`, `_CONTRACT` env vars defaulting to `agentnick` |

### Untouched (kept-not-deleted per C7)

`src/services/agent_nick_orchestrator.py`, `src/services/direct_extraction_service.py`, `src/services/intelligent_extractor.py`, `src/services/structural_extractor/**`, all `extraction_v2` files except those explicitly wrapped (`template_store_pg.py`, `parsers/`, `invariants.py`).

---

## Task 1: DB migration — `proc.bp_extraction_provenance_v3`

**Files:**
- Create: `scripts/migrations/2026-05-08-extraction-provenance-v3.sql`
- Test: `tests/extraction_v3/test_provenance_migration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/extraction_v3/test_provenance_migration.py
import psycopg2
from src.config import get_db_connection

def test_provenance_table_exists():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema='proc' AND table_name='bp_extraction_provenance'
                ORDER BY ordinal_position
            """)
            cols = {name: dtype for name, dtype in cur.fetchall()}
    expected = {
        "provenance_id": "bigint",
        "doc_type": "text",
        "doc_pk": "text",
        "field_path": "text",
        "value": "text",
        "page": "integer",
        "bbox_x0": "real",
        "bbox_y0": "real",
        "bbox_x1": "real",
        "bbox_y1": "real",
        "evidence_text": "text",
        "model": "text",
        "model_confidence": "real",
        "judge_actions": "jsonb",
        "final_confidence": "real",
        "extracted_at": "timestamp with time zone",
        "pipeline_version": "text",
    }
    for col, dtype in expected.items():
        assert cols.get(col) == dtype, f"missing or wrong type: {col}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/extraction_v3/test_provenance_migration.py -v`
Expected: FAIL — table does not exist.

- [ ] **Step 3: Write the migration**

```sql
-- scripts/migrations/2026-05-08-extraction-provenance-v3.sql
CREATE TABLE IF NOT EXISTS proc.bp_extraction_provenance_v3 (
    provenance_id    BIGSERIAL PRIMARY KEY,
    doc_type         TEXT NOT NULL,
    doc_pk           TEXT NOT NULL,
    field_path       TEXT NOT NULL,
    value            TEXT NOT NULL,
    page             INT NOT NULL,
    bbox_x0          REAL NOT NULL,
    bbox_y0          REAL NOT NULL,
    bbox_x1          REAL NOT NULL,
    bbox_y1          REAL NOT NULL,
    evidence_text    TEXT NOT NULL,
    model            TEXT NOT NULL,
    model_confidence REAL NOT NULL,
    judge_actions    JSONB,
    final_confidence REAL NOT NULL,
    extracted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pipeline_version TEXT NOT NULL,
    UNIQUE (doc_type, doc_pk, field_path, extracted_at)
);
CREATE INDEX IF NOT EXISTS idx_provenance_doc
    ON proc.bp_extraction_provenance_v3 (doc_type, doc_pk);
```

- [ ] **Step 4: Apply the migration**

Run: `psql $DATABASE_URL -f scripts/migrations/2026-05-08-extraction-provenance-v3.sql`
Expected: `CREATE TABLE`, `CREATE INDEX`.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/extraction_v3/test_provenance_migration.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/migrations/2026-05-08-extraction-provenance-v3.sql tests/extraction_v3/test_provenance_migration.py
git commit -m "feat(extraction_v3): add bp_extraction_provenance table"
```

---

## Task 2: Pydantic boundary schemas

**Files:**
- Create: `src/services/extraction_v3/__init__.py` (empty)
- Create: `src/services/extraction_v3/schemas/__init__.py`
- Create: `src/services/extraction_v3/schemas/parsed_document.py`
- Create: `src/services/extraction_v3/schemas/candidate.py`
- Create: `src/services/extraction_v3/schemas/result.py`
- Test: `tests/extraction_v3/test_parsed_document_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/extraction_v3/test_parsed_document_schema.py
import pytest
from pydantic import ValidationError
from src.services.extraction_v3.schemas.parsed_document import (
    ParsedDocument, Page, Region, Token, Cell, Table
)

def test_parsed_document_minimal():
    doc = ParsedDocument(
        source_path="/tmp/x.pdf",
        file_format="pdf-native",
        pages=[Page(index=0, width=612, height=792, rotation=0,
                    regions=[], tables=[], tokens=[])],
        full_text="",
        parser_backend="docling",
        parser_confidence=1.0,
    )
    assert doc.pages[0].index == 0

def test_parsed_document_rejects_invalid_rotation():
    with pytest.raises(ValidationError):
        Page(index=0, width=1, height=1, rotation=45,
             regions=[], tables=[], tokens=[])

def test_token_bbox_is_4_floats():
    t = Token(text="x", page=0, bbox=(0.0, 0.0, 1.0, 1.0), font_size=12.0, is_bold=False)
    assert t.bbox == (0.0, 0.0, 1.0, 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/extraction_v3/test_parsed_document_schema.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the schemas**

```python
# src/services/extraction_v3/schemas/parsed_document.py
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
```

```python
# src/services/extraction_v3/schemas/candidate.py
from typing import Literal
from pydantic import BaseModel, Field
from .parsed_document import BBox

ExtractorName = Literal[
    "layoutlmv3", "table_transformer", "sbert_anchor",
    "spacy_ner", "qa_roberta", "vendor_template"
]

class Candidate(BaseModel):
    field: str
    value: str
    page: int
    bbox: BBox
    evidence_text: str
    model: ExtractorName
    confidence: float = Field(ge=0.0, le=1.0)
```

```python
# src/services/extraction_v3/schemas/result.py
from typing import Literal
from pydantic import BaseModel
from .candidate import Candidate

ResidualReason = Literal[
    "unsupported_layout",
    "required_field_missing_no_grounding",
    "invariant_critical_failed",
    "judge_incoherent",
    "bind_error_no_resolution",
]

JudgeAction = Literal["tiebreaker", "grounded_last_resort", "schema_coherence"]

class CommittedField(BaseModel):
    field_path: str
    value: str
    page: int
    bbox: tuple[float, float, float, float]
    evidence_text: str
    model: str
    model_confidence: float
    judge_actions: list[JudgeAction] = []
    final_confidence: float

class ResidualField(BaseModel):
    field_path: str
    reason: ResidualReason
    candidates: list[Candidate] = []

class ExtractionResult(BaseModel):
    doc_type: str
    doc_pk: str | None
    committed: list[CommittedField]
    residuals: list[ResidualField]
    judge_calls: int
    pipeline_version: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/extraction_v3/test_parsed_document_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/extraction_v3/__init__.py src/services/extraction_v3/schemas/ tests/extraction_v3/test_parsed_document_schema.py
git commit -m "feat(extraction_v3): pydantic boundary schemas (ParsedDocument, Candidate, ExtractionResult)"
```

---

## Task 3: YAML schema loader + startup consistency check

**Files:**
- Create: `src/services/extraction_v3/yaml_schema/__init__.py`
- Create: `src/services/extraction_v3/yaml_schema/loader.py`
- Create: `src/services/extraction_v3/yaml_schema/registry.py`
- Create: `extraction_schemas/invoice.yaml`
- Create: `extraction_schemas/purchase_order.yaml` (stub)
- Create: `extraction_schemas/quote.yaml` (stub)
- Create: `extraction_schemas/contract.yaml` (stub)
- Test: `tests/extraction_v3/test_yaml_schema_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/extraction_v3/test_yaml_schema_loader.py
import pytest
from pathlib import Path
from src.services.extraction_v3.yaml_schema.loader import (
    load_doc_schema, load_all_schemas, SchemaDriftError
)

FIXTURES = Path(__file__).parent / "yaml_fixtures"

def test_load_invoice_schema():
    s = load_doc_schema("invoice")
    assert s.doc_type == "invoice"
    assert s.db_table == "proc.bp_invoice"
    inv_id = next(f for f in s.fields if f.name == "invoice_id")
    assert inv_id.required is True
    assert "Invoice Number" in inv_id.canonical_labels

def test_drift_fails_loud(tmp_path, monkeypatch):
    bad_yaml = tmp_path / "broken.yaml"
    bad_yaml.write_text("""
doc_type: broken
db_table: proc.does_not_exist
fields:
  - {name: x, type: string, required: true, db_column: nope, canonical_labels: ["x"], extractors: [layoutlmv3]}
line_items: {primary_extractor: table_transformer, fields: []}
""")
    with pytest.raises(SchemaDriftError) as exc:
        load_doc_schema_path(bad_yaml)
    assert "does_not_exist" in str(exc.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/extraction_v3/test_yaml_schema_loader.py -v`
Expected: FAIL — loader missing.

- [ ] **Step 3: Implement `loader.py` + minimal Pydantic models**

```python
# src/services/extraction_v3/yaml_schema/loader.py
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel
from src.config import get_db_connection

class SchemaDriftError(RuntimeError): ...

class JudgeRules(BaseModel):
    tiebreaker: bool = True
    grounded_last_resort: bool = True
    ner_type_check: Literal["none", "ORG", "PERSON", "GPE", "LOC"] = "none"

class FieldSpec(BaseModel):
    name: str
    type: Literal["string", "iso_date", "money", "decimal", "address", "postcode"]
    required: bool
    db_column: str
    canonical_labels: list[str]
    extractors: list[str]
    judge: JudgeRules = JudgeRules()
    invariants: list[str] = []

class LineItemsSpec(BaseModel):
    primary_extractor: Literal["table_transformer", "layoutlmv3"]
    fallback_extractor: str | None = None
    fields: list[FieldSpec]
    invariants: list[str] = []

class DocSchema(BaseModel):
    doc_type: str
    db_table: str
    db_lines_table: str | None = None
    fields: list[FieldSpec]
    line_items: LineItemsSpec | None = None
    document_invariants: list[str] = []

SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "extraction_schemas"

def load_doc_schema_path(path: Path) -> DocSchema:
    raw = yaml.safe_load(path.read_text())
    schema = DocSchema(**raw)
    _verify_db_consistency(schema)
    return schema

def load_doc_schema(doc_type: str) -> DocSchema:
    return load_doc_schema_path(SCHEMAS_DIR / f"{doc_type}.yaml")

def load_all_schemas() -> dict[str, DocSchema]:
    return {p.stem: load_doc_schema_path(p) for p in SCHEMAS_DIR.glob("*.yaml")}

def _verify_db_consistency(schema: DocSchema) -> None:
    table_schema, table_name = schema.db_table.split(".", 1)
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema=%s AND table_name=%s
        """, (table_schema, table_name))
        cols = {r[0] for r in cur.fetchall()}
    if not cols:
        raise SchemaDriftError(f"DB table {schema.db_table} does not exist")
    missing = [f.db_column for f in schema.fields if f.db_column not in cols]
    if missing:
        raise SchemaDriftError(f"{schema.db_table} missing columns: {missing}")
```

- [ ] **Step 4: Implement `registry.py`**

```python
# src/services/extraction_v3/yaml_schema/registry.py
from typing import Callable
_extractors: dict[str, Callable] = {}
_invariants: dict[str, Callable] = {}

def register_extractor(name: str):
    def deco(cls):
        _extractors[name] = cls
        return cls
    return deco

def register_invariant(name: str):
    def deco(fn):
        _invariants[name] = fn
        return fn
    return deco

def get_extractor(name: str): return _extractors[name]
def get_invariant(name: str): return _invariants[name]
def known_extractors() -> set[str]: return set(_extractors)
def known_invariants() -> set[str]: return set(_invariants)
```

- [ ] **Step 5: Author `extraction_schemas/invoice.yaml`** (full schema; see spec §6.1 for the canonical example, copy in entirety, ensuring every `db_column` matches `proc.bp_invoice` per `docs/procurement_table_reference.md`)

- [ ] **Step 6: Author the three stubs** (each 5 lines: `doc_type`, `db_table`, empty `fields: []`, `line_items: null`. Plan 3 fills these.)

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/extraction_v3/test_yaml_schema_loader.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/services/extraction_v3/yaml_schema/ extraction_schemas/ tests/extraction_v3/test_yaml_schema_loader.py
git commit -m "feat(extraction_v3): YAML schema loader + invoice schema + 3 stubs"
```

---

## Task 4: Universal Parser router + scanned-PDF classifier

**Files:**
- Create: `src/services/extraction_v3/parsers/__init__.py`
- Create: `src/services/extraction_v3/parsers/router.py`
- Create: `src/services/extraction_v3/parsers/scanned_classifier.py`
- Test: `tests/extraction_v3/test_scanned_classifier.py`

- [ ] **Step 1: Failing test for classifier**

```python
# tests/extraction_v3/test_scanned_classifier.py
from pathlib import Path
from src.services.extraction_v3.parsers.scanned_classifier import is_scanned_pdf

FX = Path(__file__).parent / "fixtures/invoices"

def test_native_pdf_not_scanned():
    assert is_scanned_pdf(FX / "INV-001-clean.pdf") is False

def test_scanned_pdf_detected():
    assert is_scanned_pdf(FX / "INV-005-scanned.pdf") is True
```

- [ ] **Step 2: Run test, expect FAIL**

- [ ] **Step 3: Implement classifier**

```python
# src/services/extraction_v3/parsers/scanned_classifier.py
from pathlib import Path
import pdfplumber

def is_scanned_pdf(path: Path | str) -> bool:
    with pdfplumber.open(str(path)) as pdf:
        if not pdf.pages:
            return True
        chars = sum(len((p.extract_text() or "")) for p in pdf.pages)
        avg = chars / len(pdf.pages)
        return avg <= 5
```

- [ ] **Step 4: Implement router**

```python
# src/services/extraction_v3/parsers/router.py
from pathlib import Path
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument
from .scanned_classifier import is_scanned_pdf

def parse(path: Path | str) -> ParsedDocument:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        if is_scanned_pdf(p):
            from .paddleocr_backend import parse_with_paddleocr
            return parse_with_paddleocr(p, file_format="pdf-scanned")
        from .docling_backend import parse_with_docling
        return parse_with_docling(p, file_format="pdf-native")
    if suffix == ".docx":
        from .docling_backend import parse_with_docling
        return parse_with_docling(p, file_format="docx")
    if suffix in (".png", ".jpg", ".jpeg"):
        from .paddleocr_backend import parse_with_paddleocr
        return parse_with_paddleocr(p, file_format="image")
    raise ValueError(f"unsupported file format: {suffix}")
```

- [ ] **Step 5: Acquire 2 fixture PDFs (one native, one scanned)** — copy from existing `documents/invoice/` into `tests/extraction_v3/fixtures/invoices/`. Pick a clean TECHWORLD invoice and a scanned one (search for low-text PDFs in current dataset).

- [ ] **Step 6: Run test to verify PASS**

- [ ] **Step 7: Commit**

```bash
git add src/services/extraction_v3/parsers/ tests/extraction_v3/test_scanned_classifier.py tests/extraction_v3/fixtures/invoices/INV-001-clean.pdf tests/extraction_v3/fixtures/invoices/INV-005-scanned.pdf
git commit -m "feat(extraction_v3): file-format router + scanned-PDF classifier"
```

---

## Task 5: Docling backend (native PDF + DOCX)

**Files:**
- Create: `src/services/extraction_v3/parsers/docling_backend.py`
- Test: `tests/extraction_v3/test_docling_backend.py`

- [ ] **Step 1: Add `docling` dependency** to `pyproject.toml` / `requirements.txt`. Run `pip install docling`. Confirm `from docling.document_converter import DocumentConverter` works.

- [ ] **Step 2: Failing test**

```python
# tests/extraction_v3/test_docling_backend.py
from pathlib import Path
from src.services.extraction_v3.parsers.docling_backend import parse_with_docling

FX = Path(__file__).parent / "fixtures/invoices"

def test_docling_extracts_pages_and_tokens():
    doc = parse_with_docling(FX / "INV-001-clean.pdf", file_format="pdf-native")
    assert doc.parser_backend == "docling"
    assert len(doc.pages) >= 1
    assert len(doc.pages[0].tokens) > 0
    assert doc.full_text  # non-empty

def test_docling_handles_docx():
    doc = parse_with_docling(FX / "INV-006-docx.docx", file_format="docx")
    assert doc.file_format == "docx"
    assert doc.full_text
```

- [ ] **Step 3: Run test, expect FAIL**

- [ ] **Step 4: Implement adapter** that converts Docling's output tree into `ParsedDocument`. Map Docling's `DoclingDocument.pages → Page`, `texts → Token` (with bbox via Docling's layout info), `tables → Table` (cells with row/col indices), `parser_confidence` from Docling's reported confidence (or 1.0 if not surfaced). Keep `full_text` as Docling's reading-order text.

- [ ] **Step 5: Run test to PASS**

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(extraction_v3): Docling parser backend (native PDF + DOCX)"
```

---

## Task 6: PaddleOCR backend (scanned PDF + images)

**Files:**
- Create: `src/services/extraction_v3/parsers/paddleocr_backend.py`
- Test: `tests/extraction_v3/test_paddleocr_backend.py`

- [ ] **Step 1: Install** `paddlepaddle-gpu` (matching CUDA version) + `paddleocr`. Verify `PPStructure(layout=True, table=True, ocr=True, lang="en", use_gpu=True)` initializes. **GPU dependency** — if installation fails, document blocker and surface to user; do not fall back to CPU silently (would violate C2's local-on-GPU requirement).

- [ ] **Step 2: Failing test**

```python
# tests/extraction_v3/test_paddleocr_backend.py
from pathlib import Path
from src.services.extraction_v3.parsers.paddleocr_backend import parse_with_paddleocr

FX = Path(__file__).parent / "fixtures/invoices"

def test_paddleocr_extracts_scanned_pdf():
    doc = parse_with_paddleocr(FX / "INV-005-scanned.pdf", file_format="pdf-scanned")
    assert doc.parser_backend == "paddleocr"
    assert doc.full_text
    assert len(doc.pages) >= 1
```

- [ ] **Step 3: Run test, expect FAIL**

- [ ] **Step 4: Implement adapter** — for each PDF page rasterize at 300 DPI (use `pdf2image`); pass image to PP-Structure; map its `layout_dets` → `Region`, `text_recognition` → `Token` with bbox, `table_recognition` → `Table` with cells. For `.png/.jpg`, skip rasterization. Set `parser_confidence` to the mean of detected box confidences.

- [ ] **Step 5: Run test to PASS** (may need to add a low-text-content fixture PDF)

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(extraction_v3): PaddleOCR PP-Structure backend (scanned PDF + images)"
```

---

## Task 7: Donut backend (low-confidence fallback)

**Files:**
- Create: `src/services/extraction_v3/parsers/donut_backend.py`
- Test: `tests/extraction_v3/test_donut_backend.py`

- [ ] **Step 1: Install** `donut-python` and `transformers` Donut model `naver-clova-ix/donut-base`. Confirm GPU loading.

- [ ] **Step 2: Failing test** — Donut on a hard-layout fixture should produce `parser_confidence > 0.5`.

- [ ] **Step 3: Implement adapter** — Donut returns JSON-structured output; map its detected regions to `ParsedDocument` regions. Note: Donut's "confidence" is the decoder's average token logit-prob; treat anything ≥ 0.5 as usable.

- [ ] **Step 4: Wire fallback rule into `router.py`** — if PaddleOCR returns `parser_confidence < 0.6`, retry with Donut. Don't add this until both backends exist.

- [ ] **Step 5: Run test to PASS**

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(extraction_v3): Donut parser backend (low-conf fallback) + router fallback rule"
```

---

## Task 8: Extractor base class + registry wiring

**Files:**
- Create: `src/services/extraction_v3/extractors/__init__.py`
- Create: `src/services/extraction_v3/extractors/base.py`
- Test: `tests/extraction_v3/test_extractor_registry.py`

- [ ] **Step 1: Failing test**

```python
# tests/extraction_v3/test_extractor_registry.py
from src.services.extraction_v3.extractors.base import Extractor
from src.services.extraction_v3.yaml_schema.registry import get_extractor, known_extractors

def test_layoutlmv3_registered_after_import():
    import src.services.extraction_v3.extractors.layoutlmv3  # noqa
    assert "layoutlmv3" in known_extractors()
```

- [ ] **Step 2: Implement `base.py`** — abstract `Extractor` class with `produce_candidates(parsed, schema) -> list[Candidate]`, `__init_subclass__` enforcing registration via `register_extractor` decorator.

- [ ] **Step 3: Run test, expect FAIL** (until extractors exist) — comment-out import temporarily, run, expect PASS for registry mechanism alone, then uncomment for the full check after Task 9.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(extraction_v3): extractor base class + registration mechanism"
```

---

## Task 9: LayoutLMv3 extractor (header fields)

**Files:**
- Create: `src/services/extraction_v3/extractors/layoutlmv3.py`
- Test: `tests/extraction_v3/test_layoutlmv3_extractor.py`

- [ ] **Step 1: Install** `transformers`, `microsoft/layoutlmv3-base` (auto-downloads on first use, ~500 MB).

- [ ] **Step 2: Failing test**

```python
# tests/extraction_v3/test_layoutlmv3_extractor.py
from pathlib import Path
from src.services.extraction_v3.parsers.docling_backend import parse_with_docling
from src.services.extraction_v3.extractors.layoutlmv3 import LayoutLMv3Extractor
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema

FX = Path(__file__).parent / "fixtures/invoices"

def test_layoutlmv3_emits_candidates():
    parsed = parse_with_docling(FX / "INV-001-clean.pdf", file_format="pdf-native")
    schema = load_doc_schema("invoice")
    extractor = LayoutLMv3Extractor()
    candidates = extractor.produce_candidates(parsed, schema)
    fields = {c.field for c in candidates}
    # Even without fine-tuning, base LayoutLMv3 should label SOMETHING for a clean invoice
    assert candidates, "no candidates produced"
    # Each candidate carries cited evidence
    assert all(c.evidence_text in parsed.full_text for c in candidates), \
        "candidate evidence_text missing from source"
```

- [ ] **Step 3: Run test, expect FAIL**

- [ ] **Step 4: Implement extractor**

```python
# src/services/extraction_v3/extractors/layoutlmv3.py (sketch)
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from .base import Extractor
from src.services.extraction_v3.yaml_schema.registry import register_extractor
from src.services.extraction_v3.schemas.candidate import Candidate

LABEL_MAP_INVOICE = {  # B-/I- per BIO scheme; placeholder labels for pre-trained base
    "INVOICE_ID": ["B-INVOICE_ID", "I-INVOICE_ID"],
    "SUPPLIER_NAME": ["B-SUPPLIER", "I-SUPPLIER"],
    "INVOICE_DATE": ["B-DATE", "I-DATE"],
    "INVOICE_AMOUNT": ["B-AMOUNT", "I-AMOUNT"],
    # ...
}

@register_extractor("layoutlmv3")
class LayoutLMv3Extractor(Extractor):
    def __init__(self):
        self.proc = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base").to("cuda").eval()

    def produce_candidates(self, parsed, schema):
        # rasterize each page; build words+boxes from tokens; run model; aggregate B-/I- spans
        ...
        return candidates
```

(Implementation note: pre-trained `layoutlmv3-base` has no procurement-specific labels. For Plan 1 we use it as a token-position encoder + a simple post-processor that maps decoded entity spans to schema fields by `canonical_labels` proximity. Real accuracy comes after Plan 2's fine-tune. The test threshold is "produces some candidates with valid evidence," not "produces correct values.")

- [ ] **Step 5: Run test to PASS**

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(extraction_v3): LayoutLMv3 extractor (pre-trained base, fine-tune deferred to Plan 2)"
```

---

## Task 10: Table Transformer extractor (line items)

**Files:**
- Create: `src/services/extraction_v3/extractors/table_transformer.py`
- Test: `tests/extraction_v3/test_table_transformer.py`

- [ ] **Step 1: Install** `microsoft/table-transformer-structure-recognition-v1.1-all` via transformers.

- [ ] **Step 2: Failing test** — given `INV-001-clean.pdf` with a known 3-line-item table, extractor produces 3 row-grouped candidates `(line[0..2].description / .quantity / .amount)` whose `evidence_text` matches source.

- [ ] **Step 3: Implement extractor** — rasterize page → Table Transformer → cell-level outputs → group cells into rows → map columns to `description/quantity/unit_price/amount` by header-row label matching.

- [ ] **Step 4: Run test to PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(extraction_v3): Table Transformer line-item extractor"
```

---

## Task 11: sBERT semantic anchor extractor

**Files:**
- Create: `src/services/extraction_v3/extractors/sbert_anchor.py`
- Test: `tests/extraction_v3/test_sbert_anchor.py`

- [ ] **Step 1: Install** `sentence-transformers`.

- [ ] **Step 2: Failing test** — given a token "Sold By:" near a vendor name, sBERT picks `supplier_name` as the closest field via cosine similarity to the YAML's `canonical_labels`.

- [ ] **Step 3: Implement extractor**

```python
# src/services/extraction_v3/extractors/sbert_anchor.py
from sentence_transformers import SentenceTransformer, util
from .base import Extractor
from src.services.extraction_v3.yaml_schema.registry import register_extractor
from src.services.extraction_v3.schemas.candidate import Candidate

@register_extractor("sbert_anchor")
class SbertAnchorExtractor(Extractor):
    def __init__(self):
        self.model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

    def produce_candidates(self, parsed, schema):
        # find label-shaped tokens (ends in ':' or all-caps); for each,
        # compute cosine to each field's canonical_labels embeddings;
        # if max sim > 0.6, emit a candidate for that field, value = next non-label token
        ...
```

- [ ] **Step 4: Run test to PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(extraction_v3): sBERT semantic anchor extractor"
```

---

## Task 12: spaCy NER validator

**Files:**
- Create: `src/services/extraction_v3/extractors/spacy_ner.py`
- Test: `tests/extraction_v3/test_spacy_ner.py`

- [ ] **Step 1: Install** `spacy`, `python -m spacy download en_core_web_trf`.

- [ ] **Step 2: Failing test** — given a candidate `supplier_name="INVOICE NUMBER: 4759275"` (from I-18 regression), NER validator demotes its confidence to ≤ 0.3 because zero `ORG` entities are present. Given `supplier_name="Acme Industries Ltd"`, confidence stays ≥ 0.7.

- [ ] **Step 3: Implement validator** — runs over candidates from other extractors (it's a *post-processor*, not a primary candidate generator). Reads YAML's `judge.ner_type_check` per field; downgrades confidence if the candidate's value contains zero entities of the required type.

- [ ] **Step 4: Run test to PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(extraction_v3): spaCy NER candidate validator"
```

---

## Task 13: Extractive QA gap-filler

**Files:**
- Create: `src/services/extraction_v3/extractors/qa_roberta.py`
- Test: `tests/extraction_v3/test_qa_roberta.py`

- [ ] **Step 1: Install** `deepset/roberta-base-squad2` via transformers pipeline.

- [ ] **Step 2: Failing test** — given parsed text containing "Invoice No: ABC-123" and a candidate-empty `invoice_id` field, QA returns `value="ABC-123"` with `evidence_text="ABC-123"` substring of source.

- [ ] **Step 3: Implement extractor** — for each required field with NO existing candidate from L2 components 9-11, run QA with question = first canonical_label rephrased ("What is the {label}?"). Reject answers below 0.4 confidence or that fail the substring check (defensive — QA shouldn't fabricate, but verify).

- [ ] **Step 4: Run test to PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(extraction_v3): extractive-QA gap-filler"
```

---

## Task 14: Vendor template extractor (wraps existing v2 store)

**Files:**
- Create: `src/services/extraction_v3/extractors/vendor_template.py`
- Test: `tests/extraction_v3/test_vendor_template_extractor.py`

- [ ] **Step 1: Failing test** — when `INV-001-clean.pdf`'s fingerprint matches a stored template (insert one in test setup using `extraction_v2/template_store_pg.py`), extractor emits candidates with `model="vendor_template"` and confidence 0.9.

- [ ] **Step 2: Implement extractor** — call `extraction_v2.fingerprint.compute_fingerprint(parsed)`, query `template_store_pg`, emit one candidate per stored field hint.

- [ ] **Step 3: Run test to PASS**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(extraction_v3): vendor-template extractor (wraps existing v2 store)"
```

---

## Task 15: Type binder (uses existing v2 typed parsers)

**Files:**
- Create: `src/services/extraction_v3/binding/__init__.py`
- Create: `src/services/extraction_v3/binding/type_binder.py`
- Test: `tests/extraction_v3/test_type_binder.py`

- [ ] **Step 1: Failing test**

```python
# tests/extraction_v3/test_type_binder.py
from src.services.extraction_v3.binding.type_binder import bind_typed
from src.services.extraction_v3.schemas.candidate import Candidate

def make_cand(field, value, conf=0.9):
    return Candidate(field=field, value=value, page=0, bbox=(0,0,1,1),
                     evidence_text=value, model="layoutlmv3", confidence=conf)

def test_bind_money():
    out = bind_typed(make_cand("invoice_amount", "£7,290.00"), field_type="money")
    assert out.coerced_value == 7290.00

def test_bind_iso_date():
    out = bind_typed(make_cand("invoice_date", "20th October, 2025"), field_type="iso_date")
    assert out.coerced_value == "2025-10-20"

def test_bind_failure_does_not_silently_null():
    out = bind_typed(make_cand("invoice_amount", "totally not a number"), field_type="money")
    assert out.bind_error is True
```

- [ ] **Step 2-4: Implement, run, PASS** — wraps the existing `extraction_v2/parsers/{money,iso_date,address,postcode}.py` modules; returns a small dataclass `BoundCandidate(candidate, coerced_value, bind_error)`.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(extraction_v3): type-binder using existing v2 typed parsers"
```

---

## Task 16: New `ScaleMismatch` invariant + invariants runner

**Files:**
- Create: `src/services/extraction_v3/binding/scale_mismatch.py`
- Create: `src/services/extraction_v3/binding/invariants_runner.py`
- Test: `tests/extraction_v3/test_scale_mismatch_invariant.py`

- [ ] **Step 1: Failing test** — for the TECHWORLD I-39 case (`line_sum=6750.00`, `invoice_amount=675.00`), `scale_mismatch` returns severity `CRITICAL`.

- [ ] **Step 2: Implement invariant**

```python
# src/services/extraction_v3/binding/scale_mismatch.py
from src.services.extraction_v3.yaml_schema.registry import register_invariant

@register_invariant("scale_mismatch")
def scale_mismatch(record: dict) -> tuple[str, str | None]:
    inv = record.get("invoice_amount")
    line_sum = sum((li.get("amount") or 0) for li in record.get("line_items", []))
    if not inv or not line_sum: return ("ok", None)
    ratio = max(line_sum, inv) / max(min(line_sum, inv), 0.01)
    if ratio > 9:
        return ("CRITICAL", f"line_sum/invoice_amount ratio={ratio:.1f}× — likely decimal misread")
    return ("ok", None)
```

- [ ] **Step 3: Implement runner** — loads names from YAML's per-field `invariants` + top-level `document_invariants`; runs in declared order; returns list of `(name, severity, message)`. Wires existing `extraction_v2/invariants.py` invariants into the registry.

- [ ] **Step 4: Run tests to PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(extraction_v3): ScaleMismatch invariant + invariants runner (wraps v2)"
```

---

## Task 17: Judge contracts (frozen JSON schemas)

**Files:**
- Create: `src/services/extraction_v3/judge/__init__.py`
- Create: `src/services/extraction_v3/judge/contracts.py`
- Test: `tests/extraction_v3/test_judge_contracts.py`

- [ ] **Step 1: Failing test** — `validate_tiebreaker_input(...)` and `validate_tiebreaker_output(...)` accept conformant payloads, reject malformed ones; same for grounded + coherence.

- [ ] **Step 2: Implement** — three Pydantic models per contract type (input + output) per spec §6.4.1–§6.4.3. Use Pydantic strict mode.

- [ ] **Step 3: Run tests to PASS**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(extraction_v3): judge contracts (tiebreaker, grounded, coherence)"
```

---

## Task 18: Tiebreaker judge

**Files:**
- Create: `src/services/extraction_v3/judge/tiebreaker.py`
- Test: `tests/extraction_v3/test_judge_tiebreaker.py`

- [ ] **Step 1: Failing test** — given two candidates `[Aquarius Marketing Ltd / 0.81, AuariusMarketing / 0.62]`, judge picks index 0 OR returns null. Post-validation: returned index must be in `[0, len(candidates))` or null.

- [ ] **Step 2: Implement** — call Ollama (`BeyondProcwise/AgentNick:judge` if available, else `BeyondProcwise/AgentNick:latest` as fallback for Plan 1) with the tiebreaker prompt, parse JSON output, post-validate, return `Candidate | None`.

- [ ] **Step 3: Run tests to PASS** (using a frozen Ollama response fixture for determinism)

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(extraction_v3): tiebreaker judge (call + post-validate)"
```

---

## Task 19: Grounded last-resort judge (anti-hallucination guarantee)

**Files:**
- Create: `src/services/extraction_v3/judge/grounded_last_resort.py`
- Test: `tests/extraction_v3/test_judge_grounded_last_resort.py`

- [ ] **Step 1: Failing test (CRITICAL)**

```python
# tests/extraction_v3/test_judge_grounded_last_resort.py
def test_rejects_value_not_in_doc_text(monkeypatch):
    """The structural anti-hallucination guarantee — if value not a substring, reject."""
    def fake_ollama(_prompt):
        return '{"value": "Eleanor Price", "evidence_text": "Eleanor Price", "rationale": "guess"}'
    monkeypatch.setattr("...ollama_client.generate", fake_ollama)
    result = grounded_last_resort(
        field_spec=...,  # invoice_id required string
        doc_full_text="TECHWORLD INV-005-41 ...",  # no Eleanor Price anywhere
    )
    assert result is None  # rejected because not a substring

def test_accepts_genuine_substring(monkeypatch):
    def fake_ollama(_):
        return '{"value": "INV-005-41", "evidence_text": "INV-005-41", "rationale": "found"}'
    monkeypatch.setattr("...ollama_client.generate", fake_ollama)
    result = grounded_last_resort(
        field_spec=...,
        doc_full_text="TECHWORLD INV-005-41 for PO405867",
    )
    assert result.value == "INV-005-41"
```

- [ ] **Step 2: Implement** — call Ollama with grounded prompt, parse JSON, **enforce `value == evidence_text` AND `evidence_text in doc_full_text`**; on failure return None (NOT a fallback value). This is the structural anti-hallucination contract from spec §6.4.2.

- [ ] **Step 3: Run tests to PASS**

- [ ] **Step 4: Add a property test** — for 100 random LLM-fabricated outputs, ensure none get committed.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(extraction_v3): grounded-last-resort judge with substring guarantee"
```

---

## Task 20: Schema-coherence judge

**Files:**
- Create: `src/services/extraction_v3/judge/schema_coherence.py`
- Test: `tests/extraction_v3/test_judge_schema_coherence.py`

- [ ] **Step 1: Failing test** — given a TECHWORLD invoice record with `requested_by="Eleanor Price Creative Studio"` (the I-38 cross-doc-leakage case), coherence judge returns `incoherent` with an issue naming `requested_by`.

- [ ] **Step 2: Implement** — call Ollama with coherence prompt, parse verdict, return `(verdict, issues)`. Verdict is advisory: orchestrator demotes record confidence by 0.20 if `incoherent`, doesn't mutate the record itself.

- [ ] **Step 3: Run tests to PASS**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(extraction_v3): schema-coherence judge"
```

---

## Task 21: Judge orchestrator + cost ceiling

**Files:**
- Create: `src/services/extraction_v3/judge/orchestrator.py`
- Test: `tests/extraction_v3/test_judge_orchestrator.py`

- [ ] **Step 1: Failing test** — for a doc with 2 disagreement fields and 1 missing required field, orchestrator makes ≤ 4 judge calls total (per spec §6.4.4 ceiling: `2 + 1 + 1 = 4`).

- [ ] **Step 2: Implement** — receives a `dict[field, list[Candidate]]`, decides per-field which judge invocations to fire, enforces hard cap, returns `(committed_per_field, residuals, judge_actions_per_field)`.

- [ ] **Step 3: Run tests to PASS**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(extraction_v3): judge orchestrator with hard cost ceiling"
```

---

## Task 22: Pipeline V3 orchestrator (wires L1 + L2 + L3)

**Files:**
- Create: `src/services/extraction_v3/pipeline.py`
- Test: `tests/extraction_v3/test_pipeline_e2e.py` (smoke; full E2E in Task 26)

- [ ] **Step 1: Failing test** — `PipelineV3.run("/path/to/INV-001-clean.pdf", "invoice")` returns an `ExtractionResult` with `committed` non-empty AND `judge_calls <= 4`.

- [ ] **Step 2: Implement** — sequence: parse (L1) → run all extractors named in YAML in parallel via `concurrent.futures` → bind types → run invariants → judge orchestrator → schema-coherence final pass → return `ExtractionResult`.

- [ ] **Step 3: Run test to PASS**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(extraction_v3): pipeline V3 orchestrator wiring L1+L2+L3"
```

---

## Task 23: Persistence (single-tx provenance writes)

**Files:**
- Create: `src/services/extraction_v3/persistence.py`
- Test: `tests/extraction_v3/test_persistence_provenance.py`

- [ ] **Step 1: Failing test (CRITICAL — provenance contract)**

```python
def test_provenance_written_for_every_committed_field(db):
    result = ExtractionResult(
        doc_type="invoice", doc_pk="INV-TEST-001",
        committed=[CommittedField(field_path="invoice_id", value="INV-TEST-001",
                                  page=0, bbox=(0,0,1,1), evidence_text="INV-TEST-001",
                                  model="layoutlmv3", model_confidence=0.9,
                                  judge_actions=[], final_confidence=0.9)],
        residuals=[], judge_calls=0, pipeline_version="v3.1.0",
    )
    persist(result)
    cur = db.cursor()
    cur.execute("SELECT count(*) FROM proc.bp_extraction_provenance_v3 WHERE doc_pk='INV-TEST-001'")
    assert cur.fetchone()[0] == 1
    cur.execute("SELECT invoice_id FROM proc.bp_invoice WHERE invoice_id='INV-TEST-001'")
    assert cur.fetchone()[0] == "INV-TEST-001"

def test_provenance_failure_rolls_back_data(db, monkeypatch):
    """If provenance write fails, the data write must roll back."""
    ... # cause provenance INSERT to raise; assert proc.bp_invoice has no row for that pk
```

- [ ] **Step 2: Implement** — `persist(result)` opens one transaction, writes header to `proc.bp_<doc_type>`, line items to `proc.bp_<doc_type>_line_items`, provenance rows to `proc.bp_extraction_provenance_v3`. On any error, rollback. Pipeline version is hard-coded for Plan 1 (`"v3.1.0"`).

- [ ] **Step 3: Run tests to PASS**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(extraction_v3): persistence with single-tx provenance writes"
```

---

## Task 24: Feature-flag dispatch + integration with `process_monitor_watcher`

**Files:**
- Create: `src/services/extraction_v3/dispatch.py`
- Modify: `src/services/process_monitor_watcher.py`
- Modify: `procwise.service` (add env vars) OR `.env`
- Test: `tests/extraction_v3/test_dispatch_feature_flag.py`

- [ ] **Step 1: Failing test** — when `EXTRACTION_PIPELINE_INVOICE=v3`, dispatch routes to `PipelineV3`; when `=agentnick`, routes to existing orchestrator. Default is `agentnick` (off-by-default per plan goal).

- [ ] **Step 2: Implement `dispatch.py`**

```python
# src/services/extraction_v3/dispatch.py
import os
from src.services.extraction_v3.pipeline import PipelineV3
from src.services.agent_nick_orchestrator import AgentNickOrchestrator

def dispatch(doc_path: str, doc_type: str):
    flag = os.getenv(f"EXTRACTION_PIPELINE_{doc_type.upper()}", "agentnick").lower()
    if flag == "v3":
        return PipelineV3().run(doc_path, doc_type)
    return AgentNickOrchestrator().process_document(doc_path, doc_type)
```

- [ ] **Step 3: Modify `process_monitor_watcher.py`** — replace direct call to AgentNick with `from src.services.extraction_v3.dispatch import dispatch; result = dispatch(path, doc_type)`. Convert v3's `ExtractionResult` to whatever shape the watcher's downstream expects (likely the existing dict shape — write a thin adapter).

- [ ] **Step 4: Add env vars to `.env.example`** with default `agentnick`.

- [ ] **Step 5: Run tests to PASS**

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(extraction_v3): per-category feature-flag dispatch + watcher wiring"
```

---

## Task 25: Lifespan startup — load schemas + warm models

**Files:**
- Modify: `src/api/main.py` (lifespan handler)

- [ ] **Step 1: Failing test** — start the FastAPI app with v3 enabled, hit `/health`, expect a new `extraction_v3: {"schemas_loaded": 4, "models_warm": [...]}` block in the response.

- [ ] **Step 2: Implement** — in `main.py` lifespan startup, call `load_all_schemas()` (fail-loud on drift), warm L1 + L2 model singletons, register them with the dispatch module so per-doc parsing doesn't pay model-load cost.

- [ ] **Step 3: Verify with `systemctl restart procwise`** — service should come up clean. If schema drift exists, service refuses to start with a clear message naming the offending field. **Per the loop monitor's existing watch, restart will surface any startup error.**

- [ ] **Step 4: Run tests to PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(extraction_v3): lifespan startup loads schemas + warms models"
```

---

## Task 26: Acquire 5 fixture invoices + ground-truth labels

**Files:**
- Create: `tests/extraction_v3/fixtures/invoices/INV-001-clean.pdf` and `.expected.json`
- ... `INV-002-multi-column.pdf` (DUNCAN/AQUARIUS-class) — covers I-37
- ... `INV-003-items-after-totals.pdf` — covers I-34
- ... `INV-004-vendor-quirky.pdf` (NEXASPARK or Eleanor Price-class) — covers I-18, I-38
- ... `INV-005-scanned.pdf` — covers OCR path

For each: a corresponding `INV-NNN.expected.json` with hand-labeled correct values for every field defined in `invoice.yaml`.

- [ ] **Step 1: Pick 5 documents** from `documents/invoice/` covering the 5 layout classes. Confirm their ground truth by reading the PDF.

- [ ] **Step 2: Author each `expected.json`**

```json
{
  "doc_type": "invoice",
  "header": {
    "invoice_id": "INV-005-41",
    "supplier_name": "TECHWORLD",
    "invoice_date": "2025-10-15",
    "invoice_amount": 6750.00,
    "tax_amount": 1215.00,
    "tax_percent": 18.0,
    "currency": "GBP",
    "po_id": "PO405867"
  },
  "line_items": [
    {"description": "SOFTWARE IMPLEMENTATION", "amount": 5000.00},
    {"description": "TRAINING & WORKSHOPS",     "amount": 1750.00}
  ]
}
```

- [ ] **Step 3: Commit fixtures** (use `git lfs` if PDFs > 5 MB; otherwise commit directly)

```bash
git commit -m "test(extraction_v3): 5 hand-labeled invoice fixtures (clean/multi-column/items-after-totals/vendor-quirky/scanned)"
```

---

## Task 27: End-to-end integration test

**Files:**
- Modify/extend: `tests/extraction_v3/test_pipeline_e2e.py`

- [ ] **Step 1: Add the field-by-field comparison test**

```python
import json
from pathlib import Path
from src.services.extraction_v3.pipeline import PipelineV3
from src.services.extraction_v3.persistence import persist

FIXTURES = Path(__file__).parent / "fixtures/invoices"

@pytest.mark.parametrize("fixture_id", ["INV-001-clean", "INV-002-multi-column",
                                          "INV-003-items-after-totals",
                                          "INV-004-vendor-quirky", "INV-005-scanned"])
def test_e2e_extraction_matches_ground_truth(fixture_id, db_session):
    pdf = FIXTURES / f"{fixture_id}.pdf"
    expected = json.loads((FIXTURES / f"{fixture_id}.expected.json").read_text())

    result = PipelineV3().run(str(pdf), "invoice")
    persist(result)

    # Pull persisted row back from DB
    actual_header = fetch_invoice_row(result.doc_pk)
    for field, expected_v in expected["header"].items():
        assert actual_header[field] == expected_v, \
            f"{fixture_id}: field {field}: got {actual_header[field]}, expected {expected_v}"

    # Verify zero hallucination — every committed value's evidence_text is in source
    parsed_text = ...  # re-parse to get full_text
    for prov in fetch_provenance(result.doc_pk):
        assert prov["evidence_text"] in parsed_text, \
            f"{fixture_id}: HALLUCINATION: {prov['field_path']}={prov['value']!r} (evidence not in source)"
```

- [ ] **Step 2: Run on all 5 fixtures**

Run: `pytest tests/extraction_v3/test_pipeline_e2e.py -v --tb=short`

**Expected for Plan 1**: per the iteration mandate, **all five fixtures must eventually pass before Plan 1 is closed** — including the multi-column, items-after-totals, vendor-quirky, and scanned cases. Pre-trained models are the *starting point* of Task 27, not the ending point. If a fixture fails, this is the trigger for in-plan iteration: research the failure mode, fix it, re-run. **Do NOT defer failures to Plan 2 unless they genuinely require labeled-corpus fine-tuning** (and even then, escalate explicitly with evidence rather than silently passing the buck).

- [ ] **Step 3: For each fixture that fails, immediately iterate**:
  1. Read the diff field-by-field. Classify the failure: parser produced wrong tokens? L2 extractor missed the field? Bind error? Invariant fired wrongly? Judge picked wrong candidate?
  2. Research the failure class. Examples: if Docling fragments a multi-column layout, check Docling's reading-order config / table-extraction options; if PaddleOCR produces low-confidence output on a specific font, switch to its `en_PP-OCRv4` weights; if LayoutLMv3 mis-labels for a vendor, add a vendor template via `template_store_pg` so the deterministic locator hits before the model; if the judge picks wrong, sharpen its tiebreaker prompt with vendor-specific examples.
  3. Implement the fix. Re-run the failing test. Confirm pass. Confirm no regression on other fixtures.
  4. Only after exhausting in-plan fixes, escalate to Plan 2 with a written explanation in `artifacts/log_monitor/ISSUES.md` (prefix `V3-` followed by a description) of what was tried and why fine-tuning is required.
- **Do NOT relax test thresholds. Do NOT mark fixtures as `xfail`. Do NOT skip the scanned case.** The truth is the ground truth, and the plan is closed when ground truth is met.

- [ ] **Step 4: Commit**

```bash
git commit -m "test(extraction_v3): end-to-end integration test against 5 fixtures (zero-hallucination + ground-truth)"
```

---

## Task 28: Live extraction smoke test against `proc.process_monitor` queue

**Files:**
- Create: `scripts/extraction_v3_live_smoke.py`

- [ ] **Step 1: Write the smoke script**

```python
# scripts/extraction_v3_live_smoke.py
"""
Pull the most recent N invoices from proc.process_monitor (status='Completed' or
'Extracting'), run them through PipelineV3, and report:
  - extraction success / failure rate
  - per-doc judge call count
  - per-doc latency
  - any committed value whose evidence_text is NOT a substring of source (fail-loud)

DOES NOT WRITE to the live tables — uses a transaction that rolls back at end.
"""
import argparse, json, time
from src.services.extraction_v3.pipeline import PipelineV3
from src.services.extraction_v3.parsers.router import parse
from src.config import get_db_connection

def main(n: int = 20):
    pipeline = PipelineV3()
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT record_id, file_path, category FROM proc.process_monitor
            WHERE category='invoice' AND status IN ('Completed', 'Extracting')
            ORDER BY created_at DESC LIMIT %s
        """, (n,))
        docs = cur.fetchall()
    results = []
    for record_id, path, _ in docs:
        t0 = time.time()
        try:
            result = pipeline.run(path, "invoice")
            parsed = parse(path)
            hallucinations = [
                cf for cf in result.committed
                if cf.evidence_text not in parsed.full_text
            ]
            results.append({
                "record_id": record_id, "path": path,
                "committed_count": len(result.committed),
                "residuals_count": len(result.residuals),
                "judge_calls": result.judge_calls,
                "latency_s": round(time.time() - t0, 2),
                "hallucinations": [cf.field_path for cf in hallucinations],
            })
        except Exception as e:
            results.append({"record_id": record_id, "path": path, "error": str(e),
                            "latency_s": round(time.time() - t0, 2)})
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    main(**vars(ap.parse_args()))
```

- [ ] **Step 2: Run it** — `python scripts/extraction_v3_live_smoke.py --n 20 > artifacts/log_monitor/extraction_v3_smoke_$(date +%s).json`

- [ ] **Step 3: Audit output** for:
  - **Hard fail**: any `hallucinations: [...]` non-empty list. **This is a P0 bug** per the iteration mandate. Investigate and fix immediately.
  - **Soft signal**: residuals_count and judge_calls distribution — does the pipeline match spec §10's perf budget?
  - **Accuracy proxy**: how many extracted values agree with what the legacy stack persisted (cross-reference `proc.bp_invoice` rows for the same `record_id`).

- [ ] **Step 4: Append findings to `artifacts/log_monitor/ISSUES.md`** with prefix `V3-` (e.g. `V3-001`, `V3-002`) so they're distinguishable from legacy issues.

- [ ] **Step 5: Commit script**

```bash
git commit -m "test(extraction_v3): live smoke script against process_monitor queue"
```

---

## Task 29: Documentation + handoff

**Files:**
- Create: `docs/extraction_v3_runbook.md`

- [ ] **Step 1: Write a short ops runbook** covering:
  - How to flip the per-category feature flag
  - How to interpret `proc.bp_extraction_provenance_v3`
  - How to add a new field to a YAML schema (and the matching DB migration)
  - How to run the smoke script
  - Known limitations of pre-trained models in Plan 1 (motivates Plan 2)

- [ ] **Step 2: Commit**

```bash
git commit -m "docs(extraction_v3): runbook"
```

---

## Plan completion criteria

This plan is **complete** when ALL of:

1. All 29 tasks have all checkboxes ticked.
2. `pytest tests/extraction_v3/ -v` passes — **every test green, including all five integration fixtures in Task 27 against ground truth**. No `xfail`, no `skip`, no relaxed thresholds. If a fixture won't pass with pre-trained models, iterate (Task 27 Step 3) until it does, or formally escalate to Plan 2 with a written justification (Task 27 Step 3.4) — but escalation is a last resort, not a default path.
3. The live smoke script (Task 28) runs against ≥ 20 production invoices with **zero hallucinations** (no committed value whose `evidence_text` is not a substring of source). This is gating P0.
4. Live smoke field-level agreement with legacy on documents legacy got right ≥ 99% (toward §13 criterion 1) and recovery rate on documents legacy got wrong ≥ 90% (toward §13 criterion 2). If either misses, iterate before closing the plan.
5. p95 end-to-end latency < 25 s, judge calls p95 ≤ 3 / p99 ≤ 4 (toward §13 criteria 4-5). Measure during Task 28 smoke; if budget blown, profile and fix.
6. Provenance row written for 100% of committed fields, validated on the smoke run.
7. The service starts cleanly with `EXTRACTION_PIPELINE_INVOICE=v3` set, processes a real document via `process_monitor_watcher` end-to-end, and writes both `proc.bp_invoice` and `proc.bp_extraction_provenance_v3` rows.
8. With the flag flipped back to `agentnick`, the legacy path remains functional (toggle and re-run smoke against a fresh invoice).

**On completion, immediately:**
1. Run `python scripts/extraction_v3_live_smoke.py --n 100` and append findings to `ISSUES.md`.
2. Begin Plan 2 (LayoutLMv3 fine-tune corpus + training pipeline) — that's where the real accuracy gains land.
3. Continue iterating per project memory `project_extraction_redesign_iteration_mandate.md` until all seven §13 success criteria pass.

---

## Future plans referenced from this one

- **Plan 2 — LayoutLMv3 fine-tune** (`docs/superpowers/plans/2026-XX-XX-extraction-redesign-finetune.md`): training corpus prep from `proc.bp_*`, hallucination scrub, training loop, eval against held-out vendor.
- **Plan 3 — PO/Quote/Contract YAML + fixtures** (`...-extraction-redesign-other-doctypes.md`): fills the three stub YAMLs, adds 30 fixtures total (10 per doc type), parameterises Task 27 across all four types.
- **Plan 4 — Live-shadow + cutover** (`...-extraction-redesign-cutover.md`): adds shadow-mode comparison logger, the 2-week measurement window, and the per-category cutover decision per spec §13 criteria.
