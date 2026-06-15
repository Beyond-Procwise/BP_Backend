# Engineered Extraction Pipeline V2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LLM-primary extraction with an algorithm-first pipeline where every field is asserted only after multi-strategy consensus + verification, with an LLM acting only as one optional locator. Target: **100% accuracy on every field the algorithm commits to**, with abstention as a first-class output.

**Non-technical user requirement:** Adding a new vendor / new layout must be possible **without YAML or code edits** — through a sample-upload + correction interface that derives the per-vendor extraction program from human-labeled examples.

---

## Architecture (one diagram)

```
PDF / DOCX / XLSX / JPEG
        │
        ▼
[A] Parser Stack (existing: PyMuPDF / pdfplumber / python-docx / openpyxl / Tesseract)
        │  → ParsedDocument with tokens, bboxes, tables, full text
        ▼
[B] Layout Fingerprint  ←──────────────┐
        │                              │
        ▼                              │
[C] Template Store lookup  ────────► hit: load per-vendor program (cached locators)
        │ miss
        ▼
[D] Multi-Strategy Locator Framework        per field, run N strategies → vote
        │
        ▼
[E] Type-Validated Construction              every value passes through a typed parser
        │
        ▼
[F] Verification Network                     math, FK, range, cross-field rules
        │     ├─ all pass → commit
        │     └─ fail    → demote, re-locate, or abstain
        ▼
[G] Anchored Provenance Re-check             every value substring-found in source
        │
        ▼
[H] Confidence-Routed Output
        ├─ confidence ≥ threshold + all gates → bp_invoice / bp_quote / bp_purchase_order
        └─ residuals → review queue → human correction → template store update ─┘
                                                                               (B feedback loop)
```

The vendor-onboarding loop is the same code path as review-queue corrections: a non-technical user reviews and corrects extracted values, and those corrections train the per-vendor program for next time.

---

## Module Tree (new code lives in `src/services/extraction_v2/`)

```
src/services/extraction_v2/
├── __init__.py
├── types.py                 # Pillar 4: validating types
├── parsers/                 # Pillar 6: specialized deterministic parsers
│   ├── __init__.py
│   ├── dates.py             # IsoDate parsing
│   ├── amounts.py           # Money parsing (Decimal-safe)
│   ├── postcodes.py         # UK / US / EU postcode validation + country lookup
│   ├── addresses.py         # split address block → line1/line2/city/postcode/country
│   ├── tables.py            # column-header + row segmentation for line items
│   └── currency.py          # ISO-4217 set
├── locator/                 # Pillar 1: multi-strategy consensus
│   ├── __init__.py
│   ├── base.py              # Locator protocol
│   ├── consensus.py         # vote runner, abstention, residual emitter
│   ├── strategies/
│   │   ├── label_anchored.py     # vocabulary-based label proximity
│   │   ├── position.py           # position heuristic (top-right etc.)
│   │   ├── format.py             # regex-anchored format match
│   │   ├── filename.py           # filename pattern fallback
│   │   ├── cross_doc.py          # cross-document reference
│   │   └── llm_grounded.py       # LLM as ONE optional locator (capped 0.85)
│   └── registry.py          # field → list-of-locators mapping
├── verification/            # Pillar 2: declarative rule graph
│   ├── __init__.py
│   ├── network.py           # rule graph runner
│   └── rules/
│       ├── math.py          # subtotal + tax = total, line sum = subtotal
│       ├── dates.py         # invoice_date < due_date, ranges
│       ├── cross_ref.py     # po_id ∈ bp_purchase_order
│       └── format.py        # type-validity rerun
├── provenance.py            # Pillar 3: anchor + re-ground
├── fingerprint.py           # Pillar 5: layout hash
├── template_store.py        # Pillar 5: DB-backed per-vendor program cache
├── pipeline.py              # orchestrates A→H end-to-end
└── fixtures.py              # test-harness helpers (Pillar 7)

tests/extraction_v2/
├── __init__.py
├── test_types.py
├── parsers/test_*.py        # one test file per parser, ≥20 cases each
├── locator/test_*.py
├── verification/test_*.py
├── test_pipeline.py
└── fixtures/vendors/        # Pillar 7: every supported vendor = a fixture
    └── {vendor}/{doc}.{pdf|docx}
        + {doc}.expected.yaml
```

---

## Tasks

### Task 0: Branch + worktree

**Files:**
- Create branch: `extraction-v2`

- [ ] **Step 1: Branch off Development**
```bash
git checkout -b extraction-v2 Development
```

- [ ] **Step 2: Verify clean test baseline**
```bash
python -m pytest tests/services/ tests/structural_extractor/ -q
```
Expected: 294 passed.

---

### Task 1: Validating type system

**Files:**
- Create: `src/services/extraction_v2/__init__.py`
- Create: `src/services/extraction_v2/types.py`
- Create: `tests/extraction_v2/__init__.py`
- Create: `tests/extraction_v2/test_types.py`

- [ ] **Step 1: Write the failing tests for `Money`, `IsoDate`, `Postcode`, `InvoiceId`, `PoId`, `QuoteId`, `Currency`**

```python
# tests/extraction_v2/test_types.py
import pytest
from datetime import date
from decimal import Decimal
from services.extraction_v2.types import (
    Money, IsoDate, Postcode, InvoiceId, PoId, QuoteId, Currency,
    InvalidValue,
)

class TestMoney:
    @pytest.mark.parametrize("raw,expected", [
        ("1234.56", Decimal("1234.56")),
        ("£1,234.56", Decimal("1234.56")),
        ("1.234,56", Decimal("1234.56")),       # EU
        ("(123.45)", Decimal("-123.45")),        # parens-negative
        (1234, Decimal("1234.00")),
    ])
    def test_constructs_clean(self, raw, expected):
        assert Money(raw) == expected

    @pytest.mark.parametrize("bad", ["abc", "", "  ", "1.2.3", "1e100"])
    def test_rejects_garbage(self, bad):
        with pytest.raises(InvalidValue):
            Money(bad)

    def test_negative_for_credits_only(self):
        # "(50)" allowed, "-50" allowed, but huge negatives rejected
        Money("(50.00)")  # ok
        with pytest.raises(InvalidValue):
            Money("-1e10")

class TestIsoDate:
    @pytest.mark.parametrize("raw,expected", [
        ("2025-10-10", date(2025,10,10)),
        ("10/10/2025", date(2025,10,10)),
        ("1st Oct, 2024", date(2024,10,1)),
        ("Oct 1, 2024",  date(2024,10,1)),
    ])
    def test_parses_common_formats(self, raw, expected):
        assert IsoDate(raw) == expected

    @pytest.mark.parametrize("bad", ["2099-12-31", "1999-01-01", "not a date", ""])
    def test_rejects_out_of_range(self, bad):
        with pytest.raises(InvalidValue):
            IsoDate(bad)

class TestPostcode:
    @pytest.mark.parametrize("raw,country", [
        ("RH13 5QH",  "United Kingdom"),
        ("B3 1AA",    "United Kingdom"),
        ("EC2A 3NW",  "United Kingdom"),
        ("10001",     "United States"),
        ("90210-1234","United States"),
    ])
    def test_detects_country(self, raw, country):
        pc = Postcode(raw)
        assert pc.country == country

    @pytest.mark.parametrize("bad", ["", "abc", "ZZZ 999"])
    def test_rejects_garbage(self, bad):
        with pytest.raises(InvalidValue):
            Postcode(bad)

class TestInvoiceId:
    @pytest.mark.parametrize("raw,expected", [
        ("INV600820",      "INV600820"),
        ("INV-2026-01602", "INV-2026-01602"),
        ("132548",         "INV132548"),       # bare-numeric → prefix added
    ])
    def test_normalizes_prefix(self, raw, expected):
        assert str(InvoiceId(raw)) == expected

    @pytest.mark.parametrize("bad", ["", "abc", "PO12345", "INV"])
    def test_rejects_non_invoice(self, bad):
        with pytest.raises(InvalidValue):
            InvoiceId(bad)

class TestCurrency:
    @pytest.mark.parametrize("raw,expected", [
        ("GBP", "GBP"), ("USD", "USD"), ("£", "GBP"), ("$", "USD"), ("€", "EUR"),
    ])
    def test_iso_or_symbol(self, raw, expected):
        assert Currency(raw) == expected

    def test_rejects_unknown(self):
        with pytest.raises(InvalidValue):
            Currency("XYZ")
```

Run: `python -m pytest tests/extraction_v2/test_types.py -v`
Expected: ALL FAIL (module not yet implemented).

- [ ] **Step 2: Implement `types.py`**

Key shape:
```python
# src/services/extraction_v2/types.py
"""Validating types for extraction values.

Every type's constructor either returns a valid normalized value or
raises InvalidValue. Code paths that produce a typed value have already
passed validation — there is no need to re-check downstream.
"""
from __future__ import annotations
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
import re
from typing import Optional


class InvalidValue(ValueError):
    """Raised when a typed constructor receives input that fails validation."""


class Money(Decimal):
    """Non-negative-or-credit decimal money value with 2dp normalization."""
    _MAX = Decimal("9_999_999_999.99")
    def __new__(cls, raw):
        # ... handles £/$/€ symbols, comma vs dot decimal, parens-negative
        pass


class IsoDate(date):
    """Date in [2000-01-01, today + 5 years]. Rejects far-future / pre-2000."""
    _MIN = date(2000, 1, 1)
    @classmethod
    def _max(cls): return date.today() + timedelta(days=5*365)
    def __new__(cls, raw):
        # ... parses many formats via dateparser
        pass


class Postcode(str):
    """UK / US / EU postcode with country inference."""
    UK_RE = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$", re.I)
    US_RE = re.compile(r"^\d{5}(?:-\d{4})?$")
    @property
    def country(self) -> str: ...


class InvoiceId(str): ...   # validates and normalizes prefix
class PoId(str):     ...
class QuoteId(str):  ...
class Currency(str):
    ISO = {"GBP","USD","EUR","JPY","INR","CAD","AUD","CHF","CNY","NZD","ZAR"}
    SYM = {"£":"GBP", "$":"USD", "€":"EUR", "¥":"JPY", "₹":"INR"}
```

- [ ] **Step 3: Run tests**
```bash
python -m pytest tests/extraction_v2/test_types.py -v
```
Expected: ALL PASS.

- [ ] **Step 4: Commit**
```bash
git add src/services/extraction_v2/__init__.py src/services/extraction_v2/types.py \
        tests/extraction_v2/__init__.py tests/extraction_v2/test_types.py
git commit -m "feat(extraction-v2): validating type system (Money/IsoDate/Postcode/PKs/Currency)"
```

---

### Task 2: Specialized parsers — dates, amounts, addresses, postcodes, currency

**Files:**
- Create: `src/services/extraction_v2/parsers/{__init__,dates,amounts,postcodes,addresses,currency}.py`
- Create: `tests/extraction_v2/parsers/__init__.py`
- Create: `tests/extraction_v2/parsers/test_*.py`

For each parser, the contract is:
```python
def parse_X(raw: str | None) -> Optional[X]:
    """Returns the typed value or None if input is unparseable.
    NEVER raises — abstention is encoded as None."""
```

Each parser file backed by ≥20 unit tests covering the variations seen in production data.

**Address parser specifics** (the gap class the user reported most often):
```python
@dataclass(frozen=True)
class ParsedAddress:
    line1: Optional[str]
    line2: Optional[str]
    city: Optional[str]
    postcode: Optional[Postcode]
    country: Optional[str]

def parse_address(raw: str | None) -> ParsedAddress:
    """Split a multi-line address block into structured parts.
    Strategy: scan for postcode anchor, then back-walk to find
    city, line1, line2 segments by line breaks and comma boundaries."""
```

- [ ] **Step 1: Tests for each parser** (20+ cases each, real data variants)
- [ ] **Step 2: Implement `dates.py` (uses `dateparser`)**
- [ ] **Step 3: Implement `amounts.py`**
- [ ] **Step 4: Implement `postcodes.py`**
- [ ] **Step 5: Implement `addresses.py`** (the highest-leverage one)
- [ ] **Step 6: Implement `currency.py`**
- [ ] **Step 7: Run all parser tests**
```bash
python -m pytest tests/extraction_v2/parsers/ -v
```
- [ ] **Step 8: Commit**
```bash
git commit -m "feat(extraction-v2): deterministic parsers (dates/amounts/postcodes/addresses/currency)"
```

---

### Task 3: Locator framework + consensus voting

**Files:**
- Create: `src/services/extraction_v2/locator/{base,consensus,registry}.py`
- Create: `src/services/extraction_v2/locator/strategies/{label_anchored,position,format,filename,llm_grounded}.py`
- Create: `tests/extraction_v2/locator/test_*.py`

**Core types:**
```python
# locator/base.py
class LocatorOutput(NamedTuple):
    value: Any                    # typed value (Money, IsoDate, etc.) or None
    confidence: float             # 0.0–1.0 from this strategy alone
    evidence: AnchorRef           # bbox / char-offset of source token(s)
    why: str                      # human-readable rationale

class Locator(Protocol):
    field: str
    name: str
    def locate(self, doc: ParsedDocument) -> Optional[LocatorOutput]: ...
```

**Consensus runner:**
```python
# locator/consensus.py
@dataclass
class ConsensusResult:
    field: str
    value: Any                    # the agreed value (or None on abstain)
    confidence: float             # computed from agreement-rate + locator confidences
    candidates: list[LocatorOutput]
    abstained: bool
    why: str

def run_locators(field: str, locators: list[Locator], doc: ParsedDocument) -> ConsensusResult:
    """Run all locators in parallel, vote on the result.
    
    Voting policy:
      - Group candidates by canonical-equality of value
      - Largest group wins IF size ≥ 2 OR (size == 1 AND confidence ≥ 0.95)
      - Abstain if no group reaches threshold
      - Confidence = mean(group_member_confidences) × group_size / total_locators
    """
```

- [ ] **Step 1: Tests for `LocatorOutput`, `Locator` protocol, consensus voting**
- [ ] **Step 2: Implement `base.py`, `consensus.py`, `registry.py`**
- [ ] **Step 3: Implement `label_anchored.py`** — uses externalized vocabulary
- [ ] **Step 4: Implement `position.py`, `format.py`, `filename.py`**
- [ ] **Step 5: Implement `llm_grounded.py`** — wraps existing LangExtract adapter, capped at confidence 0.85
- [ ] **Step 6: Wire `registry.py`** — `register_locator(field, locator)` + `locators_for(field)`
- [ ] **Step 7: Tests pass**
- [ ] **Step 8: Commit**

---

### Task 4: Verification network

**Files:**
- Create: `src/services/extraction_v2/verification/{network,rules/{math,dates,cross_ref,format}}.py`
- Create: `tests/extraction_v2/verification/test_*.py`

```python
@dataclass
class Rule:
    name: str
    fields: list[str]                                    # which fields participate
    check: Callable[[dict[str, Any]], bool]              # returns True if rule passes
    on_fail: Literal["demote", "abstain", "warn"]       # what to do if rule fails

def run_verification(values: dict[str, Any], rules: list[Rule]) -> VerificationResult:
    """Run all rules; return per-rule pass/fail and per-field demotions."""
```

Rules to implement:
- `subtotal_plus_tax_eq_total` (math)
- `line_amounts_sum_to_subtotal` (math)
- `tax_amount_matches_percent` (math)
- `invoice_date_before_due_date` (dates)
- `dates_in_range` (dates)
- `po_id_exists_in_purchase_orders` (cross_ref)
- `currency_iso_4217` (format)

- [ ] **Step 1: Tests for each rule + the network runner**
- [ ] **Step 2: Implement rules**
- [ ] **Step 3: Implement network runner**
- [ ] **Step 4: Commit**

---

### Task 5: Layout fingerprint + template store

**Files:**
- Create: `src/services/extraction_v2/{fingerprint,template_store}.py`
- Create: `tests/extraction_v2/test_fingerprint.py`
- DB: reuse existing `proc.bp_extraction_patterns` table (already has `layout_signature`, `vendor_id`, `success_count`, `pattern_data`)

```python
# fingerprint.py
def compute_fingerprint(doc: ParsedDocument) -> str:
    """Stable hash representing the document's layout.
    Components: page count, top-30%-token-set, font cluster signature,
    column boundary signature, table count per page.
    Same vendor/template = same fingerprint."""
```

```python
# template_store.py
@dataclass
class VendorTemplate:
    fingerprint: str
    vendor_name: Optional[str]
    field_locators: dict[str, list[str]]  # field → list of locator-strategy names that succeeded
    gates: dict[str, dict]                # tightened gates for this template
    success_count: int

class TemplateStore:
    def get(self, fingerprint: str) -> Optional[VendorTemplate]: ...
    def upsert(self, template: VendorTemplate) -> None: ...
    def record_success(self, fingerprint: str, field: str, strategy: str) -> None: ...
    def record_correction(self, fingerprint: str, field: str, value: Any, evidence: AnchorRef) -> None: ...
```

- [ ] **Step 1: Tests for fingerprint stability (same doc → same hash) + template CRUD**
- [ ] **Step 2: Implement fingerprint hashing**
- [ ] **Step 3: Implement TemplateStore against `bp_extraction_patterns`**
- [ ] **Step 4: Commit**

---

### Task 6: Pipeline orchestrator

**Files:**
- Create: `src/services/extraction_v2/pipeline.py`
- Create: `tests/extraction_v2/test_pipeline.py`

```python
class ExtractionPipelineV2:
    def extract(self, doc: ParsedDocument, doc_type: str) -> ExtractionResultV2:
        # 1. Compute fingerprint
        fp = compute_fingerprint(doc)
        # 2. Lookup template
        template = self.templates.get(fp)
        # 3. For each field:
        #    - If template hit: use template's cached locators
        #    - Else: use full registry
        # 4. Run consensus per field → ConsensusResult
        # 5. Run verification network → demote/abstain
        # 6. Re-ground every committed value
        # 7. Build final output:
        #    - committed: dict[field, ConsensusResult]
        #    - residuals: list[ResidualField]
        #    - provenance: dict[field, AnchorRef]
        return ExtractionResultV2(...)
```

- [ ] **Step 1: Test the pipeline end-to-end on a synthetic doc**
- [ ] **Step 2: Implement orchestrator**
- [ ] **Step 3: Wire feature flag in `agent_nick_orchestrator.py`** — `USE_EXTRACTION_V2=true` routes to new pipeline
- [ ] **Step 4: Commit**

---

### Task 7: Vendor onboarding (the non-technical user path)

This is the differentiating piece — adding a new vendor without YAML or code.

**Files:**
- Create: `src/api/routes/vendor_onboarding.py`
- Create: `src/api/templates/vendor_review.html`
- Create: `tests/api/test_vendor_onboarding.py`

**User workflow:**

1. **Upload sample** — `POST /vendors/onboard/upload` with a representative document
2. **System runs default extraction** — produces values with confidences and bboxes (residuals included)
3. **Review-and-correct UI** — single-page web form:
   - Renders the document on the left (PDF preview)
   - Shows field-by-field on the right with extracted value + bbox highlighted
   - User can: (a) accept value, (b) correct value, (c) click on the page to re-anchor a value to a different location
4. **Submit** — `POST /vendors/onboard/{session_id}/save`
   - Each user-confirmed-or-corrected value generates a labeled training pair: `(field, value, anchor_bbox)`
   - System derives locators from these labels:
     - For each field, find the most stable description of "where is this token relative to known anchors" (label proximity, position rank, format pattern)
     - Encode as a per-template locator config
   - Layout fingerprint computed
   - VendorTemplate stored
5. **First match-and-commit** — next document with same fingerprint uses this template; ≥3 successful auto-extractions promotes the template to "verified"

**Endpoints:**
```python
POST /vendors/onboard/upload          # creates session, runs default extraction
GET  /vendors/onboard/{session_id}    # serves the HTML review form
POST /vendors/onboard/{session_id}/correct   # user corrects a single field
POST /vendors/onboard/{session_id}/anchor    # user clicks a bbox to re-anchor
POST /vendors/onboard/{session_id}/save      # finalize + persist template
GET  /vendors/templates                       # list all known templates
GET  /vendors/templates/{fingerprint}/audit   # see template's history
```

- [ ] **Step 1: Spec the data model in detail** (OnboardingSession, FieldCorrection, derived Locator hints)
- [ ] **Step 2: Test the upload + default-extraction step**
- [ ] **Step 3: Implement upload route**
- [ ] **Step 4: Implement HTML review template** (single-page, vanilla JS — no frontend framework)
- [ ] **Step 5: Implement correction route + locator derivation**
- [ ] **Step 6: Implement save route — persist template + commit values to bp_* tables**
- [ ] **Step 7: End-to-end manual test with a real document**
- [ ] **Step 8: Commit**

---

### Task 8: Test fixture harness — every supported vendor as a test

**Files:**
- Create: `tests/extraction_v2/fixtures/vendors/.gitkeep`
- Create: `tests/extraction_v2/test_vendor_fixtures.py` (parametrized; auto-discovers fixtures)

```python
# tests/extraction_v2/test_vendor_fixtures.py
import yaml
from pathlib import Path
import pytest

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "vendors"

def discover_fixtures():
    for vendor_dir in FIXTURE_ROOT.iterdir():
        if not vendor_dir.is_dir(): continue
        for src in vendor_dir.glob("*.[pPdD][dDoO][fFcC]*"):
            expected = src.with_suffix(".expected.yaml")
            if expected.exists():
                yield pytest.param(src, expected, id=f"{vendor_dir.name}/{src.name}")

@pytest.mark.parametrize("doc_path,expected_path", discover_fixtures())
def test_vendor_extraction(doc_path, expected_path):
    """Every committed field must match the expected value exactly.
    Residuals (abstentions) are allowed; this is the 'abstain-or-be-right' contract."""
    doc = parse_any(doc_path)
    expected = yaml.safe_load(expected_path.read_text())
    result = ExtractionPipelineV2().extract(doc, expected["doc_type"])
    for field, exp_value in expected["fields"].items():
        if field in result.committed:
            actual = result.committed[field].value
            assert actual == exp_value, f"{doc_path.name}::{field}: expected {exp_value!r}, got {actual!r}"
        # If field is residual, that's allowed — algorithm correctly abstained
```

- [ ] **Step 1: Add 5 starter fixtures** (one per known vendor — DUNCAN, TECHWORLD, ELEANOR PRICE, AQUARIUS, INFOTECH)
- [ ] **Step 2: Wire the parametrized test**
- [ ] **Step 3: Run; pipeline must pass for committed fields**
- [ ] **Step 4: Commit**

---

### Task 9: Live end-to-end test

- [ ] **Step 1: Reset all process_monitor records to Completed** (forces re-extraction)
- [ ] **Step 2: Restart procwise.service** with `USE_EXTRACTION_V2=true`
- [ ] **Step 3: Watch logs**, monitor:
  - Extraction success / abstention / failure
  - Per-field commit rate (target: 90%+ of fields with consensus)
  - Verification rule pass rate
  - Review-queue size
- [ ] **Step 4: Snapshot DB** and compute:
  - Asserted-field accuracy (vs ground truth from the user's punch list)
  - Coverage rate
  - Abstention rate
- [ ] **Step 5: Compare with v1 baseline numbers**
- [ ] **Step 6: Document findings** in `docs/superpowers/results/2026-05-03-v2-rerun.md`

---

### Task 10: Migration

- [ ] **Step 1: With v2 stable, flip default `USE_EXTRACTION_V2=true`**
- [ ] **Step 2: Mark v1 path deprecated; keep behind feature flag for 2 weeks**
- [ ] **Step 3: After 2 weeks of clean v2 operation, remove v1 code**
- [ ] **Step 4: Final commit and PR**

---

## Acceptance Criteria

By end of implementation:

1. **Pillar 1 (Multi-strategy)**: Every field has ≥3 independent locators registered; consensus voting documented in tests.
2. **Pillar 2 (Verification)**: ≥7 verification rules implemented; failed rules demote field confidence (testable).
3. **Pillar 3 (Provenance)**: Every committed value has `source_bbox` + `char_offset` recorded in `bp_extraction_provenance`.
4. **Pillar 4 (Type system)**: All field-level types raise `InvalidValue` on bad input; no untyped strings reach persistence.
5. **Pillar 5 (Templates)**: Documents with previously-seen fingerprint route through cached locators. New fingerprints trigger onboarding.
6. **Pillar 6 (Specialized parsers)**: Address, date, amount, postcode parsers each with ≥20 unit tests.
7. **Pillar 7 (Test fixtures)**: ≥5 vendor fixtures pass under the parametrized test.
8. **Pillar 8 (Loud abstention)**: Residuals appear in review queue with ALL candidate values + reason; never auto-committed.
9. **Vendor onboarding UI works end-to-end**: upload → review → save → next-doc-of-same-fingerprint extracts via cache.
10. **Live test**: ≥30 of the existing 39 documents extract with full consensus on critical fields (PK / amount / date / supplier).

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Locator framework adds latency | Run locators concurrently (each locator is independent — embarrassingly parallel) |
| Fingerprint collisions across vendors | Include layout-distinguishing features (column boundaries, font clusters); fall back to full multi-strategy if cached locators yield low confidence |
| Type-system rejection of legitimate values | Each rejection is logged with input; tune validators based on real failures |
| Vendor-onboarding UI is too complex for non-tech users | Single-page form, no JS framework, click-to-anchor as the only "advanced" interaction; written in plain English |
| Template-store DB conflicts during concurrent corrections | Pessimistic lock per fingerprint during write |
| LLM strategy gets too much weight | Hard-cap LLM confidence at 0.85; never load-bearing — algorithm-locator agreement always required |

---

## Notes for the implementer

- **No commits with failing tests.** Every step ends with green pytest.
- **Reuse existing infra**: the `bp_extraction_patterns`, `bp_extraction_provenance`, `extraction_review_queue` tables already exist. Use them.
- **Don't break v1.** New code lives in `extraction_v2/`. Feature flag controls routing.
- **Address parser is the single highest-leverage parser** — most user gaps are address-related. Prioritize its test coverage.
