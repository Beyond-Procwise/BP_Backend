# Extraction Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add adaptive OCR, a validation gate, and a remediation pass to the existing extraction pipeline so that bad data is caught before entering PostgreSQL.

**Architecture:** The existing 6-strategy ML ensemble (`ml_extraction_pipeline.py`) remains untouched. Three new components wrap around it: (1) an adaptive OCR selector that detects input quality and chooses the best OCR strategy, (2) a validation gate that scores extraction confidence and blocks low-quality data from persistence, and (3) a remediation service that re-extracts low-confidence fields using targeted strategies. A new `confidence_score` column on agent tables flags records for downstream weighting.

**Tech Stack:** Python 3.12, pdfplumber, PyMuPDF, EasyOCR/Tesseract (existing), Ollama (existing), PostgreSQL (existing), existing ML ensemble pipeline

**Spec:** `docs/superpowers/specs/2026-03-31-orchestration-rearchitecture-design.md` Section 6

**Depends on:** Orchestration Core plan (completed) — uses DAG Scheduler for multi-node extraction workflow

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/services/adaptive_ocr.py` | Detects input document quality (digital/clean scan/poor scan/photo) and selects optimal OCR strategy |
| `src/services/validation_gate.py` | Per-field and per-document confidence scoring, structural/semantic validation, pass/fail decision |
| `src/services/remediation_service.py` | Re-extracts low-confidence fields using targeted strategies and cross-references with existing Postgres data |
| `src/orchestration/migrations/002_confidence_columns.sql` | DDL to add confidence_score and needs_review columns to agent tables |
| `tests/test_adaptive_ocr.py` | Adaptive OCR tests |
| `tests/test_validation_gate.py` | Validation gate tests |
| `tests/test_remediation_service.py` | Remediation service tests |

### Modified Files

| File | Change |
|------|--------|
| `src/agents/data_extraction_agent.py` | Insert validation gate + remediation into `_process_single_document()` flow, between extraction and persistence |
| `src/orchestration/workflow_definitions.py` | Update `build_extraction_workflow()` to a multi-node DAG if DAG scheduler enabled |

### Unchanged Files

| File | Why |
|------|-----|
| `src/agents/ml_extraction_pipeline.py` | Existing 6-strategy ensemble — wrapped, not modified |
| `src/services/document_extractor.py` | Legacy extractor — continues as fallback |
| `src/services/ocr_pipeline.py` | Existing OCR — adaptive_ocr wraps it |
| `src/services/document_structurer.py` | LLM jsonification unchanged |

---

## Task 1: Confidence Score Database Migration

**Files:**
- Create: `src/orchestration/migrations/002_confidence_columns.sql`

- [ ] **Step 1: Create the migration SQL**

```sql
-- src/orchestration/migrations/002_confidence_columns.sql
-- Add confidence scoring columns to extraction agent tables
-- Spec: Section 6 of orchestration-rearchitecture-design.md

BEGIN;

-- Add confidence_score and needs_review to all agent extraction tables
ALTER TABLE proc.invoice_agent
    ADD COLUMN IF NOT EXISTS confidence_score REAL,
    ADD COLUMN IF NOT EXISTS needs_review BOOLEAN DEFAULT FALSE;

ALTER TABLE proc.purchase_order_agent
    ADD COLUMN IF NOT EXISTS confidence_score REAL,
    ADD COLUMN IF NOT EXISTS needs_review BOOLEAN DEFAULT FALSE;

ALTER TABLE proc.quote_agent
    ADD COLUMN IF NOT EXISTS confidence_score REAL,
    ADD COLUMN IF NOT EXISTS needs_review BOOLEAN DEFAULT FALSE;

ALTER TABLE proc.contracts
    ADD COLUMN IF NOT EXISTS confidence_score REAL,
    ADD COLUMN IF NOT EXISTS needs_review BOOLEAN DEFAULT FALSE;

-- Index for finding records that need review
CREATE INDEX IF NOT EXISTS ix_invoice_agent_review
    ON proc.invoice_agent (needs_review) WHERE needs_review = TRUE;

CREATE INDEX IF NOT EXISTS ix_purchase_order_agent_review
    ON proc.purchase_order_agent (needs_review) WHERE needs_review = TRUE;

CREATE INDEX IF NOT EXISTS ix_quote_agent_review
    ON proc.quote_agent (needs_review) WHERE needs_review = TRUE;

CREATE INDEX IF NOT EXISTS ix_contracts_review
    ON proc.contracts (needs_review) WHERE needs_review = TRUE;

COMMIT;
```

- [ ] **Step 2: Verify file structure**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -c "
with open('src/orchestration/migrations/002_confidence_columns.sql') as f:
    sql = f.read()
print(f'Migration: {len(sql)} bytes, {sql.count(\"ALTER TABLE\")} ALTER statements, {sql.count(\"CREATE INDEX\")} indexes')
"`
Expected: `Migration: ... bytes, 4 ALTER statements, 4 indexes`

- [ ] **Step 3: Commit**

```bash
git add src/orchestration/migrations/002_confidence_columns.sql
git commit -m "feat: add confidence_score and needs_review columns to agent tables"
```

---

## Task 2: Adaptive OCR Service

**Files:**
- Create: `src/services/adaptive_ocr.py`
- Create: `tests/test_adaptive_ocr.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_adaptive_ocr.py
"""Tests for adaptive OCR quality detection and strategy selection."""
import pytest
from unittest.mock import MagicMock, patch


def test_detect_digital_pdf():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality

    ocr = AdaptiveOCR()
    # A digital PDF has extractable text and no images-only pages
    quality = ocr.detect_quality(
        text_content="Invoice #12345\nSupplier: Acme Corp\nTotal: $1,500.00",
        page_count=1,
        has_extractable_text=True,
        image_ratio=0.0,
    )
    assert quality == DocumentQuality.DIGITAL


def test_detect_clean_scan():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality

    ocr = AdaptiveOCR()
    quality = ocr.detect_quality(
        text_content="",
        page_count=1,
        has_extractable_text=False,
        image_ratio=1.0,
        estimated_dpi=300,
    )
    assert quality == DocumentQuality.CLEAN_SCAN


def test_detect_poor_scan():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality

    ocr = AdaptiveOCR()
    quality = ocr.detect_quality(
        text_content="",
        page_count=1,
        has_extractable_text=False,
        image_ratio=1.0,
        estimated_dpi=150,
    )
    assert quality == DocumentQuality.POOR_SCAN


def test_detect_photo():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality

    ocr = AdaptiveOCR()
    quality = ocr.detect_quality(
        text_content="",
        page_count=1,
        has_extractable_text=False,
        image_ratio=1.0,
        estimated_dpi=72,
        is_skewed=True,
    )
    assert quality == DocumentQuality.PHOTO


def test_strategy_for_digital():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality, OCRStrategy

    ocr = AdaptiveOCR()
    strategy = ocr.select_strategy(DocumentQuality.DIGITAL)
    assert strategy == OCRStrategy.DIRECT_TEXT


def test_strategy_for_clean_scan():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality, OCRStrategy

    ocr = AdaptiveOCR()
    strategy = ocr.select_strategy(DocumentQuality.CLEAN_SCAN)
    assert strategy == OCRStrategy.STANDARD_OCR


def test_strategy_for_poor_scan():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality, OCRStrategy

    ocr = AdaptiveOCR()
    strategy = ocr.select_strategy(DocumentQuality.POOR_SCAN)
    assert strategy == OCRStrategy.ENHANCED_OCR


def test_strategy_for_photo():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality, OCRStrategy

    ocr = AdaptiveOCR()
    strategy = ocr.select_strategy(DocumentQuality.PHOTO)
    assert strategy == OCRStrategy.ENHANCED_OCR


def test_extract_text_digital_skips_ocr():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality

    ocr = AdaptiveOCR()
    mock_extractor = MagicMock()
    mock_extractor.return_value = "Invoice #12345"

    result = ocr.extract_text(
        quality=DocumentQuality.DIGITAL,
        pdf_text="Invoice #12345",
        file_path="/tmp/test.pdf",
        ocr_func=mock_extractor,
    )
    assert result == "Invoice #12345"
    mock_extractor.assert_not_called()  # Should not call OCR for digital PDFs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_adaptive_ocr.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement AdaptiveOCR**

```python
# src/services/adaptive_ocr.py
"""Adaptive OCR: detects input document quality and selects optimal OCR strategy.

Layer 1 of the extraction pipeline. Wraps the existing OCR pipeline
(ocr_pipeline.py) with quality detection to avoid unnecessary OCR on
digital PDFs and to apply preprocessing on poor scans.

Spec reference: Section 6, Layer 1 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class DocumentQuality(Enum):
    DIGITAL = "digital"          # Generated PDF, no OCR needed
    CLEAN_SCAN = "clean_scan"    # Good quality scan, standard OCR
    POOR_SCAN = "poor_scan"      # Low DPI or noisy, enhanced OCR
    PHOTO = "photo"              # Camera photo, skewed, needs preprocessing


class OCRStrategy(Enum):
    DIRECT_TEXT = "direct_text"      # Extract text directly from PDF
    STANDARD_OCR = "standard_ocr"    # Standard OCR (EasyOCR/Tesseract)
    ENHANCED_OCR = "enhanced_ocr"    # Preprocessing + OCR (deskew, denoise)


class AdaptiveOCR:
    # DPI thresholds for quality detection
    HIGH_DPI_THRESHOLD = 200
    LOW_DPI_THRESHOLD = 100

    def detect_quality(
        self,
        text_content: str = "",
        page_count: int = 1,
        has_extractable_text: bool = False,
        image_ratio: float = 0.0,
        estimated_dpi: Optional[int] = None,
        is_skewed: bool = False,
    ) -> DocumentQuality:
        # Digital PDF: has extractable text and minimal image content
        if has_extractable_text and len(text_content.strip()) > 20:
            return DocumentQuality.DIGITAL

        # Photo: skewed or very low DPI
        if is_skewed or (estimated_dpi is not None and estimated_dpi < self.LOW_DPI_THRESHOLD):
            return DocumentQuality.PHOTO

        # Poor scan: low DPI but not skewed
        if estimated_dpi is not None and estimated_dpi < self.HIGH_DPI_THRESHOLD:
            return DocumentQuality.POOR_SCAN

        # Clean scan: high DPI, no text
        return DocumentQuality.CLEAN_SCAN

    def select_strategy(self, quality: DocumentQuality) -> OCRStrategy:
        return {
            DocumentQuality.DIGITAL: OCRStrategy.DIRECT_TEXT,
            DocumentQuality.CLEAN_SCAN: OCRStrategy.STANDARD_OCR,
            DocumentQuality.POOR_SCAN: OCRStrategy.ENHANCED_OCR,
            DocumentQuality.PHOTO: OCRStrategy.ENHANCED_OCR,
        }[quality]

    def extract_text(
        self,
        quality: DocumentQuality,
        pdf_text: str = "",
        file_path: str = "",
        ocr_func: Optional[Callable] = None,
    ) -> str:
        strategy = self.select_strategy(quality)

        if strategy == OCRStrategy.DIRECT_TEXT:
            logger.info("Digital PDF detected, using direct text extraction")
            return pdf_text

        if ocr_func is None:
            logger.warning("No OCR function provided, falling back to pdf_text")
            return pdf_text

        if strategy == OCRStrategy.ENHANCED_OCR:
            logger.info("Poor quality detected, applying enhanced OCR with preprocessing")
            # Enhanced OCR uses the same function but caller should preprocess first
            return ocr_func(file_path)

        logger.info("Clean scan detected, using standard OCR")
        return ocr_func(file_path)
```

- [ ] **Step 4: Run tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_adaptive_ocr.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/services/adaptive_ocr.py tests/test_adaptive_ocr.py
git commit -m "feat: add adaptive OCR with quality detection and strategy selection"
```

---

## Task 3: Validation Gate

**Files:**
- Create: `src/services/validation_gate.py`
- Create: `tests/test_validation_gate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_validation_gate.py
"""Tests for the extraction validation gate.

The validation gate scores extraction confidence per-field and per-document,
performs structural and semantic validation, and returns a pass/fail decision.
"""
import pytest
from unittest.mock import MagicMock


def test_structural_validation_passes_complete_invoice():
    from services.validation_gate import ValidationGate

    gate = ValidationGate()
    result = gate.validate_structural(
        doc_type="Invoice",
        extracted_fields={
            "invoice_id": "INV-001",
            "supplier_name": "Acme Corp",
            "invoice_total_incl_tax": "1500.00",
            "invoice_date": "2026-03-15",
        },
    )
    assert result.passed is True
    assert len(result.missing_required) == 0


def test_structural_validation_fails_missing_required():
    from services.validation_gate import ValidationGate

    gate = ValidationGate()
    result = gate.validate_structural(
        doc_type="Invoice",
        extracted_fields={
            "supplier_name": "Acme Corp",
            # Missing invoice_id and invoice_total_incl_tax
        },
    )
    assert result.passed is False
    assert "invoice_id" in result.missing_required


def test_semantic_validation_checks_date_format():
    from services.validation_gate import ValidationGate

    gate = ValidationGate()
    issues = gate.validate_semantic(
        doc_type="Invoice",
        extracted_fields={
            "invoice_id": "INV-001",
            "supplier_name": "Acme Corp",
            "invoice_total_incl_tax": "1500.00",
            "invoice_date": "not-a-date",
        },
    )
    assert any("date" in issue.lower() for issue in issues)


def test_semantic_validation_checks_amount_format():
    from services.validation_gate import ValidationGate

    gate = ValidationGate()
    issues = gate.validate_semantic(
        doc_type="Invoice",
        extracted_fields={
            "invoice_id": "INV-001",
            "supplier_name": "Acme Corp",
            "invoice_total_incl_tax": "not-a-number",
            "invoice_date": "2026-03-15",
        },
    )
    assert any("amount" in issue.lower() or "total" in issue.lower() for issue in issues)


def test_compute_document_confidence():
    from services.validation_gate import ValidationGate

    gate = ValidationGate()
    score = gate.compute_document_confidence(
        field_confidences={
            "invoice_id": 0.95,
            "supplier_name": 0.90,
            "invoice_total_incl_tax": 0.85,
            "invoice_date": 0.92,
        },
        structural_passed=True,
        semantic_issues=[],
    )
    assert 0.85 <= score <= 1.0  # High confidence, no issues


def test_compute_document_confidence_penalizes_issues():
    from services.validation_gate import ValidationGate

    gate = ValidationGate()
    high_score = gate.compute_document_confidence(
        field_confidences={"invoice_id": 0.95, "supplier_name": 0.90},
        structural_passed=True,
        semantic_issues=[],
    )
    low_score = gate.compute_document_confidence(
        field_confidences={"invoice_id": 0.95, "supplier_name": 0.90},
        structural_passed=False,
        semantic_issues=["bad date", "bad amount"],
    )
    assert low_score < high_score


def test_gate_decision_passes_high_confidence():
    from services.validation_gate import ValidationGate

    gate = ValidationGate(confidence_threshold=0.70)
    decision = gate.evaluate(
        doc_type="Invoice",
        extracted_fields={
            "invoice_id": "INV-001",
            "supplier_name": "Acme Corp",
            "invoice_total_incl_tax": "1500.00",
            "invoice_date": "2026-03-15",
        },
        field_confidences={
            "invoice_id": 0.95,
            "supplier_name": 0.90,
            "invoice_total_incl_tax": 0.85,
            "invoice_date": 0.92,
        },
    )
    assert decision.passed is True
    assert decision.confidence_score >= 0.70
    assert decision.needs_remediation is False


def test_gate_decision_routes_to_remediation():
    from services.validation_gate import ValidationGate

    gate = ValidationGate(confidence_threshold=0.70)
    decision = gate.evaluate(
        doc_type="Invoice",
        extracted_fields={
            "supplier_name": "Acme Corp",
            # Missing required fields
        },
        field_confidences={
            "supplier_name": 0.40,
        },
    )
    assert decision.passed is False
    assert decision.needs_remediation is True
    assert len(decision.low_confidence_fields) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_validation_gate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ValidationGate**

```python
# src/services/validation_gate.py
"""Extraction validation gate: scores confidence and blocks bad data.

Layer 3 of the extraction pipeline. Runs between extraction and PostgreSQL
persistence to catch field misclassification, missing data, and format errors.

Spec reference: Section 6, Layer 3 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Required fields per document type (matches utils/procurement_schema.py)
REQUIRED_FIELDS = {
    "Invoice": ["invoice_id", "supplier_name", "invoice_total_incl_tax"],
    "Purchase_Order": ["po_id", "supplier_name"],
    "Quote": ["quote_id", "supplier_name"],
    "Contract": ["contract_id", "supplier_name"],
}

# Fields that should look like amounts
AMOUNT_FIELDS = {
    "invoice_total_incl_tax", "invoice_total_excl_tax", "tax_amount",
    "total_amount", "unit_price", "line_total", "po_total", "quote_total",
    "contract_value", "discount_amount",
}

# Fields that should look like dates
DATE_FIELDS = {
    "invoice_date", "due_date", "po_date", "quote_date", "delivery_date",
    "contract_start_date", "contract_end_date", "expiry_date",
}

# Date patterns that are plausible
_DATE_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}",               # 2026-03-15
    r"\d{2}/\d{2}/\d{4}",               # 15/03/2026
    r"\d{2}-\d{2}-\d{4}",               # 15-03-2026
    r"\d{1,2}\s+\w+\s+\d{4}",           # 15 March 2026
    r"\w+\s+\d{1,2},?\s+\d{4}",         # March 15, 2026
]


@dataclass
class StructuralResult:
    passed: bool
    missing_required: List[str] = field(default_factory=list)
    type_errors: List[str] = field(default_factory=list)


@dataclass
class GateDecision:
    passed: bool
    confidence_score: float
    needs_remediation: bool
    low_confidence_fields: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    semantic_issues: List[str] = field(default_factory=list)


class ValidationGate:
    def __init__(self, confidence_threshold: float = 0.70):
        self._threshold = confidence_threshold

    def validate_structural(
        self, doc_type: str, extracted_fields: Dict[str, Any]
    ) -> StructuralResult:
        required = REQUIRED_FIELDS.get(doc_type, [])
        missing = [f for f in required if f not in extracted_fields or not extracted_fields[f]]
        return StructuralResult(
            passed=len(missing) == 0,
            missing_required=missing,
        )

    def validate_semantic(
        self, doc_type: str, extracted_fields: Dict[str, Any]
    ) -> List[str]:
        issues = []
        for field_name, value in extracted_fields.items():
            if not value:
                continue
            val_str = str(value).strip()

            # Check date fields
            if field_name in DATE_FIELDS:
                if not any(re.search(p, val_str) for p in _DATE_PATTERNS):
                    issues.append(f"Invalid date format in {field_name}: '{val_str}'")

            # Check amount fields
            if field_name in AMOUNT_FIELDS:
                cleaned = re.sub(r"[,$\s£€]", "", val_str)
                try:
                    float(cleaned)
                except ValueError:
                    issues.append(f"Invalid amount format in {field_name}: '{val_str}'")

        return issues

    def compute_document_confidence(
        self,
        field_confidences: Dict[str, float],
        structural_passed: bool,
        semantic_issues: List[str],
    ) -> float:
        if not field_confidences:
            return 0.0

        # Base: weighted average of field confidences
        avg_confidence = sum(field_confidences.values()) / len(field_confidences)

        # Penalty for structural failures
        if not structural_passed:
            avg_confidence *= 0.7

        # Penalty for semantic issues (each issue reduces by 5%, max 25%)
        issue_penalty = min(len(semantic_issues) * 0.05, 0.25)
        avg_confidence -= issue_penalty

        return max(0.0, min(1.0, avg_confidence))

    def evaluate(
        self,
        doc_type: str,
        extracted_fields: Dict[str, Any],
        field_confidences: Dict[str, float],
    ) -> GateDecision:
        # Structural validation
        structural = self.validate_structural(doc_type, extracted_fields)

        # Semantic validation
        semantic_issues = self.validate_semantic(doc_type, extracted_fields)

        # Document-level confidence
        confidence = self.compute_document_confidence(
            field_confidences, structural.passed, semantic_issues,
        )

        # Identify low-confidence fields
        low_conf_fields = [
            f for f, c in field_confidences.items()
            if c < self._threshold
        ]

        # Gate decision
        passed = confidence >= self._threshold and structural.passed
        needs_remediation = not passed

        return GateDecision(
            passed=passed,
            confidence_score=confidence,
            needs_remediation=needs_remediation,
            low_confidence_fields=low_conf_fields + structural.missing_required,
            missing_required=structural.missing_required,
            semantic_issues=semantic_issues,
        )
```

- [ ] **Step 4: Run tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_validation_gate.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/services/validation_gate.py tests/test_validation_gate.py
git commit -m "feat: add extraction validation gate with structural and semantic checks"
```

---

## Task 4: Remediation Service

**Files:**
- Create: `src/services/remediation_service.py`
- Create: `tests/test_remediation_service.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_remediation_service.py
"""Tests for the extraction remediation service.

Remediation re-extracts low-confidence fields using alternative strategies
and cross-references against existing PostgreSQL data.
"""
import pytest
from unittest.mock import MagicMock, patch


def test_remediate_retries_with_llm():
    from services.remediation_service import RemediationService

    mock_llm = MagicMock(return_value={"invoice_id": "INV-001"})
    service = RemediationService(llm_extract_func=mock_llm)

    result = service.remediate_fields(
        low_confidence_fields=["invoice_id"],
        document_text="Invoice Number: INV-001\nTotal: $1500",
        doc_type="Invoice",
        existing_fields={"supplier_name": "Acme"},
    )
    assert "invoice_id" in result.improved_fields
    mock_llm.assert_called_once()


def test_remediate_cross_references_postgres():
    from services.remediation_service import RemediationService

    mock_llm = MagicMock(return_value={})
    mock_db_lookup = MagicMock(return_value={"supplier_name": "Acme Corporation"})
    service = RemediationService(
        llm_extract_func=mock_llm,
        db_lookup_func=mock_db_lookup,
    )

    result = service.remediate_fields(
        low_confidence_fields=["supplier_name"],
        document_text="Supplier: Acme Corp",
        doc_type="Invoice",
        existing_fields={"supplier_name": "Acme Corp"},
    )
    # Should have tried DB cross-reference
    mock_db_lookup.assert_called()


def test_remediate_returns_improved_confidence():
    from services.remediation_service import RemediationService

    mock_llm = MagicMock(return_value={"invoice_id": "INV-999"})
    service = RemediationService(llm_extract_func=mock_llm)

    result = service.remediate_fields(
        low_confidence_fields=["invoice_id"],
        document_text="Invoice #INV-999",
        doc_type="Invoice",
        existing_fields={},
    )
    assert result.improved_confidences.get("invoice_id", 0) > 0


def test_remediate_handles_no_improvement():
    from services.remediation_service import RemediationService

    mock_llm = MagicMock(return_value={})
    service = RemediationService(llm_extract_func=mock_llm)

    result = service.remediate_fields(
        low_confidence_fields=["invoice_id"],
        document_text="Some random text with no invoice info",
        doc_type="Invoice",
        existing_fields={},
    )
    assert "invoice_id" not in result.improved_fields
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_remediation_service.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement RemediationService**

```python
# src/services/remediation_service.py
"""Extraction remediation: re-extracts low-confidence fields.

Layer 4 of the extraction pipeline. For fields below the confidence threshold,
attempts targeted re-extraction using:
1. LLM with focused prompts on specific document regions
2. Cross-reference against existing PostgreSQL data (known suppliers, PO numbers)

Spec reference: Section 6, Layer 4 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Confidence boost when LLM extraction confirms a field
LLM_REMEDIATION_CONFIDENCE = 0.80
# Confidence boost when DB cross-reference matches
DB_MATCH_CONFIDENCE = 0.85


@dataclass
class RemediationResult:
    improved_fields: Dict[str, Any] = field(default_factory=dict)
    improved_confidences: Dict[str, float] = field(default_factory=dict)
    strategies_used: Dict[str, str] = field(default_factory=dict)


class RemediationService:
    def __init__(
        self,
        llm_extract_func: Optional[Callable] = None,
        db_lookup_func: Optional[Callable] = None,
    ):
        self._llm_extract = llm_extract_func
        self._db_lookup = db_lookup_func

    def remediate_fields(
        self,
        low_confidence_fields: List[str],
        document_text: str,
        doc_type: str,
        existing_fields: Dict[str, Any],
    ) -> RemediationResult:
        result = RemediationResult()

        for field_name in low_confidence_fields:
            improved = self._try_llm_extraction(field_name, document_text, doc_type)
            if improved is not None:
                result.improved_fields[field_name] = improved
                result.improved_confidences[field_name] = LLM_REMEDIATION_CONFIDENCE
                result.strategies_used[field_name] = "llm_targeted"
                continue

            db_match = self._try_db_crossref(
                field_name, existing_fields.get(field_name), doc_type,
            )
            if db_match is not None:
                result.improved_fields[field_name] = db_match
                result.improved_confidences[field_name] = DB_MATCH_CONFIDENCE
                result.strategies_used[field_name] = "db_crossref"

        return result

    def _try_llm_extraction(
        self, field_name: str, document_text: str, doc_type: str
    ) -> Optional[Any]:
        if not self._llm_extract:
            return None
        try:
            fields = self._llm_extract(
                document_text=document_text,
                target_fields=[field_name],
                doc_type=doc_type,
            )
            return fields.get(field_name)
        except Exception:
            logger.warning("LLM remediation failed for %s", field_name, exc_info=True)
            return None

    def _try_db_crossref(
        self, field_name: str, current_value: Any, doc_type: str
    ) -> Optional[Any]:
        if not self._db_lookup or not current_value:
            return None
        try:
            matches = self._db_lookup(
                field_name=field_name,
                partial_value=str(current_value),
                doc_type=doc_type,
            )
            if matches:
                return matches.get(field_name)
        except Exception:
            logger.warning("DB cross-ref failed for %s", field_name, exc_info=True)
        return None
```

- [ ] **Step 4: Run tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_remediation_service.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/services/remediation_service.py tests/test_remediation_service.py
git commit -m "feat: add extraction remediation service for low-confidence fields"
```

---

## Task 5: Integrate Validation Gate into DataExtractionAgent

**Files:**
- Modify: `src/agents/data_extraction_agent.py`

- [ ] **Step 1: Read the current _process_single_document and _persist_to_postgres methods**

Read: `src/agents/data_extraction_agent.py` — find `_process_single_document` and `_persist_to_postgres` methods to understand where validation gate should be inserted.

- [ ] **Step 2: Add validation gate integration**

In `_process_single_document()`, after extraction is complete and before `_persist_to_postgres()`, add the validation gate:

```python
# Add import at top of file:
# from services.validation_gate import ValidationGate
# from services.remediation_service import RemediationService

# In _process_single_document(), after extraction results are available
# and before persistence, add:

# --- Validation Gate ---
gate = ValidationGate()
field_confidences = record.get("field_confidences", {})
decision = gate.evaluate(
    doc_type=doc_type,
    extracted_fields=record,
    field_confidences=field_confidences,
)

confidence_score = decision.confidence_score
needs_review = False

if decision.needs_remediation:
    # Attempt remediation for low-confidence fields
    remediation = RemediationService()
    remediation_result = remediation.remediate_fields(
        low_confidence_fields=decision.low_confidence_fields,
        document_text=text,
        doc_type=doc_type,
        existing_fields=record,
    )
    # Merge improved fields back
    if remediation_result.improved_fields:
        record.update(remediation_result.improved_fields)
        field_confidences.update(remediation_result.improved_confidences)
        # Re-evaluate after remediation
        decision = gate.evaluate(doc_type, record, field_confidences)
        confidence_score = decision.confidence_score

    if not decision.passed:
        needs_review = True
        logger.warning(
            "Document %s: low confidence (%.2f), flagged for review",
            record.get("invoice_id") or record.get("po_id") or "unknown",
            confidence_score,
        )

# Add confidence_score and needs_review to record before persistence
record["confidence_score"] = confidence_score
record["needs_review"] = needs_review
# --- End Validation Gate ---
```

The exact insertion point depends on the method structure — read it first, then insert between extraction and persistence.

- [ ] **Step 3: Verify existing extraction tests still pass**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_data_extraction_agent.py -v 2>&1 | tail -20`
Expected: No new failures

- [ ] **Step 4: Commit**

```bash
git add src/agents/data_extraction_agent.py
git commit -m "feat: integrate validation gate and remediation into extraction pipeline"
```

---

## Task 6: Update Extraction Workflow Definition

**Files:**
- Modify: `src/orchestration/workflow_definitions.py`

- [ ] **Step 1: Read the current build_extraction_workflow**

Read: `src/orchestration/workflow_definitions.py:66-84`

- [ ] **Step 2: The extraction workflow stays as a single node**

The validation gate and remediation are integrated INSIDE the DataExtractionAgent (Task 5), not as separate DAG nodes. This is because:
- The existing agent processes documents in a tight loop (download → extract → validate → persist)
- Breaking this into DAG nodes would require serializing binary file content between nodes
- The agent already has per-document threading internally

No change needed to `build_extraction_workflow()`. The single `extract_documents` node now internally runs: OCR → extraction → validation gate → remediation → persistence.

- [ ] **Step 3: Commit** (skip if no changes)

No commit needed — workflow definition unchanged.

---

## Task 7: Run Full Extraction Test Suite

- [ ] **Step 1: Run all new tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_adaptive_ocr.py tests/test_validation_gate.py tests/test_remediation_service.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run existing extraction tests to verify no regressions**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_data_extraction_agent.py tests/test_extraction.py tests/test_data_extraction_numeric.py -v 2>&1 | tail -20`
Expected: No new failures

- [ ] **Step 3: Run full orchestration test suite**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_message_protocol.py tests/test_state_manager.py tests/test_task_dispatcher.py tests/test_dag_scheduler.py tests/test_result_collector.py tests/test_worker.py tests/test_integration_workflow.py tests/test_adaptive_ocr.py tests/test_validation_gate.py tests/test_remediation_service.py -v`
Expected: All tests PASS
