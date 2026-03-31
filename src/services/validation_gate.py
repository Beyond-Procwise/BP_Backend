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
