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
