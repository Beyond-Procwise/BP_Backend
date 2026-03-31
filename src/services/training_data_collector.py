# src/services/training_data_collector.py
"""Collects extraction corrections and agent outputs as training data.

Part of the self-improving loop: remediation corrections become training
examples for the next fine-tuning cycle.

Spec reference: Section 8 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    def __init__(self, output_dir: str = "data/training"):
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def record_correction(
        self,
        doc_type: str,
        document_text: str,
        original_fields: Dict[str, Any],
        corrected_fields: Dict[str, Any],
        correction_source: str = "remediation",
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "doc_type": doc_type,
            "document_text": document_text[:2000],  # Truncate for storage
            "original_fields": original_fields,
            "corrected_fields": corrected_fields,
            "correction_source": correction_source,
        }
        self._append_jsonl("extraction_corrections.jsonl", entry)
        logger.debug("Recorded extraction correction for %s", doc_type)

    def record_negotiation(
        self,
        workflow_id: str,
        supplier_name: str,
        round_num: int,
        strategy: str,
        counter_offer: str,
        outcome: str,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workflow_id": workflow_id,
            "supplier_name": supplier_name,
            "round": round_num,
            "strategy": strategy,
            "counter_offer": counter_offer,
            "outcome": outcome,
        }
        self._append_jsonl("negotiation_examples.jsonl", entry)
        logger.debug("Recorded negotiation example for %s round %d", supplier_name, round_num)

    def get_stats(self) -> Dict[str, int]:
        stats = {}
        for filename, key in [
            ("extraction_corrections.jsonl", "extraction_corrections"),
            ("negotiation_examples.jsonl", "negotiation_examples"),
        ]:
            path = os.path.join(self._output_dir, filename)
            if os.path.exists(path):
                with open(path) as f:
                    stats[key] = sum(1 for _ in f)
            else:
                stats[key] = 0
        return stats

    def _append_jsonl(self, filename: str, entry: Dict) -> None:
        path = os.path.join(self._output_dir, filename)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
