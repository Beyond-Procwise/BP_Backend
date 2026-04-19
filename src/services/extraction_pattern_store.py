"""Extraction Pattern Store — learn from successful extractions.

When a document is extracted successfully (high confidence, verified),
the pattern of how it was structured is captured:
- File type (xlsx, pdf, docx, jpeg)
- Document layout signature (number of metadata rows, header position,
  column names, totals structure)
- Supplier name pattern
- Column-to-schema mapping that worked

When a new document arrives, its structure is compared against known
patterns. If a match is found, the proven column mapping and extraction
hints are injected into the LLM prompt for higher accuracy.

Storage: proc.bp_extraction_patterns table.
"""

import json
import logging
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExtractionPatternStore:
    """Learns and reuses document extraction patterns."""

    def __init__(self, get_db_connection):
        self._get_conn = get_db_connection
        self._ensure_table()
        self._patterns: List[Dict[str, Any]] = []
        self._load_patterns()

    def _ensure_table(self):
        try:
            conn = self._get_conn()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS proc.bp_extraction_patterns (
                        pattern_id      SERIAL PRIMARY KEY,
                        file_type       TEXT NOT NULL,
                        doc_type        TEXT NOT NULL,
                        supplier_name   TEXT,
                        layout_signature TEXT NOT NULL,
                        column_mapping  JSONB,
                        extraction_hints TEXT,
                        success_count   INTEGER DEFAULT 1,
                        last_used       TIMESTAMP DEFAULT NOW(),
                        created_date    TIMESTAMP DEFAULT NOW()
                    )
                """)
            conn.close()
        except Exception:
            logger.warning("Could not create extraction_patterns table", exc_info=True)

    def _load_patterns(self):
        try:
            conn = self._get_conn()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT pattern_id, file_type, doc_type, supplier_name,
                           layout_signature, column_mapping, extraction_hints,
                           success_count
                    FROM proc.bp_extraction_patterns
                    ORDER BY success_count DESC
                """)
                self._patterns = []
                for row in cur.fetchall():
                    self._patterns.append({
                        "pattern_id": row[0],
                        "file_type": row[1],
                        "doc_type": row[2],
                        "supplier_name": row[3],
                        "layout_signature": row[4],
                        "column_mapping": row[5],
                        "extraction_hints": row[6],
                        "success_count": row[7],
                    })
            conn.close()
            logger.info(
                "Loaded %d extraction patterns", len(self._patterns)
            )
        except Exception:
            logger.warning("Could not load extraction patterns", exc_info=True)

    @staticmethod
    def compute_layout_signature(doc_structure) -> str:
        """Compute a layout signature from a DocumentStructure.

        The signature captures the structural shape of the document
        without its content — how many metadata entries, table column
        count, header names, totals labels. Two documents with the
        same signature can use the same extraction strategy.
        """
        parts = []
        parts.append(f"meta:{len(doc_structure.metadata)}")

        for i, table in enumerate(doc_structure.tables):
            headers = table.get("headers", [])
            row_count = len(table.get("rows", []))
            # Normalize header names (lowercase, strip symbols)
            norm_headers = [
                re.sub(r"[^a-z0-9\s]", "", h.lower()).strip()
                for h in headers if h
            ]
            parts.append(
                f"table{i}:cols={len(headers)},rows={row_count},"
                f"headers=[{','.join(norm_headers)}]"
            )

        totals_labels = sorted(doc_structure.totals.keys())
        if totals_labels:
            parts.append(f"totals:[{','.join(totals_labels)}]")

        return " | ".join(parts)

    def find_matching_pattern(
        self, doc_structure, file_type: str, doc_type: str
    ) -> Optional[Dict[str, Any]]:
        """Find a previously successful pattern that matches this document.

        Returns the pattern dict with column_mapping and extraction_hints
        if a match is found, None otherwise.
        """
        if not self._patterns:
            return None

        signature = self.compute_layout_signature(doc_structure)

        best_match = None
        best_score = 0.0

        for pattern in self._patterns:
            if pattern["file_type"] != file_type:
                continue
            if pattern["doc_type"] != doc_type:
                continue

            # Compare layout signatures
            ratio = SequenceMatcher(
                None, signature, pattern["layout_signature"]
            ).ratio()

            # Boost score if supplier matches
            if pattern.get("supplier_name"):
                for meta in doc_structure.metadata:
                    val = meta.get("value", "")
                    if (
                        pattern["supplier_name"].lower() in val.lower()
                        or val.lower() in pattern["supplier_name"].lower()
                    ):
                        ratio += 0.1
                        break

            if ratio > best_score:
                best_score = ratio
                best_match = pattern

        if best_score >= 0.75 and best_match:
            logger.info(
                "Matched extraction pattern #%d (score=%.2f, supplier=%s, used %d times)",
                best_match["pattern_id"],
                best_score,
                best_match.get("supplier_name", "unknown"),
                best_match["success_count"],
            )
            # Update last_used
            try:
                conn = self._get_conn()
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE proc.bp_extraction_patterns "
                        "SET last_used = %s, success_count = success_count + 1 "
                        "WHERE pattern_id = %s",
                        (datetime.now(timezone.utc), best_match["pattern_id"]),
                    )
                conn.close()
            except Exception:
                pass
            return best_match

        return None

    def capture_pattern(
        self,
        doc_structure,
        file_type: str,
        doc_type: str,
        supplier_name: Optional[str],
        column_mapping: Optional[Dict],
        extraction_hints: Optional[str] = None,
    ) -> None:
        """Capture a successful extraction pattern for future reuse."""
        signature = self.compute_layout_signature(doc_structure)

        # Check if this signature already exists
        for pattern in self._patterns:
            if (
                pattern["file_type"] == file_type
                and pattern["doc_type"] == doc_type
                and SequenceMatcher(
                    None, signature, pattern["layout_signature"]
                ).ratio()
                > 0.90
            ):
                # Update existing pattern
                try:
                    conn = self._get_conn()
                    conn.autocommit = True
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE proc.bp_extraction_patterns "
                            "SET success_count = success_count + 1, "
                            "    last_used = %s, "
                            "    column_mapping = COALESCE(%s, column_mapping) "
                            "WHERE pattern_id = %s",
                            (
                                datetime.now(timezone.utc),
                                json.dumps(column_mapping) if column_mapping else None,
                                pattern["pattern_id"],
                            ),
                        )
                    conn.close()
                    pattern["success_count"] += 1
                    logger.debug(
                        "Updated extraction pattern #%d (count=%d)",
                        pattern["pattern_id"],
                        pattern["success_count"],
                    )
                except Exception:
                    logger.debug("Failed to update pattern", exc_info=True)
                return

        # Insert new pattern
        try:
            conn = self._get_conn()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO proc.bp_extraction_patterns
                    (file_type, doc_type, supplier_name, layout_signature,
                     column_mapping, extraction_hints)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING pattern_id""",
                    (
                        file_type,
                        doc_type,
                        supplier_name,
                        signature,
                        json.dumps(column_mapping) if column_mapping else None,
                        extraction_hints,
                    ),
                )
                pid = cur.fetchone()[0]
            conn.close()
            self._patterns.append({
                "pattern_id": pid,
                "file_type": file_type,
                "doc_type": doc_type,
                "supplier_name": supplier_name,
                "layout_signature": signature,
                "column_mapping": column_mapping,
                "extraction_hints": extraction_hints,
                "success_count": 1,
            })
            logger.info(
                "Captured new extraction pattern #%d: %s %s supplier=%s",
                pid, file_type, doc_type, supplier_name,
            )
        except Exception:
            logger.warning("Failed to capture extraction pattern", exc_info=True)
