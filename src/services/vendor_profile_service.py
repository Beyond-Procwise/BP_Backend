"""Vendor extraction profile service.

Auto-learns vendor-specific extraction patterns (date formats, currency,
label overrides) from successful extractions and applies them to future
documents from the same vendor.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VendorProfile:
    """Represents a learned vendor extraction profile."""

    __slots__ = (
        "profile_id", "supplier_id", "supplier_name", "doc_type",
        "date_format_hint", "currency_hint", "label_overrides",
        "field_positions", "extraction_count", "last_extracted_at",
    )

    def __init__(
        self,
        *,
        profile_id: int = 0,
        supplier_id: str = "",
        supplier_name: str = "",
        doc_type: str = "",
        date_format_hint: str = "",
        currency_hint: str = "",
        label_overrides: Optional[Dict[str, List[str]]] = None,
        field_positions: Optional[Dict[str, Any]] = None,
        extraction_count: int = 0,
        last_extracted_at: Optional[datetime] = None,
    ) -> None:
        self.profile_id = profile_id
        self.supplier_id = supplier_id
        self.supplier_name = supplier_name
        self.doc_type = doc_type
        self.date_format_hint = date_format_hint
        self.currency_hint = currency_hint
        self.label_overrides = label_overrides or {}
        self.field_positions = field_positions or {}
        self.extraction_count = extraction_count
        self.last_extracted_at = last_extracted_at


class VendorProfileService:
    """Manages vendor extraction profiles in proc.bp_vendor_extraction_profiles."""

    def __init__(self, agent_nick) -> None:
        self._agent_nick = agent_nick
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the profiles table if it doesn't exist."""
        sql = """
        CREATE TABLE IF NOT EXISTS proc.bp_vendor_extraction_profiles (
            profile_id SERIAL PRIMARY KEY,
            supplier_id TEXT,
            supplier_name TEXT NOT NULL,
            doc_type TEXT NOT NULL,
            date_format_hint TEXT,
            currency_hint VARCHAR(5),
            label_overrides JSONB DEFAULT '{}',
            field_positions JSONB DEFAULT '{}',
            extraction_count INTEGER DEFAULT 1,
            last_extracted_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(supplier_name, doc_type)
        );
        """
        try:
            conn = self._agent_nick.get_db_connection()
            try:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(sql)
            finally:
                conn.close()
        except Exception:
            logger.exception("Failed to ensure bp_vendor_extraction_profiles table")

    def get_profile(
        self, supplier_name: str, doc_type: str
    ) -> Optional[VendorProfile]:
        """Look up a vendor profile by supplier name and document type."""
        if not supplier_name or not doc_type:
            return None
        try:
            conn = self._agent_nick.get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT profile_id, supplier_id, supplier_name, doc_type,
                               date_format_hint, currency_hint, label_overrides,
                               field_positions, extraction_count, last_extracted_at
                        FROM proc.bp_vendor_extraction_profiles
                        WHERE LOWER(supplier_name) = LOWER(%s) AND doc_type = %s
                        """,
                        (supplier_name.strip(), doc_type),
                    )
                    row = cur.fetchone()
                    if row is None:
                        return None
                    return VendorProfile(
                        profile_id=row[0],
                        supplier_id=row[1] or "",
                        supplier_name=row[2] or "",
                        doc_type=row[3] or "",
                        date_format_hint=row[4] or "",
                        currency_hint=row[5] or "",
                        label_overrides=row[6] if isinstance(row[6], dict) else {},
                        field_positions=row[7] if isinstance(row[7], dict) else {},
                        extraction_count=row[8] or 0,
                        last_extracted_at=row[9],
                    )
            finally:
                conn.close()
        except Exception:
            logger.exception(
                "Failed to get vendor profile for '%s' doc_type='%s'",
                supplier_name, doc_type,
            )
            return None

    def learn_from_extraction(
        self,
        *,
        supplier_name: str,
        supplier_id: str = "",
        doc_type: str,
        date_format_hint: str = "",
        currency_hint: str = "",
        label_overrides: Optional[Dict[str, List[str]]] = None,
        field_positions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record or update a vendor profile from a successful extraction.

        Uses UPSERT: if a profile already exists for (supplier_name, doc_type),
        it increments extraction_count and merges new label_overrides.
        """
        if not supplier_name or not doc_type:
            return
        label_json = json.dumps(label_overrides or {})
        field_json = json.dumps(field_positions or {})
        try:
            conn = self._agent_nick.get_db_connection()
            try:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.bp_vendor_extraction_profiles
                            (supplier_id, supplier_name, doc_type,
                             date_format_hint, currency_hint,
                             label_overrides, field_positions,
                             extraction_count, last_extracted_at)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, 1, %s)
                        ON CONFLICT (supplier_name, doc_type) DO UPDATE SET
                            supplier_id = COALESCE(NULLIF(EXCLUDED.supplier_id, ''), proc.bp_vendor_extraction_profiles.supplier_id),
                            date_format_hint = COALESCE(NULLIF(EXCLUDED.date_format_hint, ''), proc.bp_vendor_extraction_profiles.date_format_hint),
                            currency_hint = COALESCE(NULLIF(EXCLUDED.currency_hint, ''), proc.bp_vendor_extraction_profiles.currency_hint),
                            label_overrides = proc.bp_vendor_extraction_profiles.label_overrides || EXCLUDED.label_overrides,
                            field_positions = proc.bp_vendor_extraction_profiles.field_positions || EXCLUDED.field_positions,
                            extraction_count = proc.bp_vendor_extraction_profiles.extraction_count + 1,
                            last_extracted_at = EXCLUDED.last_extracted_at
                        """,
                        (
                            supplier_id,
                            supplier_name.strip(),
                            doc_type,
                            date_format_hint,
                            currency_hint,
                            label_json,
                            field_json,
                            datetime.now(timezone.utc),
                        ),
                    )
            finally:
                conn.close()
            logger.info(
                "Vendor profile updated: supplier='%s' doc_type='%s'",
                supplier_name, doc_type,
            )
        except Exception:
            logger.exception(
                "Failed to learn vendor profile for '%s' doc_type='%s'",
                supplier_name, doc_type,
            )
