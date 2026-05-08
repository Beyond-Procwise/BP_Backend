"""Postgres-backed TemplateStore implementation.

Persists vendor extraction templates so they survive restarts and can be
shared across worker processes. Schema:

    proc.bp_extraction_template (
        fingerprint        TEXT PRIMARY KEY,
        vendor_name        TEXT,
        doc_type           TEXT NOT NULL,
        field_hints        JSONB NOT NULL DEFAULT '{}'::jsonb,
        line_item_hints    JSONB,
        success_count      INTEGER NOT NULL DEFAULT 0,
        correction_count   INTEGER NOT NULL DEFAULT 0,
        created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        last_used_at       TIMESTAMPTZ
    )

The `field_hints` JSONB stores `{field_name: {value, confidence, label, anchor}}`.
The `line_item_hints` JSONB stores `{header_anchors: [...], column_map: {...},
expected_min_rows: int}` when present.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.services.extraction_v2.template_store import (
    FieldHint, LineItemHints, TemplateStore, VendorTemplate,
)

logger = logging.getLogger(__name__)


DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.bp_extraction_template (
    fingerprint      TEXT PRIMARY KEY,
    vendor_name      TEXT,
    doc_type         TEXT NOT NULL,
    field_hints      JSONB NOT NULL DEFAULT '{}'::jsonb,
    line_item_hints  JSONB,
    success_count    INTEGER NOT NULL DEFAULT 0,
    correction_count INTEGER NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_bp_extraction_template_vendor
    ON proc.bp_extraction_template (vendor_name);

CREATE INDEX IF NOT EXISTS idx_bp_extraction_template_doc_type
    ON proc.bp_extraction_template (doc_type);
"""


def _hint_to_jsonable(hint: FieldHint) -> dict:
    return {
        "value": hint.value,
        "confidence": hint.confidence,
        "label": hint.label,
        "anchor": hint.anchor,
    }


def _hint_from_jsonable(field_name: str, raw: dict) -> FieldHint:
    return FieldHint(
        field=field_name,
        value=raw.get("value"),
        confidence=float(raw.get("confidence", 0.95)),
        label=raw.get("label"),
        anchor=raw.get("anchor"),
    )


def _line_item_hints_to_jsonable(hints: Optional[LineItemHints]) -> Optional[dict]:
    if hints is None:
        return None
    return {
        "header_anchors": list(hints.header_anchors),
        "column_map": dict(hints.column_map),
        "expected_min_rows": int(hints.expected_min_rows),
    }


def _line_item_hints_from_jsonable(raw: Optional[dict]) -> Optional[LineItemHints]:
    if not raw:
        return None
    return LineItemHints(
        header_anchors=list(raw.get("header_anchors", [])),
        column_map=dict(raw.get("column_map", {})),
        expected_min_rows=int(raw.get("expected_min_rows", 1)),
    )


class PostgresTemplateStore(TemplateStore):
    """Postgres-backed implementation of TemplateStore.

    The store opens a fresh connection per operation via the supplied
    connection factory — matches the pattern used by other repositories
    in this codebase (workflow_email_tracking_repo, etc.) and avoids
    long-held cursors that would otherwise serialise concurrent
    extractions on a single connection.
    """

    def __init__(self, get_conn):
        """`get_conn` is a callable returning a context-manager Postgres
        connection (psycopg2)."""
        self._get_conn = get_conn
        self.init_schema()

    def init_schema(self) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(DDL)
            conn.commit()

    def get(self, fingerprint: str) -> Optional[VendorTemplate]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT fingerprint, vendor_name, doc_type, field_hints,
                              line_item_hints, success_count, correction_count,
                              created_at, last_used_at
                         FROM proc.bp_extraction_template
                        WHERE fingerprint = %s""",
                    (fingerprint,),
                )
                row = cur.fetchone()
        if not row:
            return None
        (fp, vendor, doc_type, field_hints_raw, line_hints_raw,
         success, corrections, created_at, last_used) = row
        if isinstance(field_hints_raw, str):
            field_hints_raw = json.loads(field_hints_raw)
        if isinstance(line_hints_raw, str):
            line_hints_raw = json.loads(line_hints_raw)
        hints = {
            fname: _hint_from_jsonable(fname, raw)
            for fname, raw in (field_hints_raw or {}).items()
        }
        return VendorTemplate(
            fingerprint=fp,
            vendor_name=vendor,
            doc_type=doc_type,
            field_hints=hints,
            line_item_hints=_line_item_hints_from_jsonable(line_hints_raw),
            success_count=int(success or 0),
            correction_count=int(corrections or 0),
            created_at=created_at or datetime.now(timezone.utc),
            last_used_at=last_used,
        )

    def upsert(self, template: VendorTemplate) -> None:
        hints_json = json.dumps(
            {f: _hint_to_jsonable(h) for f, h in template.field_hints.items()},
            default=str,
        )
        line_hints_json = json.dumps(
            _line_item_hints_to_jsonable(template.line_item_hints),
            default=str,
        ) if template.line_item_hints is not None else None
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO proc.bp_extraction_template
                            (fingerprint, vendor_name, doc_type, field_hints,
                             line_item_hints, success_count, correction_count,
                             created_at, last_used_at)
                       VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s,
                               COALESCE(%s, NOW()), %s)
                       ON CONFLICT (fingerprint) DO UPDATE SET
                            vendor_name      = EXCLUDED.vendor_name,
                            doc_type         = EXCLUDED.doc_type,
                            field_hints      = EXCLUDED.field_hints,
                            line_item_hints  = EXCLUDED.line_item_hints,
                            success_count    = EXCLUDED.success_count,
                            correction_count = EXCLUDED.correction_count,
                            last_used_at     = EXCLUDED.last_used_at""",
                    (
                        template.fingerprint, template.vendor_name,
                        template.doc_type, hints_json, line_hints_json,
                        template.success_count, template.correction_count,
                        template.created_at, template.last_used_at,
                    ),
                )
            conn.commit()

    def record_success(self, fingerprint: str,
                       fields_committed: Iterable[str]) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE proc.bp_extraction_template
                          SET success_count = success_count + 1,
                              last_used_at  = NOW()
                        WHERE fingerprint = %s""",
                    (fingerprint,),
                )
            conn.commit()

    def record_correction(self, fingerprint: str, field: str, value: Any,
                          confidence: float = 0.95,
                          label: Optional[str] = None,
                          anchor: Optional[dict] = None,
                          doc_type: Optional[str] = None,
                          vendor_name: Optional[str] = None) -> None:
        existing = self.get(fingerprint)
        if existing is None:
            existing = VendorTemplate(
                fingerprint=fingerprint,
                vendor_name=vendor_name,
                doc_type=doc_type or "Unknown",
                field_hints={},
            )
        existing.field_hints[field] = FieldHint(
            field=field, value=value, confidence=confidence,
            label=label, anchor=anchor,
        )
        existing.correction_count += 1
        # Honour any caller-provided override (vendor onboarding API).
        if vendor_name is not None:
            existing.vendor_name = vendor_name
        if doc_type is not None:
            existing.doc_type = doc_type
        self.upsert(existing)

    def record_line_item_hints(self, fingerprint: str,
                               hints: LineItemHints,
                               doc_type: Optional[str] = None,
                               vendor_name: Optional[str] = None) -> None:
        """Record/replace the line-items hints for a fingerprint.

        Used by both auto-learning (pipeline detected a successful
        line-items extraction and snapshots the column layout) and by
        manual onboarding through the vendors API.
        """
        existing = self.get(fingerprint)
        if existing is None:
            existing = VendorTemplate(
                fingerprint=fingerprint,
                vendor_name=vendor_name,
                doc_type=doc_type or "Unknown",
                field_hints={},
            )
        existing.line_item_hints = hints
        if vendor_name is not None:
            existing.vendor_name = vendor_name
        if doc_type is not None:
            existing.doc_type = doc_type
        self.upsert(existing)
