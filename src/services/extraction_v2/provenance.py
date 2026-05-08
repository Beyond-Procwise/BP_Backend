"""Per-field extraction provenance.

Records WHERE each persisted field came from (LLM, structural extractor,
filename rescue, vendor template, sanitizer) so downstream consumers can
weight values by trustworthiness and so confidence scores can be
recomputed from real signals rather than a hard-coded 0.85.

Reuses the existing ``proc.bp_extraction_provenance`` table populated by
the structural extractor's row-level provenance writer. The shared
schema:

    parent_table     TEXT (e.g. 'bp_invoice')
    parent_pk        TEXT (e.g. 'INV4759276')
    field_name       TEXT
    source           TEXT (llm | structural | filename | template | sanitizer)
    anchor_ref       JSONB
    derivation_trace JSONB  -- {"value": ..., "raw_value": ...} for our writer
    confidence       NUMERIC
    attempt          INTEGER
    extracted_at     TIMESTAMPTZ

The orchestrator calls :func:`record_field_provenance` for every field
it commits to a bp_* table. The data is queryable directly:

    SELECT field_name, source FROM proc.bp_extraction_provenance
     WHERE parent_pk = 'INV4759276';
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


__all__ = [
    "DDL", "record_field_provenance", "record_extraction_provenance",
    "PARENT_TABLE_FOR_DOC_TYPE",
]


# Schema-extension DDL — adds idempotent indexes on the existing table.
# We do NOT redefine the table itself (the structural extractor's
# row-level writer owns its base shape).
DDL = """
CREATE INDEX IF NOT EXISTS idx_bp_extraction_provenance_pk
    ON proc.bp_extraction_provenance (parent_pk);

CREATE INDEX IF NOT EXISTS idx_bp_extraction_provenance_source
    ON proc.bp_extraction_provenance (source);
"""


# Doc-type → bp_ table name. Mirrors BP_TABLES in
# agent_nick_orchestrator.py; kept local so the provenance module
# doesn't pull in the orchestrator import graph.
PARENT_TABLE_FOR_DOC_TYPE = {
    "Invoice": "bp_invoice",
    "Purchase_Order": "bp_purchase_order",
    "Quote": "bp_quote",
    "Contract": "bp_contracts",
}


# Allowed source values — any other source must be remapped to one of these.
# `template`   = vendor template hint applied
# `filename`   = rescued from filename
# `sanitizer`  = nulled or normalised by ExtractionSanitizer
# `structural` = produced by the structural extractor's deterministic path
# `llm`        = produced by the legacy LLM (default)
ALLOWED_SOURCES = {"llm", "structural", "filename", "template", "sanitizer", "manual"}


def record_field_provenance(
    get_conn,
    *,
    record_id: str,
    doc_type: str,
    field_name: str,
    value: Any,
    source: str,
    confidence: Optional[float] = None,
    raw_value: Any = None,
    evidence: Optional[dict] = None,
) -> None:
    """Insert one provenance row.

    Maps to the existing ``proc.bp_extraction_provenance`` shape:
      - ``parent_table``     ← derived from ``doc_type``
      - ``parent_pk``        ← ``record_id``
      - ``derivation_trace`` ← ``{"value": ..., "raw_value": ...}``
      - ``anchor_ref``       ← ``evidence``
      - ``attempt``          ← always 1 (this writer is post-extract)
    """
    if source not in ALLOWED_SOURCES:
        logger.warning(
            "[provenance] unrecognised source=%r for field=%s — coercing to 'llm'",
            source, field_name,
        )
        source = "llm"
    parent_table = PARENT_TABLE_FOR_DOC_TYPE.get(doc_type, f"bp_{doc_type.lower()}")
    derivation_trace = {
        "value": None if value is None else str(value),
    }
    if raw_value is not None:
        derivation_trace["raw_value"] = str(raw_value)
    anchor_ref_json = json.dumps(evidence, default=str) if evidence else None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO proc.bp_extraction_provenance
                        (parent_table, parent_pk, field_name, source,
                         anchor_ref, derivation_trace, confidence, attempt)
                   VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, 1)""",
                (
                    parent_table, record_id, field_name, source,
                    anchor_ref_json,
                    json.dumps(derivation_trace, default=str),
                    confidence,
                ),
            )
        conn.commit()


def record_extraction_provenance(
    get_conn,
    *,
    record_id: str,
    doc_type: str,
    header: dict,
    rescued_fields: Iterable[str] = (),
    template_overrides: Iterable[str] = (),
    sanitizer_rejections: Iterable = (),
    default_source: str = "llm",
) -> int:
    """Bulk-record provenance for every field in ``header``.

    Source resolution per field, in priority order:
        1. ``template_overrides`` → ``template``
        2. ``rescued_fields``     → ``filename``
        3. otherwise              → ``default_source`` (typically ``llm``
           for legacy path or ``structural`` when the structural extractor
           produced the value).

    ``sanitizer_rejections`` (an iterable of records emitted by
    :class:`ExtractionSanitizer`) is used to override the source to
    ``sanitizer`` for any field that was sanitized and to populate
    ``raw_value`` with the rejected pre-sanitization value.

    Returns the number of rows written.
    """
    template_overrides = set(template_overrides)
    rescued_fields = set(rescued_fields)
    sanitizer_by_field: dict[str, Any] = {}
    for rejection in sanitizer_rejections:
        # Rejections are SimpleNamespace-style objects; tolerate dicts too.
        field_name = getattr(rejection, "field", None) or (
            rejection.get("field") if isinstance(rejection, dict) else None
        )
        if not field_name:
            continue
        raw_val = getattr(rejection, "raw_value", None) or (
            rejection.get("raw_value") if isinstance(rejection, dict) else None
        )
        sanitizer_by_field[field_name] = raw_val

    rows_written = 0
    for field_name, value in header.items():
        if field_name.startswith("_") or field_name in {
            "created_date", "created_by", "last_modified_date", "last_modified_by",
        }:
            continue
        if field_name in template_overrides:
            source = "template"
        elif field_name in rescued_fields:
            source = "filename"
        elif field_name in sanitizer_by_field:
            source = "sanitizer"
        else:
            source = default_source
        try:
            record_field_provenance(
                get_conn,
                record_id=record_id, doc_type=doc_type,
                field_name=field_name, value=value, source=source,
                raw_value=sanitizer_by_field.get(field_name),
            )
            rows_written += 1
        except Exception as exc:
            logger.warning(
                "[provenance] write failed for record=%s field=%s: %s "
                "(non-blocking)", record_id, field_name, exc,
            )
    return rows_written
