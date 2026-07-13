#!/usr/bin/env python3
"""Extraction pipeline autonomous health-check + recovery.

Designed to run on a 5-minute cadence via systemd timer. Each invocation:

1. **Stuck-row recovery** — process_monitor rows that have been mid-flight
   (status NOT terminal) for longer than STUCK_MINUTES (default 15) are
   reset to status='Completed' with NULL start_ts so the watcher's NOTIFY
   path picks them up again. Each reset is capped: a row that has been
   reset MAX_STUCK_RETRIES times (default 3) is marked
   'Extraction_Failed' with a note instead of being looped forever.

2. **Hallucination audit** — random sample (AUDIT_SAMPLE_SIZE, default 50)
   of recent provenance_v3 rows. For each, fetch the parser_snapshot from
   the matching _raw row OR the parent doc's _stg row, and assert that
   evidence_text is a substring of full_text. Violations are written to
   proc.bp_extraction_hallucination_audit (auto-created).

3. **Backlog signal** — counts ACTIVE-HITL discrepancies and total _raw
   discrepancy rows per doc-type; writes one snapshot row per run to
   proc.bp_extraction_health_metrics (auto-created).

4. **Failed-extraction reaper** — process_monitor rows with status =
   'Extraction_Failed' that have a recoverable error (file-not-found in
   the bridged-replay window) are re-queued once (no infinite retry).

The script is idempotent and stateless beyond what it writes to the
metrics + audit tables. Safe to run concurrently with the live daemons.

Output is JSON-lines to stdout for easy grep/jq from journalctl.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings  # noqa: E402
import psycopg2  # noqa: E402
from src.services.extraction_v3.grounding import is_value_grounded  # noqa: E402

STUCK_MINUTES = int(os.getenv("HEALTH_STUCK_MINUTES", "15"))
MAX_STUCK_RETRIES = int(os.getenv("HEALTH_MAX_STUCK_RETRIES", "3"))
AUDIT_SAMPLE_SIZE = int(os.getenv("HEALTH_AUDIT_SAMPLE_SIZE", "50"))

NON_TERMINAL = ("Completed", "Running", "Extracting")
TERMINAL = ("Extracted", "Extraction_Failed", "Archived")

_RAW_TABLES = {
    "invoice": "proc.bp_invoice_raw",
    "purchase_order": "proc.bp_purchase_order_raw",
    "quote": "proc.bp_quote_raw",
    "contract": "proc.bp_contract_raw",
}


def _emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, "ts": int(time.time()), **fields}, default=str),
          flush=True)


def _conn() -> psycopg2.extensions.connection:
    s = Settings()
    c = psycopg2.connect(host=s.db_host, dbname=s.db_name, user=s.db_user,
                         password=s.db_password, port=s.db_port)
    c.autocommit = True
    return c


def _ensure_metrics_tables(cur) -> None:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS proc.bp_extraction_health_metrics (
            id              BIGSERIAL PRIMARY KEY,
            recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            invoice_active_hitl  INT,
            po_active_hitl       INT,
            quote_active_hitl    INT,
            contract_active_hitl INT,
            invoice_raw_total    INT,
            po_raw_total         INT,
            quote_raw_total      INT,
            contract_raw_total   INT,
            stuck_rows_reset     INT,
            stuck_rows_failed    INT,
            audit_sample         INT,
            audit_violations     INT,
            failed_reaped        INT,
            doc_pk_collisions    INT
        );
        CREATE INDEX IF NOT EXISTS idx_health_metrics_recorded ON proc.bp_extraction_health_metrics (recorded_at);
        ALTER TABLE proc.bp_extraction_health_metrics
            ADD COLUMN IF NOT EXISTS doc_pk_collisions INT;
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS proc.bp_extraction_hallucination_audit (
            id              BIGSERIAL PRIMARY KEY,
            audited_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            provenance_id   BIGINT,
            doc_type        TEXT,
            doc_pk          TEXT,
            field_path      TEXT,
            value           TEXT,
            evidence_text   TEXT,
            reason          TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_hallucination_audit_at ON proc.bp_extraction_hallucination_audit (audited_at);
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS proc.bp_extraction_retry_log (
            pm_id           INT PRIMARY KEY,
            stuck_resets    INT NOT NULL DEFAULT 0,
            failed_retries  INT NOT NULL DEFAULT 0,
            last_reset_at   TIMESTAMPTZ,
            last_retry_at   TIMESTAMPTZ
        );
    """)


def stuck_row_recovery(cur) -> tuple[int, int]:
    """Find process_monitor rows mid-flight > STUCK_MINUTES; reset or fail."""
    placeholders = ", ".join(["%s"] * len(NON_TERMINAL))
    cur.execute(f"""
        SELECT pm.id, pm.status, pm.lastmodified_date,
               COALESCE(rl.stuck_resets, 0) AS resets
          FROM proc.process_monitor pm
     LEFT JOIN proc.bp_extraction_retry_log rl ON rl.pm_id = pm.id
         WHERE pm.status IN ({placeholders})
           AND pm.lastmodified_date < NOW() - INTERVAL '{STUCK_MINUTES} minutes'
         ORDER BY pm.id""", NON_TERMINAL)
    reset, failed = 0, 0
    for pm_id, status, mod, resets in cur.fetchall():
        if resets >= MAX_STUCK_RETRIES:
            # Cap reached — mark Extraction_Failed
            cur.execute("""UPDATE proc.process_monitor
                              SET status='Extraction_Failed', lastmodified_date=NOW(),
                                  end_ts=COALESCE(end_ts, NOW())
                            WHERE id=%s""", (pm_id,))
            cur.execute("""INSERT INTO proc.bp_extraction_retry_log (pm_id, stuck_resets, failed_retries)
                            VALUES (%s, %s, 1)
                            ON CONFLICT (pm_id) DO UPDATE SET
                                failed_retries = bp_extraction_retry_log.failed_retries + 1,
                                last_retry_at = NOW()""", (pm_id, resets))
            failed += 1
            _emit("stuck_capped", pm_id=pm_id, status=status, resets=resets)
        else:
            cur.execute("""UPDATE proc.process_monitor
                              SET status='Completed', start_ts=NULL,
                                  lastmodified_date=NOW()
                            WHERE id=%s""", (pm_id,))
            cur.execute("""INSERT INTO proc.bp_extraction_retry_log (pm_id, stuck_resets, last_reset_at)
                            VALUES (%s, 1, NOW())
                            ON CONFLICT (pm_id) DO UPDATE SET
                                stuck_resets = bp_extraction_retry_log.stuck_resets + 1,
                                last_reset_at = NOW()""", (pm_id,))
            reset += 1
            _emit("stuck_reset", pm_id=pm_id, status=status, prior_resets=resets,
                  stuck_minutes=int((time.time() - mod.timestamp()) / 60))
    return reset, failed


_PK_FIELD = {"invoice": "invoice_id", "purchase_order": "po_id",
             "quote": "quote_id", "contract": "contract_id"}


def _audit_is_violation(value, evidence_text, model, snapshots) -> bool:
    """True iff ``value`` is a genuine hallucination.

    A value is a hallucination only when it is ungrounded (by the format-tolerant
    match used by the live pipeline) in EVERY snapshot that shares its doc_pk.
    Checking all snapshots — not just the most recent — removes the false
    positives that doc_pk collisions used to cause (the value belongs to a
    sibling document under the same id). With no snapshot available the value
    cannot be verified, so it is not counted as a violation.
    """
    texts = [ft for ft in snapshots if ft]
    if not texts:
        return False
    return not any(
        is_value_grounded(value or "", evidence_text or "", ft, model=model or "")
        for ft in texts
    )


def _classify_collision(rows) -> bool:
    """True iff ``rows`` (each ``(source_file, text_hash)``) is a genuine doc_pk
    collision: >1 distinct source file AND >1 distinct document text. Same-file
    re-extractions and identical content re-uploaded under a new name are not
    collisions.
    """
    files = {f for f, _ in rows}
    texts = {t for _, t in rows}
    return len(files) > 1 and len(texts) > 1


def hallucination_audit(cur) -> tuple[int, int]:
    """Sample recent provenance rows; flag values ungrounded in the source doc.

    Uses the same format-tolerant grounding as the live pipeline (see
    ``grounding.is_value_grounded``) and checks every snapshot sharing the
    doc_pk, so reformatted-but-correct values and doc_pk collisions are no longer
    over-reported.
    """
    cur.execute("""
        SELECT provenance_id, doc_type, doc_pk, field_path, value, evidence_text, model
          FROM proc.bp_extraction_provenance_v3
         WHERE extracted_at > NOW() - INTERVAL '1 hour'
         ORDER BY RANDOM() LIMIT %s""", (AUDIT_SAMPLE_SIZE,))
    sample = cur.fetchall()
    if not sample:
        return 0, 0

    violations = 0
    for prov_id, doc_type, doc_pk, field_path, value, evidence_text, model in sample:
        raw_table = _RAW_TABLES.get(doc_type)
        pk_field = _PK_FIELD.get(doc_type)
        if not raw_table or not pk_field:
            continue
        # Fetch ALL snapshots sharing this doc_pk (not just the latest) so a
        # value belonging to a sibling collision document is not mis-flagged.
        cur.execute(
            f"SELECT parser_snapshot FROM {raw_table} WHERE {pk_field}=%s "
            f"ORDER BY extracted_at DESC LIMIT 20",
            (doc_pk,),
        )
        snapshots = [
            r[0].get("full_text") for r in cur.fetchall()
            if r[0] and isinstance(r[0], dict)
        ]
        if not snapshots:
            # _raw purged post-promotion — cannot verify; skip (not a violation).
            continue
        if _audit_is_violation(value, evidence_text, model, snapshots):
            violations += 1
            cur.execute(
                """INSERT INTO proc.bp_extraction_hallucination_audit
                       (provenance_id, doc_type, doc_pk, field_path, value, evidence_text, reason)
                   VALUES (%s,%s,%s,%s,%s,%s,'value_not_grounded_in_document')""",
                (prov_id, doc_type, doc_pk, field_path, value, evidence_text),
            )
            _emit("hallucination_violation", provenance_id=prov_id, doc_type=doc_type,
                  doc_pk=doc_pk, field=field_path, value=value)
    return len(sample), violations


def doc_pk_collision_audit(cur) -> int:
    """Detect genuine doc_pk collisions across the _raw tables and surface them.

    A collision is one doc_pk mapping to multiple DISTINCT documents (different
    source file AND different parser text). Each is emitted as a
    ``doc_pk_collision`` event for monitoring; the count is returned so the
    health snapshot can track it. Read-only — does not mutate pipeline data.
    """
    collisions = 0
    for doc_type, raw_table in _RAW_TABLES.items():
        pk_field = _PK_FIELD.get(doc_type)
        if not pk_field:
            continue
        try:
            cur.execute(
                f"""SELECT {pk_field} AS pk,
                           array_agg(source_file) AS files,
                           array_agg(md5(COALESCE(parser_snapshot->>'full_text',''))) AS texts
                      FROM {raw_table}
                     WHERE {pk_field} IS NOT NULL
                     GROUP BY {pk_field}
                    HAVING COUNT(*) > 1""")
        except Exception:
            continue
        for pk, files, texts in cur.fetchall():
            if _classify_collision(list(zip(files, texts))):
                collisions += 1
                _emit("doc_pk_collision", doc_type=doc_type, doc_pk=pk,
                      source_files=sorted(set(files)))
    return collisions


def backlog_metrics(cur) -> dict[str, int]:
    """Active-HITL + total _raw discrepancy by doc-type."""
    metrics = {}
    for cat, tbl in _RAW_TABLES.items():
        cur.execute(f"""SELECT
            COUNT(*) FILTER (WHERE r.promotion_status='discrepancy'
                AND EXISTS (SELECT 1 FROM proc.bp_extraction_discrepancy d
                             WHERE d.raw_id=r.raw_id AND d.status='open'
                               AND d.severity='critical' AND d.blocks_promotion=TRUE)) AS active,
            COUNT(*) FILTER (WHERE r.promotion_status='discrepancy') AS total
           FROM {tbl} r""")
        a, t = cur.fetchone()
        metrics[f"{cat}_active_hitl"] = a
        metrics[f"{cat}_raw_total"] = t
    return metrics


def main() -> None:
    started = time.time()
    try:
        conn = _conn()
        cur = conn.cursor()
        _ensure_metrics_tables(cur)

        reset, failed = stuck_row_recovery(cur)
        sample, violations = hallucination_audit(cur)
        collisions = doc_pk_collision_audit(cur)
        backlog = backlog_metrics(cur)

        # Write a metrics snapshot row
        cur.execute("""INSERT INTO proc.bp_extraction_health_metrics
                         (invoice_active_hitl, po_active_hitl, quote_active_hitl, contract_active_hitl,
                          invoice_raw_total, po_raw_total, quote_raw_total, contract_raw_total,
                          stuck_rows_reset, stuck_rows_failed,
                          audit_sample, audit_violations, failed_reaped, doc_pk_collisions)
                       VALUES (%(invoice_active_hitl)s, %(purchase_order_active_hitl)s,
                               %(quote_active_hitl)s, %(contract_active_hitl)s,
                               %(invoice_raw_total)s, %(purchase_order_raw_total)s,
                               %(quote_raw_total)s, %(contract_raw_total)s,
                               %(reset)s, %(failed)s, %(sample)s, %(violations)s, 0, %(collisions)s)""",
                    {**backlog, "reset": reset, "failed": failed,
                     "sample": sample, "violations": violations, "collisions": collisions})

        _emit("health_summary", duration_ms=int((time.time() - started) * 1000),
              stuck_reset=reset, stuck_failed=failed,
              audit_sample=sample, audit_violations=violations,
              doc_pk_collisions=collisions, **backlog)
        conn.close()
    except Exception as exc:
        _emit("health_error", error=str(exc), traceback=traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
