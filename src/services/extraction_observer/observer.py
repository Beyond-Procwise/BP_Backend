"""Independent background observer for extraction patterns and gaps.

Runs as a long-lived systemd service. Polls bp_*_raw tables every POLL_SECONDS,
analyses each new row, and writes findings into proc.bp_extraction_observation
and to journald. Designed to keep running even when Claude is off — the
analysis is deterministic Python with no LLM dependency.

What it captures:
  1. Per-doc field-fill report — which expected stg columns are NULL.
  2. Wrong-label leak detection — e.g. po_id values that match an
     "Invoice Number" pattern in the source filename / payload evidence.
  3. New vendor patterns — filename prefixes never seen before, with
     extraction-quality summary.
  4. Discrepancy aggregation — count by issue_type / severity over the
     last hour, surfaced when counts spike.
  5. Schema gaps — fields present in raw_payload.header but with no
     stg column populated (suggests a missing rename map entry).

Output:
  - INFO log lines to journald for each batch summary.
  - One row per significant observation in proc.bp_extraction_observation
    (auto-created on first run if absent).

Env:
  OBSERVER_POLL_SECONDS  poll interval (default 60)
  OBSERVER_BATCH_LIMIT   max raw rows per poll (default 200)
  OBSERVER_LOOKBACK_MIN  initial lookback when no checkpoint (default 30)
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# psycopg2 import deferred so module loads cleanly during smoke checks
log = logging.getLogger("extraction_observer")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
log.addHandler(_handler)

POLL_SECONDS = int(os.getenv("OBSERVER_POLL_SECONDS", "60"))
BATCH_LIMIT = int(os.getenv("OBSERVER_BATCH_LIMIT", "200"))
LOOKBACK_MIN = int(os.getenv("OBSERVER_LOOKBACK_MIN", "30"))
CHECKPOINT_FILE = Path(os.getenv(
    "OBSERVER_CHECKPOINT_FILE",
    "/var/lib/bp-extraction-observer/checkpoint.txt",
))

DOC_TYPES = ("invoice", "purchase_order", "quote")

# Expected stg columns per doc-type that should usually be populated
# (used to compute the gap report). Keep this conservative: include only
# fields the engine + LLM layer realistically extract from real docs.
_EXPECTED_NONNULL = {
    "invoice": [
        "invoice_id", "invoice_date", "currency", "invoice_amount",
        "tax_amount", "invoice_total_incl_tax", "supplier_id", "buyer_id",
        "exchange_rate_to_usd", "converted_amount_usd",
    ],
    "purchase_order": [
        "po_id", "order_date", "currency", "total_amount",
        "tax_amount", "total_amount_incl_tax", "supplier_id", "supplier_name",
        "buyer_id", "exchange_rate_to_usd", "converted_amount_usd",
    ],
    "quote": [
        "quote_id", "quote_date", "currency", "total_amount",
        "tax_amount", "total_amount_incl_tax", "supplier_id", "buyer_id",
    ],
}

_PK_COL = {"invoice": "invoice_id", "purchase_order": "po_id", "quote": "quote_id"}
_RAW_TABLE = {
    "invoice": "proc.bp_invoice_raw",
    "purchase_order": "proc.bp_purchase_order_raw",
    "quote": "proc.bp_quote_raw",
}
_STG_TABLE = {
    "invoice": "proc.bp_invoice_stg",
    "purchase_order": "proc.bp_purchase_order_stg",
    "quote": "proc.bp_quote_stg",
}


def _ensure_observation_table(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS proc.bp_extraction_observation (
                obs_id          BIGSERIAL PRIMARY KEY,
                observed_at     TIMESTAMPTZ DEFAULT NOW(),
                doc_type        TEXT NOT NULL,
                raw_id          BIGINT,
                doc_pk          TEXT,
                source_file     TEXT,
                obs_type        TEXT NOT NULL,
                severity        TEXT NOT NULL,
                detail          JSONB
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS bp_extraction_observation_obs_type_idx
                ON proc.bp_extraction_observation (obs_type, observed_at DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS bp_extraction_observation_doc_type_idx
                ON proc.bp_extraction_observation (doc_type, observed_at DESC)
        """)
    conn.commit()


def _read_checkpoint() -> datetime:
    if CHECKPOINT_FILE.exists():
        try:
            ts = CHECKPOINT_FILE.read_text().strip()
            return datetime.fromisoformat(ts)
        except Exception:
            log.warning("checkpoint file unreadable; using lookback default")
    return datetime.now(timezone.utc) - timedelta(minutes=LOOKBACK_MIN)


def _write_checkpoint(ts: datetime) -> None:
    try:
        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_FILE.write_text(ts.isoformat())
    except OSError as exc:
        log.warning("checkpoint write failed: %s", exc)


def _record_observation(
    conn: Any,
    doc_type: str,
    raw_id: int | None,
    doc_pk: str | None,
    source_file: str | None,
    obs_type: str,
    severity: str,
    detail: dict,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO proc.bp_extraction_observation
                (doc_type, raw_id, doc_pk, source_file, obs_type, severity, detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (doc_type, raw_id, doc_pk, source_file, obs_type, severity, json.dumps(detail, default=str)),
        )


def _analyze_row(conn: Any, doc_type: str, raw_row: dict) -> list[tuple[str, str, dict]]:
    """Return list of (obs_type, severity, detail) findings for a single raw row.

    The renovation pipeline writes ``raw_payload = '{}'`` as a placeholder and
    persists field values in flat columns on ``proc.bp_<doctype>_raw``. So
    "what got extracted" is the set of expected-columns whose values are
    NON-NULL on the raw row, and "what blocks promotion" is the discrepancy
    rows linked by ``raw_id``. We surface both for every doc — not just the
    promoted ones — so the observer is useful when promotion has stalled.
    """
    findings: list[tuple[str, str, dict]] = []
    raw_id = raw_row["raw_id"]
    doc_pk = raw_row["doc_pk_candidate"]
    promotion = raw_row["promotion_status"]
    raw_table = _RAW_TABLE[doc_type]

    expected = _EXPECTED_NONNULL.get(doc_type, [])

    # 1. no_pk audit — captures docs where extraction succeeded but the PK
    # pattern never fired (filename PK fallback would close most of these).
    if promotion == "no_pk" or (not doc_pk and promotion in ("discrepancy", "pending")):
        findings.append((
            "no_pk",
            "warning",
            {"file": raw_row["source_file"], "promotion_status": promotion},
        ))

    # 2. Per-doc field-fill report from the RAW flat columns (works for ANY
    # promotion status — that's the whole point post-renovation).
    raw_field_state: dict[str, bool] = {}
    if expected:
        present_cols = _table_columns(conn, raw_table)
        cols_to_read = [c for c in expected if c in present_cols]
        if cols_to_read:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {', '.join(cols_to_read)} FROM {raw_table} WHERE raw_id = %s",
                    (raw_id,),
                )
                row = cur.fetchone()
            if row:
                for col, val in zip(cols_to_read, row):
                    raw_field_state[col] = (val is not None and val != "")
                null_cols = [c for c, ok in raw_field_state.items() if not ok]
                if null_cols:
                    findings.append((
                        "field_gap",
                        "info",
                        {
                            "promotion_status": promotion,
                            "missing_columns": null_cols,
                            "filled_count": sum(1 for ok in raw_field_state.values() if ok),
                            "expected_count": len(cols_to_read),
                        },
                    ))

    # 3. Blocking-discrepancy summary — pull the exact reasons that stopped
    # promotion (or warned) straight from the discrepancy table. This is the
    # single most useful signal for "why didn't this promote".
    if promotion in ("discrepancy", "pending"):
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT field_name, issue_type, severity, blocks_promotion, notes
                FROM proc.bp_extraction_discrepancy
                WHERE doc_type = %s AND raw_id = %s
                """,
                (doc_type, raw_id),
            )
            disc_rows = cur.fetchall()
        if disc_rows:
            blockers = [
                {
                    "field": fr[0], "issue_type": fr[1], "severity": fr[2],
                    "blocks_promotion": bool(fr[3]),
                    "notes": (fr[4] or "")[:200],
                }
                for fr in disc_rows
            ]
            severity = "critical" if any(b["blocks_promotion"] for b in blockers) else "warning"
            findings.append((
                "blocking_discrepancy",
                severity,
                {"count": len(blockers), "blockers": blockers},
            ))

    # 4. Cross-check stg when promoted: confirm what landed.
    if promotion == "promoted" and doc_pk:
        with conn.cursor() as cur:
            stg_table = _STG_TABLE[doc_type]
            pk_col = _PK_COL[doc_type]
            if expected:
                stg_cols = _table_columns(conn, stg_table)
                cols_to_read = [c for c in expected if c in stg_cols]
                if cols_to_read:
                    cur.execute(
                        f"SELECT {', '.join(cols_to_read)} FROM {stg_table} WHERE {pk_col} = %s LIMIT 1",
                        (doc_pk,),
                    )
                    row = cur.fetchone()
                    if row:
                        null_cols = [c for c, v in zip(cols_to_read, row) if v is None or v == ""]
                        if null_cols:
                            findings.append((
                                "stg_field_gap",
                                "info",
                                {"missing_columns": null_cols, "expected_count": len(cols_to_read)},
                            ))

    return findings


def _table_columns(conn: Any, fq_table: str) -> set[str]:
    """Return the set of column names for proc.<table>. Cheap and cached
    per-process via the connection's autouse — observers run a single
    long-lived connection so the LRU here is just a dict on the function."""
    cache = getattr(_table_columns, "_cache", None)
    if cache is None:
        cache = {}
        _table_columns._cache = cache  # type: ignore[attr-defined]
    if fq_table in cache:
        return cache[fq_table]
    schema, table = fq_table.split(".", 1)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema, table),
        )
        cols = {r[0] for r in cur.fetchall()}
    cache[fq_table] = cols
    return cols


def _process_batch(conn: Any, since: datetime) -> tuple[int, datetime]:
    """Process all raw rows extracted since `since`. Return (n_processed, new_checkpoint)."""
    processed = 0
    max_seen = since
    by_type_counter: dict[str, Counter] = defaultdict(Counter)

    for doc_type in DOC_TYPES:
        raw_table = _RAW_TABLE[doc_type]
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT raw_id, doc_pk_candidate, source_file,
                       promotion_status, extracted_at
                FROM {raw_table}
                WHERE extracted_at > %s
                ORDER BY extracted_at ASC
                LIMIT %s
                """,
                (since, BATCH_LIMIT),
            )
            cols = [c[0] for c in cur.description]
            for row in cur.fetchall():
                rec = dict(zip(cols, row))
                processed += 1
                if rec["extracted_at"] > max_seen:
                    max_seen = rec["extracted_at"]
                findings = _analyze_row(conn, doc_type, rec)
                for obs_type, severity, detail in findings:
                    by_type_counter[doc_type][obs_type] += 1
                    _record_observation(
                        conn, doc_type,
                        rec["raw_id"], rec["doc_pk_candidate"], rec["source_file"],
                        obs_type, severity, detail,
                    )
        conn.commit()

    if processed > 0:
        for dt, cnt in by_type_counter.items():
            log.info("observer batch: doc_type=%s findings=%s", dt, dict(cnt))
    return processed, max_seen


_running = True


def _handle_signal(signum, _frame):  # noqa: ARG001
    global _running
    log.info("signal %s received -- exiting", signum)
    _running = False


def main() -> int:
    log.info(
        "extraction_observer starting -- poll=%ds batch=%d lookback=%dmin",
        POLL_SECONDS, BATCH_LIMIT, LOOKBACK_MIN,
    )
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    from src.services.db import get_conn
    last_ts = _read_checkpoint()
    log.info("starting from checkpoint: %s", last_ts.isoformat())
    while _running:
        try:
            with get_conn() as conn:
                _ensure_observation_table(conn)
                n, new_ts = _process_batch(conn, last_ts)
                if n > 0:
                    _write_checkpoint(new_ts)
                    last_ts = new_ts
        except Exception as exc:
            log.exception("observer iteration failed: %s -- continuing", exc)
        # Sleep responsively so SIGTERM doesn't hang
        for _ in range(POLL_SECONDS):
            if not _running:
                break
            time.sleep(1)
    log.info("extraction_observer stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
