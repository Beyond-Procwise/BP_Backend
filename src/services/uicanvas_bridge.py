"""uicanvas ↔ bp_sqldb process_monitor bridge.

Two-way mirror so the UI (writing to uicanvas) stays consistent with the
renovation pipeline (running on bp_sqldb).

Forward path (uicanvas → bp_sqldb):
- LISTEN on uicanvas.process_monitor_ready.
- On NOTIFY, read the new uicanvas row, INSERT a mirror row into
  bp_sqldb.process_monitor with status='Completed'. This fires bp_sqldb's
  own NOTIFY trigger which the renovation watcher consumes.
- The mirroring records a marker ``bridged_to_bp_sqldb_id=<new_id>`` in
  the uicanvas row's ``created_by`` column so we never double-mirror.

Reverse-sync (bp_sqldb → uicanvas):
- A periodic sweep (every REVERSE_SYNC_SECONDS, default 15s) reads
  bridged uicanvas rows whose status is still 'Completed'/'Running' and
  reflects the bp_sqldb status back. Once the watcher in bp_sqldb marks
  the mirror row 'Extracted' (or 'Extraction_Failed'), the originating
  uicanvas row gets the same terminal status, so the UI sees the
  pipeline outcome.

On startup the bridge runs a catch-up sweep (forward + reverse) so
anything queued or in-flight during a procwise restart is reconciled.
"""
from __future__ import annotations

import json
import logging
import os
import re
import select
import threading
import time
from typing import Any, Optional

import psycopg2

from config.settings import Settings

log = logging.getLogger(__name__)

_BRIDGED_MARKER_PREFIX = "bridged_to_bp_sqldb_id="
_BRIDGED_ID_RE = re.compile(re.escape(_BRIDGED_MARKER_PREFIX) + r"(\d+)")
# Reverse-sync interval (seconds). Bridged uicanvas rows are checked at this
# cadence and given the bp_sqldb mirror's status when it reaches a terminal
# state. Tunable via env var; default 15s for snappy UI feedback.
_REVERSE_SYNC_SECONDS = int(os.getenv("UICANVAS_BRIDGE_REVERSE_SYNC_SECONDS", "15"))
# Terminal statuses on bp_sqldb that we mirror back to uicanvas.
_TERMINAL_STATUSES = ("Extracted", "Extraction_Failed")


def _connect(dbname: str, *, autocommit: bool = True) -> psycopg2.extensions.connection:
    s = Settings()
    conn = psycopg2.connect(
        host=s.db_host, dbname=dbname, user=s.db_user,
        password=s.db_password, port=s.db_port,
    )
    conn.autocommit = autocommit
    return conn


def _already_bridged(row: dict[str, Any]) -> bool:
    created_by = row.get("created_by") or ""
    return _BRIDGED_MARKER_PREFIX in str(created_by)


def _mirror_row(uicanvas_row: dict[str, Any]) -> Optional[int]:
    """Copy a uicanvas process_monitor row into bp_sqldb. Returns the new
    bp_sqldb id, or None if skipped (already bridged, or missing required
    fields)."""
    if _already_bridged(uicanvas_row):
        return None
    file_path = uicanvas_row.get("file_path")
    category = uicanvas_row.get("category")
    if not file_path or not category:
        log.debug("skip uicanvas id=%s: missing file_path/category",
                  uicanvas_row.get("id"))
        return None

    dst = _connect("bp_sqldb")
    src = _connect("uicanvas")
    try:
        dcur = dst.cursor()
        dcur.execute("""INSERT INTO proc.process_monitor
                           (process_name, type, status, file_path, category, document_type,
                            user_id, total_count, created_date, lastmodified_date, created_by)
                         VALUES (%s, %s, 'Completed', %s, %s, %s, %s, %s, NOW(), NOW(), %s)
                         RETURNING id""",
                     (uicanvas_row.get("process_name") or "uicanvas-bridge",
                      uicanvas_row.get("type") or "inbound",
                      file_path,
                      category,
                      uicanvas_row.get("document_type"),
                      uicanvas_row.get("user_id"),
                      uicanvas_row.get("total_count"),
                      f"uicanvas-bridge-from-{uicanvas_row.get('id')}"))
        new_id = dcur.fetchone()[0]
        # Mark uicanvas row as bridged
        scur = src.cursor()
        marker = f"{_BRIDGED_MARKER_PREFIX}{new_id}"
        scur.execute(
            "UPDATE proc.process_monitor SET created_by=%s WHERE id=%s",
            (marker, uicanvas_row["id"]),
        )
        log.info("Bridged uicanvas pm_id=%s → bp_sqldb pm_id=%s (file=%s, category=%s)",
                 uicanvas_row["id"], new_id, file_path, category)
        return new_id
    finally:
        dst.close()
        src.close()


def _fetch_one(cur, pm_id: int) -> Optional[dict[str, Any]]:
    cur.execute("""SELECT id, process_name, type, status, file_path, category,
                          document_type, user_id, total_count, created_by, created_date
                     FROM proc.process_monitor WHERE id=%s""", (pm_id,))
    row = cur.fetchone()
    if row is None:
        return None
    cols = ["id", "process_name", "type", "status", "file_path", "category",
            "document_type", "user_id", "total_count", "created_by", "created_date"]
    return dict(zip(cols, row))


def _parse_bridged_id(created_by: Optional[str]) -> Optional[int]:
    if not created_by:
        return None
    m = _BRIDGED_ID_RE.search(str(created_by))
    return int(m.group(1)) if m else None


def reverse_sync_sweep() -> int:
    """For every uicanvas row marked as bridged whose status is still
    non-terminal, look up the bp_sqldb mirror row and copy its terminal
    status (Extracted / Extraction_Failed) back. Returns count updated."""
    src = _connect("uicanvas")
    dst = _connect("bp_sqldb")
    try:
        scur = src.cursor()
        scur.execute(f"""SELECT id, status, created_by
                           FROM proc.process_monitor
                          WHERE created_by LIKE %s
                            AND status NOT IN ({','.join(['%s'] * len(_TERMINAL_STATUSES))})""",
                     (f"%{_BRIDGED_MARKER_PREFIX}%", *_TERMINAL_STATUSES))
        candidates = scur.fetchall()
        if not candidates:
            return 0
        dcur = dst.cursor()
        updated = 0
        for ui_id, ui_status, ui_created_by in candidates:
            bp_id = _parse_bridged_id(ui_created_by)
            if bp_id is None:
                continue
            dcur.execute("SELECT status FROM proc.process_monitor WHERE id=%s", (bp_id,))
            row = dcur.fetchone()
            if row is None:
                continue
            bp_status = row[0]
            if bp_status not in _TERMINAL_STATUSES:
                continue
            scur.execute("""UPDATE proc.process_monitor
                               SET status=%s, lastmodified_date=NOW(),
                                   end_ts=COALESCE(end_ts, NOW())
                             WHERE id=%s""", (bp_status, ui_id))
            log.info("Reverse-sync: uicanvas pm_id=%s status %s → %s (mirror bp_sqldb pm_id=%s)",
                     ui_id, ui_status, bp_status, bp_id)
            updated += 1
        return updated
    finally:
        src.close()
        dst.close()


def _catch_up_sweep() -> int:
    """At startup, mirror any uicanvas rows that arrived while procwise was
    down. Returns number bridged."""
    src = _connect("uicanvas")
    try:
        cur = src.cursor()
        cur.execute("""SELECT id, process_name, type, status, file_path, category,
                              document_type, user_id, total_count, created_by, created_date
                         FROM proc.process_monitor
                        WHERE status IN ('Completed', 'Running')
                          AND (created_by IS NULL OR created_by NOT LIKE %s)
                          AND file_path IS NOT NULL
                        ORDER BY id""", (f"{_BRIDGED_MARKER_PREFIX}%",))
        cols = ["id", "process_name", "type", "status", "file_path", "category",
                "document_type", "user_id", "total_count", "created_by", "created_date"]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        src.close()

    bridged = 0
    for row in rows:
        try:
            if _mirror_row(row) is not None:
                bridged += 1
        except Exception as exc:
            log.warning("catch-up mirror failed for uicanvas id=%s: %s",
                        row.get("id"), exc)
    if bridged:
        log.info("uicanvas bridge catch-up: mirrored %d rows", bridged)
    return bridged


class UicanvasBridge:
    """Long-running worker that mirrors uicanvas.process_monitor → bp_sqldb."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        # Catch up forward (uicanvas → bp_sqldb) then reverse (back-fill any
        # terminal statuses that completed while procwise was down).
        try:
            _catch_up_sweep()
        except Exception:
            log.exception("uicanvas bridge catch-up failed")
        try:
            n = reverse_sync_sweep()
            if n:
                log.info("uicanvas bridge reverse-sync catch-up: updated %d rows", n)
        except Exception:
            log.exception("uicanvas bridge reverse-sync catch-up failed")
        self._thread = threading.Thread(target=self._run, name="uicanvas-bridge",
                                        daemon=True)
        self._thread.start()
        # Reverse-sync runs on its own cadence
        self._reverse_thread = threading.Thread(
            target=self._reverse_sync_loop, name="uicanvas-bridge-reverse",
            daemon=True,
        )
        self._reverse_thread.start()
        log.info("UicanvasBridge started")

    def _reverse_sync_loop(self) -> None:
        while not self._stop.is_set():
            try:
                reverse_sync_sweep()
            except Exception:
                log.exception("uicanvas reverse-sync sweep failed")
            if self._stop.wait(_REVERSE_SYNC_SECONDS):
                return

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._listen_loop()
            except Exception as exc:
                log.warning("UicanvasBridge listen loop crashed; restarting in 10s: %s", exc)
                if self._stop.wait(10):
                    return

    def _listen_loop(self) -> None:
        conn = _connect("uicanvas")
        try:
            cur = conn.cursor()
            cur.execute("LISTEN process_monitor_ready;")
            log.info("UicanvasBridge listening on uicanvas.process_monitor_ready")
            while not self._stop.is_set():
                if not select.select([conn], [], [], 2.0)[0]:
                    continue
                conn.poll()
                while conn.notifies:
                    n = conn.notifies.pop(0)
                    try:
                        pm_id = int(n.payload)
                    except (ValueError, TypeError):
                        log.debug("malformed notify payload: %r", n.payload)
                        continue
                    row = _fetch_one(cur, pm_id)
                    if row is None:
                        continue
                    if row["status"] not in ("Completed", "Running"):
                        continue
                    try:
                        _mirror_row(row)
                    except Exception:
                        log.exception("mirror failed for uicanvas pm_id=%s", pm_id)
        finally:
            try:
                conn.close()
            except Exception:
                pass
