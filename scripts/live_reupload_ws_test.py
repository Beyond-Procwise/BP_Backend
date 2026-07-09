"""Live end-to-end re-upload test against the RUNNING procwise server.

Flow (exercises the live ProcessMonitorWatcher via its real NOTIFY path):
  1. Seed an "original" processed copy of a real document (status Deal_Linked)
     carrying the content_hash of the live bytes.
  2. INSERT a "re-upload" row (status 'Completed') with a fresh session_id.
     The INSERT trigger fires pg_notify('process_monitor_ready') → the live
     watcher claims + processes it → detects the content duplicate → marks
     doc_action='duplicate', skips extraction, records a 'target' session
     outcome → the session resolves → pg_notify('session_status').
  3. Poll until the re-upload row is resolved; print its doc_action/status.
  4. Connect a REAL WebSocket to /ws/session/{id} and print the payload the UI
     receives (action_status rollup + doc_action breakdown).
  5. Clean up all test rows.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg2
import websockets
from config.settings import Settings
from src.services.extraction.content_hash import compute_content_hash

WS_URL = "ws://localhost:8000/ws/session/{sid}"


def _conn():
    s = Settings()
    c = psycopg2.connect(host=s.db_host, dbname=s.db_name,
                         user=s.db_user, password=s.db_password, port=s.db_port)
    c.autocommit = True
    return c


async def _fetch_ws_payload(session_id: str, timeout: float = 15.0) -> dict | None:
    try:
        async with websockets.connect(WS_URL.format(sid=session_id)) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            return json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"  WS error: {type(exc).__name__}: {exc}")
        return None


def main() -> int:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT file_path, category FROM proc.process_monitor
        WHERE file_path IS NOT NULL AND category IS NOT NULL
        ORDER BY id DESC LIMIT 1
    """)
    row = cur.fetchone()
    if not row:
        print("NO document available"); return 1
    file_path, category = row
    print(f"[1] Real document: {file_path} ({category})")

    content_hash = compute_content_hash(file_path)
    if not content_hash:
        print("SKIP — could not hash live bytes (S3 unavailable)"); return 2
    print(f"[1] content_hash = {content_hash[:16]}...")

    # Unique session id (no time import in workflow, but plain script is fine)
    session_id = f"DOCACTION-REUP-{int(time.time())}"
    a_id = b_id = None
    try:
        # Original processed copy (already in _trgt), carries the hash
        cur.execute("""
            INSERT INTO proc.process_monitor
                (process_name, type, status, file_path, category, content_hash,
                 created_date, lastmodified_date)
            VALUES ('reup-test-orig','inbound','Deal_Linked',%s,%s,%s,NOW(),NOW())
            RETURNING id
        """, (file_path, category, content_hash))
        a_id = cur.fetchone()[0]

        # Real re-upload — status 'Completed' fires the watcher via NOTIFY
        cur.execute("""
            INSERT INTO proc.process_monitor
                (process_name, type, status, file_path, category, session_id,
                 created_date, lastmodified_date)
            VALUES ('reup-test-dup','inbound','Completed',%s,%s,%s,NOW(),NOW())
            RETURNING id
        """, (file_path, category, session_id))
        b_id = cur.fetchone()[0]
        print(f"[2] Seeded original id={a_id}; re-uploaded id={b_id} session={session_id}")
        print(f"[2] NOTIFY fired — waiting for the LIVE watcher to process it...")

        # Poll for the watcher to resolve the re-upload
        deadline = time.time() + 90
        doc_action = status = action_status = None
        while time.time() < deadline:
            cur.execute("SELECT doc_action, status, action_status FROM proc.process_monitor WHERE id=%s", (b_id,))
            doc_action, status, action_status = cur.fetchone()
            if doc_action is not None and status in ("Extracted", "Extraction_Failed"):
                break
            time.sleep(2)
        print(f"[3] Re-upload row: doc_action={doc_action} status={status} action_status={action_status}")

        # Connect a REAL websocket to see the payload the UI receives
        print(f"[4] Connecting WebSocket /ws/session/{session_id} ...")
        payload = asyncio.run(_fetch_ws_payload(session_id))
        if payload:
            print("[4] WS payload delivered to UI:")
            print(json.dumps(payload, indent=2, default=str))
        else:
            print("[4] No WS payload received")

        ok = (doc_action == "duplicate" and status == "Extracted")
        ws_ok = bool(payload) and "action_status" in (payload or {}) and (payload or {}).get("duplicate", 0) >= 1
        print("=" * 60)
        print(f"RE-UPLOAD  → {'PASS' if ok else 'FAIL'} (doc_action=duplicate, extraction skipped)")
        print(f"WEBSOCKET  → {'PASS' if ws_ok else 'FAIL'} "
              f"(action_status='{(payload or {}).get('action_status')}', duplicate={(payload or {}).get('duplicate')})")
        return 0 if (ok and ws_ok) else 1
    finally:
        if a_id is not None:
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy "
                        "WHERE source_file=%s AND issue_type='duplicate_document' "
                        "AND doc_pk_candidate=%s", (file_path, str(a_id)))
        if b_id is not None:
            cur.execute("DELETE FROM proc.session_document_outcome WHERE session_id=%s", (session_id,))
        for rid in (b_id, a_id):
            if rid is not None:
                cur.execute("DELETE FROM proc.process_monitor WHERE id=%s", (rid,))
        conn.close()
        print("[5] Cleaned up test rows.")


if __name__ == "__main__":
    raise SystemExit(main())
