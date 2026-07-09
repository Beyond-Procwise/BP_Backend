"""Live end-to-end check for content-hash dedup + doc_action (bp_sqldb).

Self-contained scenario (does NOT run a heavy extraction):
  1. Pick a real document file_path from process_monitor whose data reached a
     _trgt table (or whose PK is not derivable → treated as present).
  2. Compute its content hash from the live bytes (local or S3), exactly as the
     watcher would.
  3. Register an "original" row A pointing at that file with the computed
     content_hash and a post-extraction status (simulating a prior success).
  4. Register a "re-upload" row B with the same file_path, status 'Completed'.
  5. Run the watcher's _process_record on B and assert B.doc_action='duplicate'
     with no re-extraction.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg2
from config.settings import Settings
from services.process_monitor_watcher import ProcessMonitorWatcher
from src.services.extraction.content_hash import compute_content_hash


def _conn():
    s = Settings()
    c = psycopg2.connect(host=s.db_host, dbname=s.db_name,
                         user=s.db_user, password=s.db_password, port=s.db_port)
    c.autocommit = True
    return c


class _Nick:
    settings = Settings()


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
        print("NO document available to test against")
        return 1
    file_path, category = row
    print(f"Using file={file_path} category={category}")

    content_hash = compute_content_hash(file_path)
    if not content_hash:
        print(f"SKIP — could not resolve/hash bytes for {file_path} "
              f"(file not local and S3 unavailable in this environment)")
        return 2
    print(f"Computed content_hash={content_hash[:12]}...")

    a_id = b_id = None
    try:
        # Original A — a prior success carrying the content hash
        cur.execute("""
            INSERT INTO proc.process_monitor
                (process_name, type, status, file_path, category, content_hash,
                 created_date, lastmodified_date)
            VALUES ('doc-action-validate-orig', 'inbound', 'Deal_Linked', %s, %s, %s, NOW(), NOW())
            RETURNING id
        """, (file_path, category, content_hash))
        a_id = cur.fetchone()[0]

        # Re-upload B — identical content, awaiting processing
        cur.execute("""
            INSERT INTO proc.process_monitor
                (process_name, type, status, file_path, category, created_date, lastmodified_date)
            VALUES ('doc-action-validate-dup', 'inbound', 'Completed', %s, %s, NOW(), NOW())
            RETURNING id
        """, (file_path, category))
        b_id = cur.fetchone()[0]
        print(f"Registered original id={a_id}, re-upload id={b_id}")

        w = ProcessMonitorWatcher(_Nick())
        with w._processing_lock:
            w._processing_ids.add(b_id)
        w._process_record({
            "id": b_id, "file_path": file_path,
            "category": category, "user_id": None,
        })

        cur.execute("SELECT doc_action, status, content_hash FROM proc.process_monitor WHERE id=%s", (b_id,))
        doc_action, status, chash = cur.fetchone()
        print(f"RESULT: id={b_id} doc_action={doc_action} status={status} "
              f"content_hash={(chash or '')[:12]}...")
        ok = (doc_action == "duplicate" and status == "Extracted")
        print("PASS" if ok else "FAIL — expected doc_action='duplicate', status='Extracted'")
        return 0 if ok else 1
    finally:
        # Remove the best-effort duplicate audit note this run may have written
        if a_id is not None:
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy "
                        "WHERE source_file=%s AND issue_type='duplicate_document' "
                        "AND doc_pk_candidate=%s", (file_path, str(a_id)))
        for rid in (b_id, a_id):
            if rid is not None:
                cur.execute("DELETE FROM proc.process_monitor WHERE id=%s", (rid,))
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
