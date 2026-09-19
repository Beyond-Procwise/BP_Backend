"""Fill proc.bp_analysis_document.doc_pk for analyses frozen before freeze()
learned to (every row written until 2026-09-19 has it NULL).

Only doc_pk is filled: which document a file IS does not change over time,
and it is traced exactly (process_monitor -> raw by process_monitor_id), so
this writes a fact rather than re-deriving a finding. Deal links and findings
are left as they were frozen. Rows nothing can be traced to stay NULL.

    ./.venv/bin/python scripts/backfill_analysis_document_doc_pk.py [--apply]
"""
import sys

from dotenv import load_dotenv

load_dotenv()

from src.services.analysis_store import _resolved_documents  # noqa: E402
from src.services.db import get_conn  # noqa: E402


def main(apply: bool) -> None:
    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT a.analysis_id, a.session_id "
            "  FROM proc.bp_analysis a "
            "  JOIN proc.bp_analysis_document d USING (analysis_id) "
            " WHERE d.doc_pk IS NULL AND a.session_id IS NOT NULL")
        filled = 0
        for analysis_id, session_id in cur.fetchall():
            for file_path, doc_pk, _deal in _resolved_documents(cur, session_id):
                cur.execute(
                    "UPDATE proc.bp_analysis_document SET doc_pk = %s "
                    "WHERE analysis_id = %s AND file_path = %s AND doc_pk IS NULL",
                    (doc_pk, analysis_id, file_path))
                filled += cur.rowcount
        cur.execute("SELECT count(*) FROM proc.bp_analysis_document WHERE doc_pk IS NULL")
        left = cur.fetchone()[0]
        print(f"filled {filled}; still NULL {left}; {'APPLIED' if apply else 'dry run, rolled back'}")
        conn.commit() if apply else conn.rollback()


if __name__ == "__main__":
    main("--apply" in sys.argv)
