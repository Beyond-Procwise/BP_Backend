"""T1: the extraction-hint proposal queue table exists in the DB."""
import pytest

from src.services.db import get_conn


def _db_or_skip():
    try:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("select 1")
            cur.fetchone()
        return True
    except Exception as exc:  # pragma: no cover - env without DB
        pytest.skip(f"no DB: {exc}")


def test_proposal_table_exists():
    _db_or_skip()
    with get_conn() as c, c.cursor() as cur:
        cur.execute("select to_regclass('proc.bp_extraction_hint_proposal')")
        assert cur.fetchone()[0] is not None


def test_dedup_unique_index_exists():
    _db_or_skip()
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            "select indexname from pg_indexes "
            "where schemaname='proc' and tablename='bp_extraction_hint_proposal'"
        )
        idx = {r[0] for r in cur.fetchall()}
    assert "ix_bp_ext_hint_proposal_dedup" in idx
