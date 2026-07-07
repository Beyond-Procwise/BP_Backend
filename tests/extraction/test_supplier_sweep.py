"""Existing-supplier duplicate sweep flags near-dup pairs, idempotently."""
import pytest

from src.services.db import get_conn
from src.services.extraction_v3 import supplier_resolver as SR

IDP = "SUP-ZZDUP"  # controlled supplier_id prefix for cleanup (names are distinctive)


def _clean():
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_supplier_review WHERE candidate_supplier_id ILIKE %s OR chosen_supplier_id ILIKE %s", (IDP+"%", IDP+"%"))
            cur.execute("DELETE FROM proc.bp_supplier WHERE supplier_id ILIKE %s", (IDP+"%",))
        c.commit()


@pytest.fixture(autouse=True)
def around():
    try:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("select 1")
    except Exception:
        pytest.skip("no DB")
    _clean(); yield; _clean()


def _seed(sid, name):
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute("INSERT INTO proc.bp_supplier (supplier_id, supplier_name, trading_name, created_date, created_by) VALUES (%s,%s,%s,NOW(),'test')", (sid, name, name))
        c.commit()


def _internal_pair_flags():
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT count(*) FROM proc.bp_supplier_review "
                    "WHERE candidate_supplier_id ILIKE %s AND chosen_supplier_id ILIKE %s AND status='pending'",
                    (IDP+"%", IDP+"%"))
        return cur.fetchone()[0]


def test_sweep_flags_near_dup_pair_only():
    # Distinctive names (no shared token) so only A~B are near-duplicates.
    _seed(f"{IDP}A", "Zephyr Widgets Solutions Ltd")
    _seed(f"{IDP}B", "Zephyr Widgets Solution Ltd")    # near-dup of A
    _seed(f"{IDP}C", "Qromax Distinct Traders Ltd")     # unrelated

    with get_conn() as c:
        SR.sweep_supplier_duplicates(c, min_score=88)
    assert _internal_pair_flags() == 1, "exactly the A~B pair should be flagged"

    # idempotent: re-running creates no new flag for the decided pair
    with get_conn() as c:
        SR.sweep_supplier_duplicates(c, min_score=88)
    assert _internal_pair_flags() == 1
