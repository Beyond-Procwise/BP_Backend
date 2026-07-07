"""Supplier entity-resolution: alias short-circuit + close-call review flagging."""
import pytest
from rapidfuzz import fuzz

from src.services.db import get_conn
from src.services.extraction_v3 import supplier_resolver as SR

PFX = "ZZTEST"


def _clean():
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_supplier_review WHERE extracted_name ILIKE %s", (PFX + "%",))
            cur.execute("DELETE FROM proc.bp_supplier_alias WHERE alias_name ILIKE %s", (PFX + "%",))
            cur.execute("DELETE FROM proc.bp_supplier WHERE supplier_name ILIKE %s OR supplier_id ILIKE %s",
                        (PFX + "%", "SUP-" + PFX + "%"))
        c.commit()


@pytest.fixture(autouse=True)
def clean_around():
    try:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("select 1")
    except Exception:
        pytest.skip("no DB")
    _clean()
    yield
    _clean()


def _seed_supplier(sid, name):
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute("INSERT INTO proc.bp_supplier (supplier_id, supplier_name, trading_name, created_date, created_by) "
                        "VALUES (%s,%s,%s,NOW(),'test')", (sid, name, name))
        c.commit()


def test_alias_short_circuits_resolution():
    _seed_supplier(f"SUP-{PFX}Canonical", f"{PFX} Canonical Corp Ltd")
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute("INSERT INTO proc.bp_supplier_alias (alias_name, supplier_id, created_by) VALUES (%s,%s,'test')",
                        (f"{PFX} Variant Spelling Ltd", f"SUP-{PFX}Canonical"))
        c.commit()
        sid = SR.resolve_or_create_supplier(f"{PFX} Variant Spelling Ltd", c)
    assert sid == f"SUP-{PFX}Canonical"  # alias wins, no new supplier created


def test_close_call_creates_review():
    seed_name = f"{PFX} Widgets Solutions Ltd"
    variant = f"{PFX} Widgets Solution Ltd"  # 'Solutions' -> 'Solution'
    score = fuzz.WRatio(SR._strip_biz_suffix(variant), SR._strip_biz_suffix(seed_name))
    assert SR._REVIEW_LOW <= score < SR._REVIEW_HIGH, f"variant not in review band (score={score})"

    _seed_supplier(f"SUP-{PFX}Widgets", seed_name)
    with get_conn() as c:
        SR.resolve_or_create_supplier(variant, c, doc_type="quote", doc_pk="ZZ1")
        c.commit()
        with c.cursor() as cur:
            cur.execute("SELECT decision, candidate_supplier_id, status FROM proc.bp_supplier_review "
                        "WHERE extracted_name=%s", (variant,))
            row = cur.fetchone()
    assert row is not None, "close-call decision should be flagged for review"
    decision, candidate, status = row
    assert candidate == f"SUP-{PFX}Widgets" and status == "pending"


def test_exact_match_no_review():
    seed_name = f"{PFX} Exactco Ltd"
    _seed_supplier(f"SUP-{PFX}Exactco", seed_name)
    with get_conn() as c:
        sid = SR.resolve_or_create_supplier(seed_name, c)
        c.commit()
        with c.cursor() as cur:
            cur.execute("SELECT count(*) FROM proc.bp_supplier_review WHERE extracted_name ILIKE %s", (PFX + "%",))
            n = cur.fetchone()[0]
    assert sid == f"SUP-{PFX}Exactco" and n == 0  # exact match, no flag
