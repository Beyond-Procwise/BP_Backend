"""Supplier web-research: grounding guard, fill-empty-only apply, sensitive-field guard."""
import pytest

from src.services.db import get_conn
from src.services.supplier_enrichment import research as R

IDP = "SUP-ZZRES"


def _clean():
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_supplier_enrichment WHERE supplier_id ILIKE %s", (IDP+"%",))
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


# ---- pure functions (no DB) ----
def test_parse_fields_extracts_json():
    c = 'ok {"fields": {"website_url": {"value":"https://x.com","source_url":"https://x.com","confidence":0.9}}} end'
    assert R._parse_fields(c)["website_url"]["value"] == "https://x.com"


def test_ground_drops_uncited_sensitive_and_unknown():
    fields = {
        "website_url": {"value": "https://acme.com", "source_url": "https://acme.com/about", "confidence": 0.9},
        "city": {"value": "London", "source_url": "https://evil.example/x", "confidence": 0.9},  # host not visited
        "vat_number": {"value": "GB123", "source_url": "https://acme.com", "confidence": 0.9},   # sensitive
        "country": {"value": "unknown", "source_url": "https://acme.com", "confidence": 0.9},     # unknown
    }
    kept = R._ground(fields, {"https://acme.com/about", "https://acme.com"})
    assert set(kept) == {"website_url"}


# ---- DB-backed apply / end-to-end (mocked research loop) ----
def _seed(sid, name, **cols):
    keys = ["supplier_id", "supplier_name", "trading_name"] + list(cols)
    vals = [sid, name, name] + list(cols.values())
    ph = ",".join(["%s"] * len(keys))
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute(f"INSERT INTO proc.bp_supplier ({','.join(keys)}, created_date, created_by) VALUES ({ph}, NOW(), 'test')", vals)
        c.commit()


def test_apply_fills_empty_only_and_skips_conflicts(monkeypatch):
    _seed(f"{IDP}1", "Acme Widgets Ltd", country="GB")  # country already set, website empty
    canned = ('{"fields": {'
              '"website_url": {"value":"https://acme.example","source_url":"https://acme.example/about","confidence":0.9},'
              '"country": {"value":"US","source_url":"https://acme.example/about","confidence":0.9},'
              '"vat_number": {"value":"GB999","source_url":"https://acme.example/about","confidence":0.9}}}')
    monkeypatch.setattr(R, "_run_loop", lambda name: (canned, {"https://acme.example/about"}))

    with get_conn() as c:
        res = R.research_and_enrich(f"{IDP}1", c)
    assert "website_url" in res["applied"]         # empty → filled
    assert "country" not in res["applied"]          # non-empty → not overwritten
    assert "vat_number" not in res["fields"]        # sensitive → never even kept

    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT website_url, country FROM proc.bp_supplier WHERE supplier_id=%s", (f"{IDP}1",))
        website, country = cur.fetchone()
    assert website == "https://acme.example" and country == "GB"  # filled empty, kept existing


def test_uncited_fields_not_applied(monkeypatch):
    _seed(f"{IDP}2", "Beta Traders Ltd")  # all empty
    canned = '{"fields": {"website_url": {"value":"https://beta.example","source_url":"https://hallucinated.example","confidence":0.95}}}'
    monkeypatch.setattr(R, "_run_loop", lambda name: (canned, {"https://realsource.example"}))  # cite ≠ visited
    with get_conn() as c:
        res = R.research_and_enrich(f"{IDP}2", c)
    assert res["fields"] == {} and res["applied"] == {}  # dropped as ungrounded
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT website_url FROM proc.bp_supplier WHERE supplier_id=%s", (f"{IDP}2",))
        assert cur.fetchone()[0] in (None, "")
