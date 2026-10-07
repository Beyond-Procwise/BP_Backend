"""Against the real tables: a tile equals direct SQL, a Buyer sees only their deals, and the parts
add up to the whole. Needs PROCWISE_TEST_LIVE_DB=1 (pytest uses a fake database otherwise)."""
import datetime as dt
import os
from types import SimpleNamespace as NS

import pytest

from src.services.report_data import registry as R, service
from src.services.report_data.scope import Scope

pytestmark = pytest.mark.skipif(os.environ.get("PROCWISE_TEST_LIVE_DB", "").lower() not in ("1", "true", "yes"),
                                reason="needs the live database")
F, T = "2023-01-01", "2026-12-31"
ADMIN = Scope("Admin", True, ())


def q(sql, params=None):
    from src.services.db import get_conn
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def value(metric, scope=ADMIN, group_by=None, **kw):
    tile = {"id": "t", "metric": metric, "viz": "kpi", "comparison": "none", **kw}
    out = service.compute(NS(subject="t"), {"from": F, "to": T, "tiles": [tile]}, scope=scope, authorise=lambda a: True)
    t = out["tiles"][0]
    assert t["status"] == "ok", t
    return t["result"]["value"]


FX = ("LEFT JOIN (SELECT currency, rate FROM proc.bp_fx_rates WHERE fetched_at=(SELECT MAX(fetched_at) FROM proc.bp_fx_rates)) fx "
      "ON fx.currency=i.currency")
USD = "i.invoice_amount / NULLIF(COALESCE(fx.rate, 1.0 / NULLIF(i.exchange_rate_to_usd,0)),0)"


@pytest.mark.parametrize("metric,sql", [
    ("committed_spend", f"SELECT SUM({USD}) FROM proc.bp_invoice_trgt i {FX} WHERE i.invoice_date >= '{F}' AND i.invoice_date < '2027-01-01'"),
    ("quote_volume", f"SELECT COUNT(*) FROM proc.bp_quote_trgt WHERE quote_date >= '{F}' AND quote_date < '2027-01-01'"),
    ("cycle_time_to_po", f"SELECT AVG(cycle_days_quote_to_po) FROM proc.bp_deal_overview WHERE COALESCE(deal_date,first_activity_date) >= '{F}' "
                         "AND COALESCE(deal_date,first_activity_date) < '2027-01-01' AND cycle_days_quote_to_po IS NOT NULL"),
    ("opportunity_pipeline", f"SELECT SUM(COALESCE(financial_impact_gbp,0)) FROM proc.bp_opportunity WHERE retired_at IS NULL AND detected_on >= '{F}' AND detected_on < '2027-01-01'"),
    ("duplicate_risk", f"SELECT COUNT(*) FROM proc.bp_extraction_discrepancy WHERE issue_type ILIKE '%duplicate%' AND created_at >= '{F}' AND created_at < '2027-01-01'"),
    ("value_reconciled_rate", f"SELECT 100.0*AVG(CASE WHEN value_reconciled THEN 1.0 ELSE 0.0 END) FROM proc.bp_deal_overview WHERE value_reconciled IS NOT NULL "
                              f"AND COALESCE(deal_date,first_activity_date) >= '{F}' AND COALESCE(deal_date,first_activity_date) < '2027-01-01'"),
])
def test_an_admin_tile_equals_direct_sql(metric, sql):
    direct = float(q(sql)[0][0] or 0)
    assert value(metric) == pytest.approx(round(direct, 2), abs=0.011)


def _buyers():
    return [r[0] for r in q("SELECT buyer_id FROM proc.bp_deal_overview WHERE buyer_id IS NOT NULL GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 4")]


def test_a_buyer_sees_only_their_deals_and_the_buyers_add_up_to_the_admin_total_less_the_unowned():
    a, b = _buyers()[:2]
    sa, sb = Scope("Buyer", False, (a,)), Scope("Buyer", False, (b,))
    va, vb = value("committed_spend", sa), value("committed_spend", sb)
    direct = lambda x: float(q(f"SELECT SUM({USD}) FROM proc.bp_invoice_trgt i {FX} WHERE i.buyer_id=%s AND i.invoice_date >= '{F}' AND i.invoice_date < '2027-01-01'", (x,))[0][0] or 0)
    assert va == pytest.approx(round(direct(a), 2), abs=0.011) and vb == pytest.approx(round(direct(b), 2), abs=0.011)
    both = value("committed_spend", Scope("Buyer", False, (a, b)))
    assert both == pytest.approx(va + vb, abs=0.02)                      # disjoint buyers add
    everyone = [r[0] for r in q("SELECT DISTINCT buyer_id FROM proc.bp_invoice_trgt WHERE buyer_id IS NOT NULL")]
    owned = value("committed_spend", Scope("Buyer", False, tuple(everyone)))
    unowned = float(q(f"SELECT SUM({USD}) FROM proc.bp_invoice_trgt i {FX} WHERE i.buyer_id IS NULL AND i.invoice_date >= '{F}' AND i.invoice_date < '2027-01-01'")[0][0] or 0)
    assert owned + unowned == pytest.approx(value("committed_spend"), abs=0.05)


def test_a_buyer_never_sees_another_buyers_supplier_or_deal_in_a_grouped_tile():
    a = _buyers()[0]
    scope = Scope("Buyer", False, (a,))
    out = service.compute(NS(subject="t"), {"from": F, "to": T, "tiles": [
        {"id": "s", "metric": "committed_spend", "groupBy": ["supplier"], "viz": "table"},
        {"id": "d", "metric": "cycle_time_to_po", "groupBy": ["deal"], "viz": "table"}]}, scope=scope, authorise=lambda x: True)
    mine_sup = {r[0] for r in q("SELECT COALESCE(s.supplier_name, i.supplier_id) FROM proc.bp_invoice_trgt i LEFT JOIN proc.bp_supplier s ON s.supplier_id=i.supplier_id WHERE i.buyer_id=%s", (a,))}
    mine_deal = {r[0] for r in q("SELECT COALESCE(deal_name, deal_id) FROM proc.bp_deal_overview WHERE buyer_id=%s", (a,))}
    assert {r[0] for r in out["tiles"][0]["result"]["rows"]} <= mine_sup
    assert {r[0] for r in out["tiles"][1]["result"]["rows"]} <= mine_deal


def test_findings_are_scoped_through_the_document_and_buyers_plus_unattributed_equal_the_admin_total():
    admin = value("duplicate_risk")
    everyone = tuple(r[0] for r in q("SELECT DISTINCT buyer_id FROM proc.bp_deal_overview WHERE buyer_id IS NOT NULL"))
    attributed = value("duplicate_risk", Scope("Buyer", False, everyone))
    out = service.compute(NS(subject="t"), {"from": F, "to": T, "tiles": [{"id": "f", "metric": "duplicate_risk", "comparison": "none"}]},
                          scope=ADMIN, authorise=lambda x: True)
    unattributed = next(c["count"] for c in out["tiles"][0]["checks"] if c["code"] == "unattributed_findings")
    assert attributed + unattributed == admin
    a = _buyers()[0]
    mine = value("duplicate_risk", Scope("Buyer", False, (a,)))
    assert 0 <= mine <= attributed


def test_every_live_metric_runs_for_every_dimension_it_lists_and_groups_tie():
    scope = Scope("Admin", True, ())
    for m in R.live_metrics():
        for d in m.dimensions:
            out = service.compute(NS(subject="t"), {"from": F, "to": T, "tiles": [{"id": "x", "metric": m.key, "groupBy": [d], "viz": "table"}]},
                                  scope=scope, authorise=lambda x: True)
            assert out["tiles"][0]["status"] == "ok", (m.key, d, out["tiles"][0])


# ---- the real scope and presentation tables ----------------------------------------------------------

def test_scope_grants_and_revokes_take_effect_on_the_next_request_and_fail_closed(monkeypatch):
    from src.services import rbac
    from src.services.db import get_conn
    from src.services.report_data import scope as scope_mod
    sub = "livecheck-buyer-subject"
    monkeypatch.setattr(rbac, "effective_role", lambda p, *a, **k: "Buyer")
    p = NS(subject=sub, claims={})
    try:
        assert scope_mod.resolve(p).assigned_nothing                       # nothing granted: nothing seen
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO proc.bp_user_buyer_scope (subject, buyer_id, granted_by) VALUES (%s,'CC000109','admin-x')", (sub,))
        assert scope_mod.resolve(p).buyers == ("CC000109",)
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("UPDATE proc.bp_user_buyer_scope SET revoked_at=now() WHERE subject=%s", (sub,))
        assert scope_mod.resolve(p).assigned_nothing                       # revoked: gone on the very next read
    finally:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_user_buyer_scope WHERE subject=%s", (sub,))


def test_a_scope_can_only_be_granted_once_while_active():
    from src.services.db import get_conn
    sub = "livecheck-dup-subject"
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO proc.bp_user_buyer_scope (subject, buyer_id, granted_by) VALUES (%s,'CC1','a')", (sub,))
            with pytest.raises(Exception, match="ix_bp_user_buyer_scope_active"):
                cur.execute("INSERT INTO proc.bp_user_buyer_scope (subject, buyer_id, granted_by) VALUES (%s,'CC1','a')", (sub,))
    finally:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_user_buyer_scope WHERE subject=%s", (sub,))


def test_presentation_activation_is_logged_per_session_in_the_real_table(monkeypatch):
    import time
    from src.services import rbac
    from src.services.db import get_conn
    from src.services.report_data import mode
    monkeypatch.setattr(rbac, "effective_role", lambda p, *a, **k: "Admin")
    now = int(time.time())
    p = NS(subject="livecheck-admin", claims={"jti": "s1", "auth_time": now, "exp": now + 600})
    try:
        assert not mode.is_active(p)
        mode.activate(p)
        assert mode.is_active(p)
        assert not mode.is_active(NS(subject="livecheck-admin", claims={"jti": "s2", "auth_time": now, "exp": now + 600}))
        mode.deactivate(p)
        assert not mode.is_active(p)
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT event FROM proc.bp_presentation_log WHERE subject='livecheck-admin' ORDER BY log_id")
            assert [r[0] for r in cur.fetchall()] == ["activate", "deactivate"]
    finally:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_presentation_log WHERE subject='livecheck-admin'")


def test_a_buyer_is_offered_only_their_own_suppliers_as_filter_choices():
    from src.services.report_data import live
    a = _buyers()[0]
    mine = {r[0] for r in q("SELECT DISTINCT i.supplier_id::text FROM proc.bp_invoice_trgt i WHERE i.buyer_id=%s AND i.supplier_id IS NOT NULL", (a,))}
    offered = {k for k, _ in live.values("invoice", "supplier", None, Scope("Buyer", False, (a,)), 500)}
    assert offered == mine
    everyone = {k for k, _ in live.values("invoice", "supplier", None, ADMIN, 100000)}
    assert mine <= everyone and len(everyone) > len(mine)
    assert live.values("invoice", "supplier", None, Scope("Buyer", False, ()), 50) == []
    hit = live.values("invoice", "supplier", "cloud", ADMIN, 10)
    assert hit and all("cloud" in l.lower() for _, l in hit)


def test_a_grant_can_be_listed_found_by_search_and_revoked_through_the_api(monkeypatch):
    from types import SimpleNamespace as NS
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routers import reports as rr
    from src.services import rbac
    from src.services.db import get_conn
    monkeypatch.setattr(rr, "gate", lambda *a, **k: None)
    monkeypatch.setattr(rbac, "effective_role", lambda p, *a, **k: "Admin")
    app = FastAPI(); app.include_router(rr.router)
    app.dependency_overrides[rr.require_user] = lambda: NS(subject="livecheck-admin", claims={})
    c = TestClient(app)
    sub, code = "livecheck-grantee", _buyers()[0]
    try:
        found = c.get("/reports/scope/buyers", params={"q": code[:5]}).json()["data"]
        assert any(b["buyer_id"] == code and b["deals"] > 0 for b in found)
        assert c.post("/reports/scope", json={"subject": sub, "buyer_id": code}).status_code == 201
        assert c.post("/reports/scope", json={"subject": sub, "buyer_id": code}).status_code == 201      # twice: still one active grant
        listed = c.get("/reports/scope", params={"subject": sub}).json()["data"]
        assert [(g["subject"], g["buyer_id"], g["granted_by"]) for g in listed] == [(sub, code, "livecheck-admin")]
        assert c.delete("/reports/scope", params={"subject": sub, "buyer_id": code}).json() == {"revoked": 1}
        assert c.get("/reports/scope", params={"subject": sub}).json()["data"] == []
        assert c.delete("/reports/scope", params={"subject": sub, "buyer_id": code}).json() == {"revoked": 0}
    finally:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_user_buyer_scope WHERE subject=%s", (sub,))
