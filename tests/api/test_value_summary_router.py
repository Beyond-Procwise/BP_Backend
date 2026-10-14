import pytest
from fastapi.testclient import TestClient

from api.auth import require_user
from src.api.routers.value_summary import trim_findings_for_response


@pytest.fixture
def client():
    """/spendiq/value-summary is mounted through _AUTHENTICATED_ROUTERS (src/api/main.py),
    so with ASK_AUTH_MODE=enforce a real caller needs a bearer token -- override the same
    require_user object the router imports from api.auth, and clear it afterwards so this
    test's override never leaks into another test module sharing the singleton app."""
    from src.api.main import app
    app.dependency_overrides[require_user] = lambda: None
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(require_user, None)


def test_value_summary_endpoint(monkeypatch, client):
    import src.services.value_summary_service as vss
    monkeypatch.setattr(vss, "build_value_summary", lambda conn=None: {
        "verified_found_gbp": 950.0, "recovered_gbp": 0.0, "potential_gbp": 0.0,
        "finding_count": 1, "since": None, "by_supplier": [], "findings": [],
        "sources": {"discrepancies": "ok", "opportunities": "ok", "benchmark": "ok"},
    })
    r = client.get("/spendiq/value-summary")
    assert r.status_code == 200
    body = r.json()
    assert body["verified_found_gbp"] == 950.0
    assert "generated_at" in body
    assert body["findings_total"] == 0
    assert body["findings_shown"] == 0


# --------------------------------------------------------------------------
# trim_findings_for_response (Task 12 / Ruling R15)
# --------------------------------------------------------------------------

def _f(id_, amount_gbp=None, ledger_state=None, superseded_by=None):
    return {"id": id_, "amount_gbp": amount_gbp, "ledger_state": ledger_state,
            "superseded_by": superseded_by}


def test_trim_keeps_every_claimed_finding_even_past_the_limit():
    claimed = [_f(f"c{i}", amount_gbp=10.0, ledger_state="claimed") for i in range(5)]
    live = [_f(f"l{i}", amount_gbp=1.0) for i in range(5)]
    out = trim_findings_for_response(claimed + live, limit=2)
    kept_ids = {f["id"] for f in out}
    assert kept_ids.issuperset({f["id"] for f in claimed})


def test_trim_orders_live_findings_by_amount_descending_none_last():
    live = [_f("small", amount_gbp=10.0), _f("big", amount_gbp=100.0),
            _f("unknown", amount_gbp=None), _f("mid", amount_gbp=50.0)]
    out = trim_findings_for_response(live, limit=10)
    assert [f["id"] for f in out] == ["big", "mid", "small", "unknown"]


def test_trim_respects_the_limit_when_no_claims_involved():
    live = [_f(f"l{i}", amount_gbp=float(i)) for i in range(10)]
    out = trim_findings_for_response(live, limit=3)
    assert len(out) == 3
    # highest amounts first
    assert [f["id"] for f in out] == ["l9", "l8", "l7"]


def test_trim_puts_superseded_findings_last_and_only_while_room():
    claimed = [_f("c0", amount_gbp=5.0, ledger_state="claimed")]
    live = [_f("live0", amount_gbp=1.0)]
    superseded = [_f("s0", amount_gbp=999.0, superseded_by="c0"),
                  _f("s1", amount_gbp=998.0, superseded_by="c0")]
    findings = claimed + live + superseded
    out = trim_findings_for_response(findings, limit=2)
    # claimed + live already fill the limit -- no room for superseded
    assert [f["id"] for f in out] == ["c0", "live0"]

    out2 = trim_findings_for_response(findings, limit=3)
    assert [f["id"] for f in out2] == ["c0", "live0", "s0"]


def test_trim_does_not_mutate_or_touch_summary_totals():
    findings = [_f(f"l{i}", amount_gbp=float(i)) for i in range(20)]
    original_len = len(findings)
    trim_findings_for_response(findings, limit=5)
    # the input list itself is untouched -- trimming only shapes the response copy
    assert len(findings) == original_len


def test_router_trims_findings_but_leaves_summary_totals_untouched(monkeypatch, client):
    import src.services.value_summary_service as vss

    synthetic = [_f(f"d:{i}", amount_gbp=float(i)) for i in range(500)]
    monkeypatch.setattr(vss, "build_value_summary", lambda conn=None: {
        "verified_found_gbp": 12345.67, "recovered_gbp": 100.0, "potential_gbp": 200.0,
        "finding_count": 500, "since": None, "by_supplier": [{"supplier_name": "Acme"}],
        "by_month": [{"month": "2026-09"}], "in_play_gbp": 300.0,
        "findings": synthetic,
        "sources": {"discrepancies": "ok", "opportunities": "ok", "benchmark": "ok"},
    })
    r = client.get("/spendiq/value-summary")
    assert r.status_code == 200
    body = r.json()
    assert body["findings_total"] == 500
    assert body["findings_shown"] == 200
    assert len(body["findings"]) == 200
    # totals are exactly what the service returned -- unaffected by trimming
    assert body["verified_found_gbp"] == 12345.67
    assert body["recovered_gbp"] == 100.0
    assert body["potential_gbp"] == 200.0
    assert body["finding_count"] == 500
    assert body["by_supplier"] == [{"supplier_name": "Acme"}]
    assert body["by_month"] == [{"month": "2026-09"}]
    assert body["in_play_gbp"] == 300.0
