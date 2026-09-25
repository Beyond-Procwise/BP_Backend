from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    from src.api.main import app
    from src.api.routers import value_ledger as router_mod
    app.dependency_overrides[router_mod.require_user] = lambda: SimpleNamespace(subject="buyer-1")
    yield TestClient(app), router_mod
    app.dependency_overrides.clear()


def test_outcome_requires_a_user():
    from src.api.main import app
    r = TestClient(app).post("/value/findings/1/outcome", json={"outcome": "claimed"})
    assert r.status_code in (401, 403, 503)


def test_outcome_passes_the_token_actor_not_the_body(client, monkeypatch):
    c, mod = client
    seen = {}
    monkeypatch.setattr(mod.value_ledger, "record_finding_outcome",
                        lambda did, outcome, amount, currency, **kw: seen.update(kw, did=did) or
                        {"outcome_id": 9, "state": outcome, "amount_gbp": 10})
    r = c.post("/value/findings/42/outcome",
               json={"outcome": "claimed", "amount": "10", "currency": "GBP", "actor": "forged"})
    assert r.status_code == 200 and r.json()["state"] == "claimed"
    assert seen["actor"] == "buyer-1" and seen["did"] == 42


@pytest.mark.parametrize("code,status", [("finding_already_moved", 409), ("no_open_claim", 409),
                                         ("already_corrected", 409), ("not_current", 409),
                                         ("not_found", 404),
                                         ("evidence_required", 422), ("invalid_amount", 422),
                                         ("not_a_money_finding", 422)])
def test_ledger_errors_map_to_http(client, monkeypatch, code, status):
    c, mod = client

    def _raise(*a, **kw):
        raise mod.value_ledger.LedgerError(code, "nope")
    monkeypatch.setattr(mod.value_ledger, "settle_claim", _raise)
    r = c.post("/value/findings/1/settle", json={"outcome": "recovered"})
    assert r.status_code == status
    assert r.json()["detail"]["error"] == code


def test_history_is_readable(client, monkeypatch):
    c, mod = client
    monkeypatch.setattr(mod.value_ledger, "finding_outcomes", lambda did: {
        "state": None, "history": [], "prefill": {"amount": "5.00", "currency": "GBP", "is_money": True}})
    r = c.get("/value/findings/3/outcomes")
    assert r.status_code == 200 and r.json()["prefill"]["amount"] == "5.00"
