import contextlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.endpoint_gate import NotPermitted
from api.routers import catalog as cr
from src.services.catalog_import import ImportResult
from src.services.sell_side._db import NotFound, StateConflict

CALLER, OTHER = "sub-real-caller", "sub-someone-else"


class _P:
    subject = CALLER


@pytest.fixture
def client(monkeypatch):
    gates = []
    monkeypatch.setattr(cr, "gate", lambda action, *a, **k: gates.append(action))
    monkeypatch.setattr(cr, "get_conn", lambda: contextlib.nullcontext("CONN"))
    monkeypatch.setattr(cr, "_max_upload_bytes", lambda: 1000)
    app = FastAPI()
    app.include_router(cr.router)
    app.dependency_overrides[cr.require_user] = lambda: _P()
    c = TestClient(app)
    c.gates = gates
    return c


def _upload(client, name="feed.csv", body=b"SKU,Description,Ccy\nA1,W,GBP\n"):
    return client.post("/catalog/import", files={"file": (name, body, "text/csv")},
                       data={"distributor_id": "SUP-1", "feed_name": "March",
                             "mapping_profile": "p1", "price_effective": "2026-03-01"})


def test_an_import_is_attributed_to_the_token(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(cr.catalog_import, "import_catalog",
                        lambda **k: seen.update(k) or ImportResult(status="imported"))
    r = _upload(client)
    assert r.status_code == 200, r.text
    assert seen["imported_by"] == CALLER
    assert client.gates == ["catalog.write"]


def test_an_oversize_feed_is_refused_before_import(client, monkeypatch):
    monkeypatch.setattr(cr.catalog_import, "import_catalog",
                        lambda **k: pytest.fail("import must not run"))
    assert _upload(client, body=b"x" * 1001).status_code == 413


def test_a_file_that_is_not_a_spreadsheet_is_refused(client, monkeypatch):
    monkeypatch.setattr(cr.catalog_import, "import_catalog",
                        lambda **k: pytest.fail("import must not run"))
    assert _upload(client, name="feed.pdf").status_code == 415


def test_a_refused_gate_stops_the_import(client, monkeypatch):
    def _refuse(*a, **k):
        raise NotPermitted("no")
    monkeypatch.setattr(cr, "gate", _refuse)
    monkeypatch.setattr(cr.catalog_import, "import_catalog",
                        lambda **k: pytest.fail("import must not run"))
    assert _upload(client).status_code == 403


def test_a_match_decision_is_the_callers_not_the_bodys(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(cr.catalog_match, "confirm_match",
                        lambda conn, mid, reviewer: seen.update(reviewer=reviewer) or {"match_id": mid})
    r = client.post("/catalog/matches/5/confirm", json={"reviewer": OTHER})
    assert r.status_code == 200 and seen["reviewer"] == CALLER


@pytest.mark.parametrize("exc,code", [(NotFound("x"), 404), (StateConflict("x"), 409),
                                      (ValueError("x"), 422)])
def test_service_errors_map_to_status_codes(client, monkeypatch, exc, code):
    def _raise(*a, **k):
        raise exc
    monkeypatch.setattr(cr.catalog_match, "reject_match", _raise)
    assert client.post("/catalog/matches/5/reject").status_code == code


def test_an_unknown_distributor_import_is_a_422_not_a_500(client, monkeypatch):
    """Finding 6: import_catalog now raises ValueError for an unknown
    distributor_id, and the router must map that to 422, not let it fall
    through to the unhandled-exception 500 handler."""
    def _raise(**k):
        raise ValueError("distributor 'SUP-1' is not a supplier")
    monkeypatch.setattr(cr.catalog_import, "import_catalog", _raise)
    assert _upload(client).status_code == 422
