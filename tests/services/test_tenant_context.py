"""Which customer a request is for: from the verified token only, carried to the database connection.

Fails closed: no valid claim means NO tenant (never a default), and a connection with no tenant says
nothing, so row-level security will show it zero rows.
"""
from __future__ import annotations

import importlib
import os
import sys
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from services import db, tenant_context as tc


# --- reading the label ---------------------------------------------------------
def test_the_tenant_is_the_verified_claim():
    assert tc.from_claims({"custom:subscription_id": "acme-01"}) == "acme-01"
    assert tc.from_principal(SimpleNamespace(claims={"custom:subscription_id": "acme-01"})) == "acme-01"


@pytest.mark.parametrize("bad", [None, "", " ", "a b", "x" * 65, "drop;table", "a'b", 7, ["a"], "é"])
def test_a_missing_or_odd_claim_is_no_tenant_never_a_repaired_one(bad):
    assert tc.from_claims({"custom:subscription_id": bad}) is None


def test_no_claims_no_principal_no_tenant():
    assert tc.from_claims({}) is None and tc.from_claims(None) is None
    assert tc.from_principal(None) is None


def test_other_claim_names_do_not_count():
    assert tc.from_claims({"subscription_id": "acme", "tenant_id": "acme"}) is None


# --- scoping -------------------------------------------------------------------
def test_bound_sets_and_restores():
    assert tc.current() is None
    with tc.bound("a"):
        assert tc.current() == "a"
        with tc.bound("b"):
            assert tc.current() == "b"
        assert tc.current() == "a"
    assert tc.current() is None


def test_binding_a_bad_value_binds_nothing():
    with tc.bound("a"):
        with tc.bound("not valid!"):
            assert tc.current() is None
        assert tc.current() == "a"


def test_system_is_scoped_and_off_by_default():
    assert tc.is_system() is False
    with tc.system():
        assert tc.is_system() is True
    assert tc.is_system() is False


def test_a_new_thread_starts_with_no_tenant():
    seen = []
    with tc.bound("a"):
        t = threading.Thread(target=lambda: seen.append(tc.current()))
        t.start(); t.join()
    assert seen == [None]


def test_both_import_names_share_one_context():
    other = importlib.import_module("src.services.tenant_context")
    assert other is tc
    with tc.bound("a"):
        assert other.current() == "a"


# --- what a connection is told -------------------------------------------------
def test_nothing_bound_says_nothing():
    assert tc.connection_settings() is None


def test_a_tenant_is_passed_and_system_is_off():
    with tc.bound("a"):
        assert tc.connection_settings() == ("a", "off")


def test_system_with_no_tenant_is_passed_with_an_empty_tenant():
    with tc.system():
        assert tc.connection_settings() == ("", "on")


class _Cur:
    def __init__(self, log): self.log = log
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params=None): self.log.append((sql, params))


class _Conn:
    def __init__(self): self.log, self.autocommit, self.closed = [], False, False
    def cursor(self): return _Cur(self.log)
    def close(self): self.closed = True


@pytest.fixture
def real_path(monkeypatch):
    conns = []
    monkeypatch.setenv("PROCWISE_TEST_LIVE_DB", "1")
    monkeypatch.setattr(db, "_pg_dsn", lambda: "host=x")

    def connect(dsn):
        c = _Conn(); conns.append(c); return c
    monkeypatch.setattr(db, "psycopg2", SimpleNamespace(connect=connect))
    return conns


def test_get_conn_gives_the_connection_the_tenant(real_path):
    with tc.bound("acme"):
        with db.get_conn() as c:
            (sql, params), = c.log
    assert "app.tenant_id" in sql and params == ("acme", "off")


def test_get_conn_with_nothing_bound_runs_no_extra_statement(real_path):
    with db.get_conn() as c:
        assert c.log == []


def test_get_conn_for_system_work_says_so(real_path):
    with tc.system():
        with db.get_conn() as c:
            assert c.log[0][1] == ("", "on")


def test_a_failure_telling_the_database_closes_the_connection(real_path, monkeypatch):
    monkeypatch.setattr(db, "psycopg2", SimpleNamespace(connect=lambda dsn: _BadConn(real_path)))
    with tc.bound("acme"):
        with pytest.raises(RuntimeError):
            with db.get_conn():
                pass
    assert real_path[-1].closed


class _BadConn(_Conn):
    def __init__(self, sink):
        super().__init__(); sink.append(self)

    def cursor(self): raise RuntimeError("no")


# --- the request middleware ----------------------------------------------------
@pytest.fixture
def app_client(monkeypatch):
    from api import auth
    from api.tenant_middleware import TenantContextMiddleware

    claims_by_token = {"good": {"custom:subscription_id": "acme"}, "noclaim": {}, "other": {"custom:subscription_id": "beta"}}

    def principal_from_token(tok):
        if tok not in claims_by_token:
            raise auth.AuthError("bad")
        return SimpleNamespace(subject="u", claims=claims_by_token[tok])
    monkeypatch.setattr(auth, "principal_from_token", principal_from_token)

    app = FastAPI()
    app.add_middleware(TenantContextMiddleware)

    @app.get("/sync")
    def sync_endpoint():          # runs in the threadpool
        return {"tenant": tc.current()}

    @app.get("/async")
    async def async_endpoint():
        return {"tenant": tc.current()}
    return TestClient(app)


def _h(tok): return {"Authorization": f"Bearer {tok}"}


@pytest.mark.parametrize("path", ["/sync", "/async"])
def test_the_endpoint_sees_the_tenant_from_the_token(app_client, path):
    assert app_client.get(path, headers=_h("good")).json() == {"tenant": "acme"}


def test_requests_do_not_leak_into_each_other(app_client):
    assert app_client.get("/sync", headers=_h("good")).json()["tenant"] == "acme"
    assert app_client.get("/sync", headers=_h("other")).json()["tenant"] == "beta"
    assert app_client.get("/sync").json()["tenant"] is None
    assert tc.current() is None


@pytest.mark.parametrize("headers", [{}, _h("noclaim"), _h("unknown"), {"Authorization": "Basic abc"},
                                     {"Authorization": "Bearer "}])
def test_no_valid_token_or_claim_means_no_tenant(app_client, headers):
    assert app_client.get("/sync", headers=headers).json() == {"tenant": None}


def test_a_tenant_header_or_query_is_ignored(app_client):
    r = app_client.get("/sync?tenant_id=evil", headers={"x-customer-id": "evil", "x-tenant-id": "evil"})
    assert r.json() == {"tenant": None}


def test_the_middleware_clears_the_tenant_when_the_request_ends(monkeypatch):
    """Same task, two requests: what the first bound must be gone before the second starts."""
    import asyncio
    from api import auth
    from api.tenant_middleware import TenantContextMiddleware

    monkeypatch.setattr(auth, "principal_from_token",
                        lambda tok: SimpleNamespace(subject="u", claims={"custom:subscription_id": "acme"}))
    seen = []

    async def inner(scope, receive, send):
        seen.append(tc.current())

    mw = TenantContextMiddleware(inner)
    scope = {"type": "http", "headers": [(b"authorization", b"Bearer good")]}

    async def go():
        await mw(scope, None, None)
        after_first = tc.current()
        await mw({"type": "http", "headers": []}, None, None)
        return after_first

    assert asyncio.run(go()) is None
    assert seen == ["acme", None]
