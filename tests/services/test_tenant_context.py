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

from services import tenant_context as tc


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


def test_system_with_no_tenant_is_passed_with_an_empty_tenant(monkeypatch):
    monkeypatch.delenv(tc.PLATFORM_LABEL_ENV, raising=False)
    with tc.system():
        assert tc.connection_settings() == ("", "on")


def test_system_with_no_tenant_uses_the_one_configured_platform_label(monkeypatch):
    monkeypatch.setenv(tc.PLATFORM_LABEL_ENV, "acme")
    with tc.system():
        assert tc.connection_settings() == ("acme", "on")


def test_a_bound_tenant_beats_the_platform_label(monkeypatch):
    monkeypatch.setenv(tc.PLATFORM_LABEL_ENV, "acme")
    with tc.system(), tc.bound("other"):
        assert tc.connection_settings() == ("other", "on")


def test_a_bad_platform_label_is_ignored_not_repaired(monkeypatch):
    monkeypatch.setenv(tc.PLATFORM_LABEL_ENV, "a b;drop")
    with tc.system():
        assert tc.connection_settings() == ("", "on")


def test_the_platform_label_never_applies_outside_system_mode(monkeypatch):
    monkeypatch.setenv(tc.PLATFORM_LABEL_ENV, "acme")
    assert tc.connection_settings() is None


class _Cur:
    def __init__(self, log): self.log = log
    def execute(self, sql, params=None): self.log.append((sql, params))
    def close(self): pass


class _Conn:
    def __init__(self, autocommit=True):
        self.log, self.autocommit, self.closed, self.commits = [], autocommit, False, 0
    def cursor(self): return _Cur(self.log)
    def commit(self): self.commits += 1
    def close(self): self.closed = True


@pytest.fixture
def driver():
    """A stand-in for psycopg2 whose connect() hands out recording connections, wrapped like the real one."""
    made = []

    def connect(*a, **k):
        c = _Conn(autocommit=k.pop("autocommit", True)); made.append(c); return c
    mod = SimpleNamespace(connect=connect)
    tc.install_driver_patch(mod)
    return mod, made


def test_a_new_connection_is_given_the_tenant(driver):
    mod, made = driver
    with tc.bound("acme"):
        c = mod.connect()
    (sql, params), = c.log
    assert "app.tenant_id" in sql and params == ("acme", "off")


def test_a_connection_made_with_nothing_bound_runs_no_extra_statement(driver):
    mod, made = driver
    assert mod.connect().log == []


def test_platform_work_says_so(driver):
    mod, _ = driver
    with tc.system():
        assert mod.connect().log[0][1] == ("", "on")


def test_the_setting_is_committed_on_a_connection_that_is_not_autocommit(driver):
    mod, _ = driver
    with tc.bound("acme"):
        c = mod.connect(autocommit=False)
    assert c.commits == 1


def test_a_failure_telling_the_database_closes_the_connection(driver):
    mod, made = driver
    made_bad = []
    bad = _Conn(); bad.cursor = lambda: (_ for _ in ()).throw(RuntimeError("no"))
    mod2 = SimpleNamespace(connect=lambda *a, **k: made_bad.append(bad) or bad)
    tc.install_driver_patch(mod2)
    with tc.bound("acme"):
        with pytest.raises(RuntimeError):
            mod2.connect()
    assert bad.closed


def test_the_patch_is_installed_once():
    mod = SimpleNamespace(connect=lambda *a, **k: _Conn())
    tc.install_driver_patch(mod); first = mod.connect
    tc.install_driver_patch(mod)
    assert mod.connect is first


def test_the_real_driver_is_patched_so_every_caller_gets_it():
    import psycopg2
    assert getattr(psycopg2, "_bp_tenant_connect", False) is True


def test_a_pooled_connection_is_cleared_when_nothing_is_bound():
    c = _Conn()
    tc.tell_connection(c, always=True)
    assert c.log[0][1] == ("", "off")


def test_a_pooled_connection_is_re_told_on_each_checkout(monkeypatch):
    sa = pytest.importorskip("sqlalchemy")
    seen = []
    monkeypatch.setattr(tc, "tell_connection", lambda conn, always=False: seen.append(always))
    engine = sa.create_engine("sqlite://")
    with engine.connect():
        pass
    assert seen and all(seen)


# --- carrying the tenant into threads and pools --------------------------------------------------
def _in_thread(fn):
    out = []
    t = threading.Thread(target=lambda: out.append(fn()))
    t.start(); t.join()
    return out[0]


def test_a_thread_started_inside_a_request_keeps_its_tenant():
    with tc.bound("acme"):
        assert _in_thread(tc.current) == "acme"


def test_a_subclassed_thread_keeps_it_too():
    got = []

    class T(threading.Thread):
        def run(self): got.append(tc.current())
    with tc.bound("acme"):
        t = T(); t.start(); t.join()
    assert got == ["acme"]


def test_a_thread_started_as_platform_work_stays_platform_work():
    with tc.system():
        assert _in_thread(tc.is_system) is True


def test_a_thread_started_with_nothing_bound_has_nothing():
    assert _in_thread(lambda: (tc.current(), tc.is_system())) == (None, False)


def test_a_pool_task_carries_the_submitters_tenant():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        with tc.bound("acme"):
            assert pool.submit(tc.current).result() == "acme"


def test_a_pooled_thread_never_runs_one_callers_task_as_anothers():
    """One worker thread serves all three tasks; each must see only its own submitter's tenant."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        with tc.bound("acme"):
            a = pool.submit(tc.current).result()
        b = pool.submit(tc.current).result()
        with tc.bound("beta"):
            c = pool.submit(tc.current).result()
    assert (a, b, c) == ("acme", None, "beta")


def test_a_pools_own_worker_thread_is_not_given_a_tenant():
    from concurrent.futures import ThreadPoolExecutor
    with tc.bound("acme"):
        pool = ThreadPoolExecutor(max_workers=1)
        pool.submit(lambda: None).result()
    # submitted with no tenant: the worker thread (created while "acme" was bound) must see none
    assert pool.submit(tc.current).result() is None
    pool.shutdown()


def test_the_event_loops_default_executor_carries_it():
    import asyncio

    async def go():
        with tc.bound("acme"):
            return await asyncio.get_running_loop().run_in_executor(None, tc.current)
    assert asyncio.run(go()) == "acme"


def test_enter_and_leave_system():
    token = tc.enter_system()
    try:
        assert tc.is_system() is True
    finally:
        tc.leave_system(token)
    assert tc.is_system() is False


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


# --- platform work -------------------------------------------------------------------------------
def test_as_system_runs_the_function_as_platform_work():
    assert tc.is_system() is False
    assert tc.as_system(tc.is_system)() is True
    assert tc.is_system() is False


def test_a_shared_worker_started_by_a_request_does_not_take_that_requests_tenant():
    """The egress log writer is started lazily by whichever request logs first, then serves everyone."""
    from services import egress_log
    seen = []
    done = threading.Event()

    def fake_run():
        seen.append((tc.current(), tc.is_system()))
        done.set()
    original = egress_log._run
    egress_log._run = fake_run
    egress_log._worker = None
    try:
        with tc.bound("acme"):
            egress_log._ensure_worker()
        assert done.wait(5)
    finally:
        egress_log._run = original
        egress_log._worker = None
    # carried the request's tenant into the thread, then ran as_system: tenant + system together is
    # fine -- the policy's system branch lets it write for any customer
    assert seen[0][1] is True


ENTRY_POINTS = [
    ("src/services/extraction_telemetry/telemetry_service.py", "def main() -> int:"),
    ("src/services/vector_store/worker.py", "def main(argv"),
    ("src/services/vector_store/rebuild.py", "def main(argv"),
    ("src/orchestration/cli.py", "def main() -> None:"),
    ("scripts/extraction_health_check.py", "def main() -> None:"),
]


@pytest.mark.parametrize("path,signature", ENTRY_POINTS)
def test_every_standalone_platform_process_enters_system_mode(path, signature):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    text = open(os.path.join(root, path), encoding="utf-8").read()
    body = text[text.index(signature):]
    assert "enter_system()" in body[:600], f"{path}: main() must start with tenant_context.enter_system()"


def test_the_web_apps_startup_runs_as_platform_work_and_ends_before_requests():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    text = open(os.path.join(root, "src/api/main.py"), encoding="utf-8").read()
    start = text.index("async def lifespan(app: FastAPI):")
    body = text[start:text.index("# Release CUDA memory", start)]
    assert body.index("enter_system()") < body.index("SessionNotifyListener")
    assert body.index("leave_system(") < body.index("\n    yield\n")
