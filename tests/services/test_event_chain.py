"""Event-driven chain: a _raw->_stg promotion fires the downstream pipeline
(stg->trgt -> deal-linking -> mining), coalescing bursts."""
from __future__ import annotations

import threading


def _bare(orchestrator=object()):
    from src.services.backend_scheduler import BackendScheduler
    s = BackendScheduler.__new__(BackendScheduler)
    s._jobs = {}
    s._lock = threading.Lock()
    s._orchestrator = orchestrator
    return s


def test_on_doc_promoted_schedules_downstream_once(monkeypatch):
    from src.services.backend_scheduler import BackendScheduler
    s = _bare()
    calls = []

    def fake_submit_once(name, runner, *, initial_delay=None):
        calls.append(name)
        s._jobs[name] = object()   # now pending

    monkeypatch.setattr(s, "submit_once", fake_submit_once)
    s._on_doc_promoted({"ok": True}, {})
    assert calls == [BackendScheduler.DOWNSTREAM_CHAIN_JOB_NAME]
    # a second promotion while a run is queued must coalesce (no duplicate)
    s._on_doc_promoted({"ok": True}, {})
    assert calls == [BackendScheduler.DOWNSTREAM_CHAIN_JOB_NAME]


def test_run_downstream_chain_order(monkeypatch):
    import src.services.linking_engine as le
    import src.services.deal_assignment_service as das
    order = []
    monkeypatch.setattr(le, "promote_ready", lambda: order.append("promote") or {"promoted": 1})
    monkeypatch.setattr(das, "assign_deals", lambda: order.append("assign") or {"forward_linked": 1})
    s = _bare()
    monkeypatch.setattr(s, "_chain_opportunity_mining", lambda r: order.append("mine"))
    s._run_downstream_chain()
    assert order == ["promote", "assign", "mine"]


def test_run_downstream_chain_stops_if_assign_fails(monkeypatch):
    import src.services.linking_engine as le
    import src.services.deal_assignment_service as das
    order = []
    monkeypatch.setattr(le, "promote_ready", lambda: order.append("promote") or {})
    def boom():
        order.append("assign-fail")
        raise RuntimeError("db down")
    monkeypatch.setattr(das, "assign_deals", boom)
    s = _bare()
    monkeypatch.setattr(s, "_chain_opportunity_mining", lambda r: order.append("mine"))
    s._run_downstream_chain()   # must not raise
    assert order == ["promote", "assign-fail"]   # mining not reached


def test_run_listener_invokes_on_promoted(monkeypatch):
    # Exercise the listener's callback hook without a real DB by faking the
    # psycopg2 connection + a single NOTIFY, and stubbing the promote call.
    import src.services.extraction.promotion as promo

    class _Notify:
        payload = '{"raw_id": 5, "doc_type": "invoice"}'

    class _FakeConn:
        def __init__(self):
            self.notifies = [_Notify()]
            self.autocommit = True
        def cursor(self):
            class _C:
                def execute(self, *a, **k): pass
            return _C()
        def poll(self): pass
        def close(self): pass

    monkeypatch.setattr(promo, "psycopg2", type("P", (), {"connect": staticmethod(lambda **k: _FakeConn())}))
    monkeypatch.setattr(promo, "Settings", lambda: type("S", (), {
        "db_host": "h", "db_name": "d", "db_user": "u", "db_password": "p", "db_port": "5432"})())
    # one select cycle returns readable, then stop
    seq = iter([([object()], [], []), ([], [], [])])
    monkeypatch.setattr(promo, "select", type("Sel", (), {"select": staticmethod(lambda *a, **k: next(seq))}))
    monkeypatch.setattr(promo, "apply_hitl_fixes_and_promote", lambda rid, dt: {"ok": True, "process_monitor_id": 7})

    seen = []
    stop = threading.Event()

    def on_promoted(result, payload):
        seen.append((result["process_monitor_id"], payload["doc_type"]))
        stop.set()   # end the loop after the first event

    promo.run_listener(stop_event=stop, on_promoted=on_promoted)
    assert seen == [(7, "invoice")]


def test_promote_pending_catches_stranded_raw(monkeypatch):
    import src.services.extraction.promotion as promo

    class _Cur:
        def execute(self, sql, params=()):
            self._rows = [(101,)] if "bp_invoice_raw" in sql else []
        def fetchall(self):
            return list(getattr(self, "_rows", []))

    class _Conn:
        autocommit = True
        def cursor(self): return _Cur()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(promo, "get_conn", lambda: _Conn())
    seen = []
    monkeypatch.setattr(promo, "apply_hitl_fixes_and_promote",
                        lambda rid, dt: seen.append((rid, dt)) or {"ok": True})
    out = promo.promote_pending(doc_types=("invoice",))
    assert seen == [(101, "invoice")]
    assert out["promoted"] == 1 and out["failed"] == 0


def test_downstream_chain_throttles_kg_sync(monkeypatch):
    import src.services.linking_engine as le
    import src.services.deal_assignment_service as das
    import src.services.extraction.promotion as promo
    monkeypatch.setattr(promo, "promote_pending", lambda: {"promoted": 0, "failed": 0})
    monkeypatch.setattr(le, "promote_ready", lambda: {})
    monkeypatch.setattr(das, "assign_deals", lambda: {"forward_linked": 0})
    monkeypatch.setenv("KG_SYNC_THROTTLE_SECONDS", "9999")
    s = _bare()
    kg = []
    monkeypatch.setattr(s, "_run_kg_sync", lambda: kg.append(1))
    monkeypatch.setattr(s, "_chain_opportunity_mining", lambda r: None)
    s._run_downstream_chain()   # first run -> KG rebuild
    s._run_downstream_chain()   # within throttle window -> skipped
    assert kg == [1]
