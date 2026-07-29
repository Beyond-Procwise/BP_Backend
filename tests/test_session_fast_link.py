"""The upload -> report race.

An upload reached _stg within a second of extraction finishing, but nothing
stamped it onto its deal or promoted it into _trgt (the only tier the product
reads) until a 15-minute scheduled sweep ran. The report opened immediately
after the redirect and showed an empty deal for up to 11 minutes.

The linking passes an upload actually needs cost ~1s. The look-back backfill
costs ~245s and never touches a document look-forward has already claimed, so it
stays on the schedule. These tests pin both halves: the fast path skips
look-back, and the session's terminal WebSocket frame is sent only after linking
has run — so "ready" means the data is really there.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.services import deal_assignment_service as das


class _Cur:
    """Stand-in cursor — the passes are stubbed, so it is never used."""


def _stub_passes(monkeypatch, calls):
    for name in ("_look_forward", "_look_back", "_reconcile_legacy",
                 "_propagate_deal_along_po", "_flag_conflict_po_chains",
                 "_backfill_deal_metadata", "_propagate_deal_date",
                 "_mirror_deal_to_raw_and_stg", "_prune_deal_document_map",
                 "reconcile_status"):
        monkeypatch.setattr(
            das, name,
            (lambda n: lambda cur: (calls.append(n), 0)[1])(name),
        )


def test_fast_run_skips_look_back(monkeypatch):
    """look_back is 245s of corpus-wide backfill — not what a new upload needs."""
    calls: list[str] = []
    _stub_passes(monkeypatch, calls)

    result = das._run(_Cur(), include_look_back=False)

    assert "_look_forward" in calls, "the upload still has to be stamped onto its deal"
    assert "_look_back" not in calls
    # Every other pass is sub-second and still runs.
    assert "_propagate_deal_along_po" in calls
    assert "reconcile_status" in calls
    # The key is reported honestly rather than claiming zero back-links happened.
    assert result["backward_linked"] is None


def test_full_run_still_includes_look_back(monkeypatch):
    """The scheduled sweep is unchanged — this is an added path, not a swap."""
    calls: list[str] = []
    _stub_passes(monkeypatch, calls)

    result = das._run(_Cur())

    assert "_look_back" in calls
    assert result["backward_linked"] == 0


def test_assign_deals_fast_does_not_generate_summaries(monkeypatch):
    """Summaries are an LLM call; holding the client's 'ready' signal behind one
    would reintroduce the very wait this removes."""
    monkeypatch.setattr(das, "_run", lambda cur, include_look_back=True: {"forward_linked": 1})

    class _Conn:
        def cursor(self):
            return _Cur()

    called = []
    import src.services.deal_analysis_service as analysis
    monkeypatch.setattr(analysis, "sync_deal_summaries",
                        lambda *a, **k: called.append("summaries") or {})

    result = das.assign_deals_fast(conn=_Conn())

    assert result["forward_linked"] == 1
    assert called == [], "the fast path must not wait on summary generation"


# ---------------------------------------------------------------------------
# The session listener: link first, then tell the client it is ready.
# ---------------------------------------------------------------------------

def _listener():
    from src.services.session_notify_listener import SessionNotifyListener
    return SessionNotifyListener(event_loop=None)


def test_terminal_frame_is_sent_only_after_linking(monkeypatch):
    order: list[str] = []

    listener = _listener()
    monkeypatch.setattr(listener, "_link_session",
                        lambda sid: order.append(f"link:{sid}"))
    monkeypatch.setattr(listener, "_broadcast",
                        lambda sid, payload: order.append(f"broadcast:{sid}"))

    listener._link_then_broadcast("SESSION-1", {"session_id": "SESSION-1"})

    assert order == ["link:SESSION-1", "broadcast:SESSION-1"]


def test_client_is_still_told_when_linking_fails(monkeypatch):
    """Fail open: a linking error must never strand the page on a spinner."""
    order: list[str] = []

    listener = _listener()

    def _boom(_sid):
        order.append("link")
        raise RuntimeError("deadlock")

    monkeypatch.setattr(listener, "_link_session", _boom)
    monkeypatch.setattr(listener, "_broadcast",
                        lambda sid, payload: order.append("broadcast"))

    listener._link_then_broadcast("SESSION-1", {"session_id": "SESSION-1"})

    assert order == ["link", "broadcast"]


def test_fast_link_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SESSION_FAST_LINK_ENABLED", "0")
    called = []
    monkeypatch.setattr(das, "assign_deals_fast", lambda *a, **k: called.append("ran"))

    _listener()._link_session("SESSION-1")

    assert called == []


# ---------------------------------------------------------------------------
# The trigger itself: promote when the session finishes EXTRACTING.
#
# _trgt inserts fire the outcome triggers that resolve a session, which is what
# ultimately tells the report page to render. So the session cannot resolve until
# promotion has run — and promotion only ever ran on a 15-minute sweep. Measured on
# a live upload: extraction finished at 12:27:47, promotion (and with it the session
# resolving) did not happen until 12:29:55.
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, pending):
        self._pending = pending

    def execute(self, *a, **k):
        pass

    def fetchone(self):
        return (self._pending,)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def __init__(self, pending):
        self._pending = pending
        self.closed = False

    def cursor(self):
        return _FakeCursor(self._pending)

    def close(self):
        self.closed = True


def _watcher(monkeypatch, pending):
    import threading
    from src.services.process_monitor_watcher import ProcessMonitorWatcher
    w = ProcessMonitorWatcher.__new__(ProcessMonitorWatcher)   # no DB in __init__
    w._finalise_lock = threading.Lock()
    w._promoting_sessions = set()
    w._promote_again = set()
    monkeypatch.setattr(w, "_get_connection", lambda *a, **k: _FakeConn(pending))
    return w


def test_no_promotion_while_documents_are_still_extracting(monkeypatch):
    ran = []
    w = _watcher(monkeypatch, pending=2)
    monkeypatch.setattr(w, "_promote_session", lambda sid: ran.append(sid))

    w._finalise_session_if_complete("SESSION-1")

    assert ran == []


def test_concurrent_workers_coalesce_into_one_promotion(monkeypatch):
    ran = []
    w = _watcher(monkeypatch, pending=0)
    monkeypatch.setattr(w, "_promote_session", lambda sid: ran.append(sid))

    # Every worker calls this as it finishes; one promotes, the rest latch.
    w._finalise_session_if_complete("SESSION-1")
    w._finalise_session_if_complete("SESSION-1")
    w._finalise_session_if_complete("SESSION-1")

    assert ran == ["SESSION-1"]
    # The latched calls are not lost — the in-flight pass runs again for them.
    assert w._finish_promotion("SESSION-1") is True


def test_a_later_type_group_still_gets_promoted(monkeypatch):
    """The regression that made this necessary.

    The Analyse screen uploads quotes, POs and invoices as separate groups under
    one session. The quotes can finish extracting before the invoice group's rows
    exist, so the first pass legitimately sees "nothing pending". Marking the
    session done at that point stranded the later groups in _stg — observed live:
    all four documents extracted, two reached _trgt, two did not.
    """
    ran = []
    w = _watcher(monkeypatch, pending=0)
    monkeypatch.setattr(w, "_promote_session", lambda sid: ran.append(sid))

    w._finalise_session_if_complete("SESSION-1")        # quotes finish
    assert w._finish_promotion("SESSION-1") is False    # that pass completes
    w._finalise_session_if_complete("SESSION-1")        # invoice finishes later

    assert ran == ["SESSION-1", "SESSION-1"]


def test_promotion_is_skipped_without_a_session(monkeypatch):
    ran = []
    w = _watcher(monkeypatch, pending=0)
    monkeypatch.setattr(w, "_promote_session", lambda sid: ran.append(sid))

    w._finalise_session_if_complete(None)
    w._finalise_session_if_complete("")

    assert ran == []


def test_fast_promotion_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SESSION_FAST_PROMOTE_ENABLED", "0")
    ran = []
    w = _watcher(monkeypatch, pending=0)
    monkeypatch.setattr(w, "_promote_session", lambda sid: ran.append(sid))

    w._finalise_session_if_complete("SESSION-1")

    assert ran == []
