import pytest

from services.people.events import DomainEvent
from services.people.wiring import emit_safe


class Cur:
    """Records SQL; fails the INSERT when told to; can refuse SAVEPOINT like an autocommit connection."""
    def __init__(self, fail_insert=False, no_savepoint=False):
        self.sql, self.fail_insert, self.no_savepoint, self.rowcount = [], fail_insert, no_savepoint, 1
    def execute(self, sql, args=None):
        self.sql.append(sql.split("(")[0].strip()[:40])
        if sql.startswith("SAVEPOINT") and self.no_savepoint:
            raise RuntimeError("SAVEPOINT can only be used in transaction blocks")
        if sql.startswith("INSERT") and self.fail_insert:
            raise RuntimeError('relation "proc.bp_domain_event" does not exist')


def ev():
    return DomainEvent("approval.requested", "u1", "approval", "7", "s", event_id="approval.requested:7")


def test_success_releases_savepoint():
    c = Cur()
    assert emit_safe(c, ev()) is True
    assert c.sql[0] == "SAVEPOINT domain_event" and c.sql[-1] == "RELEASE SAVEPOINT domain_event"


def test_failure_is_swallowed_and_rolled_back_to_savepoint():
    c = Cur(fail_insert=True)
    assert emit_safe(c, ev()) is False          # does not raise: the approval must still go through
    assert c.sql[-1] == "ROLLBACK TO SAVEPOINT domain_event"


def test_autocommit_connection_still_swallows():
    assert emit_safe(Cur(no_savepoint=True, fail_insert=True), ev()) is False
    assert emit_safe(Cur(no_savepoint=True), ev()) is True


def test_switch_off(monkeypatch):
    monkeypatch.setenv("DOMAIN_EVENTS", "off")
    c = Cur()
    assert emit_safe(c, ev()) is False and c.sql == []


def test_open_case_announces_in_the_same_transaction_and_never_blocks(monkeypatch):
    from contextlib import contextmanager
    from datetime import datetime, timezone
    from services.agent_policy import approvals

    seen = []
    @contextmanager
    def tx(conn):
        seen.append("begin"); yield conn; seen.append("commit")
    monkeypatch.setattr(approvals, "_tx", tx)
    monkeypatch.setattr(approvals, "_insert_case", lambda cur, **kw: 42)
    import services.people.wiring as w
    monkeypatch.setattr(w, "emit_safe", lambda cur, e: seen.append(e) or True)

    class Conn:
        def cursor(self):
            class C:
                def __enter__(s): return s
                def __exit__(s, *a): return False
            return C()
    doc = {"id": "P-1", "context": {"actions": {"plain": "issuing a refund"}}}
    assert approvals.open_case(Conn(), policy_doc=doc, firing_id=1, action={"tool": "t", "agent": "a"},
                               requested_by="u1", now=datetime.now(timezone.utc), mapping={}) == 42
    ev = seen[1]
    assert seen[0] == "begin" and seen[-1] == "commit"            # emitted inside the transaction
    assert (ev.type, ev.event_id, ev.actor_id) == ("approval.requested", "approval.requested:42", "u1")
    assert ev.data["requester"] == "u1" and "refund" in ev.summary.lower()
