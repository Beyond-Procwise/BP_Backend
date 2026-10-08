"""Every inbound reply is screened as it is stored, and a fault in the screen never costs a reply."""

import importlib
from datetime import datetime, timezone
from decimal import Decimal

import pytest

repo = importlib.import_module("repositories.supplier_response_repo")


def a_row(**over):
    base = dict(workflow_id="wf-1", unique_id="PROC-WF-1", supplier_id="S-1", response_text="Our bank details have changed.",
                received_time=datetime(2026, 10, 8, tzinfo=timezone.utc), response_message_id="<m1>", response_subject="Re: PO")
    base.update(over)
    return repo.SupplierResponseRow(**base)


class Reached(Exception):
    """Raised by the fake connection: proves the insert was attempted."""


@pytest.fixture
def seen(monkeypatch):
    calls = []
    inbound = importlib.import_module("services.draft_assurance.inbound")
    monkeypatch.setattr(inbound, "screen_and_record", lambda row, *a, **k: calls.append(row) or 1)

    def get_conn():
        raise Reached
    monkeypatch.setattr(repo, "get_conn", get_conn)
    return calls


def test_a_reply_is_screened_before_the_insert_is_attempted(seen):
    row = a_row()
    with pytest.raises(Reached):
        repo.insert_response(row)
    assert seen == [row]


def test_a_fault_in_the_screen_does_not_stop_the_reply_being_stored(monkeypatch):
    inbound = importlib.import_module("services.draft_assurance.inbound")
    monkeypatch.setattr(inbound, "screen_and_record", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("screen exploded")))
    monkeypatch.setattr(repo, "get_conn", lambda: (_ for _ in ()).throw(Reached()))
    with pytest.raises(Reached):                       # the insert still ran; only the screen's own error was swallowed
        repo.insert_response(a_row())


def test_the_screen_is_called_once_per_insert_and_does_not_change_the_row(seen):
    row = a_row()
    before = dict(vars(row))
    with pytest.raises(Reached):
        repo.insert_response(row)
    assert len(seen) == 1 and dict(vars(row)) == before


def test_an_import_problem_with_the_screen_is_not_fatal(monkeypatch):
    monkeypatch.setattr(repo, "_screen_inbound", repo._screen_inbound)         # the real helper
    import builtins
    real = builtins.__import__

    def refuse(name, *a, **k):
        if name.endswith("draft_assurance") or name.endswith("draft_assurance.inbound") or name == "inbound":
            raise ImportError("assurance layer unavailable")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", refuse)
    repo._screen_inbound(a_row())                                              # must simply return


# --- the sender check rides the same hook, independently ------------------------------------------------------------------------

@pytest.fixture
def both(monkeypatch):
    inbound = importlib.import_module("services.draft_assurance.inbound")
    sender = importlib.import_module("services.draft_assurance.sender_auth")
    calls = []
    monkeypatch.setattr(inbound, "screen_and_record", lambda row, *a, **k: calls.append(("screen", row)))
    monkeypatch.setattr(sender, "check_and_record", lambda row, *a, **k: calls.append(("sender", row)))
    return calls, inbound, sender


def test_every_reply_is_given_to_the_payment_screen_and_the_sender_check(both):
    calls, *_ = both
    row = a_row()
    repo._screen_inbound(row)
    assert calls == [("screen", row), ("sender", row)]


def test_a_fault_in_the_payment_screen_does_not_stop_the_sender_check(both, monkeypatch):
    calls, inbound, sender = both
    monkeypatch.setattr(inbound, "screen_and_record", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("screen exploded")))
    repo._screen_inbound(a_row())
    assert [c[0] for c in calls] == ["sender"]


def test_a_fault_in_the_sender_check_does_not_stop_the_payment_screen_or_the_insert(both, monkeypatch):
    calls, inbound, sender = both
    monkeypatch.setattr(sender, "check_and_record", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("check exploded")))
    monkeypatch.setattr(repo, "get_conn", lambda: (_ for _ in ()).throw(Reached()))
    with pytest.raises(Reached):                      # the insert still ran
        repo.insert_response(a_row())
    assert [c[0] for c in calls] == ["screen"]


def test_the_injection_screen_runs_even_when_the_other_two_fault(monkeypatch):
    inbound = importlib.import_module("services.draft_assurance.inbound")
    sender_auth = importlib.import_module("services.draft_assurance.sender_auth")
    seen_rows = []
    boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setattr(inbound, "screen_and_record", boom)
    monkeypatch.setattr(sender_auth, "check_and_record", boom)
    monkeypatch.setattr(inbound, "screen_injection_and_record", lambda row, *a, **k: seen_rows.append(row) or 1)
    monkeypatch.setattr(repo, "get_conn", lambda: (_ for _ in ()).throw(Reached()))
    row = a_row()
    with pytest.raises(Reached):
        repo.insert_response(row)
    assert seen_rows == [row]


def test_a_fault_in_the_injection_screen_does_not_stop_the_reply_being_stored(monkeypatch):
    inbound = importlib.import_module("services.draft_assurance.inbound")
    monkeypatch.setattr(inbound, "screen_injection_and_record", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(repo, "get_conn", lambda: (_ for _ in ()).throw(Reached()))
    with pytest.raises(Reached):
        repo.insert_response(a_row())
