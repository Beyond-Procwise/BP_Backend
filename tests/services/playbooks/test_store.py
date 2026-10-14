"""The store loads active playbooks and nothing else.

Rows are injected; no connection is opened. An unreadable store raises, an
empty one does not -- see the docstring on PlaybookStore for why those differ.
"""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from src.services.playbooks.store import (  # noqa: E402
    Playbook,
    PlaybookStore,
    PlaybookStoreUnavailable,
    load_playbook_store,
)


def row(**over):
    base = {
        "playbook_id": 1,
        "playbook_name": "Recover the duplicate",
        "trigger_source": "detection_finding",
        "trigger_match": {"rule_id": "duplicate"},
        "agent_workflow_id": 958,
        "params": {},
        "playbook_status": "active",
        "version": 1,
    }
    base.update(over)
    return base


def test_loads_an_active_playbook():
    store = PlaybookStore(playbook_rows=[row()])
    loaded = store.active_playbooks()
    assert len(loaded) == 1
    assert loaded[0] == Playbook(
        playbook_id=1,
        playbook_name="Recover the duplicate",
        trigger_source="detection_finding",
        trigger_match={"rule_id": "duplicate"},
        agent_workflow_id=958,
        params={},
        version=1,
    )


@pytest.mark.parametrize("status", ["draft", "pending_approval", "retired"])
def test_only_active_playbooks_load(status):
    """A retired or unapproved playbook must not be able to propose anything."""
    store = PlaybookStore(playbook_rows=[row(playbook_status=status)])
    assert store.active_playbooks() == []


def test_empty_store_does_not_raise():
    """Deliberately unlike RuleBook. An empty rule book hides work that should
    have happened; an empty playbook table simply means nobody has authored a
    strategy yet, and failing closed would make the service unbootable until
    somebody did."""
    store = PlaybookStore(playbook_rows=[])
    assert store.active_playbooks() == []


def test_unreadable_store_raises():
    def explode():
        raise RuntimeError("connection refused")

    with pytest.raises(PlaybookStoreUnavailable) as exc:
        PlaybookStore(connection_factory=explode)
    assert "connection refused" in str(exc.value)


def test_for_source_never_crosses_the_two_stores():
    store = PlaybookStore(
        playbook_rows=[
            row(playbook_id=1, trigger_source="detection_finding",
                trigger_match={"rule_id": "duplicate"}),
            row(playbook_id=2, trigger_source="opportunity",
                trigger_match={"detector_type": "Invoice Overbilling"}),
        ]
    )
    assert [p.playbook_id for p in store.for_source("detection_finding")] == [1]
    assert [p.playbook_id for p in store.for_source("opportunity")] == [2]
    assert store.for_source("invoices") == []


def test_a_row_with_an_unknown_source_is_skipped_not_crashed(caplog):
    """The CHECK constraint makes this unreachable through the endpoint, but a
    hand-edited row must not take the sweep down."""
    store = PlaybookStore(playbook_rows=[row(trigger_source="invoices")])
    assert store.active_playbooks() == []


def test_a_row_with_an_unknown_match_key_is_skipped():
    """It could only get there by hand: the endpoint validates. Loading it
    would give a playbook that never fires and no sign of why."""
    store = PlaybookStore(playbook_rows=[row(trigger_match={"sevrity": "high"})])
    assert store.active_playbooks() == []


def test_jsonb_arriving_as_text_is_parsed():
    """psycopg2 gives a dict; other drivers and some fixtures give a string."""
    store = PlaybookStore(playbook_rows=[row(trigger_match='{"rule_id": "duplicate"}')])
    assert store.active_playbooks()[0].trigger_match == {"rule_id": "duplicate"}


def test_reload_picks_up_a_change():
    """reload() re-reads through the factory rather than keeping a stale cache.

    It used to assert that a store with no factory empties itself on reload,
    which documented the silent empty read as the contract. That read is the
    bug that made the first live sweep report every finding unmatched, so the
    test now exercises a factory whose rows change.
    """

    class Cur:
        description = [
            ("playbook_id",), ("playbook_name",), ("trigger_source",),
            ("trigger_match",), ("agent_workflow_id",), ("params",),
            ("playbook_status",), ("version",), ("workflow_is_active",),
        ]

        def __init__(self, rows):
            self._rows = rows

        def execute(self, sql, params=None):
            pass

        def fetchall(self):
            return self._rows

        def close(self):
            pass

    live = [(1, "first", "detection_finding", {}, 958, {}, "active", 1, True)]

    class Conn:
        def cursor(self):
            return Cur(live)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    store = PlaybookStore(connection_factory=lambda: Conn())
    assert [p.playbook_name for p in store.active_playbooks()] == ["first"]
    live[:] = [(2, "second", "detection_finding", {}, 958, {}, "active", 2, True)]
    store.reload()
    assert [p.playbook_name for p in store.active_playbooks()] == ["second"]


def test_a_playbook_whose_workflow_is_gone_is_skipped():
    """The approve endpoint refuses an inactive workflow, but a workflow can be
    deleted AFTER its playbook was approved. Selecting it would queue a
    proposal that can only fail at the moment somebody accepts it."""
    assert PlaybookStore(
        playbook_rows=[row(workflow_is_active=False)]
    ).active_playbooks() == []
    assert len(PlaybookStore(
        playbook_rows=[row(workflow_is_active=True)]
    ).active_playbooks()) == 1


def test_loader_carries_failure_rather_than_raising():
    """Follows load_rule_book: the blast radius of a playbook outage stops at
    playbooks, so the API still boots."""
    def explode():
        raise RuntimeError("connection refused")

    nick = type("N", (), {"get_db_connection": staticmethod(explode)})()
    assert load_playbook_store(agent_nick=nick) is None
    # And a healthy one still loads.
    assert load_playbook_store(playbook_rows=[row()]) is not None


def test_the_production_loader_actually_reads_the_table(monkeypatch):
    """load_playbook_store() takes no agent_nick from the sweep, so it has to
    resolve its own connection. Without that it builds a store with no
    connection factory, reads nothing, and every sweep reports every finding
    unmatched -- which is indistinguishable from nobody having authored a
    strategy yet."""
    import src.services.playbooks.store as mod

    class Cur:
        description = [
            ("playbook_id",), ("playbook_name",), ("trigger_source",),
            ("trigger_match",), ("agent_workflow_id",), ("params",),
            ("playbook_status",), ("version",), ("workflow_is_active",),
        ]

        def execute(self, sql, params=None):
            pass

        def fetchall(self):
            return [(1, "Recover the duplicate", "detection_finding",
                     {"rule_id": "duplicate"}, 958, {}, "active", 1, True)]

        def close(self):
            pass

    class Conn:
        def cursor(self):
            return Cur()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mod, "get_conn", lambda: Conn())
    store = load_playbook_store()
    assert store is not None
    assert [p.playbook_id for p in store.active_playbooks()] == [1]


def test_a_row_with_a_null_match_value_is_skipped():
    """The endpoint refuses a null match value, but a hand-written row can
    still carry one. Loading it gives a playbook that matches nothing and says
    nothing -- the unknown-key sibling of this is already logged and skipped."""
    store = PlaybookStore(playbook_rows=[row(trigger_match={"rule_id": "duplicate",
                                                            "doc_type": None})])
    assert store.active_playbooks() == []


def test_a_row_with_a_non_scalar_match_value_is_skipped():
    store = PlaybookStore(playbook_rows=[row(trigger_match={"severity": ["critical"]})])
    assert store.active_playbooks() == []


def test_a_store_with_no_way_to_read_raises_rather_than_reporting_empty():
    """The silent empty read is what made the first live sweep report every
    finding unmatched with an active playbook in the table. An agent_nick that
    cannot supply a connection is an outage, not an empty table."""
    nick = type("N", (), {})()          # no get_db_connection
    with pytest.raises(PlaybookStoreUnavailable):
        PlaybookStore(agent_nick=nick)
