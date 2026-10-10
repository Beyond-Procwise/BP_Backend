"""The learning scan's reading of its policies, with a scripted cursor. No database."""
import copy
from datetime import datetime, timezone

from services.agent_policy import learning as L
from services.agent_policy import settings as S
from tests.agent_policy.fixtures import FORM_EXAMPLE

NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)


class _Cur:
    def __init__(self, conn):
        self.conn, self.rows, self.rowcount = conn, [], 0

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def execute(self, sql, params=None):
        self.conn.ran.append((sql, params))
        if sql == L._TARGETS_SQL:
            self.rows = list(self.conn.policies)
        elif sql == L._SAME_CONDITION_SQL:
            self.rows = [(1,), (2,)]
        else:
            self.rows = []

    def fetchall(self): return self.rows
    def fetchone(self): return self.rows[0] if self.rows else None


class _Conn:
    autocommit = True

    def __init__(self, policies):
        self.policies, self.ran = policies, []

    def cursor(self): return _Cur(self)
    def commit(self): pass
    def rollback(self): pass


def _policies():
    unreadable = copy.deepcopy(FORM_EXAMPLE)
    unreadable["hidden"]["condition"] = {"field": "args.amount", "op": "gt", "value": {1, 2}}   # will not serialise
    # the last column is the latest version's condition: FIN-0001 has a draft at 800 beside its live 500
    return [("FIN-0001", 2, copy.deepcopy(FORM_EXAMPLE), {}, _at(800)), ("FIN-0002", 1, unreadable, {}, None),
            ("FIN-0003", 3, copy.deepcopy(FORM_EXAMPLE), {}, None)]


def _at(value):
    condition = copy.deepcopy(FORM_EXAMPLE["hidden"]["condition"])
    condition["all"][1]["value"] = value
    return condition


def test_a_target_carries_its_latest_versions_condition_or_falls_back_to_the_live_one():
    first, third = L.targets(_Conn(_policies()).cursor())
    assert L._latest_condition(first) == _at(800) and first.form["hidden"]["condition"] == _at(500)
    assert third.latest_condition is None and L._latest_condition(third) == _at(500)
    injected = L.Target(key="TST-1a", live_version=1, form=copy.deepcopy(FORM_EXAMPLE), versions=[1])
    assert L._latest_condition(injected) == _at(500), "an injected target keeps working"


def test_one_unreadable_policy_is_skipped_and_the_others_are_returned(caplog):
    conn = _Conn(_policies())
    found = L.targets(conn.cursor())
    assert [(t.key, t.live_version, t.versions) for t in found] == [("FIN-0001", 2, [1, 2]), ("FIN-0003", 3, [1, 2])]
    assert [r.getMessage() for r in caplog.records if "FIN-0002" in r.getMessage()], "logged by key"


def test_the_scan_counts_the_unreadable_policy_and_does_not_close_its_cards(monkeypatch):
    monkeypatch.setattr(S, "load_settings", lambda conn=None: S.merge(None))
    conn = _Conn(_policies())
    counts = L.scan(conn, NOW)
    assert (counts["policies"], counts["errors"]) == (2, 1)
    [(_sql, params)] = [r for r in conn.ran if r[0] == L._GONE_SQL]
    assert params == (NOW, ["FIN-0001", "FIN-0003", "FIN-0002"]), \
        "a policy that could not be read is still a policy: its cards are not superseded"
    assert [p[0] for s, p in conn.ran if s == L._LOCK_SQL] == ["FIN-0001", "FIN-0003"]
