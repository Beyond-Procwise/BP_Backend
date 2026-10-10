"""What a notification points at (stage 5b, Task B4): a suggestion's gains a `target` the screen
can open; every other kind is answered exactly as it was. No database."""
from datetime import datetime, timezone

import pytest

from api.auth import Principal
from services.agent_policy import approval_views as V

AT = datetime(2026, 10, 10, 9, tzinfo=timezone.utc)
COLS = ("notification_id", "recipient", "message", "link", "read_by", "created_at", "policy_key")


class _Cur:
    def __init__(self, rows):
        self.rows, self.description = rows, [(c,) for c in COLS]

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params=None): pass
    def fetchall(self): return self.rows


class _Conn:
    def __init__(self, rows): self.rows = rows
    def cursor(self): return _Cur(self.rows)


@pytest.mark.parametrize("link,ref", [
    ("suggestion:31", {"suggestionId": 31}), ("suggestion:0", {"suggestionId": 0}),
    ("suggestion:", {}), ("suggestion:x", {}), ("suggestion:-1", {}), ("suggestion:3 1", {}),
    ("suggestion:../x", {}), ("suggestion:٣", {}), ("suggestions:31", {}), ("Suggestion:31", {}),
    ("decision:41", {"decisionId": 41}), ("agent-policy:FIN-0012", {"policyKey": "FIN-0012"}),
    ("conflict:7", {"conflictId": 7}), ("https://x/y", {}), ("", {}), (None, {})])
def test_a_link_is_read_as_ids_never_as_a_route(link, ref):
    assert V._link_ref(link) == ref


def test_a_suggestion_carries_a_target_and_every_other_kind_is_byte_for_byte_what_it_was(monkeypatch):
    monkeypatch.setattr(V, "_recipients", lambda conn, principal, is_admin=False: ["Chief Financial Officer"])
    rows = [(5, "Chief Financial Officer", "Learning suggests a change to FIN-0012: Raise the limit?", "suggestion:31",
             [], AT, "FIN-0012"),
            (4, "Chief Financial Officer", "Refund needs approval", "decision:41", ["u1"], AT, "FIN-0012"),
            (3, "Chief Financial Officer", "Policy activated", "agent-policy:FIN-0003", [], AT, None),
            (2, "Chief Financial Officer", "Policies conflict", "conflict:9", [], AT, None),
            (1, "Chief Financial Officer", "A suggestion whose id cannot be read", "suggestion:x", [], AT, "FIN-0012")]
    got = V.my_notifications(_Conn(rows), Principal(subject="u1", email=None, claims={}), 50)
    when = AT.isoformat()
    assert got == [
        {"id": 5, "recipient": "Chief Financial Officer",
         "message": "Learning suggests a change to FIN-0012: Raise the limit?", "policyKey": "FIN-0012",
         "decisionId": None, "conflictId": None, "read": False, "createdAt": when,
         "target": {"kind": "suggestion", "id": 31, "policyKey": "FIN-0012"}},
        {"id": 4, "recipient": "Chief Financial Officer", "message": "Refund needs approval", "policyKey": "FIN-0012",
         "decisionId": 41, "conflictId": None, "read": True, "createdAt": when},
        {"id": 3, "recipient": "Chief Financial Officer", "message": "Policy activated", "policyKey": "FIN-0003",
         "decisionId": None, "conflictId": None, "read": False, "createdAt": when},
        {"id": 2, "recipient": "Chief Financial Officer", "message": "Policies conflict", "policyKey": None,
         "decisionId": None, "conflictId": 9, "read": False, "createdAt": when},
        {"id": 1, "recipient": "Chief Financial Officer", "message": "A suggestion whose id cannot be read",
         "policyKey": "FIN-0012", "decisionId": None, "conflictId": None, "read": False, "createdAt": when}]
    assert [list(n) for n in got[1:4]] == [["id", "recipient", "message", "policyKey", "decisionId", "conflictId",
                                            "read", "createdAt"]] * 3
