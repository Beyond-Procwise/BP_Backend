"""Standing rules overlaid onto live policies' conflicts[] at read time (pure: a fake cursor).

Version rows are immutable and live_policies caches what it loads, so the overlay must never
change the documents it is given: it returns new ones.
"""
import copy
from datetime import datetime, timezone

from services.agent_policy import conflict_cases as CC
from services.agent_policy import contract
from services.agent_policy.compiler import compile_policy
from tests.agent_policy.fixtures import FORM_EXAMPLE, REGISTRY, SETTINGS

DECIDED = datetime(2026, 10, 9, 10, 0, tzinfo=timezone.utc)


class _Cur:
    """Answers rules_for's one query with in-force rule rows."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchall(self):
        return list(self.rows)


def _doc(key):
    form = copy.deepcopy(FORM_EXAMPLE)
    form["checked"] = {"by": "u1", "at": "2026-10-08T00:00:00Z"}
    return compile_policy(form, policy_key=key, version=2, status="live", settings=SETTINGS, never_suggest=False)


def _rule_row(prevails="FIN-0012", yields="CUS-0004", decision_id=41):
    pair = "|".join(sorted([prevails, yields]))
    # pair_key, prevails, yields, rule_text, decision_id, decided_by, decided_at
    return (pair, prevails, yields, f"{prevails} takes priority over {yields}", decision_id, "cfo@x", DECIDED)


def test_overlay_does_not_mutate_cached_docs():
    a, b, c = _doc("FIN-0012"), _doc("CUS-0004"), _doc("GEN-0001")
    cached = [a, b, c]
    before = copy.deepcopy(cached)
    out = CC.overlay(_Cur([_rule_row()]), cached)
    assert cached == before                                   # the cached copies are untouched
    assert all(o is not d for o, d in zip(out, cached))
    entry = {"with": "CUS-0004", "rule": "FIN-0012 takes priority over CUS-0004", "caseId": "pc_41",
             "decidedAt": DECIDED.isoformat(), "prevails": "FIN-0012"}
    assert out[0]["conflicts"] == [entry]
    assert out[1]["conflicts"] == [{**entry, "with": "FIN-0012"}]   # the yielding side carries it too
    assert out[2]["conflicts"] == []
    for d in out:
        assert contract.validate(d, REGISTRY) == []                 # schema-compatible entries


def test_overlay_asks_once_for_every_doc_key_and_skips_non_documents():
    cur = _Cur([])
    out = CC.overlay(cur, [_doc("FIN-0012"), None, _doc("CUS-0004")])
    assert len(cur.calls) == 1
    assert sorted(cur.calls[0][1][0]) == ["CUS-0004", "FIN-0012"]
    assert out[1] is None and out[0]["conflicts"] == [] and out[2]["conflicts"] == []


def test_overlay_of_nothing_asks_nothing():
    cur = _Cur([])
    assert CC.overlay(cur, []) == [] and cur.calls == []


def test_option_labels():
    assert CC.option_label("keep_both:FIN-0012") == "Keep both: FIN-0012 takes priority"
    assert CC.option_label("change:FIN-0012") == "Change FIN-0012"
    assert CC.option_label("limit:FIN-0012") == "Limit FIN-0012"
    assert CC.option_label("retire:FIN-0012") == "Retire FIN-0012"
