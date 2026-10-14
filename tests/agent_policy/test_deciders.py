from types import SimpleNamespace

import copy

from services.agent_policy import deciders as D
from services.agent_policy import readiness as R
from tests.agent_policy.fixtures import FORM_EXAMPLE, REGISTRY, SETTINGS

MAP = {"Finance Manager": {"groups": ["PROCWISE_FINANCE"], "emails": ["cfo@x.com"]},
       "CFO": {"groups": [], "emails": []}}


def P(groups=(), email=None):
    return SimpleNamespace(subject="s", email=email, claims={"cognito:groups": list(groups)})


def test_group_match():
    assert D.eligible(P(["PROCWISE_FINANCE"]), "Finance Manager", MAP)


def test_email_match_case_insensitive():
    assert D.eligible(P(email="CFO@X.com"), "Finance Manager", MAP)


def test_admin_not_eligible():
    assert not D.eligible(P(["PROCWISE_ADMIN"], "admin@x.com"), "Finance Manager", MAP)


def test_no_match_and_unlinked_and_unknown():
    assert not D.eligible(P(["other"], "a@b.c"), "Finance Manager", MAP)
    assert not D.eligible(P(["PROCWISE_FINANCE"]), "CFO", MAP)
    assert not D.eligible(P(["PROCWISE_FINANCE"]), "Nobody", MAP)


def test_unmapped():
    assert D.unmapped(["Finance Manager", "CFO", "Nobody", "Nobody"], MAP) == ["CFO", "Nobody"]


def test_load_map_reads_rows():
    class Cur:
        def execute(self, *a): pass
        def fetchall(self): return [("A", ["g"], ["X@y.com"])]
        def close(self): pass
    conn = SimpleNamespace(cursor=lambda: Cur())
    assert D.load_map(conn) == {"A": {"groups": ["g"], "emails": ["x@y.com"]}}


def _form():
    f = copy.deepcopy(FORM_EXAMPLE)
    f["hidden"]["condition"]["all"][0]["value"] = ["refund.issue", "credit.issue"]
    f["checked"] = {"by": "u", "at": "2026-10-08T09:14:00Z"}
    return f


def _codes(probs):
    return [(p["field"], p["code"]) for p in probs]


def test_readiness_refuses_unmapped_decider():
    probs = R.activation_problems(_form(), REGISTRY, SETTINGS, deciders={"Finance Manager": MAP["Finance Manager"]})
    assert ("deciders", "decider_unmapped") in _codes(probs)
    p = [x for x in probs if x["code"] == "decider_unmapped"][0]
    assert p["names"] == ["CFO"] and p["routeTo"] == "administrator"


def test_readiness_accepts_mapped():
    full = {n: {"groups": ["g"], "emails": []} for n in ("Finance Manager", "CFO")}
    assert R.activation_problems(_form(), REGISTRY, SETTINGS, deciders=full) == []


def test_notify_names_checked_for_notify_outcome():
    f = _form(); f["outcome"] = "notify"; f["notify"] = ["Ghost"]; f["deciders"] = []
    probs = R.activation_problems(f, REGISTRY, SETTINGS, deciders={})
    assert ("notify", "decider_unmapped") in _codes(probs)


def test_map_unavailable_is_reported(monkeypatch):
    def boom(): raise RuntimeError("db down")
    monkeypatch.setattr(R, "_load_deciders", boom)
    assert ("deciders", "decider_map_unavailable") in _codes(R.activation_problems(_form(), REGISTRY, SETTINGS))


def test_string_groups_claim():
    p = SimpleNamespace(subject="s", email=None, claims={"cognito:groups": "PROCWISE_FINANCE"})
    assert D.eligible(p, "Finance Manager", MAP)


def test_none_principal_not_eligible():
    assert not D.eligible(None, "Finance Manager", MAP)
