from datetime import datetime, timedelta, timezone

import pytest

from services.people.graph import derive as d
from services.people.graph import queries as q
from services.people.graph import sync
from services.people.graph.model import AdmissionError, NodeRef
from services.people.graph.store import InMemoryGraph
from services.people.graph.writer import GraphWriter
from tests.services.people.graph.conftest import Clock, T0, TENANT

NOW = T0 + timedelta(days=200)
U = lambda k: NodeRef("User", k)


def people(**over):
    base = [
        dict(subject="cora", display_name="Cora", title="CPO", department_id="proc", manager_subject=None, approval_limit=-1,
             is_active=True, delegate_subject=None, delegate_until=None),
        dict(subject="carl", display_name="Carl", title="Category Manager", department_id="proc", manager_subject="cora",
             approval_limit=50000, is_active=True, delegate_subject=None, delegate_until=None),
        dict(subject="jane", display_name="Jane", title="Buyer", department_id="proc", manager_subject="carl",
             approval_limit=None, is_active=True, delegate_subject=None, delegate_until=None),
        dict(subject="fay", display_name="Fay", title="Finance Approver", department_id="fin", manager_subject=None,
             approval_limit=1_000_000, is_active=True, delegate_subject=None, delegate_until=None),
    ]
    for p in base:
        p.update(over.get(p["subject"], {}))
    return base


SNAP = sync.RbacSnapshot(
    assignments={"jane": ["Buyer"], "carl": ["Buyer", "Approver"], "cora": ["Admin"], "fay": ["Approver"]},
    roles={"Buyer": ["read", "write"], "Approver": ["read", "approve"], "Admin": ["read", "write", "approve", "configure"]},
    actions={"supplier.read": "read", "supplier.write": "write", "po.approve": "approve", "policy.write": "configure"})


@pytest.fixture()
def g():
    clock = Clock(NOW)
    store = InMemoryGraph()
    w = GraphWriter(store, TENANT, now=clock)
    sync.sync_hr(w, people(), now=NOW)
    sync.sync_rbac(w, SNAP)
    return store, w, clock


# ------------------------------------------------------------------ sync
def test_hr_sync_builds_org_and_is_idempotent(g):
    store, w, clock = g
    assert [e.dst.key for e in store.edges(TENANT, src=U("jane"), type="REPORTS_TO")] == ["carl"]
    assert [e.dst.key for e in store.edges(TENANT, src=U("jane"), type="HOLDS_TITLE")] == ["Buyer"]
    assert sync.sync_hr(w, people(), now=NOW)["changed"] == 0
    assert sync.sync_rbac(w, SNAP)["changed"] == 0


def test_hr_changes_close_and_open_and_inactive_people_lose_org_edges(g):
    store, w, clock = g
    before = NOW + timedelta(minutes=1)
    clock.t = NOW + timedelta(days=5)
    sync.sync_hr(w, people(jane={"manager_subject": "cora"}, carl={"is_active": False}), now=clock.t)
    assert [e.dst.key for e in store.edges(TENANT, src=U("jane"), type="REPORTS_TO")] == ["cora"]
    assert [e.dst.key for e in store.edges(TENANT, src=U("jane"), type="REPORTS_TO", as_of=before)] == ["carl"]
    assert store.edges(TENANT, src=U("carl"), type="REPORTS_TO") == []
    assert store.edges(TENANT, src=U("carl"), type="HAS_AUTHORITY") == []


def test_delegation_only_while_valid(g):
    store, w, clock = g
    sync.sync_hr(w, people(carl={"delegate_subject": "fay", "delegate_until": NOW + timedelta(days=3)}), now=NOW)
    assert [e.dst.key for e in store.edges(TENANT, type="DELEGATES_TO", as_of=NOW)] == ["fay"]
    assert store.edges(TENANT, type="DELEGATES_TO", as_of=NOW + timedelta(days=4)) == []        # expires by its own window
    sync.sync_hr(w, people(carl={"delegate_subject": "fay", "delegate_until": NOW - timedelta(days=1)}), now=NOW)
    assert store.edges(TENANT, type="DELEGATES_TO", as_of=NOW) == []                            # already over: not asserted


def test_rbac_revocation_closes_has_role(g):
    store, w, clock = g
    clock.t = NOW + timedelta(days=1)
    snap = sync.RbacSnapshot({**SNAP.assignments, "jane": []}, SNAP.roles, SNAP.actions)
    sync.sync_rbac(w, snap)
    assert store.edges(TENANT, src=U("jane"), type="HAS_ROLE") == []
    assert [e.dst.key for e in store.edges(TENANT, src=U("jane"), type="HAS_ROLE", as_of=NOW + timedelta(hours=1))] == ["Buyer"]


# ------------------------------------------------------------------ decisions
def facts():
    f = lambda n, who, verb, days, **kw: sync.DecisionFact(verb, who, f"decision:{n}", f"approval.{verb}d:{n}", NOW - timedelta(days=days),
                                                           f"test:{n}", **kw)
    return [f(1, "carl", "approve", 10, duration_ms=3_600_000, supplier_key="supplier:S1"),
            f(2, "carl", "approve", 8, duration_ms=2 * 3_600_000, supplier_key="supplier:S1"),
            f(3, "carl", "reject", 6, duration_ms=5 * 3_600_000, supplier_key="supplier:S1"),
            f(4, "fay", "approve", 4, duration_ms=50 * 3_600_000),
            f(5, "fay", "approve", 3, duration_ms=60 * 3_600_000),
            f(6, "fay", "approve", 2, duration_ms=70 * 3_600_000)]


def test_decisions_are_one_edge_each_and_idempotent(g):
    store, w, _ = g
    assert sync.apply_decisions(w, facts()) == {"written": 6, "skipped": 0}
    assert sync.apply_decisions(w, facts()) == {"written": 0, "skipped": 6}
    assert len(store.edges(TENANT, type="APPROVED")) == 5 and len(store.edges(TENANT, type="REJECTED")) == 1


# ------------------------------------------------------------------ derivation
CFG = d.DeriveConfig()


def inter(a, b, n, entities, days_ago=5):
    return [d.Interaction(a, b, NOW - timedelta(days=days_ago, hours=i), entities[i % len(entities)] if entities else None)
            for i in range(n)]


def test_works_with_needs_n_interactions_across_m_entities():
    ents = ["po:1", "po:2", "po:3"]
    assert d.works_with(inter("a", "b", 8, ents), {}, NOW, CFG)                       # 8 across 3
    assert not d.works_with(inter("a", "b", 7, ents), {}, NOW, CFG)                   # too few
    assert not d.works_with(inter("a", "b", 50, ["po:1"]), {}, NOW, CFG)              # chat volume on one entity never qualifies
    assert not d.works_with(inter("a", "b", 50, ["po:1", "po:2"]), {}, NOW, CFG)      # two entities is not enough
    assert not d.works_with(inter("a", "b", 50, []), {}, NOW, CFG)                    # direct chat with no entity at all
    assert not d.works_with(inter("a", "b", 20, ents, days_ago=120), {}, NOW, CFG)    # outside the window


def test_confidence_decays_and_dismissals_lower_it_and_floor_prunes():
    ents = ["po:1", "po:2", "po:3"]
    fresh = d.works_with(inter("a", "b", 12, ents, days_ago=1), {}, NOW, CFG)[0]["confidence"]
    older = d.works_with(inter("a", "b", 12, ents, days_ago=60), {}, NOW, CFG)[0]["confidence"]
    assert fresh > older
    dismissed = d.works_with(inter("a", "b", 12, ents, days_ago=1), {("a", "b"): 2}, NOW, CFG)[0]["confidence"]
    assert dismissed == pytest.approx(fresh * 0.64, abs=1e-3)
    assert d.works_with(inter("a", "b", 12, ents, days_ago=1), {("a", "b"): 20}, NOW, CFG) == []   # fell below the floor


def test_active_in_only_from_decisions_weighted_by_outcome():
    dec = [("carl", "APPROVED", "supplier:S1", NOW - timedelta(days=i)) for i in range(1, 8)]
    rej = [("carl", "REJECTED", "supplier:S2", NOW - timedelta(days=i)) for i in range(1, 8)]
    out = {r["target"]: r["confidence"] for r in d.active_in(dec + rej, {}, NOW, CFG)}
    assert out["supplier:S1"] > out["supplier:S2"] > 0                                   # same volume, weighted by outcome
    assert d.active_in(dec[:2], {}, NOW, CFG) == []                                      # below the minimum


def test_rebuild_writes_only_derived_edges_in_both_directions_and_replaces_them(g):
    store, w, _ = g
    ents = ["po:1", "po:2", "po:3"]
    r = d.rebuild(w, inter("jane", "carl", 10, ents), [], {}, NOW, CFG)
    assert r["works_with"] == 1
    edges = store.edges(TENANT, type="WORKS_WITH", as_of=NOW)
    assert {(e.src.key, e.dst.key) for e in edges} == {("jane", "carl"), ("carl", "jane")}
    assert all(e.source == "derived" and e.props["window_days"] == 90 and e.props["evidence_count"] == 10 for e in edges)
    d.rebuild(w, [], [], {}, NOW, CFG)                                                    # evidence gone -> edge gone
    assert store.edges(TENANT, type="WORKS_WITH", as_of=NOW) == []
    assert len(store.edges(TENANT, type="REPORTS_TO", as_of=NOW)) == 2                                # asserted edges untouched


# ------------------------------------------------------------------ queries
def test_who_can_act_walks_roles_authority_and_delegation(g):
    store, w, _ = g
    assert [r["subject"] for r in q.who_can_act(store, TENANT, "supplier.write", as_of=NOW)] == ["carl", "cora", "jane"]
    assert [r["subject"] for r in q.who_can_act(store, TENANT, "po.approve", as_of=NOW)] == ["carl", "cora", "fay"]
    big = q.who_can_act(store, TENANT, "po.approve", amount=200_000, as_of=NOW)
    assert [r["subject"] for r in big] == ["cora", "fay"] and "unlimited" in big[0]["reason"]
    sync.sync_hr(w, people(fay={"delegate_subject": "jane", "delegate_until": NOW + timedelta(days=2)}), now=NOW)
    cover = {r["subject"]: r["reason"] for r in q.who_can_act(store, TENANT, "po.approve", amount=200_000, as_of=NOW)}
    assert "covering for Fay" in cover["jane"]
    assert "jane" not in {r["subject"] for r in q.who_can_act(store, TENANT, "po.approve", amount=200_000, as_of=NOW + timedelta(days=3))}


def test_as_of_answers_who_was_the_approver_back_then(g):
    store, w, clock = g
    later = NOW + timedelta(days=30)
    clock.t = later
    sync.sync_rbac(w, sync.RbacSnapshot({**SNAP.assignments, "carl": ["Buyer"]}, SNAP.roles, SNAP.actions))   # Carl loses Approver
    sync.sync_hr(w, people(carl={"approval_limit": None}), now=later)
    assert "carl" not in [r["subject"] for r in q.who_can_act(store, TENANT, "po.approve", as_of=later + timedelta(hours=1))]
    assert "carl" in [r["subject"] for r in q.who_can_act(store, TENANT, "po.approve", as_of=NOW + timedelta(days=1))]


def test_who_has_acted_who_knows_and_bottlenecks(g):
    store, w, _ = g
    sync.apply_decisions(w, facts())
    acted = q.who_has_acted(store, TENANT, "supplier:S1", as_of=NOW)
    assert [a["subject"] for a in acted] == ["carl", "carl", "carl"] and acted[0]["at"] > acted[-1]["at"]
    w.assert_edge(U("jane"), "OWNS", NodeRef("Entity", "supplier:S1"), source="user_action", source_ref="ui:assign",
                  asserted_by="carl", ownership_type="primary")
    know = q.who_knows_about(store, TENANT, "supplier:S1", as_of=NOW)
    assert [k["subject"] for k in know][:2] == ["jane", "carl"] and "primary owner" in know[0]["reason"]
    bn = q.find_bottlenecks(store, TENANT, as_of=NOW)
    assert [b["subject"] for b in bn] == ["fay"] and "h to decide" in bn[0]["reason"]
    assert q.find_bottlenecks(store, TENANT, as_of=NOW - timedelta(days=100)) == []             # nothing had happened yet


def test_suggest_recipients_tiers_flag_and_ranker_cannot_widen(g):
    store, w, _ = g
    sync.apply_decisions(w, facts())
    ents = ["po:1", "po:2", "po:3"]
    d.rebuild(w, inter("jane", "dan", 10, ents), [], {}, NOW, CFG)
    w.assert_edge(U("cora"), "OWNS", NodeRef("Entity", "supplier:S1"), source="user_action", source_ref="ui", asserted_by="x",
                  ownership_type="primary")
    off = q.suggest_recipients(store, TENANT, "jane", supplier_key="supplier:S1", amount=1000, as_of=NOW)
    assert {r["tier"] for r in off} == {1} and all(r["basis"] == "asserted" for r in off)
    assert off[0]["subject"] == "carl" and off[0]["reason"] == "your manager"
    on = q.suggest_recipients(store, TENANT, "jane", supplier_key="supplier:S1", amount=1000, graph_tiers=True, as_of=NOW)
    assert {r["tier"] for r in on} >= {1, 3}
    assert [r for r in on if r["tier"] == 3][0]["basis"] == "derived"

    def greedy_ranker(subjects):                       # a vector index that "knows" people the graph did not allow
        return ["mallory", "eve"] + list(reversed(subjects))
    ranked = q.suggest_recipients(store, TENANT, "jane", supplier_key="supplier:S1", amount=1000, graph_tiers=True,
                                  ranker=greedy_ranker, as_of=NOW)
    assert {r["subject"] for r in ranked} == {r["subject"] for r in on}                         # same set: never widened
    assert "mallory" not in {r["subject"] for r in ranked}


def test_no_interaction_edge_can_exist(g):
    store, w, _ = g
    for t in ("MESSAGED", "VIEWED", "COLLABORATED_ON"):
        with pytest.raises(AdmissionError):
            w.assert_edge(U("jane"), t, U("carl"), source="derived", source_ref="x", asserted_by="x", confidence=1)
