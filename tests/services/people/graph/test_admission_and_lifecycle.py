from datetime import timedelta

import pytest

from services.people.graph.model import AdmissionError, Edge, NodeRef, validate_edge
from services.people.graph.writer import GraphWriter
from tests.services.people.graph.conftest import T0, TENANT

U = lambda k: NodeRef("User", k)
E_ = lambda k: NodeRef("Entity", k)
HR = dict(source="hr", source_ref="bp_person:x", asserted_by="sync")


def writer(store, clock):
    return GraphWriter(store, TENANT, now=clock)


# ----------------------------------------------------------------------------- admission rule
def edge(type="REPORTS_TO", src=U("a"), dst=U("b"), **props):
    base = {**HR, "valid_from": T0, **props}
    return Edge(TENANT, src, type, dst, {k: v for k, v in base.items() if v is not ...})


@pytest.mark.parametrize("kw,text", [
    ({"source": ...}, "needs a source"),
    ({"source": "magic"}, "needs a source"),
    ({"source": "rbac"}, "cannot come from"),
    ({"source_ref": ""}, "source_ref"),
    ({"source_ref": ...}, "source_ref"),
    ({"asserted_by": " "}, "asserted_by"),
    ({"valid_from": ...}, "valid_from"),
    ({"valid_to": T0 - timedelta(days=1)}, "valid_to must be after"),
    ({"extra": {"nested": 1}}, "plain values"),
])
def test_edge_without_provenance_or_lifecycle_is_rejected(store, kw, text):
    with pytest.raises(AdmissionError) as e:
        store.add_edge(edge(**kw))
    assert text in str(e.value)
    assert store.edges(TENANT, include_closed=True) == []           # nothing was stored


@pytest.mark.parametrize("t", ["MESSAGED", "VIEWED", "COLLABORATED_ON", "MENTIONED"])
def test_interaction_edges_are_refused_by_name(store, t):
    with pytest.raises(AdmissionError) as e:
        store.add_edge(edge(type=t))
    assert "activity store" in str(e.value)


def test_unknown_type_wrong_labels_and_missing_extras(store):
    for bad, text in [(edge(type="LIKES"), "unknown edge type"),
                      (edge(src=NodeRef("Role", "r")), "cannot join"),
                      (edge(type="APPROVED", dst=E_("x"), source="user_action"), "missing"),          # no event_id/decided_at
                      (edge(type="WORKS_WITH", source="hr"), "cannot come from"),
                      (edge(type="WORKS_WITH", source="derived"), "missing"),
                      (edge(type="OWNS", src=U("a"), dst=E_("supplier:1"), source="inferred", ownership_type="primary"), "inferred edge needs")]:
        with pytest.raises(AdmissionError) as e:
            store.add_edge(bad)
        assert text in str(e.value)


def test_derived_and_inferred_shapes_accepted(store):
    store.add_edge(edge(type="WORKS_WITH", source="derived", confidence=0.7, evidence_count=9,
                        computed_at=T0, window_days=90))
    store.add_edge(edge(type="OWNS", dst=E_("supplier:1"), source="inferred", ownership_type="primary",
                        confirmed_by="admin", model_version="m1", confidence=0.8))
    assert {e.type for e in store.edges(TENANT, as_of=T0)} == {"WORKS_WITH", "OWNS"}


# ----------------------------------------------------------------------------- lifecycle
def test_set_single_closes_old_and_opens_new_and_is_idempotent(store, clock):
    w = writer(store, clock)
    assert w.set_single(U("jane"), "REPORTS_TO", U("carl"), **HR) == "changed"
    assert w.set_single(U("jane"), "REPORTS_TO", U("carl"), **HR) == "unchanged"
    t_before = clock()
    clock.advance(days=10)
    assert w.set_single(U("jane"), "REPORTS_TO", U("cora"), **HR) == "changed"
    now = [e.dst.key for e in store.edges(TENANT, src=U("jane"), type="REPORTS_TO")]
    assert now == ["cora"]
    # the old edge is closed, not erased: point-in-time queries still see it
    then = [e.dst.key for e in store.edges(TENANT, src=U("jane"), type="REPORTS_TO", as_of=t_before + timedelta(days=1))]
    assert then == ["carl"]
    assert len(store.edges(TENANT, src=U("jane"), type="REPORTS_TO", include_closed=True)) == 2
    assert w.set_single(U("jane"), "REPORTS_TO", None, **HR) == "changed"
    assert store.edges(TENANT, src=U("jane"), type="REPORTS_TO") == []


def test_sync_set_multi_valued(store, clock):
    w = writer(store, clock)
    D = lambda k: NodeRef("Department", k)
    w.sync_set(U("a"), "MEMBER_OF", {D("x"): {}, D("y"): {}}, **HR)
    clock.advance(days=1)
    assert w.sync_set(U("a"), "MEMBER_OF", {D("y"): {}, D("z"): {}}, **HR) == "changed"
    assert sorted(e.dst.key for e in store.edges(TENANT, src=U("a"), type="MEMBER_OF")) == ["y", "z"]
    assert w.sync_set(U("a"), "MEMBER_OF", {D("y"): {}, D("z"): {}}, **HR) == "unchanged"


def test_changed_scope_closes_and_reopens(store, clock):
    w = writer(store, clock)
    ET = NodeRef("EntityType", "approval")
    w.sync_set(U("a"), "HAS_AUTHORITY", {ET: {"action": "approve", "max_amount": 1000}}, **HR)
    clock.advance(days=3)
    assert w.sync_set(U("a"), "HAS_AUTHORITY", {ET: {"action": "approve", "max_amount": 1000}}, **HR) == "unchanged"
    assert w.sync_set(U("a"), "HAS_AUTHORITY", {ET: {"action": "approve", "max_amount": 5000}}, **HR) == "changed"
    cur = store.edges(TENANT, src=U("a"), type="HAS_AUTHORITY")
    assert [e.props["max_amount"] for e in cur] == [5000]
    old = store.edges(TENANT, src=U("a"), type="HAS_AUTHORITY", as_of=T0 + timedelta(days=1))
    assert [e.props["max_amount"] for e in old] == [1000]


def test_delegation_edge_expires_by_its_own_window(store, clock):
    w = writer(store, clock)
    w.set_single(U("a"), "DELEGATES_TO", U("b"), valid_to=T0 + timedelta(days=5), **HR)
    assert [e.dst.key for e in store.edges(TENANT, type="DELEGATES_TO", as_of=T0 + timedelta(days=2))] == ["b"]
    assert store.edges(TENANT, type="DELEGATES_TO", as_of=T0 + timedelta(days=6)) == []


def test_record_decision_is_idempotent_by_event_id_and_never_aggregated(store, clock):
    w = writer(store, clock)
    kw = dict(event_id="ev-1", decided_at=T0, asserted_by="jane", source_ref="event:ev-1", amount=500, reason_code="ok")
    assert w.record_decision(U("jane"), "APPROVED", E_("decision:9"), **kw) is not None
    assert w.record_decision(U("jane"), "APPROVED", E_("decision:9"), **kw) is None
    w.record_decision(U("jane"), "APPROVED", E_("decision:9"), **{**kw, "event_id": "ev-2"})
    assert len(store.edges(TENANT, type="APPROVED", as_of=T0 + timedelta(days=1))) == 2


def test_tenants_are_isolated(store, clock):
    w1, w2 = writer(store, clock), GraphWriter(store, "zt_other", now=clock)
    w1.set_single(U("a"), "REPORTS_TO", U("b"), **HR)
    assert store.edges("zt_other") == [] and len(store.edges(TENANT)) == 1
    w2.set_single(U("a"), "REPORTS_TO", U("c"), **HR)
    assert [e.dst.key for e in store.edges(TENANT, type="REPORTS_TO")] == ["b"]
    assert store.delete_edges("zt_other") == 1 and len(store.edges(TENANT)) == 1
