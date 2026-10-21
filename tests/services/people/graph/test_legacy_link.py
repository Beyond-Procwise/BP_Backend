"""Stubs link to the pre-existing Supplier / PurchaseOrder nodes, and only when that is unambiguous.
Runs on the in-memory store always, and on real Neo4j with PROCWISE_TEST_LIVE_DB=1 (throwaway tenant zt_graph;
the pre-existing nodes used here are throwaway too, tagged zt_marker, and removed afterwards)."""
import pytest

from services.people.graph import legacy_link
from services.people.graph.model import AdmissionError, NodeRef
from services.people.graph.store import InMemoryGraph, Neo4jGraph
from tests.services.people.graph.conftest import TENANT

E = lambda k: NodeRef("Entity", k)


@pytest.fixture()
def world(store):
    """store with three stubs and a helper to plant 'pre-existing' nodes the app would already hold."""
    for k in ("supplier:ZT-S1", "supplier:ZT-S2", "supplier:ZT-S3", "po:ZT-PO1", "decision:9"):
        store.upsert_node(TENANT, E(k))

    def plant(label, prop, value, times=1):
        if isinstance(store, InMemoryGraph):
            store.legacy[(label, prop, value)] = times
        else:
            for _ in range(times):
                store._run(f"CREATE (:{label} {{{prop}: $v, zt_marker: true}})", v=value)
    yield store, plant
    if isinstance(store, Neo4jGraph):
        store._run("MATCH (n) WHERE n.zt_marker DETACH DELETE n")


def test_links_exactly_one_match_and_reports_the_rest(world):
    store, plant = world
    plant("Supplier", "supplier_id", "ZT-S1")
    ambiguous = isinstance(store, InMemoryGraph)               # the live graph has a uniqueness constraint on
    if ambiguous:                                              # Supplier.supplier_id, so two cannot exist there
        plant("Supplier", "supplier_id", "ZT-S3", times=2)
    plant("PurchaseOrder", "po_id", "ZT-PO1")
    res = legacy_link.link_all(store, TENANT)
    assert res["supplier"] == {"linked": 1, "missing": 2 - ambiguous, "ambiguous": int(ambiguous)}   # S1 / S2 / S3
    assert res["po"] == {"linked": 1, "missing": 0, "ambiguous": 0}
    assert store.entity_keys(TENANT, "supplier:") == ["supplier:ZT-S2", "supplier:ZT-S3"]   # only unlinked remain


def test_idempotent_and_picks_up_a_node_that_arrives_later(world):
    store, plant = world
    plant("Supplier", "supplier_id", "ZT-S1")
    legacy_link.link_all(store, TENANT)
    assert legacy_link.link_all(store, TENANT)["supplier"]["linked"] == 0          # nothing left to do for S1
    plant("Supplier", "supplier_id", "ZT-S2")
    assert legacy_link.link_all(store, TENANT)["supplier"]["linked"] == 1


def test_only_supplier_and_po_stubs_are_touched(world):
    store, plant = world
    plant("Supplier", "supplier_id", "9")
    legacy_link.link_all(store, TENANT)
    assert store.entity_keys(TENANT, "decision:") == ["decision:9"]


def test_label_and_property_must_come_from_the_closed_map(world):
    store, _ = world
    with pytest.raises(AdmissionError):
        store.link_legacy(TENANT, E("supplier:ZT-S1"), "Invoice", "invoice_id", "x")
    with pytest.raises(AdmissionError):
        store.link_legacy(TENANT, E("supplier:ZT-S1"), "Supplier", "supplier_id) DETACH DELETE (n", "x")


def test_link_is_invisible_to_asserted_edge_queries_and_leaves_old_node_unchanged(world):
    store, plant = world
    plant("Supplier", "supplier_id", "ZT-S1")
    legacy_link.link_all(store, TENANT)
    assert store.edges(TENANT, include_closed=True) == []                          # not an asserted fact
    if isinstance(store, Neo4jGraph):
        row = store._run("MATCH (l:Supplier {supplier_id:'ZT-S1'}) RETURN properties(l) AS p, "
                         "size([(l)-[]-() | 1]) AS rels")[0]
        assert dict(row["p"]) == {"supplier_id": "ZT-S1", "zt_marker": True} and row["rels"] == 1
        assert store.delete_edges(TENANT) == 0                                     # deleting edges can't reach the link
        assert store._run("MATCH (l:Supplier {supplier_id:'ZT-S1'})<-[:REPRESENTS]-() RETURN count(*) AS c")[0]["c"] == 1
