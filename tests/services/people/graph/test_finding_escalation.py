"""A person escalating a finding on a purchase order becomes an ESCALATED edge to the stub po:<po_id>, which the
daily link step then ties to the existing PurchaseOrder node. Only that verb, only people, only POs.
Pure tests always; the SQL source runs with PROCWISE_TEST_LIVE_DB=1 on rows tagged zt_finding / zg_* (removed after)."""
import os
from datetime import datetime, timedelta, timezone

import pytest

from services.people.graph import derive, legacy_link, queries, sync
from services.people.graph.model import NodeRef
from services.people.graph.store import InMemoryGraph
from services.people.graph.writer import GraphWriter

T = "zt_graph"
NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def fact(did=1, po="PO-ZT-1", by="zg_alice", sup="supplier:SUP-ZT"):
    return sync.DecisionFact("escalate", by, f"po:{po}", f"finding.escalated:{did}", NOW, f"bp_decision:{did}",
                             supplier_key=sup, escalated_to=sync.FINDINGS_QUEUE)


def test_escalation_writes_an_escalated_edge_to_the_po_stub_and_is_idempotent():
    store = InMemoryGraph(); w = GraphWriter(store, T, now=lambda: NOW)
    assert sync.apply_decisions(w, [fact()]) == {"written": 1, "skipped": 0}
    assert sync.apply_decisions(w, [fact()]) == {"written": 0, "skipped": 1}
    (e,) = store.edges(T, type="ESCALATED", include_closed=True)
    assert (e.src.key, e.dst.key) == ("zg_alice", "po:PO-ZT-1")
    assert e.props["source"] == "user_action" and e.props["source_ref"] == "bp_decision:1"
    assert e.props["escalated_to"] == "findings_queue" and e.props["supplier_key"] == "supplier:SUP-ZT"
    assert store.node(T, NodeRef("Entity", "po:PO-ZT-1"))["entity_type"] == "po"


def test_the_stub_is_then_linked_to_the_existing_purchase_order():
    store = InMemoryGraph(); w = GraphWriter(store, T, now=lambda: NOW)
    sync.apply_decisions(w, [fact()])
    store.legacy[("PurchaseOrder", "po_id", "PO-ZT-1")] = 1
    assert legacy_link.link_all(store, T)["po"] == {"linked": 1, "missing": 0, "ambiguous": 0}
    assert store.linked_to(T, NodeRef("Entity", "po:PO-ZT-1")) == ("PurchaseOrder", "po_id", "PO-ZT-1")


def test_escalating_does_not_make_someone_active_in_the_supplier():
    store = InMemoryGraph(); w = GraphWriter(store, T, now=lambda: NOW)
    sync.apply_decisions(w, [fact(did=i) for i in range(1, 12)])
    assert derive.load_decisions(store, T, NOW - timedelta(days=90)) == []


# ------------------------------------------------------------------ the SQL source (live)
live = pytest.mark.skipif(os.environ.get("PROCWISE_TEST_LIVE_DB") != "1", reason="needs the live databases")


@pytest.fixture()
def rows():
    from services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        def clean():
            cur.execute("DELETE FROM proc.bp_decision WHERE created_by LIKE 'zg\\_%' OR subject_id IN "
                        "(SELECT discrepancy_id::text FROM proc.bp_extraction_discrepancy WHERE source_file='zt_finding')")
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE source_file='zt_finding'")
        clean()
        def finding(doc_type, pk):
            cur.execute("INSERT INTO proc.bp_extraction_discrepancy (doc_type, source_file, field_name, issue_type, severity, "
                        "doc_pk_candidate) VALUES (%s,'zt_finding','total','mismatch','warning',%s) RETURNING discrepancy_id", (doc_type, pk))
            return str(cur.fetchone()[0])
        def decision(fid, verb, by, status="actioned", sup=None):
            cur.execute("INSERT INTO proc.bp_decision (subject_type, subject_id, supplier_id, decision, status, actioned_by, "
                        "actioned_at, created_by) VALUES ('finding',%s,%s,%s,%s,%s,%s,%s) RETURNING decision_id",
                        (fid, sup, verb, status, by, None if by is None else datetime.now(timezone.utc), by or "zg_system"))
            return cur.fetchone()[0]
        yield finding, decision, cur
        clean()


@live
def test_only_a_persons_escalation_on_a_po_finding_becomes_a_fact(rows):
    finding, decision, cur = rows
    po1, po2, inv, blank = finding("purchase_order", "PO-ZT-1"), finding("purchase_order", " PO-ZT-2 "), finding("invoice", "INV-ZT"), finding("purchase_order", "  ")
    want1 = decision(po1, "escalate", "zg_alice", sup="SUP-ZT")
    want2 = decision(po2, "escalate", "zg_bob", status="overridden")
    decision(po1, "flag", "zg_alice")                       # other verb: not yet ruled on
    decision(po1, "dismiss", "zg_alice")
    decision(inv, "escalate", "zg_alice")                   # not a PO
    decision(blank, "escalate", "zg_alice")                 # no document id
    decision(po1, "escalate", "agent:zg_bot")               # not a person
    decision(po1, "escalate", "system:zg_job")
    decision(po1, "escalate", None, status="open")          # raised, nobody acted
    got = {f.event_id: f for f in sync.facts_from_findings(cur.connection) if "ZT" in f.entity_key}
    assert set(got) == {f"finding.escalated:{want1}", f"finding.escalated:{want2}"}
    assert got[f"finding.escalated:{want1}"].entity_key == "po:PO-ZT-1"
    assert got[f"finding.escalated:{want1}"].supplier_key == "supplier:SUP-ZT"
    assert got[f"finding.escalated:{want2}"].entity_key == "po:PO-ZT-2"           # padding trimmed
    store = InMemoryGraph(); w = GraphWriter(store, T)
    assert sync.apply_decisions(w, list(got.values()))["written"] == 2
