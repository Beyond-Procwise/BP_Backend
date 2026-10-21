"""Phase 4 against the real Postgres and Neo4j (PROCWISE_TEST_LIVE_DB=1). Postgres rows use subjects zg_*;
graph writes go to the throwaway tenant zt_int. Everything is removed afterwards."""
import os
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("PROCWISE_TEST_LIVE_DB") != "1", reason="needs the live databases")

from services.db import get_conn
from services.people import messaging as m
from services.people.admin import PeopleError
from services.people.events import DomainEvent, emit_on_cursor
from services.people.graph import activity, derive, proposed, queries, service, sync
from services.people.graph.model import NodeRef
from services.people.graph.store import Neo4jGraph
from services.people.graph.writer import GraphWriter
from services.people.wiring import drain_once

GT = "zt_int"
A, B, C = "zg_alice", "zg_bob", "zg_carol"
U = lambda k: NodeRef("User", k)


@pytest.fixture()
def env():
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    store = Neo4jGraph(driver)
    with get_conn() as conn:
        cur = conn.cursor()
        def clean():
            with driver.session() as s:
                s.run("MATCH (n) WHERE n.pw_tenant = $t DETACH DELETE n", t=GT)
            cur.execute("DELETE FROM proc.bp_notification WHERE user_subject LIKE 'zg\\_%'")
            cur.execute("DELETE FROM proc.bp_domain_event WHERE actor_id LIKE 'zg\\_%'")
            cur.execute("DELETE FROM proc.bp_conversation WHERE created_by LIKE 'zg\\_%'")
            cur.execute("DELETE FROM proc.bp_activity WHERE actor_id LIKE 'zg\\_%' OR target_user_id LIKE 'zg\\_%'")
            cur.execute("DELETE FROM proc.bp_proposed_edge WHERE src_key LIKE 'zg\\_%'")
            cur.execute("DELETE FROM proc.bp_person WHERE subject LIKE 'zg\\_%'")
            cur.execute("DELETE FROM proc.bp_admin_config WHERE config_key = 'people.graph_tiers'")
        clean()
        for s, mgr, lim in ((A, None, -1), (B, A, 50000), (C, A, None)):
            cur.execute("INSERT INTO proc.bp_person (subject, display_name, manager_subject, approval_limit, title) VALUES (%s,%s,%s,%s,%s)",
                        (s, s[3:].title(), mgr, lim, "Buyer"))
        yield conn, store, GraphWriter(store, GT)
        clean()
        driver.close()


def test_action_to_notification_to_decision_edge_and_activity_without_any_interaction_edge(env):
    conn, store, w = env
    # a conversation: behaviour
    thread = m.open_entity_thread(conn, B, "po", "PO-77", None)["id"]
    secret = "the secret negotiating position"
    m.send(conn, B, thread, f"{secret} @[Carol](zg_carol)")
    m.send(conn, B, m.open_direct(conn, B, C)["id"], "hi carol")
    # an approval: decision
    with conn.cursor() as cur:
        t0 = datetime.now(timezone.utc) - timedelta(hours=3)
        for ev in (DomainEvent("approval.requested", B, "approval", "9001", "req", event_id="approval.requested:9001", occurred_at=t0),
                   DomainEvent("approval.approved", A, "approval", "9001", "ok", event_id="approval.approved:9001:0", data={"requester": B})):
            emit_on_cursor(cur, ev)
    drain_once()
    with conn.cursor() as cur:
        cur.execute("SELECT user_subject, reason FROM proc.bp_notification WHERE user_subject LIKE 'zg\\_%' ORDER BY 1")
        notified = cur.fetchall()
    assert ("zg_carol", "mentioned") in notified and ("zg_bob", "requester") in notified          # notifications went out

    facts = sync.facts_from_events(conn, datetime.now(timezone.utc) - timedelta(days=1))
    assert sync.apply_decisions(w, facts)["written"] == 2
    approved = store.edges(GT, type="APPROVED")
    assert [(e.src.key, e.dst.key) for e in approved] == [(A, "decision:9001")]                    # decision edge in the graph
    assert approved[0].props["source"] == "user_action" and approved[0].props["source_ref"].startswith("bp_domain_event:")
    assert 2.9 * 3600e3 < approved[0].props["duration_ms"] < 3.1 * 3600e3

    with conn.cursor() as cur:                                                                     # behaviour in the activity store
        cur.execute("SELECT verb, target_user_id, entity_type, entity_id, metadata::text FROM proc.bp_activity "
                    "WHERE actor_id=%s ORDER BY id", (B,))
        rows = cur.fetchall()
    assert ("mention", C, "po", "PO-77") in [r[:4] for r in rows] and "message.sent" in [r[0] for r in rows]
    assert all(secret not in r[4] and "hi carol" not in r[4] for r in rows)                        # no content, ever
    types = {e.type for e in store.edges(GT, include_closed=True, as_of=datetime.now(timezone.utc) + timedelta(days=1)) } | \
            {e.type for e in store.edges(GT, include_closed=True)}
    assert types <= {"REQUESTED", "APPROVED"}                                                      # no interaction edge exists


def test_hr_sync_from_the_real_table_then_queries(env):
    conn, store, w = env
    people = [p for p in sync.load_people(conn) if p["subject"].startswith("zg_")]
    assert len(people) == 3 and sync.sync_hr(w, people)["changed"] > 0
    assert sync.sync_hr(w, people)["changed"] == 0
    assert [e.dst.key for e in store.edges(GT, src=U(C), type="REPORTS_TO")] == [A]
    assert [e.dst.key for e in store.edges(GT, src=U(C), type="HOLDS_TITLE")] == ["Buyer"]
    rows = queries.who_can_act(store, GT, "approve", amount=100_000)
    assert [r["subject"] for r in rows] == [A]


def test_tier_switch_is_off_by_default_and_needs_evidence(env):
    conn, store, w = env
    assert service.tiers_enabled(conn) is False
    with pytest.raises(PeopleError) as e:
        service.set_tiers(conn, True, "admin")
    assert e.value.status == 409 and "not enough evidence" in e.value.detail
    with conn.cursor() as cur:
        for i in range(40):
            activity.record(cur, "suggestion.shown", A, target_user_id=B, tier="1")
        for i in range(12):
            activity.record(cur, "suggestion.accepted", A, target_user_id=B, tier="1")
    rep = service.acceptance_report(conn)
    assert rep["tiers"]["1"]["shown"] >= 40 and rep["tiers"]["1"]["rate"] >= 0.2
    assert service.set_tiers(conn, True, "admin") == {"enabled": True} and service.tiers_enabled(conn) is True
    service.set_tiers(conn, False, "admin")
    assert service.tiers_enabled(conn) is False
    assert service.set_tiers(conn, True, "admin", force=True)["enabled"] is True        # an administrator may override


def test_proposed_edges_are_invisible_until_confirmed_and_then_tagged_inferred(env):
    conn, store, w = env
    pid = proposed.propose(conn, U(B), "ACTIVE_IN", NodeRef("Entity", "supplier:S9"), confidence=0.8, model_version="emb-1",
                           evidence={"cluster": "similar approvals"}, props={"evidence_count": 5})
    assert proposed.propose(conn, U(B), "ACTIVE_IN", NodeRef("Entity", "supplier:S9"), confidence=0.8, model_version="emb-1",
                            evidence={}) is None                                     # same inference is not queued twice
    assert store.edges(GT, type="ACTIVE_IN") == [] and queries.who_knows_about(store, GT, "supplier:S9") == []
    with pytest.raises(PeopleError):
        proposed.propose(conn, U(B), "APPROVED", NodeRef("Entity", "x"), confidence=1, model_version="m", evidence={})
    out = proposed.decide(conn, w, pid, A, confirm=True)
    edge = store.edges(GT, type="ACTIVE_IN")[0]
    assert edge.props["source"] == "inferred" and edge.props["confirmed_by"] == A and edge.props["model_version"] == "emb-1"
    assert queries.who_knows_about(store, GT, "supplier:S9")[0]["basis"] == "inferred"
    with pytest.raises(PeopleError) as e:
        proposed.decide(conn, w, pid, A, confirm=True)
    assert e.value.status == 409
    rid = proposed.propose(conn, U(C), "ACTIVE_IN", NodeRef("Entity", "supplier:S8"), confidence=0.6, model_version="emb-1", evidence={})
    proposed.decide(conn, w, rid, A, confirm=False)
    assert [x for x in store.edges(GT, type="ACTIVE_IN") if x.dst.key == "supplier:S8"] == []


def test_erase_removes_activity_and_derived_edges_keeps_decisions_and_can_pseudonymise(env):
    conn, store, w = env
    sync.apply_decisions(w, [sync.DecisionFact("approve", B, "decision:1", "approval.approved:1:0", datetime.now(timezone.utc),
                                               "test")])
    now = datetime.now(timezone.utc)
    ents = ["po:1", "po:2", "po:3"]
    inter = [derive.Interaction(B, C, now - timedelta(hours=i), ents[i % 3]) for i in range(10)]
    derive.rebuild(w, inter, [], {}, now)
    with conn.cursor() as cur:
        activity.record(cur, "view", B, entity_type="po", entity_id="1")
        activity.record(cur, "mention", C, target_user_id=B, entity_type="po", entity_id="1")
    assert len(store.edges(GT, type="WORKS_WITH")) == 2
    out = activity.erase_user(conn, store, GT, B, pseudonymise=True)
    assert out["activity_rows"] == 2 and out["derived_edges"] == 2 and out["pseudonymised_as"].startswith("erased:")
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM proc.bp_activity WHERE actor_id=%s OR target_user_id=%s", (B, B))
        assert cur.fetchone()[0] == 0
    assert store.edges(GT, type="WORKS_WITH") == []
    kept = store.edges(GT, type="APPROVED")                                           # audit trail survives, de-identified
    assert len(kept) == 1 and kept[0].src.key == out["pseudonymised_as"] and (store.node(GT, kept[0].src) or {}).get("display_name") is None


def test_activity_refuses_content_and_unknown_verbs(env):
    conn, *_ = env
    with conn.cursor() as cur:
        with pytest.raises(ValueError):
            activity.record(cur, "message.sent", A, body="hello")
        with pytest.raises(ValueError):
            activity.record(cur, "messaged", A)


def test_retention_prunes_old_rows_only(env):
    conn, *_ = env
    with conn.cursor() as cur:
        activity.record(cur, "view", A, entity_type="po", entity_id="1", occurred_at=datetime.now(timezone.utc) - timedelta(days=400))
        activity.record(cur, "view", A, entity_type="po", entity_id="2")
    assert activity.prune(conn, 365) >= 1
    with conn.cursor() as cur:
        cur.execute("SELECT entity_id FROM proc.bp_activity WHERE actor_id=%s", (A,))
        assert [r[0] for r in cur.fetchall()] == ["2"]
