"""The embedding proposer. Pure ranking tests always; the queue/guard tests need the live DB (PROCWISE_TEST_LIVE_DB=1)
and use people zg_* and made-up suppliers (removed afterwards). A fake embedder keeps meaning deterministic: each
product word is its own axis, so steel suppliers are near each other and far from software ones."""
import os
import random
from datetime import datetime, timedelta, timezone

import pytest

from services.people.graph import embed_propose as ep
from services.people.graph.embed_propose import ProposerConfig, rank_candidates
from services.people.graph.model import NodeRef
from services.people.graph.store import InMemoryGraph
from services.people.graph.writer import GraphWriter

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)
WORDS = ["steel", "software", "paper", "fuel", "cleaning", "travel"]


class FakeEmbedder:
    def __init__(self):
        self.calls = 0
        self.texts = 0

    def encode(self, texts):
        self.calls += 1
        self.texts += len(texts)
        out = []
        for t in texts:
            v = [0.05] * len(WORDS)                                  # a little of everything so nothing is orthogonal
            for i, w in enumerate(WORDS):
                if w in t.lower():
                    v[i] += 1.0
            out.append(v)
        return out


def pool(per=5, extra=None):
    spec = {f"supplier:{w[:2].upper()}{i}": f"Co {w}{i}. {w} goods" for w in WORDS for i in range(1, per + 1)}
    spec.update(extra or {})
    return spec


SUPPLIERS = pool()
STEEL3 = {"supplier:ST1": 1, "supplier:ST2": 1, "supplier:ST3": 1}
CFG = ProposerConfig()


def vecs(spec):
    return {k: ep._unit(v) for k, v in zip(spec, FakeEmbedder().encode(list(spec.values())))}


def test_proposes_the_unhandled_steel_suppliers_to_a_steel_person_and_no_others():
    got = rank_candidates(vecs(SUPPLIERS), {"ann": dict(STEEL3)}, CFG)
    assert sorted(c.supplier_key for c in got) == ["supplier:ST4", "supplier:ST5"]
    assert got[0].similarity > got[0].threshold >= CFG.min_similarity
    assert {h for h, _ in got[0].similar_to} == set(STEEL3)


def test_never_proposes_a_supplier_the_person_already_decided_on():
    hist = {f"supplier:ST{i}": 1 for i in range(1, 6)}
    assert rank_candidates(vecs(SUPPLIERS), {"ann": hist}, CFG) == []


@pytest.fixture(autouse=True)
def fresh_cache():
    ep._CACHE.clear()                       # a warm cache from another test would hide whether the model was used
    yield
    ep._CACHE.clear()


def test_too_little_history_proposes_nothing():
    h = {"ann": dict(STEEL3)}
    assert rank_candidates(vecs(SUPPLIERS), h, CFG)                                          # three is enough ...
    assert rank_candidates(vecs(SUPPLIERS), h, ProposerConfig(min_history=4)) == []          # ... unless four are required


def test_the_chance_bar_is_computed_for_the_size_of_the_persons_own_history(monkeypatch):
    sizes = []
    real = ep.chance_level
    monkeypatch.setattr(ep, "chance_level", lambda v, size, user, cfg: sizes.append(size) or real(v, size, user, cfg))
    rank_candidates(vecs(SUPPLIERS), {"ann": dict(STEEL3), "bo": {f"supplier:SO{i}": 1 for i in range(1, 5)}}, CFG)
    assert sorted(sizes) == [3, 4]


def test_history_without_a_description_does_not_count_toward_the_minimum():
    assert rank_candidates(vecs(SUPPLIERS), {"ann": {"supplier:ST1": 1, "supplier:ST2": 1, "supplier:GHOST": 1}}, CFG) == []


def test_top_k_and_best_first():
    got = rank_candidates(vecs(pool(per=8)), {"ann": dict(STEEL3)}, ProposerConfig(top_k=2))
    assert len(got) == 2 and got[0].similarity >= got[1].similarity


def test_a_mixed_history_is_not_matched_to_anything():
    # steel + software + paper: the profile is a blur, no single supplier is closer than chance
    h = {"supplier:ST1": 1, "supplier:SO1": 1, "supplier:PA1": 1}
    assert rank_candidates(vecs(SUPPLIERS), {"ann": h}, CFG) == []


def test_recent_decisions_weigh_more_than_old_ones():
    d = [("bo", "APPROVED", k, NOW - timedelta(days=170)) for k in ("supplier:ST1", "supplier:ST2", "supplier:ST3")]
    d += [("bo", "APPROVED", f"supplier:SO{i}", NOW - timedelta(days=1)) for i in (1, 2, 3)]
    got = rank_candidates(vecs(SUPPLIERS), ep.history_by_person(d, NOW, CFG), CFG)
    assert {c.supplier_key for c in got} == {"supplier:SO4", "supplier:SO5"}


def test_only_supplier_keys_count_as_history():
    d = [("ann", "APPROVED", "category:IT", NOW), ("ann", "APPROVED", "supplier:ST1", NOW)]
    assert ep.history_by_person(d, NOW, CFG) == {"ann": {"supplier:ST1": pytest.approx(1.0)}}


def test_a_corpus_with_no_real_structure_yields_no_proposals():
    """The seeded test corpus looks like this: every supplier buys a random mix, and short-text embeddings of random mixes
    are all alike. A fixed cut-off would flood the queue; the chance-level bar must reject everything."""
    rng = random.Random(7)
    v = {f"supplier:R{i}": ep._unit([1.0 + 0.1 * rng.random() for _ in range(12)]) for i in range(60)}
    hist = {"ann": {f"supplier:R{i}": 1 for i in range(4)}}
    assert rank_candidates(v, hist, ProposerConfig(min_similarity=0.6)) == []


def test_the_bar_is_higher_for_longer_histories_in_a_blurry_space():
    rng = random.Random(3)
    v = {f"supplier:R{i}": ep._unit([1.0 + 0.5 * rng.random() for _ in range(12)]) for i in range(80)}
    assert ep.chance_level(v, 12, "u", CFG) > ep.chance_level(v, 3, "u", CFG)      # bigger averages sit closer to everything


def test_unchanged_descriptions_are_not_embedded_twice():
    ep._CACHE.clear()
    emb = FakeEmbedder()
    texts = {k: t for k, t in list(SUPPLIERS.items())[:10]}
    a = ep.embed_cached(emb, texts)
    b = ep.embed_cached(emb, texts)
    assert a == b and emb.texts == 10
    ep.embed_cached(emb, dict(texts, **{"supplier:NEW": "Newco. steel"}))
    assert emb.texts == 11


# ------------------------------------------------------------------ against the live queue
live = pytest.mark.skipif(os.environ.get("PROCWISE_TEST_LIVE_DB") != "1", reason="needs the live databases")
T = "zt_graph"


@pytest.fixture()
def env():
    from services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        def clean():
            cur.execute("DELETE FROM proc.bp_proposed_edge WHERE src_key LIKE 'zg\\_%'")
            cur.execute("DELETE FROM proc.bp_person WHERE subject LIKE 'zg\\_%'")
        clean()
        for s in ("zg_ann", "zg_off"):
            cur.execute("INSERT INTO proc.bp_person (subject, display_name, is_active) VALUES (%s,%s,%s)", (s, s, s != "zg_off"))
        yield conn, cur
        clean()


def seeded_store(users):
    store = InMemoryGraph(); w = GraphWriter(store, ep.TENANT, now=lambda: NOW - timedelta(days=2))
    for u in users:
        for s in ("ST1", "ST2", "ST3"):
            w.record_decision(NodeRef("User", u), "APPROVED", NodeRef("Entity", f"decision:{u}{s}"), event_id=f"{u}{s}",
                              decided_at=NOW - timedelta(days=2), asserted_by=u, source_ref="t", supplier_key=f"supplier:{s}")
    return store


def describe(conn, limit, also=()):
    return dict(SUPPLIERS)


@live
def test_run_queues_proposals_that_are_invisible_to_the_graph_and_never_repeats(env):
    conn, cur = env
    store = seeded_store(["zg_ann"])
    emb = FakeEmbedder()
    stats = ep.run(conn, store, emb, now=NOW, cfg=CFG, describe=describe)
    assert stats["proposed"] == 2 and stats["people"] == 1
    from services.people.graph import proposed
    mine = [x for x in proposed.list_proposals(conn) if x["from"]["key"] == "zg_ann"]
    assert sorted(x["to"]["key"] for x in mine) == ["supplier:ST4", "supplier:ST5"]
    p = mine[0]
    assert p["model_version"] == ep.MODEL_VERSION and p["evidence"]["chance_level"] < p["confidence"]
    assert p["evidence"]["score_kind"] == "cosine_similarity" and len(p["evidence"]["similar_to"]) == 3
    assert store.edges(ep.TENANT, type="ACTIVE_IN", include_closed=True) == []           # nothing reached the graph
    assert ep.run(conn, store, emb, now=NOW, cfg=CFG, describe=describe)["proposed"] == 0       # not twice
    # a REJECTED proposal suppresses it for good, even under a new model version
    cur.execute("UPDATE proc.bp_proposed_edge SET status='rejected' WHERE src_key='zg_ann'")
    cur.execute("UPDATE proc.bp_proposed_edge SET model_version='older' WHERE src_key='zg_ann'")
    assert ep.run(conn, store, emb, now=NOW, cfg=CFG, describe=describe)["proposed"] == 0


@live
def test_inactive_people_get_nothing_and_the_model_is_not_loaded_without_history(env):
    conn, cur = env
    emb = FakeEmbedder()
    assert ep.run(conn, seeded_store(["zg_off"]), emb, now=NOW, cfg=CFG, describe=describe)["proposed"] == 0
    assert ep.run(conn, InMemoryGraph(), emb, now=NOW, cfg=CFG, describe=describe)["people"] == 0
    assert emb.calls == 0                                                                   # no history: no embedding work


@live
def test_pending_cap(env):
    conn, cur = env
    spec = pool(per=12)
    stats = ep.run(conn, seeded_store(["zg_ann"]), FakeEmbedder(), now=NOW,
                   cfg=ProposerConfig(top_k=20, max_pending=3), describe=lambda c, l, also=(): spec)
    assert stats["proposed"] == 3 and stats["suppressed"] > 0


@live
def test_confirming_a_proposal_writes_one_inferred_edge_naming_the_confirmer(env):
    conn, cur = env
    from services.people.graph import proposed
    store = seeded_store(["zg_ann"])
    ep.run(conn, store, FakeEmbedder(), now=NOW, cfg=CFG, describe=describe)
    p = sorted((x for x in proposed.list_proposals(conn) if x["from"]["key"] == "zg_ann"), key=lambda x: x["to"]["key"])[0]
    out = proposed.decide(conn, GraphWriter(store, ep.TENANT, now=lambda: NOW), p["id"], "zg_boss", confirm=True)
    (e,) = store.edges(ep.TENANT, type="ACTIVE_IN", as_of=NOW)
    assert (e.src.key, e.dst.key, e.props["source"], e.props["confirmed_by"]) == ("zg_ann", "supplier:ST4", "inferred", "zg_boss")
    assert e.props["model_version"] == ep.MODEL_VERSION and out["status"] == "confirmed"
