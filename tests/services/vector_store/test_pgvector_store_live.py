"""PgVectorStore against real pgvector on the test database. Each test runs in a transaction that is rolled back.

Vectors are one-hot (a 1.0 at one position), so distances are exact and rankings predictable. Needs PROCWISE_TEST_LIVE_DB=1 and
deploy/sql/2026-10-19_pgvector_upgrade.sql + 2026-10-19_vector_store.sql applied.
"""
import contextlib
import os

import pytest

from src.services.vector_store import PgVectorStore, TenantRequired
from src.services.vector_store.profiles import EmbeddingProfile, PROFILES

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = [pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1"), pytest.mark.integration]
PROFILE = PROFILES["bge-large-en-v1.5@1"]
DIM = PROFILE.dim


def onehot(i, scale=1.0):
    v = [0.0] * DIM
    v[i] = scale
    return v


@pytest.fixture()
def conn():
    from src.services.db import get_conn
    with get_conn() as c:
        c.autocommit = False
        try:
            yield c
        finally:
            c.rollback()


@pytest.fixture()
def store(conn):
    return PgVectorStore(lambda: contextlib.nullcontext(conn), profile=PROFILE)


def test_tenant_a_never_sees_tenant_b(store):
    store.upsert("tenant-a", "doc-a", onehot(1), {"document_type": "po"}, collection="documents", content="A's secret")
    store.upsert("tenant-b", "doc-b", onehot(1), {"document_type": "po"}, collection="documents", content="B's secret")
    for tenant, other in (("tenant-a", "B's"), ("tenant-b", "A's")):
        hits = store.search(tenant, onehot(1), {}, 10, collection="documents")
        assert len(hits) == 1 and other not in (hits[0].content or "")
        assert all(h.source_id == ("doc-a" if tenant == "tenant-a" else "doc-b") for h in hits)


def test_a_tenant_with_no_vectors_gets_nothing_even_when_another_tenant_has_an_identical_one(store):
    store.upsert("tenant-b", "doc-b", onehot(2), {}, collection="documents", content="B")
    assert store.search("tenant-a", onehot(2), {}, 10, collection="documents") == []
    assert store.list("tenant-a", collection="documents") == [] and store.count("tenant-a") == 0


def test_there_is_no_tenant_less_search(store):
    for bad in (None, "", " "):
        with pytest.raises(TenantRequired):
            store.search(bad, onehot(0), {}, 5, collection="documents")


def test_a_filter_cannot_reach_another_tenant(store):
    store.upsert("tenant-b", "doc-b", onehot(3), {}, collection="documents")
    with pytest.raises(ValueError):
        store.search("tenant-a", onehot(3), {"tenant_id": "tenant-b"}, 5, collection="documents")


def test_results_are_ranked_nearest_first_with_cosine_scores(store):
    for i in (1, 2, 3):
        store.upsert("t", f"s{i}", onehot(i), {}, collection="documents", content=f"c{i}")
    mixed = [0.0] * DIM
    mixed[1], mixed[2] = 1.0, 0.5
    hits = store.search("t", mixed, {}, 3, collection="documents")
    assert [h.source_id for h in hits] == ["s1", "s2", "s3"]
    assert hits[0].score > hits[1].score > hits[2].score and abs(hits[2].score) < 1e-9


def test_upsert_twice_leaves_one_vector_and_the_latest_content(store):
    store.upsert("t", "s1", onehot(4), {"document_type": "po"}, collection="documents", content="first")
    store.upsert("t", "s1", onehot(4), {"document_type": "invoice"}, collection="documents", content="second")
    assert store.count("t", collection="documents") == 1
    hit = store.list("t", collection="documents")[0]
    assert hit.content == "second" and hit.payload["document_type"] == "invoice"


def test_chunks_of_one_source_coexist_and_delete_removes_them_all(store):
    for idx in range(3):
        store.upsert("t", "doc", onehot(idx), {}, collection="documents", chunk_idx=idx)
    store.upsert("t", "other", onehot(9), {}, collection="documents")
    assert store.count("t") == 4
    assert store.delete("t", "doc") == 3
    assert store.count("t") == 1 and store.search("t", onehot(0), {}, 5, collection="documents")[0].source_id == "other"


def test_delete_is_scoped_to_the_tenant_and_the_collection(store):
    store.upsert("tenant-a", "x", onehot(1), {}, collection="documents")
    store.upsert("tenant-b", "x", onehot(1), {}, collection="documents")
    store.upsert("tenant-a", "x", onehot(2), {}, collection="policy")
    assert store.delete("tenant-a", "x", collection="documents") == 1
    assert store.count("tenant-b") == 1 and store.count("tenant-a", collection="policy") == 1


def test_clear_removes_one_tenant_only(store):
    store.upsert("tenant-a", "x", onehot(1), {}, collection="documents")
    store.upsert("tenant-b", "y", onehot(1), {}, collection="documents")
    assert store.clear("tenant-a") == 1 and store.count("tenant-a") == 0 and store.count("tenant-b") == 1


def test_filters_narrow_within_a_tenant_and_metadata_containment_works(store):
    store.upsert("t", "s1", onehot(1), {"document_type": "po", "supplier_id": "S1", "region": "uk"}, collection="documents")
    store.upsert("t", "s2", onehot(1), {"document_type": "invoice", "supplier_id": "S2", "region": "us"}, collection="documents")
    assert [h.source_id for h in store.search("t", onehot(1), {"document_type": "po"}, 5, collection="documents")] == ["s1"]
    assert {h.source_id for h in store.search("t", onehot(1), {"supplier_id": ["S1", "S2"]}, 5, collection="documents")} == {"s1", "s2"}
    assert [h.source_id for h in store.search("t", onehot(1), {"metadata": {"region": "us"}}, 5, collection="documents")] == ["s2"]


def test_metadata_that_must_not_be_indexed_is_not_stored(store, conn):
    store.upsert("t", "s1", onehot(1), {"s3_key": "bucket/secret.pdf", "uploaded_by": "someone", "keep": "yes"}, collection="documents")
    cur = conn.cursor()
    cur.execute(f"SELECT metadata::text FROM {PROFILE.table} WHERE tenant_id = 't'")
    stored = cur.fetchone()[0]
    assert "secret.pdf" not in stored and "someone" not in stored and "keep" in stored


def test_a_different_model_version_never_sees_these_vectors(store, conn):
    """Same table, but a profile for another model version: its reads filter on model/version, so it finds nothing."""
    store.upsert("t", "s1", onehot(1), {}, collection="documents")
    other = PgVectorStore(lambda: contextlib.nullcontext(conn), profile=EmbeddingProfile(
        "other@2", "some/other-model", "2", DIM, PROFILE.table))
    assert other.search("t", onehot(1), {}, 5, collection="documents") == [] and other.count("t") == 0


def test_the_database_refuses_a_row_with_the_wrong_model_version_or_a_blank_tenant(conn):
    cur = conn.cursor()
    v = "[" + ",".join(["0"] * DIM) + "]"
    for tenant, model, version in (("t", "wrong/model", "1"), ("t", PROFILE.model, "99"), ("", PROFILE.model, "1")):
        cur.execute("SAVEPOINT s")
        with pytest.raises(Exception) as exc:
            cur.execute(f"INSERT INTO {PROFILE.table} (tenant_id, collection, source_id, embedding_model, embedding_version, embedding) "
                        "VALUES (%s, 'c', 's', %s, %s, %s::vector)", (tenant, model, version, v))
        assert "check" in str(exc.value).lower()
        cur.execute("ROLLBACK TO SAVEPOINT s")


def test_a_filtered_search_still_returns_k_when_most_nearer_rows_belong_to_another_tenant(store, conn):
    """The reason iterative scans are on. 400 of tenant B's rows sit right next to the query; tenant A's 5 are unrelated. The HNSW
    index yields B's rows first, the tenant filter discards them, and a plain filtered search comes back EMPTY. The competing b-tree
    indexes are dropped (inside this rolled-back transaction) and sequential scans disabled so the planner uses HNSW as it would on
    a large table; the test first proves the problem is real in that setup, then that the store avoids it.

    Random unit vectors, not one-hot ones: orthogonal vectors leave HNSW nodes unreachable and prove nothing.
    """
    import math
    import random
    rng = random.Random(7)

    def unit(v):
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    base = unit([rng.gauss(0, 1) for _ in range(DIM)])
    for i in range(400):
        store.upsert("tenant-b", f"b{i}", unit([b + 0.02 * rng.gauss(0, 1) for b in base]), {}, collection="documents")
    for i in range(5):
        store.upsert("tenant-a", f"a{i}", unit([rng.gauss(0, 1) for _ in range(DIM)]), {}, collection="documents")
    cur = conn.cursor()
    cur.execute(f"ANALYZE {PROFILE.table}")
    for idx in ("tenant_coll", "tenant_src", "doctype", "supplier", "session", "docid"):
        cur.execute(f"DROP INDEX proc.ix_bp_vector_chunk_bge_large_v1_{idx}")
    cur.execute(f"ALTER TABLE {PROFILE.table} DROP CONSTRAINT uq_bp_vector_chunk_bge_large_v1")
    cur.execute("SET LOCAL enable_seqscan = off")

    # 1. the problem is real here: with iterative scans off, the filtered HNSW search finds none of tenant A's rows
    q = "[" + ",".join(repr(x) for x in base) + "]"
    cur.execute("SELECT set_config('hnsw.iterative_scan', 'off', false)")
    cur.execute("SELECT set_config('hnsw.ef_search', '40', false)")
    cur.execute(f"SELECT source_id FROM {PROFILE.table} WHERE tenant_id = 'tenant-a' AND embedding_model = %s AND embedding_version = %s "
                "AND collection = 'documents' ORDER BY embedding <=> %s::vector LIMIT 5", (PROFILE.model, PROFILE.version, q))
    assert len(cur.fetchall()) < 5, "the test setup no longer exercises the recall problem"

    # 2. the store avoids it
    hits = store.search("tenant-a", base, {}, 5, collection="documents")
    assert len(hits) == 5 and all(h.source_id.startswith("a") for h in hits)
