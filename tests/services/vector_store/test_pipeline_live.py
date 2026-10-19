"""The embedding pipeline on the real test database: outbox triggers, the worker, retries, delete, rebuild, model-switch backfill.

Every test runs inside one transaction that is rolled back, so no source row, job or vector survives. PROCWISE_TEST_LIVE_DB=1.
"""
import contextlib
import datetime as dt
import os

import pytest

from src.services.vector_store import PgVectorStore
from src.services.vector_store import outbox
from src.services.vector_store.loaders import DocumentsLoader, GenericSourceLoader, default_loaders
from src.services.vector_store.profiles import EmbeddingProfile, PROFILES, table_ddl
from src.services.vector_store.rebuild import rebuild
from src.services.vector_store.worker import VectorWorker
from tests.services.vector_store.helpers import HashEmbedder, words

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = [pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1"), pytest.mark.integration]
PROFILE = PROFILES["bge-large-en-v1.5@1"]


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
def connect(conn):
    return lambda: contextlib.nullcontext(conn)


@pytest.fixture()
def store(connect):
    return PgVectorStore(connect, profile=PROFILE)


def add_source(conn, tenant, collection, source_id, text, meta="{}"):
    cur = conn.cursor()
    cur.execute("INSERT INTO proc.bp_vector_source (tenant_id, collection, source_id, content, metadata) VALUES (%s,%s,%s,%s,%s::jsonb) "
                "ON CONFLICT (tenant_id, collection, source_id) DO UPDATE SET content = EXCLUDED.content, metadata = EXCLUDED.metadata",
                (tenant, collection, source_id, text, meta))


def jobs(conn, **where):
    cur = conn.cursor()
    clause = " AND ".join(f"{k} = %s" for k in where) or "true"
    cur.execute(f"SELECT action, status, attempts, tenant_id, source_id FROM proc.bp_embedding_job WHERE {clause} ORDER BY job_id", tuple(where.values()))
    return cur.fetchall()


def worker(connect, embedder=None, **kw):
    return VectorWorker(connect, embedder or HashEmbedder(), store_factory=lambda profile: PgVectorStore(connect, profile=profile or PROFILE), **kw)


# --- the outbox is written by the database, in the writer's transaction ------------------------------------------------------------

def test_writing_a_source_enqueues_one_upsert_job_in_the_same_transaction(conn):
    add_source(conn, "t1", "uploaded", "s1", "hello world")
    assert jobs(conn, tenant_id="t1") == [("upsert", "pending", 0, "t1", "s1")]


def test_a_burst_of_writes_to_one_source_is_one_pending_job(conn):
    add_source(conn, "t1", "uploaded", "s1", "one")
    add_source(conn, "t1", "uploaded", "s1", "two")
    add_source(conn, "t1", "uploaded", "s1", "three")
    assert len(jobs(conn, tenant_id="t1")) == 1


def test_deleting_a_source_enqueues_a_delete_job(conn):
    add_source(conn, "t1", "uploaded", "s1", "hello")
    conn.cursor().execute("DELETE FROM proc.bp_vector_source WHERE tenant_id = 't1' AND source_id = 's1'")
    assert ("delete", "pending", 0, "t1", "s1") in jobs(conn, tenant_id="t1")


def test_enqueue_helper_coalesces_and_reports_whether_it_queued(conn):
    cur = conn.cursor()
    assert outbox.enqueue(cur, "t1", "documents", "invoice:I1") is True
    assert outbox.enqueue(cur, "t1", "documents", "invoice:I1") is False
    assert outbox.enqueue(cur, "t1", "documents", "invoice:I1", profile="other@2") is True       # a different target profile is its own job
    assert outbox.enqueue(cur, "t1", "documents", "invoice:I1", action="delete") is True
    with pytest.raises(ValueError):
        outbox.enqueue(cur, "", "documents", "x")


# --- the worker -----------------------------------------------------------------------------------------------------------------------

def test_the_worker_embeds_a_source_into_its_own_tenant_and_marks_the_job_done(conn, connect, store):
    add_source(conn, "t1", "uploaded", "s1", "payment terms are thirty days", '{"document_type": "policy"}')
    report = worker(connect).run_once()
    assert report["done"] == 1
    hits = store.search("t1", HashEmbedder().encode(["payment terms thirty days"])[0], {}, 5, collection="uploaded")
    assert [h.source_id for h in hits] == ["s1"] and hits[0].payload["document_type"] == "policy"
    assert store.count("t2") == 0 and jobs(conn, tenant_id="t1")[0][1] == "done"


def test_processing_the_same_job_twice_leaves_one_vector(conn, connect, store):
    add_source(conn, "t1", "uploaded", "s1", "alpha beta gamma")
    w = worker(connect)
    w.run_once()
    cur = conn.cursor()
    outbox.enqueue(cur, "t1", "uploaded", "s1")                       # the same source again
    w.run_once()
    assert store.count("t1", collection="uploaded") == 1
    # and the very same job object processed twice
    job = outbox.Job(0, "t1", "uploaded", "s1", "upsert", None, 1, 5)
    w.process(conn.cursor(), job)
    w.process(conn.cursor(), job)
    assert store.count("t1", collection="uploaded") == 1


def test_a_shorter_re_read_leaves_no_stale_tail(conn, connect, store):
    add_source(conn, "t1", "uploaded", "s1", words(900))              # three chunks
    w = worker(connect)
    w.run_once()
    assert store.count("t1") == 3
    add_source(conn, "t1", "uploaded", "s1", words(40))               # one chunk now
    w.run_once()
    assert store.count("t1") == 1


def test_deleting_the_source_row_removes_its_vectors(conn, connect, store):
    add_source(conn, "t1", "uploaded", "s1", "to be deleted")
    add_source(conn, "t1", "uploaded", "s2", "to stay")
    w = worker(connect)
    w.run_once()
    assert store.count("t1") == 2
    conn.cursor().execute("DELETE FROM proc.bp_vector_source WHERE tenant_id = 't1' AND source_id = 's1'")
    w.run_once()
    assert [h.source_id for h in store.list("t1", collection="uploaded")] == ["s2"]


def test_a_job_whose_source_has_vanished_deletes_the_vectors_rather_than_failing(conn, connect, store):
    store.upsert("t1", "ghost", HashEmbedder().encode(["ghost text"])[0], {}, collection="uploaded", content="ghost text")
    outbox.enqueue(conn.cursor(), "t1", "uploaded", "ghost")
    assert worker(connect).run_once()["done"] == 1 and store.count("t1") == 0


def test_a_failure_is_retried_with_backoff_and_then_succeeds(conn, connect, store):
    add_source(conn, "t1", "uploaded", "s1", "needs a retry")
    emb = HashEmbedder(fail_times=1)
    w = worker(connect, emb)
    assert w.run_once()["retried"] == 1
    cur = conn.cursor()
    cur.execute("SELECT status, attempts, last_error, next_attempt_at > now() FROM proc.bp_embedding_job WHERE tenant_id = 't1'")
    status, attempts, err, in_future = cur.fetchone()
    assert status == "pending" and attempts == 1 and "unavailable" in err and in_future
    assert w.run_once()["done"] == 0                                    # not due yet
    cur.execute("UPDATE proc.bp_embedding_job SET next_attempt_at = now() - interval '1 second' WHERE tenant_id = 't1'")
    assert w.run_once()["done"] == 1 and store.count("t1") == 1


def test_a_job_that_keeps_failing_ends_failed_with_the_error_recorded(conn, connect):
    add_source(conn, "t1", "uploaded", "s1", "never works")
    cur = conn.cursor()
    cur.execute("UPDATE proc.bp_embedding_job SET max_attempts = 2 WHERE tenant_id = 't1'")
    w = worker(connect, HashEmbedder(fail_times=99))
    for _ in range(2):
        w.run_once()
        cur.execute("UPDATE proc.bp_embedding_job SET next_attempt_at = now() - interval '1 second' WHERE tenant_id = 't1' AND status = 'pending'")
    cur.execute("SELECT status, attempts, last_error FROM proc.bp_embedding_job WHERE tenant_id = 't1'")
    assert cur.fetchone()[0:2] == ("failed", 2)


def test_a_job_left_processing_by_a_dead_worker_is_reclaimed_after_the_visibility_timeout(conn, connect, store):
    add_source(conn, "t1", "uploaded", "s1", "abandoned by a crash")
    cur = conn.cursor()
    cur.execute("UPDATE proc.bp_embedding_job SET status = 'processing', locked_at = now() - interval '10 minutes', attempts = 1 WHERE tenant_id = 't1'")
    assert worker(connect, visibility_seconds=300).run_once()["done"] == 1 and store.count("t1") == 1
    add_source(conn, "t1", "uploaded", "s2", "being processed right now")
    cur.execute("UPDATE proc.bp_embedding_job SET status = 'processing', locked_at = now() WHERE tenant_id = 't1' AND source_id = 's2'")
    assert worker(connect, visibility_seconds=300).run_once()["done"] == 0


def test_the_documents_loader_reads_the_text_from_the_latest_raw_row(conn):
    cur = conn.cursor()
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='proc' AND table_name='bp_invoice_raw' AND is_nullable='NO' "
                "AND column_default IS NULL")
    required = [r[0] for r in cur.fetchall()]
    base = {"source_file": "probe.pdf", "pipeline_version": "t", "parser_snapshot": '{"full_text": "invoice for widgets total due"}',
            "trace_id": "00000000-0000-0000-0000-000000000001", "promotion_status": "pending", "doc_pk_candidate": "INV-PROBE-1",
            "raw_payload": "{}", "invoice_id": "INV-PROBE-1"}
    cols = [c for c in base if c in required or c in ("source_file", "pipeline_version", "parser_snapshot", "trace_id", "promotion_status",
                                                      "doc_pk_candidate", "raw_payload", "invoice_id")]
    cur.execute(f"INSERT INTO proc.bp_invoice_raw ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})", [base[c] for c in cols])
    doc = DocumentsLoader().load(cur, "default", "invoice:INV-PROBE-1")
    assert doc is not None and "widgets" in doc.text and doc.metadata["document_type"] == "invoice" and doc.metadata["record_id"] == "INV-PROBE-1"
    assert DocumentsLoader().load(cur, "default", "invoice:NOPE") is None
    assert "invoice:INV-PROBE-1" in list(DocumentsLoader().enumerate(cur, "default"))


# --- rebuild --------------------------------------------------------------------------------------------------------------------------

def _seed(conn, tenant):
    texts = {"s1": "payment terms net thirty days", "s2": "late delivery penalty clause", "s3": "supplier insurance certificate renewal"}
    for sid, text in texts.items():
        add_source(conn, tenant, "uploaded", sid, text, '{"document_type": "policy"}')


def test_wiping_the_vectors_and_rebuilding_gives_equivalent_search_results(conn, connect, store):
    _seed(conn, "t1")
    w = worker(connect)
    w.run_once()
    q = HashEmbedder().encode(["thirty days payment terms"])[0]
    def matched(hits):                       # hits that actually matched, in rank order; equal scores (here: exact zeros) have no defined order
        return [(h.source_id, round(h.score, 6)) for h in hits if h.score > 1e-9]
    everyone = lambda: {(h.source_id, h.chunk_idx, h.content) for h in store.list("t1", collection="uploaded")}
    before_hits, before_all = matched(store.search("t1", q, {}, 3, collection="uploaded")), everyone()
    assert before_hits and store.clear("t1") == 3 and store.search("t1", q, {}, 3, collection="uploaded") == []
    report = rebuild(connect, lambda p: PgVectorStore(connect, profile=p or PROFILE), lambda p: HashEmbedder(), tenant="t1", clear=True)
    assert matched(store.search("t1", q, {}, 3, collection="uploaded")) == before_hits
    assert everyone() == before_all and report["enqueued"] >= 3 and report["failed"] == 0


def test_a_tenant_rebuild_touches_only_that_tenant(conn, connect, store):
    _seed(conn, "t1")
    _seed(conn, "t2")
    worker(connect).run_once()
    t2_before = sorted(h.source_id for h in store.list("t2", collection="uploaded"))
    rebuild(connect, lambda p: PgVectorStore(connect, profile=p or PROFILE), lambda p: HashEmbedder(), tenant="t1", clear=True)
    assert sorted(h.source_id for h in store.list("t2", collection="uploaded")) == t2_before and store.count("t1") == 3


def test_a_new_embedding_profile_is_backfilled_in_parallel_while_the_old_one_keeps_serving(conn, connect, store):
    """Model switch: a second table with its own dimension is filled from the same source rows; the old table is untouched."""
    _seed(conn, "t1")
    worker(connect).run_once()
    cur = conn.cursor()
    new = EmbeddingProfile("test@2", "test/model", "2", 16, "proc.bp_vector_chunk_test_v2")
    cur.execute(table_ddl(new))
    old_before = store.count("t1")
    report = rebuild(connect, lambda p: PgVectorStore(connect, profile=p or PROFILE), lambda p: HashEmbedder(dim=16 if p else 1024),
                     tenant="t1", profile=new, clear=False)
    new_store = PgVectorStore(connect, profile=new)
    assert report["failed"] == 0 and new_store.count("t1") == 3 and store.count("t1") == old_before == 3
    q = HashEmbedder().encode(["late delivery penalty"])[0]
    assert store.search("t1", q, {}, 1, collection="uploaded")[0].source_id == "s2"
    assert new_store.search("t1", HashEmbedder(dim=16).encode(["late delivery penalty"])[0], {}, 1, collection="uploaded")[0].source_id == "s2"


def test_the_generated_table_template_matches_what_the_migration_created(conn):
    """table_ddl() is how a model switch creates its table, so it must produce exactly the columns and indexes the v1 migration did."""
    cur = conn.cursor()
    probe = EmbeddingProfile("probe@1", PROFILE.model, PROFILE.version, PROFILE.dim, "proc.bp_vector_chunk_probe_v1")
    cur.execute(table_ddl(probe))

    def columns(table):
        cur.execute("SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_schema = 'proc' "
                    "AND table_name = %s ORDER BY column_name", (table,))
        return cur.fetchall()

    def index_shapes(table):
        # the definition with the table's own name stripped out, so v1 and the probe compare equal when they are the same shape
        cur.execute("SELECT regexp_replace(indexdef, '(ix_|uq_)?' || %s || '_?', 'T_', 'g') FROM pg_indexes WHERE tablename = %s ORDER BY 1",
                    (table, table))
        return [r[0] for r in cur.fetchall()]

    assert columns("bp_vector_chunk_probe_v1") == columns("bp_vector_chunk_bge_large_v1")
    assert index_shapes("bp_vector_chunk_probe_v1") == index_shapes("bp_vector_chunk_bge_large_v1")
    cur.execute("SELECT indexdef FROM pg_indexes WHERE tablename = 'bp_vector_chunk_probe_v1' AND indexdef LIKE '%hnsw%'")
    hnsw = cur.fetchone()[0]
    assert "vector_cosine_ops" in hnsw and "m='16'" in hnsw and "ef_construction='64'" in hnsw


def test_a_source_that_is_one_chunk_of_a_document_keeps_its_chunk_id_and_a_split_source_is_numbered(conn, connect, store):
    add_source(conn, "t1", "uploaded", "doc:3", "a short chunk", '{"chunk_id": 3, "chunk_index": 3, "document_id": "doc"}')
    add_source(conn, "t1", "uploaded", "long", words(900), '{"chunk_id": 9}')
    worker(connect).run_once()
    by_source = {}
    for h in store.list("t1", collection="uploaded"):
        by_source.setdefault(h.source_id, []).append(h.payload["chunk_id"])
    assert by_source["doc:3"] == [3]
    assert sorted(by_source["long"]) == [0, 1, 2]
