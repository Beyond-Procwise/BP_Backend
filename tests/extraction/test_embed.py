"""The live extraction path writes document embeddings.

Until 2026-10-08 only the legacy DataExtractionAgent embedded documents, so on
the live path (extraction/dispatch.py, via the process-monitor watcher) every
upload reached the database and never the vector index:
procwise_document_embeddings held 0 points. These tests pin the module that
closes that gap: it reads the document's own text from its _raw row, chunks it
to fit the embedder, and replaces that document's points in the collection.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from src.services.extraction import embed


# --- chunking ---------------------------------------------------------------

def test_short_text_is_one_chunk_with_whitespace_collapsed():
    assert embed.chunk_text("Invoice  INV-1\n\nTotal   £10") == ["Invoice INV-1 Total £10"]


def test_blank_text_has_no_chunks():
    assert embed.chunk_text("  \n\t ") == []
    assert embed.chunk_text(None) == []


def test_long_text_is_windowed_with_overlap_and_every_word_is_covered():
    words = [f"w{i}" for i in range(1000)]
    chunks = embed.chunk_text(" ".join(words))

    sizes = [len(c.split()) for c in chunks]
    assert max(sizes) <= embed.CHUNK_WORDS
    # consecutive windows share CHUNK_OVERLAP words
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-embed.CHUNK_OVERLAP:] == second[:embed.CHUNK_OVERLAP]
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(words)


def test_window_fits_inside_the_embedders_512_token_limit():
    # bge-large truncates at 512 tokens; English runs ~1.3 tokens a word.
    assert embed.CHUNK_WORDS * 1.3 < 512


# --- points -----------------------------------------------------------------

def _encode(texts, **_):
    return np.array([[float(len(t)), 1.0] for t in texts])


def test_points_carry_the_payload_retrieval_reads():
    points = embed.build_points(
        ["alpha beta", "gamma"], doc_type="invoice", doc_pk="INV-1", encode=_encode,
        model_name="BAAI/bge-large-en-v1.5",
    )

    assert len(points) == 2
    p = points[1].payload
    assert p["record_id"] == "INV-1"
    assert p["document_type"] == "invoice"
    assert p["chunk_id"] == 1 and p["chunk_index"] == 1
    assert p["content"] == "gamma"
    assert p["summary"] == "gamma"
    assert p["embedding_model"] == "BAAI/bge-large-en-v1.5"
    assert points[0].vector == [10.0, 1.0]


def test_payload_never_carries_an_object_path():
    """s3_key is in DISALLOWED_METADATA_KEYS: an internal object path must not
    leave for the vector index."""
    points = embed.build_points(["x"], doc_type="invoice", doc_pk="INV-1", encode=_encode)
    assert "s3_key" not in points[0].payload
    assert "file_path" not in points[0].payload


def test_point_ids_are_stable_so_a_re_extraction_overwrites_not_duplicates():
    a = embed.build_points(["x", "y"], doc_type="invoice", doc_pk="INV-1", encode=_encode)
    b = embed.build_points(["x", "y"], doc_type="invoice", doc_pk="INV-1", encode=_encode)
    other = embed.build_points(["x"], doc_type="quote", doc_pk="INV-1", encode=_encode)

    assert [p.id for p in a] == [p.id for p in b]
    assert a[0].id != other[0].id
    uuid.UUID(a[0].id)  # Qdrant accepts UUID strings


# --- the write --------------------------------------------------------------

class _Client:
    def __init__(self):
        self.deleted = []
        self.upserted = []

    def delete(self, collection_name, points_selector, wait):
        self.deleted.append((collection_name, points_selector))

    def upsert(self, collection_name, points, wait):
        self.upserted.append((collection_name, points))


class _Cur:
    def __init__(self, row):
        self.row = row
        self.sql = None
        self.params = None

    def execute(self, sql, params=()):
        self.sql, self.params = sql, params

    def fetchone(self):
        return self.row


def _agent(client):
    return SimpleNamespace(
        qdrant_client=client,
        embedding_model=SimpleNamespace(encode=_encode),
        settings=SimpleNamespace(qdrant_collection_name="procwise_document_embeddings",
                                 embedding_model="BAAI/bge-large-en-v1.5"),
    )


def test_embed_reads_the_raw_rows_text_replaces_old_points_and_writes_new_ones():
    client = _Client()
    cur = _Cur(row=({"full_text": "Invoice INV-1 total 10"},))

    n = embed.embed_document(_agent(client), cur, doc_type="invoice", doc_pk="INV-1", raw_id=7)

    assert n == 1
    assert "proc.bp_invoice_raw" in cur.sql and cur.params == (7,)
    # old points for this document go first, so a shorter re-read leaves no tail
    assert len(client.deleted) == 1
    coll, selector = client.deleted[0]
    assert coll == "procwise_document_embeddings"
    keys = {c.key: c.match.value for c in selector.filter.must}
    assert keys == {"record_id": "INV-1", "document_type": "invoice"}
    assert client.upserted[0][1][0].payload["content"] == "Invoice INV-1 total 10"


def test_snapshot_stored_as_a_json_string_is_read_too():
    client = _Client()
    cur = _Cur(row=('{"full_text": "Quote Q-9"}',))

    assert embed.embed_document(_agent(client), cur, doc_type="quote", doc_pk="Q-9", raw_id=1) == 1


@pytest.mark.parametrize("row", [None, ({},), ({"full_text": "   "},), (None,)])
def test_no_text_writes_nothing_and_deletes_nothing(row):
    client = _Client()

    assert embed.embed_document(_agent(client), _Cur(row), doc_type="invoice",
                                doc_pk="INV-1", raw_id=7) == 0
    assert client.deleted == [] and client.upserted == []


def test_a_document_without_a_key_is_not_embedded():
    client = _Client()
    cur = _Cur(row=({"full_text": "text"},))

    assert embed.embed_document(_agent(client), cur, doc_type="invoice", doc_pk="", raw_id=7) == 0
    assert cur.sql is None and client.upserted == []


def test_an_unknown_doc_type_raises_rather_than_guessing_a_table():
    with pytest.raises(KeyError):
        embed.embed_document(_agent(_Client()), _Cur(None), doc_type="memo", doc_pk="M", raw_id=1)


# --- backfill selection -----------------------------------------------------

class _ListCur:
    def __init__(self, rows):
        self.rows, self.sql = rows, None

    def execute(self, sql, params=()):
        self.sql = sql

    def fetchall(self):
        return self.rows


def test_backfill_takes_the_newest_raw_row_per_document_that_has_text():
    cur = _ListCur([("INV-1", 31), ("INV-2", 12)])

    assert embed.latest_raw_rows(cur, "invoice") == [("INV-1", 31), ("INV-2", 12)]
    sql = " ".join(cur.sql.split())
    assert "FROM proc.bp_invoice_raw" in sql
    assert "DISTINCT ON (invoice_id)" in sql
    assert "ORDER BY invoice_id, raw_id DESC" in sql
    assert "parser_snapshot ? 'full_text'" in sql


def test_backfill_keys_a_goods_receipt_by_its_note_number():
    cur = _ListCur([])
    embed.latest_raw_rows(cur, "goods_receipt")
    assert "DISTINCT ON (grn_id)" in cur.sql
