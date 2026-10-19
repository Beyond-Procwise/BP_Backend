"""Uploads and the static policy corpus: chunks become source rows (metadata intact), nothing embeds inline."""
import types

import pytest

from src.services import document_embedding_service as des
from src.services import static_policy_loader as spl
from src.services.vector_store import DEFAULT_TENANT, sources


@pytest.fixture()
def recorded(monkeypatch):
    rec = {"put": [], "wait": [], "delete_matching": [], "find": []}
    monkeypatch.setattr(sources, "put_source", lambda *a, **k: rec["put"].append((a, k)))
    monkeypatch.setattr(sources, "wait_until_embedded", lambda *a, **k: rec["wait"].append((a, k)) or "done")
    monkeypatch.setattr(sources, "delete_sources_matching", lambda *a, **k: rec["delete_matching"].append(a) or 2)
    monkeypatch.setattr(sources, "find_source_metadata", lambda *a, **k: rec["find"].append(a) or {"s3_etag": "e1"})
    return rec


def chunks():
    return [des.DocumentChunk(content="first chunk of the document", metadata={"title": "T", "section": "1"}),
            des.DocumentChunk(content="second chunk of the document", metadata={"title": "T", "section": "2"})]


def uploader(collection="uploaded_documents"):
    s = des.DocumentEmbeddingService.__new__(des.DocumentEmbeddingService)
    s.collection_name, s.client, s.embedder = collection, None, None
    s.settings = types.SimpleNamespace(static_policy_collection_name="static_policy")
    s._embedding_model_name = "BAAI/bge-large-en-v1.5"
    s._extract_text = lambda *, file_bytes, filename: des.ExtractedContent(text="some uploaded text", method="direct_decode", metadata={})
    s._build_document_chunks = lambda *a, **k: chunks()
    s._encode_chunks = lambda *a, **k: pytest.fail("embedded inline in a request handler")
    s._ensure_collection = lambda *a, **k: pytest.fail("touched Qdrant")
    rag = types.SimpleNamespace(payloads=[], upsert_payloads=lambda payloads, text_representation_key="content": rag.payloads.extend(payloads))
    s._resolve_rag_service = lambda: rag
    s._rag = rag
    return s


def test_an_upload_becomes_one_source_per_chunk_with_the_chunk_metadata_and_nothing_embeds_inline(recorded):
    s = uploader()
    out = s.embed_document(filename="notes.txt", file_bytes=b"x", metadata={"session_id": "sess-1", "s3_key": "b/k", "uploaded_by": "u"})
    assert out.chunk_count == 2 and out.collection == "uploaded_documents"
    puts = recorded["put"]
    assert [p[0][2] for p in puts] == [f"{out.document_id}:0", f"{out.document_id}:1"]
    assert all(p[0][0] == DEFAULT_TENANT and p[0][1] == "uploaded" for p in puts)
    meta0 = puts[0][0][4]
    assert meta0["session_id"] == "sess-1" and meta0["section"] == "1" and meta0["chunk_id"] == 0 and meta0["document_id"] == out.document_id
    assert "s3_key" not in meta0 and "uploaded_by" not in meta0 and "content" not in meta0
    assert puts[0][0][3] == "first chunk of the document"


def test_an_upload_waits_for_the_worker_so_it_is_searchable_when_the_request_returns(recorded):
    out = uploader().embed_document(filename="notes.txt", file_bytes=b"x")
    waited = [w[0] for w in recorded["wait"]]
    assert (DEFAULT_TENANT, "uploaded", [f"{out.document_id}:0", f"{out.document_id}:1"]) in waited


def test_an_upload_is_also_propagated_to_the_primary_store_as_before(recorded):
    s = uploader()
    out = s.embed_document(filename="notes.txt", file_bytes=b"x")
    assert len(s._rag.payloads) == 2 and all(p["record_id"] == out.document_id for p in s._rag.payloads)
    assert [p["chunk_id"] for p in s._rag.payloads] == [0, 1]


def test_the_service_can_be_built_without_a_qdrant_client_or_an_embedder():
    nick = types.SimpleNamespace(settings=None, qdrant_client=None, embedding_model=None)
    des.DocumentEmbeddingService(nick)


# --- the static policy corpus ----------------------------------------------------------------------------------------------------

def policy_loader():
    s = spl.StaticPolicyLoader.__new__(spl.StaticPolicyLoader)
    s.collection_name, s.client, s.embedder = "static_policy", None, None
    s.settings = types.SimpleNamespace(static_policy_collection_name="static_policy")
    s._embedding_model_name = "BAAI/bge-large-en-v1.5"
    s._extract_text = lambda *, file_bytes, filename: des.ExtractedContent(text="policy text", method="direct_decode", metadata={"title": "Sourcing"})
    s._build_document_chunks = lambda *a, **k: chunks()
    s._encode_chunks = lambda *a, **k: pytest.fail("embedded inline")
    s._ensure_policy_collection = lambda *a, **k: pytest.fail("touched Qdrant")
    s._remove_disallowed_payload_fields = lambda *a, **k: pytest.fail("touched Qdrant")
    return s


def test_a_policy_file_becomes_policy_sources_with_the_official_metadata_and_the_etag(recorded):
    out = policy_loader()._ingest_object(bucket="b", key="policies/sourcing.pdf", body=b"x", etag="e2", last_modified=None)
    puts = recorded["put"]
    assert out.chunk_count == 2 and [p[0][1] for p in puts] == ["policy", "policy"]
    assert [p[0][2] for p in puts] == [f"{out.document_id}:0", f"{out.document_id}:1"]
    meta = puts[0][0][4]
    assert meta["source_path"] == "s3://b/policies/sourcing.pdf" and meta["s3_etag"] == "e2" and meta["source_type"] == "Policy"
    assert meta["document_type"] == "policy" and meta["official_policy"] is True and meta["chunk_id"] == 0 and meta["title"] == "T"
    assert "s3_key" not in meta and "content" not in meta


def test_policy_change_detection_and_purge_use_the_source_rows(recorded):
    s = policy_loader()
    assert s._lookup_existing("s3://b/k.pdf") == {"s3_etag": "e1"}
    assert recorded["find"] == [(DEFAULT_TENANT, "policy", {"source_path": "s3://b/k.pdf"})]
    s._purge_existing("s3://b/k.pdf")
    assert recorded["delete_matching"] == [(DEFAULT_TENANT, "policy", {"source_path": "s3://b/k.pdf"})]
