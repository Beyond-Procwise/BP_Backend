"""The legacy extraction agent's vector writers, the collection-ensure no-ops, and AgentNick/health wiring under pgvector."""
import types

import pytest

from src.services.vector_store import DEFAULT_TENANT, sources


@pytest.fixture(autouse=True)
def pg(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "pgvector")


@pytest.fixture()
def written(monkeypatch):
    calls = []
    monkeypatch.setattr(sources, "put_source", lambda *a, **k: calls.append(a))
    return calls


def agent():
    from src.agents.data_extraction_agent import DataExtractionAgent
    a = DataExtractionAgent.__new__(DataExtractionAgent)
    a.settings = types.SimpleNamespace(qdrant_collection_name="procwise_document_embeddings")
    a.agent_nick = types.SimpleNamespace(qdrant_client=None, embedding_model=None)
    a._chunk_text = lambda text, doc_type=None: [t for t in text.split("|") if t]
    a._ensure_qdrant_collection = lambda *x, **k: pytest.fail("touched Qdrant")
    return a


def test_document_chunks_become_source_rows_with_the_old_payload_and_without_the_s3_key(written):
    agent()._vectorize_document("alpha text|beta text", "INV-1", "Invoice", "it", "bucket/inv-1.pdf")
    assert [(a[0], a[1], a[2], a[3]) for a in written] == [(DEFAULT_TENANT, "documents", "INV-1_0", "alpha text"),
                                                          (DEFAULT_TENANT, "documents", "INV-1_1", "beta text")]
    meta = written[1][4]
    assert meta["record_id"] == "INV-1" and meta["chunk_id"] == 1 and meta["summary"] == "beta text"
    assert "s3_key" not in meta and "content" not in meta and meta["document_type"] and meta["product_type"]


def test_a_document_with_no_text_or_no_chunks_writes_nothing(written):
    a = agent()
    a._vectorize_document("", "INV-1", "Invoice", None, "k")
    a._chunk_text = lambda text, doc_type=None: []
    a._vectorize_document("something", "INV-1", "Invoice", None, "k")
    assert written == []


def test_structured_data_becomes_a_header_source_and_one_per_line_item(written):
    agent()._vectorize_structured_data({"invoice_id": "INV-1", "total": 10}, [{"sku": "A", "qty": 2}, {"sku": "B"}], "Invoice", "INV-1", "it")
    ids = [a[2] for a in written]
    assert ids == ["INV-1_header", "INV-1_line_1", "INV-1_line_2"]
    assert written[0][4]["data_type"] == "header" and written[1][4]["data_type"] == "line_item" and written[1][4]["line_number"] == 1
    assert all(a[1] == "documents" and a[3].strip() for a in written)


def test_structured_data_with_no_primary_key_writes_nothing(written):
    agent()._vectorize_structured_data({"a": 1}, [], "Invoice", "", "it")
    assert written == []


# --- RAGService.ensure_collection is a no-op: the table exists, nothing to create ---------------------------------------------------

def test_ensure_collection_does_nothing_on_pgvector_and_never_touches_a_client():
    from src.services.rag_service import RAGService
    s = RAGService.__new__(RAGService)
    s.client = object()                                   # any call on this would raise AttributeError
    assert s.ensure_collection("procwise_document_embeddings") is None


# --- the health count --------------------------------------------------------------------------------------------------------------

def test_the_health_count_reads_the_vector_store_not_a_qdrant_client(monkeypatch):
    from src.services import health_signals
    store = types.SimpleNamespace(count=lambda tenant, collection=None: 167 if (tenant, collection) == (DEFAULT_TENANT, "documents") else -1)
    assert health_signals.vector_store_points_pg(store) == 167


# --- AgentNick: no Qdrant client, no wait, a loud check that the vector table is there ---------------------------------------------

def test_agent_nick_builds_no_qdrant_client_and_skips_the_wait_on_pgvector(monkeypatch):
    from src.agents.base_agent import AgentNick
    n = AgentNick.__new__(AgentNick)
    n.settings = types.SimpleNamespace()
    monkeypatch.setattr(n, "_await_qdrant_ready", lambda: pytest.fail("waited for Qdrant"), raising=False)
    monkeypatch.setattr("src.services.vector_store.get_vector_store", lambda: types.SimpleNamespace(count=lambda t, collection=None: 5))
    n._init_vector_backend()
    assert n.qdrant_client is None


def test_a_missing_vector_table_is_logged_loudly_and_does_not_stop_the_boot(monkeypatch, caplog):
    from src.agents.base_agent import AgentNick
    n = AgentNick.__new__(AgentNick)
    n.settings = types.SimpleNamespace()

    def boom(*a, **k):
        raise RuntimeError('relation "proc.bp_vector_chunk_bge_large_v1" does not exist')
    monkeypatch.setattr("src.services.vector_store.get_vector_store", lambda: types.SimpleNamespace(count=boom))
    with caplog.at_level("ERROR"):
        n._init_vector_backend()
    assert n.qdrant_client is None and any("vector store" in r.message.lower() for r in caplog.records if r.levelname == "ERROR")


def test_conversation_memory_is_available_on_pgvector_without_a_qdrant_client():
    from src.agents.base_agent import BaseAgent
    import src.agents.base_agent as ba
    me = types.SimpleNamespace(agent_nick=types.SimpleNamespace(qdrant_client=None, embedding_model=object()))
    assert ba._vector_backend_ready(me.agent_nick) is True
