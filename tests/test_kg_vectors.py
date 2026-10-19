"""The knowledge graph's vectors go through the application's VectorStore (pgvector), not Qdrant."""
from procurement_knowledge_graph.vectors import COLLECTIONS, KnowledgeGraphVectors


class FakeStore:
    def __init__(self):
        self.rows = {}

    def upsert(self, tenant_id, source_id, embedding, metadata, *, collection, chunk_idx=0, content=None):
        self.rows[(tenant_id, collection, source_id)] = (list(embedding), dict(metadata), content)

    def search(self, tenant_id, query_embedding, filters, k, *, collection=None):
        from src.services.vector_store import VectorHit
        hits = []
        for (t, c, sid), (vec, meta, content) in self.rows.items():
            if t == tenant_id and c == collection:
                score = sum(a * b for a, b in zip(vec, query_embedding))
                hits.append(VectorHit(id=f"{c}:{sid}:0", source_id=sid, chunk_idx=0, collection=c, score=score, content=content, payload=meta))
        return sorted(hits, key=lambda h: -h.score)[:k]

    def count(self, tenant_id, *, collection=None):
        return sum(1 for (t, c, _s) in self.rows if t == tenant_id and (collection is None or c == collection))


class FakeEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0] if "alpha" in t else [0.0, 1.0] for t in texts]


def make():
    return KnowledgeGraphVectors(FakeStore(), FakeEmbedder(), tenant_id="t1")


def test_upsert_then_search_returns_payload_with_score_best_first():
    v = make()
    v.upsert("suppliers", [{"id": "S1", "text": "alpha supplier", "payload": {"supplier_id": "S1", "risk_score": 3}},
                           {"id": "S2", "text": "beta supplier", "payload": {"supplier_id": "S2"}}])
    out = v.search("suppliers", "alpha", limit=2)
    assert [o["supplier_id"] for o in out] == ["S1", "S2"]
    assert out[0]["risk_score"] == 3 and out[0]["score"] == 1.0


def test_collections_are_kg_prefixed_and_isolated():
    v = make()
    v.upsert("contracts", [{"id": "C1", "text": "alpha", "payload": {"contract_id": "C1"}}])
    assert v.search("suppliers", "alpha") == []
    assert all(c.startswith("kg_") for c in COLLECTIONS.values())
    assert v.count() == 1 and v.count("contracts") == 1


def test_empty_upsert_is_a_noop():
    v = make()
    v.upsert("suppliers", [])
    assert v.count() == 0


def test_package_no_longer_references_qdrant():
    """Source-level (the package needs loguru/neo4j to import, which the test venv lacks)."""
    import pathlib
    for f in pathlib.Path("procurement_knowledge_graph").glob("*.py"):
        text = f.read_text().lower()
        code = [line for line in text.splitlines() if "qdrant" in line and not line.lstrip().startswith(("#", '"""'))]
        assert [l for l in code if "import" in l or "qdrantclient" in l or "qdrantconfig" in l] == [], f
