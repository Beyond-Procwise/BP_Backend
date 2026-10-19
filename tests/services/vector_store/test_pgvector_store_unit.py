"""PgVectorStore SQL discipline, checked without a database: tenant is mandatory, filters are a whitelist, values are bound."""
import pytest

from src.services.vector_store import PgVectorStore, TenantRequired, VectorHit, VectorStore
from src.services.vector_store.profiles import EmbeddingProfile, PROFILES, active_profile

PROFILE = PROFILES["bge-large-en-v1.5@1"]
VEC = [0.1] * 1024


class Rec:
    """A connection that records SQL and answers `version` / row queries from scripts."""

    def __init__(self, rows=None, ext="0.8.0"):
        self.executed, self.rows, self.ext, self.autocommit = [], rows or [], ext, True

    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        rec = self

        class C:
            description = [("x",)]
            rowcount = 1
            def __enter__(s): return s
            def __exit__(s, *a): return False
            def execute(s, sql, params=()):
                rec.executed.append((sql, params))
                s._last = " ".join(sql.lower().split())
            def fetchone(s):
                return (rec.ext,) if "pg_extension" in s._last else None
            def fetchall(s):
                return list(rec.rows)
        return C()


def store(rec=None):
    rec = rec or Rec()
    return PgVectorStore(rec, profile=PROFILE), rec


def sql_of(rec, needle):
    return next((s, p) for s, p in rec.executed if needle in s.lower())


def test_the_store_is_a_vector_store():
    assert issubclass(PgVectorStore, VectorStore)


@pytest.mark.parametrize("bad", [None, "", "   ", 0])
def test_search_without_a_tenant_is_refused_before_any_query(bad):
    s, rec = store()
    with pytest.raises(TenantRequired):
        s.search(bad, VEC, {}, 5, collection="documents")
    assert rec.executed == []


@pytest.mark.parametrize("op", ["upsert", "delete", "list", "count", "clear"])
def test_every_operation_requires_a_tenant(op):
    s, rec = store()
    calls = {"upsert": lambda: s.upsert(None, "s1", VEC, {}, collection="c"), "delete": lambda: s.delete("", "s1"),
             "list": lambda: s.list(None, collection="c"), "count": lambda: s.count(""), "clear": lambda: s.clear(None)}
    with pytest.raises(TenantRequired):
        calls[op]()
    assert rec.executed == []


def test_search_always_filters_on_tenant_and_the_model_version_with_bound_parameters():
    s, rec = store()
    s.search("tenant-a", VEC, {}, 5, collection="documents")
    sql, params = sql_of(rec, "from proc.bp_vector_chunk")
    low = " ".join(sql.lower().split())
    assert "tenant_id = %s" in low and "embedding_model = %s" in low and "embedding_version = %s" in low and "collection = %s" in low
    assert "tenant-a" in params and PROFILE.model in params and PROFILE.version in params and "documents" in params
    assert "tenant-a" not in sql and "documents" not in sql                 # values are never inlined


def test_search_uses_cosine_distance_and_a_bounded_k():
    s, rec = store()
    s.search("t", VEC, {}, 5, collection="documents")
    sql, params = sql_of(rec, "from proc.bp_vector_chunk")
    assert "<=>" in sql and "order by embedding <=>" in " ".join(sql.lower().split()) and 5 in params
    for k in (0, -1, 10_000):
        with pytest.raises(ValueError):
            s.search("t", VEC, {}, k, collection="documents")


def test_filters_are_a_whitelist_and_values_are_bound():
    s, rec = store()
    s.search("t", VEC, {"document_type": "contract", "supplier_id": ["S1", "S2"]}, 5, collection="documents")
    sql, params = sql_of(rec, "from proc.bp_vector_chunk")
    assert "document_type = %s" in sql and "supplier_id = any(%s)" in sql.lower()
    assert "contract" in params and ["S1", "S2"] in params and "contract" not in sql


@pytest.mark.parametrize("key", ["tenant_id", "embedding", "1=1; drop table x", "vector_id", "content"])
def test_a_filter_cannot_name_tenant_or_an_unknown_column(key):
    s, rec = store()
    with pytest.raises(ValueError):
        s.search("t", VEC, {key: "x"}, 5, collection="documents")
    assert rec.executed == []


def test_a_filter_cannot_widen_the_tenant_scope():
    """tenant_id is not a filter key, so a caller cannot ask for another tenant by passing it as a filter."""
    s, rec = store()
    with pytest.raises(ValueError):
        s.search("tenant-a", VEC, {"tenant_id": "tenant-b"}, 5, collection="documents")


def test_metadata_containment_is_a_bound_json_parameter():
    s, rec = store()
    s.search("t", VEC, {"metadata": {"source": "x"}}, 5, collection="documents")
    sql, params = sql_of(rec, "from proc.bp_vector_chunk")
    assert "metadata @> %s::jsonb" in sql and '{"source": "x"}' in params


def test_iterative_scan_is_set_when_the_extension_supports_it():
    s, rec = store(Rec(ext="0.8.0"))
    s.search("t", VEC, {}, 5, collection="documents")
    assert any("hnsw.iterative_scan" in sql for sql, _ in rec.executed)


def test_iterative_scan_is_skipped_on_an_older_extension_without_failing():
    s, rec = store(Rec(ext="0.7.3"))
    s.search("t", VEC, {}, 5, collection="documents")
    assert not any("hnsw.iterative_scan" in sql for sql, _ in rec.executed)


def test_upsert_is_idempotent_by_key_and_checks_the_dimension():
    s, rec = store()
    s.upsert("t", "s1", VEC, {"document_type": "po"}, collection="documents", chunk_idx=2, content="hello")
    sql, params = sql_of(rec, "insert into proc.bp_vector_chunk")
    assert "on conflict (tenant_id, collection, source_id, chunk_idx) do update" in " ".join(sql.lower().split())
    assert params[:4] == ("t", "documents", "s1", 2) and PROFILE.model in params
    with pytest.raises(ValueError, match="dimension"):
        s.upsert("t", "s1", [0.1] * 10, {}, collection="documents")


def test_upsert_strips_the_metadata_keys_that_must_never_be_indexed():
    s, rec = store()
    s.upsert("t", "s1", VEC, {"s3_key": "bucket/x", "uploaded_by": "u", "keep": 1}, collection="documents")
    _, params = sql_of(rec, "insert into proc.bp_vector_chunk")
    blob = " ".join(str(p) for p in params)
    assert "bucket/x" not in blob and "uploaded_by" not in blob and '"keep": 1' in blob


def test_the_disallowed_key_set_matches_the_one_the_old_writer_used():
    from src.services.document_embedding_service import DISALLOWED_METADATA_KEYS as old
    from src.services.vector_store.pgvector_store import DISALLOWED_METADATA_KEYS as new
    assert set(new) == set(old)


def test_delete_scopes_to_the_tenant_and_optionally_the_collection():
    s, rec = store()
    s.delete("t", "s1")
    sql, params = sql_of(rec, "delete from proc.bp_vector_chunk")
    assert "tenant_id = %s" in sql and "source_id = %s" in sql and "collection" not in sql and params == ("t", "s1")
    s.delete("t", "s1", collection="policy")
    sql, params = [e for e in rec.executed if "delete from" in e[0].lower()][-1]
    assert "collection = %s" in sql and params == ("t", "s1", "policy")


def test_clear_is_per_tenant():
    s, rec = store()
    s.clear("t")
    sql, params = sql_of(rec, "delete from proc.bp_vector_chunk")
    assert "tenant_id = %s" in sql and params == ("t",)


def test_results_become_hits_that_look_like_the_old_search_results():
    row = ("documents", "s1", 0, "text", "po", "d1", "r1", "SUP1", None, None, "extraction", {"extra": 1}, 0.91)
    s, rec = store(Rec(rows=[row]))
    hit = s.search("t", VEC, {}, 5, collection="documents")[0]
    assert isinstance(hit, VectorHit) and hit.score == 0.91 and hit.source_id == "s1" and hit.id == "documents:s1:0"
    assert hit.payload["content"] == "text" and hit.payload["document_type"] == "po" and hit.payload["extra"] == 1
    assert "session_id" not in hit.payload                              # absent values are not invented


def test_the_active_profile_matches_the_embedding_model_and_dimension_in_settings():
    from config.settings import settings
    p = active_profile()
    assert p.model == settings.embedding_model and p.dim == settings.vector_size


def test_a_profile_table_name_must_be_a_plain_identifier():
    with pytest.raises(ValueError):
        EmbeddingProfile(name="x@1", model="m", version="1", dim=3, table="proc.t; drop table y")


def test_an_unknown_profile_is_refused(monkeypatch):
    monkeypatch.setenv("VECTOR_PROFILE", "nope@9")
    with pytest.raises(ValueError):
        active_profile()
