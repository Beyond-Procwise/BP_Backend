"""No new direct data-store access outside the gateway (a ratchet, modelled on test_egress_import_boundary).

``tests/gateway_direct_access_baseline.txt`` lists every file under ``src/`` that touches Postgres, Qdrant or Neo4j directly
today. The list may only shrink: a NEW file with direct access fails this test (put the read behind a gateway function), and a
file that no longer has direct access must be deleted from the list so the ratchet tightens. The gateway package is exempt, and so is
``src/services/vector_store``: it IS the pgvector adapter, the part of the gateway that owns embeddings and re-indexing (architecture:
"the gateway owns embedding generation, re-indexing on document change, and embedding-model versioning"). It is held to a stricter rule
below: nothing outside it may contain pgvector SQL.
Regenerate the baseline only when moving a read behind the gateway:  python tests/test_gateway_boundary.py --write
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "tests", "gateway_direct_access_baseline.txt")
EXEMPT_PACKAGES = ("src/services/gateway", "src/services/vector_store")
# pgvector SQL (distance operators, vector casts, the vector/outbox tables) is allowed ONLY in the adapter package, plus the style-exemplar
# module, which keeps each exemplar's vector on its own row by an earlier, deliberate decision (deploy/sql/2026-07-27_bp_style_engine.sql).
PGVECTOR_ALLOWED = ("src/services/vector_store/", "src/services/style/exemplars.py")
PGVECTOR_PATTERN = re.compile(r"<=>|<#>|::vector|bp_vector_chunk|bp_vector_source|bp_embedding_job")
PATTERN = re.compile(r"\.execute(?:many)?\(|QdrantClient|\.query_points\(|GraphDatabase\.driver|psycopg2\.connect\(")


def _scan():
    found = []
    for base, _dirs, files in os.walk(os.path.join(ROOT, "src")):
        rel_base = os.path.relpath(base, ROOT).replace(os.sep, "/")
        if "__pycache__" in rel_base or rel_base.startswith(EXEMPT_PACKAGES):
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                if PATTERN.search(fh.read()):
                    found.append(f"{rel_base}/{name}")
    return sorted(found)


def _baseline():
    with open(BASELINE, encoding="utf-8") as fh:
        return sorted(line.strip() for line in fh if line.strip() and not line.startswith("#"))


def test_no_new_file_reads_a_data_store_directly():
    new = sorted(set(_scan()) - set(_baseline()))
    assert not new, f"direct SQL/Qdrant/Neo4j access outside the gateway in new files: {new}. Add a gateway function instead."


def test_baseline_only_lists_files_that_still_have_direct_access():
    stale = sorted(set(_baseline()) - set(_scan()))
    assert not stale, f"these files no longer access a store directly; remove them from the baseline: {stale}"


def test_pgvector_sql_lives_only_in_the_vector_store_adapter():
    offenders = []
    for base, _dirs, files in os.walk(os.path.join(ROOT, "src")):
        for name in files:
            if not name.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(base, name), ROOT).replace(os.sep, "/")
            if rel.startswith(PGVECTOR_ALLOWED) or "__pycache__" in rel:
                continue
            with open(os.path.join(base, name), encoding="utf-8", errors="ignore") as fh:
                if PGVECTOR_PATTERN.search(fh.read()):
                    offenders.append(rel)
    assert not offenders, f"pgvector SQL outside the adapter: {offenders}. Call VectorStore / the outbox / sources instead."


if __name__ == "__main__" and "--write" in sys.argv:
    with open(BASELINE, "w", encoding="utf-8") as fh:
        fh.write("# Files with direct data-store access outside the gateway. May only shrink. See tests/test_gateway_boundary.py\n")
        fh.write("\n".join(_scan()) + "\n")
    print(len(_scan()), "files written")
