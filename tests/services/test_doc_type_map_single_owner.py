"""The physical document-type maps must agree with the vocabulary.

The Discovery Report (§5.1) found fifteen module-level maps keyed by document
type across ten files. They are not consolidated — several are genuinely
physical (type to table, type to primary key) and must exist. What must not
happen is drift: a type the vocabulary knows and a map does not, or the
reverse. This test is the single owner, enforced rather than refactored.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.concepts.vocabulary import SEED_VOCABULARY  # noqa: E402


def _pipelines_in_the_vocabulary() -> set[str]:
    return {
        dt.pipeline_doc_type
        for dt in SEED_VOCABULARY.document_types.values()
        if dt.pipeline_doc_type
    }


def test_the_vocabulary_names_exactly_the_four_live_pipelines():
    """A fifth pipeline value means tables that do not exist; a missing one
    means a working ingestion path has no vocabulary entry."""
    assert _pipelines_in_the_vocabulary() == {
        "invoice", "purchase_order", "quote", "contract"
    }


def test_persistence_knows_a_primary_key_for_every_pipeline():
    from src.services.extraction.persistence import _DOC_PK_FIELD
    missing = _pipelines_in_the_vocabulary() - set(_DOC_PK_FIELD)
    assert not missing, (
        f"pipelines with no primary-key field in persistence._DOC_PK_FIELD: "
        f"{sorted(missing)}"
    )


def test_kg_sync_knows_a_table_and_key_for_every_pipeline():
    from src.services.extraction.kg_sync import _PK_COL, _TRGT_TABLE
    for name, mapping in (("_TRGT_TABLE", _TRGT_TABLE), ("_PK_COL", _PK_COL)):
        missing = _pipelines_in_the_vocabulary() - set(mapping)
        assert not missing, (
            f"pipelines absent from kg_sync.{name}: {sorted(missing)} — "
            "those documents would never reach the knowledge graph"
        )


def test_provenance_knows_a_parent_table_for_every_pipeline():
    from src.services.extraction_v2.provenance import PARENT_TABLE_FOR_DOC_TYPE
    # This map is keyed in display form ("Purchase_Order"), the pipeline
    # value is lower-case; compare the spelling-insensitive key.
    keys = {k.lower() for k in PARENT_TABLE_FOR_DOC_TYPE}
    missing = _pipelines_in_the_vocabulary() - keys
    assert not missing, (
        f"pipelines absent from PARENT_TABLE_FOR_DOC_TYPE: {sorted(missing)}"
    )


def test_an_extraction_schema_exists_for_every_pipeline():
    schema_dir = Path(__file__).resolve().parents[2] / "extraction_schemas"
    present = {p.stem for p in schema_dir.glob("*.yaml")}
    missing = _pipelines_in_the_vocabulary() - present
    assert not missing, (
        f"pipelines with no extraction_schemas/<name>.yaml: {sorted(missing)}"
    )


def test_every_legacy_category_value_resolves_in_the_vocabulary():
    """utils.procurement_schema.CATEGORY_TO_DOC_TYPE is what the older path
    accepted. Every key must still resolve, or an upload that worked stops."""
    from src.services.concepts.vocabulary import resolve_alias
    from utils.procurement_schema import CATEGORY_TO_DOC_TYPE
    unresolved = [
        key for key in CATEGORY_TO_DOC_TYPE
        if not resolve_alias(key, SEED_VOCABULARY)
    ]
    assert not unresolved, (
        f"legacy category values the vocabulary cannot resolve: {sorted(unresolved)}"
    )
