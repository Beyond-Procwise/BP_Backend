"""Each §8.1 check, proven to fail on a planted violation.

A validation function that has only ever been seen to pass is not known to
check anything. Every test here builds a vocabulary that breaks the rule and
asserts the check catches it, then asserts the real seed is clean.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts import validate as VAL  # noqa: E402
from src.services.concepts import vocabulary as V  # noqa: E402


def _vocab(concepts, doc_types):
    return V.build_vocabulary(concepts, doc_types, source="test")


def _concept(code, domain, definition="x", confused=()):
    return {"concept_code": code, "domain": domain, "definition": definition,
            "not_to_be_confused_with": list(confused), "status": "active",
            "rejection_reason": None}


def _doc_type(code, role="role.master", aliases=(), parent=None, mode=None,
              pipeline="contract"):
    return {"concept_code": code, "role": role, "default_parent_type": parent,
            "execution_mode": mode, "aliases": list(aliases), "identifiers": [],
            "structural_signals": [], "pipeline_doc_type": pipeline,
            "status": "active"}


def test_colliding_alias_is_caught():
    v = _vocab(
        [_concept("role.master", "RELATIONSHIP_ROLE"),
         _concept("doctype.a", "DOCUMENT_TYPE"),
         _concept("doctype.b", "DOCUMENT_TYPE")],
        [_doc_type("doctype.a", aliases=("order form",)),
         _doc_type("doctype.b", aliases=("Order Form",))],
    )
    violations = VAL.check_aliases_are_unambiguous(v)
    assert violations, "a shared alias must be reported"
    assert "order form" in violations[0].subject


def test_dangling_role_reference_is_caught():
    v = _vocab(
        [_concept("doctype.a", "DOCUMENT_TYPE")],
        [_doc_type("doctype.a", role="role.nonexistent")],
    )
    violations = VAL.check_every_reference_resolves(v)
    assert any("role.nonexistent" in x.detail for x in violations)


def test_dangling_parent_reference_is_caught():
    v = _vocab(
        [_concept("role.master", "RELATIONSHIP_ROLE"),
         _concept("doctype.a", "DOCUMENT_TYPE")],
        [_doc_type("doctype.a", parent="doctype.ghost")],
    )
    violations = VAL.check_every_reference_resolves(v)
    assert any("doctype.ghost" in x.detail for x in violations)


def test_dangling_not_to_be_confused_with_is_caught():
    v = _vocab(
        [_concept("role.master", "RELATIONSHIP_ROLE"),
         _concept("doctype.a", "DOCUMENT_TYPE", confused=("doctype.ghost",))],
        [_doc_type("doctype.a")],
    )
    violations = VAL.check_every_reference_resolves(v)
    assert any("doctype.ghost" in x.detail for x in violations)


def test_undefined_active_concept_is_caught():
    v = _vocab([_concept("doctype.a", "DOCUMENT_TYPE", definition="  ")], [])
    violations = VAL.check_active_concepts_are_defined(v)
    assert any(x.subject == "doctype.a" for x in violations)


def test_unknown_pipeline_target_is_caught():
    v = _vocab(
        [_concept("role.master", "RELATIONSHIP_ROLE"),
         _concept("doctype.a", "DOCUMENT_TYPE")],
        [_doc_type("doctype.a", pipeline="shipping_note")],
    )
    violations = VAL.check_pipeline_targets_exist(v)
    assert any("shipping_note" in x.detail for x in violations)


def test_a_document_type_with_no_attributes_row_is_caught():
    """A type with no role or parent cannot take part in a relationship, so a
    DOCUMENT_TYPE concept with no bp_document_type row is a gap, not a choice."""
    v = _vocab([_concept("role.master", "RELATIONSHIP_ROLE"),
                _concept("doctype.orphan", "DOCUMENT_TYPE")], [])
    violations = VAL.check_every_reference_resolves(v)
    assert any(x.subject == "doctype.orphan" for x in violations)


def test_the_real_seed_passes_every_check():
    assert VAL.run_all(V.SEED_VOCABULARY, V.SEED_DOC_TYPE_ROWS) == []


def test_duplicate_alias_inside_the_seed_itself_is_caught():
    """Task 1's seed-vs-table tests compare against the UNION of table aliases,
    so a duplicate living inside seed.py alone is invisible to them. Plant one by
    giving a second seeded type an alias an existing type already claims, and
    build the vocabulary exactly as SEED_VOCABULARY is built."""
    from src.services.concepts import seed as S

    types = sorted(S.DOCUMENT_TYPES.values(), key=lambda d: d.concept_code)
    victim, thief = next((a, b) for a in types for b in types
                         if a is not b and a.aliases)
    stolen = victim.aliases[0]
    concept_rows = [
        {"concept_code": c.concept_code, "domain": c.domain,
         "definition": c.definition,
         "not_to_be_confused_with": list(c.not_to_be_confused_with),
         "status": c.status, "rejection_reason": c.rejection_reason}
        for c in S.CONCEPTS.values()]
    doc_rows = [
        {"concept_code": d.concept_code, "role": d.role,
         "default_parent_type": d.default_parent_type,
         "execution_mode": d.execution_mode,
         "aliases": list(d.aliases) + ([stolen] if d is thief else []),
         "identifiers": list(d.identifiers),
         "structural_signals": list(d.structural_signals),
         "pipeline_doc_type": d.pipeline_doc_type, "status": d.status}
        for d in S.DOCUMENT_TYPES.values()]
    v = V.build_vocabulary(concept_rows, doc_rows, source="seed-with-duplicate")

    # The plant must survive into the vocabulary, or this proves nothing.
    assert len(v.alias_index[V.fold(stolen)]) == 2
    violations = VAL.check_aliases_are_unambiguous(v)
    assert [x.subject for x in violations] == [V.fold(stolen)]
    assert VAL.run_all(V.SEED_VOCABULARY, V.SEED_DOC_TYPE_ROWS) == []
