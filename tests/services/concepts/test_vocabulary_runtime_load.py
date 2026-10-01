"""The loader must stay useful when the table does not.

These are pure-unit tests over build_vocabulary and the cache, with fake rows —
no database. The live agreement between table and seed is
tests/services/concepts/test_concept_table.py's job.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts import vocabulary as V  # noqa: E402


CONCEPT_ROWS = [
    {"concept_code": "role.master", "domain": "RELATIONSHIP_ROLE",
     "definition": "Governs a relationship.", "not_to_be_confused_with": [],
     "status": "active", "rejection_reason": None},
    {"concept_code": "doctype.invoice", "domain": "DOCUMENT_TYPE",
     "definition": "Demands payment.", "not_to_be_confused_with": [],
     "status": "active", "rejection_reason": None},
    {"concept_code": "doctype.policy_document", "domain": "DOCUMENT_TYPE",
     "definition": None, "not_to_be_confused_with": [],
     "status": "proposed", "rejection_reason": None},
]

DOC_TYPE_ROWS = [
    {"concept_code": "doctype.invoice", "role": "role.master",
     "default_parent_type": None, "execution_mode": None,
     "aliases": ["invoice", "Tax Invoice"], "identifiers": [],
     "structural_signals": ["amount due"], "pipeline_doc_type": "invoice",
     "status": "active"},
    {"concept_code": "doctype.policy_document", "role": "role.master",
     "default_parent_type": None, "execution_mode": None,
     "aliases": ["policy"], "identifiers": [],
     "structural_signals": [], "pipeline_doc_type": None,
     "status": "proposed"},
]


def test_build_vocabulary_indexes_aliases_case_and_space_insensitively():
    v = V.build_vocabulary(CONCEPT_ROWS, DOC_TYPE_ROWS, source="test")
    assert V.resolve_alias("  TAX   invoice ", v) == ("doctype.invoice",)


def test_a_proposed_type_never_resolves():
    """The load-bearing rule of the status column. If a proposed type resolved,
    the review step would be decoration."""
    v = V.build_vocabulary(CONCEPT_ROWS, DOC_TYPE_ROWS, source="test")
    assert V.resolve_alias("policy", v) == ()
    assert "doctype.policy_document" not in v.document_types


def test_a_colliding_alias_returns_both_candidates():
    """One alias on two active rows is a genuine ambiguity. Returning one of
    them would make the answer depend on row order."""
    rows = DOC_TYPE_ROWS + [{
        "concept_code": "doctype.quote", "role": "role.master",
        "default_parent_type": None, "execution_mode": None,
        "aliases": ["invoice"], "identifiers": [],
        "structural_signals": [], "pipeline_doc_type": "quote",
        "status": "active",
    }]
    concepts = CONCEPT_ROWS + [{
        "concept_code": "doctype.quote", "domain": "DOCUMENT_TYPE",
        "definition": "Offers a price.", "not_to_be_confused_with": [],
        "status": "active", "rejection_reason": None,
    }]
    v = V.build_vocabulary(concepts, rows, source="test")
    assert set(V.resolve_alias("invoice", v)) == {"doctype.invoice", "doctype.quote"}


def test_an_unknown_alias_resolves_to_nothing_rather_than_a_guess():
    v = V.build_vocabulary(CONCEPT_ROWS, DOC_TYPE_ROWS, source="test")
    assert V.resolve_alias("bill of lading", v) == ()


def test_an_empty_load_keeps_the_previous_vocabulary(monkeypatch):
    """Review Focus 4. An empty table must not blank the vocabulary: every
    document would come back unknown and the absences would be recorded as
    though the pages said nothing."""
    good = V.build_vocabulary(CONCEPT_ROWS, DOC_TYPE_ROWS, source="good")
    monkeypatch.setattr(V, "_active", good, raising=False)
    monkeypatch.setattr(V, "_loaded_at", 0.0, raising=False)
    monkeypatch.setattr(V, "_fetch_rows", lambda: ([], []))
    V.invalidate()
    result = V.ensure_vocabulary()
    assert result.source == "good"
    assert V.resolve_alias("invoice", result) == ("doctype.invoice",)


def test_an_unreadable_table_keeps_the_previous_vocabulary(monkeypatch):
    good = V.build_vocabulary(CONCEPT_ROWS, DOC_TYPE_ROWS, source="good")
    monkeypatch.setattr(V, "_active", good, raising=False)
    monkeypatch.setattr(V, "_loaded_at", 0.0, raising=False)

    def boom():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(V, "_fetch_rows", boom)
    V.invalidate()
    result = V.ensure_vocabulary()
    assert result.source == "good"


def test_with_nothing_ever_loaded_it_falls_back_to_the_seed(monkeypatch):
    monkeypatch.setattr(V, "_active", V.SEED_VOCABULARY, raising=False)
    monkeypatch.setattr(V, "_loaded_at", None, raising=False)
    monkeypatch.setattr(V, "_fetch_rows", lambda: ([], []))
    V.invalidate()
    result = V.ensure_vocabulary()
    assert result.source == "builtin-seed"
    # The seed must be able to resolve the four spellings the pipeline needs.
    for spelling in ("invoice", "purchase order", "quote", "contract"):
        assert V.resolve_alias(spelling, result), f"seed cannot resolve {spelling!r}"


def test_the_seed_resolves_every_spelling_the_old_map_accepted():
    """Review Focus 1. process_monitor_watcher's four-entry map accepted these
    ten spellings. Losing one breaks a working upload path."""
    v = V.SEED_VOCABULARY
    for spelling in ("invoice", "Invoice", "purchase_order", "PurchaseOrder",
                     "po", "PO", "quote", "Quote", "contract", "Contract"):
        assert V.resolve_alias(spelling, v), f"seed cannot resolve {spelling!r}"
