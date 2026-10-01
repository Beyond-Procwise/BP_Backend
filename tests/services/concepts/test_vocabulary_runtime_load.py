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


# ---------------------------------------------------------------------------
# The version probe. Neither scalar alone is sufficient.
# ---------------------------------------------------------------------------

def _reload_count_after(monkeypatch, first, second):
    """Load once at version ``first``, then ensure again at ``second`` with the
    probe due; return how many times the tables were read in total."""
    fetches = {"n": 0}
    state = {"v": first}

    def fake_rows():
        fetches["n"] += 1
        return CONCEPT_ROWS, DOC_TYPE_ROWS

    monkeypatch.setattr(V, "_fetch_rows", fake_rows)
    monkeypatch.setattr(V, "_fetch_version", lambda: state["v"])
    V.invalidate()
    V.ensure_vocabulary(probe_seconds=0)
    assert fetches["n"] == 1
    state["v"] = second
    V.ensure_vocabulary(probe_seconds=0)
    return fetches["n"]


def test_an_unchanged_version_does_not_reload(monkeypatch):
    v = (3, "t1", 2, "t1")
    assert _reload_count_after(monkeypatch, v, v) == 1


def test_an_edit_that_keeps_the_count_still_reloads(monkeypatch):
    """Editing a row adds none: count(*) is identical, only max(recorded_at)
    moves. A count-only probe would serve the stale vocabulary forever."""
    assert _reload_count_after(
        monkeypatch, (3, "t1", 2, "t1"), (3, "t2", 2, "t1")) == 2


def test_one_row_added_while_another_is_removed_still_reloads(monkeypatch):
    """Count identical again; the new row's recorded_at is what differs."""
    assert _reload_count_after(
        monkeypatch, (3, "t1", 2, "t1"), (3, "t2", 2, "t2")) == 2


# ---------------------------------------------------------------------------
# Live-gated: the real SQL. Skip cleanly without PROCWISE_TEST_LIVE_DB=1.
# ---------------------------------------------------------------------------

import os  # noqa: E402

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
live_only = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")


@live_only
def test_the_real_probe_sees_an_edit_that_leaves_the_count_unchanged():
    """Touches ONE bp_testdb row's recorded_at and restores it in a finally.
    Refuses to run against any database other than bp_testdb."""
    from src.services.db import get_conn

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT current_database()")
        assert cur.fetchone()[0] == "bp_testdb", "mutating test: bp_testdb only"
        cur.execute("SELECT recorded_at FROM proc.bp_concept "
                    "WHERE concept_code = 'doctype.invoice'")
        original = cur.fetchone()[0]
        before = V._fetch_version()
        try:
            cur.execute("UPDATE proc.bp_concept SET recorded_at = now() "
                        "WHERE concept_code = 'doctype.invoice'")
            after = V._fetch_version()
        finally:
            cur.execute("UPDATE proc.bp_concept SET recorded_at = %s "
                        "WHERE concept_code = 'doctype.invoice'", (original,))
        restored = V._fetch_version()

    assert restored == before, "recorded_at was not restored"
    assert after != before, "probe is blind to an edit that keeps the count"


@live_only
def test_the_active_load_excludes_a_proposed_concept_everywhere():
    """The SQL filter, not just build_vocabulary's. A proposed row leaking
    into alias_index would resolve by alias while absent from document_types."""
    V.invalidate()
    v = V.ensure_vocabulary()
    assert v.source != "builtin-seed", "did not read the live tables"
    code = "doctype.policy_document"
    assert code not in v.concepts
    assert code not in v.document_types
    assert all(code not in owners for owners in v.alias_index.values())
    assert V.resolve_alias("policy", v) == ()
    assert v.concepts and v.document_types  # and it did load the active ones


@live_only
def test_the_sql_itself_never_returns_a_proposed_row():
    """build_vocabulary also filters by status, so the test above would stay
    green if the SQL filter were dropped. This one holds the SQL on its own:
    a proposed row should never even leave the database."""
    concept_rows, doc_type_rows = V._fetch_rows()
    assert concept_rows and doc_type_rows
    assert {r["status"] for r in concept_rows} == {"active"}
    assert {r["status"] for r in doc_type_rows} == {"active"}
    assert "doctype.policy_document" not in {r["concept_code"] for r in concept_rows}
    assert "doctype.policy_document" not in {r["concept_code"] for r in doc_type_rows}
