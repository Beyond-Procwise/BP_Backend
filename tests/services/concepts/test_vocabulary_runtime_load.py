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


_CACHE_GLOBALS = ("_active", "_loaded_at", "_probed_at", "_version",
                  "_invalidated", "_failed_at")


@pytest.fixture(autouse=True)
def _restore_the_module_cache():
    """ensure_vocabulary publishes several module globals. Without this, a test
    that exercises the cache leaves its fake vocabulary live for every later
    test in the process."""
    saved = {name: getattr(V, name) for name in _CACHE_GLOBALS}
    yield
    for name, value in saved.items():
        setattr(V, name, value)


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


def _with_status(rows, code, status):
    """A copy of ``rows`` with one row's status changed."""
    return [dict(r, status=status) if r["concept_code"] == code else dict(r)
            for r in rows]


def test_a_proposed_type_never_resolves():
    """The load-bearing rule of the status column, in BOTH halves.

    `status` sits on proc.bp_concept AND proc.bp_document_type for the same
    type, so there are two ways to be proposed and a test that flips only one
    column proves only one of them. The dangerous half is the type being active
    while its concept is not: on bp_testdb, demoting ONLY
    bp_concept('doctype.invoice') to 'proposed' left resolve_alias('invoice')
    answering ('doctype.invoice',) and pipeline_for_category('Invoice') routing
    live uploads to the invoice pipeline, with validate.run_all() == [].
    """
    # both halves proposed (the seeded shape)
    v = V.build_vocabulary(CONCEPT_ROWS, DOC_TYPE_ROWS, source="test")
    assert V.resolve_alias("policy", v) == ()
    assert "doctype.policy_document" not in v.document_types

    # half one: the CONCEPT is active, the TYPE is still proposed
    concept_active = V.build_vocabulary(
        _with_status(CONCEPT_ROWS, "doctype.policy_document", "active"),
        DOC_TYPE_ROWS, source="test")
    assert "doctype.policy_document" in concept_active.concepts
    assert V.resolve_alias("policy", concept_active) == ()
    assert "doctype.policy_document" not in concept_active.document_types

    # half two: the TYPE is active, the CONCEPT is still proposed. This is the
    # half-promotion that resolved and routed.
    type_active = V.build_vocabulary(
        CONCEPT_ROWS,
        _with_status(DOC_TYPE_ROWS, "doctype.policy_document", "active"),
        source="test")
    assert "doctype.policy_document" not in type_active.concepts
    assert V.resolve_alias("policy", type_active) == (), (
        "a type resolving while its concept is absent means it classifies and "
        "routes documents with no definition behind it")
    assert "doctype.policy_document" not in type_active.document_types


def test_the_half_promoted_type_is_reported_as_a_violation():
    """The skip in build_vocabulary is silent by design (a warning in the log,
    not a refusal to load the other 18 types), so run_all must be able to NAME
    the half-promotion. It runs over a Vocabulary, and the only Vocabulary that
    can carry one is a hand-built one — which is exactly what a caller gets from
    dataclasses.replace."""
    import dataclasses

    from src.services.concepts import validate

    good = V.build_vocabulary(CONCEPT_ROWS, DOC_TYPE_ROWS, source="test")
    assert validate.check_concepts_exist_for_every_document_type(good) == []

    # The frozen dataclass is public, so a caller CAN assemble this state.
    orphan = dataclasses.replace(good.document_types["doctype.invoice"],
                                 concept_code="doctype.policy_document")
    half = dataclasses.replace(good, document_types={
        **good.document_types, "doctype.policy_document": orphan,
    })
    bad = validate.check_concepts_exist_for_every_document_type(half)
    assert [v.subject for v in bad] == ["doctype.policy_document"]
    assert bad[0].check == "concepts_exist_for_every_document_type"
    assert bad == [v for v in validate.run_all(half)
                   if v.check == "concepts_exist_for_every_document_type"]


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


def test_a_test_that_loaded_a_fake_vocabulary_leaves_no_trace(monkeypatch):
    """Runs straight after the tests above, which publish fakes through
    ensure_vocabulary. If the cache fixture stops restoring, this sees one."""
    assert not str(V._active.source).startswith("bp_concept@3")
    assert not V._active.source.startswith("bp_concept@2")


def _count_calls(monkeypatch, rows_fn):
    calls = {"rows": 0}

    def counting():
        calls["rows"] += 1
        return rows_fn()

    monkeypatch.setattr(V, "_fetch_rows", counting)
    monkeypatch.setattr(V, "_fetch_version", lambda: (1, "t", 1, "t"))
    return calls


def _boom():
    raise RuntimeError("connection refused")


def test_a_failing_load_is_not_retried_inside_the_backoff_window(monkeypatch):
    """An outage must not become one query per document."""
    monkeypatch.setattr(V, "_active", V.SEED_VOCABULARY)
    monkeypatch.setattr(V, "_loaded_at", None)
    monkeypatch.setattr(V, "_failed_at", None)
    calls = _count_calls(monkeypatch, _boom)
    V.invalidate()
    V.ensure_vocabulary()
    V.ensure_vocabulary()
    V.ensure_vocabulary()
    assert calls["rows"] == 1


def test_an_empty_load_is_not_retried_inside_the_backoff_window(monkeypatch):
    monkeypatch.setattr(V, "_active", V.SEED_VOCABULARY)
    monkeypatch.setattr(V, "_loaded_at", None)
    monkeypatch.setattr(V, "_failed_at", None)
    calls = _count_calls(monkeypatch, lambda: ([], []))
    V.invalidate()
    V.ensure_vocabulary()
    V.ensure_vocabulary()
    assert calls["rows"] == 1


def test_the_version_is_read_before_the_rows(monkeypatch):
    """An edit landing between the two reads must leave the stored version
    OLDER than the rows (one redundant reload), never newer (stale forever)."""
    order = []
    monkeypatch.setattr(V, "_fetch_version", lambda: order.append("version") or (1, "t", 1, "t"))
    monkeypatch.setattr(V, "_fetch_rows",
                        lambda: order.append("rows") or (CONCEPT_ROWS, DOC_TYPE_ROWS))
    V.invalidate()
    V.ensure_vocabulary()
    assert order == ["version", "rows"]


def test_a_load_with_no_active_concepts_keeps_the_previous_vocabulary(monkeypatch):
    """Document types alone are not a vocabulary: an empty concepts map would
    blank every definition and role lookup."""
    good = V.build_vocabulary(CONCEPT_ROWS, DOC_TYPE_ROWS, source="good")
    monkeypatch.setattr(V, "_active", good)
    monkeypatch.setattr(V, "_loaded_at", 0.0)
    monkeypatch.setattr(V, "_failed_at", None)
    proposed_only = [r for r in CONCEPT_ROWS if r["status"] != "active"]
    _count_calls(monkeypatch, lambda: (proposed_only, DOC_TYPE_ROWS))
    V.invalidate()
    assert V.ensure_vocabulary().source == "good"


def test_a_row_that_cannot_be_built_keeps_the_previous_vocabulary(monkeypatch):
    """ensure_vocabulary never raises: a malformed row (an alias that is not a
    string) must come back as the last good vocabulary, not an exception."""
    good = V.build_vocabulary(CONCEPT_ROWS, DOC_TYPE_ROWS, source="good")
    monkeypatch.setattr(V, "_active", good)
    monkeypatch.setattr(V, "_loaded_at", 0.0)
    monkeypatch.setattr(V, "_failed_at", None)
    bad = [dict(DOC_TYPE_ROWS[0], aliases=[12345])]
    _count_calls(monkeypatch, lambda: (CONCEPT_ROWS, bad))
    V.invalidate()
    assert V.ensure_vocabulary().source == "good"


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
