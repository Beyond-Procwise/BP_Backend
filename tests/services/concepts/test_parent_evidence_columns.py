"""The parent-evidence flag and its phrases reach Vocabulary from the table.

A column the loader does not read is a column the resolver cannot act on, and
the failure is silent: every flagged structure behaves as though unflagged.

Live-only for the table test. Run with:
    set -a && . ./.env && set +a
    CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/concepts/test_parent_evidence_columns.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts import validate as V          # noqa: E402
from src.services.concepts.vocabulary import build_vocabulary  # noqa: E402

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")


def _concept_row(code, domain="DOCUMENT_TYPE"):
    return {"concept_code": code, "domain": domain, "definition": "d",
            "not_to_be_confused_with": [], "status": "active", "rejection_reason": None}


def _doc_row(code, **over):
    row = {"concept_code": code, "role": "role.master", "default_parent_type": None,
           "execution_mode": None, "aliases": ["x"], "identifiers": [],
           "structural_signals": [], "pipeline_doc_type": "contract",
           "status": "active", "requires_parent_evidence": False,
           "parent_evidence_phrases": []}
    row.update(over)
    return row


def test_the_flag_and_phrases_reach_the_vocabulary():
    vocab = build_vocabulary(
        [_concept_row("doctype.a")],
        [_doc_row("doctype.a", requires_parent_evidence=True,
                  parent_evidence_phrases=["framework", "order of precedence"])],
        source="test",
    )
    dt = vocab.document_types["doctype.a"]
    assert dt.requires_parent_evidence is True
    assert dt.parent_evidence_phrases == ("framework", "order of precedence")


def test_an_unflagged_type_defaults_to_false_and_no_phrases():
    vocab = build_vocabulary([_concept_row("doctype.a")], [_doc_row("doctype.a")], source="test")
    dt = vocab.document_types["doctype.a"]
    assert dt.requires_parent_evidence is False
    assert dt.parent_evidence_phrases == ()


def test_a_missing_column_is_read_as_unflagged_not_as_an_error():
    """A row dict from an older query shape must not blow up the loader.

    vocabulary.build_vocabulary is public and callers construct rows by hand.
    """
    row = _doc_row("doctype.a")
    del row["requires_parent_evidence"]
    del row["parent_evidence_phrases"]
    vocab = build_vocabulary([_concept_row("doctype.a")], [row], source="test")
    assert vocab.document_types["doctype.a"].requires_parent_evidence is False


def test_a_flagged_type_with_no_phrases_is_a_violation():
    """The plan's Review Focus item 2 (specs/2026-10-02-contract-structures-plan.md): it could never match, muting the structure silently."""
    rows = [_doc_row("doctype.a", requires_parent_evidence=True, parent_evidence_phrases=[])]
    violations = V.check_flagged_types_have_parent_evidence_phrases(rows)
    assert len(violations) == 1
    assert "doctype.a" in violations[0].subject


def test_a_flagged_type_with_phrases_is_not_a_violation():
    rows = [_doc_row("doctype.a", requires_parent_evidence=True,
                     parent_evidence_phrases=["framework"])]
    assert V.check_flagged_types_have_parent_evidence_phrases(rows) == []


def test_an_unflagged_type_with_phrases_is_not_a_violation():
    """Phrases without the flag are inert, not wrong: they pre-stage a later UPDATE."""
    rows = [_doc_row("doctype.a", parent_evidence_phrases=["framework"])]
    assert V.check_flagged_types_have_parent_evidence_phrases(rows) == []


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_the_live_table_has_both_columns_with_the_right_defaults():
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT column_name, data_type, is_nullable, column_default
                 FROM information_schema.columns
                WHERE table_schema='proc' AND table_name='bp_document_type'
                  AND column_name IN ('requires_parent_evidence','parent_evidence_phrases')
                ORDER BY column_name"""
        )
        got = {r[0]: (r[1], r[2], r[3]) for r in cur.fetchall()}
    assert set(got) == {"requires_parent_evidence", "parent_evidence_phrases"}, got
    assert got["requires_parent_evidence"][0] == "boolean"
    assert got["requires_parent_evidence"][1] == "NO"          # NOT NULL
    assert "false" in (got["requires_parent_evidence"][2] or "")
    assert got["parent_evidence_phrases"][0] == "ARRAY"
    assert got["parent_evidence_phrases"][1] == "NO"


def test_run_all_reports_a_flagged_type_with_no_phrases():
    """The check must be reachable through run_all, which is what CI calls."""
    from src.services.concepts.vocabulary import SEED_VOCABULARY
    rows = [_doc_row("doctype.a", requires_parent_evidence=True, parent_evidence_phrases=[])]
    got = V.run_all(SEED_VOCABULARY, rows)
    assert [x.check for x in got] == ["flagged_types_have_parent_evidence_phrases"]
    assert got[0].subject == "doctype.a"


def test_phrase_order_is_preserved_by_the_loader():
    """Append order is part of the data: the seed-vs-table drift test compares
    ordered lists, so the loader must not sort or de-duplicate them."""
    vocab = build_vocabulary(
        [_concept_row("doctype.a")],
        [_doc_row("doctype.a", requires_parent_evidence=True,
                  parent_evidence_phrases=["b", "a"])],
        source="test",
    )
    assert vocab.document_types["doctype.a"].parent_evidence_phrases == ("b", "a")
