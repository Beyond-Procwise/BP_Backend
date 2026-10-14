"""The 3,051-row corpus's free-text contract_type, read as a structure.

Read-only: proc.bp_contract_master.contract_type is source data and is never
written. The mapping is a function of the vocabulary, so confirming a concept
takes effect with no backfill and leaves no second copy to drift.

Live test needs the database. Run with:
    set -a && . ./.env && set +a
    CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/concepts/test_contract_type_map.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts import contract_type_map as M   # noqa: E402

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")

#: Every distinct value in proc.bp_contract_master, measured 2026-10-02, with the
#: structure it must resolve to. None means "must not resolve".
CORPUS_VALUES = {
    "Consulting": "doctype.consulting_agreement",
    "NDA": "doctype.nda",
    "SLA": "doctype.sla",
    "Master Agreement": "doctype.master_agreement",
    "Service Agreement": "doctype.service_agreement",
    "Service Contract": "doctype.service_agreement",
    "Invoice": "doctype.invoice",
    "Amendment": "doctype.variation",
    "Purchase Order": "doctype.order",
    "Policy": None,                 # doctype.policy_document is status='proposed'
    "Service": None,
    "Indirect Procurement": None,
    None: None,
}


@pytest.fixture(scope="module")
def live_vocab():
    """The vocabulary, asserted to have come from the database.

    ensure_vocabulary() is fail-soft: on a failed or empty read it returns the
    built-in seed. Without this assertion a database outage would make every
    'live' test below pass against the seed.
    """
    from src.services.concepts import vocabulary as V
    vocab = V.ensure_vocabulary()
    assert vocab.source.startswith("bp_concept@"), (
        f"vocabulary did not come from the database: {vocab.source!r}")
    return vocab


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
@pytest.mark.parametrize("value,expected", list(CORPUS_VALUES.items()))
def test_every_corpus_value_maps_as_measured(live_vocab, value, expected):
    assert M.structure_for_contract_type(value) == expected


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_a_proposed_concept_never_resolves(live_vocab):
    """'Policy' is the mechanism working, not a gap.

    doctype.policy_document claims the alias 'policy'. Its concept is
    status='proposed', so it must not resolve -- if it did, confirming it would
    be moot and an unreviewed structure would be feeding the maths.
    """
    assert M.structure_for_contract_type("Policy") is None
    assert M.structure_for_contract_type("policy") is None


@pytest.fixture(scope="module")
def corpus_coverage(live_vocab):
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT contract_type, count(*) FROM proc.bp_contract_master GROUP BY 1"
        )
        rows = [{"contract_type": r[0], "n": r[1]} for r in cur.fetchall()]
    return sum(r["n"] for r in rows), M.coverage(rows, vocabulary=live_vocab)


# Four independent tests, not one: an assertion after a failing assertion never
# runs, so a single test could not show that each of these can fail on its own.

@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_the_corpus_is_the_measured_size(corpus_coverage):
    total, _ = corpus_coverage
    assert total == 3051, f"the corpus changed size: {total}"


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_coverage_mapped_rows_as_measured(corpus_coverage):
    assert corpus_coverage[1]["mapped"] == 3016, corpus_coverage[1]


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_coverage_unmapped_rows_as_measured(corpus_coverage):
    assert corpus_coverage[1]["unmapped"] == 35, corpus_coverage[1]


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_coverage_names_the_unmapped_values(corpus_coverage):
    assert corpus_coverage[1]["unmapped_values"] == {
        "": 29, "Policy": 3, "Service": 2, "Indirect Procurement": 1,
    }, corpus_coverage[1]["unmapped_values"]


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_the_mapping_is_case_and_spacing_insensitive(live_vocab):
    """It goes through the same fold() every alias comparison uses."""
    for spelling in ("master agreement", "MASTER AGREEMENT", "  Master   Agreement ",
                     "master-agreement", "master_agreement"):
        assert M.structure_for_contract_type(spelling) == "doctype.master_agreement", spelling


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_an_ambiguous_value_resolves_to_nothing_rather_than_picking(live_vocab):
    """Two structures claiming one spelling is a collision, not a tie to break.

    resolve_alias returns every owner; choosing the first would make the answer
    depend on row order and then record that choice as a fact.
    """
    import dataclasses

    # The unmodified vocabulary resolves 'consulting' to exactly one owner, so
    # the collision below is the only thing that can make the answer None.
    assert M.structure_for_contract_type("consulting", vocabulary=live_vocab) \
        == "doctype.consulting_agreement"
    contested = dataclasses.replace(
        live_vocab,
        alias_index={**live_vocab.alias_index,
                     "consulting": ("doctype.consulting_agreement", "doctype.sla")},
    )
    assert M.structure_for_contract_type("consulting", vocabulary=contested) is None


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_blank_and_whitespace_are_not_a_structure():
    """An absent type is not a structure, and must not become one by accident."""
    for value in (None, "", "   ", "\n", "\t  \n"):
        assert M.structure_for_contract_type(value) is None, repr(value)
