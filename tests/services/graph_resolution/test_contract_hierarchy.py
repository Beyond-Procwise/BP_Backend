"""Scoring a contract document against a candidate parent.

The one rule this file exists to hold: an exact reference match does NOT link.
The build spec's principle 3 says exact identifiers link automatically, and the
Discovery Report overturned it on measurement -- parent_contract_id is populated
on 1,561 contracts and resolves on 0, because the references were minted in
another namespace. An exact match after normalisation is the strongest signal
available; it is not a decision.

Offline — pure scoring, no database.
    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/graph_resolution/test_contract_hierarchy.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services import linking_engine as le                              # noqa: E402
from src.services.graph_resolution.profiles import contract_hierarchy as ch  # noqa: E402


def _sow(**over):
    row = {
        "contract_id": "SOW-001",
        "resolved_doc_type": "doctype.sow",
        "resolved_role": "role.master",
        "parent_agreement_ref": "MSA-4417",
        "framework_ref": None,
        "parent_contract_id": None,
        "supplier_id": "S-100",
        "contract_title": "Statement of Work Data Migration",
        "contract_start_date": "2026-03-01",
        "contract_end_date": "2026-09-30",
    }
    row.update(over)
    return row


def _msa(**over):
    row = {
        "contract_id": "MSA-4417",
        "resolved_doc_type": "doctype.master_agreement",
        "resolved_role": "role.master",
        "supplier_id": "S-100",
        "contract_title": "Master Services Agreement Data Migration",
        "contract_start_date": "2026-01-01",
        "contract_end_date": "2027-12-31",
    }
    row.update(over)
    return row


def test_the_profile_is_registered():
    """score_link must know the profile by name, independently of ch.score."""
    assert ch.PROFILE in le.PROFILES
    r = le.score_link(_sow(), _msa(), ch.PROFILE)
    assert "F" in r and "decision" in r and "signals" in r


def test_the_profile_is_capped_at_the_auto_link_band():
    """What UNCALIBRATED_PROFILES membership buys, stated precisely.

    edge_writer.cypher_for refuses an edge whose band == "auto_link" (F >= 92)
    for an uncalibrated profile. It does NOT refuse "auto_link_with_warning"
    (F 80-92), which passes through. So this is a cap at the top band, not a
    ban on every automatic link. Nothing in the current path writes graph edges
    for this profile anyway: Task 10 writes review-queue rows.
    """
    from src.services.graph_resolution.edge_writer import UNCALIBRATED_PROFILES
    assert ch.PROFILE in UNCALIBRATED_PROFILES
    # ... and what that membership actually refuses, verbatim from the guard.
    import inspect
    from src.services.graph_resolution import edge_writer
    assert 'edge.band == "auto_link"' in inspect.getsource(edge_writer.cypher_for), (
        "the guard no longer keys on the auto_link band; this profile's cap may have moved"
    )


def test_the_expected_parent_of_a_sow_is_a_master_agreement():
    assert ch.expected_parent_type("doctype.sow") == "doctype.master_agreement"


def test_the_expected_parent_of_a_call_off_is_a_framework():
    assert ch.expected_parent_type("doctype.call_off_contract") == "doctype.framework_agreement"


def test_a_structure_with_no_declared_parent_expects_none():
    assert ch.expected_parent_type("doctype.master_agreement") is None


def test_a_full_match_scores_in_a_band_a_person_sees():
    r = ch.score(_sow(), _msa())
    assert r["decision"] in ("review", "auto_link_with_warning", "auto_link"), r
    assert r["F"] > 0


def test_an_exact_reference_alone_does_not_reach_the_auto_band():
    """THE rule. Everything else missing, the reference matching exactly.

    1,561 existing parent pointers resolve to 0 real contracts. A reference that
    matches is evidence; on its own it must still put a person in the loop.
    """
    bare_child = _sow(supplier_id=None, contract_title=None,
                      contract_start_date=None, contract_end_date=None)
    bare_parent = _msa(supplier_id=None, contract_title=None,
                       contract_start_date=None, contract_end_date=None)
    r = ch.score(bare_child, bare_parent)
    assert r["decision"] != "auto_link", r
    assert r["F"] < 92, r


def test_an_exact_reference_alone_does_not_even_reach_review():
    """Tighter than the auto-band test above, which a weight of 20 slips past.

    Measured: reference-only F=30.82 (block_or_exception). Raising the weight
    to 20 gives F=85.6 (auto_link_with_warning) and the auto-band test stays
    green, so the reference would already be deciding in all but name. A
    reference with nothing corroborating it stays below the review band (65).
    """
    n = dict(supplier_id=None, contract_title=None,
             contract_start_date=None, contract_end_date=None)
    r = ch.score(_sow(**n), _msa(**n))
    assert r["F"] < 65, r
    assert r["decision"] not in ("auto_link", "auto_link_with_warning"), r


def test_the_wrong_structure_of_parent_conflicts():
    """A SOW's parent is a master agreement, not an invoice."""
    r = ch.score(_sow(), _msa(resolved_doc_type="doctype.invoice"))
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["expected_structure"] == "CONFLICT", detail


def test_a_different_supplier_conflicts():
    r = ch.score(_sow(), _msa(supplier_id="S-999"))
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["supplier"] == "CONFLICT", detail


def test_a_child_outside_the_parents_term_conflicts():
    r = ch.score(_sow(contract_start_date="2029-01-01", contract_end_date="2029-06-30"), _msa())
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["term_containment"] == "CONFLICT", detail


def test_a_child_inside_the_parents_term_is_ok():
    r = ch.score(_sow(), _msa())
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["term_containment"] == "OK", detail


def test_a_missing_field_is_missing_not_a_conflict():
    """MISSING and CONFLICT are different answers.

    Treating absence as contradiction would make every sparsely-filled contract
    look like a wrong parent, which is how a review queue fills with noise.
    """
    r = ch.score(_sow(contract_start_date=None, contract_end_date=None), _msa())
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["term_containment"] == "MISSING", detail


def test_the_reference_is_compared_after_normalisation():
    """'msa 4417' and 'MSA-4417' are the same reference printed differently."""
    r = ch.score(_sow(parent_agreement_ref=" msa 4417 "), _msa())
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["declared_reference"] == "OK", detail


def test_any_of_the_three_reference_fields_can_carry_the_pointer():
    for field in ("parent_agreement_ref", "framework_ref", "parent_contract_id"):
        pointers = {"parent_agreement_ref": None, "framework_ref": None,
                    "parent_contract_id": None}
        pointers[field] = "MSA-4417"
        child = _sow(**pointers)
        r = ch.score(child, _msa())
        detail = {d["id"]: d["status"] for d in r["signals"]}
        assert detail["declared_reference"] == "OK", (field, detail)


def test_a_reference_naming_a_different_contract_conflicts():
    r = ch.score(_sow(parent_agreement_ref="MSA-9999"), _msa())
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["declared_reference"] == "CONFLICT", detail


def test_a_shared_generic_word_is_not_title_evidence():
    """Every contract shares 'Agreement' and 'Services' with every other one.

    Without the stopword set this signal would score the vocabulary rather than
    the documents, and two unrelated contracts from one supplier would look like
    a parent and child.
    """
    r = ch.score(
        _sow(contract_title="Services Agreement"),
        _msa(contract_title="Services Agreement"),
    )
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["title_overlap"] != "OK", detail


def test_observations_are_reported_for_every_signal():
    """composition.remap_clusters needs one observation set per signal id."""
    obs = ch.observations_for(_sow(), _msa())
    assert set(obs) == {s["id"] for s in ch.SIGNALS}
