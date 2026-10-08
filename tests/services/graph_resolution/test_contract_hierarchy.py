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
    # ... and what that membership actually refuses, behaviourally.
    from src.services.graph_resolution.edge_writer import DerivedEdge, cypher_for

    def edge(band):
        return DerivedEdge(
            rel_type="CHILD_OF", from_label="Contract", from_key="contract_id",
            from_value="SOW-001", to_label="Contract", to_key="contract_id",
            to_value="MSA-4417", F=95.0, band=band, P_raw=0.9, L_evidence=0.0,
            profile=ch.PROFILE, profile_version=ch.VERSION, signals=None,
            observations="")

    with pytest.raises(ValueError):
        cypher_for(edge("auto_link"))
    cypher_for(edge("auto_link_with_warning"))   # the documented limit: not refused


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
    """THE rule. Reference matching exactly, plus the structure (both fixtures
    keep resolved_doc_type, so expected_structure is OK); supplier, dates and
    titles missing.

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

    Measured: reference + structure (no supplier/dates/titles) F=30.82
    (block_or_exception). Raising the weight
    to 20 gives F=85.6 (auto_link_with_warning) and the auto-band test stays
    green, so the reference would already be deciding in all but name. A
    reference with little corroborating it stays below the review band (65).
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
    r = ch.score(_sow(parent_agreement_ref="MSA-9999", _ref_resolves=True), _msa())
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


def test_distinctive_title_overlap_is_ok():
    """Positive counterpart: shared distinctive words ARE evidence."""
    r = ch.score(_sow(contract_title="SOW Data Migration"),
                 _msa(contract_title="Master Agreement Data Migration"))
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["title_overlap"] == "OK", detail


def test_shared_contract_boilerplate_is_not_title_evidence():
    r = ch.score(_sow(contract_title="Call Off Terms and Conditions"),
                 _msa(contract_title="Framework Terms and Conditions"))
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["title_overlap"] != "OK", detail


def test_observations_are_reported_for_every_signal():
    """composition.remap_clusters needs one observation set per signal id."""
    obs = ch.observations_for(_sow(), _msa())
    assert set(obs) == {s["id"] for s in ch.SIGNALS}


def _detail(r):
    return {d["id"]: d["status"] for d in r["signals"]}


def test_a_genuinely_reference_only_match_scores_lowest():
    n = dict(supplier_id=None, contract_title=None, contract_start_date=None,
             contract_end_date=None, resolved_doc_type=None)
    r = ch.score(_sow(**n), _msa(**n))
    assert r["F"] < 30.82 and r["decision"] == "block_or_exception", r


def test_a_dangling_reference_is_missing_and_still_proposable():
    child = _sow(parent_agreement_ref=None, parent_contract_id="C1543",
                 _ref_resolves=False)
    r = ch.score(child, _msa())
    assert _detail(r)["declared_reference"] == "MISSING", _detail(r)
    assert r["F"] >= 65, r


def test_a_reference_to_a_different_real_contract_conflicts():
    child = _sow(parent_agreement_ref="MSA-9999", _ref_resolves=True)
    assert _detail(ch.score(child, _msa()))["declared_reference"] == "CONFLICT"


def test_an_absent_resolves_flag_reads_a_non_match_as_missing():
    child = _sow(parent_agreement_ref="MSA-9999")
    assert _detail(ch.score(child, _msa()))["declared_reference"] == "MISSING"


def test_no_reference_is_missing():
    child = _sow(parent_agreement_ref=None)
    assert _detail(ch.score(child, _msa()))["declared_reference"] == "MISSING"


def test_a_missing_expected_structure_is_missing_not_conflict():
    r = ch.score(_sow(resolved_doc_type=None), _msa())
    assert _detail(r)["expected_structure"] == "MISSING"


def test_a_missing_supplier_is_missing_not_conflict():
    r = ch.score(_sow(supplier_id=None), _msa())
    assert _detail(r)["supplier"] == "MISSING"


def test_a_missing_title_is_missing_not_conflict():
    r = ch.score(_sow(contract_title=None), _msa())
    assert _detail(r)["title_overlap"] == "MISSING"


@pytest.mark.parametrize("ph", ["n/a", "N/A", "TBC", "tbd", "TBA", "none", "see",
                                "nil", "null", "NaN", "unknown", "Not Applicable", " "])
def test_a_placeholder_is_no_reference(ph):
    child = _sow(parent_agreement_ref=ph)
    assert _detail(ch.score(child, _msa(contract_id=ph)))["declared_reference"] == "MISSING"


def test_no_optional_data_scores_exactly_as_before_the_extension():
    """Review Focus 1: the corpus norm is a row with none of the corroborating fields."""
    with_ref = ch.score(_sow(), _msa())
    assert round(with_ref["F"], 1) == 96.9 and with_ref["decision"] == "auto_link"
    no_ref = ch.score(_sow(parent_agreement_ref=None), _msa())
    assert round(no_ref["F"], 1) == 75.6 and no_ref["decision"] == "review"
    assert with_ref["profile"] == "contract_hierarchy"
    assert [s["id"] for s in with_ref["signals"]] == [
        "declared_reference", "expected_structure", "supplier", "term_containment", "title_overlap"]


def test_a_matching_buyer_joins_the_score_and_a_conflicting_one_lowers_it():
    base = ch.score(_sow(parent_agreement_ref=None), _msa())["F"]
    same = ch.score(_sow(parent_agreement_ref=None, buyer_org_id="B-1"), _msa(buyer_org_id="B-1"))
    diff = ch.score(_sow(parent_agreement_ref=None, buyer_org_id="B-1"), _msa(buyer_org_id="B-2"))
    assert "buyer" in [s["id"] for s in same["signals"]]
    assert same["F"] > base > diff["F"]


def test_absent_corroborators_never_tax_a_pair_with_some_present():
    """Only the fields BOTH sides carry may enter the profile."""
    got = ch.score(_sow(currency="GBP", payment_terms="Net 30"), _msa(currency="GBP"))
    ids = [s["id"] for s in got["signals"]]
    assert "currency" in ids and "payment_terms" not in ids
