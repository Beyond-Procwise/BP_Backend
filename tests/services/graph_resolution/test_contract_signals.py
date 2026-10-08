# tests/services/graph_resolution/test_contract_signals.py
"""The seven corroborating comparators. Pure, offline.

    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/graph_resolution/test_contract_signals.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.graph_resolution.profiles import contract_signals as cs  # noqa: E402


def test_buyer():
    assert cs.cmp_buyer({"buyer_org_id": "B-1"}, {"buyer_org_id": " b-1 "}) == (1.0, "OK")
    assert cs.cmp_buyer({"buyer_org_id": "B-1"}, {"buyer_org_id": "B-2"}) == (0.0, "CONFLICT")
    assert cs.cmp_buyer({"buyer_org_id": "B-1"}, {}) == (0.5, "MISSING")
    assert cs.cmp_buyer({"buyer_org_id": "TBC"}, {"buyer_org_id": "B-1"}) == (0.5, "MISSING")


def test_currency():
    assert cs.cmp_currency({"currency": "GBP"}, {"currency": "gbp"}) == (1.0, "OK")
    assert cs.cmp_currency({"currency": "GBP"}, {"currency": "USD"}) == (0.0, "CONFLICT")
    assert cs.cmp_currency({"currency": "GBP"}, {}) == (0.5, "MISSING")


def test_payment_terms_differing_is_neutral_not_a_conflict():
    assert cs.cmp_payment_terms({"payment_terms": "Net 30"}, {"payment_terms": "net-30"}) == (1.0, "OK")
    assert cs.cmp_payment_terms({"payment_terms": "Net 30"}, {"payment_terms": "Net 60"}) == (0.5, "WEAK")
    assert cs.cmp_payment_terms({}, {"payment_terms": "Net 60"}) == (0.5, "MISSING")


def test_governing_law_compares_like_with_like():
    assert cs.cmp_governing_law({"governing_law": "England"}, {"governing_law": "england"}) == (1.0, "OK")
    assert cs.cmp_governing_law({"governing_law": "England"}, {"governing_law": "Delaware"}) == (0.0, "CONFLICT")
    assert cs.cmp_governing_law({"governing_law": "England"}, {}) == (0.5, "MISSING")
    # Neither side states a law: the jurisdictions are compared with each other.
    assert cs.cmp_governing_law({"jurisdiction": "UK"}, {"jurisdiction": "uk"}) == (1.0, "OK")
    assert cs.cmp_governing_law({"jurisdiction": "UK"}, {"jurisdiction": "France"}) == (0.0, "CONFLICT")
    # A law on one side and only a jurisdiction on the other are different facts.
    assert cs.cmp_governing_law({"governing_law": "England"}, {"jurisdiction": "England"}) == (0.5, "MISSING")


def test_a_jurisdiction_is_never_compared_with_a_governing_law():
    """Live 2026-10-08, OF-2026-0211 vs FA-2026-0077 on bp_testdb: the order form
    states only a jurisdiction, the framework both. Reading the child's
    jurisdiction against the parent's law called 'United Kingdom' vs 'England and
    Wales' a CONFLICT, though the two documents state the SAME jurisdiction."""
    order_form = {"governing_law": None, "jurisdiction": "United Kingdom"}
    framework = {"governing_law": "England and Wales", "jurisdiction": "United Kingdom"}
    assert cs.cmp_governing_law(order_form, framework) == (1.0, "OK")
    assert cs.cmp_governing_law(framework, order_form) == (1.0, "OK")


def test_value_rollup_is_necessary_not_sufficient():
    fits = cs.cmp_value_rollup({"total_contract_value": 50, "currency": "GBP"},
                               {"total_contract_value": 100, "currency": "GBP"})
    assert fits == (0.5, "WEAK"), "a child that fits under its parent proves nothing"
    over = cs.cmp_value_rollup({"total_contract_value": 150, "currency": "GBP"},
                               {"total_contract_value": 100, "currency": "GBP"})
    assert over == (0.0, "CONFLICT")


def test_value_rollup_never_compares_across_currencies_or_without_a_parent_value():
    usd = cs.cmp_value_rollup({"total_contract_value": 150, "currency": "USD"},
                              {"total_contract_value": 100, "currency": "GBP"})
    assert usd == (0.5, "MISSING"), "no FX rate is ever invented"
    assert cs.cmp_value_rollup({"total_contract_value": 5, "currency": "GBP"},
                               {"total_contract_value": None, "currency": "GBP"}) == (0.5, "MISSING")
    assert cs.cmp_value_rollup({"total_contract_value": 5, "currency": "GBP"},
                               {"total_contract_value": 0, "currency": "GBP"}) == (0.5, "MISSING")


def test_signatory_shared_name_is_positive_and_a_different_one_is_neutral():
    a = {"contract_signatory_name": "Jane Doe"}
    assert cs.cmp_signatory(a, {"contract_signatory_name": "jane  doe"}) == (1.0, "OK")
    assert cs.cmp_signatory(a, {"contract_signatory_name": "Sam Roe"}) == (0.5, "WEAK")
    assert cs.cmp_signatory(a, {}) == (0.5, "MISSING")
    b = {"buyer_signatory_name": "Ann Lee"}
    assert cs.cmp_signatory(b, {"buyer_signatory_name": "ann lee"}) == (1.0, "OK")


def test_a_signatory_is_compared_only_with_the_same_party_on_the_other_document():
    """The supplier's signatory on one document and the BUYER's on the other are two
    different people's roles; a matching name across them is not corroboration."""
    supplier_side = {"contract_signatory_name": "Jane Doe"}
    buyer_side = {"buyer_signatory_name": "Jane Doe"}
    assert cs.cmp_signatory(supplier_side, buyer_side) == (0.5, "MISSING")
    # One party matches, the other differs: still a shared signatory.
    child = {"contract_signatory_name": "Jane Doe", "buyer_signatory_name": "Ann Lee"}
    parent = {"contract_signatory_name": "Jane Doe", "buyer_signatory_name": "Bob Ray"}
    assert cs.cmp_signatory(child, parent) == (1.0, "OK")
    # Both parties comparable, neither matches: neutral, never a conflict.
    parent2 = {"contract_signatory_name": "Sam Roe", "buyer_signatory_name": "Bob Ray"}
    assert cs.cmp_signatory(child, parent2) == (0.5, "WEAK")


def test_cost_centre_any_shared_field_is_positive():
    a = {"cost_centre_id": "CC1", "spend_category": "IT"}
    assert cs.cmp_cost_centre(a, {"cost_centre_id": "CC9", "spend_category": "it"}) == (1.0, "OK")
    assert cs.cmp_cost_centre(a, {"cost_centre_id": "CC9"}) == (0.5, "WEAK")
    assert cs.cmp_cost_centre(a, {"business_unit_id": "BU"}) == (0.5, "MISSING")


def test_every_spec_row_has_the_engine_keys_and_a_registered_kind():
    from src.services import linking_engine as le
    keys = {"id", "cluster", "tier", "weight", "appl", "cap", "kind", "reads"}
    for spec in (cs.BUYER, cs.VALUE_ROLLUP, cs.CURRENCY, cs.PAYMENT_TERMS,
                 cs.GOVERNING_LAW, cs.SIGNATORY, cs.COST_CENTRE):
        assert keys <= set(spec), spec["id"]
        assert spec["kind"] in le._EXTRA_SIGNALS, spec["id"]


def test_the_amendment_list_omits_what_an_amendment_legitimately_changes():
    ids = {s["id"] for s in cs.AMENDMENT_OPTIONAL}
    assert ids == {"buyer", "currency", "governing_law", "signatory"}
    assert "payment_terms" not in ids and "value_rollup" not in ids
