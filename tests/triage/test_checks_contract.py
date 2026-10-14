"""What a contract allows, measured against what was invoiced.

The comparison used to live only in the gateway's Document match view, which draws a
screen and writes nothing. These tests pin it in the backend as checks that produce
Findings, so a contract breach is auditable and can carry a recorded outcome. Ruling of
2026-09-27: one authority, and it is this one.
"""
from __future__ import annotations

from decimal import Decimal as D

from src.services.triage.model import Outcome, Severity
from src.services.triage.writer import MIRROR_ISSUE_TYPE
from tests.triage.helpers import contract, deal, inv, line, make_cfg, pipeline, po, scored, term


def _of(results, rule):
    return [r for r in results if r.rule_id == rule]


# --- price caps -------------------------------------------------------------------

def test_charging_above_a_contract_cap_is_a_finding():
    ds = deal(contract(lines=[term(1, "cap", "10.00")]),
              po(lines=[line(1, price="12.00")]),
              inv(lines=[line(1, price="12.00")]))
    _, results = scored(ds)
    caps = _of(results, "contract_cap")
    assert len(caps) == 1
    r = caps[0]
    assert r.outcome == Outcome.CONFLICT
    assert r.auth_doc == "C-1"
    assert r.auth_value == "10.00"
    assert r.claim_value == "12.00"
    # 10 units, £2 over the cap each.
    assert r.exposure == D("20.00")
    assert r.field_class == "money"


def test_charging_at_the_cap_is_not_a_finding():
    ds = deal(contract(lines=[term(1, "cap", "12.00")]),
              po(lines=[line(1, price="12.00")]),
              inv(lines=[line(1, price="12.00")]))
    _, results = scored(ds)
    assert _of(results, "contract_cap") == []


def test_charging_below_the_cap_is_not_a_finding():
    """A cap is a ceiling, not a price: under it is exactly what it is for."""
    ds = deal(contract(lines=[term(1, "cap", "20.00")]),
              po(lines=[line(1, price="12.00")]),
              inv(lines=[line(1, price="12.00")]))
    _, results = scored(ds)
    assert _of(results, "contract_cap") == []


def test_a_lump_sum_line_is_capped_on_its_amount():
    """No quantity to multiply, so the cap applies to the line as a whole."""
    ds = deal(contract(lines=[term(1, "cap", "500.00", desc="Implementation")]),
              po(lines=[line(1, qty=None, price=None, amount="900.00", desc="Implementation")]),
              inv(lines=[line(1, qty=None, price=None, amount="900.00", desc="Implementation")]))
    _, results = scored(ds)
    caps = _of(results, "contract_cap")
    assert len(caps) == 1
    assert caps[0].exposure == D("400.00")


# --- rate cards -------------------------------------------------------------------

def test_charging_above_a_contract_rate_is_a_finding():
    ds = deal(contract(lines=[term(1, "rate", "10.00")]),
              po(lines=[line(1, price="10.00")]),
              inv(lines=[line(1, price="11.50")]))
    _, results = scored(ds)
    rates = _of(results, "contract_rate")
    assert len(rates) == 1
    assert rates[0].outcome == Outcome.CONFLICT
    assert rates[0].delta == D("1.50")
    assert rates[0].exposure == D("15.00")


def test_charging_below_a_contract_rate_is_a_finding():
    """Under-charging is still a departure from the agreed rate; it is just not money lost."""
    ds = deal(contract(lines=[term(1, "rate", "10.00")]),
              po(lines=[line(1, price="10.00")]),
              inv(lines=[line(1, price="7.00")]))
    _, results = scored(ds)
    rates = _of(results, "contract_rate")
    assert len(rates) == 1
    assert rates[0].delta == D("-3.00")


def test_charging_the_contract_rate_is_not_a_finding():
    ds = deal(contract(lines=[term(1, "rate", "12.00")]),
              po(lines=[line(1, price="12.00")]),
              inv(lines=[line(1, price="12.00")]))
    _, results = scored(ds)
    assert _of(results, "contract_rate") == []


# --- included items ---------------------------------------------------------------

def test_charging_for_an_included_item_is_a_finding():
    ds = deal(contract(lines=[term(1, "included", "0", desc="Freight", item="FREIGHT")]),
              po(lines=[line(1, item="FREIGHT", qty=None, price=None, amount="75.00",
                             desc="Freight")]),
              inv(lines=[line(1, item="FREIGHT", qty=None, price=None, amount="75.00",
                              desc="Freight")]))
    _, results = scored(ds)
    inc = _of(results, "contract_included")
    assert len(inc) == 1
    assert inc[0].outcome == Outcome.CONFLICT
    assert inc[0].exposure == D("75.00")
    assert inc[0].auth_value == "0"


def test_an_included_item_at_no_charge_is_not_a_finding():
    ds = deal(contract(lines=[term(1, "included", "0", desc="Freight", item="FREIGHT")]),
              po(lines=[line(1, item="FREIGHT", qty=None, price=None, amount="0",
                             desc="Freight")]),
              inv(lines=[line(1, item="FREIGHT", qty=None, price=None, amount="0",
                              desc="Freight")]))
    _, results = scored(ds)
    assert _of(results, "contract_included") == []


# --- what must NOT produce a finding ----------------------------------------------

def test_a_line_with_no_contract_term_produces_nothing():
    ds = deal(contract(lines=[term(1, "rate", "10.00", desc="Consultancy", item="CONS")]),
              po(lines=[line(1, item="ITEM-1", price="12.00", desc="Widget")]),
              inv(lines=[line(1, item="ITEM-1", price="12.00", desc="Widget")]))
    _, results = scored(ds)
    assert [r for r in results if r.rule_id.startswith("contract_")] == []


def test_an_unclassified_contract_term_is_never_linked():
    """classify_term_basis leaves a row None rather than guessing; None must not judge.

    This asserts on the LINK, not on the absence of a Result. Asserting "no finding
    appeared" passed even with the guard deleted, because an unknown basis also falls
    through the check's cap/rate branches -- so it proved nothing. The guard is in
    link.term_for, so that is where it has to be measured.
    """
    ds = deal(contract(lines=[term(1, None, "10.00")]),
              po(lines=[line(1, price="99.00")]),
              inv(lines=[line(1, price="99.00")]))
    links, results = scored(ds)
    assert links.term_links == [], "an unreadable term must govern nothing"
    assert [r for r in results if r.rule_id.startswith("contract_")] == []


def test_an_unknown_term_basis_is_refused_rather_than_guessed():
    """Belt and braces: even if such a term were linked, the check must not judge it.

    A future basis word added to the database CHECK but not to the check here would
    otherwise be silently ignored; this pins that it stays ignored deliberately.
    """
    from src.services.triage.checks import check_contract_terms
    from src.services.triage.model import Links, TermLink
    ds = deal(contract(lines=[term(1, "rate", "10.00")]),
              po(lines=[line(1, price="99.00")]), inv(lines=[line(1, price="99.00")]))
    forged = term(1, "some_new_basis", "10.00")
    links = Links(term_links=[TermLink(ds.invoices[0], ds.invoices[0].lines[0],
                                       ds.contracts[0], forged, 1.0)])
    assert check_contract_terms(ds, links, make_cfg()) == []


def test_a_deal_with_no_contract_produces_nothing():
    ds = deal(po(lines=[line(1, price="12.00")]), inv(lines=[line(1, price="12.00")]))
    _, results = scored(ds)
    assert [r for r in results if r.rule_id.startswith("contract_")] == []


# --- money is counted once --------------------------------------------------------

def test_the_contract_finding_absorbs_the_po_price_finding():
    """A line over both its PO and its contract is ONE sum of money, not two.

    The contract is the higher authority, so it is the cause and the PO difference
    becomes its effect -- Finding.exposure sums causes only.
    """
    ds = deal(contract(lines=[term(1, "cap", "10.00")]),
              po(lines=[line(1, price="10.50")]),
              inv(lines=[line(1, price="12.00")]))
    out = pipeline(ds)
    cap = [f for f in out.findings if f.rule_id == "contract_cap"]
    assert len(cap) == 1, "the cap breach must be its own finding"
    assert [f for f in out.findings if f.rule_id == "unit_price"] == [], \
        "the PO price difference on the same line must not be a second finding"
    assert any(e.rule_id == "unit_price" for e in cap[0].effects), \
        "it must be carried as an effect, so the reader still sees it"
    assert cap[0].exposure == D("20.00")


def test_a_price_finding_on_another_line_is_untouched():
    ds = deal(contract(lines=[term(1, "cap", "10.00", item="ITEM-1", desc="Widget")]),
              po(lines=[line(1, item="ITEM-1", price="10.00"),
                        line(2, item="ITEM-2", price="30.00", desc="Gadget")]),
              inv(lines=[line(1, item="ITEM-1", price="12.00"),
                         line(2, item="ITEM-2", price="40.00", desc="Gadget")]))
    out = pipeline(ds)
    assert [f for f in out.findings if f.rule_id == "contract_cap"]
    assert [f for f in out.findings if f.rule_id == "unit_price"], \
        "line 2 has no contract term, so its PO difference stands on its own"


# --- it has to reach a person -----------------------------------------------------

def test_every_contract_rule_reaches_the_action_centre():
    for rule in ("contract_cap", "contract_rate", "contract_included"):
        assert rule in MIRROR_ISSUE_TYPE, f"{rule} would be written but never mirrored"


def test_a_cap_breach_is_severe_enough_to_be_shown():
    ds = deal(contract(lines=[term(1, "cap", "10.00")]),
              po(lines=[line(1, price="10.00")]),
              inv(lines=[line(1, price="400.00")]))
    out = pipeline(ds)
    cap = next(f for f in out.findings if f.rule_id == "contract_cap")
    assert cap.severity >= Severity.S2
    assert cap.headline
    assert "contract_cap" not in cap.headline, "the headline must be words, not a rule id"


def test_the_finding_names_the_contract_in_its_text():
    ds = deal(contract(lines=[term(1, "cap", "10.00")]),
              po(lines=[line(1, price="10.00")]),
              inv(lines=[line(1, price="12.00")]))
    out = pipeline(ds)
    cap = next(f for f in out.findings if f.rule_id == "contract_cap")
    assert "C-1" in cap.text


def test_the_category_is_not_other():
    ds = deal(contract(lines=[term(1, "cap", "10.00")]),
              po(lines=[line(1, price="10.00")]),
              inv(lines=[line(1, price="12.00")]))
    out = pipeline(ds)
    cap = next(f for f in out.findings if f.rule_id == "contract_cap")
    assert cap.category == "contract"
