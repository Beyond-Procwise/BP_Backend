"""A price read from an email is a CLAIM until a person confirms it; the assurance layer says so instead of presenting it as a Postgres fact.

The supplier's offer lives in proc.supplier_response.price, written by software that takes the first number in the email (or whatever a
model returns). The row now records where it came from; a fact source can name that column, and the fact is a claim unless the column
says a person confirmed it. A row whose origin was never recorded (every row that exists today) is also a claim, and so is a database
without the column at all: not knowing is not the same as confirmed.
"""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from src.services import draft_assurance as da
from src.services.draft_assurance import capture, facts as F
from tests.services.test_draft_assurance import FakeConn, _family_rules

OFFER_ROW = {"id": 2, "workflow_id": "wf-1", "supplier_id": "S-1", "round_number": 2, "price": Decimal("47.50"), "currency": "GBP",
             "lead_time": "14 days", "rfq_id": "RFQ-1"}
KEYS = {"workflow_id": "wf-1", "supplier_id": "S-1"}


def src(**over):
    rules = _family_rules()
    rules["fact_sources"]["supplier_current_offer"].update({"claim_column": "extraction_status", "claim_unless": ["confirmed"], **over})
    return da.parse_family(rules).facts["supplier_current_offer"]


def resolve(status="__absent__", **srcover):
    row = dict(OFFER_ROW)
    if status != "__absent__":
        row["extraction_status"] = status
    fact = F.FactResolver(FakeConn({"supplier_response": [row]})).resolve(src(**srcover), KEYS)
    assert isinstance(fact, F.ResolvedFact), fact
    return fact


# --- the resolver ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["extracted_unverified", "rejected", None, "__absent__", "anything-else"])
def test_a_price_is_a_claim_unless_a_person_confirmed_it(status):
    fact = resolve(status)
    assert fact.claim is True and fact.value == Decimal("47.50")                     # still the value, still resolved: just not vouched for
    assert fact.provenance()["claim"] is True


def test_a_confirmed_price_is_a_plain_fact():
    fact = resolve("confirmed")
    assert fact.claim is False and "claim" not in fact.provenance()


def test_the_origin_is_reported_as_what_the_row_says_or_that_it_was_never_recorded():
    assert resolve("extracted_unverified").provenance()["origin"] == "extracted_unverified"
    assert resolve(None).provenance()["origin"] == "not recorded"
    assert resolve("__absent__").provenance()["origin"] == "not recorded"


def test_a_source_that_names_no_provenance_column_is_never_a_claim():
    rules = _family_rules()
    for k in ("claim_column", "claim_unless"):
        rules["fact_sources"]["supplier_current_offer"].pop(k, None)               # the shipped row carries them; this source must not
    plain = da.parse_family(rules).facts["supplier_current_offer"]
    assert plain.claim_column is None
    fact = F.FactResolver(FakeConn({"supplier_response": [dict(OFFER_ROW)]})).resolve(plain, KEYS)
    assert fact.claim is False


def test_a_failure_reading_the_provenance_column_makes_a_claim_and_not_an_unresolved_fact():
    class Broken(FakeConn):
        def cursor(self):
            cur = super().cursor()
            real = cur.execute

            def execute(sql, params=None):
                if "extraction_status" in sql:
                    raise RuntimeError('column "extraction_status" does not exist')
                return real(sql, params)
            cur.execute = execute
            return cur
    fact = F.FactResolver(Broken({"supplier_response": [dict(OFFER_ROW)]})).resolve(src(), KEYS)
    assert isinstance(fact, F.ResolvedFact) and fact.claim is True and fact.provenance()["origin"] == "not recorded"


# --- the config ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [{"claim_column": "x; DROP TABLE y"}, {"claim_column": "a b"}, {"claim_unless": "confirmed"}, {"claim_unless": [1]},
                                 {"claim_column": None, "claim_unless": ["confirmed"]}, {"claim_unless": []}])
def test_a_malformed_claim_setting_is_refused_at_parse_time(bad):
    rules = _family_rules()
    rules["fact_sources"]["supplier_current_offer"].update({"claim_column": "extraction_status", "claim_unless": ["confirmed"]})
    rules["fact_sources"]["supplier_current_offer"].update(bad)
    with pytest.raises(da.FamilyConfigUnavailable):
        da.parse_family(rules)


def test_the_shipped_families_mark_the_offer_and_the_lead_time_as_claims_unless_confirmed():
    sql = Path(__file__).resolve().parents[2] / "deploy/sql"
    for name in ("2026-10-07_email_family_negotiation_counter.sql", "2026-10-08_email_family_human_written.sql"):
        fam = da.parse_family(json.loads((sql / name).read_text().split("$json$")[1])["rules"])
        for key in ("supplier_current_offer", "supplier_lead_time"):
            if key in fam.facts:
                assert (fam.facts[key].claim_column, fam.facts[key].claim_unless) == ("extraction_status", ["confirmed"]), (name, key)
        assert "supplier_current_offer" in fam.facts


# --- the assurance record ---------------------------------------------------------------------------------------------------

from src.services.draft_assurance import assure as A
from tests.services.test_draft_assurance import TABLES

GOOD = ("Thank you for your latest offer of 47.50 GBP. We would like to propose 44.80 GBP. "
        "Could you please confirm by 30 October 2026?")


def family_with_claims():
    rules = _family_rules()
    rules["fact_sources"]["supplier_current_offer"]["label"] = "Supplier's latest offer"      # the labels arrive with the family_v2 migration
    return da.parse_family(rules)                        # the shipped counter row already carries the claim settings


def tables(status):
    rows = [dict(r) for r in TABLES["supplier_response"]]
    for r in rows:
        r["extraction_status"] = status
    return {**TABLES, "supplier_response": rows}


def finalized(status, family=None):
    fam = family or family_with_claims()
    data = {"current_offer": 47.5, "currency": "GBP", "counter_price": 44.8, "response_deadline": "30 October 2026",
            "reasoned_basis": {"response_deadline": ["email_thread_summary"]}}
    inp = da.prepare_inputs(FakeConn(tables(status)), fam, data, lookup_keys=KEYS)
    return inp, inp.finalize(GOOD, ["a@x.test"], ["a@x.test"])


def test_a_claimed_offer_is_listed_as_a_claim_and_the_draft_needs_a_look():
    inp, rec = finalized("extracted_unverified")
    assert "supplier_current_offer" in [c["fact"] for c in rec["claims"]]
    offer = next(c for c in rec["claims"] if c["fact"] == "supplier_current_offer")
    assert offer["origin"] == "extracted_unverified" and offer["label"] == "Supplier's latest offer"
    assert rec["status"] == "needs_review"
    assert rec["facts"]["supplier_current_offer"]["claim"] is True            # carried inside the fact's own provenance too


def test_each_claim_becomes_an_item_a_person_must_confirm_before_the_draft_is_ready():
    _, rec = finalized("extracted_unverified")
    item = next(i for i in rec["assumption_items"] if i["id"] == "claim:supplier_current_offer")
    assert "email" in item["text"] and "not been confirmed" in item["text"] and item["resolution"] is None
    assert rec["ready"] is False and "claim:supplier_current_offer" in [i["id"] for i in rec["assumption_items"] if not i.get("resolution")]


def test_a_confirmed_offer_raises_no_claim_and_leaves_the_draft_verified():
    _, rec = finalized("confirmed")
    assert rec["claims"] == [] and rec["status"] == "verified" and rec["ready"] is True
    assert not [i for i in rec["assumption_items"] if str(i["id"]).startswith("claim:")]


def test_a_row_whose_origin_was_never_recorded_is_reported_as_such():
    _, rec = finalized(None)
    offer = next(c for c in rec["claims"] if c["fact"] == "supplier_current_offer")
    assert offer["origin"] == "not recorded" and "never recorded" in offer["reason"].lower()


def test_a_claim_does_not_change_which_figures_the_checks_allow():
    """The value is still the row's value: a draft quoting it passes the figure checks. The claim is about trust, not about the number."""
    _, rec = finalized("extracted_unverified")
    assert not [v for v in rec["violations"] if v["severity"] == "fail"]


def test_the_reviewer_view_says_in_words_that_the_offer_came_from_an_email_and_is_unconfirmed():
    _, rec = finalized("extracted_unverified")
    raw = {"unique_id": "U-1", "assurance_status": rec["status"], "ready": rec["ready"], "family_id": "negotiation_counter",
           "facts": rec["facts"], "assumption_items": rec["assumption_items"], "unverified_figures": [], "violations": [], "conflicts": []}
    view = capture.to_view(raw)
    offer = view["facts"]["supplier_current_offer"]
    assert offer["claim"] is True and "email" in offer["origin_label"].lower() and "not confirmed" in offer["origin_label"].lower()
    blob = json.dumps(view)
    assert "supplier_response" not in blob and "extraction_status" not in blob               # no internal name reaches the screen
