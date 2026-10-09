"""An executive summary must exist the moment the analysis is ready — the
report card said "Summary not available yet" until a 15-minute sweep ran,
and never ran at all for held documents. Session completion now stores a
deterministic summary; the LLM narrative on the sweep still owns real
confirmed deals.

v2 (2026-10-09): the 9-document upload read "Service Desk … — 3 bidders
(3 bidders, 95% confidence)", "Findings: 87 warnings." with no word on what
they were, and told the user to "clear any critical findings" when there were
none. The text is now rendered from one facts object, verdict first, in the
same Key Outcomes / Conclusion shape as the confirmed-deal summary.
"""
import re

from src.services.session_postprocess import (
    _BIDDER_SUFFIX, compose_session_summary, issue_label)


def _proposal(**over):
    p = {"proposal_id": 202, "name": "Service Desk (24x7, 2,400 users) — annual",
         "confidence": 95.0, "status": "proposed", "deal_id": None,
         "bids": 3, "pos": 0, "invoices": 0}
    p.update(over)
    return p


def _finding(issue_type, severity, count, docs, sample_ref=None):
    return {"issue_type": issue_type, "severity": severity, "count": count,
            "docs": docs, "sample_ref": sample_ref}


def _facts(**over):
    facts = {
        "documents": {"total": 9, "linked": 9, "held": 0, "duplicates": 0},
        "proposals": [_proposal()],
        "findings": [_finding("line_missing_numbers", "warning", 87, 9)],
    }
    facts.update(over)
    return facts


def _sections(text):
    lead, outcomes, conclusion = text.split("\n\n")
    assert outcomes.startswith("Key Outcomes:\n")
    assert conclusion.startswith("Conclusion:\n")
    return lead, outcomes, conclusion


# ---- the 9-document upload that prompted v2 -------------------------------------

def test_the_nine_document_upload_reads_as_a_verdict():
    lead, outcomes, conclusion = _sections(compose_session_summary(_facts()))
    assert lead == ("1 proposed deal is ready to confirm: Service Desk (24x7, 2,400 users) — "
                    "annual, with 3 competing bids. 87 data warnings, lines with no quantity "
                    "or price; none of them block confirmation.")
    assert "• Lines with no quantity or price: 87 warnings across 9 documents" in outcomes
    assert "• Documents: 9 received — 9 analysed and linked" in outcomes
    assert conclusion == ("Conclusion:\nConfirm the deal; the warnings can be triaged later "
                          "in Data Validation & Actions.")


def test_the_bidder_count_is_stated_from_one_fact_not_two():
    text = compose_session_summary(_facts())
    assert "bidders" not in text
    # once in the verdict, once in the deal's bullet; never "(3 bidders, …)" beside a
    # name that already says it
    assert text.count("3 competing bids") == 2


def test_a_name_stored_with_its_bidder_count_is_shown_without_it():
    assert _BIDDER_SUFFIX.sub("", "Service Desk (24x7, 2,400 users) — annual — 3 bidders") \
        == "Service Desk (24x7, 2,400 users) — annual"
    assert _BIDDER_SUFFIX.sub("", "Freight — 1 bidder") == "Freight"
    assert _BIDDER_SUFFIX.sub("", "Preliminaries & site establishment") \
        == "Preliminaries & site establishment"


# ---- every number in the text is a fact -------------------------------------------

_MONEY = re.compile(r"£[\d,]+\.\d{2}")


def _numbers_in(text, facts):
    for p in facts["proposals"]:
        text = text.replace(p["name"], "")
    return [int(n) for n in re.findall(r"\d+", _MONEY.sub("", text))]


def _money_in(text):
    return [float(m[1:].replace(",", "")) for m in _MONEY.findall(text)]


def _fact_numbers(facts):
    d = facts["documents"]
    f = facts["findings"]
    allowed = {d["total"], d["linked"], d["held"], d["duplicates"], len(facts["proposals"])}
    for p in facts["proposals"]:
        allowed |= {p["bids"], p["pos"], p["invoices"], round(float(p["confidence"]))}
    for x in f:
        allowed |= {x["count"], x["docs"]}
    allowed.add(sum(x["count"] for x in f if x["severity"] == "critical"))
    allowed.add(sum(x["count"] for x in f if x["severity"] == "warning"))
    allowed |= {sum(x["count"] for x in f[3:]), len(f[3:])}
    allowed |= set((facts.get("value_at_risk") or {}).get("unvalued", {}).values())
    return allowed


def test_every_number_in_a_busy_summary_equals_a_fact():
    facts = _facts(
        documents={"total": 35, "linked": 15, "held": 18, "duplicates": 2},
        proposals=[_proposal(),
                   _proposal(proposal_id=203, name="Platform licence — Enterprise (240 seats)",
                             confidence=91.4, bids=2, pos=1, invoices=4)],
        findings=[_finding("po_not_found", "critical", 3, 2, "PO-2025-0208"),
                  _finding("line_missing_numbers", "warning", 41, 7),
                  _finding("unit_price_differs_from_po", "warning", 6, 1),
                  _finding("tax_percent_mismatch", "warning", 5, 5),
                  _finding("po_line_not_billed", "info", 11, 1)])
    text = compose_session_summary(facts)
    allowed = _fact_numbers(facts)
    stray = [n for n in _numbers_in(text, facts) if n not in allowed and n != 2025 and n != 208]
    assert stray == [], text
    # and the headline figures are actually there
    for figure in ("35 received", "3 critical findings", "52 data warnings", "4 invoices",
                   "91% grouping confidence", "16 more findings of 2 other types"):
        assert figure in text, figure


def test_the_numbers_guard_catches_a_wrong_count():
    """The guard above must fail on a number that is not a fact."""
    facts = _facts()
    text = compose_session_summary(facts).replace("87 warnings", "86 warnings")
    assert 86 not in _fact_numbers(facts)
    assert 86 in _numbers_in(text, facts)


# ---- the next step follows the findings ---------------------------------------------

def test_critical_findings_block_confirmation_and_are_named():
    facts = _facts(findings=[_finding("po_not_found", "critical", 3, 3, "PO-2025-0208"),
                             _finding("line_missing_numbers", "warning", 7, 2)])
    lead, outcomes, conclusion = _sections(compose_session_summary(facts))
    assert lead.startswith("1 proposed deal is waiting for you, but 3 critical findings "
                           "should be cleared before you confirm it.")
    assert "There are also 7 data warnings." in lead
    assert "PO-2025-0208" in outcomes
    assert conclusion == ("Conclusion:\nClear the 3 critical findings (purchase orders cited "
                          "that are not in the system) in Data Validation & Actions, then "
                          "confirm the deal.")


def test_no_critical_findings_never_says_clear_critical_findings():
    text = compose_session_summary(_facts())
    assert "critical" not in text.lower()


def test_clean_session_reads_clean():
    lead, _, conclusion = _sections(compose_session_summary(_facts(findings=[])))
    assert lead.endswith("There are no open findings.")
    assert conclusion == "Conclusion:\nConfirm the deal."


def test_warnings_name_the_top_three_types_and_count_the_rest():
    facts = _facts(findings=[_finding("line_missing_numbers", "warning", 40, 9),
                             _finding("line_total_mismatch", "warning", 5, 2),
                             _finding("tax_percent_mismatch", "warning", 3, 3),
                             _finding("sum_mismatch", "warning", 1, 1)])
    lead, outcomes, _ = _sections(compose_session_summary(facts))
    assert "49 data warnings, mostly lines with no quantity or price" in lead
    assert "• Line totals that do not add up: 5 warnings across 2 documents" in outcomes
    assert "• Tax rates that do not match: 3 warnings across 3 documents" in outcomes
    assert "Totals that do not add up" not in outcomes
    assert "• Other findings: 1 more finding of 1 other type" in outcomes


def test_a_finding_type_without_a_label_still_reads():
    assert issue_label("brand_new_check") == "brand new check"
    assert issue_label("line_missing_numbers") == "lines with no quantity or price"


# ---- other states ---------------------------------------------------------------------

def test_held_documents_are_part_of_the_next_step():
    facts = _facts(documents={"total": 35, "linked": 15, "held": 20, "duplicates": 0})
    _, outcomes, conclusion = _sections(compose_session_summary(facts))
    assert "35 received — 15 analysed and linked, 20 held for data review" in outcomes
    assert conclusion.endswith(", and review the 20 documents held for data review.")


def test_a_confirmed_proposal_is_reported_as_confirmed():
    facts = _facts(proposals=[_proposal(status="confirmed", deal_id="DEALV3-202")], findings=[])
    lead, outcomes, conclusion = _sections(compose_session_summary(facts))
    assert lead.startswith("This upload's deal has been confirmed: Service Desk")
    assert "• Confirmed deal: Service Desk" in outcomes
    assert "confirm" not in conclusion.lower()


def test_critical_findings_without_a_deal_are_stated_before_the_warnings():
    facts = _facts(proposals=[], findings=[_finding("net_exceeds_gross", "critical", 1, 1),
                                           _finding("line_missing_numbers", "warning", 7, 1)])
    lead, outcomes, conclusion = _sections(compose_session_summary(facts))
    assert lead.endswith(" 1 critical finding is open, with 7 data warnings.")
    assert "also" not in lead
    assert "• Net amounts above the gross amount: 1 critical finding across 1 document" in outcomes
    assert conclusion == ("Conclusion:\nClear the 1 critical finding (net amounts above the gross "
                          "amount) in Data Validation & Actions.")


def test_no_proposals_prompts_nothing_confusing():
    lead, _, conclusion = _sections(compose_session_summary(_facts(proposals=[])))
    assert lead.startswith("No deal could be proposed")
    assert "confirm" not in conclusion.lower()


# ---- value at risk ------------------------------------------------------------------------

def _money_facts():
    return _facts(
        proposals=[],
        # critical first, then by count: the order _session_facts returns
        findings=[_finding("invoices_exceed_po_total", "critical", 1, 1),
                  _finding("line_amount_over_po", "critical", 1, 1),
                  _finding("line_missing_numbers", "warning", 4, 1),
                  _finding("uplift_above_stated", "warning", 3, 3),
                  _finding("duplicate_invoice", "warning", 2, 2)],
        value_at_risk={"billed_gbp": {"invoices_exceed_po_total": 20000.0,
                                      "duplicate_invoice": 1003.81},
                       "uplift_up_to_gbp": 52938.6,
                       "unvalued": {"duplicate_invoice": 1}})


def test_value_at_risk_is_stated_once_per_type_and_totals_to_the_facts():
    facts = _money_facts()
    lead, outcomes, _ = _sections(compose_session_summary(facts))
    assert "Value at risk: £21,003.81." in lead
    assert "adds up to £52,938.60 over the term" in lead
    assert ("• Invoices above the purchase order total: 1 critical finding across 1 document; "
            "£20,000.00 at risk") in outcomes
    # the line superseded under its PO carries no money of its own
    assert "• Lines billed above the purchase order: 1 critical finding across 1 document\n" in outcomes
    # uplift and duplicate_invoice fall outside the top 3: the duplicate's money goes on the
    # Other row (the uplift figure is a bid's, stated in the lead, never added to it)
    assert "• Other findings: 5 more findings of 2 other types; £1,003.81 at risk" in outcomes


def test_every_money_figure_in_the_text_is_a_fact():
    facts = _money_facts()
    text = compose_session_summary(facts)
    v = facts["value_at_risk"]
    allowed = set(v["billed_gbp"].values()) | {v["uplift_up_to_gbp"],
                                                round(sum(v["billed_gbp"].values()), 2)}
    assert _money_in(text) and all(m in allowed for m in _money_in(text)), _money_in(text)
    assert [n for n in _numbers_in(text, facts) if n not in _fact_numbers(facts)] == []


def test_the_uplift_figure_names_its_bullet_when_shown():
    facts = _facts(proposals=[_proposal()],
                   findings=[_finding("uplift_above_stated", "warning", 3, 3)],
                   value_at_risk={"billed_gbp": {}, "uplift_up_to_gbp": 48327.8,
                                  "unvalued": {"uplift_above_stated": 1}})
    lead, outcomes, conclusion = _sections(compose_session_summary(facts))
    assert "Value at risk" not in lead            # nothing billed: no billed total
    assert ("• Price rises above the stated uplift: 3 warnings across 3 documents; "
            "up to £48,327.80 over the term, 1 not valued") in outcomes
    assert conclusion.startswith("Conclusion:\nConfirm the deal")


def test_no_money_findings_say_nothing_about_money():
    assert "£" not in compose_session_summary(_facts())
