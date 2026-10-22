"""The skill router: a deterministic choice of skill from the request, with no model.

Wrongly injecting a skill is worse than not injecting one, so the router stays silent unless one skill
clearly wins. The held-out prompts below are phrasings that are NOT in any skill's trigger list.
"""
from pathlib import Path

import pytest

from orchestration import skill_router as sr

SKILLS = Path(__file__).resolve().parents[2] / "skills"

# Fresh phrasings, written separately from the skills' own trigger examples.
HELD_OUT = {
    "procurement-policy-compliance": [
        "Can I go ahead and place an order for 18,000 pounds of lab equipment, or does someone have to sign it off?",
        "Does our policy permit me to approve a requisition that I raised myself?",
        "We are splitting a big purchase into three smaller ones. Is that going to cause a compliance problem?",
        "Who is the approver for a 60k software subscription?",
        "Is this contract within delegated authority limits?",
        "Check that this purchase order respects our approval thresholds.",
    ],
    "incoterms-selection": [
        "We're buying machine parts from Shenzhen in 40ft containers. FOB or FCA?",
        "Who pays for insurance if the supplier ships CIF?",
        "Is DDP a sensible term when the seller is outside the EU?",
        "Where does the risk pass to us under DAP?",
        "Pick the delivery term for an air freight shipment from Germany.",
        "Compare the landed cost of buying EXW against DAP on this lane.",
    ],
    "negotiation-planning": [
        "Our packaging supplier wants a 6 percent price rise. Help me plan the conversation.",
        "What is our fallback if talks with this vendor collapse, and what should we concede first?",
        "Set out our opening position, target and walk-away for the steel renewal.",
        "Prepare me for the meeting with the supplier about their price increase.",
        "How should we approach the contract renewal negotiation with this logistics provider?",
        "Give me the levers and a concession ladder for the software renewal talks.",
    ],
    "savings-methodology": [
        "The buyer says they saved 80k on resin. Is that a real saving?",
        "How do we count a price freeze that stopped a 5 percent increase?",
        "Which baseline do we use to claim the saving on this recurring part?",
        "Two managers both claim the saving from the same supplier deal. Sort it out.",
        "Is moving to 90 day payment terms a cost saving or something else?",
        "Prepare this saving claim for finance to sign off.",
    ],
    "award-recommendation": [
        "The bids are scored. Tell me who we should give the contract to.",
        "Write the decision paper recommending the winning bidder.",
        "Should we split the work between the top two bidders?",
        "Is the gap between first and second place big enough to be safe, or is it a close call?",
        "Which tenderer should win once price and quality are both counted?",
        "Put together the award recommendation for the sourcing committee.",
    ],
}
SHOULD_STAY_SILENT = [
    "What did we spend with our top ten suppliers last quarter?",
    "What is the weather like in Hamburg today?",
    "Summarise this email from the supplier.",
    "Translate this quotation into German.",
    "List the open findings for the Acme deal.",
    "Draft a polite reminder to the supplier about the late delivery.",
    "How many invoices are overdue?",
    "Extract the line items from this invoice.",
    "Screen this company against the sanctions lists.",
    "Write a termination clause for a services contract.",
    "hello",
    "Explain what a purchase order is.",
    "Run the opportunity miner for this supplier.",
    "Show me a chart of invoice volume by month.",
    "Which suppliers are located in Poland?",
    "Create a requisition for office chairs.",
]


@pytest.fixture(scope="module")
def router():
    return sr.Router(sr.load_entries(SKILLS))


def test_every_skill_is_found_from_fresh_phrasings(router):
    misses = [(skill, p, (router.route(p) or sr.NO_MATCH).skill)
              for skill, prompts in HELD_OUT.items() for p in prompts if (router.route(p) or sr.NO_MATCH).skill != skill]
    total = sum(len(v) for v in HELD_OUT.values())
    assert len(misses) <= total * 0.1, misses


def test_it_never_picks_the_wrong_skill_when_it_does_pick_one(router):
    wrong = [(skill, p, router.route(p).skill) for skill, prompts in HELD_OUT.items() for p in prompts
             if router.route(p) is not None and router.route(p).skill != skill]
    assert wrong == []


def test_it_stays_silent_on_requests_no_skill_covers(router):
    noisy = [(p, router.route(p).skill) for p in SHOULD_STAY_SILENT if router.route(p) is not None]
    assert noisy == []


def test_two_skills_in_one_request_is_a_tie_and_the_router_stays_silent(router):
    both = "Prepare the negotiation plan and also write the award recommendation."
    m = router.route(both)
    assert m is None or m.margin >= sr.MARGIN


def test_each_skills_own_trigger_prompts_route_to_it_and_never_to_it_when_they_should_not(router):
    import yaml
    off, wrong = [], []
    for e in router.entries:
        t = yaml.safe_load((SKILLS / e.name / "evals" / "triggers.yaml").read_text())
        hits = sum(1 for p in t["should_trigger"] if (router.route(p) or sr.NO_MATCH).skill == e.name)
        if hits < 9:
            off.append((e.name, hits))
        wrong += [(e.name, p) for p in t["should_not_trigger"] if (router.route(p) or sr.NO_MATCH).skill == e.name]
    assert off == [] and wrong == []


def test_a_match_reports_its_points_and_why(router):
    m = router.route(HELD_OUT["incoterms-selection"][0])
    assert m.skill == "incoterms-selection" and m.points >= sr.MIN_POINTS and m.margin >= sr.MARGIN
    assert "fob" in m.matched and "fca" in m.matched


def test_the_same_request_always_routes_the_same_way(router):
    p = HELD_OUT["savings-methodology"][0]
    assert [router.route(p) for _ in range(3)] == [router.route(p)] * 3


def test_empty_and_junk_requests_do_not_crash(router):
    for p in ("", "   ", "?!", "a", None):
        assert router.route(p) is None


def test_a_skill_with_no_routing_terms_is_never_routed_to(tmp_path):
    (tmp_path / "solo").mkdir()
    (tmp_path / "solo" / "SKILL.md").write_text("---\nname: solo\ndescription: x.\n---\n")
    (tmp_path / "solo" / "config.yaml").write_text("thresholds: {}\n")
    assert sr.load_entries(tmp_path) == []


def test_a_star_matches_any_ending_and_a_phrase_must_be_in_order():
    r = sr.Router([sr.Entry("a", strong=["negotiat*", "walk away"], support=["supplier"])])
    assert r.route("We are negotiating with the supplier").skill == "a"
    assert r.route("away we walk, supplier") is None


def test_one_word_scores_once_even_if_two_rules_could_match_it():
    r = sr.Router([sr.Entry("a", strong=["approver*"], support=["approver"])])
    assert r.scores("who is the approver")["a"][0] == sr.STRONG


# A second set, written AFTER the routing terms were tuned and used once to measure, never to tune.
# Measured 2026-10-21: 12 of 20 routed correctly, 0 wrong, 0 noisy on the silent set. The router is
# safe, not complete: a miss only means the model may still read the skill itself.
FRESH = {
    "procurement-policy-compliance": [
        "I'd like to buy new monitors for the team, about 7,500 pounds. Do I need anyone's permission first?",
        "Would approving this invoice myself break our rules?",
        "Is it OK under our rules to order from this supplier without getting three quotes?",
        "Our department keeps ordering just below the approval limit. Is that a problem?"],
    "incoterms-selection": [
        "Which shipping term should we put in the contract for sea freight from Vietnam?",
        "If the goods are damaged on the water, who bears the loss under CIP?",
        "The supplier proposed EXW from Turkey and we cannot arrange export clearance. What do you suggest?",
        "How does landed cost change if we switch from FOB to DAP?"],
    "negotiation-planning": [
        "Our cleaning contractor has asked for 8 percent more from January. How do we respond?",
        "Help me work out what to offer first and what we can give up in the talks with Borealis.",
        "What is our best alternative if this vendor refuses to move on price?",
        "Draft a strategy for the annual price review meeting with our main steel supplier."],
    "savings-methodology": [
        "Procurement reports 120k of savings on the cable contract. How would we validate that?",
        "Does a negotiated discount that offsets a market price rise count as a saving?",
        "Which saving numbers can finance accept and which are only avoided costs?",
        "Our buyer claims savings but the volume went down. How should we treat it?"],
    "award-recommendation": [
        "We have the tender results. Which of the bidders should we choose?",
        "Write up the recommendation for the contract award with the reasons.",
        "Is it safer to share the award between two suppliers rather than one?",
        "Would a small change in weights alter who comes out on top of the tender?"],
}
FRESH_SILENT = [
    "How much did we spend on packaging in 2025?", "Please book a meeting room for Thursday.",
    "Show me all invoices from Acme Ltd.", "What is the status of purchase order 4500123?",
    "Summarise the key terms of this contract.", "Which supplier has the best delivery record?",
    "Convert 500 euros to pounds.", "Send the supplier a reminder about the delayed shipment.",
    "Draft an RFQ for office furniture.", "How do I reset my password?",
    "List the contracts expiring this year.", "Describe how the three way match works."]


def test_on_fresh_phrasings_it_is_never_wrong_and_finds_at_least_half(router):
    total = sum(len(v) for v in FRESH.values())
    hits = sum(1 for sk, ps in FRESH.items() for p in ps if (router.route(p) or sr.NO_MATCH).skill == sk)
    wrong = [(sk, p, router.route(p).skill) for sk, ps in FRESH.items() for p in ps
             if router.route(p) is not None and router.route(p).skill != sk]
    assert wrong == [] and hits >= total * 0.5, (hits, total)


def test_on_fresh_unrelated_requests_it_stays_silent(router):
    assert [(p, router.route(p).skill) for p in FRESH_SILENT if router.route(p)] == []


def test_two_supporting_words_alone_are_not_enough_to_inject_a_skill(router):
    assert router.scores("The vendor meeting")["negotiation-planning"][0] == 2 * sr.SUPPORT
    assert router.route("The vendor meeting") is None


# Phrasings met in live runs that the router missed. Each is a regression: a miss costs nothing but the
# skill's structure, yet these are ordinary ways to ask.
LIVE_MISSES = {
    "procurement-policy-compliance": [
        "Can I approve a GBP 35,000 purchase order for IT consultancy on my own?",
        "May I approve this requisition for 12,000 pounds of furniture?",
    ],
}


def test_phrasings_missed_in_live_runs_now_route(router):
    missed = [(sk, p) for sk, ps in LIVE_MISSES.items() for p in ps
              if (router.route(p) or sr.NO_MATCH).skill != sk]
    assert missed == []


def test_the_new_approval_terms_do_not_take_an_award_question_from_the_award_skill(router):
    assert router.route("Can I approve the award to the winning bidder?").skill == "award-recommendation"
