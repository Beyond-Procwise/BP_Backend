---
name: negotiation-planning
description: Prepare a negotiation plan with a supplier - target, most likely and walk-away positions, BATNA, ZOPA, levers, a concession ladder, anticipated objections and an agenda. Use before a contract award, renewal, supplier price-increase request or claim, or when someone asks "how should we negotiate this?" Review-gated, so a named category lead approves targets and walk-away before use. Not for e-auction design, contract wording, or sending anything to the supplier.
---

# Negotiation Planning

## Purpose

Turn the evidence the platform holds into a plan a category manager can take into the room. Every price in the plan is a tool result, a user-supplied figure or an assumption marked as one.

## When to use and when not

Use before an award, a renewal, a supplier's price-increase request, or a dispute.

Not for auction design, contract wording, or anything that sends a message: this skill plans, it does not contact the supplier.

## Inputs and preconditions

Ask before planning; do not proceed on guessed spend.

- Negotiation type: new award, renewal, supplier price-increase request, or claim.
- Value at stake: annual spend and term.
- Deadline and who decides.
- Our target price and walk-away if the user already has them.

Use `tool:savings_tracker` to see what has already been counted as saved; the expected saving in this plan goes in the plan and is logged later by a person.
Use `tool:commodity_index_lookup` with the series code (or words) and the window from the last price date to today for the index move between the recorded points; quote the series, publisher and dates. If no series is loaded, index history is not available: ask the user for the index move and label it user-supplied; without either, the justified part of an increase request cannot be separated and the plan says so.
Use `tool:financial_health_score` for the financial health assessment compliance recorded (provider, rating, score, date). If none is recorded, say the supplier's financial position is not assessed; do not infer it.
Use `tool:supplier_lookup` for the supplier's recorded profile (lead time, Incoterms, currency, insurance) and `tool:get_corpus_facts` for its history; say what was missing.
NOT AVAILABLE: build_docx. Give the plan as text.

## Workflow

1. Frame the negotiation and value at stake. Ask for anything missing.
2. Fact base: `tool:get_corpus_facts` for the supplier's spend, history, open findings and quotes. `tool:list_opportunity_levers` and `tool:get_opportunity_bucket` for known savings levers on this supplier. If there are competing quotes, put them on the same terms with `bids.normalise` through `tool:run_calculation` first, using only exchange rates, freight figures and a cost of capital the user or finance gave, and compare only what it reports as comparable.
3. Costed floor, for a manufactured part only: ask the user for the bill of materials (net weight, price per kg, scrap rate), the routing (cycle time, cavities, oee, machine rate, labour rate) and the overhead, SG&A and margin percentages. Never supply a rate yourself. With those, run `cost.should_cost` through `tool:run_calculation`; its result is a modelled estimate, never the supplier's actual cost, and every rate in it is the user's. If the user cannot give them, there is no costed floor: say so and label the target an estimate.
4. Position: `tool:list_calculations` for the `negotiation` family, then `tool:run_calculation` for `negotiation.zopa_estimate` (our ceiling, the supplier's floor if evidence exists, an entry counter; pass the `cost.should_cost` unit cost as its `should_cost` input when you have one), `negotiation.batna_strength` (alternative quotes and history), `negotiation.kraljic_quadrant` (suggested style) and `negotiation.outlier_rails` (whether an offer breaches a review or escalation rail). Use only inputs the tools or the user supplied. If a result says it could not be assessed, say so and show what is missing.
5. Governed bounds: `tool:get_policy` for `negotiation_bounds_policy`. Rounds, term limits, concession steps and escalation rails come from it, not from this skill.
6. Set three positions: target, most likely, walk-away. Walk-away is not a number you choose: it is the cost of the best alternative including switching. With no alternative quote, say there is no BATNA and recommend changing the game (scope, volume, term, spec) rather than inventing one.
7. If the supplier asked for an increase: split it into the part an index justifies and the rest, using the index move from `tool:commodity_index_lookup` or, failing that, the user's. Without either, say the split cannot be made.
8. Levers: list between `config.thresholds.min_concession_levers` and `config.thresholds.max_concession_levers` tradable variables (volume, term, payment terms, forecast commitment, Incoterm, consignment, specification, rebates, index-linked clause, exclusivity, early payment, warranty, service credits). For each, give value to us, cost to us and value to them where known, and build a concession ladder in which every give has a matching ask.
9. Style and plan: from the Kraljic result choose collaborative, competitive or mixed. Add the opening position, agenda, five most likely supplier objections with data-based answers, red lines, roles, and the escalation path.
10. Run `procurement-policy-compliance` on the intended award or term before the plan is used, and state which approval the walk-away value needs as `tool:check_policy` replies (a limit is a route to an approver, not a ceiling on the deal).

## Frameworks and heuristics

- BATNA sets the walk-away; ZOPA is the overlap. If there is no overlap, say so.
- Never concede without a trade. Give small, get big, in a planned order.
- Dependency matters both ways: our share of their revenue and theirs of our category spend, when known.
- Distinguish fact (tool result), estimate (labelled) and judgement (yours).

## Policy hooks (SC-007)

`negotiation_bounds_policy` bounds rounds, terms and concessions. The approval limit comes from the procurement-policy-compliance skill. If a governed bound is missing, say it is not governed and do not substitute your own.

## Output specification

One-page summary: value at stake; target, most likely and walk-away each with its source; BATNA; ZOPA; top three asks and top three gives. Then fact base, lever and concession matrix, opening script and the five objection answers, risks and fallback, what would change the recommendation, and a capture template for agreed terms.

## Quality bar

- Every price traces to a tool result, the user, or a labelled assumption.
- The walk-away is checked against the BATNA.
- Every concession has a matching ask.
- Unavailable inputs are named, not worked around.

## Escalation and approval

Review-gated. Say plainly that a named category lead must approve targets and walk-away before use. Escalate when walk-away exceeds budget, when it is a sole source with no BATNA, or when the supplier appears to be in financial difficulty (switch to a continuity framing). Never send a message to the supplier from this skill.

## Worked example

A supplier asks for an 8 percent increase citing steel. The user gives a steel index move of 3 percent over the period (user-supplied) and the supplier's steel share of cost is not known, so the justified part cannot be computed: the plan says the increase is unproven beyond the index move and asks for the cost breakdown. `negotiation.batna_strength` returns the result for 2 alternative quotes held; the plan sets walk-away from the best alternative's landed price and labels target as an estimate because no should-cost exists.

## Anti-patterns

- Inventing a should-cost, target or walk-away.
- Treating an unassessed calculation as zero.
- Conceding without a matching ask.
- Using a bound that is not in the governed policy.
- Messaging the supplier from the plan.
