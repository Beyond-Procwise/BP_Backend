---
name: rfx-design
description: Design a sourcing event (RFI, RFQ, RFP or tender) that can be evaluated fairly and defended - scope and specification, which suppliers to invite under policy, the evaluation criteria, weights, scale and scoring method tested with the scoring calculation, like-for-like bid terms, the timeline and the governance. Use when asked to design, draft or set up a tender, RFx or sourcing event, to write evaluation criteria or weights, or to plan a go-to-market. Not for choosing the winner (award-recommendation) or screening candidates (supplier-selection).
---

# RFx Design

## Purpose

Produce a sourcing event whose result will stand: a scope suppliers can price alike, criteria and weights fixed before bids arrive and proven to combine, a timeline with enough bid time, and the policy checks done up front. The skill designs the event; a person issues it, and the platform does not create or send it.

## When to use and when not

Use when the user wants to design or draft an RFI, RFQ, RFP, ITT or tender, define evaluation criteria, weights or scoring, decide the bid terms (currency, Incoterm, payment days) or the timeline, or plan a go-to-market for a requirement.

Not for building the longlist and shortlist (supplier-selection, which this skill calls first), for scoring received bids and recommending the winner (award-recommendation), for negotiation after bids (negotiation-planning), or for the category strategy the event serves (category-strategy).

## Inputs and preconditions

- The requirement: what, how much, where, when, specification or service levels, and the budget or estimated value with currency.
- The event type the user wants, or the reasons to choose one (RFI to learn the market, RFQ for a defined item on price, RFP for a solution on value, tender where policy or regulation requires it).
- The evaluation criteria the business cares about and who evaluates; the category owner's weights, or the request to propose some for their decision.
- The policy context: competitive-bidding thresholds, minimum number of quotes, approval limits, from `tool:get_policy` and `tool:check_policy`.

Use `tool:supplier_discovery` and `tool:supplier_lookup` for known suppliers of the items and their recorded Incoterms, currency and lead time, `tool:screening_summary` for each candidate's recorded checks before invitation, `tool:similar_items` and `tool:get_corpus_facts` for how the items were specified and priced before, `tool:search_contracts` for existing agreements that bind the scope, `tool:classify_spend` for the taxonomy category, `tool:run_calculation` with `scoring.weighted_total` to prove the criteria, weights and scale combine, with `bids.normalise` to define the like-for-like terms, and with `landed_cost.total_cost` where delivered cost is a criterion.

NOT AVAILABLE: create_rfx. The event is designed here as text; a person creates and issues it in the sourcing tool.
NOT AVAILABLE: build_docx. Give the RFx pack as text sections.

## Workflow

1. Frame. Confirm the requirement, value and the event type with reasons; read the policy on competitive bidding and quotes for that value with `tool:get_policy`, and the approval route with `tool:check_policy`. Say what policy requires (for example the minimum number of bidders, `config.thresholds.min_bidders` by default when no policy states one).
2. Scope and specification. Write the scope suppliers must price from what the user gave: items or services with quantities and units, service levels, delivery points, term, options. Never write a specification, a dimension, a load rating, a budget or a value the user did not give: a missing element is a question for the user, listed as such, and the value is "not given" until they give it. Mark every element that comes from the user; use prior line descriptions and prices from the records only as context, labelled as such. A specification that names one supplier's product is flagged.
3. Invitation list. Hand the longlist and shortlist to supplier-selection; record here who is invited and that each has a screening summary with no flagged or expired check, or is invited subject to compliance.
4. Criteria, weights, scale. Define criteria that the bids can evidence, the scale (default `config.thresholds.scoring_scale_max` points), the weights summing to one hundred percent with price at or below `config.thresholds.price_weight_max_pct` percent unless the user decides otherwise, the normalisation for price, and the pass/fail gates. Prove they combine by running `scoring.weighted_total` on two made-up illustrative bids labelled as test data; a result that comes back unassessed means the design is not evaluable and is fixed before issue.
5. Like-for-like terms. Fix base currency, Incoterm, payment days and validity period so bids compare; state how deviations will be normalised with `bids.normalise` and who supplies the rates.
6. Timeline. Issue, clarification window (at least `config.thresholds.clarification_days` days), bid deadline (at least `config.thresholds.min_bid_days` days after issue), evaluation, moderation, approval, award, standstill if the regime has one (ask), contract start; backward from the need date with the notice deadlines of any contract being replaced (from `tool:obligation_tracker`).
7. Governance. Evaluators and conflict-of-interest declarations, clarification rules (one channel, answers shared with all), late-bid rule, scoring moderation, record keeping, who approves the award (from the policy reply).
8. Pack. Assemble the sections below; list the decisions the category owner must make (weights, event type, invitees) and the policy items pending.

## Frameworks and heuristics

- Criteria must be evidenced by the bid: a criterion nobody can score from the submission is removed or turned into a pass/fail gate.
- Weights are the owner's decision; the skill proposes a set, shows what each is worth in points, and proves it combines.
- Price normalisation is chosen and stated before issue (lowest-price ratio, linear to budget, or scoring bands); changing it after bids arrive is an integrity failure.
- Like-for-like first: terms that differ by bidder (Incoterm, currency, payment days) are fixed in the invitation or normalised by a stated rule with stated rates.
- Bid time fits the complexity: a solution RFP needs more than the minimum; the minimum is a floor.

## Policy hooks (SC-007)

Whether competition is required, how many bidders, and who approves the award come from governed policy via `tool:get_policy` and `tool:check_policy`; the pack quotes the policy reply as it is. Where no policy states a requirement the pack says the point is not governed and applies the configured defaults as defaults.

## Output specification

1. Event summary: requirement, value, type with reasons, policy requirements and approval route.
2. Scope and specification, with the source of each element.
3. Invitation list and screening status per invitee.
4. Evaluation model: criteria, evidence required, scale, weights, normalisation, gates; the scoring test result.
5. Bid terms: currency, Incoterm, payment days, validity, normalisation rule.
6. Timeline with dates and dependencies.
7. Governance and roles.
8. Decisions for the owner and open policy items.

## Quality bar

- Every policy statement comes from a tool reply; every default is named as a default.
- The evaluation model was proven to combine with `scoring.weighted_total` before issue.
- No invitee without a screening status; no scope element without a source.
- The event is not created or sent by the skill.

## Escalation and approval

Escalate to the category lead for weights and invitees, to compliance for any invitee with a flagged or expired check, and to legal for the standstill rule and regulated regimes. The approval route for the eventual award is the policy's; the pack records it.

## Worked example

Asked to set up an RFQ for pallets, the skill reads the policy (three quotes above the threshold, Approver signs off), scopes the items from the user's quantities with prior descriptions as context, hands the longlist to supplier-selection and records four invitees with screening status, proposes price, lead time, quality certification (gate) and sustainability with the owner's weights to decide, runs the scoring calculation on two illustrative bids (combines; price normalised as lowest-price ratio), fixes GBP, DAP and thirty days, builds a timeline backward from the need date with the clarification and bid windows at their floors, and lists the owner's decisions.

## Anti-patterns

- Issuing criteria that bids cannot evidence, or weights that were never proven to combine.
- Changing the normalisation or weights after bids arrive.
- Inviting a supplier whose screening is flagged or not performed without saying so.
- Writing a specification around one supplier's product.
- Claiming the event was created or sent.
