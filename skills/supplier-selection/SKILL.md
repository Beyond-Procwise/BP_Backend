---
name: supplier-selection
description: Shortlist and pre-qualify suppliers for a sourcing event, applying knockout gates, financial health, dependency, capacity, ESG and concentration checks, and recommend who to invite with the reason each was kept or dropped. Use after a longlist exists or a category needs new sources. Advisory only, and it never clears a screening match itself. Not for scoring received bids, deciding the award, or onboarding a supplier.
---

# Supplier Selection

## Purpose

Turn a longlist and the risk data into a qualified shortlist of suppliers to invite, and say why each one was kept or dropped. It is a funnel: hard gates first, then scored survivors, then a check that the shortlist is genuinely competitive.

## When to use and when not

Use before a sourcing event, when a longlist exists or a category needs new sources.

Not for scoring bids that have been received or deciding the winner (that is `award-recommendation`), and not for onboarding a supplier. This skill never writes to the supplier master, never contacts a supplier and never sends an invitation. Approvals and bidder-count rules come from `procurement-policy-compliance`.

## Inputs and preconditions

Ask for these; do not guess any of them.

- Category, annual spend and annual volume, sites and regions served, required certifications, risk appetite and the target shortlist size. Annual volume drives the capacity and dependency tests, so ask for it if it is missing.
- The longlist, with legal entity names and, where known, owners, directors, parent group, revenue and country.
- The type of event (RFP, RFQ or auction), because it sets the shortlist size.
- For each risk check, the result and its date, supplied by compliance or the user.
- The weights for the qualification score, from the category owner. None is assumed.

If no longlist exists, ask the user for candidates or a source.

Use `tool:supplier_discovery` to build the known-supplier part of the longlist from the organisation's own orders, invoices and quotes, and `tool:supplier_lookup` for a supplier's recorded profile. It is not a market scan: ask the user for any other source of candidates, and do not name suppliers from memory as if they were verified.
NOT AVAILABLE: screen_sanctions. Ask compliance for the result and date for each legal entity, owner and director. If none is supplied, record "not performed"; that supplier cannot be shortlisted.
NOT AVAILABLE: forced_labour_screen. Ask for the result and date where the geography or material is high risk; if none, record "not performed".
NOT AVAILABLE: conflict_minerals_check. Ask for the result and date where tin, tantalum, tungsten or gold is in scope; if none, record "not performed".
NOT AVAILABLE: financial_health_score. Ask for the probability of default, the score and its 12-month change with their date; if none, record "not performed".
NOT AVAILABLE: esg_rating_lookup. Ask for the rating, the provider and the date; if none, record "not available" and request it by questionnaire.
NOT AVAILABLE: monitor_risk_events. Ask the user or compliance for current country and event risk, with the index version and date; if none, record "not performed".
For incumbents, use `tool:calc_supplier_scorecard` with the supplier and period for spend, purchase price variance, on-time delivery and open findings measured from this organisation's own records, and report each with its record count or its not-assessed reason. Quality is not recorded: record quality track record as "not available" unless the user supplies it with its period.
NOT AVAILABLE: build_xlsx. Give the funnel and the registers as text tables.
NOT AVAILABLE: build_docx. Give the memo as text using templates/memo.md.

## Workflow

1. Confirm the inputs above and the event type. Count the longlist; this is the first funnel number.
2. Knockout gates. Any failed gate drops the supplier, and the drop is recorded with the gate and the evidence.
   a. Sanctions: use the result compliance supplied for the legal entity, owners and directors. A potential match, including a partial-name hit on a director, puts the supplier on hold for compliance. Never clear it yourself and never shortlist a supplier with an unresolved match.
   b. Forced labour and conflict minerals: use the supplied results for high-risk geographies and materials. A potential match goes to compliance in the same way.
   c. Mandatory certifications, with evidence on file for each (for example IATF 16949 for automotive production parts, ISO 13485 for medical devices). A certificate that is claimed but not evidenced is "not evidenced", not a pass.
3. Financial health, from the supplied result. Drop a supplier whose 12-month probability of default is above `config.thresholds.pd_drop_pct` percent. Flag, but keep, one at or above `config.thresholds.pd_flag_pct` percent, or whose score fell by more than `config.thresholds.score_decline_flag_points` points in 12 months. No result means "not performed", and the supplier stays open rather than being scored as healthy.
4. Dependency and capacity. Divide our annual spend by the supplier's annual revenue and show the working. Flag a ratio above `config.thresholds.dependency_max_pct` percent (we would become their lifeline). Deprioritise one below `config.thresholds.irrelevance_min_pct` percent unless the supplier is strategic (we would not matter to them). Compare their stated capacity headroom with our peak volume; no capacity figure means "not available".
5. ESG, from the supplied rating and provider. Apply the minimum level the Ethical Sourcing Policy sets, from the user or compliance; do not state one yourself. A missing rating is requested by questionnaire and is not an exclusion by default.
6. Concentration and geography. Check parent links for each survivor: two suppliers under one parent are one group. Avoid a shortlist where every supplier shares a country, a port or a parent group, using the supplied country risk and event data.
7. Qualification score. Rate each survivor on capability fit, quality track record, financial, risk and ESG, on one scale, from evidence, and mark any rating with no evidence as not available. Combine them with `scoring.weighted_total` through `tool:run_calculation`, passing the criteria and weights from the category owner, the scale and the scores; use one evaluator per row. If it comes back unassessed, say why (for example weights that do not sum to 1) and do not total by hand. Rank the survivors from its result.
8. Shortlist size. Use `config.thresholds.rfp_shortlist_min` to `config.thresholds.rfp_shortlist_max` for an RFP and `config.thresholds.rfq_shortlist_min` to `config.thresholds.rfq_shortlist_max` for an RFQ or auction, and at least `config.thresholds.min_independent_groups` independent parent groups where policy requires competition. These are configurable defaults. If fewer survive, say so and offer options: widen the region, relax a non-mandatory criterion, or run an RFI first. Do not pad the list with a supplier that failed a gate.
9. Policy. Call `tool:check_policy` for the minimum bidder count and for preferred or diverse supplier rules, and report its reply as it is, including anything it lists as not assessed.
10. Write the funnel and memo using templates/memo.md.

## Frameworks and heuristics

- Knockout first, then score: a gate failure is never offset by a high score.
- Kraljic decides how many sources to keep: run `negotiation.kraljic_quadrant` through `tool:run_calculation` only if the user supplies the spend, alternative supplier count and risk score; otherwise say the quadrant is not determined.
- Dependency is a ratio of our spend to their revenue, read both ways.
- Country risk indices come from the user or compliance and carry a version and date.
- Separate fact (supplied result), estimate (labelled) and judgement (yours).

## Policy hooks (SC-007)

The minimum number of bidders, preferred and diverse supplier rules, and any approver come from `tool:check_policy` and the skill `procurement-policy-compliance`, not from this skill. Do not state a count, limit or role the tool did not return. The size and risk thresholds above are configurable defaults, not policy.

## Output specification

A memo from templates/memo.md with: funnel counts by stage; the shortlist table (supplier, qualification score, key evidence, risks, confidence high, medium or low); the dropped-supplier register with the gate and evidence for each; and open items (pending compliance review, missing data). Every number is from a tool result or the user and says which.

## Quality bar

- Every exclusion cites the gate and the evidence.
- No supplier with an unresolved screening match is shortlisted.
- Independence is checked at parent level, not by supplier name.
- Checks not performed are shown as not performed, never as passed.
- Confidence and assumptions are stated.

## Escalation and approval

Advisory only; nothing is written to the supplier master and nobody is invited. Potential sanctions or forced-labour matches go to compliance through the case route. Escalate when fewer suppliers qualify than the policy minimum, or when the user wants to include a supplier that failed a gate. Pressure to add a favoured supplier "just this once" is refused: restate the gate that failed, and say that an exception needs the approval route and not this skill.

## Worked example

A buyer shares a longlist of 40 machining suppliers across 3 countries, annual spend 2,000,000 GBP, for an RFQ. Compliance supplied sanctions results dated last week: 2 suppliers have a potential match (one a partial-name hit on a director), so both are held for compliance and left out. 5 more lack evidence of the required certificate and are dropped. One supplier has a 12-month probability of default of 7 percent and is dropped at the financial gate. Our spend is 8 percent of one supplier's 25,000,000 GBP revenue, so no dependency flag. An incumbent has no ESG rating: it stays, with a questionnaire requested. Two suppliers share one parent, so they count as one group. The survivors are scored with weights the category owner supplied, and the shortlist is the top ranked within the RFQ size, covering the independent groups. The memo shows the funnel 40, 33, 32 up to that point, then the later stages from the counts the remaining checks give, with the held suppliers as open items.

## Anti-patterns

- Clearing, or shortlisting past, a potential screening match.
- Treating a missing check as a pass.
- Excluding a supplier only for a missing ESG rating.
- Counting two suppliers with one parent as independent.
- Totalling the qualification score by hand or inventing its weights.
- Padding the shortlist with a supplier that failed a gate.
- Stating a bidder count, limit or approver that `tool:check_policy` did not return.
