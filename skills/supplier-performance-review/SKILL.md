---
name: supplier-performance-review
description: Review one supplier's performance over a period from the organisation's own records - spend, price variance against PO prices, on-time delivery, open findings, contract obligations and recorded risk events - and prepare the review pack and the asks for a business review. Use when asked how a supplier is performing, for a scorecard, a QBR or performance review pack, on-time delivery or OTIF, or price variance. Not for the renew-or-exit decision (renewal-and-exit-strategy), risk clearance (supplier-risk-assessment) or negotiation tactics (negotiation-planning).
---

# Supplier Performance Review

## Purpose

Say how a supplier has actually performed for us, from records rather than impressions, and turn that into a review pack: what is measured, what is not measured and why, where the supplier is strong or weak against configured thresholds, and the asks to raise with them. Nothing here is scored by opinion; every dimension states how many records it rests on.

## When to use and when not

Use when the user asks for a supplier's performance, scorecard, delivery record, price variance, open issues, or a pack for a quarterly or annual business review.

Not for deciding renewal or exit (renewal-and-exit-strategy, which takes this pack as its performance input), for risk clearance (supplier-risk-assessment), for comparing candidates (supplier-selection), or for the negotiation itself (negotiation-planning).

## Inputs and preconditions

- The supplier: name or supplier_id; confirm with the user when the match is partial.
- The period: default the trailing `config.thresholds.window_months` months, or the user's dates; the reporting currency.
- Quality, responsiveness, service-level and credit records from the user, labelled user-supplied: the platform does not record them.
- The category owner's weights, if a composite score is wanted.

Use `tool:calc_supplier_scorecard` for spend, price variance (`ppv.purchase_price_variance`), on-time delivery (`otif.delivery_performance`, tolerances from the governed policy) and open findings, each with its record count or its not-assessed reason. Use `tool:search_contracts` for the live contracts and `tool:obligation_tracker` for their dates and the clauses that carry service, delivery or review obligations, quoted word for word. Use `tool:monitor_risk_events` for recorded risk-event reviews and the platform's risk signals, `tool:supplier_lookup` for the master record, `tool:build_spend_cube` for the supplier's share of total spend, and `tool:savings_tracker` for savings recorded against the relationship. For a composite, use `tool:run_calculation` with `supplier.composite_score` and the category owner's weights only.

NOT AVAILABLE: build_docx. Give the review pack as text sections.
NOT AVAILABLE: render_chart. Describe trends as period-by-period tables.

## Workflow

1. Identify and frame. The round budget is small: call `tool:calc_supplier_scorecard` FIRST with the supplier as the user named it (it resolves the name, asks if several match, and reads the on-time tolerance policy itself; do not call supplier_lookup or get_policy before it). State the period and currency and say up front which dimensions the records can and cannot measure.
2. Measure. Run the scorecard for the period and, where the user wants a trend, for the equal period before it. Report each dimension with its record count: spend (and share of total from the cube), price variance as a percentage of standard spend, on-time rate with the number of purchase orders assessed, open findings by kind. A dimension the tool marks not assessed is reported as not assessed with its reason.
3. Judge against thresholds. On-time at or above `config.thresholds.on_time_target_pct` percent is on target; price variance within plus or minus `config.thresholds.ppv_tolerance_pct` percent is within tolerance; open findings at or above `config.thresholds.findings_escalate` escalate. These are configurable defaults, not policy, and are said to be so.
4. Obligations, when the user asks about the contract or rounds remain after the scorecard and `tool:search_contracts`. From the live contracts, list the service, delivery and review obligations with their clause and wording, and mark each as met, breached or not measurable from the records; never infer a breach from a figure the contract does not reference.
5. Risk events, when asked or when a round remains. List recorded reviews and signals in the period with type, severity, source and date; none recorded is "none recorded".
6. User-supplied dimensions. Add quality, responsiveness, SLA credits and the user's experience as reported, labelled user-supplied; they carry no record count.
7. Composite, only with weights. With the category owner's weights, run `supplier.composite_score`; a strong result is at or above `config.thresholds.score_strong_min`, a weak one below `config.thresholds.score_weak_max`. Without weights, present the dimensions separately and say the overall score is not available.
8. Asks and actions. For each weak or breached dimension, draft the ask for the review meeting with the evidence behind it, and the owner on our side. Hand a renew-or-exit question to renewal-and-exit-strategy and a price discussion to negotiation-planning.

## Frameworks and heuristics

- Record count before rate: an on-time rate over two purchase orders is reported with the two, not as a percentage alone.
- Our gaps are not their failures: a missing receipt, a missing promised date or a missing price is unassessed, never a miss.
- Trend only on equal windows with the same currency rule; otherwise "no trend".
- Price variance is against PO prices on matched lines; it is not a market comparison (that is negotiation-planning's).
- Findings are open items in our triage: they show where our process caught something, and the kind matters more than the count.

## Policy hooks (SC-007)

The on-time tolerances come from the governed `supplier_scorecard` policy through the scorecard tool; if the policy is not set the on-time dimension is not assessed and the pack says so. Review thresholds in config are not policy. Any action arising that changes the supplier relationship or commits spend is checked by the receiving skill with `tool:check_policy`.

## Output specification

1. Supplier, period, currency; what the records can and cannot measure.
2. Scorecard table: dimension, value, record count, threshold verdict (on target / within tolerance / weak / not assessed with reason), trend if read.
3. Contract obligations: clause, wording, met / breached / not measurable.
4. Risk events and signals in the period.
5. User-supplied dimensions, labelled.
6. Composite (with weights) or "overall score not available".
7. Asks for the review meeting with evidence and owners; hand-offs to other skills.

## Quality bar

- Every measured figure carries its record count and comes from a tool result; every not-assessed dimension carries its reason.
- No composite without the category owner's weights.
- No breach is asserted unless the clause names the measure and the record shows the miss.
- Thresholds are named as configurable defaults, not policy.

## Escalation and approval

Escalate to the category lead when open findings reach the escalation count, a contractual obligation is breached on the record, or a flagged risk event is recorded. The pack is advice for the review; it commits nothing and sends nothing.

## Worked example

Asked how Clearwater Logistics performed this year, the skill resolves the supplier (exact name), runs the scorecard: spend converted with the dashboard rule, price variance assessed over thirty-four matched lines in two currencies reported separately, on-time not assessed because no goods receipt carries a promised date, ten open findings mostly on quantity invoiced above PO. It lists the live contract's delivery clause as not measurable, no risk events recorded, and prepares two asks: evidence for the over-invoicing findings and promised delivery dates on future orders so on-time can be measured. The overall score is not available without weights, and the renew question is handed to renewal-and-exit-strategy.

## Anti-patterns

- Turning "no receipt recorded" into "delivered late".
- Averaging user-supplied impressions with measured rates.
- Stating an overall score with weights the skill chose.
- Calling a figure a breach of contract when the contract does not reference that measure.
- Guessing the supplier from a partial name.
