---
name: rebates-and-incentives
description: "Work out what a rebate, volume discount or growth incentive is worth and whether it is being earned and claimed - the clause's own formula and thresholds quoted, the qualifying spend from the records, the position against each threshold, the amount due under the formula, the claim procedure and deadline, and whether the structure is doing us any good. Use when asked what rebate we are owed, whether we will hit a tier, how to claim, or whether to accept a rebate structure a supplier offers. Not for designing index or collar mechanisms (pricing-mechanisms) or for logging the value (savings-methodology)."
---

# Rebates and Incentives

## Purpose

Make rebates real money: the clause read as written, the qualifying spend counted the way the clause counts it, the amount computed by the clause's formula on recorded figures, claimed on time, and the structure judged on whether it changes anything we would not have done anyway. The skill computes and advises; it claims nothing and signs nothing.

## When to use and when not

Use when the question is about a rebate, retrospective or volume discount, growth incentive, tier threshold or bonus: how much, whether earned, how to claim, or whether to accept one.

Not for designing price mechanisms in a new contract (pricing-mechanisms), for claiming the value as a saving (savings-methodology), for the negotiation of the rebate (negotiation-planning) or for invoice-level price errors (overbilling-investigation).

## Inputs and preconditions

- The supplier and the agreement with the rebate, or the structure the supplier offers, as written.
- The period in question.

- Use `tool:search_contracts` for the agreement
- Use `tool:extract_clauses` for the rebate, discount, incentive, measurement, payment and claim clauses, quoted with thresholds and formula
- Use `tool:obligation_tracker` for the claim deadline and the measurement period dates
- Use `tool:build_spend_cube` for invoiced spend with the supplier over the measurement period
- Use `tool:invoice_reconciliation` for the invoiced lines by item where the clause counts only some items
- Use `tool:run_calculation` with `deal.pct_change` for the growth against the base period and for the gap to a threshold, and `critic.annualised_rate` to project the period's rate
- Use `tool:get_policy` for how rebates must be recorded and who claims them
NOT AVAILABLE: send_message. the claim is drafted; finance or the owner sends it
NOT AVAILABLE: build_docx. the claim and the analysis are given as text

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.tier_proximity_pct`.

## Workflow

1. Quote the clause: what spend qualifies (items, entities, net of what), the measurement period, thresholds and rates per tier, whether the rate applies to all spend or the increment, the base period for growth, the claim procedure, deadline and payment form (credit, cash).
2. Count the qualifying spend from the records the way the clause counts it; state what was included and excluded; spend in a currency without a rate is named.
3. Position: spend against each threshold; the gap to the next tier via the registry; the period's rate projected via the registry where the period is not complete, labelled a projection.
4. Amount due: the clause's formula applied to the qualifying spend by the registry; where the formula is in prose only, the arithmetic is shown step by step with the inputs; a figure within `config.thresholds.tier_proximity_pct` of a threshold is flagged as sensitive to the count.
5. Claim: deadline from the tracker, procedure from the clause, the draft claim text, who sends it per policy; whether earlier periods were claimed (findings if not).
6. Judgement where a structure is offered: whether the thresholds sit where our spend already is (money for nothing is fine; stretching for a tier is a cost), whether the increment or whole-spend basis is stated, the cash-flow and accounting treatment to raise with finance.

## Frameworks and heuristics

- The clause counts the spend, not the cube's default; read the definition before counting.
- Whole-spend versus increment basis changes the value several times over; never assume.
- A tier just out of reach is a lever for the supplier, not for us.
- Unclaimed rebates expire; the deadline is the first fact to find.
- A rebate is not a saving until the methodology says so.

## Policy hooks (SC-007)

Recording and claiming rules come from `tool:get_policy` and are quoted. The skill claims nothing and signs nothing; a draft claim goes to the owner the policy names. Accepting an offered structure that changes commitments is checked with `tool:check_policy` and reported as replied.

## Output specification

1. Clause summary with thresholds, rates, basis, period, claim procedure and deadline, quoted.
2. Qualifying spend with inclusions and exclusions stated.
3. Position per tier with gaps and projection, via the registry.
4. Amount due with the arithmetic shown; sensitivity flag near a threshold.
5. Claim draft, deadline and owner; unclaimed earlier periods.
6. Judgement of an offered structure, where asked.

## Quality bar

- Thresholds, rates and basis are quoted from the clause, never assumed.
- Qualifying spend is counted as the clause defines it, with exclusions named.
- All arithmetic is from the registry or shown step by step with inputs.
- Nothing is claimed, sent or accepted.

## Escalation and approval

Escalate to finance for the claim and the accounting treatment, to legal where the clause's basis is ambiguous, and to the category lead where a structure would pull volume to a supplier for the wrong reason.

## Worked example

Asked how much rebate Acme owes us this year, the skill quotes the clause (tiers at stated spend levels, a rate on whole spend above each, net of credit notes, calendar-year measurement, claim within a stated period after year end), counts the qualifying spend from the cube net of credits, finds spend above the second tier with the third just out of reach, computes the amount due by the formula, flags that the count is close to the second threshold, reads the claim deadline from the tracker, drafts the claim for finance, and notes that last year's rebate was never claimed.

## Anti-patterns

- Applying the rate to the cube's gross spend when the clause nets credits.
- Assuming the increment basis or the whole-spend basis.
- Stretching spend to reach a tier without costing the stretch.
- Claiming or signing from the skill.
