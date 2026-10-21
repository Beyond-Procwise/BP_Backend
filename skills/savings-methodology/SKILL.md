---
name: savings-methodology
description: Classify and validate a procurement saving - hard cost reduction, soft saving or cost avoidance - choose the baseline, separate price, FX, index and volume effects, check for double counting, and prepare it for finance sign-off. Use when someone logs a saving, reviews a savings pipeline or disputes a savings claim. Review-gated, so a finance reviewer independent of the owner signs off. Not for negotiation targets or budget variance without a sourcing action.
---

# Savings Methodology

## Purpose

Make every reported saving reproducible from its baseline, price and volume, so finance can reconcile it to the ledger. A saving that cannot be recalculated from stored evidence stays a draft.

## When to use and when not

Use when a saving is logged, reviewed or disputed.

Not for setting negotiation targets (use negotiation-planning) or for budget variances that have no sourcing action behind them.

## Inputs and preconditions

Ask for evidence. No evidence means status stays DRAFT with a missing-evidence list.

- The initiative, the contract or PO that proves it, and the effective date.
- Before and after prices, currency, and volumes.
- Whether the item has purchase history and how many purchases in the baseline window.

NOT AVAILABLE: savings_tracker. There is no tool that records or phases savings: produce the validation sheet as text and leave the status DRAFT.
NOT AVAILABLE: commodity_index_lookup. Ask the user for the index series; without it say "not neutralised for index movement" and do not net anything.
NOT AVAILABLE: build_spend_cube. Use `tool:get_corpus_facts` for invoiced prices and volumes, or ask the user.
NOT AVAILABLE: build_xlsx. Give the validation sheet as a table.
NOT AVAILABLE: build_docx. Give the memo as text.

## Workflow

1. Gather the evidence above. If there is no price evidence, stop with DRAFT and the missing list.
2. Classify. Price paid falls against a valid baseline: hard saving. A benefit that does not reduce cost (payment terms, inventory, process time): soft. An increase prevented, or a price below market or below the requested increase: cost avoidance. Split mixed initiatives into components.
3. Baseline. Recurring item with at least three purchases in `config.thresholds.baseline_window_months` months: weighted-average price over that window (use `tool:run_calculation` with `deal.weighted_unit_price`). Infrequent item: last price paid, only if newer than `config.thresholds.last_price_max_age_months` months. New item with no history: the first competitive quote, and the saving is cost avoidance only. A budget baseline is allowed for hard savings only where policy permits; check `tool:get_policy`.
4. Calculate the price effect only: `tool:run_calculation` with `deal.realised_savings` using the quoted and invoiced unit prices and the invoiced volume. Report any volume change separately and never book a volume drop as a price saving. Use `deal.pct_change` for percentages.
5. FX and index. Restate at a constant rate only with a rate the user or a tool supplied; say which. Index neutralisation needs a user-supplied series; otherwise report "not neutralised". If neutralising makes the saving negative, report the negative figure; do not hide it.
6. Soft savings (payment terms): value only with a cost of capital the user or a governed policy supplies, and keep them out of P&L totals. Cost avoidance is reported separately and is never added to hard savings.
7. Validate. Check the same supplier or item is not claimed in another initiative, that it does not repeat a prior-year saving, and that realised results are within `config.thresholds.realised_vs_plan_variance_pct` percent of plan, else flag for re-review. Check the saving does not depend on a specification downgrade nobody approved.
8. Materiality. If the annualised value reaches `config.thresholds.finance_review_min_annualised`, it needs finance review; above `config.thresholds.escalate_above_annualised`, escalate. Confirm who may approve with `procurement-policy-compliance`.
9. Prepare for sign-off. Status stays DRAFT until a finance reviewer, independent of the initiative owner, accepts it.

## Frameworks and heuristics

- Price-volume-mix: saving = (adjusted baseline minus new price) times volume; volume effect separate.
- Constant-currency reporting; index neutralisation only with evidence.
- Hard, soft and cost avoidance are never summed into one headline figure.

## Policy hooks (SC-007)

Savings-recognition rules, the materiality limit and who may approve come from governed policy. If a policy row is missing, say it is not governed; the thresholds in config are this skill's defaults for review, not policy.

## Output specification

A validation sheet per initiative: type and sub-type; baseline method and evidence; bridge from baseline to price saving with FX, index and volume effects (each "not neutralised" where it could not be calculated); phasing note (not computed, since no tracker exists); risks to realisation; status DRAFT; missing evidence. Then a pipeline summary by type.

## Quality bar

- Each saving is recalculable from the stated inputs.
- Type and baseline follow the rules above.
- No double counting; adjustments shown.
- Every input is sourced or labelled.

## Escalation and approval

Review-gated: finance sign-off by someone independent of the owner. Escalate claims above `config.thresholds.escalate_above_annualised`, baseline overrides, retro-dated savings, and any pressure to relabel cost avoidance as a hard saving. Never mark a saving approved.

## Worked example

A buyer reports a 6 percent price cut on aluminium parts. The weighted-average baseline over the window is 4.20 per unit (from `deal.weighted_unit_price`) and the new price is 3.95; invoiced volume 10,000 units. `deal.realised_savings` returns 2,500 before any neutralisation. The user says the aluminium index fell 9 percent but has not given a series, so the saving is reported as "not neutralised for index movement" and flagged as at risk of being negative once neutralised. Status DRAFT; missing evidence: the index series and the PO.

## Anti-patterns

- Booking a volume reduction as a price saving.
- Adding cost avoidance to hard savings.
- Using a budget baseline without a governed permission.
- Netting an index move the user has not evidenced.
- Marking a saving approved.
