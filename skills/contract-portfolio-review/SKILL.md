---
name: contract-portfolio-review
description: Review the contract portfolio as a whole - what is on the register, what expires or auto-renews over a horizon, which agreements carry notice clauses, how the suppliers behind them perform and screen, where spend runs with no contract on record - and turn it into a renewal pipeline with owners and dates. Use when asked for a renewal calendar or pipeline, the state of the contract portfolio, upcoming renewals to plan for, or contract coverage of spend. Not for the decision on one contract (renewal-and-exit-strategy) or for reading one contract's clauses (contract-clause-review).
---

# Contract Portfolio Review

## Purpose

Give the procurement lead a dated view of the whole register: which contracts need a decision and by when, which renew by themselves unless someone acts, which suppliers behind them deserve attention, and where money is spent with no contract on record. The pipeline is built from the register and the stored clauses; nothing is scheduled, sent or changed.

## When to use and when not

Use when the user asks for a renewal calendar or pipeline, upcoming renewals over a period, the state or health of the contract portfolio, contracts with auto-renewal, or contract coverage of spend.

Not for deciding one contract's renewal, renegotiation or exit (renewal-and-exit-strategy, which takes each pipeline item), for one contract's wording (contract-clause-review), for a plain list of expiring contracts (the search_contracts tool answers that directly in the Ask box), or for a supplier's clearance (supplier-risk-assessment).

## Inputs and preconditions

- The horizon (default the next `config.thresholds.horizon_days` days) and the period for spend (default the trailing `config.thresholds.spend_period_months` months), the reporting currency, and any scope (a category's suppliers, a region, a business unit the user names).
- The organisation's renewal rules (who decides, lead times, mandatory re-tender thresholds) from ONE call to `tool:get_policy` with the query `approval_threshold_policy` (the only purchasing-authority policy on this platform; there is no renewal-rules policy to search for); where it states nothing about renewals the configured lead times are defaults and the review says so. The round budget is small: policy once, the contracts once, then the trackers.

Use `tool:search_contracts` with `expires_within_days` set to the horizon (and `status` as needed) for the contracts falling due, with their end dates, values, auto-renewal flags and renewal terms as stored; `tool:obligation_tracker` per contract in the pipeline for its calendar and the clauses that carry a notice, review or renewal, quoted word for word; `tool:build_spend_cube` by supplier for the period's spend, matched to the register's suppliers for coverage; `tool:calc_supplier_scorecard` and `tool:screening_summary` for the suppliers behind the largest or soonest contracts; `tool:savings_tracker` for savings recorded against renewals; and `tool:run_calculation` with `deal.pct_change` for shares and changes.

NOT AVAILABLE: build_xlsx. Give the pipeline as a text table.
NOT AVAILABLE: render_chart. Give the expiry profile as a table by month.

## Workflow

1. Frame. Fix the horizon, the spend period, the currency and the scope; read the renewal rules and say which points are not governed.
2. Falling due. Read the contracts expiring within the horizon; order by end date; mark each auto-renewing one. The register covers only uploaded contracts: say so, and that a contract not on it is not covered by this review.
3. Notice and decision dates. For each contract in the pipeline (up to `config.thresholds.max_contracts_detailed` in detail, the rest as a list), read the tracker: expiry, auto-renewal, and the notice or review clauses quoted; the notice period is read from the clause, never computed from a period the records do not state; where no clause is stored the decision date is "unknown: clause not read" and the contract is flagged.
4. Lead time. Working back from the notice deadline where known, or the expiry where not, with the buffer `config.thresholds.decision_buffer_days` days: a contract whose decision date is within `config.thresholds.urgent_days` days is urgent; one whose date has passed is overdue, said first.
5. Suppliers. For the largest and soonest contracts, read the scorecard (dimensions with record counts) and the screening summary (not performed is not clear); a supplier with a flagged screening or many open findings is flagged in the pipeline.
6. Coverage. From the spend cube by supplier for the period, list the suppliers with spend above `config.thresholds.uncovered_spend_min` in the reporting currency and no live contract on the register as "no contract on record" (not "no contract").
7. Pipeline. The table: contract, supplier, value, end date, auto-renewal, notice clause status, decision date, urgency, supplier flags, suggested route (renew, renegotiate, re-tender, exit: a question for renewal-and-exit-strategy per contract), owner to assign, and the rule or policy that applies.
8. Expiry profile. Count and value of contracts ending per month over the horizon; share of portfolio value due within the horizon via `deal.pct_change` where both figures are read.

## Frameworks and heuristics

- Overdue first, then urgent, then by decision date; value breaks ties.
- Auto-renewal is a deadline, not a convenience: every auto-renewing contract without a read notice clause is flagged.
- Decision dates come from clauses and the register; the skill never infers a notice period from a renewal term or a "usual" ninety days.
- Coverage means spend against a contract on record; the register's gaps are stated as gaps.
- Supplier flags inform the route; they do not decide it.

## Policy hooks (SC-007)

Renewal rules, mandatory re-tender thresholds and who decides come from governed policy via `tool:get_policy`; where none is stated the pipeline says the point is not governed and applies the configured defaults as defaults. Decisions on each contract run through renewal-and-exit-strategy and its approvals; nothing is committed here.

## Output specification

1. Frame: horizon, period, currency, scope, rules or "not governed", register coverage caveat. Write every section for the contract owner: contracts, suppliers, dates and clauses in plain words; never name tools, tables, files or how the records are stored.
2. Overdue and urgent items, each with the clause quoted or "clause not read".
3. Pipeline table ordered by decision date.
4. Supplier flags for the largest and soonest contracts.
5. Spend with no contract on record, by supplier.
6. Expiry profile by month; share of value due.
7. Owners to assign, hand-offs per contract, data gaps (contracts not read, clauses not stored).

## Quality bar

- Every date, value and flag comes from the register, the tracker, the cube, the scorecard or the screening summary.
- No notice deadline is computed from an unstated period; "clause not read" is a status.
- Every auto-renewing contract within the horizon appears in the pipeline.
- Spend without a contract on record is labelled as on record, not as uncovered in fact.

## Escalation and approval

Escalate overdue items and auto-renewing contracts without a read notice clause to the contract owners now; escalate suppliers with flagged screening to compliance. The pipeline is advice; each decision follows renewal-and-exit-strategy.

## Worked example

Asked for the renewal pipeline for the next six months, the skill reads the contracts expiring within the horizon (several, two auto-renewing), reads each tracker (notice clauses quoted for most; two contracts never read for clauses, flagged), works back with the buffer (one decision date already passed: overdue, stated first), reads the scorecard and screening summary for the three largest suppliers (one with many open findings, none screened), lists four suppliers with material spend and no contract on record, and hands each pipeline item to renewal-and-exit-strategy with a suggested owner.

## Anti-patterns

- Computing a notice deadline from a "usual" period or from the renewal term.
- Treating "no contract on record" as "no contract".
- Leaving an auto-renewing contract out of the pipeline because its value is small.
- Deciding renew or exit here instead of handing each contract to renewal-and-exit-strategy.
- Scheduling or sending anything.
