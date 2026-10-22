---
name: procurement-reporting
description: "Assemble a procurement performance report for a period - spend and its concentration, contract coverage, savings and value delivered as the ledger states them, supplier performance and findings, compliance to process - with every figure traced to its source tool and the period and scope stated once. Use when asked for a monthly, quarterly or board report on procurement, a KPI pack, a dashboard-style summary or 'how is procurement doing'. Not for one supplier's review (supplier-performance-review) or for the spend analysis alone (spend-analysis)."
---

# Procurement Reporting

## Purpose

Give leadership a period report that is true to the records: the same figures finance and the ledger would give, the gaps named, the movements explained only where the records explain them. The skill assembles and formats; it invents no target and claims no value the ledger does not hold.

## When to use and when not

Use when the request is a report, pack, KPI set or summary of procurement's results over a period for a reader (board, CFO, CPO, team).

Not for one supplier (supplier-performance-review), for the spend analysis in depth (spend-analysis), for validating a savings claim (savings-methodology), for the opportunity pipeline detail (opportunity-pipeline-review) or for the document file itself.

## Inputs and preconditions

- The period and the reader; the prior period for comparison if wanted.
- The sections the reader expects, if the organisation has a template; otherwise the sections here.
- Any targets, from policy or the user, labelled by source.

- Use `tool:build_spend_cube` for spend by supplier and category in the period and the prior period
- Use `tool:contract_pipeline` for contract coverage, expiries and unread notice clauses
- Use `tool:savings_tracker` for savings and value as recorded, by status
- Use `tool:invoice_reconciliation` for invoices without orders and open findings in the period
- Use `tool:calc_supplier_scorecard` for the top suppliers' on-time and findings
- Use `tool:run_calculation` with `deal.pct_change` for period-on-period movements
- Use `tool:get_policy` for targets and KPI definitions the organisation has set
NOT AVAILABLE: build_docx. the report is given as structured text; the document is built elsewhere
NOT AVAILABLE: build_chart. figures are given as tables

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.expiry_horizon_days`.

## Workflow

1. Frame: period, prior period, reader, sections, targets by source.
2. Spend: total, by category, top suppliers and concentration, movement against the prior period.
3. Coverage and contracts: share of spend under a live agreement, expiries in the next `config.thresholds.expiry_horizon_days` days, unread notice clauses.
4. Value: savings and value delivered by status exactly as the ledger states (claimed, validated, realised); no reclassification; the methodology skill decides what counts.
5. Compliance: invoices without orders, off-contract spend, open findings; movements.
6. Suppliers: top suppliers' delivery and findings from the scorecard.
7. Assemble: a one-page summary (five to eight figures with movement and target where one exists), the sections, the data gaps, and the questions the figures raise. Each figure carries its source and period.

## Frameworks and heuristics

- One scope, one period, stated once; every figure obeys them.
- Movements are reported; causes are reported only where the records show them.
- Value is the ledger's word, by status; a claimed saving is not a realised one.
- Targets come from policy or the user, never from the skill.
- A gap in the data is a line in the report, not a blank.

## Policy hooks (SC-007)

KPI definitions and targets come from `tool:get_policy` and are quoted with their source. The report commits nothing and changes no record; where a figure would be read as a claim to finance, the savings-methodology rules apply and are referenced.

## Output specification

1. Summary page: key figures with movement and target (source stated).
2. Sections: spend, coverage and contracts, value, compliance, suppliers.
3. Data gaps and caveats.
4. Questions the figures raise, for the reader.
5. Appendix: figure, source tool, period.

## Quality bar

- Every figure traces to a tool result and a period.
- Value is by ledger status; nothing reclassified.
- Targets are quoted from policy or labelled user-supplied.
- Nothing is created, sent or recorded.

## Escalation and approval

Escalate to finance where value figures differ from the ledger's, to the CPO where a target is missing, and to the data owner where a source is incomplete for the period.

## Worked example

Asked for the quarterly procurement report for the board, the skill reads the cube for the quarter and the prior one, the contract pipeline for coverage and expiries, the ledger for value by status, the reconciliation for no-order invoices and findings, and the scorecards of the top suppliers; it writes a summary page of seven figures with movements and the two targets the policy holds, the five sections with sources, names that one category has no contract coverage data, and lists the three questions the figures raise.

## Anti-patterns

- Reporting claimed savings as delivered.
- Explaining a movement with a cause the records do not show.
- Inventing a target to compare against.
- Mixing periods or scopes between sections.
