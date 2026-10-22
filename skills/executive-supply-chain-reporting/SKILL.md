---
name: executive-supply-chain-reporting
description: "Assemble the executive view of the supply chain for a period - service (on-time delivery where measurable), cost (spend and its movement), cash (payment terms, open commitments), risk (screening coverage, flagged suppliers, contracts near expiry), value (as the ledger states it) and the actions decided - from the records and the figures the user supplies, with every number sourced and the gaps named, as a deck or a document the reader can act on. Use when asked for the supply chain board pack, the executive summary of operations, or a leadership update. Dormant pending review. Not for procurement's own function report (procurement-reporting) or the daily alerts (control-tower-monitoring)."
---

# Executive Supply Chain Reporting

## Purpose

Give leadership one page that is true: the few measures that matter, their movement, what caused what the records show, the decisions needed; nothing claimed the records cannot carry.

## When to use and when not

Use when a period summary of the supply chain is wanted for leadership: board, executive committee, operations review.

Not for procurement's function report (procurement-reporting), for the daily list (control-tower-monitoring), for one supplier (supplier-performance-review) or for the ESG report (csrd-esg-reporting).

## Inputs and preconditions

- The period, the reader, the measures leadership expects, targets with their sources.
- Figures the records do not hold (stock, service to customers), from the user, labelled.

- Use `tool:build_spend_cube` for spend by supplier, country and month in the period and the prior period
- Use `tool:calc_supplier_scorecard` for on-time delivery and findings of the largest suppliers
- Use `tool:open_orders` for open commitments by supplier
- Use `tool:invoice_status` for overdue invoices and no-order invoices by supplier
- Use `tool:contract_pipeline` for contracts ending in the horizon and unread
- Use `tool:screening_summary` for coverage and flags of the largest suppliers
- Use `tool:savings_tracker` for value as the ledger states it
- Use `tool:run_calculation` with `deal.pct_change` for movements, `deal.share_pct` for shares and `finance.average_per_period` for a run rate
- Use `tool:get_policy` for targets and KPI definitions if governed
- Use `tool:build_pptx` for the executive deck
- Use `tool:build_docx` for the written summary
- Use `tool:render_chart` for the few charts the deck carries
NOT AVAILABLE: calc_kpis. no cross-cutting KPI tool; each measure comes from the tool that computes it, named per figure
NOT AVAILABLE: query_wms. no stock data; stock measures are the user's, labelled

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.max_headline_measures`.

## Workflow

1. Frame: period and prior period, reader, the measures expected with targets and their sources; what the records hold and what the user supplies.
2. Service: on-time delivery as the scorecard measures it for the largest suppliers (and its basis), findings counts; customer service only as the user supplies.
3. Cost: spend and its movement via the registry; the largest movements by supplier or country from the cube; run rate via the registry.
4. Cash: open commitments from open_orders; overdue and no-order invoices from invoice_status; payment-terms spend from the cube by payment terms.
5. Risk: screening coverage and flags via the registry from the register; contracts ending and unread from the pipeline; supplier risk events.
6. Value and decisions: value as the ledger states it; decisions taken and needed, each with an owner; the summary page of at most `config.thresholds.max_headline_measures` measures with movement and target; the deck or document built; charts only where a movement needs one.

## Frameworks and heuristics

- Six measures leadership will act on beat sixteen they will scroll.
- A movement is reported; its cause only where the records show it.
- Value is the ledger's word.
- Every slide names its source in small print.

## Policy hooks (SC-007)

Targets and KPI definitions come from `tool:get_policy` and are quoted; value follows the ledger and the savings-methodology rules. Nothing is decided or committed by the skill; the decisions page asks, it does not answer.

## Output specification

1. Summary page: measures, movement, target, source.
2. Sections: service, cost, cash, risk, value, decisions.
3. Data gaps and the user-supplied figures labelled.
4. Deck or document export id; chart export ids.

## Quality bar

- Every figure traces to a tool, the ledger or a labelled user figure.
- Movements, shares and run rates come from the registry.
- Targets are quoted or absent.
- Nothing is decided.

## Escalation and approval

Escalate to finance where cash figures differ from theirs, to the data owner for gaps that blind a measure, and to the executive for the decisions page.

## Worked example

Asked for the quarterly supply chain pack for the board, the skill reads the cube for the quarter and the prior one, the scorecards of the ten largest suppliers, open commitments and overdue invoices, the contract pipeline, screening coverage and the ledger, computes movements and shares via the registry, writes the six-measure summary with targets quoted from policy, the five sections with sources, the decisions page with owners, labels the stock figure as the user's, and builds the deck with two charts.

## Anti-patterns

- Sixteen measures on the summary page.
- A cause the records do not show.
- Claimed value reported as delivered.
- A stock figure with no label when the records hold none.
