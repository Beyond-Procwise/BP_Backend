---
name: price-increase-challenge
description: "Test a supplier's cost justification for a surcharge or an increase against the evidence - the contract's price-review or index clause, the index series the supplier cites versus what the index actually did, the price history on record, the share of cost the input really represents - and say what the evidence supports and what to ask for. Use when asked whether a surcharge, inflation claim or cost pass-through is justified or in line. Not for planning the negotiation itself (negotiation-planning), which takes this analysis as input."
---

# Price Increase Challenge

## Purpose

Separate the part of a supplier's increase the evidence supports from the part it does not, with every figure traced to a clause, an index reading, a record or the supplier's own letter, so the negotiator starts from facts.

## When to use and when not

Use when a supplier has notified a surcharge, an inflation-based increase or a cost pass-through and the question is whether it is justified, in line with the contract or the index, and how to challenge it.

Not for the negotiation plan and concessions (negotiation-planning), for designing index clauses for a new contract (pricing-mechanisms), for benchmarking the absolute price (price-benchmarking) or for savings claims (savings-methodology).

## Inputs and preconditions

- The supplier and the increase as notified: amount or percentage, effective date, stated reasons, labelled user-supplied unless the letter is on record.
- The items or services affected and the contract, if any.
- Any cost breakdown the supplier has given.

- Use `tool:search_contracts` for the agreement and its stored summary
- Use `tool:extract_clauses` for the price-review, index, surcharge and pass-through clauses, quoted
- Use `tool:commodity_index_lookup` for the index series the clause or the supplier cites, over the period the supplier claims
- Use `tool:similar_items` for the price range as billed and the last date of the affected items across our lines (a range per source, not a dated series) and the prices other suppliers' lines carry for the same items
- Use `tool:run_calculation` with `deal.pct_change` for the index movement and the price movement over the same window, and `critic.excess_over_index` for the part of the increase above the index
- Use `tool:calc_supplier_scorecard` for price variance already seen and open findings
- Use `tool:check_policy` where accepting the increase commits spend above a band
- Use `tool:supplier_items` for the supplier's own items with their unit-price range and last date in the period, read for two periods to see whether prices moved
NOT AVAILABLE: send_message. the challenge letter is drafted as text; a person sends it
NOT AVAILABLE: build_docx. the analysis is given as text

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.index_tolerance_pct`.

## Workflow

1. Frame the increase: amount, items, effective date, reasons, and whether the contract is on record.
2. Contract position: quote the price-review clause (when prices may change, on what basis, with what notice), any index clause (series, weighting, frequency, cap or collar), any surcharge or pass-through clause. If no clause allows a change in the term, the increase is a request, not an entitlement.
3. Index test: read the series the clause or the supplier cites over the window the supplier claims; compute its movement; where the clause or the supplier's breakdown states the input's share of the price, the weighted movement is `deal.scaled_total` of the index movement and that share, and the excess is `critic.excess_over_index`; where no share is stated, show the unweighted movement and say the share is unknown. No arithmetic in the answer.
4. History test: the supplier's unit-price range for the items over two periods (supplier_items) and the range other suppliers' lines carry (similar_items); the records give ranges and last dates, not a dated series, so an earlier rise or fall is stated only where two periods show it or the user gives dated prices.
5. Reasonableness: an increase within `config.thresholds.index_tolerance_pct` of the weighted index movement is in line; above it, the excess is the challenge. Open findings and price variance from the scorecard are context.
6. Output what the evidence supports, what it does not, the questions and documents to request from the supplier (cost breakdown, index references, effective dates), and the draft challenge text; hand the negotiation to negotiation-planning.

## Frameworks and heuristics

- A contract fixes prices unless a clause says otherwise; the clause, not the supplier's letter, decides whether a change is due.
- Inflation is not a cost breakdown; the input's share of the price is what the index moves.
- Symmetry: an index that justifies an increase justified a decrease when it fell; ask for the history.
- Timing: an index that moved a year ago and has since fallen does not justify an increase today.
- Surcharges are temporary by name; ask for the removal trigger.

## Policy hooks (SC-007)

Accepting an increase that takes the committed spend above a policy band is checked with `tool:check_policy` and reported as replied. No reply is sent to the supplier and no price is changed on any record by the skill.

## Output specification

1. Frame: supplier, increase, items, effective date, reasons as stated.
2. Contract position with clauses quoted; entitlement or request.
3. Index test: series, window, movement, weighting basis, excess over index.
4. History test: price history, other suppliers' prices for the same items, earlier movements.
5. Verdict per item or group: supported, partly supported (with the excess), not supported; evidence for each.
6. Requests to the supplier and draft challenge text; hand-off to negotiation-planning.

## Quality bar

- Every index figure comes from the index tool with its series and dates.
- Every price comes from the records with its date; no market price from memory.
- The weighting is the clause's or the supplier's stated share, or is declared unknown.
- Nothing is sent or changed.

## Escalation and approval

Escalate to legal where the contract gives no right to change the price or the notice given is short of the clause, to the category lead where the excess is large, and to finance where the increase moves committed spend.

## Worked example

Asked whether a packaging supplier's surcharge citing paper prices is fair, the skill quotes the index clause (annual review, a named pulp index, paper at a stated share of the price, capped), reads the index over the window the supplier cites, finds the weighted movement is below the surcharge asked, computes the excess, finds from two periods of supplier_items that the supplier's price range did not fall when the index fell the year before (put to the supplier as a question, since the records hold ranges, not a dated series), and reports: the surcharge is partly supported up to the weighted movement, the excess is not, the clause caps it anyway, and the request to the supplier is the cost breakdown and the removal trigger.

## Anti-patterns

- Taking the supplier's inflation figure as the price movement.
- Reading the index at the peak the supplier chose rather than over the clause's window.
- Treating a surcharge as permanent.
- Sending the challenge or changing a price.
