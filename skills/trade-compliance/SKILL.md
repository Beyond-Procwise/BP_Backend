---
name: trade-compliance
description: "Check a cross-border shipment or supplier against the trade rules - the classification and origin the user or the supplier states, the duty rate and preferences the user supplies, export controls and sanctions screening on record, the documents the lane needs, and the landed cost effect - and say what is compliant, what is missing and who clears it. Use when asked about customs, duties, tariff codes, origin, export controls or the paperwork for an import or export. Dormant until tariff data can be read; until then codes and rates are the user's or the broker's. Not for the Incoterm (incoterms-selection) or sanctions on a supplier alone (supplier-risk-assessment)."
---

# Trade Compliance

## Purpose

Make sure nothing crosses a border on a guess: the code, the origin, the rate and the documents each have a source, the screening is on record, and the gaps go to the broker or compliance before the goods move.

## When to use and when not

Use when goods cross a border and the question is classification, duty, origin, preference, export control, sanctions or documents.

Not for the delivery term (incoterms-selection), for the landed-cost comparison of quotes (total-cost-of-ownership), for sanctions screening of a supplier alone (supplier-risk-assessment) or for the mode (mode-selection).

## Inputs and preconditions

- The goods: description, the tariff code and origin as the supplier or the user states them, value, Incoterm, lane.
- The duty rate and any preference claim from the broker or the user, labelled by source and date.

- Use `tool:screening_summary` for the supplier's sanctions and risk-event screening on record
- Use `tool:screen_sanctions` when the sanctions check alone is asked about
- Use `tool:supplier_lookup` for the supplier's country and record
- Use `tool:search_contracts` for the agreement's Incoterm, origin and compliance clauses
- Use `tool:extract_clauses` for the customs, origin, export-control and documentation clauses, quoted
- Use `tool:run_calculation` with `landed_cost.total_cost` for the landed cost with the stated duty rate, and `deal.scaled_total` for the duty amount from the value and the rate
- Use `tool:get_policy` for the trade-compliance rules (who classifies, who signs declarations, restricted goods, required screening)
NOT AVAILABLE: hs_code_lookup. no tariff database; the code is the supplier's or the broker's, labelled, never guessed
NOT AVAILABLE: duty_calc. no tariff rates; the rate is the broker's or the user's, labelled; the landed-cost calculation applies it

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.rate_change_alert_pct`.

## Workflow

1. Frame: goods, lane, Incoterm (who is importer and exporter of record), value, the sources of the code, origin and rate.
2. Classification and origin: as stated with their source; a code or origin with no source is a gap for the broker; a preference claim needs the origin evidence the rule requires, as the user states it.
3. Screening: the supplier's sanctions and risk events on record; not performed is not clear; export-control status of the goods as the user or policy states, never guessed.
4. Documents: the lane's documents from the clauses and the user's checklist (invoice, packing list, origin certificate, licences); each present, missing or unknown.
5. Cost: duty via `deal.scaled_total` from the value and the stated rate; landed cost via `landed_cost.total_cost`; a rate changing by more than `config.thresholds.rate_change_alert_pct` points since the last shipment is flagged.
6. Verdict: compliant, compliant with gaps, or blocked; the gaps with their owners (broker, compliance, supplier); nothing cleared by the skill.

## Frameworks and heuristics

- A tariff code is a legal statement; it has an owner, not a guess.
- Origin is evidence, not a label on the box.
- Not screened is not clear.
- The importer of record carries the risk; the Incoterm says who that is.

## Policy hooks (SC-007)

Trade-compliance rules come from `tool:get_policy` and are quoted; screening comes from the register and is never performed here. The skill signs no declaration and clears nothing; the broker and compliance do.

## Output specification

1. Frame with the importer and exporter of record.
2. Classification and origin with sources; preference claim and its evidence.
3. Screening and export-control status.
4. Document checklist: present, missing, unknown.
5. Duty and landed cost via the registry; verdict and gaps with owners.

## Quality bar

- Every code, origin and rate has a source or is a gap.
- Screening is read, never performed or assumed.
- Duty and landed cost come from the registry.
- Nothing is cleared or signed.

## Escalation and approval

Escalate to the customs broker for codes, rates and declarations, to compliance for screening and export controls, and to legal for a restricted-goods question.

## Worked example

Asked whether the pumps from Turkey can be imported under preference, the skill frames the lane with us as importer of record under DAP, takes the supplier's stated code and origin with their source, reads the supplier's screening (clear, dated), lists the documents the agreement's clause requires and marks the origin certificate as missing, computes duty at the broker's stated rate and the landed cost via the registry, and returns compliant with one gap for the supplier to close before shipment.

## Anti-patterns

- Guessing a tariff code.
- Treating an unperformed screening as clear.
- Claiming preference without origin evidence.
- Clearing the shipment from the skill.
