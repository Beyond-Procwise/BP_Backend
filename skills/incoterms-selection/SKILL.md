---
name: incoterms-selection
description: Choose or check an Incoterms 2020 rule and named place for a purchase or supply contract - where risk and cost transfer, who insures, who clears customs, and the landed-cost effect of each candidate rule. Use when someone asks which Incoterm to use, whether a requested one (FOB, EXW, DDP, CIF) is valid for the transport mode, or what the landed cost is under different rules. Advisory only. Not for HS classification or duty rates, or choosing a carrier.
---

# Incoterms Selection

## Purpose

Recommend one Incoterms 2020 rule and a precise named place, explain where risk and cost pass, and show the landed-cost effect using the platform's landed-cost calculation. Incoterms do not transfer title and do not set payment terms; say so when it matters.

## When to use and when not

Use when picking, checking or explaining an Incoterm, or comparing the cost of two to four candidate rules on one lane.

Not for HS codes or duty rates, carrier choice, or contract wording. Refer risk-clause conflicts to the contract review skill when it exists.

## Inputs and preconditions

Ask for anything missing that changes the answer; do not guess.

- Transport mode: sea container, sea bulk, air, road, rail or multimodal. The mode decides which rules are valid.
- Origin and destination countries, and the cargo and its value.
- Who controls freight today: do we have forwarder contracts that beat the supplier's freight?
- Who can act as exporter and importer of record, and whether the buyer is VAT registered.
- Payment method. A letter of credit needs a transport document.
- Whether our own cargo policy already covers transit.

NOT AVAILABLE: duty_calc. Ask the user for the duty rate and label it user-supplied in the landed-cost inputs.
NOT AVAILABLE: get_freight_quote. Ask for the freight and insurance amounts; label them user-supplied.
NOT AVAILABLE: hs_code_lookup. Ask for the HS code or duty rate; do not infer one.

Use `tool:build_docx` when the user asks for a Word file of the result, after the answer is written, with only what the tools returned or the user stated.

## Workflow

1. Gather the inputs above. If the mode is missing, stop and ask.
2. Filter valid rules by mode using `config.rules_any_mode` and `config.rules_sea_and_inland_waterway_only`. The sea-only rules do not fit air, road or rail, and ICC guidance calls them inappropriate for containers the seller does not load on board under its control; recommend FCA, CPT or CIP instead for containerised sea freight, and say why if the user still wants a sea-only rule.
3. State the transfer point for each remaining candidate: EXW at the seller's premises; FCA on handover to the carrier; FAS alongside the ship; FOB on board; CFR and CIF on board with the seller paying main carriage; CPT and CIP at handover to the first carrier with the seller paying carriage; DAP ready for unloading at destination; DPU unloaded at destination; DDP at destination cleared for import. The C-rules have two critical points, origin for risk and destination for cost.
4. Check clearance. The seller exports in every rule except EXW; the buyer imports in every rule except DDP. If the buyer cannot export, EXW is wrong: recommend FCA. If the seller cannot be importer of record or recover import VAT, DDP is wrong.
5. Check insurance using `config.insurance_obliged_to_seller` and `config.thresholds.insurance_minimum_pct`. Only CIF and CIP oblige the seller to insure. If our own cargo policy covers transit, prefer CPT or CFR over CIP or CIF to avoid double cover.
6. Compare cost for two to four candidate rules on the same lane. First `tool:list_calculations` with name `landed_cost.total_cost` to read its inputs, then `tool:run_calculation` once per rule. Pass the price `lines` (unit_price, quantity, currency) and `buyer_currency`, `incoterm` as the rule plus place, `cross_border`, the `components` the user gave (freight, insurance, origin and destination charges) with each marked included or stated, and `duty_rate` if the user gave one. Use only values the user supplied or a tool returned. A component the quote does not mention is reported as not stated, never as zero.
7. Recommend one rule with a named place specific enough to locate (building, street, city), the transfer point, who insures, who clears, and what would change the recommendation.
8. Call `tool:get_policy` with `preferred incoterm` and report any governed preference. If none is found, say there is no governed Incoterm policy.

## Frameworks and heuristics

- Put risk where control is: a buyer with strong forwarder contracts usually does better with FCA than with a supplier-arranged rule.
- The common mistake is FOB for containers: the seller loses control at the container yard but keeps risk until on board.
- DAP and DPU differ only in who unloads.
- Compare rules on the same lane, volume, currency and date, or the comparison means nothing.

## Policy hooks (SC-007)

Before recommending a rule for a contract above the approval limit, run the procurement-policy-compliance skill. Preferred Incoterms and mandatory forwarder or cargo-insurance programmes are policy matters; if no governed policy exists, say so.

## Output specification

A recommendation card: rule and named place with "Incoterms 2020"; transfer-of-risk point; cost-split table (seller / buyer for carriage, insurance, export clearance, import clearance, duty); insurance; clearance; landed-cost comparison with each figure marked tool result, user-supplied or not stated; confidence; assumptions; suggested contract wording.

## Quality bar

- The rule is valid for the mode.
- The named place can be located.
- Every landed-cost figure comes from `tool:run_calculation` or is labelled user-supplied.
- CIF and CIP are not confused on insurance level.

## Escalation and approval

Advisory only; nothing is written anywhere. Refer sanctions or dual-use flags to Trade Compliance. If the user wants a rule that does not fit the mode, explain why and give the better alternative; draft what they ask for only with the risk stated beside it.

## Worked example

A buyer asks for FOB Shenzhen for 40-foot containers. FOB is a sea-only rule and the containers are handed to the terminal before loading, so the seller's risk ends at a point it no longer controls. Recommend FCA Shenzhen, named terminal, Incoterms 2020: risk passes when the goods are handed to our carrier. Landed cost: run `landed_cost.total_cost` for FCA and for FOB with the freight the user gave (USD 2,400 per container, user-supplied), duty rate not supplied so duty is reported as not stated rather than zero.

## Anti-patterns

- Recommending FOB, CFR, CIF or FAS for air, road or container shipments without stating the risk-transfer gap.
- Treating a missing duty or freight figure as zero.
- Naming a country or port instead of a place.
- Saying the Incoterm decides when the buyer owns the goods or when payment is due.
- Presenting a landed-cost number that no tool produced.
