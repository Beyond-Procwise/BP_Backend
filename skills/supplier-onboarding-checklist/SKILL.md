---
name: supplier-onboarding-checklist
description: Work out what is still needed before a new or re-activated supplier can be used - which master-record fields are filled or empty (legal identity, country, registration, currency, Incoterms, lead time, insurance and its expiry, certifications), which compliance checks are recorded or not performed, whether a contract or quote exists, and which approvals apply - and produce the checklist with owners. Use when asked to set up, onboard or activate a supplier, what is missing to use a supplier, or whether a supplier is ready to order from. Not for the risk verdict itself (supplier-risk-assessment) or for choosing the supplier (supplier-selection).
---

# Supplier Onboarding Checklist

## Purpose

Turn "can we start using this supplier" into a checklist with evidence: what the master record holds and what is empty, which checks compliance has recorded, what contract or quote exists, what policy requires for the first order, and who owns each open item. The skill reads and lists; it creates no supplier, records no check and approves nothing.

## When to use and when not

Use when the user wants to set up, onboard, activate or re-activate a supplier, asks what is missing before a supplier can be used or paid, or whether a supplier is ready for a first order.

Not for the risk status and clearance (supplier-risk-assessment, which this checklist points to for the compliance items), for choosing between candidates (supplier-selection), for the contract wording (contract-clause-review) or for the purchase approval itself (procurement-policy-compliance).

## Inputs and preconditions

- The supplier: name or supplier_id (or "not on record yet", in which case the checklist is the full set and the master record is a gap); what will be bought from them, where they deliver from, the expected annual value and currency.
- The organisation's onboarding requirements (mandatory fields, required checks, insurance minimums, contract-before-order rules) from `tool:get_policy`; where none is stated the configured defaults apply and are named as defaults.

Use `tool:supplier_lookup` for the master record (given a partial or similar name it lists the candidate records, which is how a near-duplicate is found; a duplicate record is a finding) and `tool:supplier_discovery` by the item or service to be bought, for records that already supply it (it takes an item, never a supplier name), `tool:screening_summary` for every compliance check recorded or not, `tool:search_contracts` for an agreement with the supplier, `tool:get_corpus_facts` with the supplier's name for quotes, orders or invoices already on file, `tool:check_policy` with the expected value for the approval route of the first order, and `tool:classify_spend` for the category of what will be bought where the taxonomy marks it.

NOT AVAILABLE: onboard_supplier. The supplier is created or activated by a person in the supplier master; the skill lists what they need.
NOT AVAILABLE: build_xlsx. Give the checklist as a text table.

## Workflow

1. Identify. Resolve the supplier; if several records match, ask which; if none matches, say the supplier is not on record and run discovery on the item and the name to catch a near-duplicate record (two records for one supplier is a finding for the master-data owner).
2. Master record. From the lookup: legal name, trading name, type, legal structure, registered country and number, country, default currency, Incoterms, lead time, insurance type, amount and expiry, certifications on file, website. Each is "filled" or "empty"; an insurance expiry in the past or within `config.thresholds.insurance_warn_days` days is "expiring"; an amount below `config.thresholds.insurance_min_amount` in the reporting currency is "below default minimum" unless policy states another.
3. Compliance. From the screening summary: each check performed or not, flagged, expired; the checklist points the compliance items to supplier-risk-assessment for the verdict and says the status is not clear until compliance records the checks.
4. Commercial basis. Contract on the register (status, dates, terms) or none; quotes, orders or invoices already on file (a supplier with invoices but no record of onboarding is a finding); the category of what will be bought as the taxonomy records it.
5. Policy. Read the onboarding requirements; map each requirement to the evidence found; the first order's approval route from `tool:check_policy` with the expected value; a contract-before-order rule, where policy states one, makes the contract a blocking item.
6. Checklist. Items in three groups: blocking (cannot order until done), required (needed before payment or within `config.thresholds.required_within_days` days), recommended; each with the evidence found, the gap, the owner (buyer, supplier, compliance, master-data, legal, finance) and what to ask the supplier for.
7. Readiness. Ready to order only when no blocking item is open; otherwise "not ready" with the blocking list. Hand the compliance verdict to supplier-risk-assessment and the first negotiation or contract to their skills.

## Frameworks and heuristics

- Empty is a fact: an empty field is listed as empty, never filled from the model's knowledge of the company.
- Compliance is recorded or not: a supplier's own assurance is a request item, not a check.
- A near-duplicate record is cheaper to catch now than to merge later; discovery runs even when a record exists.
- Insurance and certifications are evidence only with dates; expired is not held.
- Blocking items come from policy or from the inability to pay or receive (no currency, no legal identity, no bank details on file as the master record indicates by its own flag, never shown here).

## Policy hooks (SC-007)

Mandatory fields, required checks, insurance minimums and contract-before-order rules come from governed policy via `tool:get_policy`; where none is stated the configured defaults apply as defaults and the checklist says so. The first order's approval route comes from `tool:check_policy`. Creation, activation and approval are a person's acts.

## Output specification

1. Supplier identity: record found or not, how matched, near-duplicate finding if any.
2. Master-record table: field, value or "empty", status.
3. Compliance table: check, recorded or not performed, flagged, expired; pointer to supplier-risk-assessment.
4. Commercial basis: contract, documents on file, category.
5. Policy requirements mapped to evidence; approval route for the first order.
6. Checklist by group with owner and the request to the supplier; readiness verdict.

## Quality bar

- Every field value and check status comes from a tool result; no company fact from memory.
- Bank details and personal contacts are never shown or requested through the skill.
- "Not performed" is never clear; "empty" is never assumed.
- Readiness follows the blocking rule exactly.

## Escalation and approval

Escalate a near-duplicate record to the master-data owner, a flagged or unperformed check on a supplier about to be ordered from to compliance, and an insurance gap on a high-value supplier to the category lead. Activation and the first order follow their own approvals.

## Worked example

Asked to onboard Falconridge Systems 15 for pallets, the skill finds the record (exact id), lists the master fields (currency and Incoterms empty, insurance expired, no certifications), finds no check recorded in the screening summary, no contract on the register and fifteen invoices already on file (a finding: trading without onboarding), reads the policy (no onboarding rule stated: defaults named), reports the Approver route for the expected value, and produces a checklist with insurance renewal and compliance checks blocking, currency and Incoterms required, certifications recommended, each with an owner and the request to the supplier.

## Anti-patterns

- Filling an empty field from what the model believes about the company.
- Treating a supplier's own statement or website as a performed check.
- Calling a supplier ready while a blocking item is open.
- Showing or asking for bank details or personal contacts.
- Creating, activating or approving anything from the skill.
