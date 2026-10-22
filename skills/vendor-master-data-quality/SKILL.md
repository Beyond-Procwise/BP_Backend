---
name: vendor-master-data-quality
description: "Review the supplier master records for a scope - duplicates and near-duplicates, missing or stale fields, dormant records, suppliers with spend but no onboarding, and bank-detail or contact change requests that need verification before anything changes - and say what the master-data owner should fix, merge, deactivate or verify. Use when asked about duplicate suppliers, cleaning the vendor file, record quality, or a request to change a supplier's bank or contact details. The skill changes no record and never shows bank details. Not for onboarding one supplier (supplier-onboarding-checklist)."
---

# Vendor Master Data Quality

## Purpose

Keep the supplier file trustworthy: one record per real supplier, fields complete and current, dormant records closed, and every change to payment details verified by a person through a known channel before it is made. The skill finds and proposes; the master-data owner acts.

## When to use and when not

Use when the question is about the quality of supplier records, duplicates, dormant or incomplete records, a clean-up, or a request to change a supplier's bank, address or contact details.

Not for onboarding a new supplier (supplier-onboarding-checklist), for screening (supplier-risk-assessment), for offboarding (supplier-offboarding) or for making any change to a record.

## Inputs and preconditions

- The scope: the whole file, a category, a list of suppliers, or one change request.
- For a change request: what is asked, by whom, through which channel, labelled user-supplied.

- Use `tool:supplier_lookup` for a supplier's record, status and the fields filled or empty (never bank details or personal contacts); given a partial or similar name it lists the candidate records, which is how near-duplicate names are found
- Use `tool:supplier_discovery` by item or service, for separate records that supply the same things (a second sign of a duplicate); it takes an item, never a supplier name
- Use `tool:build_spend_cube` for spend and invoice count by supplier, to find dormant records and spend on unonboarded suppliers
- Use `tool:invoice_reconciliation` for invoices from suppliers whose records are incomplete
- Use `tool:screening_summary` for records with no screening
- Use `tool:get_policy` for the master-data rules (required fields, dormancy period) and to confirm a bank-change verification procedure exists; the procedure's steps are routed to, never reproduced in an answer that could reach a supplier
NOT AVAILABLE: update_supplier_record. no record is changed; the master-data owner acts on the findings
NOT AVAILABLE: merge_suppliers. merges are the owner's act after confirmation

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.dormant_days`.

## Workflow

1. Frame the scope and quote the master-data rules: required fields, dormancy period, verification procedure for payment-detail changes.
2. Duplicates: near-duplicate names from the candidates the lookup lists for the name, and records for the same items from discovery by item; each pair listed with the evidence (name similarity, shared items, same address if on record) and a proposed survivor, for the owner to confirm; never merged by the skill.
3. Completeness: per record the required fields that are empty or stale; records with invoices but no onboarding or screening.
4. Dormancy: records with no invoices for `config.thresholds.dormant_days` days proposed for deactivation review, with the last invoice date.
5. Change requests: a request to change bank, address or contact details is never actioned from the skill; the request goes to finance and the master-data owner for the policy's verification procedure (verification through a contact already on record, never the one in the request); the answer names the owners and does not reproduce the procedure's steps; the request's channel and sender are recorded as facts. The skill shows no bank details, old or new.
6. Findings list by type with owner and the proposed action; counts per type.

## Frameworks and heuristics

- A change of bank details is a fraud attempt until verified through a known channel.
- Near-duplicates are confirmed by a person with evidence, not merged on a name score.
- Spend on a record with no onboarding is a control gap, not just a data gap.
- Dormant is a date rule from policy; the proposal is a review, not a deletion.

## Policy hooks (SC-007)

Required fields, dormancy and the payment-detail verification procedure come from `tool:get_policy` and are quoted. No record is created, changed, merged or deactivated by the skill. Bank details and personal contacts are never shown or requested; a change request is routed to the verification procedure.

## Output specification

1. Frame and rules quoted.
2. Duplicate pairs with evidence and proposed survivor.
3. Incomplete and unscreened records with the empty fields; invoices on unonboarded records.
4. Dormant records with last invoice date.
5. Change requests: routed to finance and the master-data owner for the verification procedure; the steps are not reproduced; no details shown.
6. Findings by type with counts, owners and actions.

## Quality bar

- Every finding is evidenced by a record, a spend figure or a rule.
- No bank details or personal contacts appear anywhere.
- Nothing is changed, merged or deactivated.
- Dormancy and completeness use the policy's rules or labelled defaults.

## Escalation and approval

Escalate to finance and the master-data owner for any payment-detail change, to compliance for spend on unscreened records, and to the category lead for duplicate survivors.

## Worked example

Asked to clean up the supplier file for packaging, the skill quotes the master-data rules, finds three near-duplicate pairs by name and shared items with a proposed survivor each, lists records with the required fields empty, two suppliers with invoices but no screening, four records with no invoices beyond the dormancy period, and, for a pending email asking to change a supplier's bank account, routes it to finance and the master-data owner for the verification procedure without reproducing its steps, showing no account details.

## Anti-patterns

- Actioning or displaying a bank-detail change.
- Merging records on a name score.
- Deleting dormant records rather than proposing a review.
- Requesting bank details through the skill.
