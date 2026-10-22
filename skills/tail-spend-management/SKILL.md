---
name: tail-spend-management
description: "Work out how to manage the long tail of small and infrequent suppliers and low-value purchases - which to consolidate onto existing agreements, which to move to a catalogue or card channel, which to leave alone - from the spend cube, the alternatives on record and the policy on buying channels. Use when asked how to manage the tail, reduce the number of suppliers, rationalise small purchases or set up channels for low-value buying. Not for measuring the tail (spend-analysis) or for the maverick findings themselves (maverick-spend-review)."
---

# Tail Spend Management

## Purpose

Turn the measured tail into a plan: the small suppliers whose items are already bought from a contracted supplier (consolidate), the repeat low-value items that belong in a catalogue, the one-off purchases that belong on a card or a light channel, and the rest that stays. Every move names its evidence and its owner; nothing is switched off.

## When to use and when not

Use when the question is what to do about the tail: too many suppliers, small purchases everywhere, which to consolidate, which channel to give them.

Not for sizing the tail (spend-analysis), for finding off-process purchases (maverick-spend-review), for pooling demand across business units (demand-aggregation) or for choosing a channel for one purchase (buying-channel-guidance).

## Inputs and preconditions

- The scope and period, default the whole base over the last full year.
- The tail definition the user wants; otherwise the configured per-supplier share and invoice-count cuts.
- Any channels the organisation already runs (catalogue, card), from policy or the user.

- Use `tool:build_spend_cube` for spend, share and invoice count by supplier, read with order set to bottom so the smallest suppliers come first
- Use `tool:similar_items` for whether a tail supplier's items are already bought from a contracted or larger supplier
- Use `tool:search_contracts` for the agreements the consolidation targets are under
- Use `tool:screening_summary` for the status of the suppliers the tail would move onto
- Use `tool:get_policy` for the buying-channel and card rules and any supplier-count target
- Use `tool:check_policy` for any new agreement the plan proposes
NOT AVAILABLE: build_docx. the plan is given as text and tables

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.tail_cell_share_pct`, `config.thresholds.min_invoices`.

## Workflow

1. Frame the scope and read the policy for channels and targets.
2. Cut the tail from the cube read by supplier with order set to bottom (and top_n at its maximum, repeated with the period narrowed if the cell count exceeds it): a supplier whose share from the cube is below `config.thresholds.tail_cell_share_pct`, or with fewer than `config.thresholds.min_invoices` invoices in the period, is in the long tail. Report the tail's supplier count and invoice count as the cube gives them; do no arithmetic on shares.
3. For each tail supplier (or the largest group of them), look for the same items on record from a contracted supplier (similar items plus contracts). A match is a consolidation candidate; the target's screening status is read.
4. Items bought repeatedly at low value with no contracted alternative are catalogue candidates; one-off low-value purchases are card or light-channel candidates per policy.
5. What remains stays, with the reason (specialist, regulated, one-off project).
6. Plan: moves grouped by target supplier or channel, with the spend and invoice count moved, the owner and the dependency; approval routes for any new agreement.

## Frameworks and heuristics

- The tail is defined by both a supplier's share and its frequency; a supplier with three invoices and a large share is not tail.
- Consolidate onto agreements that exist before creating new ones.
- Processing cost per invoice is the lever in the tail, so invoice count moved matters as much as spend moved.
- A supplier that stays in the tail for a good reason is listed with the reason, not hidden.

## Policy hooks (SC-007)

Channel rules, card limits and any supplier-count target are read with `tool:get_policy` and quoted. A new agreement or a catalogue set-up that commits spend is checked with `tool:check_policy` and reported as replied. No supplier is deactivated, blocked or told anything by the skill.

## Output specification

1. Tail definition and its size: suppliers, spend, invoices, share.
2. Consolidation candidates: tail supplier, items, target supplier and agreement, target screening status, spend and invoices moved.
3. Catalogue and card candidates with the policy rule they fall under.
4. Suppliers that stay, with reasons.
5. Plan with owners, dependencies and approval routes.

## Quality bar

- The tail cut is stated and reproducible from the cube's own share and invoice figures.
- Every consolidation target is evidenced by items on record; none from memory.
- Channel assignments cite the policy rule or say there is none.
- No supplier is removed or blocked.

## Escalation and approval

Escalate to the category lead for consolidation targets that change a strategic relationship, to compliance where a target's screening is incomplete, and to the process owner where no channel policy exists.

## Worked example

Asked to reduce the number of suppliers, the skill cuts the tail from the cube (most suppliers, a small share of spend, a large share of invoices), finds a third of them supply items already bought from contracted suppliers, lists repeat low-value items for a catalogue, puts one-off purchases on the card channel the policy allows, and leaves a handful with reasons; the plan moves most of the invoice count with little spend and names an owner per move.

## Anti-patterns

- Defining the tail by spend alone.
- Proposing a new supplier for the tail when an agreement already covers the items.
- Deactivating suppliers from the skill.
- Counting spend moved as a saving (savings-methodology decides what counts).
