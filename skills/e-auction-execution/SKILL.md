---
name: e-auction-execution
description: "Carry out an e-auction that has been designed and approved - confirm the design, the qualified bidders and the approval are on record, set the event up on the platform exactly as designed, open and close it on the published times, watch it for rule breaches, record every bid and the closing position, and hand the result to the award decision. Use when asked to set up, run, open or close an auction. Dormant until an auction tool exists; until then the skill prepares the set-up sheet for a person to run it. Not for the design (e-auction-design) or the award (award-recommendation)."
---

# E-Auction Execution

## Purpose

Run the event the bidders were promised: the published rules, the published times, every bid recorded, no change mid-event, and a closing record that the award can rest on. The skill runs the event through the platform's auction action when it exists; it never extends, reopens or edits an event on its own.

## When to use and when not

Use when the auction design is approved, the bidders are qualified, and the user asks to set it up, run it, open it, monitor it or close it.

Not for deciding whether to auction or how (e-auction-design), for the award after close (award-recommendation), for inviting bidders (supplier-selection) or for a negotiation (negotiation-planning).

## Inputs and preconditions

- The approved design: lots, format, starting price basis, decrement, duration, extension rule, award basis, from e-auction-design.
- The qualified bidders with screening status and their acceptance of the rules sheet.
- The approval for the event's value.

- Use `tool:get_policy` for the auction rules (minimum bidders, disclosure, what may change after open, award basis)
- Use `tool:check_policy` for the approval the event's value needs, as replied
- Use `tool:screening_summary` for each bidder's screening; an unscreened bidder does not bid
- Use `tool:supplier_lookup` for each bidder's record
- Use `tool:build_xlsx` to produce the set-up sheet (lots, bidders, prices, times, rules) and, after close, the bid record
- Use `tool:run_calculation` with `bids.normalise` to put bids on one basis and `deal.pct_change` for the movement from the starting price
NOT AVAILABLE: run_e_auction. the auction action does not exist yet; the skill prepares the set-up sheet exactly as designed and a person runs the event on the sourcing platform
NOT AVAILABLE: send_message. bidders are not messaged by the skill; the auction action, when it exists, notifies them under the policy gate

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.record_retention_days`.

## Workflow

1. Confirm the preconditions: the approved design, at least the policy's minimum of qualified, screened, independent bidders who accepted the rules sheet, and the approval as `tool:check_policy` replied; a missing precondition stops the set-up and is named first.
2. Set up exactly as designed: lots, format, starting or ceiling price with its source, decrement, open and close times with the extension rule, award basis; produce the set-up sheet with `tool:build_xlsx` and have the user confirm it before anything opens.
3. Open on the published time through the auction action; no change to any rule after open. A rule problem after open means pause and refer, never a quiet edit.
4. Monitor: bids recorded as they land; a bid below the decrement or from an unqualified bidder flagged; extensions only as the published rule allows; no contact with a single bidder during the event.
5. Close on the published time or extension; record the closing position per lot: every bid with time and bidder, the leading bid, the movement from the starting price via `deal.pct_change`, bids put on one basis via `bids.normalise` where the design allows.
6. Hand the record to award-recommendation; the auction result is an input to the award, not the award. Keep the full bid record for `config.thresholds.record_retention_days` days at least.

## Frameworks and heuristics

- The published rules are a promise; the event keeps every one of them.
- No change after open; a problem after open is a pause and a referral.
- Every bid is a record with a time; the leading bid is not the award.
- Talk to all bidders or none during the event.

## Policy hooks (SC-007)

Auction rules come from `tool:get_policy` and are quoted; the approval comes from `tool:check_policy` as replied. The auction action runs under the agent-policy gate with the user's name; the skill never opens, extends, reopens or closes an event by any other route, never contacts a bidder, and never awards.

## Output specification

1. Precondition table: design, bidders with screening and rule acceptance, approval; stop reasons.
2. The set-up sheet's export id and its confirmation by the user.
3. The event record: open and close times, extensions, bids per lot with time and bidder, flags.
4. The closing position per lot with movement from the starting price (or the ready-to-run state while the action does not exist).
5. Hand-off to the award decision.

## Quality bar

- The event matches the approved design line for line.
- No rule changes after open.
- Every bid is recorded with its time and bidder.
- Nothing is awarded; nothing is reported as run without the action's record.

## Escalation and approval

Escalate to the category lead where a precondition fails or a rule problem appears after open, to compliance where a bidder's screening lapses, and to legal where a bidder disputes the record.

## Worked example

Asked to run the approved pallet auction, the skill confirms the design, the three screened bidders who accepted the rules sheet, and the approval; produces the set-up sheet with the lot, the ceiling at the latest price paid, the decrement and the times; has the user confirm it; and, while the auction action does not exist, reports the event as ready to run for a person on the sourcing platform, with the monitoring checklist and the bid record template, awarding nothing.

## Anti-patterns

- Opening with fewer bidders than the policy minimum.
- Extending or reopening outside the published rule.
- Messaging one bidder during the event.
- Treating the leading bid as the award.
