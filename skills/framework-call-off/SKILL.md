---
name: framework-call-off
description: "Guide a purchase under an existing framework or panel agreement - which framework and lot covers the need, whether a direct award or a mini-competition is required under its rules and our policy, the call-off terms and limits, the suppliers on the lot and their status, and the approval the value needs. Use when asked whether something can be called off a framework, how to run a mini-competition, or what the framework allows. Not for setting up a new framework (rfx-design) or for a one-off purchase outside any agreement (buying-channel-guidance)."
---

# Framework Call-Off

## Purpose

Keep call-offs inside the framework's own rules and ours: the right lot, the right route (direct award or competition), the right terms, the right approval, with every rule quoted from the agreement or the policy. The skill guides; it awards nothing and places no order.

## When to use and when not

Use when a framework, panel, dynamic purchasing arrangement or call-off agreement is in play and the question is whether and how to buy under it.

Not for designing a new framework (rfx-design), for the award decision after a mini-competition (award-recommendation), for a purchase no agreement covers (buying-channel-guidance) or for the contract wording of the call-off (contract-clause-review).

## Inputs and preconditions

- The need: items or services, value, duration, location.
- The framework the user has in mind, if any.

- Use `tool:search_contracts` for the frameworks, panels and call-off agreements on the register (title, supplier, dates, value as stored; lots are not a stored field)
- Use `tool:extract_clauses` for the call-off procedure, direct-award conditions, mini-competition rules, limits and term, and the lot wording (containing 'lot'), quoted; lot membership comes from these clauses or from the user, labelled
- Use `tool:obligation_tracker` for the framework's expiry and the last date a call-off may be placed
- Use `tool:supplier_lookup` for each supplier the lot clause or the user names, one at a time, for its status
- Use `tool:screening_summary` for each lot supplier's screening status
- Use `tool:get_policy` for the organisation's rule on direct awards and mini-competitions
- Use `tool:check_policy` for the approval the call-off value needs
- Use `tool:build_docx` when the user asks for a Word file of the result, after the answer is written, with only what the tools returned or the user stated
NOT AVAILABLE: create_rfx. the mini-competition is issued by a person on the sourcing platform; the skill gives the design

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.direct_award_limit_gbp`.

## Workflow

1. Frame the need and find the frameworks on the register whose scope and lots cover it; quote the scope wording. No framework covers it means buying-channel-guidance.
2. Quote the framework's call-off procedure: when a direct award is allowed, when a mini-competition is required, the ranking or rotation rule if any, the limits per call-off and in total, the expiry and last call-off date.
3. Apply our policy: a direct award is allowed only where both the framework and the policy allow it; a call-off above `config.thresholds.direct_award_limit_gbp` goes to mini-competition unless the framework's own rule is stricter.
4. Suppliers on the lot as the framework's lot clause or the user names them (the register stores no lot list), each looked up for status and screening; a supplier whose screening is incomplete is not called off from.
5. Route: direct award with its justification under the quoted condition, or mini-competition with the criteria the framework allows, the suppliers to invite (all on the lot unless the framework says otherwise), the timetable and the award basis.
6. Terms: the call-off contract form the framework prescribes; the approval route for the value as the policy replies.

## Frameworks and heuristics

- The framework's rules are the contract; our policy sits on top, and the stricter applies.
- A direct award needs the framework's stated condition to be met, written down.
- A mini-competition invites every capable supplier on the lot unless the framework says otherwise.
- The last call-off date is a hard stop; a call-off after expiry is a new contract.
- The framework's call-off terms, not the supplier's quote terms, govern.

## Policy hooks (SC-007)

Direct-award permission and competition rules come from `tool:get_policy` and the framework's clauses, both quoted; the stricter applies. The call-off value is checked with `tool:check_policy` and the route reported as replied. The skill awards nothing, invites nobody and places no order.

## Output specification

1. Frameworks that cover the need, with scope wording and lots.
2. Call-off procedure quoted: direct award conditions, competition rules, limits, expiry.
3. Route decision with the rule that decides it (framework or policy).
4. Suppliers on the lot with status and screening.
5. For a mini-competition: criteria, invitees, timetable, award basis; for a direct award: the justification.
6. Call-off terms and the approval route.

## Quality bar

- Every rule is quoted from the framework or the policy, and the stricter one is applied.
- Every supplier named comes from the framework's lot wording or the user, labelled, with its recorded status and screening.
- Expiry and limits are from the tracker, with dates.
- Nothing is awarded, issued or ordered.

## Escalation and approval

Escalate to legal where the framework's rules are unclear or the direct-award condition is doubtful, to compliance where a lot supplier is unscreened, and to the approver the policy names for the value.

## Worked example

Asked whether a facilities repair can be called off the maintenance framework, the skill finds the framework and the lot covering the location, quotes the call-off clause (direct award to the top-ranked supplier below a limit, mini-competition above it), reads the expiry and the last call-off date, finds three suppliers on the lot with two screened, applies our policy's stricter limit, concludes a mini-competition is required at this value, and lists the criteria the framework allows, the two screened invitees with a note on the third, the timetable and the approval route.

## Anti-patterns

- Direct awarding where the framework's condition is not met.
- Calling off after the framework's last call-off date.
- Using the supplier's quote terms instead of the framework's call-off terms.
- Awarding the mini-competition from the skill.
