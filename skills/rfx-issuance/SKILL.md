---
name: rfx-issuance
description: "Carry out the issue of an RFI, RFQ or RFP that has already been designed and approved - confirm the design, the invitee list and the approval are on record, assemble the pack from the stored documents, issue it to the invited suppliers through the platform, record the issue and the deadline, and confirm what went out to whom. Use when asked to issue, send out or publish a sourcing event. Dormant until an issuing tool exists; until then the skill explains what a person issues and from where. Not for designing the event (rfx-design) or choosing invitees (supplier-selection)."
---

# RFx Issuance

## Purpose

Make issuing an event a checked act: nothing goes out without a designed pack, a confirmed invitee list, the approval the value needs and a record of the issue. The skill carries out the issue through the platform's issuing action when that action exists; it never issues by any other route.

## When to use and when not

Use when the design and the invitees are settled and the user asks to issue, send out, publish or launch the event.

Not for the design (rfx-design), the invitee choice (supplier-selection), the evaluation (award-recommendation) or an auction (e-auction-execution).

## Inputs and preconditions

- The event (its design, pack documents and deadline) as the user names it.
- The invitee list with each supplier's screening status.
- The approval record for the event's value, if the policy requires one.

- Use `tool:get_policy` for the issuing rules (who may issue, minimum invitees, required pack contents, notice to invitees)
- Use `tool:check_policy` for the approval the event's value needs, as replied
- Use `tool:supplier_lookup` for each invitee's record and status
- Use `tool:screening_summary` for each invitee's screening status; an unscreened invitee is not issued to
- Use `tool:search_contracts` for any framework or agreement the event must sit under
- Use `tool:build_docx` to assemble the cover letter and instructions to bidders as a file for the pack, from the stored design
NOT AVAILABLE: create_rfx. the issuing action does not exist yet; the skill assembles the pack and the invitee list, records what is ready, and a person issues the event from the sourcing screen
NOT AVAILABLE: send_message. invitations are not sent by the skill; the issuing action, when it exists, notifies invitees under the policy gate

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.clarification_window_days`.

## Workflow

1. Confirm the preconditions: the design exists (criteria, weights, timetable, pack contents as rfx-design left them), the invitee list is final, every invitee's screening is performed and clear, and the approval for the value is on record as `tool:check_policy` replied. Any missing precondition stops the issue and is named first.
2. Quote the issuing rules from policy: minimum number of invitees, required pack contents, the notice period invitees get, who may issue.
3. Assemble the pack: the stored design documents plus a cover letter and instructions to bidders built with `tool:build_docx`, with the deadline, the clarification window of `config.thresholds.clarification_window_days` days and the submission route.
4. Issue through the platform's issuing action, once per event, to the confirmed invitees only, with the deadline and the pack attached; the action is gated by the agent-policy gate and recorded with the user's name.
5. Record and confirm: the event reference, the issue time, the invitees it went to, the deadline, and the clarification window; the confirmation lists exactly what went out and to whom, and nothing it cannot prove.
6. After issue: no change to the pack except through a published clarification to all invitees; the skill records clarifications and never answers a bidder privately.

## Frameworks and heuristics

- Issue once, to everyone on the list, at the same time; a late invitee is a new issue decision.
- An unscreened invitee is not an invitee.
- The deadline is a date and a time in the invitees' time zone, stated in the pack.
- A clarification goes to all invitees or to none.

## Policy hooks (SC-007)

Issuing rules come from `tool:get_policy` and are quoted; the approval for the value comes from `tool:check_policy` as replied. The issuing action runs under the agent-policy gate (tool.call.before) and is recorded with the user's name; the skill never issues by any other route and never claims an issue the action did not confirm.

## Output specification

1. Precondition table: design, invitees with screening status, approval; stop reasons if any.
2. Issuing rules quoted.
3. The pack contents and the built cover letter's export id.
4. The issue record: reference, time, invitees, deadline, clarification window (or the ready-to-issue state while the action does not exist).
5. Open clarifications and who answers them.

## Quality bar

- No issue without every precondition on record.
- Every invitee named is screened and on the final list.
- The confirmation states only what the issuing action returned.
- No private answers to bidders.

## Escalation and approval

Escalate to the category lead where an invitee fails screening after the list was approved, to legal for the instructions to bidders, and to the approver where the value or the invitee count changed since approval.

## Worked example

Asked to issue the packaging RFQ, the skill confirms the design and timetable from rfx-design, the five invitees with clear screening, and the approval on record; quotes the policy's minimum of three invitees and the required pack contents; builds the cover letter with the deadline and the clarification window; and, while the issuing action does not exist, reports the event as ready to issue with the pack and list for a person to issue from the sourcing screen, recording nothing as sent.

## Anti-patterns

- Issuing to an invitee whose screening is not performed.
- Changing the pack after issue without a clarification to all.
- Reporting the event as issued when the action did not confirm it.
- Answering one bidder's question privately.
