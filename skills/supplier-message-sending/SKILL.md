---
name: supplier-message-sending
description: "Carry out the sending of a message to a supplier that has been drafted and approved - confirm the draft, the recipient role on record, the sender's authority and any approval the content needs, send it through the platform's messaging action from the organisation's own mailbox, record the send, and confirm what went to whom. Use when asked to send, dispatch or transmit a message, reminder or notice to a supplier. Dormant until a sending tool exists; until then the skill hands the approved draft to a person to send. Not for writing the message (supplier-communication-drafting) or for legal notices (legal and supplier-offboarding)."
---

# Supplier Message Sending

## Purpose

Send what was approved, to the right supplier contact role, once, from the organisation's mailbox, with a record; never send what was not drafted and checked, never to a personal address the record does not hold, and never a notice that commits or waives anything without its approval.

## When to use and when not

Use when a drafted message exists and the user asks to send it, dispatch it, or transmit it to the supplier.

Not for drafting (supplier-communication-drafting), for termination or legal notices (supplier-offboarding, legal), for tender issue (rfx-issuance) or for messages to colleagues.

## Inputs and preconditions

- The approved draft (by export id or text) and the supplier it goes to.
- The sender's role and any approval the content needs (a commitment, a waiver, a notice).

- Use `tool:supplier_lookup` for the supplier's record and the contact role on file (the role, never a personal address typed by the user)
- Use `tool:get_policy` for the messaging rules (who may write to suppliers, what needs approval before sending, what may never be sent)
- Use `tool:check_policy` where the message commits spend or changes terms, for the approval it needs, as replied
- Use `tool:search_contracts` for the agreement a notice refers to and its notice clause
- Use `tool:extract_clauses` for the notice clause, quoted, when the message is a contractual notice
NOT AVAILABLE: send_message. the messaging action does not exist yet; the skill confirms the draft, recipient role and authority, and hands the message to a person to send from the organisation's mailbox

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.resend_block_hours`.

## Workflow

1. Confirm the draft exists and has not changed since it was checked; a message that commits spend, waives a right or gives contractual notice needs the approval `tool:check_policy` names, or legal's sign-off for a notice, before anything is sent.
2. Confirm the recipient: the supplier's record and the contact role on file; an address the record does not hold is not used; a personal address is never used.
3. Quote the messaging rules: who may send, what must be approved, what may never be sent (bank details, personal data, commitments outside approval).
4. Check the content once more against the rules: no bank details, no personal data beyond the role, no commitment beyond the approval, the ask and the deadline present.
5. Send once through the platform's messaging action from the organisation's mailbox, under the agent-policy gate, with the sender's name; the action returns the send record. Within `config.thresholds.resend_block_hours` hours the same message is not sent again.
6. Confirm: what was sent, to which role, when, and the record reference; what the sender watches for next (a reply, a deadline).

## Frameworks and heuristics

- A message is sent once; a second send is a new decision.
- The record's contact role is the recipient; a typed address is not.
- A notice is sent the way the clause says, or it is not a notice.
- What the message commits, the approval must cover.

## Policy hooks (SC-007)

Messaging rules come from `tool:get_policy` and are quoted; a message that commits spend or changes terms needs the approval `tool:check_policy` names. The messaging action runs under the agent-policy gate with the sender's name, from the organisation's mailbox only; the skill never sends by any other route, never to an address the record does not hold, and never reports a send the action did not confirm.

## Output specification

1. Draft confirmed (export id or text) and its approval status.
2. Recipient: supplier and contact role on record.
3. Rules quoted and the content check.
4. The send record: time, role, reference (or the ready-to-send state while the action does not exist).
5. What the sender watches for next.

## Quality bar

- No send without a checked draft and the approval its content needs.
- The recipient is the contact role on record.
- No bank details or personal data in what is sent.
- A send is reported only with the action's record.

## Escalation and approval

Escalate to legal for any notice, to finance for any payment commitment, to the approver the policy names for a commitment, and to the master-data owner where the contact role on record is missing.

## Worked example

Asked to send the approved reminder about three overdue deliveries, the skill confirms the draft from supplier-communication-drafting is unchanged and commits nothing, reads the supplier's record for the contact role, quotes the messaging rules, checks the content holds no bank or personal data, and, while the messaging action does not exist, hands the message to the user to send from the organisation's mailbox, recording the ready state and the deadline to watch.

## Anti-patterns

- Sending to an address the user typed.
- Sending a commitment the approval does not cover.
- Sending a notice by a route the clause does not allow.
- Reporting a message as sent without the action's record.
