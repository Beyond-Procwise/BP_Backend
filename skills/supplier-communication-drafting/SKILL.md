---
name: supplier-communication-drafting
description: "Draft an email, letter or message to a supplier - a chaser, a query, a notice of a problem, a reply to their request, a request for documents - with the facts taken from the records (orders, invoices, contract clauses, dates) and the tone the situation calls for. Use when asked to write to a supplier, reply to them, word something or chase them. The draft is text for a person to send; the skill never sends. Not for the negotiation strategy behind a reply (negotiation-planning) or for legal notices (legal reviews those)."
---

# Supplier Communication Drafting

## Purpose

Give the user a draft they can send as it stands: correct facts with their sources, the right register, one clear ask, a deadline where one is due, and nothing that commits the organisation beyond what the user said. The skill writes; a person sends.

## When to use and when not

Use when the request is to write, draft, word, reply, respond, chase, remind, inform, notify or tell a supplier something, or to improve the tone or wording of a message to one.

Not for working out the position behind the message (negotiation-planning, price-increase-challenge, dispute-and-claims give the content), for termination notices (supplier-offboarding and legal), for tenders (rfx-design) or for internal messages.

## Inputs and preconditions

- The supplier and the purpose of the message, in the user's words.
- What the user wants to happen (the ask) and by when.
- Any facts the user has that the records do not hold, labelled user-supplied.

- Use `tool:supplier_lookup` for the supplier's record and the contact role on file (never personal contact details in the draft)
- Use `tool:invoice_reconciliation` for the order and invoice line facts and the open findings a chaser or query cites (numbers, dates, amounts)
- Use `tool:search_contracts` for the agreement the message refers to
- Use `tool:extract_clauses` for the clause wording a message relies on, quoted
- Use `tool:obligation_tracker` for the dates and obligations the message cites
- Use `tool:get_policy` for any communication rule (who may write to suppliers, confidentiality, what may not be promised)
- Use `tool:invoice_status` for an invoice's status, due and paid dates as stored ('not recorded' is said as such) when the message concerns an invoice
NOT AVAILABLE: send_message. the draft is given as text; a person sends it from their own mailbox
NOT AVAILABLE: build_docx. a letter is given as text

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.reply_days`.

## Workflow

1. Frame: supplier, purpose, ask, deadline, audience (which role at the supplier).
2. Collect the facts the message needs from the records: order and invoice references, amounts, dates, clause wording. Anything the user asserts that the records do not hold is used only as the user's statement.
3. Read the policy for communication rules; a message that would commit spend, waive a right or accept liability is flagged and left to the right skill or to legal.
4. Draft: subject; one-sentence purpose; the facts with references; the ask, singular; the deadline with `config.thresholds.reply_days` business days where none is given; what happens next; sign-off with the user's role. Register: courteous and direct by default; firm where a deadline has passed; never threatening.
5. Second version where tone matters (first chaser versus final reminder), with the difference explained.
6. List what the user should check before sending and what must not be added (bank details, personal data, commitments).

## Frameworks and heuristics

- One message, one ask; two asks get two messages or a numbered list.
- Facts carry references; the supplier can find what you mean.
- Escalate tone in steps: reminder, firm reminder, formal notice; never jump.
- Never promise what the user has not decided; never accept fault; never quote a figure the records do not hold.
- Write to the role, not the person: contact details stay out of the draft.

## Policy hooks (SC-007)

Communication rules come from `tool:get_policy` and are followed; a message that commits spend, waives a right, accepts liability or gives notice is flagged and routed to the owning skill or legal before any draft. The draft is never sent by the skill and contains no bank details or personal data.

## Output specification

1. Frame: supplier, purpose, ask, deadline, audience.
2. Facts used with their sources; user-supplied statements marked.
3. The draft (subject, body, sign-off); a second version where tone differs.
4. Pre-send checklist and the things the draft deliberately leaves out.
5. Flags: anything that needs legal, finance or another skill first.

## Quality bar

- Every reference and figure is from the records or marked user-supplied.
- No commitment, waiver or admission beyond the user's instruction.
- No personal contact details or bank details in the draft.
- The ask and the deadline are explicit.

## Escalation and approval

Escalate to legal for notices, breaches and anything that waives a right; to finance for payment commitments; to the category lead where the message changes a commercial position.

## Worked example

Asked to chase a supplier for three overdue deliveries, the skill reads the orders and their dates from the records, finds the delivery clause and its remedy wording, and drafts a firm reminder citing the three order numbers and dates, asking for a revised delivery date by a stated day, noting the clause, signed by the user's role; it gives a softer first-reminder version, and lists for the user to check that none of the three has been received since.

## Anti-patterns

- Quoting an amount or date the records do not hold.
- Accepting fault or promising a payment to smooth the tone.
- Putting someone's phone number or bank details in the draft.
- Sending the message.
