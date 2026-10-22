---
name: procurement-process-guide
description: "Explain how a procurement process works here, step by step, from the governed policy and the records - source to contract, procure to pay, what happens after a requisition, who approves at each stage, where the three-way match sits, what a finding or a hold means and who clears it. Use when someone asks how the process works, what the steps are, what happens next or why something is stuck in it. Not for whether a specific purchase is allowed (procurement-policy-compliance) or which channel to use for one item (buying-channel-guidance)."
---

# Procurement Process Guide

## Purpose

Give a plain-language walk-through of the process as this organisation actually runs it, with the governed rules quoted where they exist and the platform's own records used to say where a specific item sits. The skill explains; it approves, clears and moves nothing.

## When to use and when not

Use when the request is an explanation: how does it work, what are the steps, what happens after, why is this stuck, who does what, what does this status mean.

Not for a ruling on a specific purchase (procurement-policy-compliance), for the channel for one item (buying-channel-guidance), for an exception (policy-exception-request), for an invoice's status (invoice-query-handling) or for training material in a document.

## Inputs and preconditions

- The process or stage asked about, and the role of the asker if given.
- A specific item (requisition, order, invoice or contract) when the question is about where it sits.

- Use `tool:get_policy` for the governed rules at each stage (approval bands, required quotes, receipt rules, payment terms) quoted
- Use `tool:list_governance` for the policies and rules that exist, so the explanation names what governs and what does not
- Use `tool:invoice_reconciliation` for where a specific order or invoice sits and any finding or hold on it
- Use `tool:contract_pipeline` for where a specific contract sits in its life
NOT AVAILABLE: build_docx. the walk-through is given as text

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.max_stages`.

## Workflow

1. Frame: which process or stage, for whom, and whether a specific item is in question.
2. Read what governs: the policies and rules that exist for the stages asked about; quote the ones that apply; name the stages where nothing is governed and describe the default practice as a default.
3. Walk the stages in order with, for each, what happens, who acts, what the system records, what can stop it and who clears a stop. Keep each stage to a few sentences; `config.thresholds.max_stages` stages at most before splitting into two answers.
4. Where a specific item is in question, read its record and say where it sits, what is open on it and who acts next.
5. Point to the skill (by name, from the ones this playbook mentions) or the person for the next step; list the terms used with one-line meanings.

## Frameworks and heuristics

- Explain the process the organisation runs, not a textbook one; the policy is the source and its absence is said.
- Each stage has an actor, an input, an output and a way it can stop.
- A status is explained by what caused it and who clears it, never by its code alone.
- Plain words first, the system's term in brackets once.

## Policy hooks (SC-007)

Every rule quoted comes from `tool:get_policy` or the governance listing; stages with no governed rule are described as default practice and labelled so. The skill approves nothing, clears nothing and changes no record.

## Output specification

1. Frame: process, stage, asker's role.
2. What governs: rules quoted; stages without a rule named.
3. Stage-by-stage walk-through: what, who, record, stop, who clears.
4. Where the asked item sits, if one was given.
5. Next step and who to go to; glossary of the terms used.

## Quality bar

- Every rule is quoted from policy or labelled a default.
- Statuses are explained with cause and owner, from the record where an item was given.
- No step is invented for a stage the organisation does not run.
- Nothing is approved, cleared or changed.

## Escalation and approval

Escalate to the process owner where a stage has no governed rule and the asker needs a ruling, and to the owning skill for the next action.

## Worked example

Asked what happens after a requisition is submitted, the skill reads the approval and receipt rules, finds the approval bands governed and the receipt rule governed but no rule on quotes, and walks through: approval by the band the policy sets, purchase order issue, delivery and receipt recorded against the order, invoice matched to order and receipt, a finding raised where they differ and cleared by the buyer or finance, payment on the agreement's terms; it names the quotes stage as default practice and points to buying-channel-guidance for the requester's own purchase.

## Anti-patterns

- Describing a textbook process the organisation does not run.
- Giving a status code without its cause and owner.
- Ruling on the asker's specific purchase.
- Inventing a rule where the policy is silent.
