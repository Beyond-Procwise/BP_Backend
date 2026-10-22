---
name: buying-channel-guidance
description: "Tell a requester the right way to buy a specific thing - catalogue, requisition and purchase order, contracted supplier, card, or a sourcing event - from the policy on channels and limits, the agreements on record and the value involved. Use when someone asks how to buy or order something, whether they can put it on a card, whether a requisition is needed, or which supplier the channel points to. Not for whether the purchase is allowed at all (procurement-policy-compliance) or for running the event the channel leads to (rfx-design)."
---

# Buying Channel Guidance

## Purpose

Answer the everyday question 'how do I buy this?' with the channel the policy and the records point to, in plain steps, so purchases land in the right place the first time. The skill advises; it raises nothing and buys nothing.

## When to use and when not

Use when a requester asks how to buy, order or pay for a specific item or service, which channel to use, whether a card or expense claim is acceptable, or whether an existing agreement already covers it.

Not for whether the purchase or approval is permitted (procurement-policy-compliance), for a one-off exception (policy-exception-request), for designing a tender (rfx-design), for choosing between quotes (quote-comparison) or for emergencies (urgent-sourcing).

## Inputs and preconditions

- What is being bought, roughly how much, how often, and by when.
- Who is asking and their cost centre, if known.

- Use `tool:get_policy` for the channel rules, card and expense limits, catalogue rules and the value bands that require a requisition, quotes or a tender
- Use `tool:search_contracts` for a live agreement that already covers the item or service
- Use `tool:similar_items` for suppliers we already buy the item from and the prices paid
- Use `tool:supplier_lookup` for the contracted supplier's record and status
- Use `tool:check_policy` for the approval the value needs on the chosen channel
NOT AVAILABLE: create_requisition. the requester raises the requisition themselves; the skill gives the steps and the supplier to name
NOT AVAILABLE: build_docx. guidance is given as text

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.quote_band_gbp`, `config.thresholds.tender_band_gbp`.

## Workflow

1. Frame: what, how much, how often, by when.
2. Read the policy for channel rules and value bands; quote the band the value falls in. Where no rule exists for the case, say so and give the default route (requisition and purchase order).
3. Check for coverage: a live agreement for the item or category, and suppliers we already buy it from with the prices paid. A covered item goes to the contracted supplier through the catalogue or a requisition against the agreement.
4. Uncovered: below the configured quote band, requisition with one quote; within it, the quotes the policy requires; above the configured tender band, a sourcing event (hand to rfx-design).
5. Card or expense only where the policy allows it for the value and category; a purchase that would need to be split to fit a limit is not split (hand to procurement-policy-compliance).
6. Give the steps: channel, supplier to name, what to attach, who approves (as the policy replied), and the lead time to expect.

## Frameworks and heuristics

- Covered before new: an agreement on record beats a new supplier even when the requester has a quote in hand.
- The value band is the whole need over the period, not the first order.
- Card and expense are for what the policy lists, not for what is convenient.
- The requester's time matters: give steps in order, not a policy lecture.

## Policy hooks (SC-007)

Channel rules, card and expense limits and the value bands come from `tool:get_policy` and are quoted. The approval for the value on the chosen channel comes from `tool:check_policy` and is reported as replied. The skill never raises a requisition, places an order or approves.

## Output specification

1. Frame: item, value, frequency, deadline.
2. Channel: the route, the policy rule quoted, and why.
3. Coverage: the agreement or supplier on record, with prices paid if present.
4. Steps in order with who approves and the expected lead time.
5. What not to do (split, card outside the rule) where relevant, with the pointer to the right skill.

## Quality bar

- The rule cited is quoted from policy; where none exists the default is named as a default.
- Any supplier named is on our records; none from memory.
- Prices paid are from the records with their dates.
- No requisition, order or approval is made.

## Escalation and approval

Escalate to the category owner where an agreement exists but the requester has reasons not to use it, to the policy owner where no channel rule covers the case, and to procurement-policy-compliance where the value or the splitting raises a compliance question.

## Worked example

Asked how to buy twelve monitors for a new team, the skill reads the policy (IT hardware goes through the catalogue; the value falls in the one-quote band), finds a live agreement with a hardware reseller and the price paid last quarter for the same model, and gives the steps: pick the model from the catalogue under the agreement, raise the requisition with the cost centre, the line manager approves as the policy replied, delivery in the agreement's lead time; a card purchase is outside the policy for this category.

## Anti-patterns

- Sending the requester to a new supplier when an agreement covers the item.
- Sizing the band on the first order rather than the need over the period.
- Suggesting a card purchase outside the policy's list.
- Raising the requisition for the requester.
