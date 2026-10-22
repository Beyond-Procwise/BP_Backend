---
name: e-auction-design
description: "Decide whether an e-auction fits a requirement and design it if so - lot structure, auction format (English reverse, Dutch, Japanese), starting price basis from the records, bid decrements, duration and extensions, qualification before bidding, what is awarded on price and what is not, and the policy on auctions. Use when asked about running a reverse auction, bidding event or online bidding, or how to structure lots and decrements. Not for the full tender design around it (rfx-design) or for who to invite (supplier-selection)."
---

# E-Auction Design

## Purpose

Make the auction decision and design honest: an auction only where the specification is settled and competition exists, with a price basis the records support, rules suppliers can trust and a plan for what price does not decide. The skill designs; it runs no auction and awards nothing.

## When to use and when not

Use when an e-auction, reverse auction, bidding event or online bidding is proposed or asked about, or when lots, decrements, auction rules or starting prices are in question.

Not for the tender documents and evaluation model (rfx-design), for choosing the invitees (supplier-selection), for the award decision after the event (award-recommendation) or for negotiation outside an event (negotiation-planning).

## Inputs and preconditions

- The requirement: items or services, volumes, duration, and whether the specification is settled.
- The suppliers qualified to bid and whether they are known to be independent.
- The organisation's auction rules, from policy.

- Use `tool:get_policy` for the auction rules (when permitted, minimum bidders, disclosure, award basis) quoted
- Use `tool:similar_items` for the prices paid for the same items, as the starting-price basis
- Use `tool:commodity_index_lookup` for the index movement behind a commodity lot
- Use `tool:supplier_discovery` for the suppliers on record for the items
- Use `tool:screening_summary` for each bidder's screening status before qualification
- Use `tool:run_calculation` with `quote.weighting_score` where price is one weighted criterion and `bids.normalise` to put bids on one basis
- Use `tool:check_policy` for the approval the event's value needs
NOT AVAILABLE: run_e_auction. the event is run on the sourcing platform by a person; the skill gives the design
NOT AVAILABLE: create_rfx. the event set-up is done by a person on the sourcing platform

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.min_bidders`, `config.thresholds.max_decrement_pct`.

## Workflow

1. Fit test: specification settled and comparable across bidders; at least `config.thresholds.min_bidders` qualified, independent bidders; value worth the event; no relationship the auction would damage beyond the saving. Any failed test is a reason not to auction, said plainly.
2. Policy: quote the auction rules; the approval route for the value.
3. Lots: group items so each lot has competition; a lot with one capable bidder is not a lot; state what is in and out of price (freight, duty, payment terms) using the bid-normalisation calculation.
4. Format: English reverse for comparable goods with several bidders; Japanese or Dutch where the user needs to protect bidder anonymity or the market is thin; justify the choice in one paragraph.
5. Price basis: the starting or ceiling price from prices paid on record and the index, with dates; no number from memory. Decrements as a share of the starting price within `config.thresholds.max_decrement_pct`; duration and extension rule stated.
6. Non-price: what is scored before bidding (qualification) and what after (weighted score); the award basis stated to bidders in advance.
7. Output the design, the rules sheet for bidders in plain words, and the risks.

## Frameworks and heuristics

- An auction prices a settled specification; an unsettled one gets a tender.
- Competition is the engine: fewer than the minimum bidders means no event.
- Tell bidders the rules once and keep them; a changed rule mid-event ends the trust.
- The starting price anchors; a wrong anchor from memory costs more than the event saves.
- Price is one criterion unless the policy says it is the only one.

## Policy hooks (SC-007)

Auction permission, minimum bidders, disclosure and award basis come from `tool:get_policy` and are quoted. The event's value is checked with `tool:check_policy` and the route reported as replied. The skill runs no event, invites nobody and awards nothing.

## Output specification

1. Fit verdict with the tests and their evidence.
2. Policy rules quoted and the approval route.
3. Lot table: items, volume, what price includes, capable bidders on record with screening status.
4. Format and justification; price basis with sources and dates; decrement, duration, extension.
5. Non-price scoring and the award basis; bidder rules sheet; risks and owners.

## Quality bar

- Every price basis is a record or an index reading with a date.
- Every bidder named is on our records with a screening status.
- The fit tests are answered with evidence, not assumed.
- Nothing is run, sent or awarded.

## Escalation and approval

Escalate to the category lead where the fit test fails on competition or relationship, to compliance where a bidder's screening is incomplete, and to legal for the bidder rules sheet.

## Worked example

Asked to run a reverse auction for pallets, the skill reads the auction policy (permitted above a value, three bidders minimum, award basis disclosed), finds four suppliers on record for the pallet items with prices paid over the year, three screened, sets one lot with freight included and payment terms fixed, proposes an English reverse format with a ceiling at the latest price paid, decrements within the configured share, a timed close with extensions, and qualification before bidding; it flags that the fourth supplier is unscreened and that two of the four share a parent on record, which the fit test must resolve.

## Anti-patterns

- Auctioning an unsettled specification.
- A starting price from memory.
- Running an event with bidders that share a parent as if they competed.
- Changing rules during the event or awarding from the skill.
