# Ranking evidence chase — design

Status: design approved in conversation 2026-10-08; written spec awaiting review.
Origin: capability audit 2026-10-08, gap "shortlist scored without price".

## 1. Intent

When the supplier-ranking agent cannot score a shortlisted supplier on a
criterion the governed decision weights call for, it must not drop that
criterion silently. It records the gap, asks the supplier for exactly that
detail, takes the answer in, re-scores, and repeats until the gap is closed or
a person takes it over. Until then the ranking is labelled provisional.

Said by the user: "there needs to be a rule that the agent keeps going back to
the Supplier to get the details needed to assess per the weightings of the
decision."

Decided by the user: the agent **prepares each request for a person to
approve**. It does not send on its own.

## 2. What exists today (read from the code, 2026-10-08)

- Weights: one active `WeightAllocationPolicy` row in `proc.bp_policy`:
  price 0.4, delivery 0.3, risk 0.2, payment_terms 0.1
  (`rules.default_weights`).
- `composite_scores` (`src/agents/supplier_ranking_agent.py`) scores each
  supplier only on criteria it holds data for and renormalises the weights over
  those. This is deliberate (a gap in our data is not a fault in the bid) and
  stays. It also means a missing criterion is invisible except in `scored_on`.
- Price is scored only deal-scoped, when two or more candidates quoted on the
  same deal. A new requirement has no such deal, so price is always absent.
- Replies: `email_watcher` matches inbound mail to a dispatch and writes
  `proc.supplier_response`, which has `price`, `currency`, `payment_terms`,
  `warranty`, `validity`, `lead_time`. **Only 7 rows exist in bp_testdb and 2
  have a price.** Parsing quality is unproven; see section 7, task 1.
- Sending: `EmailDispatchApprovalPolicy` denies any send without an approved
  `proc.bp_approval` row. Drafts live in `proc.draft_rfq_emails`.
- `EmailReplyAutonomyPolicy` already escalates `price_change` and
  `terms_change` intents to a person.

## 3. Rules versus policies (ruling 2026-09-27)

A **rule** reads facts and asserts something is the case. A **policy** answers
allowed / needs approval / forbidden for a proposed action.

- Detecting "supplier S has no price for requirement R" is a **rule**:
  new row in `proc.bp_rule`, detector slug `ranking_evidence_gap_check`,
  finding type `non_conformance`, scope `supplier_ranking`.
- Whether to ask, how often and how many times is a **policy**:
  new `bp_policy` row `SupplierInfoRequestPolicy`
  (`policy_identifier` `supplier_info_request`).
- Whether the email may go out stays with the existing
  `EmailDispatchApprovalPolicy`. This design adds nothing to that path.

## 4. Design

### 4.1 The gap rule

Input: the shortlist (supplier ids), the requirement, the governed weights.
For each supplier and each criterion with weight > 0, the criterion is a **gap**
when the scoring frame holds no value for it. Output: one gap per
(requirement, supplier, criterion). No gap is recorded for a criterion the
supplier is not asked about by policy (4.3 `askable`).

Which criteria a supplier can be asked for:

| Criterion | Askable from the supplier | Source of an answer |
|---|---|---|
| price | yes | `supplier_response.price` + `currency` |
| payment_terms | yes | `supplier_response.payment_terms` |
| delivery | yes, as lead time | `supplier_response.lead_time` |
| risk | **no** | our own assessment; recorded as a gap but never emailed |

A criterion that is not askable is shown as "not assessed" and never
chased. Nothing is invented for it.

### 4.2 Request state

New table `proc.bp_supplier_info_request` (prefix per the `bp_` convention,
indexes `ix_bp_supplier_info_request_*`):

| Column | Meaning |
|---|---|
| request_id | identity |
| requirement_id | the buyer's requirement (`bp_requirement`) |
| supplier_id | who is asked |
| criterion | price / payment_terms / delivery |
| state | `open`, `drafted`, `asked`, `answered`, `escalated`, `closed` |
| attempts | number of requests drafted for this gap |
| draft_id | latest `draft_rfq_emails` row |
| last_asked_at | when the latest request was approved and sent |
| answered_value | the value taken from the reply, with `response_id` |
| escalated_reason | why a person took over |
| created_date, last_modified_date | audit |

Unique on (requirement_id, supplier_id, criterion) so a re-run never
duplicates a gap. The migration is applied to **both** databases and ships
with a rollback.

### 4.3 The policy

`SupplierInfoRequestPolicy.details.rules` (a missing value RAISES, per the
governed-limits ruling, so none is defaulted in code):

```
askable_criteria:   ["price", "payment_terms", "delivery"]
max_attempts:       3
wait_working_days:  3
```

### 4.4 Asking

For each supplier with open askable gaps, **one** draft listing all of that
supplier's missing items, through the existing email agent. The draft states
the buyer's requirement from stated fields only. It never contains the target
budget. It is saved as an ordinary draft and enters the existing approval
step. The request moves `open -> drafted`; once the approved email is
dispatched it moves to `asked` and `attempts` increments.

### 4.5 Taking the answer in

A `supplier_response` row that matches the dispatch (existing matcher) and
carries a value for the criterion closes the gap: `asked -> answered`, then
the ranking is recomputed with that value. A reply whose value is missing or
unparsed leaves the gap `asked` and raises a review item so a person can
read the reply and enter the value. No value is inferred from prose.

### 4.6 Keep going, then stop

A scheduled job (existing backend scheduler) every working day:

- `asked` gaps older than `wait_working_days` with `attempts < max_attempts`
  go back to `open` and get a new draft (a chase), again held for approval.
- `attempts >= max_attempts` and still unanswered: `escalated`, with
  "supplier did not provide <criterion> after N requests", shown in the
  Action Centre. The ranking states this plainly.
- A gap is `closed` when its criterion is scored, or the requirement is
  cancelled.

### 4.7 Provisional ranking

The ranking output gains `evidence_status`: `complete` when every weighted
criterion is scored for every supplier, otherwise `provisional`, plus
`missing` listing criterion per supplier and the state of its request. The
score itself is unchanged. The UI/consumers can then no longer mistake a
delivery-and-risk score for a full one.

## 5. Components

| Unit | Does | Depends on |
|---|---|---|
| `services/ranking_evidence.py` | find gaps from frame + weights; `evidence_status` | scoring frame only (pure) |
| `services/supplier_info_requests.py` | state transitions and persistence | `bp_supplier_info_request`, policy |
| `agents/supplier_ranking_agent.py` | calls the above, attaches `evidence_status` | the two services |
| email agent path | drafts the request | existing, unchanged interface |
| scheduler job | chases and escalates | `supplier_info_requests` |
| migration + rule row + policy row | data | both databases |

The pure gap finder and the state machine are kept apart so each can be tested
without a database or a model.

## 6. Error handling

- Policy value missing or unreadable: raise, do not default.
- A draft that fails to save leaves the gap `open` and logged; it is retried by
  the job, never marked `drafted`.
- Reply matches more than one open gap: close only the ones whose criterion
  has a value; leave the rest.
- The job is idempotent: running it twice in a day makes no second draft.

## 7. Plan outline (detailed plan comes after spec approval)

1. **Prove reply parsing first.** Build 6 synthetic replies (price in body,
   price in attachment, terms only, vague, injection text, wrong currency),
   run them through the real watcher/parser, and report which fields come out
   correctly. This sets the honest ceiling for the whole feature. If parsing is
   weak, the design falls back to 4.5's review item for most replies and we say so.
2. Pure gap finder + `evidence_status`, with tests.
3. Migration, rule row, policy row (both DBs, with rollback).
4. State machine and persistence.
5. Drafting the request through the email agent, held for approval.
6. Reply intake and re-score.
7. Scheduler job: chase and escalate.
8. Live run on the laptop requirement; prove no draft is sent unapproved, and
   break the guard on purpose to watch it fail.

## 8. Out of scope

Automatic sending; changing the weights; the shortlist filter (shipped 8cec22e);
getting risk from the supplier; negotiation rounds.

## 9. Success criteria

- A laptop requirement's ranking reads `provisional` and lists exactly which
  supplier is missing which criterion.
- One approved-only draft per supplier is produced; nothing is sent without an
  approval.
- A reply with a price closes the gap and the ranking changes; a reply without
  one does not.
- After `max_attempts` the gap is escalated to a person, visibly.
- Re-running every step changes nothing the second time.

## Amendments found while planning (2026-10-08; these override the sections above)

1. **Rule registration (4.1, section 3).** The opportunity miner logs an ERROR and
   skips every `bp_rule` row it has no handler for. The gap rule is evaluated by the
   ranking agent per ranking, not by the sweep, so the miner is changed to skip rules
   scoped `supplier_ranking` quietly. The `bp_rule` row stays, for governance visibility.
2. **Wording of the request (4.4).** The request text is built from a fixed template
   from stated requirement fields, not written by the email agent. A model adds no
   value to a factual list of missing items, could vary it between runs, and would need
   the GPU. The draft is still an ordinary `draft_rfq_emails` row in the same approval path.
3. **Reading replies (4.5).** The existing reply parser takes the first number in the
   text as the price and does not read payment terms (see "Probe result"). This feature
   uses its own strict reader (`services/reply_values.py`) and treats
   `supplier_response.price` as untrusted. A reply naming two amounts, or an amount with
   no unit/total marker, yields no value and a review flag.
4. **How an answer reaches the score (4.5, 4.7).** Answered unit prices in one currency
   feed the existing deal-price score (ratio to cheapest, two or more bidders). Prices in
   different currencies are not compared and the ranking says so. A "total" is not
   converted to a unit price.
