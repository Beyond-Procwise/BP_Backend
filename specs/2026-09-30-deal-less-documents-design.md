# Deal-less documents: recover what is provable, count what is not

**Status:** proposed, awaiting review
**Date:** 2026-09-30

## The problem

A document that reaches `_trgt` — the final tier, the only tier the product reads —
with `deal_id IS NULL` belongs to no deal. It is therefore absent from
`proc.bp_deal_documents`, from every deal's counts and totals, and from every screen.

Nothing reports it. Each safety net misses it for a different reason:

| Surface | Why it misses a deal-less `_trgt` document |
|---|---|
| `/promotion/review-queue` | lists documents the promotion gate **held**. These promoted cleanly; `_evaluate` returns `reason=None` and the queue skips them |
| `proc.bp_deal_orphans` | covers `po` and `invoice` only, and reads through `bp_deal_documents`, which a NULL `deal_id` already excludes from any deal |
| `deal_assignment_service` | walks `proc.process_monitor` and matches on `process_monitor_id`; rows written without one are unreachable |
| `_attach_rival_quotes` | joins on `requirement_id`, which is NULL on **all 3,395** deal-less quotes — the pass cannot fire for any of them |
| `/promotion/link-proposals` | reads `_stg` and targets documents with no `po_id`; a promoted `_trgt` row is out of its population |

The observed consequence: four quotes extracted correctly on 2026-09-23 — including
`ORB-Q-6612 (V3)` at £1,096,000, the awarded best-and-final offer on
`TESTDEAL2026072901` — were written to `proc.bp_quote_trgt` with `deal_id` NULL and
were visible nowhere for a week. `proc.bp_agent_actions` records
`promote_to_trgt / ok` for all four at `2026-09-23 11:22:58`. The promotion succeeded.
The attachment never happened, and nothing said so.

### Measured population (bp_testdb, 2026-09-30)

```
proc.bp_quote_trgt           deal_id IS NULL   3,395
proc.bp_invoice_trgt         deal_id IS NULL   2,154
proc.bp_purchase_order_trgt  deal_id IS NULL       1
                                               -----
                                               5,550
```

Recoverable by relationship math:

| Route | Count |
|---|---|
| Quote → deal holding its own base version (`X (V3)` → deal holding `X`) | **14** |
| Invoice → deal of its referenced PO | **0** (1,964 of 2,154 carry no `po_id` at all) |
| Neither | 5,536 |

The 5,536 are standalone documents — overwhelmingly seeded corpus — with no deal
partner to attach to. **They are not a linking defect and this spec does not
fabricate deals for them.** It makes them counted instead of silent.

## Scope

**In scope**

1. Attach quote revisions to the deal their own base version sits on.
2. Count only the latest round of a quote toward a deal's quote totals.
3. Report deal-less `_trgt` documents, separating the actionable from the merely standalone.

**Out of scope**

- Chasing the 5,536 standalone documents. Nothing links them because there is
  nothing to link them to; that is a question about the seeded corpus, not about
  linking.
- The stale `ORB-Q-6612` total of £1,116,000 on `TESTDEAL2026072901`, which matches
  no raw extraction (£3,371,910 / £1,103,000 / £1,096,000) and equals the deal's
  invoice total. Separate defect, separate investigation.
- Backfilling pre-fix frozen analysis records (`bp_analysis_deal` rows written
  before `c755303`). Separate work.

## Part 1 — Attach quote revisions to their base version's deal

### Why this is safe to do automatically

`ORB-Q-6612 (V3)` and `ORB-Q-6612` are not two documents that resemble each other.
They carry the same quote number and the same supplier; the suffix is a round
marker this codebase mints itself, in `context_layer.canonical_quote_revision`.
Reattaching a round to its own quote's deal is reading an identity that is already
recorded, not inferring a relationship from supplier and amount — the mechanism
behind the open mis-grouping bug, which this deliberately does not use.

### Rule

A deal-less quote `Q` is attached to deal `D` when **all** hold:

1. `base_reference(Q.quote_id)` (from `src/services/version_collapse.py`, regex
   `\s*\(\s*v(\d+).*\)\s*$`, case-insensitive) matches the base reference of at
   least one quote already on a deal.
2. Every deal-bearing quote sharing that base names exactly **one** distinct
   `deal_id`. Two candidate deals is a coin toss, not evidence — it holds.
3. `Q.supplier_id` equals the supplier on those rows. A shared quote number
   across suppliers is a collision, not a revision.

On attach, the row receives `deal_id`, `deal_name`, and
`document_id = '{deal_id}::quote::{quote_id}'`, matching the existing minting
convention.

`award_status` is **left as-is**. A later round of the deal's own quote is part of
the transaction. This is the deliberate difference from `_attach_rival_quotes`,
which sets `not_awarded` to keep a rival bid out of `bp_deal_documents`: a rival
bid is evidence about the sourcing event, a revision is the event.

### Placement

New `_attach_quote_revisions(cur) -> int` in
`src/services/deal_assignment_service.py`, called from `_run()` immediately
**before** `_attach_rival_quotes(cur)`, and returned in the run dict as
`quote_revisions_attached`.

Order matters: a revision attached first acquires a `deal_id` and is therefore
skipped by the rival pass's `deal_id IS NULL` filter — which is correct, because
it is not a rival.

### Expected effect

**14 quotes attached, across four deals.** All 14 satisfy all three conditions;
on current data none are held for ambiguity or supplier mismatch (both guards
measure 0 — they are there for the case that has not arrived yet, and the tests
force them).

| quote_id | amount | attaches to |
|---|---|---|
| `CPS-Q-3380 (V3)` | £994,000 | `TESTDEAL2026072901` |
| `NXF-2024-441 (V3)` | £1,020,000 | `TESTDEAL2026072901` |
| `ORB-Q-6612 (V2)` | £1,103,000 | `TESTDEAL2026072901` |
| `ORB-Q-6612 (V3)` | £1,096,000 | `TESTDEAL2026072901` |
| `AUR-2025-0619`, `(V2)` | £3,485,184, £1,272,000 | `DEALV3-77` |
| `LSL-Q-4408`, `(V2)` | £1,264,000, £1,210,000 | `DEALV3-77` |
| `MCP-Q-7740`, `(V2)` | £1,312,000, £1,250,000 | `DEALV3-77` |
| `COB-2025-0884`, `(V2)` | £1,486,440, £1,459,660 | `DEALV3-78` |
| `IMI-MSA-4470`, `(V2)` | £1,498,220, £1,465,920 | `DEALV3-78` |

Note the direction. On `DEALV3-77` and `DEALV3-78` the deal already holds the
**V3** of each base and the earlier rounds are the deal-less ones — the rule
matches on base reference, not on "child joins parent", so it works either way.
Those two deals therefore gain visible negotiation history and, once Part 2
applies, **no change at all to their counts or totals**, because the round that
already counted is still the surviving round. That is the intended behaviour.

## Part 2 — Only the latest round counts

### Why

Counting V1, V2 and V3 of one negotiation as three quotes inflates both
`quote_count` and `quote_total`. `collapse_versions` already exists for exactly
this reason and is documented as "a prerequisite for correct clustering"; the deal
overview never applied it. Part 1 makes the problem worse before it makes it
better — attaching the missing rounds would treble-count three negotiations.

### Change

A migration, `scripts/migrations/2026-09-30-deal-overview-quote-rounds.sql`,
redefines `proc.bp_deal_overview` so its quote aggregates keep only the highest
round per `(deal_id, base_reference)`:

- `quote_count`
- `quote_total`
- `converted_total_usd` — the quote contribution only; excluding superseded rounds
  here as well, or it would contradict `quote_total`
- `cycle_days_quote_to_po` reads `min(doc_date)` over quotes; superseded rounds are
  excluded from the deduped set, so the first activity date is taken from the
  surviving round. This is a behaviour change and is intended: the cycle is measured
  from the bid that stands.

The base reference is computed in SQL as
`regexp_replace(quote_id, '\s*\(\s*[vV][0-9]+.*\)\s*$', '')`, mirroring
`version_collapse.base_reference`. The two must not drift; the test suite asserts
they agree on a shared fixture set.

`proc.bp_deal_documents` is **not** changed. Every round stays listed, so the deal
screen still shows the full negotiation history. Only the arithmetic changes.

### Measured impact

Measured against live `bp_testdb`, not estimated, and re-measured as an acceptance
check before and after the change.

**Part 2 applied alone** (today's rows, no attachment): 2 deals, 5 rows stop
counting.

**Parts 1 and 2 applied together** — the state that will actually ship — the same
2 deals change, and no others:

| Deal | quote_count | quote_total |
|---|---|---|
| `TESTDEAL2026072901` | 5 → 3 | £5,327,600 → £3,110,000 |
| `TESTDATA_3007262026073025` | 6 → 3 | £1,164,750 → £574,396 |

`DEALV3-76`, `DEALV3-77` and `DEALV3-78` receive attached rounds from Part 1 but
their counts and totals are **unchanged**, because the round they already held is
the one that survives the dedupe. Both affected deals are test deals. **No
production deal moves.**

The two effects must be measured together, not separately: Part 1 adds rows that
Part 2 then removes from the arithmetic, so the blast radius of either in
isolation is not the blast radius of the pair. Applying Part 1 without Part 2
would treble-count three negotiations on `TESTDEAL2026072901`, taking its quote
total to £11,796,510. The two ship together.

## Part 3 — Report what is not attached

### Why a count and not a list

5,536 of the 5,550 are standalone by nature. A queue containing all of them would
bury the 14 real defects and would be ignored within a week. The product value
here is the **alarm**, not the inventory.

### Shape

New `unattached_documents(conn=None) -> dict` in
`src/services/linking_engine.py`, exposed read-only as
`GET /promotion/unattached` in `src/api/routers/promotion.py`:

```json
{
  "counts": { "quote": 3381, "invoice": 2154, "po": 1, "total": 5536 },
  "items": [
    { "doc_type": "quote", "doc_pk": "ORB-Q-6612 (V3)",
      "supplier_id": "SUP-OrbisPlatformSolutionsLtd",
      "amount": 1096000.00, "currency": "GBP",
      "candidate_deal_id": "TESTDEAL2026072901",
      "evidence": "base quote ORB-Q-6612 is on this deal" }
  ]
}
```

- `items` carries **only** documents with a provable candidate deal — the same
  rule as Part 1. Small, actionable, worth a human's attention.
- `counts` states the full remainder plainly, so the standalone population is
  visible as a number without drowning the signal. Following the precedent set by
  `considered` on `/promotion/link-proposals`, whose docstring already makes this
  argument: "a bare empty list reads as 'every document has an order' when the
  truth is that 1,964 have none."

Read-only. Nothing is written by this call.

In steady state, after Part 1 has run, `items` is empty and `counts` holds only
genuinely standalone documents. A non-empty `items` means the attach pass has
something it could not take — which is precisely the alarm that did not exist.

### Output safety

Per the withheld-routes rule, the response returns identifiers only — no route
paths, no table names in any field value.

## Files

| File | Change |
|---|---|
| `src/services/deal_assignment_service.py` | `_attach_quote_revisions()`; wired into `_run()`; new key in the run dict |
| `scripts/migrations/2026-09-30-deal-overview-quote-rounds.sql` | redefine `proc.bp_deal_overview` quote aggregates |
| `src/services/linking_engine.py` | `unattached_documents()` |
| `src/api/routers/promotion.py` | `GET /promotion/unattached` |
| `tests/test_quote_revision_attach.py` | new |
| `tests/test_deal_overview_quote_rounds.py` | new |

## Testing

Every guard is proved to fail before it is made to pass.

**`test_quote_revision_attach.py`**

1. Base `X` on deal `D`, `X (V2)` deal-less, same supplier → attached to `D`.
   Red before Part 1.
1b. **The reverse, which is the majority case on real data** (10 of the 14):
   `X (V3)` on deal `D`, bare `X` and `X (V2)` deal-less → both attached to `D`.
   The rule matches on base reference, so it must not be implemented as
   "child joins parent"; this test fails if it is.
2. Base `X` on deal `D1` **and** deal `D2`, `X (V2)` deal-less → stays unattached.
   This is the coin-toss guard; it must fail if the "exactly one deal" condition is
   removed.
3. Base `X` on deal `D` under supplier `S1`, `X (V2)` under `S2` → stays unattached.
4. Attached revision keeps `award_status` NULL — it is not marked `not_awarded`.
5. `document_id` is minted as `{deal_id}::quote::{quote_id}`.

**`test_deal_overview_quote_rounds.py`**

6. Deal with `X`, `X (V2)`, `X (V3)` → `quote_count = 1`, `quote_total` = the V3
   amount. Red before Part 2.
6b. Deal holding only `X (V3)`, with `X` and `X (V2)` newly attached by Part 1 →
   `quote_count` and `quote_total` **unchanged** from before the attach. This is
   the `DEALV3-77` case and the one that proves the dedupe keeps the highest
   round rather than an arbitrary or first row.
7. The same deal still lists **three** rows in `bp_deal_documents`.
8. Two distinct bases on one deal → `quote_count = 2`. Guards against the dedupe
   collapsing unrelated quotes.
9. The SQL base-reference expression and `version_collapse.base_reference` agree on
   a shared fixture set, including `(V3 (BAFO))`, `( v2 )` and unversioned ids.

**Endpoint**

10. A document with a candidate deal appears in `items`; a standalone one appears
    only in `counts`.

DB-backed tests require `PROCWISE_TEST_LIVE_DB=1`; without it pytest uses the fake
DB and these assert nothing. The suite runs with `CUDA_VISIBLE_DEVICES=""` and a
dead Ollama port, and never concurrently with another suite.

## Live verification

On the running local server against `bp_testdb`, not tests alone:

1. Snapshot `quote_count` / `quote_total` for **every** deal before the change,
   not just the two expected to move. Step 7 compares against this.
2. Apply the migration; run deal assignment.
3. `TESTDEAL2026072901`: 8 → 12 documents listed; quote total
   £5,327,600 → £3,110,000; count 5 → 3; `ORB-Q-6612 (V3)` at £1,096,000 present
   and attributed to Orbis Platform Solutions Ltd.
4. `TESTDATA_3007262026073025`: quote total £1,164,750 → £574,396, count 6 → 3.
5. `DEALV3-77` and `DEALV3-78`: document list grows, `quote_count` and
   `quote_total` **identical** to the snapshot. This is the check that the dedupe
   is keeping the right round rather than an arbitrary one.
6. `GET /promotion/unattached` → `items` empty, `counts.total` 5,536.
7. Re-run the deal-less census: 5,550 → 5,536, and the 14 named rows are absent
   from it. Diff the full snapshot from step 1: exactly two deals differ.

## Risks

| Risk | Mitigation |
|---|---|
| Part 2 changes numbers users have seen | Blast radius measured at 2 test deals / 5 rows, re-measured as an acceptance check; no production deal affected |
| SQL and Python base-reference logic drift | Test 9 asserts they agree on shared fixtures |
| A quote number reused across suppliers is wrongly merged | Supplier equality is a required condition; test 3 |
| Ambiguous base on multiple deals | Holds rather than guesses; test 2 |
| `items` grows large if the attach pass regresses | That is the alarm working as designed |

## Open question for review

Part 2 excludes superseded rounds from `cycle_days_quote_to_po`, so the
quote-to-PO cycle is measured from the surviving round rather than the first bid.
That is the defensible reading — the cycle is from the bid that stands — but it is a
judgement call, and measuring from the *opening* bid is equally arguable as "how
long did this sourcing event take". Flagging it rather than deciding it silently.
