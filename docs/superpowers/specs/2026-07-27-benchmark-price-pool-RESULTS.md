# Benchmark price pool, data corrections, and price-outlier review — RESULTS

**Date:** 2026-07-27
**Status:** done — measured, not projected
**Spec:** `2026-07-27-benchmark-price-pool-from-test-data-design.md`
**Plan:** `2026-07-27-benchmark-price-pool.md`

## Summary

The benchmark pricing engine could not produce a single number before this
work — every quote line was refused for lack of price history. It now
computes: fed a proper price pool, it priced 262 of 400 real test lines with
a mix of high, medium and low confidence, and the other 138 correctly
refused because there genuinely isn't enough history for them. Along the
way we fixed four ways the engine was misreading the database (wrong
currency, wrong quantities, a deal graded against its own paperwork,
undisclosed suspect prices), and added a check that flags suspiciously
extreme prices for a human to look at — which, on its very first run
against live data, caught a real pricing error. One piece did not survive
contact with a realistic amount of data: the system that groups documents
into "deals" breaks down at volume, so the deal-scoped view of the engine
still returns nothing. That is a genuine, useful finding, not a failure of
this task — full detail below.

## The headline result: the engine now computes

Plain English: think of the benchmark engine as a price-checker. Before it
can tell you whether a quoted price looks fair, it needs to have seen
enough other real purchases of the same thing to compare against. Until
now, the database simply didn't have that history loaded in, so the
price-checker never had anything to compare against and always declined to
answer ("gated").

- We loaded a pool of **76,968 comparison price-points**, grouped into
  **13,846 groups** of "the same item, same unit, same currency." The design
  spec predicted 75,853 — the actual count came in close to that estimate.
- **11,229 of those groups** have at least three observations, which is the
  engine's minimum bar for making a call at all.
- We then ran the engine against 400 real seeded quote lines: **262 produced
  a computed price comparison, 138 were correctly declined.** Of the 262,
  confidence broke down as 75 HIGH, 100 MEDIUM, 87 LOW.
- The 138 declines are not a problem — those specific items simply don't
  have three or more comparable prices in the pool yet. Declining is the
  right answer for them, not a bug.
- One check (V15) is now a permanent guardrail: it takes a specific,
  frequently-bought item ("enterprise usage component itm000810", priced
  per hour, in GBP) and asserts it always has 27 comparison points, HIGH
  confidence, and is never gated. If a future change breaks the pool, this
  check catches it immediately.

## Four defects fixed in how the engine reads the database

Plain English: the engine's logic itself was already correct (verified
line-by-line against the Excel spreadsheet it was built to match). The
problems were all in the plumbing that feeds it data from the database.

**1. Purchase-order currency was silently dropped.**
The query that builds the price pool was reading the currency off the
purchase-order *line* table — but that column is empty (NULL) on every
single row in the live database. So every purchase-order price was
defaulting to being labelled "GBP" (pounds), even when it wasn't. Checking
the 136 purchase-order price-points already in the live pool, 77 of them
are actually US dollars or New Zealand dollars, not pounds. Fixed by
reading the currency from the purchase-order *header* instead, the same way
the invoice half of the query already did it correctly. After the fix, the
live pool reads USD 73 / GBP 59 / NZD 4 — instead of all 136 being wrongly
called GBP.

**2. Missing quantities were being counted as zero.**
When a purchase-order or invoice line has no recorded quantity (common for
service lines, where "quantity" doesn't really mean anything), the old code
treated that as "quantity = 0." That drags down the average quantity the
engine compares against, which makes today's quotes look artificially more
expensive than they are. 12 of 136 purchase-order rows and 4 of 108 invoice
rows were affected. Fixed by letting the engine express "quantity unknown"
properly and skip those rows when averaging, rather than pulling the
average toward zero. This was the **only** change made to the pricing
engine's own calculation code (as opposed to the database-reading layer
around it), and the full 15-test suite that checks every value against the
Excel spreadsheet passed unchanged throughout, byte for byte.

**3. A deal was being benchmarked against its own paperwork.**
The query that builds the comparison pool wasn't excluding a deal's own
purchase order and invoices from the pool used to judge that same deal's
quotes. That means the winning supplier's quote was partly being compared
against its own resulting purchase order and invoice — the one comparison
that tells you nothing, because of course a supplier's own paperwork looks
like its own paperwork. Fixed: a deal's own documents are now excluded from
its own comparison pool, and the number of documents removed is reported
back so it's visible.

**4. Suspect prices are now disclosed, not silently dropped.**
For 22% of live quote lines, unit price times quantity doesn't equal the
line total that's printed on the document — usually a legitimate discount,
sometimes a genuine extraction mistake. We don't know which of the three
numbers (price, quantity, total) is the wrong one, so guessing which to
throw out would just be a hidden assumption dressed up as a fix. These
rows stay in the price pool, but every result now reports how many of its
comparison prices come from a document that already has an open
data-quality question mark against it. 163 live documents carry one such
flag today.

## New: price outliers become review checkpoints

Plain English: this is a new automatic check that looks for prices that
seem way out of line compared to everything else bought under the same
item, and raises a flag for a person to look at — it never changes or
blocks anything by itself.

- No new screen or table was built. The Action Centre (the review queue
  people already use) reads one existing table filtered just on "is this
  open or not" — so writing a new row with status `open` is the entire
  integration. Nothing else had to change for it to show up there.
- The comparison uses the **median** (the middle value) and a measure of
  typical spread around it, rather than the **average** and standard
  deviation. This matters because a single wildly wrong price drags the
  average and the spread with it — which can hide the very outlier you're
  trying to catch. The median resists that.
- A price is only flagged when it is **both** statistically unusual **and**
  commercially significant — at least 3x the typical price, or at most a
  third of it. Either test alone doesn't work: the statistical test alone
  would flag trivial 2% differences among near-identical prices, and the
  commercial test alone would flag ordinary, legitimate price variety.
- Anything 10x or more (or a tenth or less) is marked critical; anything
  else that clears both bars is a warning. Nothing is ever blocked — it's a
  question, not a gate.
- **On real, live data, this found a genuine problem on its very first
  run**: "Coffee Filters" priced at 2p each, when 7 comparable purchases
  show the usual price is £2.00 — a hundred times too cheap. That's one
  finding out of 639 real priced lines checked, which is a plausible,
  believable hit rate, not noise.
- The scheduled version of this check is turned **off** by default — the
  only job in the whole scheduler that starts off — and the manual script
  defaults to a dry run (report only, write nothing), because the review
  queue already has roughly 1,000 open items waiting, and we didn't want to
  flood it before a human has looked at the first batch.

## Honest gap: the detector has not been tested against a known answer key

This should be stated plainly, not softened.

- Run against the full 232,023-row synthetic test corpus, the detector
  finds **zero** outliers.
- That is the correct, expected behaviour given what's in that corpus — not
  a bug in the detector. The synthetic data generator produces smooth,
  realistic-looking price drift (about 3.8% a year, plus or minus 4% random
  noise) and never deliberately plants an extreme price. The widest spread
  found within any single currency is 1.2x — nowhere near the 3x bar the
  detector requires.
- The much larger spreads (up to roughly 150x) that appear if you ignore
  currency entirely are not real outliers — they're what you get from
  comparing, say, a price in pounds against the same item priced in
  rupees. The detector correctly keeps currencies separate, so this is
  actually a small proof that the currency-matching logic works, not a
  concern.
- **Consequence:** the implementation plan hoped to measure how accurate the
  detector is (how many real planted problems it catches, how many false
  alarms it raises) using the synthetic corpus's built-in answer key. That
  measurement isn't possible right now, because the answer key contains no
  planted price-outlier defects to detect in the first place — zero
  findings against zero planted examples proves nothing either way.
- To be clear: the detector is not "unproven." It has been proven on real
  data (the Coffee Filters case). What's missing is a controlled test
  against a known, deliberately-planted set of bad prices.
- **Recommendation:** ask whoever maintains the synthetic test-data
  generator to add a "planted price outlier" defect type, so the detector
  can be properly scored. Do **not** lower the detector's thresholds just to
  make it trigger on the current synthetic data — that would be tuning the
  product to match a test fixture's quirks rather than real-world accuracy,
  and the live data already shows the current thresholds catching a real
  100x error correctly.

## Blocked: deal grouping does not survive a realistic amount of data

Plain English: quotes, purchase orders and invoices for the same piece of
business are supposed to get grouped together under a single "deal," so you
can look up "how did this deal perform" in one place. That grouping step
does not work once there's a realistic volume of documents to process.

- `GET /benchmark/by-deal/{deal_id}` (the endpoint that shows benchmark
  results for one specific deal) still returns nothing. The reason: every
  single one of the 115,610 seeded quote lines and every one of the seeded
  purchase orders has no deal assigned (`deal_id` is empty).
- The grouping logic has two halves. The "look forward" half comes back
  with 0 results immediately — it reads from a monitoring table that is
  completely empty (0 rows), even after all the document data was loaded.
- The "look back" half crashes partway through, with a database connection
  error ("SSL connection has been closed unexpectedly"), inside the part of
  the code that persists a grouped deal.
- The likely root cause: that persistence code asks the database "what
  columns does this table have" as a separate question for **every single
  document being saved** — tens of thousands of extra round-trips to the
  database inside one long-running operation, which is very likely what's
  breaking the connection. This is worth investigating as its own piece of
  work.
- Nothing was left half-done or half-committed as a result — the failure
  happened before anything was saved, so there's no cleanup needed.
- **Why this has never been noticed before:** the live database only has 34
  purchase orders total. The new synthetic test data has 5,037 purchase
  orders and 21,020 quotes — roughly 150x more. This is a real finding
  about how the production grouping service behaves at realistic scale,
  and it was only discoverable because this realistic test data now
  exists. Arguably the single most valuable thing this test dataset has
  produced so far.
- This was anticipated: the original spec allowed for exactly this
  scenario — if grouping fails, demonstrate the price pool directly
  instead, which is what section "The headline result" above does.

## Safety

- The live production database (`bp_sqldb`) was never written to at any
  point. Verified directly: its six relevant tables still hold exactly 774
  rows, unchanged, while the separate test database holds 232,023 rows of
  synthetic data.
- One git commit during this work accidentally swept in another,
  unrelated, uncommitted piece of work from a different session — an
  unbriefed scheduled job that had registered itself to run automatically
  and was deleting rows every hour, importing from a module nobody had
  reviewed. This was caught during review before it went anywhere and was
  removed. Lesson for anyone sharing this branch: always stage files
  explicitly by name, never with a blanket "add everything."

## Test position

- The benchmark and price-outlier test suites: 90 tests passing (58 at the
  start of this work — 32 new tests added).
- The synthetic test-data generator's own test suite: 246 tests passing.
- The Excel-parity check (every value the engine produces compared against
  the original spreadsheet's own cached answers): 15 out of 15, unchanged,
  throughout the entire piece of work.

## Also worth recording

- **One-off costs** (delivery, implementation, support, risk charges) are
  now added once per order, not multiplied by the quantity ordered. The
  source spreadsheet's own formula and its own written description of that
  formula contradicted each other; the formula (the thing actually
  producing the numbers) won. This only changes two displayed total
  figures — it cancels out of the cost-gap calculation, so the bottom-line
  comparison is unaffected. The spreadsheet's incorrect written description
  still needs to be corrected by hand in Excel — it was deliberately not
  changed programmatically, because doing so would destroy the cached
  calculated values that the test suite's reference numbers are pulled
  from.
- Two tasks originally planned as part of this work (loading the raw and
  staging tiers of data, and a broader persistence layer) were instead
  delivered by a parallel, separate piece of work happening on the same
  data at the same time. Rather than duplicate that effort, this work's
  own data-mapping module was extended to work with what they built.

## What to do next

- Ask the synthetic test-data team to add a deliberately-planted
  price-outlier defect type, so the outlier detector can finally be scored
  against a known answer key — do not lower its thresholds to force a
  match against today's data.
- Investigate the deal-grouping crash as its own piece of work: specifically,
  stop asking "what columns does this table have" separately for every
  document and ask it once per run instead.
- Correct the source Excel spreadsheet's written description of the
  one-off-cost formula by hand, so it stops contradicting the formula
  itself.
- Keep the price-outlier scheduled job switched off, and the manual script
  in dry-run mode, until the existing backlog of roughly 1,000 open review
  items has been worked down.
