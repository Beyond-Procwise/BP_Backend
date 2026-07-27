# Benchmark price pool, data corrections, and price-outlier review

**Date:** 2026-07-27
**Status:** design, awaiting approval
**Depends on:** `2026-07-24-full-scope-test-dataset-design.md` (Plan 1 complete at `1db0767`)

Three strands, in dependency order: give the benchmark engine a real price pool
(§1–4), correct four defects in how it reads the database (§5), and flag extreme
prices for human review in the Action Centre (§6).

## 1. Why

The deterministic benchmark pricing engine (`src/services/benchmark/`, shipped
2026-07-16) is complete and penny-exact against the Excel prototype — 58 tests
pass, including a golden-fixture parity suite over 54 data points. It is wired
to two live endpoints, `GET /benchmark/by-deal/{deal_id}` and
`POST /benchmark/preview`.

It has nothing to work on. The engine needs several comparable historical prices
for the same item before it will produce a number; below that threshold it fails
closed by design. In `bp_sqldb` only one deal has any deal-linked quote line
with matching history, and that history is a single observation of the deal's
own purchase order. Every line gates. The engine has never been exercised at
realistic evidence depth.

The test-data generator already produces exactly the missing raw material, as a
side effect of how it was built: a fixed 5,000-item catalogue
(`scripts/testdata/catalogue.py`) whose items are drawn repeatedly across 6,000
document chains, priced by `price_on()` at 3.8% annual drift plus a
deterministic per-item-per-date noise band of ±4%. Regenerating at seed 42
against the real taxonomy yields 21,020 quotes (115,610 lines), 5,037 purchase
orders (22,537 lines) and 11,908 invoices (53,316 lines).

The price pool the engine reads is the purchase-order and invoice lines:
**75,853 priced observations across 5,000 catalogue items, averaging about 15
per item** — comfortably above the three-point minimum the engine requires, with
three and a half years of drift between them. The 115,610 quote lines are the
things being benchmarked, not pool members.

## 2. The gap

Those documents are never written to a database. `scripts/testdata/build.py`
generates the chains, plants the defects, writes the answer key, and then
persists **only** suppliers and the supplier crosswalk (`build.py:132-133`).
All 37,965 documents and their 191,463 lines are checksummed in memory and then
discarded. `docs/testdata/BUILD_LOG.md` records this as known gap 1 and defers
persistence to a Plan 3 that does not exist yet. (The log's own counts —
44,455 documents, 193,568 lines — predate the D05 fix in `1db0767` and no longer
reproduce; the figures used throughout this spec were regenerated at that
commit.)

So the work here is not "generate price points" — the price points already
exist. It is "persist the documents that already contain them".

**Current database state (verified 2026-07-27 10:22 UTC):** `bp_testdb` holds
111 tables and zero rows; `proc.bp_supplier_id_crosswalk` is absent.
`uicanvas_test` holds `proc.supplier` at 5,000 rows plus the reference tables.
`BUILD_LOG.md` describes a fuller state that no longer exists — `bp_testdb` was
re-cloned after the recorded run and not repopulated. A clean rebuild is a
prerequisite for every acceptance criterion below.

## 3. Scope

**In scope:** persist quote, purchase-order and invoice headers and their line
items into the target tables; assign deal identifiers using the real service;
correct two generator defects that would otherwise corrupt the pool; extend
verification to cover the new data; correct four defects in how the benchmark
reads the database (§5); and flag extreme prices into the Action Centre (§6).

The last two strands touch `src/`, not just the seeder. They earn their place
here because both were found while establishing whether the pool would produce
trustworthy numbers, and shipping the pool without them would mean seeding
190,000 rows into a reader with a known currency defect.

**Out of scope, deliberately:** the four adjustment factors the engine supports
but the corpus cannot feed — specification score, service-level score, location
cost index, and a dated price index. `src/services/benchmark_live.py:32`
neutralises all four today by supplying identical values to both sides of each
comparison, and discloses that it has done so. That behaviour stays exactly as
it is, so the test database exercises the engine the same way live data does.
Volume is the only live adjustment, and quantities in the generated data are
real. Generating the other four dimensions is a separate decision with its own
spec; it would make the test database diverge from live behaviour, which is a
cost worth paying only once someone needs those factors demonstrated.

**Also out of scope:** the organisation tables, downstream rankings and
decisions, golden document files, and scenario harnesses. Those belong to Plans
2 and 3.

## 4. Design

### 4.1 `scripts/testdata/persist.py` — new module

Maps the existing `Document` and `LineItem` dataclasses from `documents.py`
onto six existing tables, and bulk-loads them through the `copy_rows` helper
already in `scripts/testdata/db.py`.

| Generated | Target table |
|---|---|
| `Document(doc_type="Quote")` | `proc.bp_quote_trgt` |
| its `lines` | `proc.bp_quote_line_items_trgt` |
| `Document(doc_type="Purchase_Order")` | `proc.bp_purchase_order_trgt` |
| its `lines` | `proc.bp_po_line_items_trgt` |
| `Document(doc_type="Invoice")` | `proc.bp_invoice_trgt` |
| its `lines` | `proc.bp_invoice_line_items_trgt` |

Note the purchase-order header table is `bp_purchase_order_trgt`, not
`bp_po_trgt` — the latter does not exist. Only `bp_po_line_items_trgt` follows
the `po` abbreviation.

Interface:

```python
COLUMNS: dict[str, tuple[str, ...]]          # table -> ordered column list
def write_chains(conn, chains: Sequence[Chain]) -> dict[str, int]
```

`write_chains` truncates the six tables before loading, so a rebuild is
idempotent. It returns rows written per table for the build log.

Every column written is populated from a field the generator already produces.
Columns with no generated source are left NULL rather than invented. The one
judgement call worth naming: organisation attribution. The header tables carry
`buyer_id`, `country` and `region` but no business-unit or cost-centre column,
so `buyer_id` takes the cost-centre identifier and `country`/`region` come from
the owning entity. The fuller organisation model has nowhere to land until Plan
3 creates it, and the seeder discards it today regardless.

`created_by` and `last_modified_by` are set to a constant `testdata` marker so
generated rows are distinguishable from anything else by query alone.

Each of the three header tables carries an `AFTER INSERT` trigger
(`trg_quote_trgt_outcome`, `trg_po_trgt_outcome`, `trg_invoice_trgt_outcome`)
that looks the document up in the corresponding `_raw` table and records an
outcome when it finds one. Loading 37,965 headers fires it 37,965 times. The
`_raw` tables stay empty in the test database, so each call finds nothing and
returns immediately, but the cost is measured during implementation rather than
assumed. The triggers are left enabled: they are part of the structure the clone
is supposed to reproduce.

### 4.2 `build.py` wiring

One call to `write_chains` after the supplier write, and its row counts added to
the printed summary.

### 4.3 Deal identifiers

`GET /benchmark/by-deal/{deal_id}` selects quote lines by `deal_id`, so without
deal assignment the endpoint returns nothing even with a full pool. (The pool
query itself does not filter by deal and needs no deal work.)

Deal assignment runs through the existing production service,
`src/services/deal_assignment_service.assign_deals()`, pointed at the test
database after loading. The seeder does not stamp `deal_id` itself. Running the
real grouping code over 6,000 synthetic chains is worth having on its own
merits.

Worth recording, because it contradicts a standing assumption: **the cloned
schema contains no deal-assignment routine.** All ten functions in `bp_testdb`
are outcome, discrepancy or process-monitor related, and none of the nine
triggers touches `deal_id`. If a deal-assigning stored procedure exists
somewhere, it is not part of what `pg_dump --schema-only` brought across from
`bp_sqldb`, so a freshly built test database will not assign deal identifiers on
its own. The Python service is the only mechanism available here.

This is the one step with genuine uncertainty. `assign_deals` derives look-back
deal identifiers from a canonical purchase order using `linking_engine` scores,
and it has never been run at this volume or against data it did not extract
itself. If it groups the chains poorly, that is a product finding about the
grouping service, reported as such — not something the seeder works around by
writing deal identifiers directly. The design document for the test dataset
takes the same position in its §13 note: a red result from a correctly built
test is a finding, not a defect in the test.

### 4.4 Two generator corrections

Both are defects in already-committed generator code that would corrupt the
pool. Both are small.

**Description collisions.** The engine pools on item description, unit of
measure and currency. Run against the real 246-leaf taxonomy, `build_catalogue`
produces 50 key combinations shared by 102 of the 5,000 catalogue items: the
description is assembled from a 10-word qualifier list, the leaf name and a
10-word noun list, giving only 100 distinct shapes per leaf for roughly 20 items
per leaf. The worst colliding pair carries base prices of £4.81 and £8,865.99 —
a 1,843-fold spread that would pool as though the two were the same product,
and would drag any benchmark built from that pool into nonsense. Fix: make the
description unique per item. This changes the generated corpus and therefore the
determinism checksum and the answer key; both are regenerated and the new
checksum recorded.

**Currency labels without conversion.** `documents.py:88` stamps the cost
centre's currency on each line, while `unit_price` comes from `price_on()`,
which returns the catalogue's untranslated sterling figure. A US cost centre
therefore records a sterling magnitude labelled USD. Benchmarking survives this,
because currency is a match key and the pools stay separate — but spend roll-up
does not, and check V05 ("FX conversions re-derive") exists to catch exactly
this. Fix: convert through `proc.bp_fx_rates`, which `reference.py` already
copies verbatim. That table is currently empty in `bp_testdb`, which the
prerequisite rebuild resolves.

### 4.5 Verification

- Extend V01 to assert document and line-item counts, not just suppliers.
- Implement V04: line totals sum to their header totals, per document type.
- Implement V05: a converted amount re-derives from the copied FX rate.
- Add one benchmark check: take the most frequently purchased catalogue item,
  run `compute_benchmark` against the loaded pool, and assert the result is not
  gated and reports HIGH confidence.

The last is the acceptance test for this work in a single line: the engine
computes where it previously gated.

### 4.6 What does not change

No file under `src/services/benchmark/` is touched — the engine's arithmetic is
verified against the workbook (§5.1) and stays exactly as it is. The four
corrections in §5 are all in `benchmark_live.py`, which is data plumbing, not
calculation. The neutralised adjustments stay neutralised and stay disclosed.

Pool size was measured rather than assumed: constructing 120,000
`BenchmarkPoint` objects takes 0.41 s, and one `compute_benchmark` call over the
full pool takes 8 ms, so a twenty-line deal costs roughly 0.6 s of engine time
plus the row fetch. No optimisation is warranted, and none is proposed —
narrowing the pool query would risk changing which points match, and the engine's
accuracy is not negotiable for a saving this size.

## 5. Corrections to how the engine reads the database

### 5.1 The arithmetic is not in question

Verified formula-by-formula against `Benchmark Calculations.xlsx`, reading the
formula strings out of the cells rather than only comparing outputs. The five
adjustment factors, their clamps (and the deliberate absence of clamps on
location and inflation), the per-column rounding digits — `2,2,4,2,2` on the
weighted profiles, including the odd 4 on location — the confidence bands and
all eleven configuration values match. The golden fixtures are independent
evidence: `scripts/export_benchmark_fixtures.py:62` loads the workbook with
`data_only=True`, so the expected values are Excel's own cached results, not our
engine's output.

Two intentional divergences, both already covered by tests:

- **Median.** The workbook's median formula reads the internal data sheet only
  and returns 0 for four of the five sample deals. Ours is a true median across
  the pooled set. Selecting the median method will therefore not reproduce the
  workbook.
- **One-off costs.** The workbook's formula adds delivery, implementation,
  support and risk once per order; the description written in that same column
  says to multiply them by quantity. The two contradict each other. We follow
  the formula, which is also what the cached values reflect. **This remains an
  open question for the business** — on a 220-unit line it is the difference
  between one delivery charge and 220. Recorded here rather than silently
  settled; the code does not change until someone decides.

### 5.2 Purchase-order currency is dropped (defect)

`load_benchmark_pool` selects `p.currency` from `bp_po_line_items_trgt` with no
join to the purchase-order header. That column is NULL on every row in the live
database, so `_norm_currency` defaults all of it to sterling. Of the 136
purchase-order lines currently in the pool, **77 are not sterling** — 73 US
dollar and 4 New Zealand dollar. The invoice half of the same query already
joins its header correctly; the purchase-order half does not.

No wrong answer is being produced today, because no dollar purchase order shares
an item description with any quote, so nothing actually pools together. It is
latent, not benign: any corpus with overlap triggers it.

Fix: join `proc.bp_purchase_order_trgt` and coalesce line currency to header
currency, mirroring the invoice branch. Note that this defect would be *masked*
rather than fixed by the test data, because §4.1 writes line-level currency on
every generated row — which is a reason to fix it in the query and to add a
regression test that pins the header fallback.

### 5.3 Missing quantities are counted as zero

`benchmark_live.py:111` coerces a NULL quantity to `0.0`. Twelve of 136
purchase-order pool rows and four of 108 invoice rows have no quantity — for
services lines, legitimately so. Counting them as zero drags down the weighted
reference quantity, which tilts the volume adjustment and makes quotes look
dearer than they are. The distortion is bounded by the 0.85–1.15 clamp but it is
real and it is silent.

Fix: exclude rows with no quantity from the reference-quantity average rather
than counting them as zero. The row still contributes its price to the
benchmark; only the volume profile ignores it. The count of such rows is added
to the response so the omission is visible.

### 5.4 A deal is benchmarked against its own documents

The pool query has no deal filter, so a deal's own purchase order and invoices
sit in the comparison set used to judge its quotes. The winning supplier is
therefore partly benchmarked against its own price — precisely the supplier most
worth scrutinising. This is disclosed today but not corrected, and the test data
makes it much worse: every generated purchase order copies the awarded quote's
lines verbatim, and every invoice copies the purchase order's.

Fix: exclude documents belonging to the deal under analysis from its own pool,
and report the excluded count. Where that drops a line below the evidence
threshold, the line gates — which is the correct outcome, not a regression.

### 5.5 Suspect prices stay in the pool, and are disclosed

For 22% of live quote lines, unit price times quantity does not equal the line
total. Most are legitimate line discounts of 5–11%. Some are extraction errors:
one line records a notebook at £11.69 each for a quantity of 100 against a
printed total of £116.90 — the unit price is ten times too high, and the
benchmark engine trusts it.

These are **not** silently dropped from the pool. We do not know which of the
three numbers is wrong, so excluding the row is a guess dressed as a
correction. Instead the response reports how many pool points carry an open
finding against them, and §6 raises the extreme cases for a human to judge.

The existing `line_total_mismatch` check already detects the within-row
inconsistency (160 findings live), so §6 does not duplicate it — §6 is the
cross-document check that no existing detector performs.

## 6. Price outliers as review checkpoints

**Requirement:** where a line's price is extreme against comparable purchases,
raise it as a checkpoint so someone can confirm whether the line is accurate,
and surface it in the Action Centre.

### 6.1 Route to the Action Centre

No new table, endpoint or screen. The Action Centre reads
`proc.bp_extraction_discrepancy`: the gateway's `getDiscrepancies` filters on
`status` alone, with no issue-type allowlist, orders critical findings first,
and resolves the owning deal by joining `proc.bp_deal_documents`. Writing a row
with `status='open'` is the whole integration. The compliance exception chart
and the deal-detail Checks list pick the new type up on the same basis.

### 6.2 `src/services/price_outlier.py` — new module

Detection is separated from the benchmark API on purpose: a GET request must not
write findings. The detector runs as a scheduled job alongside the existing
`trgt-promotion` and `deal-assignment` jobs, following the same
env-flag-plus-interval pattern in `backend_scheduler`, and is also callable on
demand.

For each priced line in the quote, purchase-order and invoice target tables, the
peer set is the pooled purchase-order and invoice history for the same
normalised item, unit and currency, excluding the line's own document. This is
deliberately the same match rule the benchmark engine uses, so a flag and a
benchmark never disagree about what counts as comparable.

**The test, and why this one.** Compare against the peer **median** and the
median absolute deviation, not the mean and standard deviation: the mean is
dragged by the very outliers being hunted, so a single ten-times-wrong price
raises the bar enough to hide itself. A line is flagged when it is both

- statistically extreme — at least 5 robust deviations from the peer median
  (`|price − median| / (1.4826 × MAD)`), and
- commercially material — at least 3× the peer median, or at most a third of it.

Both conditions must hold. The first alone flags trivia when peers are nearly
identical, where a 2% difference is arithmetically enormous; the second alone
flags genuine price variety. Where more than half the peers share an identical
price the deviation measure collapses to zero, so the ratio test stands alone in
that case.

At least 5 peers are required. The benchmark engine's own floor is 3, but a
median over three points is too thin to call a fourth one extreme.

Severity follows magnitude: 10× or more (or a tenth or less) is `critical`,
anything else `warning`. `blocks_promotion` is false — this is review signal, not
a gate, consistent with every other finding of this kind.

### 6.3 The row

| Column | Value |
|---|---|
| `doc_type` | `quote`, `purchase_order`, `invoice` |
| `doc_pk_candidate` | the header id, so the deal join resolves |
| `field_name` | `line_items[N].unit_price` |
| `issue_type` | `price_outlier` |
| `raw_value` | the line's unit price |
| `expected_value` | the peer median |
| `computed_value` | left NULL — see below |
| `severity` | `critical` or `warning` |
| `status` | `open` |
| `blocks_promotion` | false |

`computed_value` is deliberately empty. That column carries two different
conventions across the codebase — in some rows an expected value, in others a
signed delta — and the gateway reads `expected_value ?? computed_value` as the
expected figure and derives the delta itself. Populating only `expected_value`
leaves no room for the ambiguity to bite.

`notes` carries the plain-English reason, because that is what a reviewer
actually reads: *"line 3: 'A4 Ruled Notebook' at £11.69 each is 10.0× the usual
£1.17 across 18 comparable purchases since May 2023 — check the unit price and
quantity."*

Re-running the detector must not stack duplicates: a line already carrying an
open `price_outlier` finding is skipped. A resolved finding whose price later
changes can be raised again.

### 6.4 What this is not

It does not alter any price, exclude anything from the benchmark pool, or block
promotion. It raises a question for a human. The one thing it changes about the
benchmark response is the disclosure in §5.5: the count of pool points carrying
an open finding.

## 7. Testing

**Seeder.** Unit tests over a small chain fixture, no database: `write_chains`
produces one row per document and one per line; column lists match the live
schema; totals survive the mapping; the marker fields are set. Database writes
follow the integration-marked pattern already used in `tests/testdata/test_db.py`.
Regression tests for the two generator corrections: descriptions are unique
across the full 5,000-item catalogue, and a converted line re-derives from its
FX rate.

**Live-read corrections.** Each of §5.2–5.5 gets a test that fails before the
fix: a purchase-order line whose currency lives only on the header reaches the
pool in that currency; a NULL quantity leaves the reference quantity unchanged
rather than pulling it toward zero; a deal's own documents do not appear among
its matched point ids; the response reports the suspect-point count.

**Outlier detector.** Pure-function tests on the decision rule, no database: a
ten-times price among tight peers flags as critical; a 3× price among tight
peers flags as warning; a 2% deviation among identical peers does not flag
despite being statistically enormous; genuine price spread does not flag; fewer
than five peers never flags; identical peers (zero deviation) fall back to the
ratio test. Persistence tests: the row carries the header id and populates
`expected_value` not `computed_value`; a second run raises no duplicate.

## 8. Acceptance criteria

1. `python -m scripts.testdata.build --target bp_testdb --seed 42 --drop-first`
   exits 0 and `bp_testdb` holds the suppliers, reference data, documents and
   line items — verified by query, not by the build log.
2. `proc.bp_po_line_items_trgt` and `proc.bp_invoice_line_items_trgt` together
   hold at least 70,000 priced lines (75,853 at seed 42 before defect planting),
   and the median catalogue item has at least three observations in that pool.
3. No two catalogue items share description, unit of measure and currency.
4. V01, V02, V03, V04, V05 and V14 pass; the benchmark check passes.
5. `GET /benchmark/by-deal/{deal_id}` returns non-gated lines at HIGH confidence
   for at least one deal — or, if `assign_deals` fails to group the synthetic
   chains, that failure is documented with its cause and the pool is
   demonstrated directly instead.
6. Two builds at seed 42 in separate processes reproduce one checksum, and
   `BUILD_LOG.md` records the new value.
7. `bp_sqldb` and `uicanvas` are provably unchanged (V14).
8. A purchase-order line whose currency is set only on its header enters the
   pool in that currency, proven against live data where 77 of 136 pool lines
   are not sterling.
9. The benchmark response reports, per line, how many matched points were
   excluded as the deal's own documents and how many carry an open finding.
10. The outlier detector, run against the seeded database, raises
    `price_outlier` findings that appear in `GET /spendiq/discrepancies` with a
    resolved `deal_id`, and each one names its comparison in plain English.
11. Running the detector twice raises no duplicate findings.
12. The whole benchmark suite still passes unchanged — the arithmetic is not
    touched by any of this.

## 9. Risks

**`assign_deals` at volume.** Covered in §4.3. The fallback is explicit and does
not involve fabricating deal identifiers.

**Checksum and answer key churn.** The two corrections change the generated
corpus, so the recorded determinism checksum and the 7,070-instance answer key
both change. This is expected; the criterion is that the new values are
reproducible, not that they match the old ones.

**Planted defects interacting with unique descriptions.** The defect planter
mutates chains after generation. Making descriptions unique should not affect
it, but the existing defect regression tests must still pass, and the per-type
instance counts must still hit their declared targets.

**Flooding the Action Centre.** The seeded corpus is roughly 190,000 lines
against 1,065 open findings today. If the outlier thresholds are loose, a single
detector run could bury every other finding in the queue. Two mitigations: the
detector reports how many findings it would raise before it writes any, and the
first run against the seeded database is reviewed for volume and precision
before the scheduled job is enabled. If the count is implausible, the thresholds
are wrong and get tightened — a detector nobody can keep up with is a detector
nobody reads.

**Outlier findings on planted defects.** Several of the 30 planted defect types
manipulate prices deliberately. The detector should find those, and the answer
key gives us a rare chance to measure precision and recall against known truth
rather than guess at them. Worth doing as part of the first run; a detector
whose hit rate is unknown is not finished.

**The one-off cost question in §5.1 is unresolved.** Until it is settled, total
cost gap on multi-unit lines carries whichever interpretation the workbook's
formula encodes. It is stated, not hidden, and no code changes on a guess.
