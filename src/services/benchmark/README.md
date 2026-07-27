# Benchmark Pricing Engine

Deterministic, pure-Python reproduction of the `Benchmark Calculations.xlsx`
prototype. **No LLM, no I/O, no framework** — same inputs always produce
identical outputs. Fails closed: below the evidence threshold it returns no
benchmark rather than a weak one.

## What it does (plain English)

Given one supplier quote line, it finds all comparable historical price
points (same item, unit, currency), averages them into a raw benchmark
price, then applies five multiplicative corrections so the benchmark is fair
for *this* quote — how much you're buying (volume), the spec level you asked
for, where it's delivered (location), the service level you asked for (SLA),
and price inflation since the historical prices were recorded. It then
reports how far the quoted price sits above or below that adjusted benchmark,
per unit and across the whole deal.

## Usage

```python
from services.benchmark.engine import compute_benchmark
from services.benchmark.models import (
    BenchmarkPoint, BenchmarkSettings, QuoteLine,
)

result = compute_benchmark(quote, points, location_index_table, index_table,
                           BenchmarkSettings())
```

## Formulas (field by field)

Let `M` = benchmark points matching the quote's `item_name` + `uom` +
`currency` with `include=True` (internal and external pooled, treated
identically). `n = len(M)`, `W = Σ source_weight`.

| Field | Formula | Rounding |
|---|---|---|
| `simple_benchmark` | `mean(raw_unit_price over M)` | 2dp |
| `weighted_benchmark` | `Σ(price·weight) / W` | 2dp |
| `median_benchmark` | true median of pooled prices (Decision 1) | 2dp |
| `selected_benchmark` | candidate chosen by `settings.method` | — |
| `ref_quantity` | `Σ(historical_quantity·weight)/W` | 2dp |
| `avg_spec_score` | `Σ(specification_score·weight)/W` | 2dp |
| `avg_loc_index` | `Σ(location_cost_index·weight)/W` (row-static, Decision 2) | 4dp |
| `avg_sla_score` | `Σ(sla_score·weight)/W` | 2dp |
| `avg_hist_index` | `Σ(index_value_at_price_date·weight)/W` | 2dp |
| `target_loc_index` | `location_index_table[quote.location]`, default 1.0 (audited, Decision 3) | — |
| `current_index` | `index_table[quote.index_id]`, default 1.0 (audited, Decision 3) | — |
| `volume_adjustment` | `clamp((quantity/ref_quantity)^(−elasticity), 0.85, 1.15)` | 4dp |
| `spec_adjustment` | `clamp(1 + (req_spec − avg_spec)·0.03, 0.90, 1.20)` | 4dp |
| `location_adjustment` | `target_loc_index / avg_loc_index` (no clamp) | 4dp |
| `sla_adjustment` | `clamp(1 + (req_sla − avg_sla)·0.025, 0.90, 1.25)` | 4dp |
| `inflation_adjustment` | `current_index / avg_hist_index` (no clamp) | 4dp |
| `combined_factor` | product of the five rounded factors | 4dp |
| `final_benchmark` | `selected × volume × spec × location × sla × inflation` | 2dp |
| `quoted_total` | `quoted_unit_price·qty + delivery + implementation + support + risk_premium − discount_rebate` | none |
| `benchmark_total` | same with `final_benchmark` in place of the quoted price | none |
| `unit_variance_gbp` | `quoted_unit_price − final_benchmark` (positive = quote above benchmark) | 2dp |
| `unit_variance_pct` | `unit_variance_gbp / final_benchmark` — a fraction, ×100 for display | 4dp |
| `total_cost_gap` | `quoted_total − benchmark_total` (positive = savings opportunity) | 0dp |
| `confidence` | `n==0 → "No Data"`, `< min → "Insufficient"`, `≥10 → HIGH`, `≥6 → MEDIUM`, else LOW | — |

All factor defaults live in `BenchmarkSettings` and are injectable — the
numbers above are the defaults, not constants in the code.

**Fail-closed gate (Step 0):** if `n < min_data_points` (default 3) every
computed field is `None`, `gated=True`, and no exception is raised. The
match evidence (`matched_point_ids`, counts, `total_weight`) and
`confidence` are still reported so a reviewer can see *why* it gated.

**Second fail-closed gate (zero total weight):** when `method="weighted"` and
`sum(source_weight)` across the matched set is `0`, no reliable weighted
benchmark exists, so the engine also fails closed (`gated=True`) even if
`n >= min_data_points`. Its `confidence` is reported as `"Insufficient"`
(not the n-based value) so the record never reads as gated-but-HIGH.
Consumers must always check `gated` before rendering any result — do not
infer reliability from `confidence` alone.

**Error handling:** each adjustment factor mirrors the prototype's
`IFERROR(..., 1)` — any arithmetic failure (divide-by-zero `ref_quantity`,
`avg_loc_index`, `avg_hist_index`, etc.) yields a neutral `1.0` rather than
an exception. `unit_variance_pct` is `None` if `final_benchmark` is 0.

**Rounding is part of the contract.** Each intermediate is rounded before
the next step consumes it (the five factors to 4dp, then the final benchmark
is the rounded product of those rounded factors), exactly as the workbook's
cell-by-cell `ROUND(...)`s do. All rounding is Excel-style half-away-from-
zero via `excel_round` — Python's built-in `round()` is half-to-even and
provably drifts (D002's total gap is −172.5: Excel −173, `round()` −172).

## The three resolved design decisions

1. **Median: fixed.** The workbook's `MEDIAN(IF(...))` reads only internal
   data and, not being array-entered, returns 0 for 4 of the 5 example rows.
   This engine computes the true median of the pooled internal+external
   matched set. Default method is `weighted`, so golden outputs are
   unaffected — but `method="median"` is now correct.
2. **Location index source: preserved.** The benchmark-side location profile
   averages the **static** `location_cost_index` stored on each data row;
   only the quote's target location is looked up live. Known, accepted
   limitation: editing the Location Index Table later does NOT re-price
   historical rows — refresh the row indices when reloading benchmark data.
3. **Silent defaults: surfaced.** A missing location or index lookup still
   substitutes `1.0` (numeric parity with the prototype), but every
   substitution is appended to `fallbacks_used`
   (`"location_default"` / `"index_default"`) and logged at WARNING. A
   missing lookup is evidence of a data gap; the audit record says so.

## Other prototype defects found (and how they're handled)

- **Quoted-total description bug:** the workbook column's *description* says
  `(Unit Price + … − Discount) × Quantity`, but its *formula* multiplies only
  the unit price by quantity and adds the cost adders once. The formula (and
  the golden values) win; this engine adds adders once.
- **Row-500 cap:** every workbook aggregation reads data rows 4–500 only —
  a silent-truncation trap if the sheets ever grow. Harmless today (no data
  beyond row 500). The engine takes full lists and has no such cap.
- **Gate scope:** the workbook still shows a quoted total and lookup values
  on gated rows; per the fail-closed contract this engine returns `None` for
  *all* computed fields when gated (spec explicitly overrides the prototype
  here).

## Informational-only fields

`supplier_risk_penalty`, `contract_risk_penalty`, `strategic_supplier_bonus`
are carried through to the result verbatim for governance visibility and are
**never** used in any formula (verified by test).

## Parity fixtures

`tests/fixtures/benchmark/golden.json` is exported from the workbook by
`scripts/export_benchmark_fixtures.py` (re-run only if the workbook changes,
then re-verify the parity suite). The suite validates the full D001
intermediate trace, final outputs for D002–D005, the gate, both lookup
fallbacks, and median correctness.

## Decision: one-off costs are added once (2026-07-27)

Delivery, implementation, support and risk are added once per order, and the
discount is subtracted once. They are NOT multiplied by quantity.

The workbook contradicts itself here: the formula in column AT adds them once,
while the description written beside it says to multiply by quantity. The
formula wins, for three reasons. The field names all denote one-off charges —
multiplying a negotiated £500 rebate by 220 units to reach £110,000 is not what
a rebate means. The formula is what Excel has actually been computing, so it
produced the numbers the author reviewed. And a charge that genuinely scales
with quantity is part of the unit price, not a separate charge.

It also changes less than it appears: the same amount is added to the quoted
total AND the benchmark total, so it cancels in the total cost gap. A 220-unit
line at £120 against a £102 benchmark returns a £3,978 gap with no charges,
with £1,750 added once, and with £1,750 multiplied by quantity. Only the two
displayed absolute totals move.

When these charges are eventually extracted from documents, extraction
normalises to a one-off amount: "delivery £250 per order" contributes £250
once, and "£3 per unit delivery" belongs in the unit price.

**The workbook's column description still needs correcting, by hand, in Excel.**
Do NOT edit the workbook with openpyxl: it does not evaluate formulas and drops
cached values on save, which would destroy the very numbers
`scripts/export_benchmark_fixtures.py` reads with `data_only=True` to build the
golden parity fixtures.
