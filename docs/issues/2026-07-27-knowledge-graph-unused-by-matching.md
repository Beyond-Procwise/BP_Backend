# The knowledge graph is not used by any matching path

**Raised:** 2026-07-27
**Severity:** Medium — a capability gap, not a defect. Nothing is broken; something is missing.
**Component:** `src/services/benchmark_live.py`, `src/services/price_outlier/detector.py`, `src/services/deal_assignment_service.py`, `src/services/linking_engine.py`
**Found by:** the benchmark price-pool work

## In plain English

We decide "is this the same product as that one?" by comparing text exactly. Two lines
match only if their description, unit and currency are character-for-character identical
after trimming and lower-casing. A knowledge graph exists in this codebase and could
answer that question far better — but no matching path consults it.

The consequence is that we under-count comparable purchases. When the pricing engine
says "not enough evidence", some of the time the evidence exists and we simply failed
to recognise it as the same thing.

## What was checked

Four graph modules exist:

```
src/services/procwise_knowledge_graph.py
src/services/procurement_kg_builder.py
src/services/platform_kg.py
src/services/kg_ingestion_service.py
```

None of them is referenced by:

| Path | Uses the graph? |
|---|---|
| Benchmark price history (`benchmark_live`) | No |
| Price-outlier detection (`price_outlier/detector`) | No |
| Deal assignment (`deal_assignment_service`) | No |
| Document linking (`linking_engine`) | No |

Matching is exact normalised string equality on `(item_description, unit_of_measure,
currency)` — see `benchmark_live._norm_item`, `_norm_uom`, `_norm_currency`.

## Why it matters, with a number

Benchmarking 400 real quote lines against the seeded corpus: **262 computed, 138 gated**.
A line gates when it has fewer than three comparable purchases.

Some of those 138 are genuinely first-time buys and should gate — that is correct
behaviour. But some almost certainly have comparable history recorded under a different
wording, and exact-string matching cannot see it. We do not currently know the split,
and that is the first thing worth measuring.

Two specific things a graph could resolve that strings cannot:

1. **The same product described differently.** "A4 Ruled Notebook, White Cover" and
   "Notebook A4 ruled (white)" are one product to a buyer and two to us.
2. **The same supplier under different names.** Relevant to deal grouping and supplier
   ranking, both of which currently narrow candidates by exact `supplier_id`.

## Suggested first step — measure before building

Do not wire the graph in yet. Establish the size of the prize:

1. Take the 138 gated lines. For each, ask the graph whether any *other* item in the
   corpus is the same product.
2. Count how many would clear the three-observation threshold if those were pooled.
3. If the answer is "a handful", exact matching is adequate and this can be closed.
   If it is "most of them", it justifies real work.

That measurement is a contained piece of work and answers whether anything further is
worth doing.

## Design caution if it does proceed

The benchmark and the outlier detector **must keep using the same match rule as each
other**. They deliberately share `_norm_item` / `_norm_uom` / `_norm_currency` today,
so a flag and a benchmark can never disagree about what counts as comparable. If graph
resolution is introduced, it has to be introduced to both through one shared path — two
independent implementations would let the detector and the engine contradict each other,
which is worse than either being imprecise alone.

Also note the pricing engine is a **pure function** with no I/O by design, and its
arithmetic is locked to an Excel workbook by a golden-fixture parity suite. Any graph
lookup must happen in the data layer that builds its inputs, never inside the engine.

## Related

- `docs/issues/2026-07-27-deal-assignment-does-not-scale.md`
- `docs/superpowers/specs/2026-07-27-benchmark-price-pool-RESULTS.md`
