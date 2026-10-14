# Formula Registry — Gap Report

**Date:** 2026-09-05 · **Branch:** Development @ `2e27c58`
**Companion:** [`formula-inventory.md`](./formula-inventory.md) — F-numbers below refer to it.

Assessment of the current codebase against the eight target-spec items, then the two
sections the brief requires: suspected formula defects (found, deliberately **not** fixed)
and corrections applied / outstanding.

---

## 1. Spec-item assessment

| # | Spec item | State | Evidence |
|---|---|---|---|
| 1 | Single `@formula(...)` decorator registering pure functions in an in-memory registry | **PARTIAL** | see §1.1 |
| 2 | `inputs` is a typed contract — each input a term with unit and allowed range; `output` declares type and unit | **MISSING** | see §1.2 |
| 3 | Every call validates before evaluating; contract failure returns `UNASSESSED`, never zero/None/default, and emits a finding | **PARTIAL** | see §1.3 |
| 4 | Every evaluation writes an audit-spine record (name, version hash, inputs, provenance IDs, output, tri-state confidence, evaluated_at) | **MISSING** | see §1.4 |
| 5 | Golden vectors beside each formula; module refuses to import if any fails; CI runs them | **PARTIAL** | see §1.5 |
| 6 | Callers never invoke formula functions directly; they call `evaluate` / `evaluate_many`; batch writes one aggregated record | **MISSING** | see §1.6 |
| 7 | `model_inventory()` generates a report from the registry | **MISSING** | see §1.7 |
| 8 | ADR recording what is deferred and the triggers for building each | **MISSING** | see §1.8 |

### 1.1 Decorator + registry — PARTIAL

**What exists.** `src/services/linking_engine.py` carries a working in-memory registry with a
registration API and a named-handler dispatch table:

- `PROFILES: dict[str, dict]` (`:265`) — profile name → `{p0, alpha, floor, signals, date_field}`
- `register_profile(name, profile)` (`:283`)
- `register_signal(kind, fn)` (`:277`) → `_EXTRA_SIGNALS` (`:274`), dispatched by
  `_signal_match` (`:289-315`)

Four profiles are registered across three modules: `invoice_po` and `quote_po` inline
(`:265-271`), `quote_rival` from `requirement_similarity.py:110`, `invoice_duplicate` from
`duplicate_invoice_detector.py:149`. Nine signal kinds are registered.

An adjacent second registry exists **on an unmerged branch only** —
`worktree-conformance-phase1` adds `src/engines/detector_registry.py`
(`DETECTOR_REGISTRY: dict[str, DetectorSpec]` over 11 opportunity detectors) and
`src/engines/rule_book.py`. Neither is on Development.

**What is missing.** No decorator. No version. No `effective_from`. No `owner`/`purpose`.
No vocabulary binding. Registration is by explicit call, so a formula is only in the registry
if some module happens to have been imported. The 50 formulas outside `linking_engine`'s
profile model (F12–F61) are in no registry at all.

**Assessment.** The seam is right; the metadata is entirely absent. Extend, do not replace.

### 1.2 Typed input contract with units and ranges — MISSING

No formula anywhere declares a contract. Inputs are `dict`, `pd.DataFrame`, `pd.Series` or
loose positional floats. Nothing declares a unit; nothing declares an allowed range.

The nearest existing thing is `src/services/benchmark/models.py`, where `QuoteLine`,
`BenchmarkPoint` and `BenchmarkSettings` are Pydantic models with types and some field
constraints — but no units and no ranges. `price_outlier/rule.py:21` `OutlierSettings` uses
`Field(default=5, ge=2)` and `Field(default=3.0, gt=1.0)`, which is a range on a *setting*,
not on an input.

The consequence is visible in the inventory: `risk_score` is consumed on a 0–1 scale (F20),
a 0–100 scale (F30) and a coerce-either scale (F52). A declared unit would have made that
a load error rather than three quiet conventions.

There is a **vocabulary** to bind terms to, and it is not GPSS. `src/services/facts/concept_codes.py`
derives the valid concept set at import time from `extraction_schemas/*.yaml`. GPSS does not
exist in this project; `docs/remediation/00_seam_map.md` §B3 records the decision (2026-08-07)
and the reasoning. See §4, Decision D1.

### 1.3 Validate-then-evaluate, `UNASSESSED` on failure — PARTIAL

**What exists — and it is more than nothing.** Three modules already implement exactly this
discipline under different names, and one of them shipped specifically to fix a defect the
default-value habit caused:

| Module | Sentinel | Meaning |
|---|---|---|
| `facts/arithmetic.py` + `facts/models.py:68` | `ArithmeticState.UNTESTABLE_QUANTITY_ONE`, `UNTESTABLE_MISSING_INPUT` | "could not be checked" is a distinct state from "checked and passed". Docstring: *"abstaining is not the same as passing"* |
| `facts/fx.py:35` | `FX_UNAVAILABLE` | refuses to return a rate rather than defaulting to 1.0 |
| `negotiation_advice/grounding.py:12` | `not_applicable` (of `ready` / `groundwork` / `not_applicable`) | a play whose precondition cannot hold |
| `supplier_ranking_agent.py:2031-2042` | `np.nan` | *"Unmeasured, not zero. A 0.0 here asserts 'we measured this and it was the worst possible'"* |
| `benchmark/engine.py:156` | `gated=True`, all computed fields `None` | fail-closed below the evidence threshold |
| `linking_engine` comparators | status `"MISSING"` with `q=0` | evidence absent contributes nothing rather than counting against |

**What is missing.** There is no shared sentinel, no pre-evaluation validation step, and no
finding emitted on contract failure. And the discipline is *not* universal — the same
calculation defaults where its sibling refuses:

- `quote_comparison_agent.py:757` — `entry["weighting_score"] = 0.0` when no weight applies.
  Supplier ranking's identical calculation returns `NaN` here, deliberately (F35/F37).
- `quote_evaluation_agent.py:858,867` — non-numeric weight → `0.0`.
- `opportunity_miner_agent.py:1400` — unparseable risk → `0.0`, i.e. *no risk*. Fail-open.
- `risk_intelligence_service.py:52-55` — missing metrics default to the **best possible**
  values (`on_time=1.0`, `quality=1.0`, `anomaly=0.0`), so an unmeasured supplier scores safe.
- `supplier_ranking_agent.py:2025` — categorical mapping miss → `mapping.get("default", 0)`.
- `opportunity_miner_agent.py:4799` — no discount rate on file → invent `× 0.05`.

Six places where a missing input currently produces a number that reads as a measurement.

### 1.4 Audit-spine record per evaluation — MISSING

**No formula in the codebase writes an evaluation record.** Verified by tracing every
`record_action` call site.

An audit spine exists and is healthy: `proc.bp_agent_actions`, written through
`src/services/agent_actions.py:102` `record_action`. Live schema (read from the RDS cluster,
`bp_testdb`, 2026-09-05):

```
action_id bigint, created_at timestamptz, deal_id text, document_id text, doc_pk text,
doc_type text, process_monitor_id integer, trace_id text, phase text, action_type text,
agent text, field_name text, status text, summary text, details jsonb,
confidence numeric, pipeline_version text
```

Row counts by `action_type` show it records **decisions**, not evaluations:
`promote_held` 1 897, `discrepancy` 794, `governed_reasoning` 320, `staging_sweep` 223,
`grounding_gate` 139, `persist` 135, `promote_to_trgt` 69.

`linking_engine` is the only formula module that writes here at all (`:755, 765, 817, 829, 1062`)
— and it logs the *promotion decision*, carrying `F` inside `details`. A `score_link` call from
`deal_clustering` or `duplicate_invoice_detector` writes nothing.

Missing columns for the spec's record: `formula_name`, `version_hash`, resolved `inputs`,
`provenance_ids`, and a tri-state `confidence` (the existing column is `numeric`).
`proc.bp_model` exists but is an LLM-provider registry (3 rows) and is unrelated.

**No tri-state confidence type exists anywhere.** `FactProvenance.confidence` and
`CommercialFact.confidence` (`facts/models.py:148,249`) are `Optional[float]`.

### 1.5 Golden vectors, import-time enforcement, CI — PARTIAL

**What exists.** Two genuine golden-vector sets:

- `tests/fixtures/benchmark/golden.json` — drives `tests/test_benchmark_parity.py`, pinning
  F16 to the `Benchmark Calculations.xlsx` prototype **to the penny**. `excel_round`
  (`engine.py:34`) exists solely to hold that parity. This is the strongest example in the
  codebase and the template the rest should follow.
- `tests/fixtures/deal_clustering/golden_batch.py` — F8's `p0=0.03 / alpha=0.55` were
  *calibrated against it*, with the tuning reasoning recorded at
  `requirement_similarity.py:93-108`.

**What is missing.** Both live under `tests/`, not beside their formula. Neither is checked at
import time — a mis-edit to `alpha` loads fine and only fails under pytest. And there is
**no CI at all**: no `.github/workflows/`, no `.pre-commit-config.yaml`, no git hooks.
`pytest.ini` declares only two markers.

38 of the 61 formulas have no test pinning their arithmetic (see the inventory's Tests column).

### 1.6 `evaluate` / `evaluate_many` — MISSING

Every formula is called directly. Sample of call sites:

- `benchmark_live.py:274` → `compute_benchmark(...)` per quote line
- `price_outlier/detector.py:196` → `assess(...)` per line, inside a loop
- `duplicate_invoice_detector.py:218` → `score_pair(...)` per candidate pair
- `deal_clustering.py:22` → `scorer(...)` per bid pair
- `deal_clustering.py:232` → `awarded_po_scored(...)` per bid
- `advisor.py:148,233`, `negotiation_agent.py:7566` → `rank_plays(...)`
- `negotiation_agent.py:1510` → `plan_counter(...)`
- `api/routers/benchmark.py:56` → `compute_benchmark(...)`

Two partial abstractions exist: `linking_engine._signal_match` dispatches comparators by
kind, and `deal_clustering` accepts an injectable `scorer=` parameter. Neither validates,
versions or audits.

No batch entry point exists anywhere. Six hot loops would become per-item `evaluate` calls
if migrated naively (listed in the inventory §3).

### 1.7 `model_inventory()` — MISSING

Nothing generates a report from any registry. No formula declares an owner or a purpose.
`docs/` contains no model inventory. The closest artefact is `src/services/benchmark/README.md`,
a hand-written formula table for F16 — accurate, but hand-maintained and covering 1 of 61.

### 1.8 ADR — MISSING

`docs/adr/` does not exist. Architectural decisions are recorded, but as design specs and
seam maps under `docs/superpowers/specs/` and `docs/remediation/` — for example
`00_seam_map.md` §B3 (the GPSS decision), which *is* an ADR in everything but name and
location.

---

## 2. Suspected formula defects

**Recorded, not fixed.** Behaviour is unchanged. Each is a candidate for a separate, deliberate
change with its own before/after numbers.

**D-1 — `PredictiveRiskModel.evaluate` reads the wall clock.**
`risk_intelligence_service.py:38` — `now = datetime.now(timezone.utc)` drives the exponential
signal decay `0.5 ** (age_hours / half_life)`. The same supplier, the same signals, evaluated a
week apart, returns a different score with no input having changed. It is not reproducible, not
back-testable, and a stored score cannot be recomputed to check it. As-of time must be an input.

**D-2 — Missing performance metrics default to the *best* possible values.**
`risk_intelligence_service.py:52-55` — `on_time_delivery_rate` and `quality_score` default to
`1.0`, `anomaly_index` to `0.0`. A supplier we hold no performance data for scores as a perfect
performer. The failure is fail-open on a *risk* model: absence of evidence becomes evidence of
safety. `supplier_ranking_agent.py:2031-2042` fixed exactly this class of bug in its own module
and documented why; this module never got the same treatment.

**D-3 — `quote_comparison_agent` scores an unmeasured quote as the worst.**
`quote_comparison_agent.py:757` — `entry["weighting_score"] = 0.0` when no metric weight
applies. This is the identical calculation to `supplier_ranking_agent`'s composite (F35+F37),
which returns `NaN` for this case, with the in-code reasoning: *"That punishes gaps in OUR data
as if they were faults in THEIR bid."* The two agents disagree about the same quote.

**D-4 — Unparseable risk becomes zero risk.**
`opportunity_miner_agent.py:1397-1400` — `_normalise_risk_score` returns `0.0` on
`TypeError`/`ValueError`. `risk_score` is stored as `VARCHAR` in `proc.bp_supplier`
(recorded at `negotiation_advice/signals.py:118-124`), so a malformed value is a live
possibility, and it silently reads as the safest supplier on the table. It then multiplies
into the finding weightage at `:2183`.

**D-5 — A 5% saving is invented when no discount rate is on file.**
`opportunity_miner_agent.py:4797-4799` — when the fallback path is taken and the computed
saving is `<= 0`, the code substitutes `total_spend × 0.05`. The number has no source. It is
then persisted to `bp_opportunity` as `financial_impact_gbp` and rolled into the dashboard
savings pipeline, where it is indistinguishable from an evidenced figure.

**D-6 — The separation stage of the relationship model is a no-op, but the model is used for
selection.** `linking_engine.py:398-399` pins `S = 1.0` with the comment *"single explicitly-
referenced candidate (1:1)"*. That holds for the promotion path. It does **not** hold for
`deal_clustering.awarded_po_scored` (`:76-85`), which scores a bid against **every** PO and
takes the best. A bid that matches four POs at F=81 and a bid that matches exactly one PO at
F=81 are reported with identical confidence, when the second is far stronger evidence.
Separation is precisely the stage that would distinguish them.

**D-7 — Four incompatible amount-tolerance conventions.**
`linking_engine.cmp_numeric_tol` (linear decay to 0 at 10% drift, default tol 1%),
`extraction/completeness.py:34,49` (5% **or** £1.00), `three_way_match.py:32-33`
(1% **or** £0.01), `reconciliation.py:28-30` (1% **or** £1.00, env-tunable). The same two
numbers can agree in one module and conflict in another. None references the others.

**D-8 — Four unrelated quantities are all called "confidence".**
F18 (count of evidence points), F43 (proportion of schema fields filled), F28 (distance from a
classification bar), F12 (minimum pairwise correlation). This is not cosmetic: the comment at
`linking_engine.py:36-53` and `:340-357` records that using F43 as an evidence-quality proxy
made the promotion gate **mathematically unreachable by a perfect match** (F = 75.5 against a
gate of 80), so no invoice could promote and no deal could form. The naming collision caused it.

**D-9 — `cluster_confidence` returns 100.0 for a singleton.**
`deal_clustering.py:60-61` — a cluster of one has no pair to measure, and the function returns
maximum confidence. The docstring says callers treat singletons separately, which is a
convention, not an enforcement. A caller that does not is handed a confident number derived
from no evidence.

**D-10 — Data-fitted constants are compiled into source.**
`negotiation_advice/classification.py:20-27` — `high_spend = 98175.0` and
`many_alternatives = 93` are documented as the live p90 and per-deal median. They are
distribution parameters living in code; when the corpus moves they are silently stale, and
nothing measures the drift. The docstring calls them *"governed data rather than constants"*,
which is the correct intent — but they are constants.

**D-11 — The same bar is defined twice.**
`classification.py:27` `many_alternatives: 93` and `signals.py:110`
`THIN_MARKET_ALTERNATIVES = 93`. Two modules, one number, no shared definition.

**D-12 — `_combine_weight_values` sums whatever it is handed.**
`quote_evaluation_agent.py:861-889` — if none of `price/delivery/risk/value` is present it
falls through to summing **every** remaining numeric key except `tenure`/`volume`. A weights
dict carrying an unrelated numeric field silently inflates the total.

**D-13 — The vector-similarity bonus is unpinned.**
`supplier_ranking_agent.py:2002` — `final_score = final_score * (1 + similarity_score * 0.1)`
mutates the composite after it is computed, is guarded only by a bare `except`, and no test
pins it. Whether it fires depends on whether embeddings were available, so the same frame can
score two ways.

**D-14 — Benchmark's confidence ladder is the one thing not injectable.**
`benchmark/engine.py:86-96` — every other constant in the module is on `BenchmarkSettings`;
the `HIGH ≥ 10 / MEDIUM ≥ 6 / LOW` ladder is hardcoded. Inconsistent with the module's own
stated design.

---

## 3. Corrections applied / outstanding

Phases 2–4 have now run. This section records what changed, what did not, and why.

### 3.1 Applied

| ID | What | Where | Numeric effect |
|---|---|---|---|
| A-1 | **`PredictiveRiskModel.evaluate` takes an explicit `as_of`.** It read `datetime.now()` internally, so the same supplier and the same signals gave a different score a week later. Now a parameter, defaulting to now. | `risk_intelligence_service.py:31` | **None.** Every existing caller omits it and gets the identical number. What changes is that an evaluation can now pin the time it was measured from. Discharges C-4. |
| A-2 | **`composite_scores` extracted from `SupplierRankingAgent.run()`.** The flagship supplier composite lived inline in a 2,600-line method and could not be named, versioned or tested on its own. | `supplier_ranking_agent.py` | **None** — moved verbatim; `run()` calls it. 35 existing tests pass unchanged. |
| A-3 | **`finding_weight_factor` and `normalise_weightages` extracted from `OpportunityMinerAgent.run()`.** Same reason. | `opportunity_miner_agent.py` | **None** — moved verbatim. 70 existing tests pass unchanged. |
| A-4 | **Six batch paths converted to `evaluate_many`.** Price-outlier sweep, duplicate-invoice sweep, rivalry pairwise matrix, benchmark deal run, supplier composite, opportunity weightage shares. Each now writes ONE audit record per sweep rather than one per item. | 6 modules | **None.** Discharges C-3. |
| A-5 | **`deal.pct_change`, `deal.weighted_unit_price`, `deal.realised_savings` routed through `evaluate`.** `realised_savings` gives one name to a quantity previously computed twice (`efficiency_score`; dashboard `savings`). | `deal_analysis_service.py` | **None** for the deal-analysis path. The dashboard is untouched — see O-4. |
| A-6 | **`benchmark_preview` returns 422 with the reason on a refused contract**, rather than an empty result that reads like "no benchmark exists". | `api/routers/benchmark.py` | **None** for valid payloads. A previously-crashing payload now gets a 422. |
| A-7 | **Inventory correction.** F55 and F58 were reported as duplicate percentage-change implementations. They are not: F58 divides by `abs(prev)`, rounds to whole percent, returns a string and maps a zero baseline to `"+0%"`/`"+100%"`. Corrected in place in the inventory. | `formula-inventory.md` §2 | Documentation only. |

**47 formulas registered by this work; 50 in the registry as of writing** (three were
added by a concurrent workstream — see §3.4). **154 golden vectors, all reproducing.**
57 new tests in
`tests/services/formulas/` pass, plus the existing suites of every migrated module.

### 3.1a Regression evidence

`tests/services` was run twice: once on this working tree, once on a **detached git
worktree at `2e27c58`** (the commit this work started from) so the comparison is a
real before/after rather than a recollection.

| | Failed | Passed | Skipped |
|---|---|---|---|
| Baseline (`2e27c58`, clean worktree) | 93 | 1,438 | 53 |
| Current | 94 | 7,537 | 53 |

**The failing test sets are identical**, name for name, except one: 
`test_deal_clustering_awards.py::test_the_attached_order_carries_how_forced_its_award_was`,
which **passes when re-run** — it belongs to a concurrent workstream (see §3.4) and was
mid-edit when the sweep ran. **No failure in this run is attributable to the formula
registry.** All 93 are pre-existing.

The pass-count difference is a collection difference, not a behavioural one: the clean
worktree lacks the untracked fixtures and test files present in the working tree, so it
collected far fewer parametrised cases. The failure-set comparison is the meaningful
signal, and it is unchanged.

### 3.4 A concurrent workstream shares this checkout

**Another session is actively editing `src/services/deal_clustering.py` and
`src/services/resolution/` in this same working tree**, building an award-resolution
layer (`awarded_pos`, `CardinalityRule`, an evidence-scaled margin). Those changes are
**not part of this work** and are not claimed by it. Two consequences:

1. **`deal_clustering.awarded_po` is registered against a moving target.** It delegated
   to an argmax over candidate POs when it was registered; it now delegates to the new
   resolver. Its golden vector still reproduces because it pins the *no-award* case,
   which neither implementation changes — but it does not pin the award path. The
   formula's `notes` say so and ask for a re-snapshot once that work lands.
2. **That workstream has started registering its own formulas.** Three
   (`negotiation.zopa_estimate`, `negotiation.outlier_rails`,
   `negotiation.batna_strength`) were appended to
   `src/services/formulas/definitions/negotiation.py` while this work was finishing,
   alongside a new `src/services/negotiation/leverage.py`. They are theirs, not part
   of this delivery. One of their golden vectors was briefly failing, which — correctly
   — took the whole registry's import down until they fixed it. That is the guard doing
   its job, and it is also a coordination cost worth knowing about: **one bad vector
   blocks every importer, not just its own module.**
3. **That workstream appears to address D-6.** The separation gap this report flagged —
   a bid matching four POs equally well reporting the same confidence as one matching
   exactly one — is what an evidence-scaled resolution margin exists to fix. D-6 should
   be re-checked against their implementation rather than acted on independently.

That workstream committed repeatedly while this one was in progress, moving
`Development` on from `2e27c58` several times. Their commits contain **none** of this
work, and this work's change to `deal_clustering.py` (the `pairwise_matrix` batch
conversion) stays cleanly separable from theirs — both independently changed a
`scorer=` default from a function to `None`, consistently and without conflict. The
baseline run above is against `2e27c58`, which is where this work started.

Nothing here has been committed. Committing this tree as it stands would sweep up
whatever they have in flight now.

### 3.2 The spec's Phase 4 checklist, item by item

**Callers that now receive `UNASSESSED` where they previously got a default.**
Per decision D2, a term the current body tolerates is declared *optional*, so no migrated
formula returns `UNASSESSED` where it previously returned a number. Every migrated call
site nonetheless handles the tri-state explicitly, and none treats it as zero or falsy:

| Call site | On `UNASSESSED` |
|---|---|
| `price_outlier/detector.py` | logs the line and **skips** it. Not "no outlier" — a line we could not assess. |
| `duplicate_invoice_detector.find_duplicates` | logs the pair and skips it. Explicitly *not* a clean bill of health for the later invoice. |
| `deal_clustering.pairwise_matrix` | leaves the pair **out** of the matrix rather than entering 0.0. `_corr` already reads a missing pair as 0.0 for linkage, but "not scored" and "scored, unrelated" stay distinguishable to anything inspecting the matrix. |
| `supplier_ranking_agent` composite / price | `NaN` for the whole frame, with a warning. Never 0.0 — the module's own comments explain why. |
| `opportunity_miner_agent` weightage | share set to `None`; the finding leaves the ranking rather than being pinned to the bottom of it. |
| `benchmark_live.benchmark_deal` | skips the line with a warning rather than publishing a variance computed from nothing. |
| `api/routers/benchmark.py` | HTTP 422 carrying `result.why()`. |
| `negotiation_agent.decide_strategy` | returns `strategy: "clarify"` — asks for structured pricing rather than countering at a number derived from an input it would not accept. |
| `negotiation_advice.advisor` | falls back to the module's own `indeterminate` shape, so a refused contract and an unplaceable deal reach the rest of the module identically — neither is a quadrant, and neither is guessed. |

`UNASSESSED` itself raises on `bool()` and on arithmetic, so a consumer that *did* treat
it as falsy would fail loudly rather than quietly. Four tests assert exactly that,
including that `result or 0` cannot silently zero it.

**Hardcoded constants now visible in contracts.** **36 of the 129 declared terms carry an
enforced range or allowed-set.** No value changed. The complete list of formula constants
is the "Hardcoded constants" column of `formula-inventory.md`; the ones now enforced at
the boundary include: relationship band cuts 92/80/65/45; cluster dampening 1.0/0.85/0.70;
line-pair weights 0.5/0.25/0.25; amount-agreement decay ceiling 0.10; temporal windows
365d/730d; `quote_rival` p0=0.03 / alpha=0.55; `invoice_duplicate` p0=0.02 / alpha=0.40;
benchmark clamps 0.85–1.15 / 0.90–1.20 / 0.90–1.25 and the 10/6 confidence ladder;
`_MAD_TO_SIGMA` 1.4826, min_peers 5, robust 5.0, material 3.0, critical 10.0; risk
weights 0.4/0.25/0.25/0.1, blend 0.55/0.45, logistic steepness 8.0, half-life 168h;
counter-plan 0.88 / 60% / 0.06 / 0.12; Kraljic bars 98,175 and 93; play nudges
+0.6/+0.3/−0.3/−0.7; payment-terms scale 0–90 days; three-way-match 0.01 / 0.005.

**Batch paths converted.** Six, listed in A-4.

**Evaluations that used "now".** Exactly one: `PredictiveRiskModel.evaluate` (D-1),
fixed in A-1. Its contract declares `as_of` as a `TIMESTAMP` term. No other registered
formula reads a clock; `datetime.now()` appears in no formula body.

**LLM-sourced inputs cap the result at `ASSERTED`.** The confidence ladder is
`OBSERVED > ASSERTED > UNVERIFIED` and a result is capped at the weakest input, so the
cap is structural rather than special-cased. Two tests assert it. Note the *default* is
`UNVERIFIED`, not `OBSERVED`: a caller that states no provenance has not earned more.
The seven formulas consuming extraction output (F16, F19, F34, F41, F51, F53, F56) will
report `ASSERTED` once their call sites pass provenance — see O-1.

**LLM calls inside formula bodies.** **None.** Re-verified across every registered
formula: zero hits for `ollama|llm|agent_nick|generate(` in the pure modules. The only
nearby call, `supplier_ranking_agent.py:2373`, generates justification *prose* after the
score is final and has a deterministic fallback. Nothing to move.

**Audit-spine schema change.** Proposed as DDL in `docs/adr/0002-formula-registry.md`
(`proc.bp_formula_evaluation`, three indexes). **Not run.** Records currently ride in
`proc.bp_agent_actions.details`, which needs no migration.

**Existing tests broken by confidence and provenance on outputs.** **None.** Confidence
and provenance live on the `Result` wrapper, not on the value, so `result.value` is
byte-identical to what the original function returned. Every migrated module's existing
suite passes unchanged.

**RLS coverage.** **Not applicable.** No tenant dimension exists in this schema. The
proposed table carries no RLS clause and the ADR records that adding a tenant column to
the underlying data makes RLS on that table a prerequisite of *that* change.

### 3.3 Outstanding

| ID | Item | Why it is still open |
|---|---|---|
| O-1 | **Provenance is not yet passed at any call site.** Every migrated evaluation therefore records `UNVERIFIED`. | Threading provenance IDs through requires each caller to know where its inputs came from, which is a separate piece of work in the extraction layer. The mechanism is built and tested; the callers do not use it yet. |
| O-2 | **31 of the 47 formulas have no call site outside their own module.** They are registered, contract-checked and pinned, but their callers still call the underlying function directly. | Named individually at the top of `docs/model-inventory.md`, generated from the registry, so this stays visible rather than being assumed done. Mostly sub-signal comparators reached *through* `linking.relationship_confidence`, which is itself registered. |
| O-3 | **14 of the 61 inventory formulas are not registered.** The opportunity detectors' individual impact formulas (F53) are embedded in detector bodies with their own guards; extracting them changes numbers. | Each needs its own before/after snapshot, as A-2 and A-3 got. `opportunity.price_variance_impact` registers the *shape* they share so it has a name to migrate onto. |
| O-4 | **The four amount-tolerance conventions (D-7) are unreconciled.** | Consolidating them changes which documents reconcile. That is a deliberate decision with a measurable blast radius, not a refactor. |
| O-5 | **`quote.weighting_score` still returns 0.0 for a quote with no data (D-3),** while `supplier.composite_score` returns `NaN` for the identical situation. | Pinned by a golden vector with a note saying so. Fixing it changes live rankings and needs a decision. |
| O-6 | **`risk.predictive_supplier_score` still fails open (D-2).** | Same: pinned, noted, and a fix changes every risk score in the system. |
| O-7 | **No retention policy for the proposed evaluation table.** | Should land in the same migration, not after it. Noted in the ADR. |
| O-8 | **`negotiate_dashboard` still computes savings itself.** `deal.realised_savings` exists and the deal-analysis path uses it; the dashboard's copy also formats and substitutes `"awaiting quote"`. | Migrating it changes what the dashboard displays for a deal with no quote. Small, but visible to a user. |

## 4. Decisions taken without an answer, and their reasoning

You asked me to proceed without settling the two open questions. Both are recorded here as
decisions I made, so they can be overturned cheaply.

**D1 — GPSS.** The decorator will accept `gpss_version` exactly as the spec names it, because
the spec names it. It defaults to `None`, and input terms bind to the existing derived
`concept_code` vocabulary (`src/services/facts/concept_codes.py`), not to an invented GPSS
dictionary. A non-`None` `gpss_version` is recorded and reported by `model_inventory()` but
resolves against nothing, and the ADR says so plainly. This keeps the spec's signature intact
while refusing to manufacture the authority that `00_seam_map.md` §B3 decided against. If GPSS
is later adopted, `concept_code` is the column it maps into and the field is already there.

**D2 — behaviour preservation vs. `UNASSESSED`.** The spec requires contract failure to return
`UNASSESSED`; it also requires the refactor to reproduce current outputs exactly. Where today's
code accepts a missing input and returns a default, those cannot both hold. Resolution: an input
that the current body tolerates is declared **optional** in its contract, so the body still runs
and returns the identical number, and the tolerance is listed under C-1 as a candidate for
deliberate tightening with its numeric consequence. `UNASSESSED` fires only on a genuine contract
violation — a required term absent, a value outside its declared range, a unit mismatch. This is
the reading that satisfies "no silent behaviour changes".

**D3 — the conformance worktree.** The registry is built as a new, additive package
(`src/services/formulas/`) that touches no file `worktree-conformance-phase1` modifies. That
branch's `detector_registry.py` maps slugs to handlers; this registry versions and audits the
maths inside those handlers. They compose rather than compete, and neither merge conflicts
with the other.

**D4 — population-scoped formulas.** `evaluate_many` is defined as *one evaluation over a set*,
not a loop of independent evaluations. This is what F35–F42 need (min-max ranges and
ratio-to-cheapest are properties of the set) and it is also what the spec's "batch evaluation
writes one aggregated audit record" implies. Scalar and set formulas are distinguished at
registration.

**D5 — tri-state confidence ladder.** `OBSERVED > ASSERTED > UNVERIFIED`, with `UNASSESSED`
as the separate no-result state rather than a fourth confidence level. `OBSERVED` = read
directly from a source document or a system of record; `ASSERTED` = claimed by a model or
derived from a claim; `UNVERIFIED` = provenance unknown. A result is capped at the minimum
confidence of its inputs, which is what makes the spec's "LLM inputs cap at `ASSERTED`" fall
out automatically rather than needing a special case.
