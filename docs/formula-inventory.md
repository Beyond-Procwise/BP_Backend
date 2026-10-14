# Formula Inventory — Phase 0 Discovery

**Date:** 2026-09-05 · **Branch:** Development (started at `2e27c58`) · **Scope:** every
site in `src/` that computes a number from domain inputs.

> **Correction, 2026-09-05.** One claim in this document was wrong when written and is
> corrected in place: F55 and F58 are *not* duplicate implementations of percentage
> change. The difference was found while migrating them and is set out in §2. Everything
> else here was re-checked during the migration and holds.

**Method.** Grep sweep across all 450 `src/**/*.py` files for scoring, weighting,
normalisation, percentile, threshold, log-odds and ratio patterns; then read of every
candidate. Test coverage established by cross-referencing `tests/`. Audit behaviour
established by tracing `record_action` / `proc.bp_agent_actions` writes. Live schema of
the audit table read from the RDS cluster (`bp_testdb`, `proc.bp_agent_actions`).

---

## 0. Headline findings before the table

Four facts change how the target spec should be read. Three of them affect whether parts
of the spec are buildable as written.

**F1 — GPSS does not exist in this codebase, and that was a deliberate, documented decision.**
The spec requires `@formula(..., gpss_version=...)` and "each input is a GPSS term". There is
no GPSS dictionary here and never has been. `docs/remediation/00_seam_map.md` §B3 raised this
as a blocker on 2026-08-07 and resolved it: the vocabulary is `concept_code`, derived at
import time from `extraction_schemas/*.yaml`, and the column was deliberately *not* named
`gpss_code` so no future reader would assume external authority behind it
(`src/services/facts/concept_codes.py:17-25`). Binding formula inputs to "GPSS terms" would
re-introduce exactly the shadow vocabulary that decision exists to prevent.

**F2 — Five of the nine formulas the brief names by name do not exist.**
Present: supplier ranking criteria scoring, document relationship matching log-odds,
opportunity detector scoring, weighted sums / normalisation / threshold comparisons.
Absent: **CISE criticality inference** (no criticality model anywhere; `CISE` matches only
as a substring inside unrelated words), **clause conformance log-odds** (the conformance work
is on an unmerged worktree branch and contains no log-odds), **guided-buying channel
thresholds** (nothing), **deal review checklist scoring** (nothing), **`evaluate_counter`**
(nothing by that name; the nearest real thing is `plan_counter`, F21).

**F3 — A registry seam already exists, and it is good.**
`linking_engine.PROFILES` + `register_profile()` + `register_signal()` is a working
in-memory registry of scoring profiles with a dispatch table of named comparators. Four
profiles are registered across three modules (`invoice_po`, `quote_po`, `quote_rival`,
`invoice_duplicate`). It has no versioning, no input contract and no audit record — but it
is the shape of the thing the spec asks for, and the migration should extend it rather than
compete with it. Separately, an **unmerged** branch (`worktree-conformance-phase1`) already
adds `src/engines/detector_registry.py` (slug→handler registry over 11 detectors) and
`src/engines/rule_book.py`.

**F4 — Two of the largest formulas are population-scoped, not row-scoped.**
`supplier_ranking_agent` and `quote_comparison_agent` normalise each supplier against the
*other suppliers in the same frame* (min-max ranges, ratio-to-cheapest). Their inputs are a
population, not a row. `evaluate(name, ctx)` as specified — one context, one output — cannot
express them without changing the numbers. `evaluate_many` can, but only if it is defined as
*one evaluation over a set*, not *a loop of independent evaluations*.

---

## 1. Core domain formulas

Columns: **Audit** = writes an audit record on evaluation. **Tests** = has tests that pin the
arithmetic. **Call** = direct call vs. through an abstraction. **Dup** = same calculation
exists elsewhere under another name.

### 1.1 Document relationship / linking (log-odds fusion)

| # | File · function | Computes | Inputs | Hardcoded constants | Audit | Tests | Call | Dup |
|---|---|---|---|---|---|---|---|---|
| F1 | `src/services/linking_engine.py:326` `score_link` | Relationship confidence `F` (0–100) between two documents: per-signal contribution `c = w·r·(2s−1)`, cluster dampening, log-odds → sigmoid, coverage, tier-1 conflict cap | source row, target row, profile name, source lines, target lines, optional set amount | `S=1.0`, `Q=1.0` (both hard-pinned, with the reasoning in-comment); band cuts `92/80/65/45` at `:59-62`; per-profile `p0`, `alpha`, `floor` | **No** — returns a full auditable dict, but nothing persists it unless the caller is the promotion path | Indirect only (`test_linking_engine_cache.py` covers caching, not the math); the math is pinned via `test_duplicate_invoice_detector.py`, `test_requirement_similarity.py`, `test_deal_clustering.py` | **Abstraction** (`PROFILES` + `register_profile`) | — |
| F2 | `linking_engine.py:64` `_dampen` | Cluster dampening factor by count of active signals | `n_active` | `1.0 / 0.85 / 0.70` | No | Indirect | Direct (internal) | — |
| F3 | `linking_engine.py:164` `_line_pair_score` | Per-line-pair composite over description Jaccard / qty equality / unit-price equality | two line dicts | weights `0.5 / 0.25 / 0.25`; tolerances `1e-9`, `1e-6` | No | Indirect | Direct (internal) | — |
| F4 | `linking_engine.py:182` `cmp_line_composite` | Best-match line-set score with source-extra coverage penalty | source lines, target lines | status cuts `0.8 / 0.5` | No | Indirect | Via `_signal_match` | — |
| F5 | `linking_engine.py:143` `cmp_numeric_tol` | Amount agreement with linear decay to 0 at 10% drift | two amounts, tolerance | decay ceiling `0.10`; status cut `0.5` | No | Indirect | Via `_signal_match` | Near-dup of `reconciliation._check_numeric` (F47) and `three_way_match._agrees` (F46) — see §2 |
| F6 | `linking_engine.py:219` `cmp_temporal` | Order-precedence + plausibility window | doc date, PO order date, due date | `365d → 1.0`, `730d → 0.7`, else `0.0`; cuts `0.7`, `>0` | No | Indirect | Via `_signal_match` | — |
| F7 | `linking_engine.py:239` `cmp_location` | Country/region blend | 4 location fields | `0.5/0.5` split; cuts `0.8 / 0.4` | No | Indirect | Via `_signal_match` | — |
| F8 | `src/services/requirement_similarity.py:116` `rivalry_score` | Rivalry correlation between two bids = `score_link(quote_rival).F / 100` | two bids, two line lists | profile `p0=0.03, alpha=0.55, floor=0.55`; signal weights `5/4/2/1/1`, caps `0.90/0.90/0.90/0.95/0.95` — comment says calibrated against `tests/fixtures/deal_clustering/golden_batch.py` | No | **Yes** `tests/services/test_requirement_similarity.py` | Wraps F1 | — |
| F9 | `requirement_similarity.py:23,39,55,67` `cmp_desc_overlap`, `cmp_volume`, `cmp_price_prox`, `cmp_buyer` | Four rivalry sub-signals (token Jaccard, qty ratio, price ratio, buyer equality) | line lists / rows | cuts `0.5`, `0.9/0.5`, `0.85/0.5` | No | Yes (same file) | Registered via `register_signal` | `cmp_desc_overlap` duplicates F3's description half at aggregate level |
| F10 | `src/services/duplicate_invoice_detector.py:208` `score_pair` | Duplicate-invoice confidence via `invoice_duplicate` profile | two invoice rows | profile `p0=0.02, alpha=0.40, floor=0.55`; weights `5/4/5/5/4/2/1`, caps `0.60/0.45/0.45/0.70/0.55/0.80/0.90` | No | **Yes** `tests/services/test_duplicate_invoice_detector.py` | Wraps F1 | — |
| F11 | `duplicate_invoice_detector.py:77,107` `cmp_ref_prox`, `cmp_date_prox` | Reference edit-distance proximity; date proximity ladder | two refs / two dates | `1.0/0.9/0.5/0.0`; date ladder `1.0/0.85/0.6/0.35/0.0` | No | Yes | `register_signal` | — |

### 1.2 Deal clustering

| # | File · function | Computes | Inputs | Hardcoded constants | Audit | Tests | Call | Dup |
|---|---|---|---|---|---|---|---|---|
| F12 | `src/services/deal_clustering.py:57` `cluster_confidence` | Cluster confidence = min pairwise correlation × 100 | cluster members, pair matrix | singleton → `100.0`; 1dp rounding | No | **Yes** `tests/services/test_deal_clustering.py` | Direct | — |
| F13 | `deal_clustering.py:33` `complete_linkage` | Agglomerative clustering, complete linkage | bids, matrix, threshold | `THRESHOLD = 0.70` | No | Yes | Direct | — |
| F14 | `deal_clustering.py:76` `awarded_po_scored` | Best PO award link + its `F` | bid, POs, lines | `min_score = 80.0` | No | Yes | Direct (injectable `scorer`) | — |
| F15 | `deal_clustering.py:101` `_REVIEW_BAND`, `_NEAR` | Review-flag thresholds | — | `80.0`, `0.05` | No | Yes | Direct | Restates `linking_engine._BAND_WARN` |

### 1.3 Benchmark pricing

| # | File · function | Computes | Inputs | Hardcoded constants | Audit | Tests | Call | Dup |
|---|---|---|---|---|---|---|---|---|
| F16 | `src/services/benchmark/engine.py:120` `compute_benchmark` | Adjusted benchmark price and variance: match set → simple/weighted/median → five multiplicative adjustments (volume, spec, location, SLA, inflation) → variance | `QuoteLine`, benchmark points, location index table, index table, `BenchmarkSettings` | **All injectable via `BenchmarkSettings`** (elasticity, clamps `0.85–1.15`, `0.90–1.20`, `0.90–1.25`, spec `0.03`, SLA `0.025`, min points, defaults `1.0`). Confidence ladder `_confidence` is **not** injectable: `10 → HIGH`, `6 → MEDIUM`, else `LOW` | No | **Yes, golden-vector style** — `tests/fixtures/benchmark/golden.json` + `test_benchmark_parity.py`, `test_benchmark_engine.py`, `test_benchmark_models.py` | Direct from `benchmark_live.py:274` and `api/routers/benchmark.py:56` | — |
| F17 | `benchmark/engine.py:34` `excel_round` | Half-away-from-zero rounding (contractual, penny-parity with the workbook) | value, digits | — | No | Yes | Direct (internal) | — |
| F18 | `benchmark/engine.py:86` `_confidence` | Evidence-count → confidence label | n, min points | `0 → No Data`, `<min → Insufficient`, `≥10 → HIGH`, `≥6 → MEDIUM`, else `LOW` | No | Yes | Direct (internal) | Conceptual dup of F43 (both called "confidence", measuring different things) |

### 1.4 Price outlier

| # | File · function | Computes | Inputs | Hardcoded constants | Audit | Tests | Call | Dup |
|---|---|---|---|---|---|---|---|---|
| F19 | `src/services/price_outlier/rule.py:42` `assess` | Robust outlier verdict: median + MAD z-score AND commercial materiality ratio | price, peer prices, `OutlierSettings` | `_MAD_TO_SIGMA = 1.4826` (fixed); everything else injectable (`min_peers=5`, `robust_threshold=5.0`, `material_ratio=3.0`, `critical_ratio=10.0`) | Findings persisted to `bp_extraction_discrepancy`, but **no evaluation record** | **Yes** `tests/test_price_outlier_rule.py` | Direct, in a loop (`detector.py:196`) | — |

### 1.5 Risk

| # | File · function | Computes | Inputs | Hardcoded constants | Audit | Tests | Call | Dup |
|---|---|---|---|---|---|---|---|---|
| F20 | `src/services/risk_intelligence_service.py:31` `PredictiveRiskModel.evaluate` | Forward-looking supplier risk 0–1: exponentially time-decayed signal severity blended with a weighted performance term, squashed through a logistic | supplier metrics dict, risk signals | half-life `168h` (injectable); performance weights `0.4/0.25/0.25/0.1`; blend `0.55/0.45`; logistic steepness `8.0`, centre `0.5`; **defaults on missing metrics: `on_time=1.0`, `quality=1.0`, `anomaly=0.0`, `resilience=0.5`** | No | **Yes** `tests/test_risk_intelligence_service.py` | Direct | — |
| F20a | same, line 38 | **Used `datetime.now(timezone.utc)` as the as-of time.** Same inputs on two different days gave different outputs | — | — | — | — | — | **Non-deterministic — fixed, see gap report A-1** |

### 1.6 Negotiation

| # | File · function | Computes | Inputs | Hardcoded constants | Audit | Tests | Call | Dup |
|---|---|---|---|---|---|---|---|---|
| F21 | `src/agents/negotiation_agent.py:304` `plan_counter` | Counter price + decision (`hold`/`clarify`/`accept`/`decline`/`counter`) by round | `NegotiationContext`, `SupplierSignals` | R1 anchor `×0.88` at gap `>10%`, else midpoint; R2 capture `60%` of gap; R3+ buffer `max(min_abs_buffer, target×risk_buffer_pct)`, soft-landing `×(1−step_pct_of_gap)` | No | **Yes** `tests/test_negotiation_skills.py` | Direct, via `compute_decision` (`:1467`) | — |
| F22 | `negotiation_agent.py:1467` `compute_decision` | Builds the context for F21 | payload dict | **`aggressiveness=0.75`, `leverage=0.6`, `urgency=0.3`, `risk_buffer_pct=0.06`, `min_abs_buffer=3.0`, `step_pct_of_gap=0.12`, `min_abs_step=4.0`, `ask_early_pay_disc=0.02` — all hardwired here, overriding the dataclass defaults and ungovernable** | No | Yes | Direct | — |
| F23 | `src/services/negotiation_advice/ranking.py:250` `rank_plays` | Play score = `1.0 + idx·0.01 + policy + performance + market` | supplier type, style, policy guidance, performance, market, playbook | base `1.0`, tie-break `0.01`; see F24–F26 | No | **Yes** `tests/services/negotiation_advice/test_ranking.py` | Direct from `advisor.py:148,233` and `negotiation_agent.py:7566` | — |
| F24 | `ranking.py:113` `_score_policy_alignment` | Policy nudge | lever, guidance sets | `+0.6 / +0.3 / −0.3 / −0.7` | No | Yes | Internal | — |
| F25 | `ranking.py:133` `_score_supplier_performance` | Performance nudge | lever, performance dict | cuts `0.9`, `0.97`, `0.6`, `0.6`, `0.5`; nudges `+0.4/+0.2/−0.1/+0.3/+0.2/+0.3/+0.25` | No | Yes | Internal | — |
| F26 | `ranking.py:185` `_score_market_context` | Market nudge | lever, market dict | nudges `+0.3/+0.2/+0.25/+0.2/+0.2`; keyword sets | No | Yes | Internal | — |
| F27 | `src/services/negotiation_advice/classification.py:38` `classify` | Kraljic quadrant + confidence from (deal value, alternative supplier count) | signals dict, thresholds | `high_spend = 98175.0`, `many_alternatives = 93` — **documented as derived from the live distribution, i.e. data-fitted constants living in code** | No | **Yes** `test_classification.py` | Direct | — |
| F28 | `classification.py:29` `_confidence` | Distance-from-bar confidence, `min(1.0, 0.5 + |v/bar − 1|)` | value, bar | `0.5` floor, `1.0` cap | No | Yes | Internal | — |
| F29 | `src/services/negotiation_advice/grounding.py:39` `assess` | Play readiness tri-state (`ready` / `groundwork` / `not_applicable`) against deal evidence | play, signals | quote-count cut `2`, alt-count cut `2` | No | **Yes** `test_grounding.py` | Loop in `apply_states` | **Already a tri-state with an explicit "cannot assess" arm — the closest existing analogue to `UNASSESSED`** |
| F30 | `negotiation_advice/signals.py:109` | Market-context thresholds | — | `RISK_ELEVATED = 60.0`, `THIN_MARKET_ALTERNATIVES = 93` | No | Yes `test_signals.py` | Direct | `93` restates F27's `many_alternatives` — **same constant, two definitions** |
| F31 | `negotiation_agent.py:4588` `_calculate_round_timeout` | Response-wait timeout | supplier count, round | `base 900s`, `per-supplier 300s`, multiplier `1 + 0.2·(round−1)`, cap `3600s` (all settings-overridable) | No | No | Direct | — |
| F32 | `src/agents/email_drafting_agent.py:403` `_calculate_tone_guidance` | Tone string by round and gap | round, gap % | cut `20%` | No | No | Direct | Returns prose, not a number — borderline |

### 1.7 Supplier ranking and quote comparison (population-scoped — see F4 above)

| # | File · function | Computes | Inputs | Hardcoded constants | Audit | Tests | Call | Dup |
|---|---|---|---|---|---|---|---|---|
| F33 | `src/agents/supplier_ranking_agent.py:95` `_normalize_days_to_score` | Payment-terms days → 0–100 (lower is better) | days, min, max | `min_days=0`, `max_days=90`; 2dp | No | **Yes** `tests/test_supplier_ranking_agent.py` | Direct via `ensure_payment_terms_score` | — |
| F34 | `supplier_ranking_agent.py:1579` `_score_deal_price` | Deal-scoped price score = `100 × cheapest / price`; NULL with <2 bidders | price column of the deal frame | bidder floor `2`; 2dp | No | Yes | Direct | Same intent as F35's `lower` direction, different math |
| F35 | `supplier_ranking_agent.py:2027` `_normalize_numeric_scores` | Min-max normalise each criterion to 0–100 across the frame; all-tied → 100 for measured rows only, NaN elsewhere | frame, direction map | scale `0–100` | No | Yes | Direct | **Duplicate of F41** |
| F36 | `supplier_ranking_agent.py:2063` `_normalise_weight_map` | Drop unusable criteria, renormalise weights to sum 1 | frame, weights | — | No | Yes | Direct | **Duplicate of the weight-renormalisation half of F41** |
| F37 | `supplier_ranking_agent.py:995` (inline in `run`) | Composite `final_score` = weighted mean over *present* criteria only, renormalised per supplier; all-absent → NaN | scored frame, weights | equal-weight fallback `1/n` | No | Yes | **Inline in a 2,597-line `run()` — no named function to register** (extracted during migration) | **Duplicate of F41** |
| F38 | `supplier_ranking_agent.py:2002` (inline) | Vector-similarity bonus: `final_score × (1 + similarity × 0.1)` | frame, embeddings | `0.1` | No | Not pinned | Inline | — |
| F39 | `supplier_ranking_agent.py:2008` `_score_categorical_criteria` | Map categorical values to scores from `CategoricalScoringPolicy` | frame, criteria, policies | `default 0` when the mapping misses | No | Yes | Direct | — |
| F40 | `supplier_ranking_agent.py:568` `_governed_default_weights` | Reads default weights from the governance envelope | context | — | No | Yes `tests/governance/test_ranking_consumes_governance.py` | Direct | — |
| F41 | `src/agents/quote_comparison_agent.py:690` `_calculate_weighting_scores` | Min-max normalise per metric → weighted mean → `×100` | quote entries, weights | `EPSILON = 1e-9`; **missing weight total → `weighting_score = 0.0`** | No | **Yes** `tests/test_quote_comparison_agent.py` | Direct | **Duplicate of F35+F36+F37 — and it diverges: this returns `0.0` where supplier ranking deliberately returns NaN. Same calculation, opposite treatment of "no data".** |
| F42 | `src/agents/quote_evaluation_agent.py:861` `_combine_weight_values` | Collapse a multi-factor weight dict to one number | weights dict | key list `price/delivery/risk/value`; excludes `tenure/volume`; **non-numeric → `0.0`** | No | **Yes** `tests/test_quote_evaluation_agent.py` | Direct | Third weight-collapsing convention |

### 1.8 Extraction quality / reconciliation

| # | File · function | Computes | Inputs | Hardcoded constants | Audit | Tests | Call | Dup |
|---|---|---|---|---|---|---|---|---|
| F43 | `src/services/extraction/promotion.py:163` `_compute_confidence_score` | Completeness 0–100: required fields 2 pts, optional 1 pt | doc type, row, required set | weights `2` / `1` | Yes — surfaced via `record_action(confidence=…)` on promote/hold | **Yes** `tests/extraction/*`, `tests/services/test_accuracy_score.py` | Direct | Named "confidence" but measures completeness — the comment at `linking_engine.py:36-53` records the harm this caused |
| F44 | `promotion.py:192` `_compute_accuracy_score` | Mean historical hit-rate of the readers that produced this doc, ×100 | columns, candidates, accuracy map | — | Yes (same path) | **Yes** `tests/services/test_accuracy_score.py` | Direct | — |
| F45 | `src/services/extraction/completeness.py:135` `assess` | Line-sum vs header reconciliation | doc type, columns, lines | `_RECONCILE_TOLERANCE = 0.05`, `_RECONCILE_ABS_TOLERANCE = 1.00` | Yes, via discrepancies | **Yes** `tests/extraction/test_completeness.py` | Direct (3 call sites in `dispatch.py`) | Third amount-tolerance convention (cf. F5, F46) |
| F46 | `src/services/extraction/three_way_match.py:43` `_agrees` / `:197` `check_against_po` | Invoice-vs-PO agreement and over-billing | invoice row, PO, lines | `_ABS_TOL = 0.01`, `_REL_TOL = 0.005` | Yes, via discrepancies | **Yes** `tests/extraction/test_three_way_match_pending_po.py`, `test_non_charge_lines.py` | Direct | Fourth amount-tolerance convention |
| F47 | `src/services/reconciliation.py:80` `_check_numeric` | Cross-document field agreement | docs, field, tolerance fn | `RECON_AMOUNT_TOLERANCE_PCT=0.01`, `_ABS=1.00`, `_TAX_PCT=0.1` (env-overridable) | No | **Yes** `tests/sql/test_bp_deal_overview_reconciliation_sql.py` | Direct | Fifth amount-tolerance convention |
| F48 | `src/services/facts/arithmetic.py:49` `check_line_arithmetic` | Whether `qty × unit_rate = extended_line` corroborates the assigned roles; returns a 4-state `ArithmeticState` including two explicit *untestable* arms | qty, unit rate, extended line | Decimal, no float | No | **Yes** `tests/services/facts/*` | Direct | **Second existing tri-/quad-state analogue to `UNASSESSED`** |
| F49 | `src/services/facts/fx.py` | FX resolution and conversion; returns explicit `FX_UNAVAILABLE` rather than a default rate | currency pair, table | sentinel `FX_UNAVAILABLE` | Stamped onto the fact (rate, rate date, source) | **Yes** `tests/services/facts/test_fx.py` | Direct | **Third existing "refuse rather than default" analogue** |
| F50 | `src/services/field_accuracy.py:133` `clean_numeric` | Numeric coercion from raw extracted text | raw value | — | No | **Yes** `tests/test_field_accuracy.py` | Direct | — |

### 1.9 Opportunity detection

| # | File · function | Computes | Inputs | Hardcoded constants | Audit | Tests | Call | Dup |
|---|---|---|---|---|---|---|---|---|
| F51 | `src/agents/opportunity_miner_agent.py:2182` (inline in `run`) | Finding weightage: `impact × (1+risk) × (1+coverage)`, normalised over the run | findings, risk map, flow coverage | — | Findings persisted to `bp_opportunity`; **no evaluation record** | Partly `tests/test_opportunity_miner_agent.py` | Inline (extracted during migration) | — |
| F52 | `opportunity_miner_agent.py:1397` `_normalise_risk_score` | Coerce a risk value onto 0–1, dividing by 100 when >1 | raw value | `100.0` divisor, `1.0` cap, **`0.0` on unparseable** | No | Partly | Direct | Different convention from F20 (0–1 native) and F30 (0–100) |
| F53 | `opportunity_miner_agent.py` × 14 `_policy_*` handlers (`:4612, 4840, 5607, 5763, 5858, 6227, 6340, 6416, 6505, 6618, 6751, 6894, 7134, 7241`) | 14 detector impact formulas — e.g. `(actual − benchmark) × qty` (`:5578, 5730`), `invoice_total − anchor` (`:5820`), `spend × discount_rate` (`:4797`, fallback `× 0.05`), `risk_score × risk_weight` (`:6385`), `actual − budget` (`:6709`), `(invoice_price − po_price) × qty` (`:6850`), `total_value × late_ratio` (`:7112`), `max(0, threshold − score) × penalty` (`:7194`) | table frames + `conditions` dict | Fallback `0.05` consolidation rate (`:4799`), `savings_pct` default `5.0` (`:4856`), `performance_threshold 0.9`, `negotiation_window_days 90`, `risk_weight 1000.0` (the last three visible in the unmerged `detector_registry.py`) | Findings persisted; no evaluation record | Partly `tests/test_opportunity_miner_agent.py` | Direct method dispatch on Development; **already behind a registry on the unmerged conformance branch** | Several restate F16's variance idea in cruder form |

### 1.10 Deal metrics and requirements

| # | File · function | Computes | Inputs | Hardcoded constants | Audit | Tests | Call | Dup |
|---|---|---|---|---|---|---|---|---|
| F54 | `src/services/deal_analysis_service.py:58` `_weighted_unit_price` | Σ(qty×price) / Σqty across a doc type's lines | doc list | — | No | **Yes** `tests/services/test_deal_analysis_service.py` | Internal | — |
| F55 | `deal_analysis_service.py:76` `_pct_change` | `(new − base) / base × 100`, 2dp; None on a zero baseline | new, base | — | No | Yes | Internal | See the F58 correction in §2 |
| F56 | `deal_analysis_service.py:103` `_compute` | Deal value, volume, unit price, price/volume change, `efficiency_score = (quote_unit − inv_unit) × inv_vol` | deal documents | 2dp / 4dp rounding | No | Yes | Direct via `compute_deal_metrics` | `efficiency_score` is realised savings — same quantity as F57 |
| F57 | `src/services/negotiate_dashboard.py:110,186` | `savings_pct = (quote − actual)/quote × 100`; `savings = quote − actual` | deal row | 1dp | No | Not pinned | Inline | Second savings convention |
| F58 | `src/services/opportunity_dashboard.py:49` `_pct_change` | Month-over-month % change, formatted | current, previous | zero baseline → `"+0%"` / `"+100%"` | No | Not pinned | Internal | **NOT a duplicate of F55** — see §2 |
| F59 | `src/services/requirement_service.py:223` `evaluate_completeness` | Filled-required-fields ratio 0–1 | requirement, required fields | empty-required → `1.0` | No | Partly | Direct | Same shape as F43 at a different scale (0–1 vs 0–100) |
| F60 | `src/services/feedback_service.py:116` `_pattern_match_score` | `min(1.0, matches / (0.3 × len(patterns)))`; short-message boost `×1.5` | text, patterns | `0.3`, `1.5`, cuts `0.3/0.4/0.2` | No | Not pinned | Internal | — |
| F61 | `src/services/rbac.py:160` `role_rank` | Role → integer rank | role, policy engine | governed table | No | **Yes** `tests/test_policy_engine.py` | Direct | — |

---

## 2. Duplicate calculations, consolidated

| Calculation | Implementations | Divergence |
|---|---|---|
| Min-max normalise + weighted composite | **F35+F36+F37** (supplier ranking), **F41** (quote comparison), **F42** (quote evaluation, weight collapse only) | Supplier ranking returns **NaN** on no data; quote comparison returns **0.0**; quote evaluation returns **0.0**. Same maths, three different answers to "we don't know". |
| Amount agreement within tolerance | **F5** (10% linear decay), **F45** (5% or £1.00), **F46** (1% or £0.01), **F47** (1% or £1.00, env-tunable) | Four tolerance conventions, no shared definition. |
| Percentage change | **F55**, **F58** | ~~Identical bodies, two files.~~ **Corrected 2026-09-05 during migration:** they are not equivalent. F55 divides by `base`, rounds to 2dp and returns `None` for a zero baseline; F58 divides by `abs(prev)`, rounds to whole percent, returns a formatted string, and maps a zero baseline to `"+0%"` or `"+100%"`. They disagree on sign for a negative baseline. F58 is a display helper and was deliberately left alone. |
| Realised savings | **F56** `efficiency_score`, **F57** dashboard `savings` | Same quantity, different names, no shared function. |
| "Confidence" | **F18** (evidence count), **F43** (field completeness), **F28** (distance from a bar), **F12** (min pairwise correlation) | Four unrelated meanings under one word. F43's misuse as a quality proxy is the documented cause of a live outage-class bug (`linking_engine.py:36-53`). |
| Alternative-supplier "thin market" bar = 93 | **F27** `many_alternatives`, **F30** `THIN_MARKET_ALTERNATIVES` | Same number defined twice in two modules. |
| Risk score scale | **F20** (0–1), **F30** (0–100), **F52** (coerces either) | Three conventions for one field. |

---

## 3. Cross-cutting observations

**Audit.** No formula anywhere writes an evaluation record. `proc.bp_agent_actions` exists and
is the natural spine — live schema is `(action_id, created_at, deal_id, document_id, doc_pk,
doc_type, process_monitor_id, trace_id, phase, action_type, agent, field_name, status, summary,
details jsonb, confidence numeric, pipeline_version)`. It records **decisions** (`promote_held`
1,897 rows, `discrepancy` 794, `promote_to_trgt` 69) but never an evaluation. There is no
`formula_name`, no `version_hash`, no `inputs`, no `provenance_ids` column. `proc.bp_model`
exists but is an LLM-provider registry (3 rows), unrelated.

**Golden vectors.** Two already exist in spirit: `tests/fixtures/benchmark/golden.json`
(F16, penny-parity against the workbook) and `tests/fixtures/deal_clustering/golden_batch.py`
(F8's `p0`/`alpha` were calibrated against it). Neither is enforced at import time; both run
only under pytest.

**Determinism.** All pure-calculation modules are LLM-free — verified by grep for
`ollama|llm|agent_nick|generate(` across F1–F19 and F48–F49: zero hits. The one LLM call
near a formula is `supplier_ranking_agent.py:2373`, and it generates the *justification prose*
after the score is final, with a deterministic fallback. **No LLM call sits inside a formula
body.** One genuine non-determinism: **F20a** reads `datetime.now()`.

**LLM-derived inputs.** F16, F19, F34, F41, F51, F53, F56 all consume `unit_price`,
`total_amount` and `quantity` from the extraction pipeline, whose L3 arm is an AI judge.
Those inputs are asserted, not observed, and nothing downstream currently records that.

**Batch paths that would become `evaluate` loops.** `price_outlier/detector.py:196` (per line),
`duplicate_invoice_detector.py:218` (per candidate pair), `benchmark_live.py:274` (per quote
line), `deal_clustering.pairwise_matrix` (per bid pair), `linking_engine._promote` (per staged
row), `opportunity_miner` (per finding × 14 detectors).

**Tenancy.** No tenant dimension exists in this schema (recorded in
`project_ask_path_authorization`). RLS on persisted formula metadata is not applicable
until one does.

---

## 4. What the brief named that is not here

| Named in brief | Status |
|---|---|
| Supplier ranking criteria scoring | **Present** — F33–F42 |
| Document relationship matching log-odds | **Present** — F1–F11 |
| Opportunity detector scoring | **Present** — F51–F53 |
| Weighted sum / normalisation / threshold | **Present** — throughout |
| Percentile | **Absent** — nothing computes a percentile; F19 uses median + MAD instead |
| CISE criticality inference | **Absent** — no criticality model in `src/`; `CISE` matches only as a substring |
| Clause conformance log-odds | **Absent on Development** — conformance work is on the unmerged `worktree-conformance-phase1`, and contains rules, not log-odds |
| Negotiation `evaluate_counter` | **Absent by that name** — nearest is `plan_counter` (F21) |
| Deal review checklist scoring | **Absent** |
| Guided-buying channel thresholds | **Absent** |
