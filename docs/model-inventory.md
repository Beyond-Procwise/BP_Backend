# Model Inventory

Generated from the formula registry by `model_inventory()` — 2026-09-05 22:06 UTC.
Do not hand-edit: regenerate with `python -m src.services.formulas.inventory`.

**50 registered formulas.**

**34 with no call site outside their own module** — either not yet migrated onto `evaluate`, or dead: `benchmark.evidence_confidence`, `benchmark.excel_round`, `deal_clustering.awarded_po`, `deal_clustering.cluster_confidence`, `duplicate_invoice.date_proximity`, `duplicate_invoice.reference_proximity`, `extraction.amount_agreement`, `extraction.completeness_score`, `extraction.line_arithmetic_state`, `linking.amount_agreement`, `linking.cluster_dampening`, `linking.decision_band`, `linking.line_pair_score`, `linking.line_set_composite`, `linking.location_agreement`, `linking.temporal_plausibility`, `negotiation.batna_strength`, `negotiation.market_context_score`, `negotiation.outlier_rails`, `negotiation.play_readiness`, `negotiation.policy_alignment_score`, `negotiation.supplier_performance_score`, `negotiation.threshold_confidence`, `negotiation.zopa_estimate`, `opportunity.price_variance_impact`, `opportunity.risk_normalisation`, `quote.weighting_score`, `requirement.completeness_score`, `risk.predictive_supplier_score`, `rivalry.description_overlap`, `rivalry.price_proximity`, `rivalry.volume_agreement`, `supplier.criterion_normalisation`, `supplier.weight_renormalisation`

| Formula | Ver | Owner | Purpose | GPSS | Last validated | Vectors | Dependents |
|---|---|---|---|---|---|---|---|
| `benchmark.adjusted_price` | 1.0.0+d0939572 | commercial | Fair benchmark unit price for one quote line, and the variance against it | — | 2026-09-05 22:06:33 | 2 | 3 |
| `benchmark.evidence_confidence` | 1.0.0+8a74bdde | commercial | Confidence label for a benchmark, from how many comparable points backed it | — | 2026-09-05 22:06:33 | 5 | — |
| `benchmark.excel_round` | 1.0.0+b600c6e4 | commercial | Excel ROUND semantics: half away from zero | — | 2026-09-05 22:06:33 | 2 | — |
| `deal.pct_change` | 1.0.0+39441e5c | analytics | Percentage change of a value against a baseline | — | 2026-09-05 22:06:33 | 4 | 1 |
| `deal.realised_savings` | 1.0.0+09ac2b97 | analytics | Money actually saved: the quote-to-invoice unit price gap times invoiced volume | — | 2026-09-05 22:06:33 | 3 | 1 |
| `deal.weighted_unit_price` | 1.0.0+08d83ed7 | analytics | Volume-weighted unit price across a document type's line items | — | 2026-09-05 22:06:33 | 2 | 1 |
| `deal_clustering.awarded_po` | 1.0.0+7b0692e8 | linkage | Which purchase order a bid won, by continuity scoring rather than name matching | — | 2026-09-05 22:06:33 | 1 | — |
| `deal_clustering.cluster_confidence` | 1.0.0+885404a4 | linkage | Confidence that every member of a cluster belongs to the same sourcing event | — | 2026-09-05 22:06:33 | 2 | — |
| `duplicate_invoice.date_proximity` | 1.0.0+a120547b | assurance | How close two invoice dates are, on a graded ladder | — | 2026-09-05 22:06:33 | 3 | — |
| `duplicate_invoice.pair_score` | 1.0.0+bc300d89 | assurance | How likely two invoices are the same invoice billed twice | — | 2026-09-05 22:06:33 | 2 | 1 |
| `duplicate_invoice.reference_proximity` | 1.0.0+ef75b67d | assurance | Whether two invoice references are the same reference | — | 2026-09-05 22:06:33 | 3 | — |
| `extraction.amount_agreement` | 1.0.0+d7f5c294 | extraction | Whether two money amounts agree within the three-way-match tolerance | — | 2026-09-05 22:06:34 | 3 | — |
| `extraction.completeness_score` | 1.0.0+facea3ec | extraction | How much of a document's schema the extraction actually filled (0-100) | — | 2026-09-05 22:06:34 | 3 | — |
| `extraction.line_arithmetic_state` | 1.0.0+93f89c05 | extraction | Whether quantity x unit rate = extended line corroborates the assigned roles | — | 2026-09-05 22:06:34 | 4 | — |
| `linking.amount_agreement` | 1.0.0+9136b135 | linkage | Whether two amounts agree, decaying linearly to zero at 10% drift | — | 2026-09-05 22:06:34 | 4 | — |
| `linking.cluster_dampening` | 1.0.0+89fdee3d | linkage | Correlated-signal dampening factor within one evidence cluster | — | 2026-09-05 22:06:34 | 3 | — |
| `linking.decision_band` | 1.0.0+4f959f09 | linkage | Which action band a relationship confidence F falls into | — | 2026-09-05 22:06:34 | 5 | — |
| `linking.line_pair_score` | 1.0.0+cb656046 | linkage | Similarity of two individual line items (description / qty / unit price) | — | 2026-09-05 22:06:34 | 2 | — |
| `linking.line_set_composite` | 1.0.0+b4cdfa68 | linkage | Best-match line-set agreement with a source-extra coverage penalty | — | 2026-09-05 22:06:34 | 3 | — |
| `linking.location_agreement` | 1.0.0+eac1a040 | linkage | Country and region agreement between two documents | — | 2026-09-05 22:06:34 | 2 | — |
| `linking.relationship_confidence` | 1.0.0+3e3c340e | linkage | How confidently two procurement documents are the same relationship (F, 0-100) | — | 2026-09-05 22:06:34 | 2 | 1 |
| `linking.temporal_plausibility` | 1.0.0+ba1d1b32 | linkage | Whether a child document's date is plausible against its parent's order date | — | 2026-09-05 22:06:34 | 4 | — |
| `negotiation.batna_strength` | 1.0.0+aeabae8b | negotiation | The buyer's walk-away position from alternative quotes and trading history | — | 2026-09-05 22:06:47 | 5 | — |
| `negotiation.counter_plan` | 1.0.0+39f65e1a | negotiation | What to counter at, and whether to counter at all, given the round and the gap | — | 2026-09-05 22:06:47 | 8 | 2 |
| `negotiation.kraljic_quadrant` | 1.0.0+82ee8709 | negotiation | Kraljic quadrant and suggested negotiation style for a deal | — | 2026-09-05 22:06:47 | 5 | 1 |
| `negotiation.market_context_score` | 1.0.0+60b018ae | negotiation | How far market conditions argue for one lever | — | 2026-09-05 22:06:47 | 2 | — |
| `negotiation.outlier_rails` | 1.0.0+35afd9c8 | negotiation | Whether an offer breaches a review or escalation rail on price, volume or term | — | 2026-09-05 22:06:47 | 5 | — |
| `negotiation.play_rank` | 1.0.0+9c252e65 | negotiation | Rank playbook plays for a (supplier type, style) pair against live signals | — | 2026-09-05 22:06:47 | 2 | 1 |
| `negotiation.play_readiness` | 1.0.0+3936fa3b | negotiation | Whether a play's precondition actually holds on this deal's evidence | — | 2026-09-05 22:06:47 | 4 | — |
| `negotiation.policy_alignment_score` | 1.0.0+6224b68c | negotiation | How far governed policy pushes for or against one negotiation lever | — | 2026-09-05 22:06:47 | 3 | — |
| `negotiation.supplier_performance_score` | 1.0.0+fad2575a | negotiation | How far a supplier's measured performance argues for one lever | — | 2026-09-05 22:06:47 | 3 | — |
| `negotiation.threshold_confidence` | 1.0.0+08a1494e | negotiation | How far a value sits from a classification bar, as a 0.5-1.0 confidence | — | 2026-09-05 22:06:47 | 3 | — |
| `negotiation.zopa_estimate` | 2.0.0+3f03aa12 | negotiation | Buyer's ceiling, the supplier's cost floor if any evidence exists, and an entry counter | — | 2026-09-05 22:06:47 | 3 | — |
| `opportunity.finding_weight_factor` | 1.0.0+c361a830 | opportunity | One finding's unnormalised weight: money at stake, amplified by risk and coverage | — | 2026-09-05 22:06:47 | 2 | 2 |
| `opportunity.price_variance_impact` | 1.0.0+5527a64c | opportunity | Money at stake when a paid price sits above a benchmark | — | 2026-09-05 22:06:47 | 2 | — |
| `opportunity.risk_normalisation` | 1.0.0+bf07873b | opportunity | Coerce a supplier risk value onto 0-1, whatever scale it arrived on | — | 2026-09-05 22:06:47 | 5 | — |
| `opportunity.weightage_shares` | 1.0.0+b908dedf | opportunity | Turn a run's weight factors into shares of that run | — | 2026-09-05 22:06:47 | 2 | 1 |
| `price_outlier.verdict` | 1.0.0+5f72312b | assurance | Whether a unit price is both statistically extreme and commercially material | — | 2026-09-05 22:06:47 | 5 | 2 |
| `quote.weighting_score` | 1.0.0+903ffe16 | sourcing | Weighted 0-100 comparison score across competing quotes | — | 2026-09-05 22:06:47 | 2 | — |
| `requirement.completeness_score` | 1.0.0+b744edd2 | analytics | Proportion of a requirement's required fields that are genuinely filled | — | 2026-09-05 22:06:33 | 2 | — |
| `risk.predictive_supplier_score` | 1.0.0+573d6ee2 | risk | Forward-looking supplier risk 0-1 from decayed incident signals and performance | — | 2026-09-05 22:06:47 | 2 | — |
| `rivalry.correlation` | 1.0.0+955cbb2c | linkage | How strongly two bids look like rivals for the same sourcing event | — | 2026-09-05 22:06:47 | 2 | 1 |
| `rivalry.description_overlap` | 1.0.0+0565dc1f | linkage | Token Jaccard over two bids' aggregated line descriptions | — | 2026-09-05 22:06:47 | 2 | — |
| `rivalry.price_proximity` | 1.0.0+1580e47b | linkage | Graded price closeness between two bids (proximity, never equality) | — | 2026-09-05 22:06:47 | 2 | — |
| `rivalry.volume_agreement` | 1.0.0+d7b35f94 | linkage | Quantity-total ratio between two bids | — | 2026-09-05 22:06:47 | 2 | — |
| `supplier.composite_score` | 1.0.0+488a3cd8 | sourcing | Weighted composite supplier score over only the criteria we hold for them | — | 2026-09-05 22:06:47 | 2 | 2 |
| `supplier.criterion_normalisation` | 1.0.0+2bc2ceb8 | sourcing | Min-max normalise one raw criterion to 0-100 across the suppliers being compared | — | 2026-09-05 22:06:47 | 4 | — |
| `supplier.deal_price_score` | 1.0.0+d877d447 | sourcing | Price scored against the cheapest RIVAL bid on the same deal (100 = cheapest) | — | 2026-09-05 22:06:47 | 3 | 2 |
| `supplier.payment_terms_score` | 1.0.0+36b959b6 | sourcing | Payment terms in days scored 0-100, longer terms being better for the buyer | — | 2026-09-05 22:06:47 | 6 | 2 |
| `supplier.weight_renormalisation` | 1.0.0+cca0ca38 | sourcing | Drop criteria nobody can be scored on, and renormalise the rest to sum to 1 | — | 2026-09-05 22:06:47 | 2 | — |

---

## Contracts

### `benchmark.adjusted_price`

*Fair benchmark unit price for one quote line, and the variance against it* — scalar formula, owner **commercial**, effective from 2026-09-05, source `src.services.formulas.definitions.benchmarking:33`.

Rounding is contractual: every intermediate is rounded exactly where the prototype rounds, Excel-style (half away from zero). Never substitute Python's round() here. Below `min_data_points` matching points the engine returns a gated result with every computed field None, which `evaluate` reports as a value (not UNASSESSED) because the gate is a MEASURED outcome -- 'we looked and there is not enough evidence' -- rather than a contract failure.

| Input | Unit | Range | Required |
|---|---|---|---|
| `quote` | record | unbounded | yes |
| `points` | rows | unbounded | yes |
| `location_index_table` | record | unbounded | no |
| `index_table` | record | unbounded | no |
| `settings` | record | unbounded | no |

**Output:** `BenchmarkResult` in GBP — adjusted benchmark, the five factors, variance and evidence

**Dependents:** `src/api/routers/benchmark.py`, `src/services/benchmark_live.py`, `tests/services/formulas/test_registered_formulas.py`

### `benchmark.evidence_confidence`

*Confidence label for a benchmark, from how many comparable points backed it* — scalar formula, owner **commercial**, effective from 2026-09-05, source `src.services.formulas.definitions.benchmarking:95`.

The 10 / 6 cuts are the one thing in this module NOT on BenchmarkSettings (gap report D-14). Registered separately so the inconsistency is visible and so a later move onto settings is a version bump rather than a silent edit.

| Input | Unit | Range | Required |
|---|---|---|---|
| `n_total` | count | [0, +inf] | yes |
| `min_points` | count | [0, +inf] | yes |

**Output:** `str` in label — No Data | Insufficient | LOW | MEDIUM | HIGH

**Dependents:** *none*

### `benchmark.excel_round`

*Excel ROUND semantics: half away from zero* — scalar formula, owner **commercial**, effective from 2026-09-05, source `src.services.formulas.definitions.benchmarking:124`.

Python's round() is half-to-EVEN and drifts from the workbook on ties: -172.5 rounds to -172 rather than Excel's -173. This is contractual, not stylistic.

| Input | Unit | Range | Required |
|---|---|---|---|
| `value` | money | unbounded | yes |
| `digits` | count | [0, 10] | yes |

**Output:** `float` in money — the rounded value

**Dependents:** *none*

### `deal.pct_change`

*Percentage change of a value against a baseline* — scalar formula, owner **analytics**, effective from 2026-09-05, source `src.services.formulas.definitions.deal:58`.

**Replaces:** `deal_analysis_service._pct_change`

None against a zero or absent baseline: there is no percentage change from nothing, and 0.0 would read as 'measured, and it did not move'.

`opportunity_dashboard._pct_change` LOOKS like a duplicate of this and is NOT one: it divides by `abs(prev)` (so it disagrees on sign for a negative baseline), rounds to whole percent, returns a formatted string, and maps a zero baseline to '+0%' or '+100%' rather than to no answer. It is a display helper, not this calculation, and consolidating them would change what the dashboard shows. Left alone deliberately.

| Input | Unit | Range | Required |
|---|---|---|---|
| `new` | money | unbounded | no |
| `base` | money | unbounded | no |

**Output:** `float` in percent — percentage points, 2dp; None against a zero baseline

**Dependents:** `src/services/deal_analysis_service.py`

### `deal.realised_savings`

*Money actually saved: the quote-to-invoice unit price gap times invoiced volume* — scalar formula, owner **analytics**, effective from 2026-09-05, source `src.services.formulas.definitions.deal:91`.

**Replaces:** `deal_analysis_service._compute.efficiency_score`, `negotiate_dashboard.savings`

One name for a quantity previously computed twice: `efficiency_score` in the deal analysis service and `savings` on the negotiate dashboard. The dashboard additionally reported 0.0 when there was no quote to compare against, which reads as 'we saved nothing' rather than 'there is no baseline'; this returns None, and callers that want the old display string ask for it explicitly.

| Input | Unit | Range | Required |
|---|---|---|---|
| `quoted_unit_price` | money | [0, +inf] | no |
| `invoiced_unit_price` | money | [0, +inf] | no |
| `invoiced_volume` | count | [0, +inf] | no |

**Output:** `float` in money — positive = saved against the quote; None if unknowable

**Dependents:** `src/services/deal_analysis_service.py`

### `deal.weighted_unit_price`

*Volume-weighted unit price across a document type's line items* — scalar formula, owner **analytics**, effective from 2026-09-05, source `src.services.formulas.definitions.deal:31`.

Returns None, not 0.0, when no line carries both a quantity and a unit price. Services deals are lump-sum and carry no quantity at all, so None is the normal outcome for them rather than an error.

| Input | Unit | Range | Required |
|---|---|---|---|
| `documents` | rows | unbounded | yes |

**Output:** `float` in money — total line value / total quantity; None if unknowable

**Dependents:** `src/services/deal_analysis_service.py`

### `deal_clustering.awarded_po`

*Which purchase order a bid won, by continuity scoring rather than name matching* — scalar formula, owner **linkage**, effective from 2026-09-05, source `src.services.formulas.definitions.clustering:58`.

**Under active concurrent development.** `awarded_po_scored` was an argmax over every candidate PO when this formula was registered; it now delegates to `src.services.resolution`, which is being built in a parallel workstream and appears to address the separation gap this formula's registration flagged (gap report D-6: with S pinned to 1.0, a bid matching four POs equally well reported the same confidence as one matching exactly one). The vector below holds across both implementations because it pins the no-award case, which neither changes. **Re-snapshot this formula's vectors once that work lands** -- the current set does not pin the award path.

| Input | Unit | Range | Required |
|---|---|---|---|
| `bid` | record | unbounded | yes |
| `purchase_orders` | rows | unbounded | yes |
| `po_lines` | record | unbounded | no |
| `bid_lines` | rows | unbounded | no |
| `min_score` | score_0_100 | [0, 100] | no |

**Output:** `tuple` in score_0_100 — (winning po_id or None, its F)

**Dependents:** *none*

### `deal_clustering.cluster_confidence`

*Confidence that every member of a cluster belongs to the same sourcing event* — scalar formula, owner **linkage**, effective from 2026-09-05, source `src.services.formulas.definitions.clustering:32`.

A singleton returns 100.0 -- maximum confidence from no evidence at all (gap report D-9). Callers are expected to treat a one-bid event separately; that is a convention, not something this function enforces.

| Input | Unit | Range | Required |
|---|---|---|---|
| `cluster` | rows | unbounded | yes |
| `matrix` | record | unbounded | yes |

**Output:** `float` in score_0_100 — minimum pairwise correlation x 100, 1dp

**Dependents:** *none*

### `duplicate_invoice.date_proximity`

*How close two invoice dates are, on a graded ladder* — scalar formula, owner **assurance**, effective from 2026-09-05, source `src.services.formulas.definitions.duplicates:78`.

| Input | Unit | Range | Required |
|---|---|---|---|
| `a` | date | unbounded | no |
| `b` | date | unbounded | no |

**Output:** `tuple` in ratio — (score 0-1, status)

**Dependents:** *none*

### `duplicate_invoice.pair_score`

*How likely two invoices are the same invoice billed twice* — scalar formula, owner **assurance**, effective from 2026-09-05, source `src.services.formulas.definitions.duplicates:26`.

Consecutive reference numbers score CONFLICT on ref_prox, not OK: INV-0455 and INV-0456 are ordinary sequential invoices, and treating near-identical references as evidence of duplication would flag every supplier who invoices twice in a week.

| Input | Unit | Range | Required |
|---|---|---|---|
| `earlier` | record | unbounded | yes |
| `later` | record | unbounded | yes |

**Output:** `dict` in score_0_100 — F plus the full signal breakdown and band

**Dependents:** `src/services/duplicate_invoice_detector.py`

### `duplicate_invoice.reference_proximity`

*Whether two invoice references are the same reference* — scalar formula, owner **assurance**, effective from 2026-09-05, source `src.services.formulas.definitions.duplicates:57`.

| Input | Unit | Range | Required |
|---|---|---|---|
| `a` | text | unbounded | no |
| `b` | text | unbounded | no |

**Output:** `tuple` in ratio — (score 0-1, status)

**Dependents:** *none*

### `extraction.amount_agreement`

*Whether two money amounts agree within the three-way-match tolerance* — scalar formula, owner **extraction**, effective from 2026-09-05, source `src.services.formulas.definitions.extraction:81`.

Tolerances `_ABS_TOL = 0.01` and `_REL_TOL = 0.005`. One of FOUR amount-tolerance conventions in this codebase (gap report D-7): the linking engine decays linearly to zero at 10% drift, extraction completeness allows 5% or GBP 1.00, and reconciliation allows 1% or GBP 1.00. The same two numbers can agree in one module and conflict in another.

| Input | Unit | Range | Required |
|---|---|---|---|
| `a` | money | unbounded | yes |
| `b` | money | unbounded | yes |

**Output:** `bool` in money — True when within 1p absolute or 0.5% relative

**Dependents:** *none*

### `extraction.completeness_score`

*How much of a document's schema the extraction actually filled (0-100)* — scalar formula, owner **extraction**, effective from 2026-09-05, source `src.services.formulas.definitions.extraction:27`.

**This measures COMPLETENESS, not correctness, and its name has caused real harm.** Required fields score 2, optional 1, as a percentage of the schema, so a row with every required field and no optionals lands at ~50%. Using it as an evidence-quality proxy in the linking engine made the promotion gate mathematically unreachable by a perfect match (F = 75.5 against a gate of 80): no invoice could promote and no deal could form. A document can be entirely complete and entirely wrong -- `extraction.accuracy_score` is the other question.

| Input | Unit | Range | Required |
|---|---|---|---|
| `doc_type` | label | unbounded | yes |
| `row` | record | unbounded | yes |
| `required` | rows | unbounded | yes |

**Output:** `Decimal` in score_0_100 — 0-100; None when the doc type has no field list

**Dependents:** *none*

### `extraction.line_arithmetic_state`

*Whether quantity x unit rate = extended line corroborates the assigned roles* — scalar formula, owner **extraction**, effective from 2026-09-05, source `src.services.formulas.definitions.extraction:110`.

Two untestable states exist because abstaining is not the same as passing. Roughly a fifth of real lines have quantity = 1, where a unit rate and a total are numerically identical and no arithmetic can separate them -- exactly the blind spot that let a line total ship booked as a unit price. A fact whose role could not be verified must not look identical to one that was checked.

| Input | Unit | Range | Required |
|---|---|---|---|
| `quantity` | count | unbounded | no |
| `unit_rate` | money | unbounded | no |
| `extended_line` | money | unbounded | no |

**Output:** `ArithmeticState` in label — consistent | inconsistent | untestable_quantity_one | untestable_missing_input

**Dependents:** *none*

### `linking.amount_agreement`

*Whether two amounts agree, decaying linearly to zero at 10% drift* — scalar formula, owner **linkage**, effective from 2026-09-05, source `src.services.formulas.definitions.linking:183`.

One of FOUR amount-tolerance conventions in this codebase (gap report D-7). Consolidating them changes numbers and is therefore a separate, deliberate decision.

| Input | Unit | Range | Required |
|---|---|---|---|
| `a` | money | unbounded | no |
| `b` | money | unbounded | no |
| `tol` | ratio | [0, 0.09] | no |

**Output:** `tuple` in ratio — (score 0-1, status)

**Dependents:** *none*

### `linking.cluster_dampening`

*Correlated-signal dampening factor within one evidence cluster* — scalar formula, owner **linkage**, effective from 2026-09-05, source `src.services.formulas.definitions.linking:107`.

| Input | Unit | Range | Required |
|---|---|---|---|
| `n_active` | count | [0, 64] | yes |

**Output:** `float` in factor — multiplier applied to the cluster's summed contribution

**Dependents:** *none*

### `linking.decision_band`

*Which action band a relationship confidence F falls into* — scalar formula, owner **linkage**, effective from 2026-09-05, source `src.services.formulas.definitions.linking:273`.

Cuts are 92 / 80 / 65 / 45, env-overridable nowhere -- they are module constants.

| Input | Unit | Range | Required |
|---|---|---|---|
| `F` | score_0_100 | [0, 100] | yes |

**Output:** `str` in label — auto_link | auto_link_with_warning | review | weak_relation | block_or_exception

**Dependents:** *none*

### `linking.line_pair_score`

*Similarity of two individual line items (description / qty / unit price)* — scalar formula, owner **linkage**, effective from 2026-09-05, source `src.services.formulas.definitions.linking:127`.

| Input | Unit | Range | Required |
|---|---|---|---|
| `a` | record | unbounded | yes |
| `b` | record | unbounded | yes |

**Output:** `float` in ratio — 0-1 weighted over only the sub-signals present on both

**Dependents:** *none*

### `linking.line_set_composite`

*Best-match line-set agreement with a source-extra coverage penalty* — scalar formula, owner **linkage**, effective from 2026-09-05, source `src.services.formulas.definitions.linking:152`.

A source covering a SUBSET of the target is not penalised (split shipment).

| Input | Unit | Range | Required |
|---|---|---|---|
| `source_lines` | rows | unbounded | no |
| `target_lines` | rows | unbounded | no |

**Output:** `tuple` in ratio — (score 0-1, status OK|WEAK|CONFLICT|MISSING)

**Dependents:** *none*

### `linking.location_agreement`

*Country and region agreement between two documents* — scalar formula, owner **linkage**, effective from 2026-09-05, source `src.services.formulas.definitions.linking:248`.

| Input | Unit | Range | Required |
|---|---|---|---|
| `a_country` | text | unbounded | no |
| `a_region` | text | unbounded | no |
| `b_country` | text | unbounded | no |
| `b_region` | text | unbounded | no |

**Output:** `tuple` in ratio — (score 0-1, status)

**Dependents:** *none*

### `linking.relationship_confidence`

*How confidently two procurement documents are the same relationship (F, 0-100)* — scalar formula, owner **linkage**, effective from 2026-09-05, source `src.services.formulas.definitions.linking:52`.

S (separation) and Q (evidence quality) are pinned to 1.0 in the engine. See docs/formula-registry-gap-report.md D-6: S=1.0 is sound for the 1:1 promotion path and is NOT sound for `deal_clustering.awarded_po`, which scores one bid against every PO and takes the best.

| Input | Unit | Range | Required |
|---|---|---|---|
| `source_row` | record | unbounded | yes |
| `target_row` | record | unbounded | yes |
| `profile_name` | label | unbounded | yes |
| `source_lines` | rows | unbounded | no |
| `target_lines` | rows | unbounded | no |
| `set_amount_usd` | money | [0, +inf] | no |

**Output:** `dict` in score_0_100 — F plus the full per-signal breakdown, band decision and stage values

**Dependents:** `tests/services/formulas/test_registered_formulas.py`

### `linking.temporal_plausibility`

*Whether a child document's date is plausible against its parent's order date* — scalar formula, owner **linkage**, effective from 2026-09-05, source `src.services.formulas.definitions.linking:218`.

| Input | Unit | Range | Required |
|---|---|---|---|
| `doc_date` | date | unbounded | no |
| `parent_order_date` | date | unbounded | no |
| `parent_due_date` | date | unbounded | no |

**Output:** `tuple` in ratio — (score 0-1, status)

**Dependents:** *none*

### `negotiation.batna_strength`

*The buyer's walk-away position from alternative quotes and trading history* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:607`.

Salvaged from NegotiationStrategyEngine._build_batna and select_strategy before that module was deleted (it was constructed at every boot and reachable from nothing). Two corrections in the move: a missing count is UNASSESSED rather than 0, and no-BATNA scores 0.0 rather than selecting the engine's most aggressive strategy (STRATEGY_ANCHORING, target_discount 0.15). Bars: 2 alternatives for strong, 5 orders for an established relationship -- both inherited from the engine and ungoverned.

| Input | Unit | Range | Required |
|---|---|---|---|
| `alternative_quotes` | count | [0, +inf] | no |
| `supplier_history_count` | count | [0, +inf] | no |

**Output:** `dict` in label — strength, score, confidence, narrative, findings

**Dependents:** *none*

### `negotiation.counter_plan`

*What to counter at, and whether to counter at all, given the round and the gap* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:41`.

`compute_decision` hardwires aggressiveness 0.75, leverage 0.6, urgency 0.3, risk_buffer_pct 0.06, min_abs_buffer 3.0, step_pct_of_gap 0.12 and ask_early_pay_disc 0.02, overriding the dataclass defaults. They are not governed and not configurable at the call site (gap report F22). Round behaviour: R1 anchors at x0.88 above a 10% gap else midpoint; R2 captures 60% of the gap; R3+ enforces target + max(3.0, target x 6%).

| Input | Unit | Range | Required |
|---|---|---|---|
| `current_offer` | money | [0, +inf] | yes |
| `target_price` | money | [0, +inf] | yes |
| `round` | count | [1, +inf] | no |
| `max_rounds` | count | [1, +inf] | no |
| `walkaway_price` | money | [0, +inf] | no |
| `currency` | currency_code | unbounded | no |
| `ask_early_pay_disc` | ratio | [0, 1] | no |
| `ask_lead_time_keep` | label | unbounded | no |
| `supplier_message_text` | text | unbounded | no |
| `offer_prev` | money | [0, +inf] | no |

**Output:** `dict` in money — decision, counter_price, asks, lead_time_request, message, log

**Dependents:** `src/agents/negotiation_agent.py`, `tests/services/formulas/test_registered_formulas.py`

### `negotiation.kraljic_quadrant`

*Kraljic quadrant and suggested negotiation style for a deal* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:161`.

Default bars `high_spend = 98,175` and `many_alternatives = 93` are the live deal-value p90 and per-deal median alternative count. They are distribution parameters compiled into source: when the corpus moves they go stale silently and nothing measures the drift (gap report D-10). `many_alternatives = 93` is also defined a second time as `signals.THIN_MARKET_ALTERNATIVES` (D-11). Note bp_supplier.supplier_type is NOT a Kraljic axis and is deliberately unused.

| Input | Unit | Range | Required |
|---|---|---|---|
| `signals` | record | unbounded | yes |
| `thresholds` | record | unbounded | no |

**Output:** `dict` in label — quadrant, reasons, confidence, style, style_reasons, indeterminate

**Dependents:** `src/services/negotiation_advice/advisor.py`

### `negotiation.market_context_score`

*How far market conditions argue for one lever* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:367`.

| Input | Unit | Range | Required |
|---|---|---|---|
| `lever` | label | unbounded | yes |
| `market` | record | unbounded | no |

**Output:** `tuple` in ratio — (score nudge, notes)

**Dependents:** *none*

### `negotiation.outlier_rails`

*Whether an offer breaches a review or escalation rail on price, volume or term* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:525`.

PROVISIONAL -- vectors pin current arithmetic, not validated behaviour. Both price rails test (reference - offer)/reference, so they fire when the supplier's offer is BELOW the reference -- an error/fraud check, not an authority limit. Nothing here stops US emitting a counter or an acceptance above the walk-away; `plan_counter` clamps to target_price, never to walkaway_price (audit 1.6). Constants NEG_MARKET_REVIEW_PCT 0.2, NEG_MARKET_ESCALATION_PCT 0.4, NEG_MAX_VOLUME_LIMIT 1000, NEG_MAX_TERM_DAYS 120, with x1.5 and x2 escalation multipliers. The escalation log line reads 'more than 20% below our walk-away price' while the constant tested is 0.4.

| Input | Unit | Range | Required |
|---|---|---|---|
| `supplier_offer` | money | [0, +inf] | no |
| `target_price` | money | [0, +inf] | no |
| `walkaway_price` | money | [0, +inf] | no |
| `market_floor` | money | [0, +inf] | no |
| `volume_units` | count | [0, +inf] | no |
| `term_days` | days | [0, +inf] | no |

**Output:** `dict` in label — requires_review, human_override, review_recommendation, alerts

**Dependents:** *none*

### `negotiation.play_rank`

*Rank playbook plays for a (supplier type, style) pair against live signals* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:244`.

Score = 1.0 + index x 0.01 + policy + performance + market. The index term is a tie-break that preserves playbook order, not a judgement.

| Input | Unit | Range | Required |
|---|---|---|---|
| `supplier_type` | label | unbounded | no |
| `negotiation_style` | label | unbounded | no |
| `lever_priorities` | rows | unbounded | no |
| `policy_guidance` | record | unbounded | no |
| `supplier_performance` | record | unbounded | no |
| `market_context` | record | unbounded | no |
| `playbook` | record | unbounded | no |
| `limit` | count | [1, +inf] | no |

**Output:** `dict` in ratio — ranked plays with scores, rationale and trade-offs

**Dependents:** `src/services/negotiation_advice/advisor.py`

### `negotiation.play_readiness`

*Whether a play's precondition actually holds on this deal's evidence* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:389`.

A tri-state that already existed before this registry did, and the closest prior art in the codebase to UNASSESSED. 'Leverage competitor quotes' is strong with a second quote and damaging without one -- a buyer who bluffs a competing quote they do not have loses standing. A known sole-source deal returns not_applicable rather than groundwork: there is no second supplier to go and find, so advising a competitive event would be noise.

| Input | Unit | Range | Required |
|---|---|---|---|
| `play` | record | unbounded | yes |
| `signals` | record | unbounded | no |

**Output:** `dict` in label — the play plus state (ready|groundwork|not_applicable)

**Dependents:** *none*

### `negotiation.policy_alignment_score`

*How far governed policy pushes for or against one negotiation lever* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:307`.

Nudges are +0.6 required, +0.3 preferred, -0.3 discouraged, -0.7 restricted.

| Input | Unit | Range | Required |
|---|---|---|---|
| `lever` | label | unbounded | yes |
| `guidance` | record | unbounded | no |

**Output:** `tuple` in ratio — (score nudge, human-readable notes)

**Dependents:** *none*

### `negotiation.supplier_performance_score`

*How far a supplier's measured performance argues for one lever* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:337`.

An empty performance dict yields (0.0, []) -- the honest 'no signal' outcome. `negotiation_advice.signals.supplier_performance_dict` deliberately omits unknown keys rather than defaulting them, so an unmeasured supplier produces no nudge instead of an invented one.

| Input | Unit | Range | Required |
|---|---|---|---|
| `lever` | label | unbounded | yes |
| `performance` | record | unbounded | no |

**Output:** `tuple` in ratio — (score nudge, notes)

**Dependents:** *none*

### `negotiation.threshold_confidence`

*How far a value sits from a classification bar, as a 0.5-1.0 confidence* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:221`.

| Input | Unit | Range | Required |
|---|---|---|---|
| `value` | ratio | unbounded | no |
| `bar` | ratio | unbounded | no |

**Output:** `float` in ratio — 0.5 at the bar, rising to 1.0 away from it

**Dependents:** *none*

### `negotiation.zopa_estimate`

*Buyer's ceiling, the supplier's cost floor if any evidence exists, and an entry counter* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:465`.

PROVISIONAL -- vectors pin current arithmetic, not validated behaviour. v2.0.0 REMOVED the `price * 0.85` fallback floor (authorised behaviour change, 2026-09-05): with no should-cost, no benchmark and no history the floor is None and a finding is emitted, where it previously invented a number from the supplier's own offer. Nothing in this repository produces a should_cost, so the None branch is the ordinary case. `entry_counter` remains heuristic: price x (1 - clamp(concession, 0.03, 0.12)), default concession 0.05, all four constants ungoverned.

| Input | Unit | Range | Required |
|---|---|---|---|
| `price` | money | [0, +inf] | no |
| `target` | money | [0, +inf] | no |
| `history` | record | unbounded | no |
| `benchmarks` | record | unbounded | no |
| `should_cost` | money | [0, +inf] | no |
| `signals` | record | unbounded | no |

**Output:** `dict` in money — buyer_max, supplier_floor, supplier_floor_basis, entry_counter, findings

**Dependents:** *none*

### `opportunity.finding_weight_factor`

*One finding's unnormalised weight: money at stake, amplified by risk and coverage* — scalar formula, owner **opportunity**, effective from 2026-09-05, source `src.services.formulas.definitions.opportunity:54`.

The guard is load-bearing: a finding with real money behind it must never fall to a zero weight because risk or coverage arrived out of range, or it would vanish from the ranking entirely -- the opposite of what a risk signal should do.

| Input | Unit | Range | Required |
|---|---|---|---|
| `base_impact` | GBP | [0, +inf] | yes |
| `risk_score` | ratio | [0, 1] | yes |
| `coverage` | ratio | [0, 1] | yes |

**Output:** `float` in GBP — unnormalised weight factor

**Dependents:** `src/agents/opportunity_miner_agent.py`, `tests/services/formulas/test_registered_formulas.py`

### `opportunity.price_variance_impact`

*Money at stake when a paid price sits above a benchmark* — scalar formula, owner **opportunity**, effective from 2026-09-05, source `src.services.formulas.definitions.opportunity:106`.

The shape `(actual - benchmark) x quantity` recurs across several detectors (price benchmark variance, invoice-vs-PO variance, cheapest-alternative). Registered once so the shape has a name; the detectors are not yet rewired onto it, because each carries its own guards and rewiring them is a separate change with its own before/after numbers.

| Input | Unit | Range | Required |
|---|---|---|---|
| `actual_price` | money | [0, +inf] | yes |
| `benchmark_price` | money | [0, +inf] | yes |
| `quantity` | count | [0, +inf] | yes |

**Output:** `float` in GBP — positive = overpaying against the benchmark

**Dependents:** *none*

### `opportunity.risk_normalisation`

*Coerce a supplier risk value onto 0-1, whatever scale it arrived on* — scalar formula, owner **opportunity**, effective from 2026-09-05, source `src.services.formulas.definitions.opportunity:25`.

**Fails open (gap report D-4).** An unparseable value returns 0.0, i.e. NO risk. `risk_score` is stored as VARCHAR in proc.bp_supplier, so a malformed value is a live possibility, and it currently reads as the safest supplier on the table before multiplying into the finding weightage. Behaviour unchanged and pinned by the fourth vector.

| Input | Unit | Range | Required |
|---|---|---|---|
| `value` | ratio | unbounded | no |

**Output:** `float` in ratio — 0-1

**Dependents:** *none*

### `opportunity.weightage_shares`

*Turn a run's weight factors into shares of that run* — set formula, owner **opportunity**, effective from 2026-09-05, source `src.services.formulas.definitions.opportunity:83`.

Population-scoped: a share only means anything relative to the rest of the run. A zero total yields zeros rather than an even 1/n split, which would invent a ranking the evidence does not support.

| Input | Unit | Range | Required |
|---|---|---|---|
| `factor` | GBP | [0, +inf] | yes |

**Output:** `list[float]` in ratio — shares summing to 1.0, or all zeros

**Dependents:** `src/agents/opportunity_miner_agent.py`

### `price_outlier.verdict`

*Whether a unit price is both statistically extreme and commercially material* — scalar formula, owner **assurance**, effective from 2026-09-05, source `src.services.formulas.definitions.price_outlier:23`.

Both tests must pass. Either alone is useless: the statistical test flags trivia when peers are nearly identical, and the materiality test flags ordinary price variety. Below `min_peers` the answer is 'not flagged' with no median -- an abstention, not a clearance.

| Input | Unit | Range | Required |
|---|---|---|---|
| `price` | money | [0, +inf] | yes |
| `peers` | rows | unbounded | yes |
| `settings` | record | unbounded | no |

**Output:** `Verdict` in money — flagged, peer_count, median, ratio, robust_score, severity

**Dependents:** `src/services/price_outlier/detector.py`, `tests/services/formulas/test_registered_formulas.py`

### `quote.weighting_score`

*Weighted 0-100 comparison score across competing quotes* — set formula, owner **sourcing**, effective from 2026-09-05, source `src.services.formulas.definitions.ranking:256`.

**Duplicates `supplier.composite_score` and disagrees with it (gap report D-3).** Both min-max normalise then take a weighted mean; this one scores a quote with no usable metric as **0.0**, which ranks it as the worst on the table, while supplier ranking returns NaN for the identical situation and documents why. Behaviour unchanged and pinned by the last vector, so a consolidation is a deliberate version bump with visible numbers.

| Input | Unit | Range | Required |
|---|---|---|---|
| `total_cost_gbp` | money | [0, +inf] | no |
| `tenure` | count | [0, +inf] | no |
| `volume` | count | [0, +inf] | no |
| `weights` | record | unbounded | no |

**Output:** `list[float]` in score_0_100 — 0-100 per quote

**Dependents:** *none*

### `requirement.completeness_score`

*Proportion of a requirement's required fields that are genuinely filled* — scalar formula, owner **analytics**, effective from 2026-09-05, source `src.services.formulas.definitions.deal:141`.

A blank string counts as missing. No fabrication: only genuinely-filled fields score.

| Input | Unit | Range | Required |
|---|---|---|---|
| `requirement` | rows | unbounded | no |
| `required_fields` | rows | unbounded | no |

**Output:** `tuple` in ratio — (0-1 score, list of missing field names)

**Dependents:** *none*

### `risk.predictive_supplier_score`

*Forward-looking supplier risk 0-1 from decayed incident signals and performance* — scalar formula, owner **risk**, effective from 2026-09-05, source `src.services.formulas.definitions.risk:26`.

**Fails open on missing data (gap report D-2).** An absent `on_time_delivery_rate` or `quality_score` defaults to 1.0 and an absent `anomaly_index` to 0.0, so a supplier we hold NO performance data for scores as a perfect performer. On a risk model that is the wrong direction: absence of evidence becomes evidence of safety. Behaviour is unchanged here and pinned by the second vector below, so a fix is a visible version bump.

| Input | Unit | Range | Required |
|---|---|---|---|
| `supplier_metrics` | record | unbounded | yes |
| `signals` | rows | unbounded | no |
| `as_of` | timestamp | unbounded | no |
| `half_life_hours` | hours | [1, +inf] | no |

**Output:** `dict` in ratio — score plus its signal and performance components

**Dependents:** *none*

### `rivalry.correlation`

*How strongly two bids look like rivals for the same sourcing event* — scalar formula, owner **linkage**, effective from 2026-09-05, source `src.services.formulas.definitions.rivalry:33`.

The quote_rival profile deliberately carries NO supplier_id and NO exact-amount signal: divergence on those is the signature of a competitive event, not a defect. p0=0.03 / alpha=0.55 were calibrated against tests/fixtures/deal_clustering/golden_batch.py.

| Input | Unit | Range | Required |
|---|---|---|---|
| `bid_a` | record | unbounded | yes |
| `bid_b` | record | unbounded | yes |
| `lines_a` | rows | unbounded | no |
| `lines_b` | rows | unbounded | no |

**Output:** `dict` in ratio — score_link's full result plus `correlation` = F/100 in [0,1]

**Dependents:** `src/services/deal_clustering.py`

### `rivalry.description_overlap`

*Token Jaccard over two bids' aggregated line descriptions* — scalar formula, owner **linkage**, effective from 2026-09-05, source `src.services.formulas.definitions.rivalry:70`.

| Input | Unit | Range | Required |
|---|---|---|---|
| `lines_a` | rows | unbounded | no |
| `lines_b` | rows | unbounded | no |

**Output:** `tuple` in ratio — (score 0-1, status)

**Dependents:** *none*

### `rivalry.price_proximity`

*Graded price closeness between two bids (proximity, never equality)* — scalar formula, owner **linkage**, effective from 2026-09-05, source `src.services.formulas.definitions.rivalry:109`.

| Input | Unit | Range | Required |
|---|---|---|---|
| `row_a` | record | unbounded | yes |
| `row_b` | record | unbounded | yes |

**Output:** `tuple` in ratio — (min/max ratio, status)

**Dependents:** *none*

### `rivalry.volume_agreement`

*Quantity-total ratio between two bids* — scalar formula, owner **linkage**, effective from 2026-09-05, source `src.services.formulas.definitions.rivalry:87`.

Services deals carry no quantity at all (lump-sum lines), so MISSING is the normal outcome for them rather than an anomaly.

| Input | Unit | Range | Required |
|---|---|---|---|
| `lines_a` | rows | unbounded | no |
| `lines_b` | rows | unbounded | no |

**Output:** `tuple` in ratio — (score 0-1, status)

**Dependents:** *none*

### `supplier.composite_score`

*Weighted composite supplier score over only the criteria we hold for them* — set formula, owner **sourcing**, effective from 2026-09-05, source `src.services.formulas.definitions.ranking:200`.

Each supplier's weights are renormalised over the criteria they actually have, so a gap in OUR data is not charged to THEIR bid. A supplier with nothing measurable scores NaN rather than 0.0 -- we have no opinion, and 0 would be inventing one. Compare `quote.weighting_score`, which returns 0.0 for the same situation (gap report D-3).

| Input | Unit | Range | Required |
|---|---|---|---|
| `scores` | record | unbounded | yes |
| `weights` | record | unbounded | yes |

**Output:** `list[float]` in score_0_100 — composite per supplier; NaN when no criterion was measurable

**Dependents:** `src/agents/supplier_ranking_agent.py`, `tests/services/formulas/test_registered_formulas.py`

### `supplier.criterion_normalisation`

*Min-max normalise one raw criterion to 0-100 across the suppliers being compared* — set formula, owner **sourcing**, effective from 2026-09-05, source `src.services.formulas.definitions.ranking:119`.

All-unmeasured yields NaN for everyone, so `supplier.weight_renormalisation` drops the criterion rather than dragging every composite down by its weight. When every MEASURED supplier ties, only the measured ones score 100 -- assigning the tied score to the whole column once handed an unmeasured supplier a perfect 100 they never earned.

| Input | Unit | Range | Required |
|---|---|---|---|
| `value` | ratio | unbounded | no |
| `direction` | label | one of 'lower_is_better', 'higher_is_better' | yes |

**Output:** `list[float]` in score_0_100 — 0-100 per supplier; NaN where unmeasured

**Dependents:** *none*

### `supplier.deal_price_score`

*Price scored against the cheapest RIVAL bid on the same deal (100 = cheapest)* — set formula, owner **sourcing**, effective from 2026-09-05, source `src.services.formulas.definitions.ranking:82`.

Ratio-to-best, not min-max. Min-max stretched whatever gap existed across the full 0-100 scale: on the live TEST005 deal it scored a bid 1.1% more expensive as 0.00 against the winner's 100.00, when all three bids sat within GBP 3,050 of each other. The ordering was right and the numbers slandered the runners-up. A lone bidder scores NaN, not 100 -- an uncontested quote is no evidence of a good price.

| Input | Unit | Range | Required |
|---|---|---|---|
| `price` | money | [0, +inf] | no |

**Output:** `list[float]` in score_0_100 — 100 x cheapest / price, 2dp; NaN for every bid when fewer than two bid

**Dependents:** `src/agents/supplier_ranking_agent.py`, `tests/services/formulas/test_registered_formulas.py`

### `supplier.payment_terms_score`

*Payment terms in days scored 0-100, longer terms being better for the buyer* — scalar formula, owner **sourcing**, effective from 2026-09-05, source `src.services.formulas.definitions.ranking:39`.

Returns None, not 0.0, for a supplier with no terms recorded. Zero would assert 'they offered the worst terms available', which is a claim about the supplier rather than about our data.

| Input | Unit | Range | Required |
|---|---|---|---|
| `payment_terms_days` | days | [0, +inf] | no |
| `min_days` | days | [0, +inf] | no |
| `max_days` | days | [0, +inf] | no |

**Output:** `float` in score_0_100 — 0-100, 2dp; None when no terms are on file

**Dependents:** `src/services/formulas/__init__.py`, `tests/services/formulas/test_registered_formulas.py`

### `supplier.weight_renormalisation`

*Drop criteria nobody can be scored on, and renormalise the rest to sum to 1* — scalar formula, owner **sourcing**, effective from 2026-09-05, source `src.services.formulas.definitions.ranking:161`.

A criterion only counts if a real `_score` column exists AND is not entirely NaN. Accepting one on the strength of its raw column let weights and scoring disagree: the weight map kept the criterion, the scoring loop skipped it for want of a score column, and its weight silently vanished from the composite.

| Input | Unit | Range | Required |
|---|---|---|---|
| `weights` | record | unbounded | yes |
| `scored` | record | unbounded | yes |

**Output:** `dict` in ratio — criterion -> normalised weight, summing to 1.0

**Dependents:** *none*


