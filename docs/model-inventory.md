# Model Inventory

Generated from the formula registry by `model_inventory()` — 2026-10-22 19:38 UTC.
Do not hand-edit: regenerate with `python -m src.services.formulas.inventory`.

**85 registered formulas.**

**47 with no call site outside their own module** — either not yet migrated onto `evaluate`, or dead: `benchmark.evidence_confidence`, `benchmark.excel_round`, `deal_clustering.awarded_po`, `deal_clustering.cluster_confidence`, `duplicate_invoice.date_proximity`, `duplicate_invoice.reference_proximity`, `extraction.amount_agreement`, `extraction.completeness_score`, `extraction.line_arithmetic_state`, `finance.average_per_period`, `finance.carrying_cost`, `finance.early_payment_decision`, `finance.net_present_value`, `finance.payback_years`, `forecast.accuracy`, `forecast.exponential_smoothing`, `inventory.abc_xyz`, `inventory.eoq`, `inventory.reorder_point`, `inventory.safety_stock`, `inventory.turns`, `linking.amount_agreement`, `linking.cluster_dampening`, `linking.decision_band`, `linking.line_pair_score`, `linking.line_set_composite`, `linking.location_agreement`, `linking.temporal_plausibility`, `negotiation.batna_strength`, `negotiation.market_context_score`, `negotiation.outlier_rails`, `negotiation.play_readiness`, `negotiation.policy_alignment_score`, `negotiation.supplier_performance_score`, `negotiation.threshold_confidence`, `negotiation.zopa_estimate`, `opportunity.price_variance_impact`, `opportunity.risk_normalisation`, `quality.ppm`, `quote.weighting_score`, `requirement.completeness_score`, `risk.predictive_supplier_score`, `rivalry.description_overlap`, `rivalry.price_proximity`, `rivalry.volume_agreement`, `supplier.weight_renormalisation`, `sustainability.scope3`

| Formula | Ver | Owner | Purpose | GPSS | Last validated | Vectors | Dependents |
|---|---|---|---|---|---|---|---|
| `benchmark.adjusted_price` | 1.0.0+d0939572 | commercial | Fair benchmark unit price for one quote line, and the variance against it | — | 2026-10-22 19:37:44 | 2 | 3 |
| `benchmark.evidence_confidence` | 1.0.0+8a74bdde | commercial | Confidence label for a benchmark, from how many comparable points backed it | — | 2026-10-22 19:37:44 | 5 | — |
| `benchmark.excel_round` | 1.0.0+b600c6e4 | commercial | Excel ROUND semantics: half away from zero | — | 2026-10-22 19:37:44 | 2 | — |
| `bids.normalise` | 1.0.0+f2eb58d4 | bids | Supplier bids on a like-for-like basis: same lines, units, currency and payment terms, with every adjustment shown beside the submitted price | — | 2026-10-22 19:37:44 | 2 | 1 |
| `cost.should_cost` | 1.0.0+baae1fb7 | cost | Bottom-up modelled estimate of what a manufactured part should cost: material, conversion, labour, overhead, SG&A, logistics and margin per unit, with a tornado of drivers | — | 2026-10-22 19:37:45 | 2 | 1 |
| `critic.addressable_value` | 1.0.0+5ea664fc | opportunity_critic | What is left of the detector's number after every haircut the tests justified | — | 2026-10-22 19:37:45 | 4 | 1 |
| `critic.anchor_age_days` | 1.0.0+84414816 | opportunity_critic | How old the comparator is, so staleness can be tested against a governed rule | — | 2026-10-22 19:37:45 | 3 | 1 |
| `critic.annualised_rate` | 1.0.0+82e26cae | opportunity_critic | Compound annual rate of change between an anchor price and the current one | — | 2026-10-22 19:37:45 | 5 | 2 |
| `critic.excess_over_index` | 1.0.0+07b239a9 | opportunity_critic | The part of a price rise that is above index -- the only part that is an opportunity | — | 2026-10-22 19:37:45 | 4 | 1 |
| `critic.fabricated_anchor` | 1.0.0+39078c99 | opportunity_critic | Whether an anchor price is an artefact of the miner's missing-data fallbacks | — | 2026-10-22 19:37:45 | 4 | 1 |
| `critic.friction_haircut` | 1.0.0+0de054cd | opportunity_critic | Deduction for the real cost of switching or renegotiating | — | 2026-10-22 19:37:45 | 3 | 1 |
| `critic.min_confidence` | 1.0.0+1d197c09 | opportunity_critic | A claim inherits the weakest confidence of the evidence it rests on | — | 2026-10-22 19:37:45 | 5 | 1 |
| `critic.normalise_unit_rate` | 1.0.0+2b9e905a | opportunity_critic | Reduce a line to a per-unit rate so anchor and current can be compared like for like | — | 2026-10-22 19:37:45 | 3 | 1 |
| `critic.relative_gap` | 1.0.0+ead52e09 | opportunity_critic | The gap as a proportion of its base -- 40% of GBP 3k is not 4% of GBP 3m | — | 2026-10-22 19:37:45 | 3 | 1 |
| `critic.unit_basis_match` | 1.0.0+622ed8ee | opportunity_critic | Whether anchor and current are priced on the same basis -- mismatch invalidates outright | — | 2026-10-22 19:37:45 | 4 | 1 |
| `critic.volume_delta` | 1.0.0+b847708d | opportunity_critic | Proportional change in volume between anchor and current, to test like-for-like | — | 2026-10-22 19:37:45 | 4 | 1 |
| `deal.pct_change` | 1.0.0+39441e5c | analytics | Percentage change of a value against a baseline | — | 2026-10-22 19:37:45 | 4 | 7 |
| `deal.realised_savings` | 1.0.0+09ac2b97 | analytics | Money actually saved: the quote-to-invoice unit price gap times invoiced volume | — | 2026-10-22 19:37:45 | 3 | 2 |
| `deal.scaled_total` | 1.0.0+c758e942 | analytics | A unit amount scaled by a count (a monthly value over twelve months, a unit price over a quantity) | — | 2026-10-22 19:37:45 | 4 | 1 |
| `deal.share_pct` | 1.0.0+4e079d17 | analytics | One part as a percentage of a whole (a supplier's share of spend, a tail's share of invoices) | — | 2026-10-22 19:37:45 | 5 | 1 |
| `deal.weighted_unit_price` | 1.0.0+08d83ed7 | analytics | Volume-weighted unit price across a document type's line items | — | 2026-10-22 19:37:45 | 2 | 1 |
| `deal_clustering.awarded_po` | 1.0.0+7b0692e8 | linkage | Which purchase order a bid won, by continuity scoring rather than name matching | — | 2026-10-22 19:37:45 | 1 | — |
| `deal_clustering.cluster_confidence` | 1.0.0+dc15cf32 | linkage | Confidence that every member of a cluster belongs to the same sourcing event | — | 2026-10-22 19:37:45 | 2 | — |
| `demand.price_harmonisation` | 1.0.0+dc2180c9 | demand | The saving from buying one identical item at a reference price (a percentile of the prices paid, not the lowest) instead of the higher prices actually paid | — | 2026-10-22 19:37:45 | 2 | 1 |
| `duplicate_invoice.date_proximity` | 1.0.0+a120547b | assurance | How close two invoice dates are, on a graded ladder | — | 2026-10-22 19:37:45 | 3 | — |
| `duplicate_invoice.pair_score` | 1.0.0+bc300d89 | assurance | How likely two invoices are the same invoice billed twice | — | 2026-10-22 19:37:45 | 2 | 1 |
| `duplicate_invoice.reference_proximity` | 1.0.0+ef75b67d | assurance | Whether two invoice references are the same reference | — | 2026-10-22 19:37:45 | 3 | — |
| `extraction.amount_agreement` | 1.0.0+d7f5c294 | extraction | Whether two money amounts agree within the three-way-match tolerance | — | 2026-10-22 19:37:45 | 3 | — |
| `extraction.completeness_score` | 1.0.0+facea3ec | extraction | How much of a document's schema the extraction actually filled (0-100) | — | 2026-10-22 19:37:45 | 3 | — |
| `extraction.line_arithmetic_state` | 1.0.0+93f89c05 | extraction | Whether quantity x unit rate = extended line corroborates the assigned roles | — | 2026-10-22 19:37:45 | 4 | — |
| `finance.average_per_period` | 1.0.0+a817af6d | analytics | A total spread evenly over a number of periods (a monthly run rate from a year-to-date figure, a weekly consumption from a quarter's quantity) | — | 2026-10-22 19:37:45 | 4 | — |
| `finance.carrying_cost` | 1.0.0+49aa161b | analytics | What holding or releasing an amount for a number of days is worth at an annual rate (a payment-terms change, an early-payment discount, stock held) | — | 2026-10-22 19:37:45 | 4 | — |
| `finance.early_payment_decision` | 1.0.0+c38b3526 | analytics | Whether an early-payment discount beats paying at net days: the discount's annualised return against the cost of capital | — | 2026-10-22 19:37:45 | 3 | — |
| `finance.net_present_value` | 1.0.0+0f9fe2b0 | analytics | Net present value of dated cash flows at an annual rate; year 0 is undiscounted, costs negative, benefits positive | — | 2026-10-22 19:37:45 | 4 | — |
| `finance.payback_years` | 1.0.0+2282289a | analytics | Simple payback: years for an annual benefit to repay an initial cost, undiscounted | — | 2026-10-22 19:37:45 | 5 | — |
| `forecast.accuracy` | 1.0.0+7d70bad6 | analytics | Forecast accuracy: mean absolute percentage error, weighted MAPE and bias over paired actuals and forecasts | — | 2026-10-22 19:37:52 | 4 | — |
| `forecast.exponential_smoothing` | 1.0.0+f29e2a71 | analytics | Simple exponential smoothing of a demand series, with the next-period forecast | — | 2026-10-22 19:37:52 | 4 | — |
| `inventory.abc_xyz` | 1.0.0+6bba11a6 | analytics | ABC class by share of annual value (cumulative 80/95 percent) and XYZ class by demand variability (coefficient of variation) | — | 2026-10-22 19:37:52 | 2 | — |
| `inventory.eoq` | 1.0.0+e9f71e1d | analytics | Economic order quantity: the order size that balances ordering cost against holding cost | — | 2026-10-22 19:37:52 | 4 | — |
| `inventory.reorder_point` | 1.0.0+82989e9d | analytics | Reorder point: demand over the lead time plus safety stock | — | 2026-10-22 19:37:52 | 3 | — |
| `inventory.safety_stock` | 1.0.0+f7dfe988 | analytics | Safety stock for a stated service level from demand variability over the lead time | — | 2026-10-22 19:37:52 | 4 | — |
| `inventory.turns` | 1.0.0+eefb0e41 | analytics | Inventory turns and days of inventory from cost of goods sold and average inventory value | — | 2026-10-22 19:37:52 | 3 | — |
| `landed_cost.total_cost` | 1.0.0+d672ffee | cost | What it costs to get the ordered goods to the buyer's delivery point, with every cost that is owed but not stated flagged rather than counted as zero | — | 2026-10-22 19:37:45 | 3 | 4 |
| `linking.amount_agreement` | 1.0.0+9136b135 | linkage | Whether two amounts agree, decaying linearly to zero at 10% drift | — | 2026-10-22 19:37:45 | 4 | — |
| `linking.cluster_dampening` | 1.0.0+89fdee3d | linkage | Correlated-signal dampening factor within one evidence cluster | — | 2026-10-22 19:37:45 | 3 | — |
| `linking.decision_band` | 1.0.0+4f959f09 | linkage | Which action band a relationship confidence F falls into | — | 2026-10-22 19:37:45 | 5 | — |
| `linking.line_pair_score` | 1.0.0+cb656046 | linkage | Similarity of two individual line items (description / qty / unit price) | — | 2026-10-22 19:37:45 | 2 | — |
| `linking.line_set_composite` | 1.0.0+b4cdfa68 | linkage | Best-match line-set agreement with a source-extra coverage penalty | — | 2026-10-22 19:37:45 | 3 | — |
| `linking.location_agreement` | 1.0.0+eac1a040 | linkage | Country and region agreement between two documents | — | 2026-10-22 19:37:45 | 2 | — |
| `linking.relationship_confidence` | 1.0.0+3e3c340e | linkage | How confidently two procurement documents are the same relationship (F, 0-100) | — | 2026-10-22 19:37:45 | 2 | 1 |
| `linking.temporal_plausibility` | 1.0.0+ba1d1b32 | linkage | Whether a child document's date is plausible against its parent's order date | — | 2026-10-22 19:37:45 | 4 | — |
| `negotiation.batna_strength` | 1.0.0+aeabae8b | negotiation | The buyer's walk-away position from alternative quotes and trading history | — | 2026-10-22 19:37:52 | 5 | — |
| `negotiation.counter_plan` | 1.1.0+9952a0ef | negotiation | What to counter at, and whether to counter at all, given the round and the gap | — | 2026-10-22 19:37:52 | 8 | 2 |
| `negotiation.kraljic_quadrant` | 1.1.0+c696bef0 | negotiation | Kraljic quadrant and suggested negotiation style for a deal | — | 2026-10-22 19:37:52 | 7 | 1 |
| `negotiation.market_context_score` | 1.0.0+60b018ae | negotiation | How far market conditions argue for one lever | — | 2026-10-22 19:37:52 | 2 | — |
| `negotiation.outlier_rails` | 1.0.0+35afd9c8 | negotiation | Whether an offer breaches a review or escalation rail on price, volume or term | — | 2026-10-22 19:37:52 | 5 | — |
| `negotiation.play_rank` | 1.0.0+9c252e65 | negotiation | Rank playbook plays for a (supplier type, style) pair against live signals | — | 2026-10-22 19:37:52 | 2 | 2 |
| `negotiation.play_readiness` | 1.2.0+b7b9d068 | negotiation | Whether a play's precondition actually holds on this deal's evidence | — | 2026-10-22 19:37:52 | 5 | — |
| `negotiation.policy_alignment_score` | 1.0.0+6224b68c | negotiation | How far governed policy pushes for or against one negotiation lever | — | 2026-10-22 19:37:52 | 3 | — |
| `negotiation.supplier_performance_score` | 1.0.0+fad2575a | negotiation | How far a supplier's measured performance argues for one lever | — | 2026-10-22 19:37:52 | 3 | — |
| `negotiation.threshold_confidence` | 1.0.0+08a1494e | negotiation | How far a value sits from a classification bar, as a 0.5-1.0 confidence | — | 2026-10-22 19:37:52 | 3 | — |
| `negotiation.zopa_estimate` | 2.0.0+3f03aa12 | negotiation | Buyer's ceiling, the supplier's cost floor if any evidence exists, and an entry counter | — | 2026-10-22 19:37:52 | 3 | — |
| `opportunity.finding_weight_factor` | 1.0.0+c361a830 | opportunity | One finding's unnormalised weight: money at stake, amplified by risk and coverage | — | 2026-10-22 19:37:52 | 2 | 2 |
| `opportunity.price_variance_impact` | 1.0.0+5527a64c | opportunity | Money at stake when a paid price sits above a benchmark | — | 2026-10-22 19:37:52 | 2 | — |
| `opportunity.risk_normalisation` | 1.0.0+bf07873b | opportunity | Coerce a supplier risk value onto 0-1, whatever scale it arrived on | — | 2026-10-22 19:37:52 | 5 | — |
| `opportunity.weightage_shares` | 1.0.0+b908dedf | opportunity | Turn a run's weight factors into shares of that run | — | 2026-10-22 19:37:52 | 2 | 1 |
| `otif.delivery_performance` | 1.0.0+f93ba46f | supplier_performance | The share of deliveries that were on time AND in full, over the deliveries that can be assessed | — | 2026-10-22 19:37:52 | 2 | 2 |
| `ppv.purchase_price_variance` | 1.0.0+dbb6b52c | supplier_performance | What was paid above (+) or below (-) the PO price, on the quantity billed | — | 2026-10-22 19:37:52 | 2 | 2 |
| `price_outlier.verdict` | 1.0.0+5f72312b | assurance | Whether a unit price is both statistically extreme and commercially material | — | 2026-10-22 19:37:52 | 5 | 2 |
| `quality.ppm` | 1.0.0+013b83d0 | analytics | Supplier defect rate in parts per million from defects found and units received or inspected | — | 2026-10-22 19:37:52 | 4 | — |
| `quote.weighting_score` | 1.0.0+903ffe16 | sourcing | Weighted 0-100 comparison score across competing quotes | — | 2026-10-22 19:37:52 | 2 | — |
| `requirement.completeness_score` | 1.0.0+b744edd2 | analytics | Proportion of a requirement's required fields that are genuinely filled | — | 2026-10-22 19:37:45 | 2 | — |
| `risk.predictive_supplier_score` | 1.0.0+573d6ee2 | risk | Forward-looking supplier risk 0-1 from decayed incident signals and performance | — | 2026-10-22 19:37:52 | 2 | — |
| `rivalry.correlation` | 1.0.0+955cbb2c | linkage | How strongly two bids look like rivals for the same sourcing event | — | 2026-10-22 19:37:52 | 2 | 1 |
| `rivalry.description_overlap` | 1.0.0+0565dc1f | linkage | Token Jaccard over two bids' aggregated line descriptions | — | 2026-10-22 19:37:52 | 2 | — |
| `rivalry.price_proximity` | 1.0.0+1580e47b | linkage | Graded price closeness between two bids (proximity, never equality) | — | 2026-10-22 19:37:52 | 2 | — |
| `rivalry.volume_agreement` | 1.0.0+d7b35f94 | linkage | Quantity-total ratio between two bids | — | 2026-10-22 19:37:52 | 2 | — |
| `scoring.weighted_total` | 1.0.0+09f22cb8 | scoring | Ranked supplier totals from evaluator scores and criterion weights, with gates, a named normalisation, consistency flags and weight sensitivity | — | 2026-10-22 19:37:52 | 2 | 2 |
| `supplier.composite_score` | 1.0.0+488a3cd8 | sourcing | Weighted composite supplier score over only the criteria we hold for them | — | 2026-10-22 19:37:52 | 2 | 3 |
| `supplier.criterion_normalisation` | 1.0.0+2bc2ceb8 | sourcing | Min-max normalise one raw criterion to 0-100 across the suppliers being compared | — | 2026-10-22 19:37:52 | 4 | 2 |
| `supplier.deal_price_score` | 1.0.0+d877d447 | sourcing | Price scored against the cheapest RIVAL bid on the same deal (100 = cheapest) | — | 2026-10-22 19:37:52 | 3 | 2 |
| `supplier.payment_terms_score` | 1.0.0+36b959b6 | sourcing | Payment terms in days scored 0-100, longer terms being better for the buyer | — | 2026-10-22 19:37:52 | 6 | 4 |
| `supplier.weight_renormalisation` | 1.0.0+cca0ca38 | sourcing | Drop criteria nobody can be scored on, and renormalise the rest to sum to 1 | — | 2026-10-22 19:37:52 | 2 | — |
| `sustainability.scope3` | 1.0.0+5164408c | analytics | Scope 3 emissions from activity data and emission factors the user supplies, per line and in total | — | 2026-10-22 19:37:52 | 2 | — |
| `tco.total_cost` | 1.0.0+2ad3ef49 | cost | The present value, at the evaluation date, of every cash outflow to acquire, use and dispose of the goods over the horizon, minus inflows | — | 2026-10-22 19:37:45 | 2 | 5 |

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

### `bids.normalise`

*Supplier bids on a like-for-like basis: same lines, units, currency and payment terms, with every adjustment shown beside the submitted price* — scalar formula, owner **bids**, effective from 2026-10-21, source `src.services.formulas.definitions.bids:24`.

Per cell: tier at the evaluated volume (a break uses the higher tier), price per unit and per master unit of measure, currency at the supplied rate, then payment terms: price x (1 + r) ^ ((target days - bid days) / 365). Freight, insurance and exception loadings are shown as bid-level adjustments beside the price. A rate that is missing, zero or negative makes the line non-comparable, never one-to-one. A bid on a different Incoterm with no priced adjustment, or on different payment terms with no cost of capital, is listed as not normalised. Alternates are kept out of the base total. A missing line is flagged by default; any fill is labelled an assumption. Sealed events refuse a comparison before the opening time. Submitted values are never changed.

| Input | Unit | Range | Required |
|---|---|---|---|
| `line_master` | rows | unbounded | yes |
| `bids` | rows | unbounded | yes |
| `base_currency` | currency_code | unbounded | yes |
| `fx_rates` | record | unbounded | yes |
| `fx_type` | text | unbounded | no |
| `fx_date` | text | unbounded | no |
| `target_incoterm` | text | unbounded | no |
| `target_payment_days` | days | [0, +inf] | no |
| `cost_of_capital` | ratio | [0, 1] | no |
| `uom_conversions` | rows | unbounded | no |
| `missing_line_policy` | label | one of 'flag_only', 'exclude_supplier_from_total', 'fill_with_highest_bid', 'fill_with_average_bid', 'fill_with_last_paid_price' | no |
| `last_paid_prices` | record | unbounded | no |
| `award_date` | text | unbounded | no |
| `sealed` | boolean | unbounded | no |
| `opened_at` | text | unbounded | no |
| `as_of` | text | unbounded | no |

**Output:** `dict` in ratio — matrix (line x supplier cells tracing raw to normalised price), suppliers (totals, completeness, missing and unmatched lines, bid adjustments), exceptions, alternates, lowest_per_line, cherry_pick_total, best_single_supplier, warnings, assumptions; UNASSESSED if the inputs cannot be normalised

**Dependents:** `tests/services/bids/test_registered.py`

### `cost.should_cost`

*Bottom-up modelled estimate of what a manufactured part should cost: material, conversion, labour, overhead, SG&A, logistics and margin per unit, with a tornado of drivers* — scalar formula, owner **cost**, effective from 2026-10-20, source `src.services.formulas.definitions.cost:192`.

yield = 1 - scrap; gross = net / yield; material = gross x price less scrap at the recovery share. Conversion per unit = (cycle / 3600) / (cavities x oee) x machine rate + setup hours x rate / batch. Labour per unit = the same hours x operators x labour rate. Overhead on conversion plus labour; SG&A on manufacturing cost; margin on value-add only unless margin_base is total_cost. Every price and rate is supplied by the caller and listed with its source: a missing labour rate, an oee outside (0, 1], cavities below 1 or scrap of 1 or more make the whole estimate UNASSESSED. It is a modelled estimate, never a supplier's actual cost.

| Input | Unit | Range | Required |
|---|---|---|---|
| `bom` | rows | unbounded | yes |
| `routing` | rows | unbounded | yes |
| `batch_size` | count | [1, +inf] | yes |
| `overhead_pct` | ratio | [0, +inf] | yes |
| `sga_pct` | ratio | [0, +inf] | yes |
| `margin_pct` | ratio | [0, +inf] | yes |
| `logistics_per_unit` | money | unbounded | no |
| `currency` | currency_code | unbounded | no |
| `margin_base` | label | one of 'value_add_only', 'total_cost' | no |
| `annual_volume` | count | [0, +inf] | no |
| `sensitivity_pct` | percent | [0, +inf] | no |
| `monte_carlo_n` | count | [0, +inf] | no |
| `seed` | count | unbounded | no |
| `distributions` | record | unbounded | no |
| `quote_breakdown` | record | unbounded | no |

**Output:** `dict` in factor — unit_cost, per_unit by element, shares, operations, annual, rates_used with source, assumptions, sensitivity (tornado, monte_carlo), quote_gap, label 'modelled estimate'; UNASSESSED if the inputs cannot be costed

**Dependents:** `tests/services/cost/test_cost_formulas.py`

### `critic.addressable_value`

*What is left of the detector's number after every haircut the tests justified* — scalar formula, owner **opportunity_critic**, effective from 2026-09-09, source `src.services.formulas.definitions.critic:296`.

Clamped at BOTH ends, and the upper clamp is the load-bearing one. The prompt forbids ever emitting VALID with a value above the detector's, so a negative haircut must not be able to raise it. Enforcing that here as well as in the invariant guard means a bad policy row cannot inflate a number even if the guard is bypassed.

| Input | Unit | Range | Required |
|---|---|---|---|
| `detector_proposed` | money | unbounded | yes |
| `haircuts` | text | unbounded | no |

**Output:** `float` in money — addressable value, floored at 0 and capped at the detector's own figure

**Dependents:** `tests/services/formulas/test_critic_value.py`

### `critic.anchor_age_days`

*How old the comparator is, so staleness can be tested against a governed rule* — scalar formula, owner **opportunity_critic**, effective from 2026-09-09, source `src.services.formulas.definitions.critic:123`.

An undated anchor is UNASSESSED, not age zero. The live detector emits no anchor date at all (spec 6.1), so this returning UNASSESSED is the normal case until the evidence subagent dates it via invoices.

| Input | Unit | Range | Required |
|---|---|---|---|
| `anchor_date` | date | unbounded | no |
| `current_date` | date | unbounded | no |

**Output:** `int` in days — days between, or UNASSESSED if either is undated

**Dependents:** `tests/services/formulas/test_critic_anchor.py`

### `critic.annualised_rate`

*Compound annual rate of change between an anchor price and the current one* — scalar formula, owner **opportunity_critic**, effective from 2026-09-09, source `src.services.formulas.definitions.critic:37`.

Returns UNASSESSED rather than a number when the anchor is zero or the span is zero. A zero anchor is not a 0% rise; it is an unusable comparator, and returning 0.0 would present that as 'no change'.

| Input | Unit | Range | Required |
|---|---|---|---|
| `anchor_value` | money | unbounded | yes |
| `current_value` | money | unbounded | yes |
| `years` | count | [0, +inf] | yes |

**Output:** `float` in percent — percent per year, or UNASSESSED

**Dependents:** `tests/services/formulas/test_critic_anchor.py`, `tests/services/test_critic_store.py`

### `critic.excess_over_index`

*The part of a price rise that is above index -- the only part that is an opportunity* — scalar formula, owner **opportunity_critic**, effective from 2026-09-09, source `src.services.formulas.definitions.critic:81`.

UNASSESSED with no index is the whole point. The prompt forbids guessing a rate, and there is no index source in this deployment -- so this formula returns UNASSESSED for every candidate until one is supplied, and the gap register says so out loud.

| Input | Unit | Range | Required |
|---|---|---|---|
| `annualised_pct` | percent | unbounded | yes |
| `index_pct` | percent | unbounded | no |
| `band_pp` | percent | [0, +inf] | yes |

**Output:** `float` in percent — excess percentage points, 0.0 if inside band, or UNASSESSED with no index

**Dependents:** `tests/services/formulas/test_critic_anchor.py`

### `critic.fabricated_anchor`

*Whether an anchor price is an artefact of the miner's missing-data fallbacks* — scalar formula, owner **opportunity_critic**, effective from 2026-09-09, source `src.services.formulas.definitions.critic:198`.

THIS RULE IS NOT IN THE ORIGINAL PROMPT. It exists because of what the live detector actually does (spec 6.1): at opportunity_miner_agent.py:5545 a missing quantity becomes 1.0 and a missing unit price becomes the whole line value, and the anchor is then a .min() over that column -- the aggregation that selects for whichever row is most corrupted downward. A unit price identical to its line value at quantity exactly 1.0 is the fingerprint. It INVALIDATES rather than downgrades: a fabricated floor is not a weak comparator, it is not a price.

| Input | Unit | Range | Required |
|---|---|---|---|
| `unit_price` | money | unbounded | no |
| `line_value` | money | unbounded | no |
| `quantity` | count | unbounded | no |

**Output:** `bool` in boolean — True if the anchor is an artefact, not a price

**Dependents:** `tests/services/formulas/test_critic_anchor.py`

### `critic.friction_haircut`

*Deduction for the real cost of switching or renegotiating* — scalar formula, owner **opportunity_critic**, effective from 2026-09-09, source `src.services.formulas.definitions.critic:372`.

No default band. An invented friction percentage is an invented number, and the governed policy row is the only source.

| Input | Unit | Range | Required |
|---|---|---|---|
| `gross_value` | money | unbounded | yes |
| `friction_pct` | percent | [0, 100] | no |

**Output:** `float` in money — the amount to deduct, or UNASSESSED without a band

**Dependents:** `tests/services/formulas/test_critic_value.py`

### `critic.min_confidence`

*A claim inherits the weakest confidence of the evidence it rests on* — scalar formula, owner **opportunity_critic**, effective from 2026-09-09, source `src.services.formulas.definitions.critic:402`.

Nothing upgrades evidence. An empty list is UNASSESSED, not CORROBORATED: 'we checked nothing' and 'we checked and it was fine' must not return the same answer. Ladder matches services/analytics/models.py:70.

| Input | Unit | Range | Required |
|---|---|---|---|
| `confidences` | text | unbounded | no |

**Output:** `str` in text — ASSERTED | CORROBORATED | UNASSESSED

**Dependents:** `tests/services/formulas/test_critic_value.py`

### `critic.normalise_unit_rate`

*Reduce a line to a per-unit rate so anchor and current can be compared like for like* — scalar formula, owner **opportunity_critic**, effective from 2026-09-09, source `src.services.formulas.definitions.critic:241`.

Zero or absent quantity is UNASSESSED. Dividing by a defaulted 1.0 is how the miner produced fabricated anchors in the first place.

| Input | Unit | Range | Required |
|---|---|---|---|
| `total_value` | money | unbounded | yes |
| `quantity` | count | unbounded | yes |

**Output:** `float` in money — value per unit, or UNASSESSED

**Dependents:** `tests/services/formulas/test_critic_value.py`

### `critic.relative_gap`

*The gap as a proportion of its base -- 40% of GBP 3k is not 4% of GBP 3m* — scalar formula, owner **opportunity_critic**, effective from 2026-09-09, source `src.services.formulas.definitions.critic:343`.

A percentage off a base too small to mean anything is noise. Live, one supplier's +345,261% was GBP 26.72 the year before.

| Input | Unit | Range | Required |
|---|---|---|---|
| `gap_value` | money | unbounded | yes |
| `base_value` | money | unbounded | yes |

**Output:** `float` in ratio — fraction of base, or UNASSESSED on a negligible base

**Dependents:** `tests/services/formulas/test_critic_value.py`

### `critic.unit_basis_match`

*Whether anchor and current are priced on the same basis -- mismatch invalidates outright* — scalar formula, owner **opportunity_critic**, effective from 2026-09-09, source `src.services.formulas.definitions.critic:155`.

Compared after case-folding and separator-stripping only. No synonym table: deciding that 'per user' means 'per seat' is a judgement, and judgement belongs to the agent, not to a formula.

| Input | Unit | Range | Required |
|---|---|---|---|
| `anchor_basis` | text | unbounded | no |
| `current_basis` | text | unbounded | no |

**Output:** `bool` in boolean — True if comparable, False if not, UNASSESSED if unknown

**Dependents:** `tests/services/formulas/test_critic_anchor.py`

### `critic.volume_delta`

*Proportional change in volume between anchor and current, to test like-for-like* — scalar formula, owner **opportunity_critic**, effective from 2026-09-09, source `src.services.formulas.definitions.critic:268`.

| Input | Unit | Range | Required |
|---|---|---|---|
| `anchor_qty` | count | unbounded | no |
| `current_qty` | count | unbounded | no |

**Output:** `float` in ratio — signed fraction of the anchor volume, or UNASSESSED

**Dependents:** `tests/services/formulas/test_critic_value.py`

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

**Dependents:** `src/orchestration/agentnick_control.py`, `src/services/cost/compare_tco.py`, `src/services/deal_analysis_service.py`, `tests/orchestration/test_argument_check.py`, `tests/services/scoring/test_decline_reason.py`, `tests/services/test_cost_compare.py`, `tests/test_calculation_tools.py`

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

**Dependents:** `src/services/deal_analysis_service.py`, `tests/skills/test_checks.py`

### `deal.scaled_total`

*A unit amount scaled by a count (a monthly value over twelve months, a unit price over a quantity)* — scalar formula, owner **analytics**, effective from 2026-10-22, source `src.services.formulas.definitions.deal:203`.

Registered so a projected or annual total is a traceable calculation rather than a figure the model multiplies in its head (live, 2026-10-22: 4,900 a month became 58,800 a year and the verifier struck it out). The multiplier is an assumption: it must come from the request, a result or the skill's settings, never be chosen by the model.

| Input | Unit | Range | Required |
|---|---|---|---|
| `unit_value` | money | unbounded | no |
| `multiplier` | count | [0, +inf] | no |

**Output:** `float` in money — unit_value times multiplier, 2dp; None if either is absent

**Dependents:** `tests/orchestration/test_argument_check.py`

### `deal.share_pct`

*One part as a percentage of a whole (a supplier's share of spend, a tail's share of invoices)* — scalar formula, owner **analytics**, effective from 2026-10-22, source `src.services.formulas.definitions.deal:168`.

Registered so a share is a calculation the verifier can trace rather than arithmetic the model does in its head (live, 2026-10-22: a tail review divided two cube figures and then struck the result out as unverifiable). None against a zero or absent whole: a share of nothing is not 0 percent.

| Input | Unit | Range | Required |
|---|---|---|---|
| `part` | money | unbounded | no |
| `whole` | money | unbounded | no |

**Output:** `float` in percent — percentage points, 2dp; None against a zero or absent whole

**Dependents:** `tests/orchestration/test_argument_check.py`

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

### `demand.price_harmonisation`

*The saving from buying one identical item at a reference price (a percentile of the prices paid, not the lowest) instead of the higher prices actually paid* — scalar formula, owner **demand**, effective from 2026-10-21, source `src.services.formulas.definitions.demand:16`.

Reference price is the percentile by linear interpolation between the two nearest prices. Saving per line is max(0, price - reference) x quantity, so a line at or below the reference saves nothing. Incoterms, payment terms, currency and unit of measure are not adjusted: normalise first. Fewer prices than the minimum give no reference.

| Input | Unit | Range | Required |
|---|---|---|---|
| `lines` | rows | unbounded | yes |
| `percentile` | percent | [0, 100] | yes |
| `min_observations` | count | [1, +inf] | no |

**Output:** `dict` in percent — reference_price, lines (each with its saving), total_saving, lowest_price, highest_price, method, assumptions; UNASSESSED if the inputs cannot be harmonised

**Dependents:** `tests/services/demand/test_price_harmonisation.py`

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

### `finance.average_per_period`

*A total spread evenly over a number of periods (a monthly run rate from a year-to-date figure, a weekly consumption from a quarter's quantity)* — scalar formula, owner **analytics**, effective from 2026-10-22, source `src.services.formulas.definitions.finance:30`.

An average is not a forecast: it says nothing about seasonality. The caller names which periods it covers.

| Input | Unit | Range | Required |
|---|---|---|---|
| `period_total` | money | unbounded | no |
| `periods` | count | [0, +inf] | no |

**Output:** `float` in money — period_total / periods, 2dp; None when periods is zero or absent

**Dependents:** *none*

### `finance.carrying_cost`

*What holding or releasing an amount for a number of days is worth at an annual rate (a payment-terms change, an early-payment discount, stock held)* — scalar formula, owner **analytics**, effective from 2026-10-22, source `src.services.formulas.definitions.finance:86`.

Simple, not compounded: the figure finance quotes for a terms change. Registered because the payment-terms skill had been feeding a spend figure into tco.total_cost, which needs a landed-cost result and so never ran.

| Input | Unit | Range | Required |
|---|---|---|---|
| `amount` | money | [0, +inf] | no |
| `annual_rate` | ratio | [0, +inf] | no |
| `days` | count | [0, +inf] | no |

**Output:** `float` in money — amount * annual_rate * days / 365, 2dp; None when an input is absent

**Dependents:** *none*

### `finance.early_payment_decision`

*Whether an early-payment discount beats paying at net days: the discount's annualised return against the cost of capital* — scalar formula, owner **analytics**, effective from 2026-10-22, source `src.services.formulas.definitions.finance:116`.

The standard comparison: annualised return = discount / (100 - discount) * 365 / (net_days - discount_days). take_discount is true when that return beats the cost of capital; whether AP can reliably pay inside the window is the user's statement, not computed here.

| Input | Unit | Range | Required |
|---|---|---|---|
| `discount_pct` | percent | [0, +inf] | no |
| `discount_days` | count | [0, +inf] | no |
| `net_days` | count | [0, +inf] | no |
| `annual_rate` | ratio | [0, +inf] | no |

**Output:** `dict` in percent — {annualised_return_pct, cost_of_capital_pct, take_discount}; None when net_days is not after discount_days or an input is absent

**Dependents:** *none*

### `finance.net_present_value`

*Net present value of dated cash flows at an annual rate; year 0 is undiscounted, costs negative, benefits positive* — scalar formula, owner **analytics**, effective from 2026-10-22, source `src.services.formulas.definitions.finance:147`.

The rate is an assumption the policy or finance supplies; the argument check refuses one the model made up. A row without a numeric year and amount makes the whole result None rather than a partial sum.

| Input | Unit | Range | Required |
|---|---|---|---|
| `cash_flows` | rows | unbounded | yes |
| `cost_of_capital` | ratio | [0, +inf] | yes |

**Output:** `float` in money — sum of amount / (1 + rate) ** year, 2dp; None when a row is malformed or the rate is absent

**Dependents:** *none*

### `finance.payback_years`

*Simple payback: years for an annual benefit to repay an initial cost, undiscounted* — scalar formula, owner **analytics**, effective from 2026-10-22, source `src.services.formulas.definitions.finance:56`.

Undiscounted by design: it is the figure finance templates ask for beside the present value, not a substitute for it. A benefit of zero or less never pays back, and that is None, not infinity.

| Input | Unit | Range | Required |
|---|---|---|---|
| `initial_cost` | money | [0, +inf] | no |
| `annual_benefit` | money | unbounded | no |

**Output:** `float` in count — years, 2dp; None when the benefit is zero or negative or an input is absent

**Dependents:** *none*

### `forecast.accuracy`

*Forecast accuracy: mean absolute percentage error, weighted MAPE and bias over paired actuals and forecasts* — scalar formula, owner **analytics**, effective from 2026-10-22, source `src.services.formulas.definitions.planning:240`.

MAPE skips periods with a zero actual (and says how many); WMAPE and bias use the sums, so a zero actual is fine there. Bias positive means the forecast ran high.

| Input | Unit | Range | Required |
|---|---|---|---|
| `actuals` | rows | unbounded | yes |
| `forecasts` | rows | unbounded | yes |

**Output:** `dict` in percent — {mape_pct, wmape_pct, bias_pct, periods}; None when the lists differ in length, are empty, or actuals sum to zero

**Dependents:** *none*

### `forecast.exponential_smoothing`

*Simple exponential smoothing of a demand series, with the next-period forecast* — scalar formula, owner **analytics**, effective from 2026-10-22, source `src.services.formulas.definitions.planning:210`.

No trend or seasonality: it is the flat baseline a planner starts from, and the alpha is the planner's, never chosen here. Default alpha 0.3 when absent, said so by the caller.

| Input | Unit | Range | Required |
|---|---|---|---|
| `series` | rows | unbounded | yes |
| `alpha` | ratio | [0, +inf] | no |

**Output:** `dict` in count — {forecast, level, periods}; the forecast for the next period, 2dp; None when the series is empty or alpha is off range

**Dependents:** *none*

### `inventory.abc_xyz`

*ABC class by share of annual value (cumulative 80/95 percent) and XYZ class by demand variability (coefficient of variation)* — scalar formula, owner **analytics**, effective from 2026-10-22, source `src.services.formulas.definitions.planning:147`.

Defaults when the cutoffs are absent: A up to 80 percent of cumulative value, B up to 95, X at CV up to 0.5, Y up to 1.0. An item with fewer than two demand periods has no CV and is class Z (unknown variability), said so by the caller.

| Input | Unit | Range | Required |
|---|---|---|---|
| `items` | rows | unbounded | yes |
| `a_cutoff_pct` | percent | [0, +inf] | no |
| `b_cutoff_pct` | percent | [0, +inf] | no |
| `x_cv_max` | ratio | [0, +inf] | no |
| `y_cv_max` | ratio | [0, +inf] | no |

**Output:** `list` in count — one row per item: item, annual_value, value_share_pct, abc, cv, xyz; None when no item has a positive value

**Dependents:** *none*

### `inventory.eoq`

*Economic order quantity: the order size that balances ordering cost against holding cost* — scalar formula, owner **analytics**, effective from 2026-10-22, source `src.services.formulas.definitions.planning:36`.

The classic Wilson formula; it assumes steady demand and no price breaks. A price break is compared separately with deal.scaled_total and finance.carrying_cost.

| Input | Unit | Range | Required |
|---|---|---|---|
| `annual_demand` | count | [0, +inf] | no |
| `order_cost` | money | [0, +inf] | no |
| `holding_cost_per_unit` | money | [0, +inf] | no |

**Output:** `float` in count — sqrt(2 * annual_demand * order_cost / holding_cost_per_unit), 1dp; None when holding cost is zero or an input is absent

**Dependents:** *none*

### `inventory.reorder_point`

*Reorder point: demand over the lead time plus safety stock* — scalar formula, owner **analytics**, effective from 2026-10-22, source `src.services.formulas.definitions.planning:93`.

Safety stock absent is taken as zero and said so by the caller; demand or lead time absent is None.

| Input | Unit | Range | Required |
|---|---|---|---|
| `demand_per_period` | count | [0, +inf] | no |
| `lead_time_periods` | count | [0, +inf] | no |
| `safety_stock` | count | [0, +inf] | no |

**Output:** `float` in count — demand_per_period * lead_time_periods + safety_stock, 1dp; None when demand or lead time is absent

**Dependents:** *none*

### `inventory.safety_stock`

*Safety stock for a stated service level from demand variability over the lead time* — scalar formula, owner **analytics**, effective from 2026-10-22, source `src.services.formulas.definitions.planning:63`.

A service level off the fixed table is refused, not interpolated; the table is the usual one-sided normal factors.

| Input | Unit | Range | Required |
|---|---|---|---|
| `demand_sd_per_period` | count | [0, +inf] | no |
| `lead_time_periods` | count | [0, +inf] | no |
| `service_level_pct` | percent | unbounded | no |

**Output:** `float` in count — z * demand_sd * sqrt(lead_time), 1dp; None when the service level is off the table or an input is absent

**Dependents:** *none*

### `inventory.turns`

*Inventory turns and days of inventory from cost of goods sold and average inventory value* — scalar formula, owner **analytics**, effective from 2026-10-22, source `src.services.formulas.definitions.planning:120`.

Turns = COGS / average inventory; days = period_days / turns. Both figures are the user's or finance's.

| Input | Unit | Range | Required |
|---|---|---|---|
| `cost_of_goods_sold` | money | [0, +inf] | no |
| `average_inventory_value` | money | [0, +inf] | no |
| `period_days` | count | [0, +inf] | no |

**Output:** `dict` in ratio — {turns, days_of_inventory}; None when inventory or the period is zero or an input is absent

**Dependents:** *none*

### `landed_cost.total_cost`

*What it costs to get the ordered goods to the buyer's delivery point, with every cost that is owed but not stated flagged rather than counted as zero* — scalar formula, owner **cost**, effective from 2026-10-20, source `src.services.formulas.definitions.cost:32`.

Statuses: stated, included, not_applicable, derived, estimated, missing, and not_stated (an optional charge the quote does not mention: listed to confirm, not counted as missing). The Incoterm decides what the buyer owes; with none, every unstated item except goods is missing. An amount in a currency the dated FX table has no rate for is missing, never converted 1:1. Insurance defaults to policy rate x 110% x (goods + freight) and is then 'estimated'; duty is rate x customs value ('CIF' = goods + freight + insurance). A total with anything missing is a lower bound (status incomplete).

| Input | Unit | Range | Required |
|---|---|---|---|
| `lines` | rows | unbounded | yes |
| `buyer_currency` | currency_code | unbounded | yes |
| `cross_border` | boolean | unbounded | yes |
| `incoterm` | label | unbounded | no |
| `evaluation_date` | date | unbounded | no |
| `fx_rates` | record | unbounded | no |
| `fx_table_date` | date | unbounded | no |
| `components` | rows | unbounded | no |
| `duty_rate` | ratio | [0, 1] | no |
| `customs_basis` | label | one of 'CIF', 'transaction' | no |
| `default_insurance_rate` | ratio | [0, 1] | no |
| `vat_recoverable` | boolean | unbounded | no |

**Output:** `dict` in money — components (each with a status), landed_total, unit_landed, landed_status, missing, estimated, not_stated; UNASSESSED if the goods cannot be priced

**Dependents:** `src/services/cost/compare_tco.py`, `tests/orchestration/test_argument_check.py`, `tests/services/cost/test_cost_formulas.py`, `tests/services/test_cost_compare.py`

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

*The buyer's walk-away position from alternative quotes and trading history* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:646`.

Salvaged from NegotiationStrategyEngine._build_batna and select_strategy before that module was deleted (it was constructed at every boot and reachable from nothing). Two corrections in the move: a missing count is UNASSESSED rather than 0, and no-BATNA scores 0.0 rather than selecting the engine's most aggressive strategy (STRATEGY_ANCHORING, target_discount 0.15). Bars: 2 alternatives for strong, 5 orders for an established relationship -- both inherited from the engine and ungoverned.

| Input | Unit | Range | Required |
|---|---|---|---|
| `alternative_quotes` | count | [0, +inf] | no |
| `supplier_history_count` | count | [0, +inf] | no |

**Output:** `dict` in label — strength, score, confidence, narrative, findings

**Dependents:** *none*

### `negotiation.counter_plan`

*What to counter at, and whether to counter at all, given the round and the gap* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:41`.

`compute_decision` hardwires aggressiveness 0.75, leverage 0.6 and urgency 0.3, overriding the dataclass defaults. They are not governed and not configurable at the call site (gap report F22). Since 1.1.0 no ask carries a figure the deal did not state: the early-payment discount is cost of capital x term_days / 365 (none without term_days, unless the caller gives one), and the volume ask names volume_units (none without it). Round behaviour: R1 anchors at x0.88 above a 10% gap else midpoint; R2 captures 60% of the gap; R3+ enforces target + max(3.0, target x 6%).

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
| `term_days` | days | [0, +inf] | no |
| `volume_units` | count | [0, +inf] | no |

**Output:** `dict` in money — decision, counter_price, asks, lead_time_request, message, log

**Dependents:** `src/agents/negotiation_agent.py`, `tests/services/formulas/test_registered_formulas.py`

### `negotiation.kraljic_quadrant`

*Kraljic quadrant and suggested negotiation style for a deal* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:172`.

Since 1.1.0 spend is compared in pounds (`deal_value_gbp`; a foreign value with no rate is no verdict, never a native comparison) and `many_alternatives` is a supplier count, 3, not the corpus median (93, which went stale as the data grew: D-10). `signals.market_context_dict` reads the same governed bar (D-11 closed). A deal whose items only its own supplier has quoted carries no count, so it is no verdict until a buyer states one. Note bp_supplier.supplier_type is NOT a Kraljic axis and is deliberately unused.

| Input | Unit | Range | Required |
|---|---|---|---|
| `signals` | record | unbounded | yes |
| `thresholds` | record | unbounded | no |

**Output:** `dict` in label — quadrant, reasons, confidence, style, style_reasons, indeterminate

**Dependents:** `src/services/negotiation_advice/advisor.py`

### `negotiation.market_context_score`

*How far market conditions argue for one lever* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:393`.

| Input | Unit | Range | Required |
|---|---|---|---|
| `lever` | label | unbounded | yes |
| `market` | record | unbounded | no |

**Output:** `tuple` in ratio — (score nudge, notes)

**Dependents:** *none*

### `negotiation.outlier_rails`

*Whether an offer breaches a review or escalation rail on price, volume or term* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:564`.

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

*Rank playbook plays for a (supplier type, style) pair against live signals* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:270`.

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

**Dependents:** `src/services/negotiation_advice/advisor.py`, `tests/services/negotiation_advice/test_advisor.py`

### `negotiation.play_readiness`

*Whether a play's precondition actually holds on this deal's evidence* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:415`.

A tri-state that already existed before this registry did, and the closest prior art in the codebase to UNASSESSED. 'Leverage competitor quotes' is strong with a second quote and damaging without one -- a buyer who bluffs a competing quote they do not have loses standing. A known sole-source deal returns not_applicable rather than groundwork: there is no second supplier to go and find, so advising a competitive event would be noise. Since 1.1.0 a play no evidence family can test is groundwork, not ready (fail closed). Since 1.2.0 money evidence names its currency (the deal's), with pounds beside a foreign deal value; an unrecorded currency is said, not assumed.

| Input | Unit | Range | Required |
|---|---|---|---|
| `play` | record | unbounded | yes |
| `signals` | record | unbounded | no |

**Output:** `dict` in label — the play plus state (ready|groundwork|not_applicable)

**Dependents:** *none*

### `negotiation.policy_alignment_score`

*How far governed policy pushes for or against one negotiation lever* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:333`.

Nudges are +0.6 required, +0.3 preferred, -0.3 discouraged, -0.7 restricted.

| Input | Unit | Range | Required |
|---|---|---|---|
| `lever` | label | unbounded | yes |
| `guidance` | record | unbounded | no |

**Output:** `tuple` in ratio — (score nudge, human-readable notes)

**Dependents:** *none*

### `negotiation.supplier_performance_score`

*How far a supplier's measured performance argues for one lever* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:363`.

An empty performance dict yields (0.0, []) -- the honest 'no signal' outcome. No table holds on-time/OTIF data, so the negotiation advisor passes an empty dict rather than default one, and an unmeasured supplier produces no nudge instead of an invented one.

| Input | Unit | Range | Required |
|---|---|---|---|
| `lever` | label | unbounded | yes |
| `performance` | record | unbounded | no |

**Output:** `tuple` in ratio — (score nudge, notes)

**Dependents:** *none*

### `negotiation.threshold_confidence`

*How far a value sits from a classification bar, as a 0.5-1.0 confidence* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:247`.

| Input | Unit | Range | Required |
|---|---|---|---|
| `value` | ratio | unbounded | no |
| `bar` | ratio | unbounded | no |

**Output:** `float` in ratio — 0.5 at the bar, rising to 1.0 away from it

**Dependents:** *none*

### `negotiation.zopa_estimate`

*Buyer's ceiling, the supplier's cost floor if any evidence exists, and an entry counter* — scalar formula, owner **negotiation**, effective from 2026-09-05, source `src.services.formulas.definitions.negotiation:504`.

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

### `otif.delivery_performance`

*The share of deliveries that were on time AND in full, over the deliveries that can be assessed* — scalar formula, owner **supplier_performance**, effective from 2026-10-20, source `src.services.formulas.definitions.performance:78`.

Each rate is over the deliveries assessable for it. A delivery with no promised date, no receipt, no quantity or an unreadable date is unassessed for that measure, not a failure: our missing data is not charged to the supplier. The tolerances are inputs from governed policy, not defaults in code. As of 2026-10-20 the corpus holds no promised dates and no receipts, so on live data this answers UNASSESSED until they are loaded.

| Input | Unit | Range | Required |
|---|---|---|---|
| `deliveries` | rows | unbounded | yes |
| `late_tolerance_days` | days | [0, +inf] | yes |
| `in_full_tolerance` | ratio | [0, 1] | yes |
| `early_tolerance_days` | days | [0, +inf] | no |

**Output:** `dict` in ratio — on_time_rate, in_full_rate, otif_rate, otif_score (0-100) and the number assessed for each; UNASSESSED if nothing could be assessed

**Dependents:** `src/services/supplier_performance/scorecard.py`, `tests/services/supplier_performance/test_performance_formulas.py`

### `ppv.purchase_price_variance`

*What was paid above (+) or below (-) the PO price, on the quantity billed* — scalar formula, owner **supplier_performance**, effective from 2026-10-20, source `src.services.formulas.definitions.performance:24`.

(invoice unit price - PO unit price) x quantity per matched line; positive = the supplier billed MORE than the PO price. ppv_pct is of the standard spend (PO price x quantity). A line with no usable price, quantity or currency is counted as unassessed, never as zero variance. Mixed currencies are not added without a dated table: by_currency still reports each, and a currency with no rate is excluded and named, never taken 1:1.

| Input | Unit | Range | Required |
|---|---|---|---|
| `lines` | rows | unbounded | yes |
| `buyer_currency` | currency_code | unbounded | no |
| `fx_rates` | record | unbounded | no |
| `fx_table_date` | date | unbounded | no |

**Output:** `dict` in money — total_ppv, unfavourable, favourable, standard_spend, ppv_pct, by_currency, lines_assessed, lines_unassessed, excluded_currencies; UNASSESSED if no line could be assessed

**Dependents:** `src/services/supplier_performance/scorecard.py`, `tests/services/supplier_performance/test_performance_formulas.py`

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

### `quality.ppm`

*Supplier defect rate in parts per million from defects found and units received or inspected* — scalar formula, owner **analytics**, effective from 2026-10-22, source `src.services.formulas.definitions.quality:26`.

The denominator is what was inspected, not what was ordered; the caller says which.

| Input | Unit | Range | Required |
|---|---|---|---|
| `defective_units` | count | [0, +inf] | no |
| `units_inspected` | count | [0, +inf] | no |

**Output:** `float` in ratio — defective_units / units_inspected * 1,000,000, 1dp; None when nothing was inspected

**Dependents:** *none*

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

### `scoring.weighted_total`

*Ranked supplier totals from evaluator scores and criterion weights, with gates, a named normalisation, consistency flags and weight sensitivity* — scalar formula, owner **scoring**, effective from 2026-10-21, source `src.services.formulas.definitions.scoring:37`.

Weights at each sibling level must sum to 1 (decimal arithmetic, tolerance 1e-9); gates have no weight and a failed gate removes the supplier. Scored criteria are normalised on the scale; price and numeric criteria use the named method. Global weight is the product down the tree. Equal totals to six decimals are a tie with a shared rank: nothing breaks a tie quietly. A missing score is refused, never counted as zero. Krippendorff alpha, ICC and evaluator rank-reversal checks are NOT computed and are listed in not_computed. It ranks; it never chooses the winner.

| Input | Unit | Range | Required |
|---|---|---|---|
| `criteria` | rows | unbounded | yes |
| `scale_min` | factor | unbounded | yes |
| `scale_max` | factor | unbounded | yes |
| `scores` | rows | unbounded | yes |
| `metrics` | rows | unbounded | no |
| `consensus_scores` | rows | unbounded | no |
| `normalisation` | label | one of 'none', 'ratio_to_best', 'min_max', 'price_interpolation', 'rank_based' | no |
| `aggregation` | label | one of 'consensus_if_present', 'mean', 'median', 'trimmed_mean' | no |
| `price_reference` | money | unbounded | no |
| `monte_carlo_n` | count | [0, +inf] | no |
| `seed` | count | unbounded | no |

**Output:** `dict` in factor — ranking (total, rank, tied, contributions), excluded, global_weights, aggregated_scores, evaluator_flags, criterion_flags, sensitivity, warnings, not_computed, top_supplier, top_total; UNASSESSED if the inputs cannot be scored

**Dependents:** `tests/services/scoring/test_decline_reason.py`, `tests/services/scoring/test_registered.py`

### `supplier.composite_score`

*Weighted composite supplier score over only the criteria we hold for them* — set formula, owner **sourcing**, effective from 2026-09-05, source `src.services.formulas.definitions.ranking:200`.

Each supplier's weights are renormalised over the criteria they actually have, so a gap in OUR data is not charged to THEIR bid. A supplier with nothing measurable scores NaN rather than 0.0 -- we have no opinion, and 0 would be inventing one. Compare `quote.weighting_score`, which returns 0.0 for the same situation (gap report D-3).

| Input | Unit | Range | Required |
|---|---|---|---|
| `scores` | record | unbounded | yes |
| `weights` | record | unbounded | yes |

**Output:** `list[float]` in score_0_100 — composite per supplier; NaN when no criterion was measurable

**Dependents:** `src/agents/supplier_ranking_agent.py`, `tests/services/formulas/test_registered_formulas.py`, `tests/services/supplier_performance/test_performance_formulas.py`

### `supplier.criterion_normalisation`

*Min-max normalise one raw criterion to 0-100 across the suppliers being compared* — set formula, owner **sourcing**, effective from 2026-09-05, source `src.services.formulas.definitions.ranking:119`.

All-unmeasured yields NaN for everyone, so `supplier.weight_renormalisation` drops the criterion rather than dragging every composite down by its weight. When every MEASURED supplier ties, only the measured ones score 100 -- assigning the tied score to the whole column once handed an unmeasured supplier a perfect 100 they never earned.

| Input | Unit | Range | Required |
|---|---|---|---|
| `value` | ratio | unbounded | no |
| `direction` | label | one of 'lower_is_better', 'higher_is_better' | yes |

**Output:** `list[float]` in score_0_100 — 0-100 per supplier; NaN where unmeasured

**Dependents:** `tests/api/test_update_agent.py`, `tests/test_calculation_tools.py`

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

**Dependents:** `src/services/formulas/__init__.py`, `tests/api/test_update_agent.py`, `tests/services/formulas/test_registered_formulas.py`, `tests/test_calculation_tools.py`

### `supplier.weight_renormalisation`

*Drop criteria nobody can be scored on, and renormalise the rest to sum to 1* — scalar formula, owner **sourcing**, effective from 2026-09-05, source `src.services.formulas.definitions.ranking:161`.

A criterion only counts if a real `_score` column exists AND is not entirely NaN. Accepting one on the strength of its raw column let weights and scoring disagree: the weight map kept the criterion, the scoring loop skipped it for want of a score column, and its weight silently vanished from the composite.

| Input | Unit | Range | Required |
|---|---|---|---|
| `weights` | record | unbounded | yes |
| `scored` | record | unbounded | yes |

**Output:** `dict` in ratio — criterion -> normalised weight, summing to 1.0

**Dependents:** *none*

### `sustainability.scope3`

*Scope 3 emissions from activity data and emission factors the user supplies, per line and in total* — scalar formula, owner **analytics**, effective from 2026-10-22, source `src.services.formulas.definitions.quality:52`.

Factors are never supplied here: a line without a factor is counted in lines_without_factor and left out of the total, which the caller reports as a lower bound. factor_source is carried through so the answer can cite it.

| Input | Unit | Range | Required |
|---|---|---|---|
| `activities` | rows | unbounded | yes |

**Output:** `dict` in count — {total_kgco2e, total_tco2e, lines: [...], lines_without_factor}; None when no line carries both a quantity and a factor

**Dependents:** *none*

### `tco.total_cost`

*The present value, at the evaluation date, of every cash outflow to acquire, use and dispose of the goods over the horizon, minus inflows* — scalar formula, owner **cost**, effective from 2026-10-20, source `src.services.formulas.definitions.cost:116`.

PV(amount, days) = amount / (1 + r) ** (days / 365), days from the evaluation date. Unstated timing is day 0. The goods are paid on net terms unless AP takes the early-payment discount, in which case the cheaper present value is used and recorded. Lifetime items are included only when supplied; every unsupplied ownership item is listed in excluded_lifetime. Inventory carrying cost excludes capital cost, because r already covers it. Contingent charges and supply-disruption risk are not included.

| Input | Unit | Range | Required |
|---|---|---|---|
| `landed` | record | unbounded | yes |
| `cost_of_capital` | ratio | [0, 1] | yes |
| `horizon_years` | money | [0, +inf] | yes |
| `payment_terms` | record | unbounded | no |
| `ap_can_take_discount` | boolean | unbounded | no |
| `lifetime_items` | rows | unbounded | no |

**Output:** `dict` in money — tco, tco_status (inherits landed_status), discount_choice, components with pv, excluded_lifetime, warnings

**Dependents:** `src/services/cost/compare_tco.py`, `tests/orchestration/test_argument_check.py`, `tests/services/cost/test_cost_formulas.py`, `tests/services/test_cost_compare.py`, `tests/skills/test_checks.py`


