# ADR 0001 — Deferring the Track B quantitative services for negotiation

- **Status:** Accepted
- **Date:** 2026-09-05
- **Owner:** negotiation
- **Evidence:** `docs/negotiation-agent-state-audit.md` §1.2, §1.7

## Context

The negotiation agent is specified to reason with six quantitative services —
collectively "Track B". An audit on 2026-09-05 established that **none of the six
exists in BP_Backend**, and that the agent has been computing substitutes for them
without saying so. This ADR records what is absent, what stands in its place, what
that costs us in confidence, and what should trigger building each one.

We are **not** building Track B in this work. The point of writing it down is that
the substitutes are currently indistinguishable from the real thing at the point of
use, and that is the defect worth fixing first.

## Decision

Defer all six services. Make each substitute **self-declaring** — a consumer must be
able to tell a computed number from a stood-in one without reading the source — and
cap the confidence of any negotiation recommendation at what its weakest input earns.

## The six services

### 1. Lever valuation (NPV with country-aware discount rates) — ABSENT

**Stands in today:** a single flat constant, `COST_OF_CAPITAL_APR = 0.12`
(`src/agents/negotiation_agent.py:72`), divided by 365 and multiplied by the days
of a payment term (`_optimize_multi_issue`, :6559). Four fixed term packages
(Net15/30/45/60) are priced this way.

**Classification:** HEURISTIC. It gets the *sign* and rough magnitude of a
payment-term trade right, which is most of what a term comparison needs. It is not
an NPV: no cash-flow schedule, no per-country rate, no currency dimension, and 12%
is an unsourced global default.

**Trigger to build:** the first negotiation whose levers span more than one country
of supply, **or** any deal where a payment-term concession is traded against a price
concession and the two must be compared in the same units. Until then the ranking of
four term options is not sensitive enough to the rate to justify the build.

### 2. MILP package optimiser — ABSENT

**Stands in today:** exhaustive enumeration of a bounded grid in
`_optimize_multi_issue` (:6570-6584): at most 5 candidate prices × 4 term packages
× 1 lead package × 4 volume tiers = **≤80 options**, scored by a normalised
weighted sum and argmaxed.

**Classification:** HEURISTIC, and defensible at this size. At 80 options the
optimum of the stated objective is found exactly; a solver would return the same
answer more slowly. `scipy.optimize.milp` is already a dependency
(`src/services/resolution/solver.py`), so the build cost is low when it is needed.

**Trigger to build:** when the option space stops being enumerable — the first of
(a) more than ~3 lever dimensions, (b) cross-lever constraints that cannot be
expressed as a post-hoc clamp (e.g. "volume tier 500 is only available with Net30"),
or (c) multi-supplier award splitting. Any of those breaks the grid; none of them
exist today.

### 3. Bayesian benchmark posterior — ABSENT

**Stands in today:** a point estimate, `benchmarks["p10"]` or `benchmarks["low"]`,
used as a candidate supplier cost floor (`_estimate_zopa`, :6452-6455). No
uncertainty is carried, so a p10 drawn from 3 observations and one drawn from 300
are used identically.

**Classification:** HEURISTIC when a benchmark is supplied; the benchmark is
usually not supplied.

**Trigger to build:** when benchmark data acquires a sample-size dimension. Today
benchmarks arrive as caller-supplied scalars with no `n`, so there is nothing to
form a posterior over. Building this before the data carries `n` would produce a
credible interval computed from an assumed prior — a second fiction on top of the
first.

### 4. Country risk / total equity risk premium (Damodaran) — ABSENT

**Stands in today:** nothing. There is no country dimension in the negotiation
domain at all.

**Classification:** MISSING, with no substitute — which is the honest state. No
number in the negotiation path claims to be country-risk-adjusted.

**Trigger to build:** the first cross-border negotiation where a discount rate or a
supplier-viability judgement must differ by country of supply. Note this depends on
(1): a country-aware rate has nowhere to be consumed until an NPV exists.

### 5. Tax / duty and withholding-tax calculator — ABSENT

**Stands in today:** nothing. Negotiated prices are compared gross, in the document's
own currency, with no duty or withholding leg.

**Classification:** MISSING, with no substitute.

**Risk while deferred:** this is the deferral most likely to produce a *wrong*
answer rather than an absent one. Two offers from suppliers in different duty
regimes are today compared as though landed cost equalled quoted price. Nothing in
the code claims otherwise, but nothing warns the buyer either.

**Trigger to build:** the first negotiation comparing suppliers across a customs
border or a withholding-tax jurisdiction. This should be treated as a **blocking**
gap for that scenario, not a nice-to-have.

### 6. Should-cost model — ABSENT

**Stands in today, until 2026-09-05:** `price * 0.85` — a supplier cost floor
invented from the supplier's own offer whenever should-cost, benchmark and history
were all missing, which is the ordinary case because **nothing in this repository
produces a should-cost**. `should_cost` is only ever read from
`context.input_data` (`negotiation_agent.py:5166`) and no producer exists.

That fallback has been **removed** in this work. `_estimate_zopa` now returns
`supplier_floor = None` with `supplier_floor_basis = None` and an explicit finding.

**Consequence of the removal, recorded deliberately:** the counter-email
justification gated its "based on our market analysis and benchmarking across
similar products" claim on a >15% gap to this floor. Since
`(offer − 0.85·offer) / (0.85·offer) = 0.1765`, the gate was **always open** — that
sentence was sent to every supplier on every negotiation regardless of whether any
benchmarking had happened. Removing the fabricated floor removes an unfounded claim
from outbound correspondence.

**Trigger to build:** the first category where we hold bill-of-materials or
cost-driver data. Until then `supplier_floor` stays honestly absent, and a counter
must not be presented to a supplier as cost-justified.

## Consequences

### Honest confidence ceiling on negotiation recommendations

With Track B absent, a negotiation recommendation rests on:

- a counter price derived from a **regex first-match over supplier prose**
  (`_extract_price_from_response`, :4787) — `Confidence.ASSERTED` at best;
- a supplier cost floor that is now **absent** rather than invented;
- a Kraljic position from thresholds **frozen from a 2026 corpus snapshot**
  (`classification.py:26-27`);
- **no leverage input to the price ladder at all** — `NegotiationContext.leverage`
  is set and never read (audit §1.1).

**Ceiling: `Confidence.ASSERTED`.** No negotiation recommendation may be presented
as OBSERVED until (a) counter-party moves arrive structurally rather than by regex,
and (b) at least one of should-cost or a sized benchmark exists. The *decision*
(counter / accept / decline / clarify) and the *play ranking* are evidence-gated and
defensible; the *numbers* are not, and must not be shown as computed positions.

### What must not happen while this ADR stands

1. No golden vector may pin the current output of `_estimate_zopa`,
   `_adaptive_strategy` or `_optimize_multi_issue` as validated behaviour. They are
   registered `PROVISIONAL` (see `docs/formula-inventory.md`).
2. No user-facing surface may present a negotiation number without its confidence
   marker.
3. No substitute may be made to *look* like the service it stands in for. A flat
   12% is allowed; a flat 12% called an NPV is not.

## Revisiting

Revisit when any single trigger above fires, or unconditionally at the first
cross-border negotiation, whichever is sooner.
