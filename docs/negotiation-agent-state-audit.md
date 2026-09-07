# Negotiation Agent — State Audit

**Date:** 2026-09-05 · **Branch:** Development · **Baseline commit:** `2e27c58`
**Test baseline:** `tests/test_negotiation_agent.py`, `tests/test_negotiation_strategy_engine.py`,
`tests/test_negotiation_skills.py`, `tests/services/negotiation_advice/` — **214 passed** in 165s.

**Re-verified 2026-09-07** at `fbc4b56`, by reading the whole agent and *running* the load-bearing
claims rather than re-reading them. Five findings below were wrong or had gone stale; each is
corrected in place under a **`↻ 2026-09-07`** marker, and §6 lists them together. New test
baseline, same suites plus `tests/services/formulas/` and `tests/services/negotiation/`:
**250 passed** in 188s.

---

## 0. Finding that reframes the whole audit

> The brief states: *"The negotiation agent was refactored to a single deterministic state
> machine with four narrow LLM-callable functions: `generate_plan`, `select_tactic`,
> `evaluate_counter`, `reconcile`."*

**That refactor does not exist in this repository.** No function, method, or symbol by any of
those four names exists anywhere under `src/` or `tests/`:

```
$ grep -rnE 'def (generate_plan|select_tactic|evaluate_counter|reconcile)\b' --include='*.py' src/ tests/
(no output)
```

The only `reconcile` hits in the codebase are in unrelated services
(`src/services/reconciliation.py` — document reconciliation; `scripts/reconcile_uom_vocabulary.py`).

What actually exists is the pre-refactor shape:

| Expected | Actual |
|---|---|
| single deterministic state machine | `src/agents/negotiation_agent.py` — **13,292 lines**, one `NegotiationAgent` god class plus 8 module-level helper classes |
| four narrow LLM-callable functions | zero. One LLM call exists, in `_extract_negotiation_signals` (:6372), and it is a free-text JSON prompt with no grammar and no schema validation |
| `negotiation_strategy_engine.py` as a "remnant" | still present (405 lines), still imported at process start, still under test (`tests/test_negotiation_strategy_engine.py`) |

> **↻ 2026-09-07 — this finding stands.** The grep is still empty at `fbc4b56`. The class has
> *grown* to **13,342 lines**. The strategy-engine row is the one thing that changed: it was
> salvaged and deleted in `343b5a0` (§4).

Consequently **§1.1 is answered as MISSING throughout**, and the rest of the audit reports what
the code does *instead*, mapped onto the four intended responsibilities. The remediation plan in
Phase 2 assumed the refactor had landed; it has not, so Phase 2 as written cannot be executed
literally. See §8 for what I recommend instead.

---

## 1.1 The four functions — **MISSING**

None of the four exist. Their responsibilities are discharged by the following, which I classify
against the intended design.

### `generate_plan` → **HEURISTIC** (`plan_counter`)

`src/agents/negotiation_agent.py:304-465`, reached via `compute_decision` (:1467) →
`decide_strategy` (:1513).

Signature (actual):

```python
def plan_counter(ctx: NegotiationContext, signals: SupplierSignals) -> Dict[str, object]
```

Returns an **unvalidated `dict`**, not a schema. Keys, by inspection of the five return
statements: `decision` (`hold|clarify|accept|decline|counter`), `counter_price`, `asks`,
`lead_time_request`, `message`, `log`, `finality`. There is no `TypedDict`, no pydantic model,
no JSON Schema — nothing enforces this contract, and the five branches do not all return
the same value types (`counter_price` is `None` on three paths).

**LLM boundary: none.** `plan_counter` is pure Python. This is the one place the intended design
and the code agree by accident — the price ladder is deterministic.

**Deterministic computation and its inputs.** A hardcoded round ladder:

- `ctx.round_index > ctx.max_rounds` → hold at `min(current_offer, walkaway_price or current_offer)`
- non-positive offer or target → `clarify`
- final-offer language detected in `signals.message_text` → `accept` if `offer <= walkaway or target`, else `decline`
- `gap <= 0` → `accept`
- Round 1: gap > 10% → `max(target, offer × 0.88)`; else midpoint
- Round 2: gap ≤ 10% → midpoint; else `offer − gap × 0.6`
- Round 3+: `target + max(min_abs_buffer, target × risk_buffer_pct)`, else `max(target, offer × (1 − step_pct_of_gap))`
- finally `counter_price = round(max(counter_price, target_price), 2)`

**Three context fields are set and never read.** `compute_decision` (:1489-1501) constructs
`NegotiationContext` with `aggressiveness=0.75, leverage=0.6, urgency=0.3`. Verified by
extracting every `ctx.*` reference inside `plan_counter`:

```
ctx.ask_early_pay_disc  ctx.ask_lead_time_keep  ctx.currency  ctx.current_offer
ctx.max_rounds  ctx.min_abs_buffer  ctx.risk_buffer_pct  ctx.round_index
ctx.step_pct_of_gap  ctx.target_price  ctx.walkaway_price
```

`aggressiveness`, `leverage`, `urgency` and `min_abs_step` do not appear.
**There is no leverage input to counter pricing at all**, despite a field named `leverage`.
Likewise `SupplierSignals.offer_prev` and `offer_new` are populated (:1503-1507) and never
read — only `message_text` is. The supplier's *movement between rounds* is computed and discarded.

**Missing-input behaviour: SILENT DEFAULT — defect.**
- `decide_strategy` (:1524) returns `strategy="clarify"` when price or target is `None`. This is
  the *correct* shape but it is indistinguishable from a genuine "ask them to clarify" decision;
  there is no UNASSESSED marker and no finding is emitted.
- `compute_decision` (:1484-1486) does `float(payload["current_offer"])` — a **`KeyError`** if the
  key is absent rather than present-and-`None`. Two different failure modes for the same defect.
- `ctx.max_rounds` silently defaults to `3` (:1497), `ask_early_pay_disc` to `0.02` (:1471).

> **↻ 2026-09-07 — CORRECTION 1 of 5. `generate_plan`'s analogue now fails closed, and the price
> is single-sourced.** `decide_strategy` no longer calls `compute_decision` directly: it calls
> `evaluate("negotiation.counter_plan", …)` through the formula registry (:1535-1547), which
> validates every input against the declared contract *before* the maths runs. Three consequences,
> all verified by running them:
>
> - **A refused contract yields `clarify`, marked.** `current_offer = -5` returns
>   `strategy="clarify"` with `decision_origin="contract_refused"` and the reason
>   *"current_offer=-5 money is below the declared minimum 0 (range [0, +inf])"*. The two failure
>   modes above are now one, and it is named. The bare `KeyError` is unreachable from this path —
>   `decide_strategy` null-checks first and the contract rejects second.
> - **A successful plan is marked too**, with `decision_origin="plan_counter"` and
>   `price_plan_locked=True`.
> - **`price_plan_locked` makes the counter price authoritative.** `_adaptive_strategy` skips its
>   price branch when locked (:6528), and `_run_single_negotiation_locked` strips `counter_price`
>   out of `_optimize_multi_issue`'s overrides (:5545-5546). See the correction under
>   `select_tactic`.
>
> **What has NOT changed:** the never-read fields above. `aggressiveness`, `leverage`, `urgency`,
> `min_abs_step`, `offer_prev` and `offer_new` are still set and still unread. The registry pins
> that in its own `notes`: *"not governed and not configurable at the call site (gap report F22)"*.
> There is still no leverage input to counter pricing.

### `select_tactic` → **HEURISTIC, split across three places, none authoritative**

Three separate things select a tactic and they do not agree:

1. `NegotiationStrategyEngine.select_strategy` — `src/engines/negotiation_strategy_engine.py:186-215`.
   Six named strategies chosen by a first-match rule ladder on `supplier_history_count` and
   `alternative_quotes`. **DEAD — see §1.5.**
2. `NegotiationAgent._adaptive_strategy` — `:6450-6503`. Overwrites `decision["strategy"]` to
   `"package-trade"` on a finality hint; otherwise adjusts `counter_price` toward a ZOPA midpoint.
3. `negotiation_advice.classification.classify` — `src/services/negotiation_advice/classification.py:38-90`.
   Kraljic quadrant → negotiation style. This is the only one that reaches a user (via
   `/deals/{id}/advice` and the Negotiate dashboard's `plays`), and it is the **best-built code in
   the domain**: it fails closed (`indeterminate: True` with named reasons when spend or
   alternatives are missing, :47-56), carries a confidence, and documents that
   `bp_supplier.supplier_type` is *not* a Kraljic axis.

> **↻ 2026-09-07 — CORRECTION 2 of 5. They no longer disagree about the price, and (1) is gone.**
> Item 1 was salvaged and deleted in `343b5a0`. Item 2's price branch is dead whenever the plan is
> locked, which is every path where a counter price exists at all. **The counter price is now
> single-sourced from `negotiation.counter_plan`.** The audit's sentence "adjusts `counter_price`
> toward a ZOPA midpoint" describes a branch that no longer executes in production.
>
> **But the `strategy` is not locked, only the price — and that is a live bug.** On any finality
> hint `_adaptive_strategy` still overwrites `decision["strategy"]` with `"package-trade"` and
> returns early (:6515-6523), *including when the plan said `accept`*. The multi-round loop closes
> a supplier only on `accept` or `decline` (`_execute_negotiation_round`:3794). Verified
> end-to-end: a supplier offering 79.00 against an 80.00 target and writing *"this is our final
> offer"* produces `plan_counter → accept`, then `_adaptive_strategy → package-trade`, then
> `strategy_lower in {"accept","decline"}` is **False**, so the supplier is never marked ACCEPTED
> and the negotiation runs another round against an offer we had already decided to take.
> **A best-and-final we want to accept cannot close the negotiation.** Not in the original audit.

### `evaluate_counter` → **HEURISTIC** (`_estimate_zopa` + `_optimize_multi_issue`)

`:6399-6448` and `:6505-6651`. No LLM. `_optimize_multi_issue` scores an exhaustively enumerated
grid (≤5 prices × 4 term packages × 1 lead package × 4 volume tiers = ≤80 options) with a
normalised weighted sum, default weights `price .5 / delivery .2 / risk .2 / terms .1`.

**Missing-input behaviour: silent.** `_estimate_zopa` falls back to `price * 0.85` for
`supplier_floor` when should-cost, benchmark p10 and historic minimum are all absent (:6431) —
i.e. **it invents a 15%-below-offer supplier cost floor out of nothing**, and the caller cannot
tell that from a floor derived from a real should-cost model. This is the single most
consequential placeholder in the domain.

> **↻ 2026-09-07 — fixed in `e4eda2e`, verified live.** With no should-cost, no benchmark and no
> history, `_estimate_zopa` now returns `supplier_floor: None`, `supplier_floor_basis: None`, and
> a finding naming the absence: *"zopa.supplier_floor UNASSESSED: … The supplier's cost base is
> unknown; do not present a counter as cost-justified."* The fabrication is gone.
>
> **The marker reaches nobody, though** — see the correction in §1.4. `zopa["findings"]` is
> written at :6497 and read at no call site: `zopa` never enters the agent's output `data`, and
> its only rendering point (`_build_prompt_context`'s `zopa_summary`) is in dead code.

### `reconcile` → **MISSING, with no analogue**

`_consolidate_multi_round_results` (:4823) assembles round outcomes for reporting. There is no
function that reconciles a negotiated position back against source documents, policy, or the
extraction fact model.

---

## 1.2 Track B services

| Service | Status | Evidence |
|---|---|---|
| Lever valuation, NPV with country-aware discount rates | **ABSENT** | zero hits for `npv\|net_present_value` in `src/`. The only discounting is `COST_OF_CAPITAL_APR = 0.12` (`negotiation_agent.py:72`), a flat env constant divided by 365 (:6559) to price payment terms. Not country-aware; not an NPV. |
| MILP package optimiser | **ABSENT** | `scipy.optimize.milp` exists in the repo but at `src/services/resolution/solver.py:20,166` — entity resolution, unrelated. Negotiation uses brute-force grid enumeration (`_optimize_multi_issue`:6570-6584). |
| Bayesian benchmark posterior | **ABSENT** | zero hits for `bayes\|posterior`. |
| Country risk / total equity risk premium (Damodaran) | **ABSENT** | zero hits for `damodaran\|equity_risk_premium\|country_risk`. |
| Tax/duty and withholding-tax calculator | **ABSENT** | zero hits for `withholding_tax\|duty_rate\|customs_duty`. The `withholding` hit at `negotiation_agent.py:5088` is log text about withholding *price data*, not tax. |
| Should-cost model | **ABSENT as a service; consumed as an optional caller input** | `should_cost` is only ever read from `context.input_data.get("should_cost")` (:5166) and used at :6423. Nothing in the repo produces it. In practice it is always `None`. |

**What is computed instead, classified honestly:**

| Intended | Actual substitute | Class |
|---|---|---|
| supplier cost floor from should-cost | `min(should_cost, benchmark_p10, history_min)` — all three normally absent → `offer × 0.85` | **PLACEHOLDER** |
| NPV of payment terms | `apr_equiv = (0.12/365) × days` on four fixed term packages | **HEURISTIC** |
| MILP over the lever package | argmax over an ≤80-row enumerated grid, weighted sum | **HEURISTIC** (defensible at this option count) |
| benchmark posterior | `benchmarks["p10"]` or `benchmarks["low"]`, point estimate, no uncertainty | **HEURISTIC** |
| country risk premium | not represented anywhere | **MISSING** |
| lead-time value | `LEAD_TIME_VALUE_PCT_PER_WEEK = 0.01` — 1% of price per week saved, flat | **PLACEHOLDER** |

---

## 1.3 CISE wiring — **CISE DOES NOT EXIST**

There is **no criticality assessment service** in BP_Backend. No `CISE`, no
`ASSESSED_CRITICAL` / `ASSESSED_NON_CRITICAL` / `UNASSESSED` criticality vocabulary. The single
`unassessed` symbol in the codebase is `src/services/price_outlier/detector.py:213`, an unrelated
price-outlier verdict.

- **Does `select_tactic` consume CISE?** No — neither the function nor the service exists.
- **What drives Kraljic positioning today?** `negotiation_advice.classification.classify` (:38) **infers** it from two
  measured signals: `deal_value` and `alternative_supplier_count`, against two thresholds
  compiled into source (`classification.py:26-27`):
  `high_spend = 98_175.0` (live deal-value p90) and `many_alternatives = 93` (per-deal median
  alternative-supplier count). These are *testdata-derived distribution parameters hardcoded as
  constants* — already logged as gap-report items D-10 (silent staleness) and D-11 (`93`
  duplicated as `signals.THIN_MARKET_ALTERNATIVES`).
- **What happens on unknown criticality?** `classify` **fails closed correctly** (:47-56): if `deal_value`
  or `alternative_supplier_count` is `None` it returns
  `{"quadrant": None, "quadrant_confidence": 0.0, "indeterminate": True}` with named reasons.
  This is the one place in the negotiation domain that already behaves the way the brief asks for.
  Note it is *not* wired to the counter ladder — `plan_counter` never sees the quadrant.

> **↻ 2026-09-07 — CORRECTION 4 of 5. The thresholds are no longer "compiled into source".**
> `advisor.load_thresholds` reads them from `proc.bp_policy` via `PolicyEngine`, slug
> `negotiation_advice_thresholds`; `classification.default_thresholds()` is now only the fallback
> when that lookup fails. Confirmed present in the live DB (`.env` → **bp_testdb**):
> `policy_id 605`, `policy_type 'negotiation'`, `policy_details {"high_spend": 98175.0,
> "many_alternatives": 93}`.
>
> The values are identical, so nothing about the *classifications* changes — but they are governed
> data now, editable without a deploy. **D-10 is narrowed, not closed**: drift is still measured by
> nothing, and D-11 (`93` duplicated as `signals.THIN_MARKET_ALTERNATIVES`) is untouched.
>
> The last sentence stands and is the point: `plan_counter` still never sees the quadrant. This is
> Phase 2 step 4, still deferred into the refactor.

---

## 1.4 Input provenance and confidence

### Path from correspondence to scored move

```
supplier email
  └─ imap_supplier_response_watcher / email_watcher        (transport)
     └─ ResponseMatcher.match_response          negotiation_agent.py:662
        └─ _map_responses_to_suppliers                             :4650
           └─ _extract_price_from_response                         :4787   ← EXTRACTION STEP 1
              ├─ dict keys: price|quoted_price|unit_price|offer_price|current_offer
              └─ else REGEX over free text:
                    r"[£$€₹]\s*(\d+(?:,\d{3})*(?:\.\d{2})?)"
                    r"(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:GBP|USD|EUR|INR)"
           └─ _extract_negotiation_signals                         :6332   ← EXTRACTION STEP 2 (LLM)
        └─ compute_decision → plan_counter                         :1467
```

**Two extraction steps, both unmarked:**

1. **`_extract_price_from_response` (:4787) — regex, first-match-wins.** It takes the *first*
   currency-looking number in the message body. A supplier writing *"our previous quote of
   £12,400 stands, but we can do £11,900 on 500 units"* is scored at **£12,400**. There is no
   confidence, no candidate list, no marker that the number came from prose rather than a
   structured field. This number then becomes `current_offer` and drives the entire price ladder.

2. **`_extract_negotiation_signals` (:6372-6394) — ungrammared LLM call.** A free-text prompt
   asking for JSON, parsed by `content[content.find("{"):content.rfind("}")+1]` and
   `json.loads`. No grammar constraint, no schema validation, no field-type checking — parsed
   values are copied straight over the regex-derived defaults (`for key in signals: if key in
   parsed ... signals[key] = parsed[key]`). A model returning `"moq": "about 500"` writes a
   string into an int field. The whole block is wrapped in `except Exception: logger.debug(...)`,
   so a total failure is invisible and the caller sees the keyword-only defaults with no
   indication the LLM leg was skipped. This is the same class of defect recorded for the
   extraction domain in [`project_extraction_unit_price_as_total_bug`].

### Tri-state confidence — **ABSENT**

There is **no tri-state confidence vocabulary anywhere in the negotiation domain**. Searching
`ASSERTED|CONFIRMED|tri-state` across `src/services/facts/`, `negotiation_agent.py` and
`src/services/negotiation_advice/` returns nothing relevant.

What confidence exists is a single float, `quadrant_confidence`, produced by
`classification._confidence` (:29-36) as `min(1.0, 0.5 + |value/bar − 1.0|)`. **This is a
distance-from-threshold measure, not a probability and not a provenance marker.** It says how far
the deal is from a classification boundary. It says nothing about whether the inputs were read
reliably. Two deals with identical spend get identical confidence whether the spend came from a
signed PO or a regex over an email.

- **Is `evaluate_counter`'s result capped at the lowest input confidence?** No — there is no
  input confidence to cap against.
- **Is there a path where a score from an unverified read of a supplier message reaches a
  user-facing recommendation without an ASSERTED marker?** **Yes, and it is the primary path.**
  `_extract_price_from_response` (regex over prose) → `compute_decision` → `counter_price` →
  `_finalize_round_email_bundle` (:2476) → drafted supplier email, and → `_record_round_status`
  (:3907) → the round record the Action Centre reply panel renders
  (`src/api/routers/workflows.py:1396`). Nothing on that path carries provenance.

> **↻ 2026-09-07 — the finding stands, and is now precisely diagnosable.** The vocabulary now
> *exists* and is *produced*: `Confidence` (`src/services/formulas/unassessed.py`) with
> OBSERVED/ASSERTED/UNVERIFIED, and every registry evaluation attaches one to its
> `EvaluationRecord`. What is missing is no longer the vocabulary — it is the **wire**. Three
> concrete breaks, all verified:
>
> 1. **`zopa["findings"]` is written and never read.** Produced at :6497; `zopa` is passed to
>    message composers but never into the agent's output `data`. Its only rendering point is
>    `_build_prompt_context`'s `zopa_summary` key (:9026) — and that method has **zero callers**
>    (§1.5). So the UNASSESSED cost-floor marker that `e4eda2e` added reaches no consumer.
> 2. **The evaluation records die with the process.** See §1.7 — the default audit sink is an
>    in-process ring buffer and nothing installs the durable one.
> 3. **The regex-over-prose path is unchanged.** `_extract_price_from_response` still takes the
>    first currency-looking number with no confidence and no candidate list.
>
> Also verified live on 2026-09-07: the LLM leg of `_extract_negotiation_signals` **failed while
> being observed** — Ollama returned `500` three times, `except Exception: logger.debug(…)`
> swallowed it, and the agent proceeded on keyword-only defaults with nothing in the output to say
> the model had never answered. The defect described two paragraphs above is not hypothetical.

### Additional placeholder reaching the UI directly

`src/services/negotiate_dashboard.py:259-282` computes the Negotiate page's `currentStandpoint`
and `preferredOutcome`:

```python
supplier_rate = 50
our_aim = 47
walk_away = 50
if quote and actual and quote > 0:
    supplier_rate = round(actual / quote * 50)
    our_aim = max(0, supplier_rate - 3)
    walk_away = supplier_rate
```

`ourAim` — labelled **"optimalPrice"** in the payload (:281) — is *"three index points below
wherever the supplier is"*. `walkAway` is set **equal to the supplier's own rate**, which as a
walk-away threshold is meaningless. When `quote` or `actual` is missing, the literals 50/47/50
ship unchanged. This is a **PLACEHOLDER presented to buyers as a recommended negotiating
position**, and it is not registered, not tested, and carries no marker.

> **↻ 2026-09-07 — fixed in `e4eda2e`, verified.** `ourAim` and `walkAway` are now `None`, and the
> payload carries `unavailableReason`: *"An optimal price needs a should-cost or benchmark, and a
> walk-away needs an authority limit. Neither is available for this deal."* `supplierRate` remains
> — but it is a measured index (`actual / quote × 50`), not an invented position, and it is `None`
> when either figure is missing. The page now says it does not know.

---

## 1.5 Dead code

### `negotiation_strategy_engine.py` — **importable, constructed at startup, never invoked**

- Importable: yes. `src/orchestration/reasoning_engine.py:20-21`.
- Constructed: yes, at process start. `reasoning_engine.py:115` (`self._negotiation_engine =
  NegotiationStrategyEngine()`), and `ReasoningEngine` itself is instantiated in
  `src/api/main.py:150` on every boot.
- **Reachable? No.** Its only call site is `reasoning_engine.py:381`, inside
  `_rule_based_plan`. `_rule_based_plan` is called only from `create_plan` (:269).
  **`create_plan` has zero callers**:

```
$ grep -rn '\.create_plan(\|rule_based_plan(' --include='*.py' . | grep -v .claude/ | grep -v /venv/
src/orchestration/reasoning_engine.py:269:        return self._rule_based_plan(task, context or {})
src/orchestration/reasoning_engine.py:342:    def _rule_based_plan(self, task: dict, context: dict) -> WorkflowPlan:
```

This confirms the standing note in [`project_dynamic_planner_diagnosis`] that `ReasoningEngine`
is unwired. `NegotiationStrategyEngine` is **DEAD** — 405 lines plus a 400-line test file
(`tests/test_negotiation_strategy_engine.py`, currently green) exercising code no request path
can reach. It is also the *only* place the six named strategies and the BATNA text exist.

### BATNA-proxy fields — **NOT salvaged**

`alternative_quotes` and `supplier_history_count` exist in exactly two places, both dead:

- `src/engines/negotiation_strategy_engine.py:28-29` (the dataclass), read at :195, :199, :203,
  :330, :337, :383, :390, :396
- `src/orchestration/reasoning_engine.py:377-378`, populating that dataclass on the dead path

**Neither field appears anywhere in `negotiation_agent.py`.** They were never salvaged, because
the refactor that was supposed to salvage them never happened. Deleting the strategy engine today
would delete the only BATNA reasoning in the codebase.

Note the *adjacent* concept survives elsewhere under a different name:
`classification.classify` uses `alternative_supplier_count` (from
`negotiation_advice/signals.py`) as its market-contest axis. That is a genuine measured signal
from the deal's items — arguably a better BATNA proxy than `alternative_quotes` ever was. It is
not, however, the same field, and nothing reads `supplier_history_count`'s relationship signal.

### Other overlapping computation

| Concern | Strategy engine | The agent | Advice service |
|---|---|---|---|
| pick a posture | `select_strategy` (6 strategies) | `_adaptive_strategy` (`package-trade` override) | `classify` → 5 styles |
| set a target price | `generate_position`: `order_value × (1 − discount)` | `plan_counter` round ladder | — |
| continue / escalate / walk away | `should_continue` → `ContinueDecision` | `_detect_outliers` → `requires_review` / `human_override` | play `state` |

Three independent, mutually inconsistent vocabularies for the same three decisions.

### ↻ 2026-09-07 — dead code this audit missed, all inside `negotiation_agent.py`

The original pass looked for dead *modules* and found the strategy engine. Reading the whole file
turns up roughly **900 further lines that no request path can reach**, three of which matter
because the audit reasoned about them as if they ran:

| What | Evidence | Why it matters |
|---|---|---|
| `_get_prompt_template` (:8859), `_build_prompt_context` (:8896), `_apply_prompt_template` (:9061) | zero callers; grep returns only the three `def` lines | This is the **only** place `zopa_summary` — and therefore the UNASSESSED cost-floor finding — was ever rendered. Its unreachability is why §1.4's break (1) exists. |
| `_compose_negotiation_message` (:8686), the "simple" composer | zero callers; superseded by `_compose_negotiation_message_rich` | ~150 lines plus five `_craft_*_simple` helpers. |
| `_compose_negotiation_message_rich` (:7689) and its ~15 `_craft_*` / `_weave_*` helpers | reachable, but only when `NEG_USE_ENHANCED_MESSAGES` is `true`; it **defaults to `"false"` (:1616) and is not set in `.env`** | ~700 lines that do not run in production. **`_craft_position_statement` (:7876) lives here** — the "based on our market analysis and benchmarking" claim §4 discusses was behind a disabled flag as well as gated on the fabricated floor. |

The live message is therefore `_build_summary_fallback` (:8572): a bulleted round plan, not prose.
Any future work that assumes the agent writes persuasive negotiation copy should check this flag
first.

---

## 1.6 Governance and state

### Persistence — **not a bitemporal spine, and there is no GPSS**

Negotiation state persists two ways, neither temporal:

- **Redis**, key `negotiation_session:{workflow_id}` — `src/services/negotiation_session.py:31`.
  Last-write-wins, no history, no valid-time.
- **Postgres**, `proc.negotiation_sessions` and `proc.negotiation_session_state` (the only two
  `proc.*` tables `negotiation_agent.py` writes besides `proc.bp_approval`). Created by
  commit `3019b5c` *"create the two tables the pipeline reads but never made"*.

There is no bitemporal audit spine for negotiation. `valid_from` exists only in
`src/services/facts/models.py:255,375` (the extraction fact model) and negotiation does not
write there.

**On GPSS specifically** — the repo is explicit that it does not exist:

> `src/services/formulas/registry.py:273-274` — *"`gpss_version` is accepted because the
> specification names it. It is recorded and reported, and it resolves against nothing: there is
> no GPSS…"*
> `src/services/facts/concept_codes.py:5` — *"There is no GPSS dictionary in this project…"*

So "persisting as a GPSS entity type" is not achievable as stated; the brief's §2.5 requirement
to give formulas "typed contracts (GPSS terms…)" resolves to the local `contract.py` vocabulary
instead, which is what the existing registered formulas already use.

### Authority limits and walk-away — **enforced only partially, and in the wrong direction**

- **Walk-away as a floor on our own counter: NOT enforced.** `plan_counter` ends with
  `counter_price = round(max(counter_price, target_price), 2)` (:452) — it clamps to the
  *target*, never to `walkaway_price`. `walkaway_price` is consulted on only two branches: the
  max-rounds hold (:312) and the finality accept/decline test (:349). A round-2 counter above
  the walk-away is emitted without complaint.

> **↻ 2026-09-07 — CORRECTION 3 of 5. This finding is WRONG, and it was the most consequential
> error in the audit.** It is true of `plan_counter` read in isolation, which is what was
> inspected. It is not true of the agent. `_run_single_negotiation_locked` passes every counter
> through `_respect_positions` (:10607) immediately after `decide_strategy` returns (:5247, and
> again at :5288 on the review path), and that helper clamps in all three directions:
>
> ```python
> candidate = max(candidate, positions.desired)    # floor at the target
> candidate = min(candidate, positions.no_deal)    # CEILING AT THE WALK-AWAY
> candidate = min(candidate, positions.start)      # never above our own last position
> ```
>
> `positions.no_deal` **is** `walkaway_price` (`_build_positions`:10571). Verified by running it:
> a counter of `88.00` against a walk-away of `85.00` returns **`85.00`**; with no walk-away
> supplied it returns `88.00` unchanged. The guardrail exists, is reached on the live path, and
> works. A round-2 counter above the walk-away is *not* emitted.
>
> The rest of this section stands: the direction of `_detect_outliers` is unchanged, and the
> `0.2` / `0.4` log-string mismatch at :10637 is still there.
- **`_detect_outliers` (:10585-10660) guards the opposite direction.** Both its price rails —
  `market_gap` and `walkaway_gap` — are `(reference − offer)/reference`, i.e. they fire when the
  supplier's offer is *below* the reference. Thresholds `MARKET_REVIEW_THRESHOLD = 0.2` and
  `MARKET_ESCALATION_THRESHOLD = 0.4` (:484-485). This detects a suspiciously *cheap* offer
  (error/fraud), which is useful — but it is not an authority-limit check, and the log string at
  :10637 says *"more than 20% below our walk-away price"* while the constant tested is `0.4`.
- Volume (`MAX_VOLUME_LIMIT = 1000`) and payment term (`MAX_TERM_DAYS = 120`) rails do fire on
  the over-limit direction, with `×1.5` and `×2` escalation multipliers.
- **HITL is enforced before dispatch and is genuinely solid.** `_hitl_enforced` (:2918),
  `_resolve_hitl_decision` (:2979) — and critically, since `f01e0b5` / `d712468`, a payload's
  own `hitl_decisions` claim is **not** trusted: it is verified against a real `proc.bp_approval`
  row (:3061, :3080) before a round proceeds. This is the strongest control in the domain.

### Approval matrix

`proc.bp_approval` is the record (`src/services/approval_store.py`, append-only; a revocation is
a new row, :135). Negotiation consults it through the HITL path above, and
`src/services/email_dispatch_guard.py:244` gates dispatch. There is **no row-typed approval
matrix** — no table mapping (value band × category × action) → required approver. The rails are
the four env constants above, not a matrix.

---

## 1.7 Formula inventory

`R` = registered under `@formula` in `src/services/formulas/definitions/negotiation.py` (built
earlier today, 2026-09-05). `PH` = **currently computing on placeholder inputs — must not be
snapshotted as a golden vector.**

| # | File · function | Produces | Hardcoded constants | Tests | Audit rec | R | PH |
|---|---|---|---|---|---|---|---|
| 1 | `negotiation_agent.py:304` `plan_counter` | counter_price | 0.88, 0.10, 0.6, 0.02 tolerance | yes | no | ✅ `negotiation.counter_plan` | — |
| 2 | `negotiation_agent.py:1467` `compute_decision` | ctx params | aggressiveness .75, leverage .6, urgency .3, risk_buffer .06, min_abs_buffer 3.0, step .12, ask_disc .02 — **first three never read** | yes | no | (folded into #1) | — |
| 3 | `negotiation_agent.py:6399` `_estimate_zopa` | buyer_max, **supplier_floor**, entry_counter | ×1.03 capacity, ×1.01 firm tone, **×0.85 floor fallback**, concession clamp 0.03–0.12, default 0.05 | no | no | ❌ | **PH** |
| 4 | `negotiation_agent.py:6450` `_adaptive_strategy` | counter_price override | midpoint | partial | no | ❌ | **PH** (consumes #3) |
| 5 | `negotiation_agent.py:6505` `_optimize_multi_issue` | best package score, counter_price | weights .5/.2/.2/.1, `COST_OF_CAPITAL_APR` .12, `LEAD_TIME_VALUE_PCT_PER_WEEK` .01, `AGGRESSIVE_FIRST_COUNTER_PCT` .12, risk +.02, OTIF bars .9/.97, tiers 100/250/500 | no | no | ❌ | **PH** (consumes #3) |
| 6 | `negotiation_agent.py:10585` `_detect_outliers` | requires_review, human_override | .2, .4, 1000, 120, ×1.5, ×2 | partial | no | ❌ | — |
| 7 | `negotiation_agent.py:10504` `_build_positions` | start/desired/no_deal | — | yes | no | ❌ | — |
| 8 | `classification.py:38` `classify` | quadrant, style, confidence | **high_spend 98175.0, many_alternatives 93** (D-10/D-11) | yes | no | ✅ `negotiation.kraljic_quadrant` | — |
| 9 | `classification.py:29` `_confidence` | 0.5–1.0 | 0.5 base | yes | no | ✅ `negotiation.threshold_confidence` | — |
| 10 | `ranking.py:15` `rank_plays` base score (:250) | play base | 1.0 + idx×0.01 | yes | no | ✅ `negotiation.play_rank` | — |
| 11 | `ranking.py:113` `_score_policy_alignment` | −0.7…+0.6 | .6, .3, −.3, −.7 | yes | no | ✅ `negotiation.policy_alignment_score` | — |
| 12 | `ranking.py:133` `_score_supplier_performance` | 0…+0.4 | .9, .97, .4, .2, −.1, .3, .6, .2, .5, .25 | yes | no | ✅ `negotiation.supplier_performance_score` | — |
| 13 | `ranking.py:185` `_score_market_context` | 0…+0.3 | .3, .2, .25 | yes | no | ✅ `negotiation.market_context_score` | — |
| 14 | `grounding.py:39` `assess` | play state | family keyword table | yes | no | ✅ `negotiation.play_readiness` | — |
| 15 | **`negotiate_dashboard.py:259` (negotiation-strategy block)** | **supplierRate, ourAim ("optimalPrice"), walkAway** | **50, 47, 50, −3, ×50** | **no** | no | ❌ | **PH — user-facing** |
| 16 | `negotiation_strategy_engine.py:264` `generate_position` | target_price | 6 discount rates | yes | no | ❌ | DEAD |

**Registered: 11 of 16 as of 2026-09-05 (was 8).** Items 3, 6 and the salvaged
BATNA are now registered as `negotiation.zopa_estimate`,
`negotiation.outlier_rails` and `negotiation.batna_strength`, with every vector
marked PROVISIONAL — they pin *arithmetic*, not correctness. Items 4, 5, 7 and
15 remain unregistered: 4 and 5 are pure functions of item 3 and gain nothing
until it has a real cost basis; 15 no longer computes a number at all.
**Zero of the sixteen write an audit record.**

> **↻ 2026-09-07 — CORRECTION 5 of 5. The last line is wrong; the 11 registered formulas all
> write one.** `evaluate()` builds a full `EvaluationRecord` — inputs, output, version + version
> hash, findings, confidence, duration, trace/deal/document ids — and calls `_audit.emit(record)`
> on every path, including both refusal paths (`evaluate.py:178, 198, 214, 232`). Refusals are
> recorded as `status="unassessed"` with the blocking finding attached.
>
> The real gap is narrower and worth stating exactly: **the default sink is
> `MemoryAuditSink` — a 5,000-entry in-process ring buffer (`audit.py:94`) — and nothing anywhere
> installs `DbAuditSink`.** Grep for `DbAuditSink` finds only its definition and two re-exports.
> So the records are produced correctly and then die with the process; nothing reaches
> `proc.bp_agent_actions`. Installing the durable sink is a small, self-contained job, and it is
> the cheapest of the outstanding items.
>
> The five unregistered formulas (4, 5, 7, 15) still write nothing, since they never go near the
> registry.

Registering item 6 immediately surfaced a defect the audit had missed:
`volume_units > MAX_VOLUME_LIMIT * 1.5` is a strict `>` against exactly
`1500.0`, so a buyer entering the round number 1500 breaches the review rail
but never the escalation rail. Pinned as a golden vector so it stays visible. Items **3, 4, 5 and 15** must
not receive behaviour-snapshot golden vectors: 3/4/5 compute on a fabricated cost floor, and 15
is a literal placeholder shown to buyers.

> **↻ 2026-09-07 — the `1500` boundary bug is still live**, still pinned as a golden vector, still
> unfixed. The caveat above has aged, though: since `e4eda2e` items 3/4/5 no longer compute on a
> fabricated floor — item 3 returns `None` plus a finding, and 4/5 consume that. They are safe to
> snapshot now, and would gain little until there is a real cost basis to snapshot against.

---

## 2. Summary classification

| | Item |
|---|---|
| **REAL** | HITL enforcement verified against `proc.bp_approval` (:2979-3090); `classification.classify` fail-closed `indeterminate` path; the append-only `approval_store`; the formula registry itself and the 8 registered negotiation formulas; `plan_counter`'s determinism |
| **HEURISTIC** | `plan_counter` round ladder; `_optimize_multi_issue` grid search; payment-term APR pricing; Kraljic thresholds (measured, but frozen in source) |
| **PLACEHOLDER** | `_estimate_zopa` `offer × 0.85` supplier floor; `LEAD_TIME_VALUE_PCT_PER_WEEK = 0.01`; **`negotiate_dashboard` 50/47/50 standpoint shown to buyers**; `compute_decision`'s three never-read behavioural knobs |
| **DEAD** | `negotiation_strategy_engine.py` (405 lines) + `tests/test_negotiation_strategy_engine.py`; `ReasoningEngine.create_plan` / `_rule_based_plan`; `SupplierSignals.offer_prev` / `offer_new`; `NegotiationContext.min_abs_step` |
| **MISSING** | all four target functions; every Track B service (6/6); CISE entirely; tri-state confidence; provenance on extracted counters; bitemporal negotiation state; GPSS; a row-typed approval matrix; walk-away as a ceiling on our own counter |

### ↻ 2026-09-07 — the same table, re-verified

| | Item |
|---|---|
| **REAL** | everything above, plus: **`_respect_positions` enforcing the walk-away ceiling** (moved up from MISSING); the counter price single-sourced through `negotiation.counter_plan`; contract refusal → marked `clarify`; **11 registered formulas, all emitting an `EvaluationRecord`**; the honest `unavailableReason` on the Negotiate page |
| **HEURISTIC** | unchanged, except Kraljic thresholds are now **governed rows in `proc.bp_policy`**, not source constants |
| **PLACEHOLDER** | `LEAD_TIME_VALUE_PCT_PER_WEEK = 0.01`; `compute_decision`'s never-read knobs. **The `offer × 0.85` floor and the 50/47/50 standpoint are gone** (`e4eda2e`) |
| **DEAD** | the strategy engine is now actually deleted; `ReasoningEngine.create_plan`; `offer_prev`/`offer_new`; `min_abs_step`; **plus ~900 lines inside the agent the first pass missed — the three prompt methods, `_compose_negotiation_message`, and the flag-disabled prose composer** (§1.5) |
| **MISSING** | all four target functions; Track B 6/6; CISE; tri-state confidence *on the wire* (the vocabulary now exists and is produced — it reaches no consumer); provenance on extracted counters; bitemporal state; GPSS; a row-typed approval matrix; **a durable audit sink**; **leverage as an input to counter pricing** |

## 3. Honest confidence ceiling

A negotiation recommendation today rests on: a counter price from a **regex first-match over
supplier prose**, against a supplier cost floor that is **`offer × 0.85` when should-cost,
benchmark and history are absent — which is the normal case**, positioned by a quadrant whose
thresholds are **frozen 2026-era corpus percentiles**, with **no leverage input reaching the
price ladder at all**.

**The honest ceiling is: directionally useful, numerically unsupported.** The *decision*
(counter / accept / decline / clarify) and the *play ranking* are defensible and evidence-gated.
The *numbers* — counter_price, supplier_floor, optimalPrice, walkAway — should not be presented
to a buyer as computed positions until Track B exists. Item 15 in particular is presenting a
literal constant as "optimalPrice" today.

> **↻ 2026-09-07 — the ceiling is unchanged, but for one fewer reason.** Two of the four props
> under the paragraph above are gone: there is no invented cost floor (the floor is `None` with a
> named finding), and no literal constant is presented as "optimalPrice". The two that remain are
> the load-bearing ones — **the counter price still originates in a regex first-match over
> supplier prose, and no leverage signal reaches the price ladder.**
>
> So the verdict holds word for word: **directionally useful, numerically unsupported.** What
> changed is that the system now *says so* in the two places it previously guessed, instead of
> presenting a guess as a computation. That is the difference between wrong and unassessed, and it
> was the point of `e4eda2e`.
>
> Worth recording alongside it: `proc.bp_negotiation_advice`, `proc.negotiation_sessions` and
> `proc.negotiation_session_state` are **all empty** in bp_testdb as of 2026-09-07. No negotiation
> has ever run against this corpus, so none of the above has been exercised on real data.

---

## 4. Remediation status (updated 2026-09-05, post-approval)

All four decisions in §5 were approved. Landed so far, in the order requested:

| Commit | Step |
|---|---|
| `343b5a0` | BATNA salvaged into `src/services/negotiation/leverage.py`; `negotiation_strategy_engine.py` and its test file deleted; planner rewired onto `BatnaAssessment` |
| `e4eda2e` | The `price * 0.85` cost floor and the 50/47/50 dashboard standpoint removed |
| `b86e851` | This audit and `docs/adr/0001-negotiation-track-b-deferral.md` |
| *(uncommitted)* | Three formulas registered PROVISIONAL — the registry itself is another session's untracked work in progress |

**Still open**, deferred into the four-function refactor rather than retrofitted
onto the 13k-line class: fail-closed on every function (§2 step 2), confidence
propagation to the Action Centre (step 3), and criticality as a required
leverage input (step 4).

> **↻ 2026-09-07 — the registry landed, and one of the three deferred items is partly done.**
> The "remaining open question" in §5 is closed: `src/services/formulas/` is tracked, committed in
> `9988eaa` (the registry, 25 files) and `d76b5d0` (callers, the ADR and the gap report). It is not
> a passive record — `decide_strategy` and `negotiation_advice.advisor` both *call* `evaluate()`
> on the live path, so the contract runs in production.
>
> Revised status of the three deferred items:
>
> | Item | Status 2026-09-07 |
> |---|---|
> | fail-closed on every function (step 2) | **partly landed.** The contract refuses bad input to `negotiation.counter_plan` and `decide_strategy` returns a marked `clarify`. `classify` already failed closed. The other functions are untouched. |
> | confidence propagation (step 3) | **untouched, now precisely diagnosable.** The marker is produced and read by nobody — §1.4 names the three breaks. Fixing the sink and surfacing `zopa["findings"]` are separable from the refactor. |
> | criticality as a required leverage input (step 4) | **untouched.** `plan_counter` still never sees the quadrant. |

### The path that was recommended, and taken

1. ~~**Stop the bleeding.**~~ Done in `e4eda2e`. Note the consequence found while making the
   change: the counter-email justification gated its "based on our market analysis and
   benchmarking" claim on a >15% gap to the fabricated floor, and
   `(offer − 0.85·offer)/(0.85·offer) = 0.1765` — so that sentence was sent to **every supplier
   on every negotiation**, benchmark or no benchmark. Removing the floor removed the claim.
   Separately, no UI consumer of `currentStandpoint`/`optimalPrice`/`walkAway` was found in
   `beyond_procwise_ui`, so item 15 was served but not rendered.
2. ~~**Do not delete the strategy engine yet.**~~ Done in `343b5a0`, salvage first.
3. **Then** the refactor to four functions, with fail-closed contracts and confidence
   propagation designed in from the start — rather than retrofitting them onto the 13k-line class.
4. Register items 3–7 and 15 with `PROVISIONAL` vectors as part of step 1, not before it.
5. ADR for Track B deferral — writable now, independent of the above.

## 5. Decisions (answered 2026-09-05 — all four approved)

1. **Is the four-function refactor still the target?** → **Yes.** Treated as a build to be
   planned, not started at the tail of the audit session. Steps 2–4 of Phase 2 belong inside it.
2. **May I change the numbers?** → **Yes**, both changed in `e4eda2e`.
3. **CISE** → Kraljic-from-signals is the intended substitute; §1.3's assumption stands. There
   is no criticality service to wire, so Phase 2 step 4 becomes "make the Kraljic quadrant a
   required leverage input to tactic selection, with `indeterminate` yielding a provisional
   recommendation" — `classification.classify` already fails closed correctly.
4. **`negotiation_strategy_engine.py`** → salvage-then-delete. Done in `343b5a0`.

### Remaining open question

The formula registry (`src/services/formulas/`) is **entirely untracked** in this checkout —
another session's work in progress. The three PROVISIONAL registrations from this work sit in
its `definitions/negotiation.py` and cannot be committed independently without landing a file
that imports untracked modules. How that lands is not this work's call.

> **↻ 2026-09-07 — closed.** It landed as `9988eaa` + `d76b5d0`, tracked, with 11 negotiation
> formulas registered and their goldens under CI (`.github/workflows/formula-goldens.yml`).

---

## 6. ↻ Re-verification, 2026-09-07

Method: read `negotiation_agent.py` end to end (13,342 lines) plus the advice services, the
registry, the dashboard and the router; then **execute** each load-bearing claim against the live
`.env` database rather than re-reading the source. Suites re-run: **250 passed** in 188s.

### The five corrections

| # | Original finding | Verdict | Evidence |
|---|---|---|---|
| 1 | `generate_plan`'s analogue silently defaults / raises `KeyError` | **stale** | `decide_strategy` routes through the registry; a refused contract returns `clarify` + `decision_origin="contract_refused"`. Ran it with `current_offer=-5`. |
| 2 | Three things set the price and disagree | **stale** | `price_plan_locked` makes `negotiation.counter_plan` authoritative; the other two price branches are dead in production. |
| 3 | **Walk-away is not a ceiling on our own counter** | **WRONG** | `_respect_positions` clamps `min(counter, walkaway)` on the live path. Counter 88.00 vs walk-away 85.00 → **85.00**. |
| 4 | Kraljic thresholds frozen in source | **stale** | Governed row `proc.bp_policy` id 605. Values unchanged; D-10 narrowed, not closed. |
| 5 | **Zero of the sixteen write an audit record** | **WRONG** | All 11 registered formulas emit an `EvaluationRecord`, refusals included. The gap is the *sink*: `MemoryAuditSink` by default, `DbAuditSink` installed nowhere. |

Correction 3 is the one to remember: it was derived by reading `plan_counter` in isolation and
never checking what the agent does with its return value. **A guardrail can live one call frame
away from the function that appears to be missing it.**

### Found on re-verification, not in the original pass

1. **A best-and-final we want to accept cannot close the negotiation.** `_adaptive_strategy`
   overwrites `accept` with `package-trade`; the round loop only closes on `accept`/`decline`.
   Verified end-to-end. This is a bug, not a design gap — see §1.1 `select_tactic`.
2. **The UNASSESSED cost-floor marker reaches no consumer.** Produced at :6497, read nowhere;
   its only rendering point is in dead code. §1.4.
3. **~900 dead lines inside the agent**, including the three prompt methods and a ~700-line prose
   composer behind a flag that defaults off. §1.5.
4. **The LLM signal leg was observed failing** — Ollama `500` ×3, swallowed, no trace in the
   output. §1.4.
5. **Nothing has ever run here.** The three negotiation tables are empty in bp_testdb.

### Outstanding, in the order worth doing

1. **The four-function refactor** — still the target, still not started. Fail-closed is partly
   landed; confidence propagation and criticality-as-leverage are untouched.
2. **Close the accept path** (finding 1). Small, and it stops finished negotiations finishing.
3. **Install `DbAuditSink`** — the records already exist; only the sink is missing.
4. **Surface `zopa["findings"]`**, and delete or wire the dead code around it.
5. Standing defects: leverage unread; outlier rails one-directional; the `1500` strict-`>`
   boundary; the swallowed LLM failure.
6. Formula registration 11/16; items 4 and 5 deliberately blocked on a real cost basis.
7. **Track B 6/6** — deferred with written triggers (ADR 0001), none scheduled.
