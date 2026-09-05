# Negotiation Agent — State Audit

**Date:** 2026-09-05 · **Branch:** Development · **Baseline commit:** `2e27c58`
**Test baseline:** `tests/test_negotiation_agent.py`, `tests/test_negotiation_strategy_engine.py`,
`tests/test_negotiation_skills.py`, `tests/services/negotiation_advice/` — **214 passed** in 165s.

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

### `evaluate_counter` → **HEURISTIC** (`_estimate_zopa` + `_optimize_multi_issue`)

`:6399-6448` and `:6505-6651`. No LLM. `_optimize_multi_issue` scores an exhaustively enumerated
grid (≤5 prices × 4 term packages × 1 lead package × 4 volume tiers = ≤80 options) with a
normalised weighted sum, default weights `price .5 / delivery .2 / risk .2 / terms .1`.

**Missing-input behaviour: silent.** `_estimate_zopa` falls back to `price * 0.85` for
`supplier_floor` when should-cost, benchmark p10 and historic minimum are all absent (:6431) —
i.e. **it invents a 15%-below-offer supplier cost floor out of nothing**, and the caller cannot
tell that from a floor derived from a real should-cost model. This is the single most
consequential placeholder in the domain.

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

Registering item 6 immediately surfaced a defect the audit had missed:
`volume_units > MAX_VOLUME_LIMIT * 1.5` is a strict `>` against exactly
`1500.0`, so a buyer entering the round number 1500 breaches the review rail
but never the escalation rail. Pinned as a golden vector so it stays visible. Items **3, 4, 5 and 15** must
not receive behaviour-snapshot golden vectors: 3/4/5 compute on a fabricated cost floor, and 15
is a literal placeholder shown to buyers.

---

## 2. Summary classification

| | Item |
|---|---|
| **REAL** | HITL enforcement verified against `proc.bp_approval` (:2979-3090); `classification.classify` fail-closed `indeterminate` path; the append-only `approval_store`; the formula registry itself and the 8 registered negotiation formulas; `plan_counter`'s determinism |
| **HEURISTIC** | `plan_counter` round ladder; `_optimize_multi_issue` grid search; payment-term APR pricing; Kraljic thresholds (measured, but frozen in source) |
| **PLACEHOLDER** | `_estimate_zopa` `offer × 0.85` supplier floor; `LEAD_TIME_VALUE_PCT_PER_WEEK = 0.01`; **`negotiate_dashboard` 50/47/50 standpoint shown to buyers**; `compute_decision`'s three never-read behavioural knobs |
| **DEAD** | `negotiation_strategy_engine.py` (405 lines) + `tests/test_negotiation_strategy_engine.py`; `ReasoningEngine.create_plan` / `_rule_based_plan`; `SupplierSignals.offer_prev` / `offer_new`; `NegotiationContext.min_abs_step` |
| **MISSING** | all four target functions; every Track B service (6/6); CISE entirely; tri-state confidence; provenance on extracted counters; bitemporal negotiation state; GPSS; a row-typed approval matrix; walk-away as a ceiling on our own counter |

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
