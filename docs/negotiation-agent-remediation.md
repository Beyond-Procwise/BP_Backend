# Negotiation Agent — Remediation Brief

**Date:** 2026-09-07 · **Branch:** `Development` · **Baseline commit:** `1b18257`
**Subject:** `src/agents/negotiation_agent.py` (13,365 lines) and everything that reaches it
**Evidence:** static trace of the live execution path, plus execution of the load-bearing claims
against `./venv`. Design docs, comments, schemas and agent registers were not treated as evidence
of behaviour.

**Relationship to the existing audit:** this supersedes the *body* of
[`docs/negotiation-agent-state-audit.md`](negotiation-agent-state-audit.md) (2026-09-05) but **not
its §6 corrections**, which still apply. That document asks "does the agent do what the brief
claims"; this one asks "what is it wired to, and what has to change". Read §6 of the audit before
quoting anything from its 2026-09-05 body.

**Published copy:** <https://claude.ai/code/artifact/116b5502-75e4-4fb8-be51-0add5afa5f8e>

**Status, 2026-09-07:** **Phase 0 is complete** — 0a `25318ac`, 0b `a6655b6`, 0c `b5aa637`,
0d `260c8f2`. Only 0e (the learning-snapshot drain, carried out of 0b) remains open. §5 records what
shipped, what changed behaviour, and what was carried forward. Everything below the status line describes the code
as it was found; where a finding has since been acted on it is marked in §5, not rewritten in place.

---

## 0. How to read this

Every claim carries a `file:line`. Where a claim was cheap to prove by running the code, the
output is reproduced verbatim rather than described.

Grades used throughout:

| Grade | Meaning |
|---|---|
| **IMPLEMENTED** | Reachable from a live entry point and does what its name says. |
| **PARTIAL** | Works under conditions that are not the ordinary case. |
| **ORPHANED** | The code exists and is often good; nothing on the live path calls it. |
| **ABSENT** | No implementation found. |

> **The one-sentence version.** A fail-closed spend mandate, a fail-closed human-approval gate, a
> grounded Kraljic advisor, a BATNA assessor and a benchmark pricing engine all exist in this
> repository. None of them is connected to the code that decides what price to offer a supplier.

---

## 1. What actually runs

Five routes reach `NegotiationAgent` on paper. One runs in production.

| Route | Where | State | Why |
|---|---|---|---|
| EmailWatcher | `email_watcher.py:1401` | **LIVE** | The only route that runs end to end. No orchestrator, so no workflow context, no authority block, no HITL resolution. The returned `AgentOutput` is discarded. |
| `POST /workflows` | `workflows.py:2072` | **LIVE, thin** | `"negotiation"` is not in `WORKFLOW_REGISTRY`, so it falls to `_execute_generic_workflow`. Chains `next_agents` correctly — the only route that does. |
| `supplier_interaction` graph | `workflow_definitions.py:441` | **UNREACHABLE** | Edge condition can never be true. See §2.1. |
| ~~`_execute_supplier_interaction_workflow`~~ | ~~`orchestrator.py:2202`~~ | **DELETED** `260c8f2` | Was shadowed in normal operation and reachable only when `WorkflowEngine` failed to construct. See §2.2. |
| `quote_evaluation` graph | `workflow_definitions.py:305` | **CONDITIONAL** | Fires only if the API caller supplies `supplier`, `current_offer`, `target_price` and `rfq_id` themselves. `QuoteEvaluationAgent` returns none of them. |

### 1.1 The decision chain, per supplier per round

Inside `_run_single_negotiation_locked` (`:5056–6365`):

```
_normalise_negotiation_inputs   :10356   offer / target / walkaway parsed from payload
_build_positions                :10572   start · desired · no_deal   ← all asserted, none computed
_extract_negotiation_signals    :6365    regex + one LLM call (:6417, no grammar)
_estimate_zopa                  :6432    supplier_floor, or None + an honest finding
decide_strategy                 :1522    → formula registry → plan_counter :313
_adaptive_strategy              :6508    finality → strategy = "package-trade"
_respect_positions              :10630   clamp to target / walkaway / previous counter
_detect_outliers                :10658   4 rails → strategy = "review"   ← one rail inverted
_optimize_multi_issue           :6578    80-cell weighted grid  → price DISCARDED at :5553
_invoke_email_drafting_agent    :5864    LLM writes the outbound body
```

There are exactly **two LLM call sites** on this path: `negotiation_agent.py:6417`
(`ollama_generate`, temperature 0.1, free-form JSON, no grammar and no schema validation — its
`finality_hint` flips the strategy at `:6539`) and `email_drafting_agent.py:240-248` (`_chat` →
`call_ollama`, which writes the body the supplier receives). Neither has a structured-output
contract.

### 1.2 Integration facts

**Shared state: none.** `NegotiationAgent` never calls `set_workflow_context`, `update_shared` or
`emit_signal`. The orchestrator attaches a `WorkflowContext` at `orchestrator.py:2746`; the agent
ignores it. The three agents that do use the blackboard are `base_agent`, `email_drafting_agent`
and `data_extraction_agent`.

**Identity spaces do not overlap.** The agent is keyed on `workflow_id` / `rfq_id` /
`session_reference` / `supplier_id`. The whole `negotiation_advice` package and
`NegotiationAdvisorAgent` are keyed on `deal_id`.

```
$ grep -c deal_id src/agents/negotiation_agent.py
0
```

**Nothing reads what the agent persists.** `proc.negotiation_session_state` and
`proc.negotiation_sessions` are read in exactly one other place —
`supplier_interaction_agent.py:3585-3605` — and only to resolve a `supplier_id` from an `rfq_id`.
The Negotiate dashboard builds its rounds from `proc.supplier_response.round_number`
(`negotiate_dashboard.py:159, 219, 248`) and its prices from `_trgt` tables. What the agent decided
never reaches the screen named after it.

**`next_agents` is collected but not dispatched.** The agent emits `["EmailDraftingAgent"]`
(`:5919`), `["SupplierInteractionAgent"]` (`:5511, :6284`) and `{"QuoteEvaluationAgent"}`
(`:3510-3511`). On the supplier-interaction path these are appended to `results["next_agents"]`
(`orchestrator.py:2271-2274`) and nothing runs them — compare `orchestrator.py:1673-1679` and
`:2496-2503`, which do chain.

---

## 2. Three broken seams

Ranked by what closing them unlocks, not by size. All three are wiring, not capability.

### 2.1 The `supplier_interaction` graph cannot reach the negotiate node

`EmailWatcherAgent` returns `data["supplier_responses"]` (`email_watcher_agent.py:572`). The graph
node asks for a key that does not exist:

- `output_to_shared=["responses"]` — `workflow_definitions.py:449`
- `input_mapping={"watch_responses.responses": "responses"}` — `:427`
- The engine copies with `result.data.get(output_field)` — `workflow_engine.py:613`

So `shared_data["responses"]` is never set, `_has_responses` (`:141`) is permanently `False`, and
the edge at `:441` never fires.

```
$ ./venv/bin/python  # simulating the engine with EmailWatcherAgent's real data keys
negotiate node input_mapping: {'watch_responses.responses': 'responses'}
watch_responses output_to_shared: ['responses']

shared_data after watch_responses: {}
edge watch_responses->negotiate fires?  False
negotiate input_data: {}
_extract_batch_inputs on that payload: ([], {})
```

```
draft_emails → dispatch_emails → watch_responses  ─╳→  negotiate  ─╳→  compare_quotes
                                 writes                reads              never runs
                        "supplier_responses"       "responses"
```

Two nodes downstream have therefore never executed in this graph. If the edge were fixed but the
mapping were not, the node would receive `{}`, fall through to `_run_single_negotiation`, skip HITL
entirely, and return `clarify`.

**Fix.** In `build_supplier_interaction_workflow`, use the key the watcher actually emits and
update the edge predicate to match:

- `:449` → `output_to_shared=["supplier_responses"]`
- `:427` → `input_mapping={"watch_responses.supplier_responses": "supplier_responses"}`
- `:141` → `_has_responses` reads `supplier_responses`

Add a test asserting the negotiate node receives a non-empty batch given `EmailWatcherAgent`'s
documented return shape. This class of bug — a node declaring a key its upstream agent does not
emit — is caught by no current test, and `workflow_engine.py:617-622` records a previous instance
of exactly the same failure with `supplier_candidates`.

### 2.2 The one working orchestrated call into negotiation is shadowed

`execute_workflow` checks the declarative engine before the bespoke handlers:

```python
# src/orchestration/orchestrator.py:461
use_engine = (self._workflow_engine is not None
              and workflow_name in self._workflow_registry
              and not workflow_config
              and enriched_input.get("use_workflow_engine", True))
```

`"supplier_interaction"` is in `WORKFLOW_REGISTRY` (`workflow_definitions.py:556`), and
`use_workflow_engine` is set to `False` nowhere in the repository — the default at `:465` is its
only occurrence. So in normal operation `_execute_supplier_interaction_workflow`
(`orchestrator.py:493`, defined `:1989`), which contains the only code that builds a real
negotiation payload and dispatches it (`:2202-2233`), never runs. It is shadowed by the graph in
§2.1, which cannot reach the node.

> **↻ Corrected 2026-09-07.** An earlier revision of this document called that method *dead code in
> production*. That is wrong, and the distinction changes what to do about it. `orchestrator.py:176-179`
> catches **any** `WorkflowEngine` construction failure and sets `_workflow_engine = None` with an
> empty `_workflow_registry`, logging *"Workflow engine init failed, using legacy routing"*. `use_engine`
> is then false and the bespoke path runs. It is a **degraded-mode fallback**, not dead code.

Two further facts bear on the decision:

* **Its own tests stopped exercising it and nobody noticed.** All four tests in
  `tests/test_orchestrator_supplier_workflow.py` fail on an empty `.calls` list — the stub agent is
  never invoked, because `execute_workflow("supplier_interaction", …)` takes the engine path. They
  broke when the declarative engine began shadowing the method, and the failure was tolerated. So
  the fallback is not merely unused; it is unverified.
* **It is not ungoverned.** `_apply_authority` runs at `orchestrator.py:429`, before the branch at
  `:442`, so both paths receive the injected mandate.

> **✔ Closed 2026-09-07 by 0d** (`260c8f2`). The fallback was deleted and a graph-backed workflow
> now fails rather than routing elsewhere. See §5 for what that means operationally.

**Fix — the decision taken.** Fixing §2.1 makes the declarative graph the
single control flow in normal operation, which is the intended direction (12-Factor #8, per the
comment at `orchestrator.py:459`). What to do with the fallback is a real choice:

* **Delete it, and let a failed engine init fail loudly.** One control flow, fail-closed. Silently
  rerouting a workflow through a second, unverified implementation with different behaviour is the
  class of problem this document exists to describe — and since the fallback's own tests do not
  pass, it would likely fail anyway. **← chosen.**
* ~~Keep it as a documented degraded mode, and fix its four tests so it is genuinely exercised.~~

### 2.3 The spend mandate is resolved, injected, and never read

> **✔ Closed 2026-09-07 by 0c** (`b5aa637`). Described below as it was found; see §5 for what the
> enforcement does and the two decisions behind it.

`src/services/governance_tools/authority.py` is a well-built fail-closed mandate resolver:
`limit_gbp`, `auto_intents`, `escalate_intents`, resolved per agent through
`PolicyEngine.get_policy()`, with every failure path returning `ungoverned_block()` meaning
*escalate*. The orchestrator resolves it for the negotiation workflow specifically —
`"negotiation": ["email_drafting_agent", "negotiation_agent"]` (`orchestrator.py:326-328`) — and
injects it:

```python
# src/orchestration/orchestrator.py:363
if isinstance(enriched_input, dict):
    enriched_input["authority"] = resolved
try:
    context.input_data["authority"] = resolved      # ← lands in the agent's own input
```

```
$ grep '"authority"' src/agents/negotiation_agent.py
(no output)
```

The only consumer in the codebase is `DecisionEngine` (`decision_engine.py:713-729`), which
resolves its own and governs email replies on the `/decisions` API — a system that never touches
`NegotiationAgent`. The agent therefore has no enforced ceiling on what it may commit to. The only
limit in its code is `_respect_positions` (`:10630`) clamping to a `walkaway_price` that nothing in
the repository produces.

**Fix.** Read `input_data["authority"]["negotiation_agent"]` in `_run_single_negotiation_locked`
before the decision is returned. Refuse to counter above `limit_gbp`; when `governed` is `False`,
return `strategy="review"` with the block's own `reason` — it is written to be shown to a buyer
verbatim (`authority.py:32-40`). On the EmailWatcher route there is no orchestrator, so the agent
must resolve authority itself rather than treat a missing block as permission.

---

## 3. What the agent does today

Condensed from the full capability matrix. The pattern is consistent: the sophisticated component
exists and is disconnected; the connected component is a fixed table.

| Capability | Grade | Evidence | What is actually there |
|---|---|---|---|
| Concession schedule | IMPLEMENTED | `:313-528` | Deterministic and under formula-registry contract. R1 `offer×0.88` or midpoint; R2 60% of gap; R3+ `target + max(3.0, target×0.06)`. |
| Leverage / urgency effect on price | ABSENT | `:282-288`, `:1502` | Fields defined and hardwired; `plan_counter` reads none of them. Proven: leverage 0.0 and 1.0 both return 88.0. |
| Counterparty modelling | ABSENT | `:297-298`, `:1515` | `offer_prev`/`offer_new` threaded through two signatures, never read. Supplier moving £1 and £40 produce identical counters. |
| Multi-issue optimiser | ORPHANED | `:6578`, discarded `:5553` | Real weights, real 80-cell grid. Its price is filtered out because `decide_strategy` always sets `price_plan_locked` (`:1576`). Only three ask-strings survive. |
| Trade-off / log-rolling | ABSENT | `:6599` | `_ = policy, constraints  # reserved for future`. With no coupling the objective is separable, so the argmax is the best value on every axis at once — asking for everything, not trading. |
| MESOs (equal-value packages) | ABSENT | `:6698-6721` | Appends only `best_option`; `counter_options` is always length 1. |
| BATNA object | ORPHANED | `services/negotiation/leverage.py:82` | Good code, correct UNASSESSED-is-not-zero discipline. Sole caller is `ReasoningEngine._rule_based_plan`, whose chain terminates in `process_task` — no callers in `src/`. |
| Reservation value / walk-away | ABSENT | `:10356-10387` | Three payload keys. No producer anywhere. Benchmark engine exists at `src/services/benchmark/`, unwired. |
| ZOPA estimation | PARTIAL | `:6432-6506` | Buyer max is our own ceiling, not inferred. Supplier floor from should-cost / benchmark p10 / history, else `None` plus an honest `findings` marker. |
| `should_cost` input | ABSENT | one read at `:5199` | Zero producers in the repository. The costed-floor branch has never executed in production. |
| Kraljic / play selection | ORPHANED | `:7596-7615` | Returns `{"plays": []}` unless `input_data["supplier_type"]` is set. No live caller sets it. The classifier itself is the best-built code in the domain — and it is `deal_id`-keyed. |
| Deal-size tiering | ABSENT | — | No size bands on the negotiation path. Same 3-round loop and same gate for a £500 order and a £5M framework. `max_rounds` hard-capped at 3 (`:3257`). |
| HITL approval gate | PARTIAL → **reachable** | `:3012-3125` | Excellent and genuinely fail-closed. Its only call site (`:3312`) is in `_run_multi_round_negotiation`, which nothing could reach until 0a. Still gates only that path — `_run_single_negotiation` and `_run_batch_negotiations` never call it. |
| Spend mandate | ABSENT → **IMPLEMENTED** (0c) | `:10700-10850`, `authority.py` | Live limit GBP 10,000. Withholds the counter price when the commitment is over it, uncomputable, or in a currency the limit is not set in; escalates a terminal accept rather than rewriting it. |
| Approval → resume | ABSENT | `:2530`, `:3499-3506` | `POST /approvals/round/{wf}/{n}` writes a `bp_approval` row and nothing consumes it. A paused negotiation cannot be resumed. |
| Dispatch approval gate | IMPLEMENTED | `email_dispatch_guard.py:356` | Requires a `proc.bp_approval` row before any send, on every path. Currently carrying the entire governance load on its own. |
| Round audit trail | ABSENT | `:1685-1718` | Ends in `logger.info("NEGOTIATION_ROUND_EVENT %s", …)`. No table. Positions, rationale and evidence are not co-recorded anywhere queryable. |
| Outcome learning | PARTIAL | `learning_repository.py:205` | Records what we proposed — strategy, counter, asks. Never what happened: no accepted price, no win/loss. The drain is orchestrator-only, so the live EmailWatcher route drops every snapshot. |
| Conduct guardrails | COUNTER-INDICATED | `email_drafting_agent.py:107, 117, 292, 397` | Prompt instructs the model to imply competing proposals and assert deadlines, on round number alone, with nothing checking either is true. |
| Test harness | ABSENT | `tests/test_negotiation_*.py` | 70 tests, all unit tests of `plan_counter` branches, HITL and plumbing. No scenario suite, no outcome assertions, no adversarial counterparty. |

---

## 4. Where it produces a bad outcome silently

These three fail confidently rather than visibly. Fix them before any capability work.

### 4.1 The walk-away rail is inverted

Both price rails in `_detect_outliers` compute `(reference − offer) / reference` (`:10683-10711`),
so they fire only when the offer is *below* the reference. For a buyer the walk-away is the maximum
payable, so the rail is watching the wrong direction:

```
$ ./venv/bin/python  # target 80, walk-away 95
offer 58% ABOVE walk-away    review=False  escalate=False  alerts=[]
offer just above walk-away   review=False  escalate=False  alerts=[]
offer 26% BELOW walk-away    review=True   escalate=False  alerts=['26.3% below walk-away 95.00']
```

The exact condition the guardrail exists to catch produces silence; a good outcome triggers a
review. The escalation copy also claims "more than 20% below our walk-away" while
`MARKET_ESCALATION_THRESHOLD` is `0.4` (`:73-74`, `:10707`).

**Fix.** Add an over-ceiling rail — fire when `offer > walkaway` — and align the message to the
constant. Keep the existing under-market rail; a suspiciously low bid is also worth flagging.

### 4.2 Two different prices ship in one payload

The optimiser's price is dropped from the decision at `:5553` but survives in `counter_options`
(`:5556`), which is published to `draft_stub["counter_proposals"]` (`:5768`) and
`data["counter_proposals"]` (`:12940`):

```
$ ./venv/bin/python  # offer 100, target 80, walk-away 95, hard_constraints max_price 72
decide_strategy ->  88.0   locked: True
optimiser       ->  78.4   counter_options: [{'price': 78.4, 'terms': 'Net15 with 2% disc', ...}]
LIVE MERGE      -> counter_price on the wire: 88.0
hard_constraints max_price=72 respected by optimiser? 78.4      ← ignored
```

Whichever surface a reviewer approves from, the other is wrong — and the approval binds to neither,
because a round approval records no content hash (`approvals.py:246`).

> **◐ Partly closed 2026-09-07 by 0c** (`b5aa637`): `counter_options` is now cleared when the
> mandate withholds a counter, because the leak fired inside 0c's own test. The general divergence
> below — an in-mandate counter shipping 88.00 in the email and 78.40 in `counter_proposals` — is
> untouched and remains task 1b.

**Fix.** Pick one price as authoritative. Either stop publishing `counter_options`, or drop
`price_plan_locked` and let the optimiser's price through with the hard-constraint clamp actually
implemented. Then bind the round approval to a content hash of the counter, the way draft approvals
already do (`approvals.py:230-235`).

### 4.3 The outbound email is licensed to make claims nothing verifies

`email_drafting_agent.py` instructs the model to *"Reference competitive pressure subtly: 'We're
evaluating multiple proposals…'"* (`:107`) and *"Create urgency with real deadlines"* (`:117`,
`:292`, `:397`), triggered by round number or an `urgent` flag. Nothing checks that competing
proposals exist or that the deadline is real. `check_dispatch` validates recipients and content
sensitivity, not truthfulness.

**Fix.** Shortest path is to delete those four prompt lines — the tactic is not worth the exposure
for a regulated buyer. The fuller fix is a post-generation check that any competitive or deadline
claim maps to a fact in the payload, on the model already used for grounding elsewhere.

---

## 5. The change list

Ordered by dependency. **Phase 0 is entirely extraction from code that already exists** and should
land before anything else — it is the difference between a system with governance and a system that
owns governance code.

### Phase 0 — Close the seams *(extraction only)*

No new capability. Every piece already exists; these tasks connect them. Phase 0 alone makes the
multi-round HITL path reachable for the first time and gives the agent the spend mandate the
platform already resolves for it.

| # | Change | Files | Status | Blocks |
|---|---|---|---|---|
| 0a | Rename the graph keys to `supplier_responses` in node output, input mapping and edge predicate; add a shape test | `workflow_definitions.py:141, :427, :449` | **DONE** `25318ac` | 0b, 0d |
| 0b | Capture the `AgentOutput` on the watcher route; persist `hitl_email_tasks` | `email_watcher.py:1401`, `negotiation_agent.py:2530` | **DONE** `a6655b6` | 1c |
| 0c | Read and enforce `input_data["authority"]["negotiation_agent"]`; resolve it directly on the watcher route | `negotiation_agent.py:5056-6365` | **DONE** `b5aa637` | — |
| 0d | Delete the degraded-mode fallback; a graph-backed workflow now fails rather than routing elsewhere | `orchestrator.py:176-181, :448-470` | **DONE** `260c8f2` | — |
| 0e | Drain the learning snapshot on the watcher route — carried out of 0b, see below | `email_watcher.py`, `orchestrator.py:3025-3065` | ~0.5d | 1c |

### What shipped, and what changed behaviour

**0a.** `watch_responses` published `responses`; `EmailWatcherAgent` returns `supplier_responses`.
Renamed in all three places. Before the fix the engine reported
`negotiate: <NodeStatus.SKIPPED: 'skipped'>`; the node now executes. Five tests, including one that
runs the whole graph through the engine and one that feeds the node's built `input_data` to the real
`_extract_batch_inputs`.

**0b.** Held rounds now persist onto the session via `_persist_held_email_tasks`, reduced to the
JSON-safe form the round loop already uses — storing them raw makes `_save_session_state_obj` raise
into its own `except` and store nothing. On the watcher side the `AgentOutput` is captured and the
`delete_responses` call is gated on it: a failed or raising negotiation no longer destroys the only
record of what the supplier offered. `AgentStatus` was not imported in `email_watcher.py` and the
resulting `NameError` was being swallowed by the surrounding `except`. Nine tests; both persistence
guards were verified by breaking them on purpose.

**0c — this one changes commercial behaviour, deliberately.** Two decisions were taken rather than
defaulted:

* *The limit is a total spend; the agent only sees a unit price.* `proc.supplier_response` has no
  quantity column and `supplier_interaction_agent.py` never mentions one, so on the live route the
  commitment a counter represents **cannot be established**. Unknown is treated as outside the
  mandate. The rejected alternative was comparing the unit price to the £10,000 limit, which would
  have passed £250,000 of commitment on a £50 unit price — a guardrail that checks nothing.
* *The number is withheld, not flagged.* A review flag stops nothing here: `_detect_outliers` sets
  one and the email is still drafted around the price.

Verified against the live policy (`ApprovalThresholdPolicy`, GBP 10,000):

```
120 units @ GBP 80  = 9,600      -> counter 80.0  [counter]
120 units @ GBP 90  = 10,800     -> counter None  [review]
   this counter commits GBP 10,800.00, above the governed GBP 10,000 limit
unit price GBP 80, qty unknown   -> counter None  [review]
   the quantity this counter would commit is not stated, so its total value
   cannot be established against the GBP 10,000 limit
120 units @ EUR 80               -> counter None  [review]
   this counter is priced in EUR and the limit is set in GBP; the two cannot be compared
```

Two things fell out of building it:

* **Risk 2 (§4.2) fired during 0c's own test.** Clearing `decision["counter_price"]` withheld
  nothing — `counter_options` still shipped GBP 1,144.00, because `_optimize_multi_issue` knows
  nothing about the mandate and its price is published to `counter_proposals`. Now cleared with it.
* **A terminal `accept` must be escalated, never rewritten.** Turning it into `review` leaves the
  supplier unclosed and reopens bargaining on an offer already agreed — the bug
  `_adaptive_strategy` returns early to avoid. An out-of-mandate accept raises
  `human_override_required` instead. Four tests hold that line.

Two existing tests asserted a counter composed with no mandate at all. They now supply one
explicitly; the gate was not weakened to keep them green.

**0d.** `_execute_supplier_interaction_workflow` is gone (288 lines), with its branch. A failed
engine build is still tolerated at construction — workflows with no declarative graph are
unaffected and the orchestrator still starts — but a workflow that *is* in `WORKFLOW_REGISTRY` now
raises, and `execute_workflow`'s handler turns that into a failed result naming the reason. The init
log moved from `warning` to `error` and no longer claims a fallback exists. Five tests, including
the narrowness guard: a workflow outside the registry still reaches `_execute_generic_workflow`.
`tests/test_orchestrator_supplier_workflow.py` keeps the three tests of helpers that still exist and
loses the four obsolete ones plus their unused stubs — 424 lines to 69. Baseline 28 failures, now
24: exactly the four removed, and no new ones.

**Operationally this is a behaviour change worth knowing.** Before, a broken `WorkflowEngine` meant
`supplier_interaction` quietly ran a different pipeline. Now it returns
`{"status": "failed", "error": "workflow 'supplier_interaction' runs on the declarative workflow
engine, which failed to initialise (…). Refusing to run it another way."}`. If the engine starts
failing in an environment, that surfaces immediately instead of as odd downstream behaviour.

**0e, carried forward.** The learning-snapshot drain was scoped into 0b and is not done. The drain
lives in the orchestrator (`:3025-3065`) and the watcher route has no orchestrator, so doing it
properly means extracting that ~40 lines into a shared function rather than copying it. Left for
its own change.

Regression method for all three: a detached `git worktree` at the pre-Phase-0 commit, never
`git stash` — another session shares this checkout. **50 pre-existing failures before and after,
byte-identical sets**, across the negotiation, watcher, orchestration, governance and guardrail
suites.

### Phase 1 — Stop the silent bad outcomes *(correctness)*

These are defects, not gaps. Each one currently produces a confident wrong answer.

| # | Change | Files | Effort | Blocks |
|---|---|---|---|---|
| 1a | Add the over-ceiling walk-away rail; align escalation copy to the constant | `negotiation_agent.py:10658` | ~2h | — |
| 1b | Resolve the counter-price divergence; one authoritative price on the wire | `negotiation_agent.py:5553, :5768, :12940` | ~0.5d | 1d |
| 1c | Round events to `proc.bp_negotiation_round` instead of a log line; positions + rationale + evidence in one row | `negotiation_agent.py:1685` + new migration | ~2d | 1d |
| 1d | Bind round approvals to a content hash of the counter | `approvals.py:238`, `approval_store.py` | ~1d | — |
| 1e | Remove the competitive-pressure and deadline instructions from the drafting prompt | `email_drafting_agent.py:107, :117, :292, :397` | ~1h | — |
| 1f | Fix `_optimize_multi_issue` sign errors: extended payment terms are a buyer benefit; the `on_time` term is dead weight with an inverted sign | `negotiation_agent.py:6669, :6683` | ~3h | 2b |

### Phase 2 — Make the decisions real *(capability)*

Only start once Phase 0 and 1 are green. Every item here is worthless while the counter price is
unbounded and the round is unrecorded.

| # | Change | Files | Effort | Blocks |
|---|---|---|---|---|
| 2a | Compute a reservation value with provenance; wire `src/services/benchmark/` into `_estimate_zopa` | `negotiation_agent.py:6432` | ~1-2w | 2b, 2c |
| 2b | Wire `assess_batna` into the agent and feed `NegotiationContext.leverage`; make `plan_counter` read leverage, urgency and `offer_prev` | `leverage.py:82`, `negotiation_agent.py:313` | ~1w | — |
| 2c | Join the two identity spaces: give the agent a `deal_id` so `negotiation_advice` classification and plays can reach it | `negotiation_agent.py`, `negotiation_advice/*` | ~1w | 2d |
| 2d | Deal-size tiers: bands, differentiated round caps, evidence bars and approval roles | new + `orchestrator.py` | ~2w | 2e |
| 2e | Genuine multi-issue: per-issue utility, coupling constraints, equal-value package generation | `negotiation_agent.py:6578` | ~4-6w | 2f |
| 2f | Scenario harness with adversarial counterparty simulation and outcome pass criteria | `tests/` (new) | ~2w | — |

---

## 6. Do not tidy these

Several modules on and near this path look redundant and are not. Each encodes a decision that cost
something to learn.

- **`governance_tools/authority.py` vs `envelope.py`** — deliberately separate. The envelope is
  fail-open because it carries prompts; authority is fail-closed because it carries a spend limit.
  The module's own header says not to merge them (`authority.py:1-22`).
- **`_hitl_enforced` returning `False`** — narrows nothing. `hitl_enabled=false` is logged and
  ignored (`:3028-3040`). That is intentional; do not "restore" it as a switch.
- **`_estimate_zopa`'s `None` floor** — the `price × 0.85` fallback was removed on purpose
  (`:6455-6470`). It also gated an unfounded "based on our market analysis and benchmarking" claim
  in outbound email that fired on every supplier, every time, because
  `(offer − 0.85·offer) / (0.85·offer)` is always 0.1765 and the gate was 15%. Do not reintroduce a
  default floor.
- **`assess_batna`'s `UNASSESSED`** — "we never looked" and "there is none" must stay
  distinguishable. The predecessor coerced both to zero and then anchored hardest on the weakest
  position (`leverage.py:28-46`).
- **`TERMINAL_STRATEGIES` as one module-level definition** (`:126`) — two readings disagreeing was
  the bug that stopped an accepted best-and-final from ever closing the negotiation.

One more that looks live and is not: **`src/services/negotiation_session.py` (132 lines) has zero
importers.** `grep -rn "from services.negotiation_session"` over `src/`, `tests/` and `scripts/`
returns nothing. The agent defines its *own* `NegotiationSession` and `SupplierNegotiationState` at
`negotiation_agent.py:13279-13365`, with different fields (`negotiation_parameters`, not
`parameters`; no `workflow_id`), and that is the one `_load_session_state_obj` uses. There are four
classes called `NegotiationSession` in this repository — the agent's, this dead one,
`procurement_workflow.py:194` and `email_thread.py:112`. Importing the wrong one costs an
afternoon; it cost one during 0b.

Two things genuinely are dead and can go: the ~900 lines behind `NEG_USE_ENHANCED_MESSAGES`
(`:1624`, `:7712-8360`), which defaults off and has never run; and the second
`class NegotiationAgent` at `procurement_workflow.py:582`, a template renderer against a
`MockDatabaseConnection` that is exported in `__all__` and misleads anyone reading that file for how
negotiation works.

---

## 7. What to tell a stakeholder

The honest position, stated plainly: **the negotiation agent negotiates one variable — price —
using a fixed three-round percentage table, and mentions the other variables in prose.** It does not
model the counterparty, compute a walk-away, trade issues against each other, or behave differently
for a large deal than a small one.

What it does have is a real deterministic pricing heuristic under formula-registry contract, a
genuinely fail-closed approval gate on dispatch, and an honest refusal to fabricate a supplier cost
floor. That is a defensible foundation, and the decisions it produces are auditable in kind even
where the numbers are not yet evidenced.

The gap between that and "best-in-class" is not one large build. It is Phase 0 — four wiring tasks
against code already written — followed by six correctness fixes. Only after those does the
capability work in Phase 2 become worth starting.

---

*Line numbers are as at `1b18257` and will drift — re-grep the identifier, not the line.*
