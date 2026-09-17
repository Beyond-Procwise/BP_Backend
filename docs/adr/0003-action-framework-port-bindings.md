# ADR 0003 — Action Framework: Phase 0 port bindings (discovery report)

- **Status:** Proposed — awaiting review. No framework code written.
- **Date:** 2026-09-15
- **Brief:** "Build Prompt — ProcureIQ Action Framework v1 (Revision 2)"
- **Numbering note:** the brief asks for `0001-...`; `0001` and `0002` already exist locally.
- **Location note:** `/docs/*` is gitignored (`.gitignore:12`); this file is tracked beside 0001
  and 0002 by an explicit `git add -f`, as the `.gitignore` header prescribes. Decision D10 below.

All citations were checked against the working tree on `Development` (`a789f1e`). Database
facts were checked against **both** `bp_testdb` and `bp_sqldb` on 2026-09-15.

---

## 1. Headline

**The brief describes a platform that is only partly this one.** Of the eight ports, two bind
cleanly, two bind with gaps, and **four have no engine behind them at all**.

§2.4 of the brief says that when an engine cannot answer, the answer is a proposed engine change
or a narrowed framework, never a local implementation. For four ports that rule leaves only
"narrow" or "build the engine first". Building the engine first is a larger programme than the
framework itself.

| Port | Binds to | Verdict |
|---|---|---|
| `AuditPort` | `agent_actions.record_action_or_fail` → `proc.bp_agent_actions` | **Binds, with 3 gaps** (not immutable, no returned ref, no read API) |
| `AgentToolPort` | `orchestration/agentnick_control.build_tools` + `tool_runtime.Tool` | **Binds, with 1 gap** (no registry; no authorisation in the tool loop) |
| `AuthorityPort` | `guardrail.authorize` (role + effect policies) | **Partial.** No approval matrix, no `NEEDS_APPROVAL`, no hard-block list |
| `PolicyPort` (autonomy) | `guardrail.authorize` / `proc.bp_policy` | **Partial.** No autonomy concept anywhere; no bitemporality; no write path |
| `RulesPort` | — | **No engine.** No addressable predicate-rule catalogue |
| `LifecyclePort` | — | **No engine.** Per-object ad-hoc status writes; one real guard (sell-side quotes) |
| `EntitlementPort` | — | **No engine.** The code states it (`analytics/next_steps.py:155`) |
| `PreferencePort` | — | **No engine.** No shared scorer, no observation feed, no presented-set record |

## 2. Brief premises that are false in this codebase

Each one changes the design, not just a name.

| Brief assumes | Reality | Evidence |
|---|---|---|
| Postgres `FORCE ROW LEVEL SECURITY`, tenant context propagation | **No RLS on any table in either DB.** No `set_config`/`current_setting`. `tenant_id` columns are placeholders defaulted to one constant. The comment says a policy "would be theatre". | `pg_class` query (0 rows, both DBs); `deploy/sql/2026-08-07_commercial_fact.sql:17-20`; `src/api/auth.py:18-21` |
| Alembic migrations | Hand-applied SQL in `deploy/sql/` (112 files, `_rollback.sql` pairs), `psql -v ON_ERROR_STOP=1 -f` | `deploy/sql/*` headers |
| React/TypeScript front end | React 19 **JavaScript/JSX**, 0 `.ts/.tsx`. SpendIQ is a 17,388-line `engine.js` | `beyond_procwise_ui/package.json`, `src/modules/SpendIQ/engine.js` |
| Package `procwise/actions/` | `procwise` is a shell script; code lives under `src/` | `file procwise` |
| Approval matrix (281 rows, 32 categories, 7 sheets, hard-block list) | **Does not exist.** Code says so. One £10k `ApprovalThresholdPolicy` + 10 role-based authority policies | `src/services/rga/pipeline.py:206`; `deploy/sql/2026-07-13_bp_approval.sql:61`; `2026-09-09_action_authority_policies.sql:57-87` |
| Bitemporal licence entitlement service | **Does not exist.** Only `AllowAll` / `AllowList` stand-ins | `src/services/analytics/next_steps.py:146-175` |
| GPSS as canonical object vocabulary | **Does not exist**; recorded as blocker B3 | `docs/remediation/00_seam_map.md:16,281-292` |
| Profile → Capability → Composition architecture | **Does not exist.** Roles are Viewer/Buyer/Approver/Admin in `bp_policy`; "personas" are strings that reorder next steps | `deploy/sql/2026-08-06_guardrail_enforcement.sql:33-41`; `next_steps.py:~136` |
| Findings lifecycle CheckDefinition → Observation → Finding → FindingState | **Does not exist.** "Finding" means two things (opportunity row / extraction discrepancy row) | §3.5 |
| Tri-state confidence incl. INFERRED / MEASURED | Three separate definitions; INFERRED/MEASURED absent; values are set at creation and never transition | `analytics/models.py:70-73`; `formulas/definitions/critic.py:405-440`; `formulas/unassessed.py:85-99` |
| Shared log-odds scoring service with registerable, calibrated, versioned profiles and `observe()` | **Does not exist as a service.** One document-link scorer with in-memory profiles; no support counts, no version, no observations | §3.7 |
| Guided-buying `SlateDecision` | **Does not exist.** No "options presented, one chosen" record anywhere | §3.8 |
| C0–C7 guided-buying channel resolver; 105 sub-process definitions | **Neither exists.** The only sub-process ladder is sell-side (3 phases / 9 sub-processes) | `src/services/sell_side/ladder.py:11-23` |
| Idempotency-key convention on mutating endpoints | **None.** Only natural-key `ON CONFLICT` | whole repo |
| Action Centre sources affordances per level | Surface → Batch → Type is **one of three switchable groupings**. Every node offers the same accept/dismiss; no role or entitlement check; dismiss is local-only; audit lives in `window.__SIQ_AUDIT__` | `actionTree.js:131-166,433`; `engine.js:3415-3440` |

**Important conflict with §4.3.** An action vocabulary already exists:
`src/services/actions.py:28` holds 45 closed names shaped **`domain.verb`** (`finding.resolve`,
`email.send`, `spend.approve`), each with a class that drives irreversibility. Every guardrail
policy matches on those names through `applies_to`. The brief's **`verb.object`** key
(`dismiss.finding`) would be a second spelling of the same idea. The module header warns exactly
against that ("a policy written against `email_send` simply never matches `email.send`").
Decision D2.

---

## 3. Per-port detail

### 3.1 `AuditPort` → `proc.bp_agent_actions`

- **Binds to** `record_action_or_fail(*, phase, action_type, conn=None, **fields)`
  (`src/services/agent_actions.py:133`). It raises `AuditWriteError`.
  - With `conn=` it writes inside a SAVEPOINT on the caller's transaction (`:75-99`). That meets
    the brief's "audit in the same transaction as the state write".
  - The caller must use `sell_side._db.transactional_conn()`. `db.get_conn()` is autocommit
    (`src/services/db.py:1141`), so a rollback on it does nothing.
- **Record shape:** `deal_id, document_id, doc_pk, doc_type, process_monitor_id, trace_id, phase*,
  action_type*, agent, field_name, status, summary, details jsonb, confidence, pipeline_version`,
  plus `action_id`, `created_at`.
- **Unknown/malformed:** `record_action` (the best-effort variant) swallows errors. The framework
  must only ever use `record_action_or_fail`.
- **Gaps:**
  1. ~~**Not immutable.**~~ **DONE 2026-09-16, commit `c17c1a3`** —
     `deploy/sql/2026-09-16_bp_agent_actions_immutable.sql`, applied to both DBs.
     The proposed REVOKE was **abandoned on discovery**: the app role `procwisedb123` is the
     table's OWNER in both databases, and an owner re-grants to itself at will, so a privilege
     change would have looked like a control and stopped nothing. The control is a `BEFORE
     UPDATE OR DELETE` row trigger that RAISEs, plus a **separate `BEFORE TRUNCATE` statement
     trigger** — Postgres does not route TRUNCATE through a row-level trigger, so without it one
     statement could still empty all 54,769 rows. `truncate_for_fresh_extraction.sql` listed this
     table and no longer does. INSERT stays open by design.
     Still open from this port: **no returned `AuditRef`** (gap 2) and **no read API** (gap 3).
     The missing committed DDL is also still missing — the live shape was captured and is
     identical across both DBs (17 columns, 7 indexes), but recording it was kept out of this
     change to keep the invariant reviewable.
  2. **No returned reference.** The function returns `None`. *Proposed:* `RETURNING action_id`,
     returned to the caller as the `AuditRef`.
  3. **No read API** for `GET /v1/action-requests/{id}/audit`. *Proposed:* a read function keyed by
     `trace_id`.
- **Latency:** one INSERT; safe to call per transition.
- **Caveat:** the endpoint gate writes `phase="authorize", status="allowed"` for a shadowed denial
  (`api/endpoint_gate.py:75`, `guardrail.py:488`). An audit reader must check
  `details.evidence.shadowed`.

### 3.2 `AgentToolPort` → `agentnick_control.build_tools`

- **Shape:** `Tool(name, description, parameters, handler)` (`src/services/tool_runtime.py:54-60`),
  sent to Ollama in the standard function format.
  - `build_tools` (`agentnick_control.py:259`) assembles `_agent_tools`, `_governance_tools` and
    `_corpus_tools` each call.
  - Agents become `run_<slug>` tools via `AutoRegistry.tool_schemas()` (`auto_registry.py:300-350`).
- **Gap: no registry and no authorisation in the loop.** `tool_runtime.py:342,449` call
  `tool.handler(**args)` directly; the only guard is a name exclusion list
  (`base_agent.py:127`). This does not block the framework: every `action.*` tool handler would
  create an ActionRequest, and that path runs the checks. *Proposed engine change:* a registration
  seam in `build_tools` for a tool provider, not a second registry.
- **Hand-written state-changing tools that would overlap the catalogue:** `run_email_dispatch`,
  `run_approvals` (inserts `bp_approval`), `run_negotiation`. None creates POs or dismisses
  findings; those are HTTP-only.
- **Note:** Ollama ignores `oneOf`/discriminators in `format=` schemas. Generated `params_schema`
  tool inputs must be union-free.

### 3.3 `AuthorityPort` → `guardrail.authorize`

- **Signature:** `authorize(action, action_class, principal, context, policy_engine) -> Decision`
  (`src/services/guardrail.py:396`).
- **Return shape:** `Decision(allowed, reason, policy_id, policy_name, policy_version, evidence,
  resolution)` (`:49`).
- **Order:** role cap, then any `effect:"deny"` wins, then a permit must be an explicit
  `effect:"allow"`.
- **Fail behaviour: closed.** An unknown role denies (`:215`); any exception denies (`:312`); an
  uncovered irreversible class returns `unresolved` + `allowed=False` and opens a `bp_decision`
  escalation (`decision_engine.py:1709`).
- **Mapping to the brief's four outcomes:**

  | Brief | guardrail | Notes |
  |---|---|---|
  | `ALLOWED` | `allowed=True`, `resolution=resolved`, **not** `evidence.shadowed` | |
  | `BLOCKED` | `allowed=False`, `resolved` | |
  | `NEEDS_APPROVAL` | *nearest:* `allowed=False`, `unresolved` (goes to a person) | Not the same thing: "no rule settled this" ≠ "a rule says an approver must sign". Gap. |
  | `UNEVALUABLE` | exception → deny | |
  | `hard_block` | *nearest:* non-shadowable actions `email.send`, `approval.email` (`:337`) | No hard-block list exists. Gap. |
  | allowed-but-shadowed | `allowed=True`, `evidence.shadowed=True` | **No brief equivalent.** Decision D6. |

- **`matrix_row_ref`:** `(policy_id, policy_version)` is present. Reproducibility is limited
  because a version bump overwrites the row in place (no history table, §3.4).
- **Preview vs at-commit:** no distinction exists. Calling the same function at both edges is
  legitimate, because the brief only requires a re-check at commit.
- **Batch:** none; sync, one action per call. Policies are loaded through the `PolicyEngine` cache.
  **I did not measure latency.** It needs measuring before Phase 1 decides on per-render calls.
- **Engine change needed for `NEEDS_APPROVAL`:** a policy `effect:"require_approval"` (with
  `required_role` of the approver) returned as a third resolution. This is new policy-engine
  semantics and needs your ruling.

### 3.4 `PolicyPort` (autonomy) → `proc.bp_policy` / guardrail

- **Storage:** `proc.bp_policy` (`deploy/sql/2026-06-08_create_bp_prompt_bp_policy.sql:24-37`).
  - Active rows: 43 in both DBs.
  - Addressed by `details.policy_identifier` via `PolicyEngine.get_policy(slug)`
    (`policy_engine.py:348`), or by `details.applies_to` via `policies_for_action`.
- **Gaps (each needs a ruling; none can be patched in the framework):**
  1. **No autonomy concept.** A0–A4 appears nowhere. *Proposed:* an optional `rules.autonomy` on
     authority policies, keyed by actor class, with guardrail returning it alongside the decision.
     Absent means A0.
  2. **No bitemporality.** Only an integer `version` bumped in place: no `valid_from/valid_to`, no
     epoch, no history. `as_of` resolution and the brief's `policy_epoch` are therefore not
     reproducible. *Proposed:* a `bp_policy_history` append table written on version bump, or
     narrow `as_of` to "now".
  3. **No policy write path.** No endpoint writes policies (`policy.write` is only a name in
     `actions.py:76`); changes are SQL migrations. So the "reject `govern` at A3/A4 at policy-write
     time" hook has nowhere to live. `govern` actions whose commit "calls the policy engine's write
     API" have no API to call. This is its own programme.
  4. **No change signal.** No trigger/NOTIFY. rbac uses a 60s TTL (`rbac.py:27`);
     `governed_limits.reset_cache()` has **no callers**, so limit values live for the process
     lifetime even after `/reload-policies`. The resolver cache invalidation "on policy write"
     cannot be done as briefed.
  5. **Fail-open corners remain:** `PolicyEngine` load failure returns `[]` (`policy_engine.py:210`).
     `guardrail` and `envelope.resolve_governance` are closed; `PolicyEngine` and `rbac._rules` are
     not. The adapter must go through the closed path only.

### 3.5 `LifecyclePort` → no engine

There is no lifecycle service and no transition table. State lives per object:

| Object | State | Illegal transitions refused? |
|---|---|---|
| Opportunity | `bp_opportunity.stage` | **No.** CHECK limits values; `set_stage` does a plain UPDATE, so `realised → identified` succeeds (`opportunity_store.py:144-156`) |
| Extraction discrepancy (the Action Centre "finding") | `bp_extraction_discrepancy.status` | No from-state check (`decision_engine.py:1289-1309`) |
| Deal | booleans `is_tracked`, `is_saved_reference` | No |
| Deal proposal | `bp_deal_proposal.status` | **Yes** — acts only from `proposed`, 409 if stale |
| Approval | append rows, newer hides older | n/a |
| Negotiation | `negotiation_session_state.status` (runtime-created varchar) | No |
| Sales quote | `bp_sales_quote.status` | **Yes** — `_transition` locks the row and checks the from-state (`sell_side/quotes.py:171-192`). The one model to copy |

- **Confirmed defect found during discovery — FIXED 2026-09-15, commit `db746c2`.** The Action
  Centre's **flag / hold / escalate / assign / investigate / query** actions wrote `status` values
  `flagged`, `on_hold`, `escalated`; `resolution_action`'s CHECK also allows only `apply_value,
  keep_null, dismiss`, so `confirm`, `approve` and `reject` failed the same way. Probed per action
  against **both** DBs: **9 of the 11 actions were rejected**; only `apply_value` and `dismiss`
  ever worked. The user saw HTTP 400 and the toast "Could not apply: could not update the
  finding" — the buttons visibly failed rather than silently no-opping. Severity was higher than
  it first appears: `flag` is the **primary** option for fifteen issue types in the UI's
  `DECISION_TYPES`, and `reportResolveAllInCase` posts each finding's primary verb, so bulk
  "Resolve all" failed on every row of those categories.
  **The fix changed no constraint.** It adopts the mapping the Node gateway already writes to this
  same table (`spendiq.service.ts RESOLUTION_STATUS`), so the two writers now agree:
  apply_value/confirm/approve → `resolved`; dismiss/reject → `ignored`; flag/hold/escalate/… →
  `open`. `resolution_action` records what happened to the *raw extracted value* —
  `extraction/promotion.py` switches on exactly `apply_value`/`keep_null` — so a verb with no
  honest fit records NULL, and the human's actual verb continues to be recorded on
  `proc.bp_decision`. Re-opening now clears `resolved_at`, a path that was unreachable while
  `flag` always failed.
- **Two further defects found in passing, both UNFIXED and outside this change:**
  1. **`bp_sqldb` is missing `ix_bp_extraction_discrepancy_open_key`** (present in `bp_testdb`), so
     `deploy/sql/2026-07-30_discrepancy_dedup.sql` was never applied there. `write_discrepancies`'
     `ON CONFLICT … WHERE` upsert therefore **fails on bp_sqldb** — verified: "there is no unique
     or exclusion constraint matching the ON CONFLICT specification". Duplicate findings stack up
     there, which is the very problem that migration existed to stop.
  2. **A flagged finding is not visually distinct.** Neither `_fetch_finding`, `analysis_findings`,
     `value_summary_service` nor the gateway's `getDiscrepancies` selects `resolution_action`, so
     a flagged finding renders exactly like an untouched one. The gateway shares this limitation.
     Surfacing it is a small read-path change in both repos.
- **Proposed engine change (the prerequisite for `LifecyclePort`):** give each owning store a
  `_transition(frm, to)` guard modelled on `sell_side/quotes.py:171`, plus a declared transition
  table and a `can_apply` query. Start with the two object types the seed catalogue needs first
  (discrepancy-finding, opportunity). `lifecycle_effect_ref` then resolves to a named edge in that
  table.

### 3.6 `RulesPort` → no engine

- `bp_policy` rows are configuration JSON, not evaluable predicates over a subject.
- `guardrail` evaluates "may this principal do this action", not "is this precondition true of
  this object".
- **Closest existing home:** the formula registry (`src/services/formulas/`). It already has named,
  semver-versioned, source-hashed definitions with golden vectors and `evaluate` / `evaluate_many`
  batch entry points (`evaluate.py:151,236`). It also has an UNASSESSED sentinel that maps
  naturally onto `UNEVALUABLE`.
- *Proposed engine change:* a predicate kind of formula returning TRUE/FALSE/UNASSESSED;
  `precondition_ref` = `formula_name@version`. The catalogue loader can then verify that each ref
  resolves. Needs your ruling, because it makes the formula registry the rules engine.

### 3.7 `PreferencePort` → no engine

- The one log-odds scorer is `linking_engine.score_link(source_row, target_row, profile_name, ...)`
  (`src/services/linking_engine.py:349`). It is a **document-pair relationship** scorer:
  hard-coded weights, in-memory `register_profile` (`:302`), no persistence, no version returned,
  no support counts, no decay, no `observe`.
- Clause conformance (`decision_engine.py`) and supplier ranking (`supplier_ranking_agent.py:177-217`,
  a weighted mean) do not use it.
- No ranking anywhere learns from past acceptance.
- **There is nothing to delegate to.** Using `score_link` for action choice would be forcing a
  relationship scorer into a preference model, which is the reimplementation the brief forbids
  under a different name.
- **Recommendation:** Phase 2 learned ranking is out of v1. The ranker ships catalogue order
  only, with the explanation "No history yet", which the brief already defines. Building a real
  choice-scoring service is a separate decision with its own owner.

### 3.8 Feedback store / `SlateDecision`

- There is no generic feedback store. About ten feature-specific tables exist: `bp_decision`,
  `bp_approval`, `bp_extraction_verdict`, `bp_extraction_hint_proposal`, `bp_supplier_review`,
  `bp_negotiation_advice`, `opportunity_feedback`, `rag_feedback`, `bp_policy_observation`,
  `bp_workflow_input_request`.
- None records the set of options that was presented.
- `bp_decision` stores only the chosen value. `SlateDecision` does not exist, so §5.3
  `action_presented_set` has nothing to extend. It is only needed if Phase 2 ranking goes ahead.

### 3.9 `EntitlementPort` → no engine

- `Entitlements.allows(action_id) -> bool` Protocol (`next_steps.py:146`), with `AllowAll`
  (documented as the stand-in) and `AllowList`. Passing `None` fails closed.
- **There is nothing to bind.** Binding `AllowAll` would be a permissive default, which the brief
  forbids. Binding a code-held `AllowList` would be inventing an entitlement service.
- **Recommendation:** narrow. Remove `capability_key` and step 4 of the resolver from v1, and state
  in the catalogue README that licensing is not enforced.

### 3.10 Identity and scheduling (supporting facts)

- **Identity:** `require_user(request) -> Optional[Principal]` (`src/api/auth.py:260`). The actor is
  `principal.subject` from the Cognito token, used 114× across 29 routers. It returns `None` only
  when auth is off.
- **Scheduler:** in-process `BackendScheduler` (`src/services/backend_scheduler.py`), polling every
  60s (`:79`), with `register_job(name, runner, interval)` (`:359`). A per-minute veto scheduler
  fits at exactly its resolution.

---

## 4. Recommendation

**Do not build the framework as briefed.** Half its ports would be stubs, and §2.3 would, correctly,
refuse to let it start.

Build it in this order, each step small and independently useful:

1. **Engine hardening that pays off regardless** (no framework code):
   - ~~Fix the discrepancy CHECK / status mismatch (§3.5).~~ **DONE — `db746c2`.**
   - Make `bp_agent_actions` immutable ~~and give it a returned ref~~ (§3.1).
     **Immutability DONE — `c17c1a3`.** The returned `AuditRef` is still outstanding.
   - ~~Add a from-state transition guard to the discrepancy and opportunity stores (§3.5).~~
     **DONE 2026-09-17** — `deploy/sql/2026-09-17_bp_lifecycle_transitions.sql`, applied to both
     DBs. Enforced by a trigger over one declared table, `proc.bp_lifecycle_transition`, not by a
     Python `_transition` per store: the Node gateway writes finding statuses too, so a Python
     guard would not bind it. `src/services/lifecycle.can_apply` reads the same table; a refusal
     is SQLSTATE `BP409`, surfaced as HTTP 409 / a clear Action Centre message.
   - **NEXT:** the returned `AuditRef` (§3.1 gap 2).
2. **Policy-engine extensions, on your rulings:**
   - `require_approval` effect (§3.3)
   - `rules.autonomy` (§3.4.1)
   - policy history (§3.4.2)
   - the predicate-formula kind (§3.6)
3. **A narrowed Action Framework v1:**
   - Catalogue metadata extends `services/actions.py` names (D2).
   - Resolver over Lifecycle + Rules + Authority/Autonomy.
   - ActionRequest envelope and state machine.
   - Audit through the hardened spine.
   - Agent tools through `build_tools`.
   - **Single-tenant, no entitlements, catalogue-order ranking only.**
4. **Deferred until an owner exists:** `PreferencePort` / learned ranking / governance loop
   (Phases 2–3 of the brief), the entitlement service, the policy write API, RLS/tenancy.

## 5. Decisions

**All rulings below were accepted as recommended on 2026-09-15.** D11 is done; the rest are
authorised but not started — §4's order stands, so the remaining step-1 engine hardening (the
returned `AuditRef`, lifecycle transition guards) comes next, then the step-2 policy extensions.

- **D1 — Scope.** Accept the narrowed v1 in §4, or treat §4 step 2 as the programme and defer the
  framework entirely?
- **D2 — Key shape.** Reuse `services/actions.py` `domain.verb` names (recommended: one
  vocabulary, and policies already match on it), or adopt `verb.object` and migrate every policy's
  `applies_to`?
- **D3 — Tenancy.** Confirm v1 is explicitly single-tenant, with no RLS and no tenant-isolation
  tests, since there is no tenant to isolate.
- **D4 — `NEEDS_APPROVAL`.** Approve a new `require_approval` policy effect in guardrail?
- **D5 — Autonomy.** Approve `rules.autonomy` on authority policies (absent means A0)?
- **D6 — Shadowed allows. ACCEPTED: treat as BLOCKED.** Shadow mode exists to avoid breaking
  behaviour that already ships while a policy is trialled. A brand-new framework action has no
  legacy behaviour to protect, so the fail-closed reading wins: the framework does not commit on
  an allow whose only basis is `evidence.shadowed`. The observation is still recorded, so the
  trial keeps learning. Revisit when enrolment lapses on 2026-10-09.
- **D7 — Rules home.** Make the formula registry the rules catalogue (predicate formulas)?
- **D8 — Entitlements and learned ranking.** Confirm both are out of v1.
- **D9 — Lifecycle. DONE (2026-09-17).** Transition guards for findings and opportunities, as a
  database trigger over `proc.bp_lifecycle_transition` (see §4 step 1).
- **D10 — ADR location. DONE (2026-09-16).** Kept in `docs/adr/`, force-added like 0001 and 0002;
  `.gitignore` is unchanged.
- **D11 — Discrepancy CHECK defect. DONE (commit `db746c2`).** Fixed by mapping the actions onto
  the statuses the table already allows, not by widening the CHECK. Widening `status` was
  rejected on evidence: eight queries filter `status = 'open'` (`benchmark_live.py:176`,
  `corpus_facts.py:223/229/240`, `price_outlier/detector.py:267`, `deal_link_proposals.py:121/185`,
  `duplicate_invoice_detector.py:373`), so a new `flagged` status would have dropped flagged
  findings out of open counts and out of the dedup probes, silently re-creating duplicates.
  Verified: 23 unit tests, 1 live-constraint test, 230 neighbouring tests green; all 11 actions
  accepted by both DBs; and the real `execute()` path run against live data (flag → stays open,
  confirm → resolved), with the probe row restored.

## 6. Places I could not verify

- Guardrail / PolicyEngine per-call latency: not measured.
- Whether any deployed environment differs from the two DBs checked.
- The fix could not be exercised end-to-end over HTTP: `_actor` (`decisions.py:46`) requires a
  token subject and `ask_auth` is off, so an unauthenticated call is refused by the auth gate
  before reaching the engine. It was proven through the real `execute()` path against the live
  database instead.
