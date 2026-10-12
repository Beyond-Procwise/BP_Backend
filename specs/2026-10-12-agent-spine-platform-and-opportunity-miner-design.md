# Agent spine platform, and the Opportunity Miner as its first agent: design

**Date:** 2026-10-12
**Status:** Proposed. Awaiting the product owner's review. No code has been changed.
**Basis:** two assessments run on 2026-10-12 against `BP_Backend` (branch `Development`), the gateway, the UI, the predecessor repository `procwise_agents`, and read-only queries on `bp_testdb` and `bp_sqldb`. File and line references were correct on that date; re-check before relying on them.
**Read alongside:** `specs/2026-10-12-orchestrator-routing-and-agent-governance-brief.md` (another session's brief, also proposed). Section 3 says how the two fit.

---

## 1. What this is for

Every agent in the product should work the same way: a short list of things it must assess and report on every run, which a person owns and the agent cannot change, with freedom to investigate inside that list. Actions that cannot be undone are refused unless that report is complete and checked.

This document designs the shared parts once, and puts the Opportunity Miner on them first.

**Success looks like:** a user opening an opportunity sees a figure that was recomputed and challenged before it reached them, the rows it came from, and what the miner could not assess and why. Adding the next agent means writing its list, its tools and its test cases, not rebuilding the platform.

## 2. Decisions already made by the product owner (2026-10-12)

| Decision | Ruling |
|---|---|
| Who owns the fixed list | Category Manager or SuperAdmin |
| Agent order | Opportunity Miner, discrepancy triage, negotiation, quote comparison and award, contract review and renewals, sell-side quotes, sourcing (placeholder only) |
| Budget, ESG, supplier performance | Reported as "insufficient evidence". That status is carried into the negotiation report. |
| Opportunity validation | Needs its own checker, with a different angle from the negotiation plan reviewer |
| Duplicate-invoice detector | Switched on once the validator is live, not before |
| Accepting a supplier's offer | Always needs a person. The rule is a policy, changeable through the learning loop. |
| Negotiation rounds | Multi-round stays on. The number of rounds is to be learned, so the cap must be a policy value. |
| Authority | Whatever the user's policy sets. The agent may ask a supplier for options. |
| Data | Dummy data now; real production documents later, volume unknown |

## 3. How this fits the orchestrator brief

The two documents cover different layers and do not overlap in code.

| Question | Answered by |
|---|---|
| May this agent run at all, and by which route? | The orchestrator brief: WP1 (one policy check on every agent run) and WP2 (`determine_path`) |
| What must the agent assess on every run, and was it assessed? | This design: the spine and the run report |
| May this particular action go ahead? | This design: the action gate, which asks for a complete, verified report |
| How should negotiation use the model inside coded limits? | The orchestrator brief WP6, built on this design's platform (stage 3 below) |

This design does not depend on WP1 or WP2 landing first. If they land, the run report gains two fields (the route taken and the policy verdict) and nothing else changes.

## 4. Programme

| Stage | What | Covered by |
|---|---|---|
| 0 | Four negotiation control holes | Section 11 of this document |
| 1 | Shared platform, Opportunity Miner, validator | Sections 5 to 10 of this document |
| 2 | Discrepancy triage on the platform | Its own design, after stage 1 |
| 3 | Negotiation on the platform, including the gaps against the original requirements | Its own design, after stage 2 |
| 4+ | Quote comparison and award; contracts and renewals; sell-side; sourcing | Later |

Stage 1 is one coherent slice and is planned in two parts: A (sections 5, 6, 7) and B (sections 8, 9, 10).

---

## 5. The spine store (shared)

**What it is.** One governed row per agent holding its fixed list and its assertions.

**Where.** `proc.bp_policy`, the same place email rule sets live (`src/services/draft_assurance/family.py`). A spine is a policy row named `AgentSpine_<agent_key>`, with `policy_type = 'agent_spine'`. The row's `version` column is the spine version.

**Shape of `policy_details.rules`:**

```json
{
  "agent_key": "opportunity_miner",
  "mode": "shadow",
  "items": [
    {"key": "price_position", "label": "Price position",
     "detectors": ["price_variance_check", "volume_consolidation_check", "duplicate_supplier_check"]}
  ],
  "assertions": [
    {"key": "figures_from_formulas", "label": "Every figure comes from a registered formula"}
  ]
}
```

**Rules.**
- A missing or malformed spine raises. The agent does not run on a default list. This copies `load_family`, which was written for the same reason.
- `mode` is `shadow` (record only) or `enforce` (the gate may refuse). It starts as `shadow`.
- Only a Category Manager or SuperAdmin may change a spine row. This is a new policy row, `AgentSpineAuthorityPolicy`, read through the existing endpoint gate (`src/api/endpoint_gate.py`). **To confirm during build:** that "Category Manager" exists in the product's role mapping; the comparable rule today (`PlaybookAuthorityPolicy`) is Admin-only.
- Every change is a new version. Reports record the version they ran under.

**Code.** New package `src/services/agent_spine/`: `spine.py` (load, parse, raise), no other responsibilities.

## 6. The run report (shared)

**What it is.** One record per agent run, in one shape for every agent.

**Where.** A new append-only table, `proc.bp_agent_run_report`. Updates and deletes are blocked by trigger, the way `proc.bp_agent_actions` already is (`deploy/sql/2026-09-16_bp_agent_actions_immutable.sql`).

| Column | Meaning |
|---|---|
| `report_id` | Key |
| `agent_key`, `spine_policy_id`, `spine_version` | Which agent, under which version of its list |
| `subject_type`, `subject_id` | What the run was about (a deal, a supplier, the whole corpus) |
| `trigger`, `initiated_by` | What started it, and who |
| `brief` | What was asked for. For the miner: scope and the governed minimum impact. |
| `items` | One entry per spine item: status, reason, counts, evidence references |
| `assertions` | One entry per assertion: pass, fail or could-not-verify, with detail |
| `strategy` | The case-specific approach. Empty for the miner. |
| `open_questions` | What is unknown or assumed |
| `proposed_additions` | Anything the agent suggests adding to the spine. Kept separate; never applied. |
| `decision`, `decision_reason` | `proceed`, `revise` or `escalate` |
| `created_at` | When |

**Item status** is one of `investigated`, `not_applicable`, `insufficient_evidence`, always with a reason. `investigated` carries how many findings were raised and how many survived validation.

**Rules.**
- A report with any spine item missing is invalid and cannot be written as `proceed`.
- The report is the only source the gate reads. Nothing in memory counts.

**Code.** `src/services/agent_spine/report.py` (build, validate against the spine, write) and one migration under `deploy/sql/`, applied to both databases.

## 7. The Opportunity Miner made honest

The miner stays what it is: detectors written as code over documents, switched on by `proc.bp_rule` rows. It is not turned into a model-driven agent, because its value is arithmetic and a model adds risk there without adding judgement. The changes are inside `src/agents/opportunity_miner_agent.py` and `src/services/opportunity_store.py`.

### 7.1 Spine for the miner

| Item | Detectors behind it | Expected status today |
|---|---|---|
| Price position | Price variance, volume consolidation, duplicate supplier | Investigated |
| Billing integrity | Invoice overbilling, duplicate invoice, inflation pass-through | Investigated |
| Contract | Expiry, expiry buckets, unused contract value, maverick spend | Investigated once repointed (7.4) |
| Supplier | Risk, performance | Insufficient evidence: no performance source |
| Budget | Category overspend | Insufficient evidence: no budgets supplied |
| ESG | ESG switch | Insufficient evidence: no ESG inputs |
| Benchmark and index | Unit price against a real benchmark; contract price drift against an index | Insufficient evidence: no benchmark or index tables |
| Payment and logistics | Early-payment discount missed; logistics cost outliers | Insufficient evidence: no discount-terms or shipment data |

The last two items come from the original requirements (`procwise_agents/business_requirements/Agents/Opportunity Finder Agent/Opportunity_Agent_MVP_Engineering_Requirements_v1.4.docx`), which asked for seven detectors of which four were never built. They are listed so the report says so on every run. Building them is not part of this stage.

### 7.2 Assertions for the miner

1. Every figure comes from a registered formula.
2. Every finding names source rows that exist.
3. The currency label is true.
4. Nothing is shown below the governed minimum impact.
5. No finding is retired by a detector that could not read its data.
6. No finding a person has progressed is changed.

### 7.3 Figures

- Each detector's figure moves to a registered formula in `src/services/formulas/definitions/opportunity.py`, with golden test values. `opportunity.price_variance_impact` is already registered and unused; the detectors are rewired onto it and its siblings.
- Each finding is labelled as a **saving** or a **value at stake**, using the existing `value_basis` column. Contract expiry (whole contract value), maverick spend (whole PO total) and duplicate supplier (total item spend) are values at stake. They are never summed into "potential savings".
- The contract expiry buckets detector stops writing an unconverted contract value into a GBP column (`opportunity_miner_agent.py:6203`).

### 7.4 Behaviour removed

| Today | After |
|---|---|
| Missing quantity becomes 1; missing unit price becomes the line value (`:5440-5445`) | The row is excluded and counted under "insufficient evidence" |
| A failed table read becomes an empty table (`:2353-2366`) | The item is "insufficient evidence: could not read `<table>`", and nothing is retired |
| At most two findings per detector per run (`:961-970`) | No cap |
| If nothing clears the minimum, the best finding is shown anyway (`:1964-1980`) | Nothing is shown; the report says none cleared the minimum |
| Thresholds default to 0.0 in code | A missing threshold makes the detector "insufficient evidence" |
| Older contract detectors read `proc.contracts`, which does not exist in either database | They read `proc.bp_contract_master` (3,051 rows in each database) |
| Prompt text can change `workflow` and `min_financial_impact` (`:728-769`) | Removed; only policy rows set them |

### 7.5 Identity and state

- Findings are keyed by `opportunity_ref_id`, the stable reference, everywhere. Today the table's primary key is a per-run counter, so a new finding can collide with an old row and the failed save is swallowed (`:7558-7568`). The migration changes the key and the save stops swallowing errors.
- Production's `bp_opportunity` lacks the `retired_at` column that test has. The migration adds it.
- The per-run record of which detectors ran, were skipped or found nothing is written into the run report for every path, not only the API path.

### 7.6 The adaptive part

Small on purpose in this stage. For each finding the miner records, in the report's item entry, what evidence would confirm it and what would disprove it, and for "insufficient evidence" what to ask for and from whom. No model chooses detectors.

## 8. The validator

**Angle.** The negotiation reviewer (stage 3) checks a plan: were all the levers examined, is each claim sourced. The opportunity validator tries to disprove one finding. They share the harness in 8.3 and nothing else.

### 8.1 Checks

| # | Check | How | Outcome when it cannot be done |
|---|---|---|---|
| 1 | Recompute the figure from the source rows, without the miner's working | Code, registered formulas | Could not assess |
| 2 | Like-for-like: unit, quantity basis, currency, date | Code | Could not assess |
| 3 | A contract term explains the variance | Code over captured commercial terms and obligations | Recorded as not checked; does not fail the finding |
| 4 | Still open: not rejected, realised or already in negotiation | Code | Could not assess |
| 5 | Same money counted by another detector | Code, comparing source rows across findings | n/a |
| 6 | Source rows exist and carry the numbers used | Code | Invalid |
| 7 | Addressable in practice | The existing critic's model call, given the finding and evidence only | Could not assess |

Checks 1 to 6 run first. Check 7 runs only if they pass. The eleven `critic.*` formulas already written and never executed (`src/agents/opportunity_critic_agent.py:155-163`) are executed here; the arithmetic leaves the prompt.

### 8.2 Verdicts

The existing vocabulary is kept: `VALID`, `VALID_REFRAMED` (valid at a different figure), `INVALID`, `DUPLICATE`, `UNASSESSED`. A recomputed figure that differs from the miner's by more than a governed tolerance makes the finding `VALID_REFRAMED` at the recomputed figure, or `INVALID` if the recomputed saving is zero or negative.

### 8.3 Storage and harness

- New table `proc.bp_finding_validation`: the finding's stable reference, the report it belongs to, validator version, each check's outcome and detail, the miner's figure, the recomputed figure, the verdict. Append-only.
- Shared harness `src/services/agent_spine/review.py`: runs a list of checks, guarantees the reviewer receives only the subject and its evidence, records every check. The opportunity checks live in `src/services/opportunity_validation/`.
- The validator runs inside the mining run, immediately after detection, for every finding. `scripts/critic_run.py` is retired.
- If the model is unavailable, checks 1 to 6 still decide; check 7 is recorded as not run and the verdict is `UNASSESSED`. That is a state a person sees, not a silent pass.

## 9. The gate at the action (shared)

**What it is.** One function, `require_report(agent_key, subject, action)`, in `src/services/agent_spine/gate.py`. It reads the latest report and validation for the subject and answers allow, refuse or escalate, and writes its answer to `proc.bp_agent_actions`.

**It refuses when** there is no report, a spine item is missing, an assertion failed or could not be verified, or the verdict is not `VALID` or `VALID_REFRAMED`. `UNASSESSED` escalates to a person.

**Where it is called for the miner:**

| Action | Call site |
|---|---|
| Counting a finding in "value found" and listing it | `src/services/opportunity_dashboard.py`, `src/api/routers/opportunities.py`, the gateway's opportunity queries |
| Sending a supplier query about a finding | `src/api/routers/value_summary.py:88-104` (the existing dispatch guard still applies) |
| Approving a playbook proposal raised on a finding | `src/api/routers/playbooks.py:235-275` |
| Moving a finding into negotiation | `src/services/deal_lifecycle.py:15-19` |
| Recording realised value | `src/api/routers/value_ledger.py:98-102` |

**Recording realised value** gains two further rules: the finding must have been validated, and an amount above the validated figure needs a stated reason.

**Shadow first.** While the spine's `mode` is `shadow`, the gate records what it would have refused and allows the action. It moves to `enforce` per agent once the proof in section 10 passes. Counting and listing are the exception: unvalidated findings are labelled "not yet checked" from the first day.

## 10. Proof

### 10.1 Scenario set

A fixed set of dummy findings with known right answers, run automatically on every change, built like `evals/email/` (its own throwaway database, `evals/email/db.py`). At least:

- A genuine price saving. Expected: valid.
- The same money raised by two detectors. Expected: one valid, one duplicate.
- A unit mismatch (each against box). Expected: invalid.
- A contract clause that explains the price rise. Expected: invalid, with the clause cited.
- A finding already realised. Expected: invalid, not open.
- A finding whose source rows do not carry the stated numbers. Expected: invalid.
- A missing quantity. Expected: item reported as insufficient evidence, no finding.
- A duplicate invoice pair that is a legitimate repeat order. Expected: invalid.
- An unreadable table. Expected: nothing retired, item insufficient evidence.
- The same run twice. Expected: identical report apart from timestamps.

Scored on: correct verdict, correct item status, no unsupported figure, no change between runs.

### 10.2 Where dummy data lives

**Recommendation:** dummy rows live only in the scenario set's own database, never in `bp_sqldb` business tables. Production is proven by running the platform in `shadow` mode over whatever real rows exist. This keeps invented invoices out of real totals, which matters because production already has 51 real invoices and a known canonical spend figure.

This is open decision 1 in section 13.

### 10.3 Switch-on order

1. Platform and miner changes on `bp_testdb`, spine in `shadow`.
2. Scenario set passing in CI.
3. Validator live on `bp_testdb`; re-validate the 308 existing findings and report the result.
4. Same migrations and policy rows applied to `bp_sqldb`, spine in `shadow`.
5. Duplicate-invoice detector switched on (`duplicate_invoice_detector_enabled` in `AutonomousOperationPolicy`), both databases.
6. Spine moved to `enforce`, test first, then production.

Each step is reported with its evidence before the next begins.

### 10.4 Screen

The opportunity view shows each spine item's status and reason, the source rows behind a figure, the saving-or-value-at-stake label, and the validation verdict with its checks. The `source_records` and `calculation_details` already stored are returned by the API, which they are not today. Built last, on the existing drawer and panel styles.

---

## 11. Stage 0: four negotiation control holes

Small, independent of the rest, and worth closing before any negotiation work.

| Hole | Evidence | Fix |
|---|---|---|
| A caller-supplied `authority` block is trusted outside three workflows | `src/agents/negotiation/analysis.py:194-198`; the orchestrator overwrites it only for the workflows in `_AUTHORITY_WORKFLOWS` (`src/orchestration/orchestrator.py:419-423`) | The agent always resolves its mandate from policy itself and ignores any block in the request unless it carries a marker only the orchestrator sets |
| The quantity in the value-limit check comes from the request | `src/agents/negotiation_agent.py:9744` | Quantity is read from the quote lines on record; if none is on record the counter is withheld, as it already is when quantity is missing |
| The single-supplier and plain-batch paths have no round hold | `_resolve_hitl_decision` has one caller (`negotiation_agent.py:2922`) | The hold is applied on every path that produces a counter |
| A held round cannot be released | Held tasks are written (`negotiation_agent.py:2206`) and read by nothing | A release path that re-checks the approval and the mandate before drafting |

Each gets a failing test first, and a deliberate break to prove the test goes red.

## 12. Not in this design

- Building the four detectors from the original requirements that never existed (they are reported as insufficient evidence).
- Discrepancy triage and negotiation on the platform.
- Versioned playbooks with owners and usage data.
- The review queue for agent-proposed additions (they are recorded in the report only).
- Routing and the single run-time policy check (the orchestrator brief).
- Loading real production documents.

## 13. Open decisions for the product owner

1. **Dummy data in production.** Recommended: none in production tables; use the scenario set's own database and run production in shadow (10.2). The alternative is tagged dummy rows inside `bp_sqldb`.
2. **The four never-built detectors.** Recommended: list them as insufficient evidence now and decide later whether to build them, since each needs data the product does not hold (benchmarks, indices, discount terms, shipments).
3. **Existing findings.** Recommended: re-validate the 308 on `bp_testdb` and show the result before anything is hidden or retired.
4. **Category Manager role.** Confirm it should be created if the role mapping does not have it.

## 14. Risks

| Risk | Handling |
|---|---|
| Most existing findings fail validation, and "value found" drops sharply | Expected. Shown as a before-and-after report at step 3 of 10.3 before any screen changes. |
| Changing the opportunity table's key breaks references from value outcomes and playbook proposals | The migration maps both; it runs on a copy first and the row counts are compared. |
| Two sessions work in this checkout and on these databases | New files and new tables only where possible; migrations follow the existing reviewed-script pattern and are applied to both databases deliberately. |
| The orchestrator brief changes the run path under this work | The gate and report are called from action sites, not from the run path, so the two can land in either order. |
| Enforcing the gate stops something that works today | Shadow first; enforcement is per agent and is a one-row change to reverse. |
