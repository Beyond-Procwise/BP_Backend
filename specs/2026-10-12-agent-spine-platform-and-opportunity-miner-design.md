# Orchestrator checkpoints, the agent checklist and report, and the Opportunity Miner as the first agent: design

**Date:** 2026-10-12 (second version, rewritten the same evening after the product owner's review)
**Status:** Proposed. Not approved. No code has been changed.
**Basis:** two assessments run on 2026-10-12 against `BP_Backend` (branch `Development`), the gateway, the UI, the predecessor repository `procwise_agents`, and read-only queries on `bp_testdb` and `bp_sqldb`. File and line references were correct on that date; re-check before relying on them.
**Supersedes:** the first version of this file (commits `96b0f7b6` to `bd3e40cf`). That version stored the checklist as a policy and checked actions in a gate beside the Orchestrator. The product owner rejected both; section 3 records the rulings.
**Read alongside:** `specs/2026-10-12-orchestrator-routing-and-agent-governance-brief.md` (another session's brief, also proposed). Section 4 says how the two fit.

---

## 1. What this is for

Every agent in the product should work the same way:

- The **Orchestrator** applies policy. Agents report back to it at key points, and it checks the policies that matter at that point before the agent carries on.
- Each agent has a short **checklist** of things it must assess and report on every run. A person owns it; the agent cannot change it.
- Each run produces one **report** in one shape.
- A separate **check** challenges the agent's output before anything is done with it.

This document designs those shared parts once and puts the Opportunity Miner on them first.

**Success looks like:** a user opening an opportunity sees a figure that was recomputed and challenged before it reached them, the rows it came from, what could not be assessed and why, and which policies were applied along the way. Adding the next agent means writing its checklist, its checkpoints, its tools and its test cases.

## 2. One finding, start to finish

A duplicate invoice: the same £111,784 invoice from one supplier appears twice against one purchase order.

1. The mining run starts. The agent tells the Orchestrator what it has been asked to do and what data it needs. The Orchestrator checks the few policies that bear on that and answers.
2. The miner runs its detectors, which are code, as today.
3. It raises the finding. Because the value is above a set amount, it reports that decision to the Orchestrator, which checks the policies relevant to a finding of that size.
4. The miner fills in its checklist: for each heading, looked at, does not apply, or could not assess, with the reason.
5. A separate check tries to prove the finding wrong from the invoices and the order, and separately asks what should be done about it.
6. Before the finding is shown as value, emailed about, handed to negotiation or recorded as a saving, the agent reports the intended action. The Orchestrator checks the policies for that action together with the checklist and the check's verdict.
7. If two policies disagree at any of these points, the decision engine works out where, settles it if a standing rule or an earlier ruling covers it, and otherwise puts it to a person. Only the step in question waits.

## 3. Decisions made by the product owner (2026-10-12)

### 3.1 Architecture

| Decision | Ruling |
|---|---|
| Where policy is applied | By the Orchestrator. Not in a separate gate. |
| How many policies | Many policies may bear on one activity. The checklist is not a policy and must not be treated as the one policy for a run. |
| When | Agents report back at key points of access or decision, not on every turn. |
| Which policies, when | Only those relevant to the point reached. At the start, only the key ones for the instruction and for access to data. |
| Overload | The agent must not be loaded with every policy, and a clash between policies must not stop it completing its task. |
| Conflicts | Go to the decision engine first, which works out where the conflict is. It may settle one on a standing rule or an earlier ruling, as it does today. Otherwise it presents the conflict to a person. |
| Order | The Orchestrator's part is built first. |

### 3.2 Agents and data

| Decision | Ruling |
|---|---|
| Who owns a checklist | The existing user types Category Manager (`PROCWISE_CATEGORY_MANAGER`) and tenant super admin (`PROCWISE_TENANT_SUPER_ADMIN`). No new role. |
| Agent order | Opportunity Miner, discrepancy triage, negotiation, quote comparison and award, contract review and renewals, sell-side quotes, sourcing (placeholder only) |
| Budget, ESG, supplier performance | Reported as "could not assess", and carried into the negotiation report |
| Opportunity validation | Its own check, with a different angle from the negotiation plan reviewer |
| Duplicate-invoice detector | Switched on once the check is live |
| Price-variance detector | Connected to the existing benchmark engine in this work |
| Detectors in the original requirements that were never built | Re-checked (9.1) and listed as could not assess |
| Existing findings | All 308 on `bp_testdb` re-checked and shown before anything is hidden. First result in 10.4. |
| Accepting a supplier's offer | Always needs a person. The rule is a policy, changeable through the learning loop. |
| Negotiation rounds | Multi-round stays on. The cap becomes a policy value so it can be learned. |
| Dummy data | Used now, kept out of production tables |

## 4. How this fits the orchestrator brief

The earlier brief proposed eight work packages. This design takes two of them further and leaves the rest where they are.

| Brief | This design |
|---|---|
| WP1: one policy check on every agent run | Absorbed and extended: a check at the start and at each later checkpoint, with only the relevant policies at each (section 6) |
| WP3: results change the route | Partly absorbed: the "cannot proceed" and "new information" checkpoints, and the answers the Orchestrator gives (section 6) |
| WP2: `determine_path` routing | Unchanged, still the brief's. The run report records the route once it exists. |
| WP4 (Ask reaches the Orchestrator), WP5 (agents read what they are given), WP7 (remove what is not real), WP8 (what AgentNick is) | Unchanged, still the brief's |
| WP6: negotiation with the model inside coded limits | Stage 7 here, built on this platform |

The brief's rule "no applicable policy means an acting agent does not run" is kept for the action checkpoint only, and starts record-only, for the reason the brief gives: `bp_sqldb` has no live agent policies.

## 5. Stages

| Stage | What | Section |
|---|---|---|
| 1 | Orchestrator checkpoints. Record-only. | 6 |
| 2 | The checklist and the run report | 7, 8 |
| 3 | Opportunity Miner: honest figures, benchmark connected | 9 |
| 4 | The check on each finding | 10 |
| 5 | Switch from record-only to enforcing, `bp_testdb` first | 11 |
| 6 | Discrepancy triage | Own design |
| 7 | Negotiation, including its four safety fixes (section 12) | Own design |

---

## 6. Orchestrator checkpoints (shared)

### 6.1 The eight points

| # | Point | The agent reports | Why it is separate |
|---|---|---|---|
| 1 | Instruction received | What it has been asked to do, and for whom | Decides which policies are in play at all |
| 2 | Access | What sensitive data or outside source it is about to read | Access rules differ from action rules |
| 3 | Plan set or changed | Its approach, and again if new evidence changes it | A changed plan can bring different policies into play |
| 4 | Decision | Something material it has settled | The moment a later action becomes likely |
| 5 | New information from outside | A supplier reply or new document has arrived | An earlier "carry on" may no longer hold |
| 6 | Hand-off | It is about to pass work or data to another agent | The next agent may have different permissions |
| 7 | Cannot proceed | Low confidence, missing evidence, a limit about to be reached, unclear authority | The agent asks instead of guessing |
| 8 | Action | Something is about to leave the product or change a record permanently | The last point at which it can be stopped |

Disclosure (what an outside message reveals) and changes to records are checked at point 8. The product's closed list of named actions (`src/services/actions.py`, 73 entries, each classed as reversible or not) identifies which actions need it.

Routine work between points (reading rows, arithmetic) does not report.

### 6.2 What the Orchestrator answers

| Answer | Meaning |
|---|---|
| Carry on | Nothing that applies objects |
| Carry on, with a change | For example a lower limit, or a field that must be withheld |
| Hold this step | This step waits for a person or a ruling; the rest of the task continues |
| Stop | The run ends. Only for an instruction a policy forbids outright, or policies that cannot be read. |

### 6.3 The right policies at the right moment

- Policies already carry the checkpoint they apply at: `enforcement.check` filters on `policy.context.checkpoint` (`src/services/agent_policy/enforcement.py:98-104`). Today there is one value, `tool.call.before`. This design adds the eight above as the permitted values.
- At point 1 the Orchestrator applies only policies marked for the instruction and for access. Every other policy waits for its own point.
- The agent is never handed policy text in bulk. It receives the answer and any limit it must work within.
- A policy with no checkpoint is treated as an action-point policy, which is how every existing policy behaves.

### 6.4 Conflicts

When policies that apply at one point disagree:

1. The decision engine identifies the conflict: which policies, on what, and what each would allow. `decide_live_conflict` (`src/engines/decision_engine.py:2079`) and `conflict_engine.classify` already do this for tool calls.
2. If a standing rule covers it, or a person has already ruled on the same clash, the engine settles it. The self-approval bar still holds.
3. Otherwise it presents the conflict to a person with the facts and the history.

Rules that follow from "do not stop the task":
- A conflict holds only the step it affects. The run continues.
- A conflict between policies that never become relevant in this run is never raised.
- The report lists every held step and what it is waiting for.

### 6.5 One door

The same check must apply however the work started. Today it does not: the brief (its section 2.5) lists five entry paths checked five different ways, and built-in agents bypass the agent-policy gate entirely.

- One function, `Orchestrator.checkpoint(point, agent, subject, facts)`, is the only place a policy verdict is produced.
- `BaseAgent` gains `self.checkpoint(...)`, which calls it. Built-in and canvas-built agents use the same call.
- The existing tool-call gate (`src/services/agent_policy/gate.py`) becomes one caller of it, at point 8, with unchanged behaviour.
- The five entry paths (named workflow, canvas run, stored flow, supplier-reply watcher, scheduled job) each pass point 1.
- The existing agent signals (`CONFIDENCE_LOW`, `NEEDS_ATTENTION`, `RECOMMEND_ESCALATION` in `src/orchestration/workflow_context.py:20-24`) become point 7.
- Holding a step reuses what exists: approval cases, and runs that wait for input (`proc.bp_workflow_input_request`).

### 6.6 Record-only first

Every checkpoint writes what it would have answered. While an agent is in record-only mode the answer returned is always "carry on". Moving an agent to enforcing is a setting on its checklist (section 7).

### 6.7 What is checked that is not a policy

At point 8 the Orchestrator also requires, from sections 7 to 10: that the run's earlier listed checkpoints were passed; that the report is complete; and that the subject of the action has a verdict that permits it. These are conditions of the action, recorded beside the policy verdicts, not policies.

## 7. The checklist (shared)

**What it is.** The fixed list an agent must report on, its pass-or-fail assertions, and which of the eight checkpoints it must use and where.

**Where.** Its own table, `proc.bp_agent_checklist`, one active version per agent, every change a new version. It is not stored in `proc.bp_policy`.

**Who may change it.** Only a Category Manager or a tenant super admin. The product's four roles (Admin, Approver, Buyer, Viewer) are too coarse for this, because Category Manager shares the Buyer role with Buyer Analyst (`deploy/sql/2026-09-18_admin_area_role_mapping.sql`). The rule therefore names the two user types, checked against the user-type claim `src/services/rbac.py` already reads. **To confirm during build:** that the endpoint check can be given the user type as well as the role.

**Rules.**
- A missing or unreadable checklist stops the agent. It does not run on a default.
- It carries the agent's mode: record-only or enforcing.
- Reports record the version they ran under.

## 8. The run report (shared)

One record per run, in one shape for every agent, in a new append-only table `proc.bp_agent_run_report` (updates and deletes blocked by trigger, as `proc.bp_agent_actions` already is).

| Part | Content |
|---|---|
| Header | Agent, checklist version, what the run was about, what started it, who |
| Brief | What was asked for |
| Checklist items | For each: looked at, does not apply, or could not assess, with reason, counts and evidence references |
| Assertions | For each: pass, fail or could not verify |
| Checkpoint trail | For each checkpoint passed: the point, what the agent reported, each policy that applied with its version and what it said, how any conflict was settled and by whom, and the answer given |
| Held steps | Each step waiting, and for what |
| Approach | The case-specific plan. Short for the miner. |
| Open questions | What is unknown or assumed |
| Proposed additions | Anything the agent suggests adding to its checklist. Recorded, never applied. |
| Outcome | Proceed, revise or escalate, with reason |

A report with a checklist item missing cannot be recorded as "proceed".

## 9. The Opportunity Miner

The miner stays what it is: detectors written as code over documents, switched on by `proc.bp_rule` rows. It is not made model-driven, because its value is arithmetic.

### 9.1 Checklist

| Item | Detectors behind it | Expected status after this work |
|---|---|---|
| Price position | Volume consolidation, duplicate supplier | Looked at |
| Benchmark | Price variance against the benchmark engine | Looked at |
| Billing integrity | Invoice overbilling, duplicate invoice, inflation pass-through | Looked at |
| Contract | Expiry, expiry buckets, unused contract value, maverick spend | Looked at, once repointed (9.4) |
| Supplier | Risk, performance | Could not assess: no performance source |
| Budget | Category overspend | Could not assess: no budgets supplied |
| ESG | ESG switch | Could not assess: no ESG inputs |
| Index | Contract price drift against an index | Could not assess: no index data held |
| Payment and logistics | Early-payment discount missed; logistics cost outliers | Could not assess: no discount terms or shipment data held |

The last two come from the original requirements (`procwise_agents/business_requirements/Agents/Opportunity Finder Agent/Opportunity_Agent_MVP_Engineering_Requirements_v1.4.docx`). Re-checked 2026-10-12:

| Detector | Finding |
|---|---|
| Unit price against a benchmark | A tested benchmark engine exists (`src/services/benchmark/engine.py`, fed by `src/services/benchmark_live.py` from past PO and invoice lines). The miner does not call it; it uses the cheapest other supplier's average (`opportunity_miner_agent.py:5478`). Connected in this work. |
| Price drift against an index | No index table in either database. |
| Early-payment discount missed | Payment terms filled on 10 of 12,408 invoices on `bp_testdb` and 35 of 51 on `bp_sqldb`; no field records discount terms. |
| Logistics cost outliers | No shipment table in either database. |

### 9.2 Assertions

1. Every figure comes from a registered formula.
2. Every finding names source rows that exist.
3. The currency label is true.
4. Nothing is shown below the governed minimum impact.
5. No finding is retired by a detector that could not read its data.
6. No finding a person has progressed is changed.

### 9.3 Checkpoints

| Point | Where in the miner |
|---|---|
| 1 Instruction | Start of a run: scope and what data it will read |
| 4 Decision | A finding above the governed value is raised |
| 7 Cannot proceed | A table cannot be read, or a threshold is missing |
| 6 Hand-off | A finding is passed to negotiation |
| 8 Action | Retiring existing findings; showing a finding as value; a supplier query email; recording realised value |

### 9.4 Figures and behaviour

- Each detector's figure moves to a registered formula in `src/services/formulas/definitions/opportunity.py` with test values. `opportunity.price_variance_impact` is registered and unused today.
- Each finding is labelled a **saving** or a **value at stake**, using the existing `value_basis` column. Whole contract values, whole PO totals and duplicates with no payment status are values at stake and are never summed into "potential savings".

| Today | After |
|---|---|
| Missing quantity becomes 1; missing unit price becomes the line value (`:5440-5445`) | The row is excluded and counted under "could not assess" |
| A failed table read becomes an empty table (`:2353-2366`) | Reported at point 7; the item is "could not assess"; nothing is retired |
| At most two findings per detector per run (`:961-970`) | No cap |
| If nothing clears the minimum, the best finding is shown anyway (`:1964-1980`) | Nothing is shown; the report says none cleared it |
| Thresholds default to 0.0 in code | A missing threshold is "could not assess" |
| Older contract detectors read `proc.contracts`, which does not exist in either database | They read `proc.bp_contract_master` (3,051 rows in each) |
| Prompt text can change the scope and the minimum impact (`:728-769`) | Removed; only governed rows set them |
| An unconverted contract value is written into a GBP column (`:6203`) | Converted, or "could not assess" |

### 9.5 Identity

- Findings are keyed by the stable `opportunity_ref_id`. Today the primary key is a per-run counter, a new finding can collide with an old row, and the failed save is swallowed (`:7558-7568`). The migration changes the key, maps value outcomes and playbook proposals, and the save stops swallowing errors.
- Production's `bp_opportunity` lacks the `retired_at` column that test has. The migration adds it.

## 10. The check on each finding

### 10.1 Two questions, answered separately

The existing critic asks one question, "would this survive a negotiator's scrutiny". On 2026-10-12 it had rejected sound duplicate-invoice findings as "not a negotiable opportunity", which is a correct answer to the wrong question. The check therefore asks two:

| Question | How | Result |
|---|---|---|
| **Is it true?** | Code only, from the source rows | Sound, sound at a different amount, wrong, duplicate of another finding, could not tell |
| **What should be done about it?** | The existing critic's model call, given the finding and evidence only | A route: recover through accounts payable, raise with the supplier, take to negotiation, or note only |

A finding that is true but not negotiable is still a finding. Only the first question decides whether it is shown and counted.

### 10.2 The truth checks

| # | Check | When it cannot be done |
|---|---|---|
| 1 | Recompute the figure from the source rows, without the miner's working | Could not tell |
| 2 | Like-for-like: unit, quantity basis, currency, date | Could not tell |
| 3 | A contract term explains the variance | Recorded as not checked; does not fail the finding |
| 4 | Still open: not rejected, realised or already in negotiation | Could not tell |
| 5 | The same money is counted by another finding | n/a |
| 6 | The source rows exist and carry the numbers used | Wrong |

The eleven `critic.*` formulas already written and never executed (`src/agents/opportunity_critic_agent.py:155-163`) are executed here.

### 10.3 Storage and running

- New append-only table `proc.bp_finding_check`: the finding's stable reference, the report, the check version, each check's outcome and detail, the miner's figure, the recomputed figure, the truth verdict and the route.
- The check runs inside the mining run, straight after detection, for every finding. `scripts/critic_run.py` is retired.
- If the model is unavailable the truth verdict still stands; the route is recorded as not determined.
- A shared harness, `src/services/agent_spine/review.py`, guarantees a reviewer receives only the subject and its evidence. The negotiation plan reviewer (stage 7) uses the same harness with different checks.

### 10.4 First re-check of the existing 308 findings (`bp_testdb`, read-only, 2026-10-12)

Checks 5 and 6 and part of check 1, run as queries. Not the full check.

| Detector | Findings | Result |
|---|---|---|
| Duplicate invoice | 300, £3.23m | All hold on the facts: both invoices exist with the same supplier, total, date and currency, and on all 180 purchase orders the invoices exceed the order. None has a payment status, so the figure is value at stake. |
| Invoice overbilling | 6, £262k | Three single-invoice variances of 13% to 16% look genuine (about £7,000). Three are several invoices on one order (about £255,000); at least two of those deals also carry duplicate-invoice findings, so that money is counted twice. |
| Price variance | 2, £44k | Not verifiable until the benchmark connection replaces "cheapest other supplier". |

## 11. Proof and switching on

### 11.1 Scenario set

Dummy cases with known right answers, run on every change, in their own throwaway database as `evals/email/` does. No dummy rows enter `bp_sqldb` business tables.

**Findings:** a genuine saving; the same money raised by two detectors; a unit mismatch; a contract clause that explains a price rise; a finding already realised; source rows that do not carry the stated numbers; a missing quantity; a legitimate repeat order that looks like a duplicate; an unreadable table; the same run twice.

**Checkpoints and policy:**
- A policy relevant only at the action point is not applied at the start.
- Two policies that disagree at one point: only that step is held, and the run finishes the rest.
- Two policies that disagree but never become relevant: nothing is raised.
- A clash covered by an earlier ruling is settled by the decision engine without a person.
- A clash with no rule and no ruling reaches a person, with the conflict identified.
- An action attempted without its earlier checkpoints is refused.
- Policies cannot be read: the run stops and says so.
- The same work started through each of the five entry paths gets the same answers.

### 11.2 Order

1. Checkpoints on `bp_testdb`, record-only. Report what would have been held or stopped.
2. Checklist, report, miner changes and the check on `bp_testdb`.
3. Scenario set passing in CI.
4. Full re-check of the 308 findings; before-and-after shown to the product owner.
5. The same migrations and rows applied to `bp_sqldb`, record-only.
6. Duplicate-invoice detector switched on (`duplicate_invoice_detector_enabled` in `AutonomousOperationPolicy`), both databases.
7. Miner moved to enforcing, `bp_testdb` first, then `bp_sqldb`.

Each step is reported with its evidence before the next begins.

### 11.3 Screen

The opportunity view shows each checklist item's status and reason, the source rows behind a figure, the saving-or-value-at-stake label, the truth verdict and the route, and any held step. `source_records` and `calculation_details` are returned by the API, which they are not today. Built last, on the existing drawer and panel styles.

## 12. Negotiation safety fixes (done at the start of stage 7)

| Hole | Evidence | Fix |
|---|---|---|
| A caller-supplied `authority` block is trusted outside three workflows | `src/agents/negotiation/analysis.py:194-198`; `src/orchestration/orchestrator.py:419-423` | Replaced by the instruction and access checkpoints: the mandate always comes from the Orchestrator |
| The quantity in the value-limit check comes from the request | `src/agents/negotiation_agent.py:9744` | Read from the quote lines on record; withheld if none |
| The single-supplier and plain-batch paths have no round hold | `_resolve_hitl_decision` has one caller (`negotiation_agent.py:2922`) | Each counter is a decision checkpoint on every path |
| A held round cannot be released | Held tasks written at `negotiation_agent.py:2206`, read by nothing | Held steps use the shared hold-and-release from 6.5 |

They were first planned as a stage of their own. With checkpoints in place three of the four are closed by the platform itself. They affect drafts only; nothing can be sent without approval today.

## 13. Not in this design

- Routing (`determine_path`) and the other work packages left with the brief.
- Building the index, discount and logistics detectors.
- Discrepancy triage and negotiation on the platform.
- Versioned playbooks with owners and usage data.
- The review queue for agent-proposed additions.
- Loading real production documents.

## 14. To confirm during build

1. That the endpoint check can be given the user type as well as the role (section 7).
2. How many existing policies carry no checkpoint, and that treating them as action-point policies changes nothing for them (6.3).
3. Whether the gateway's direct queries on `proc.bp_opportunity` need the same "checked or not yet checked" label (11.3).

## 15. Risks

| Risk | Handling |
|---|---|
| Checkpoints slow agents down or hold too much | Record-only first; the step-1 report shows how often each point would have held before anything enforces |
| Too few policies are marked for the early points, so the start check is empty | Expected at first. The report shows which points had no applicable policy. |
| Most existing findings change status, and "value found" falls | Shown as a before-and-after at step 4 before any screen changes |
| Changing the opportunity table's key breaks references | The migration maps them, runs on a copy first, and row counts are compared |
| Two sessions work in this checkout and on these databases, and the brief's author may start WP1 | Agree ownership of the Orchestrator change before stage 1 starts |
| Enforcing stops something that works today | Per agent, and one setting to reverse |
