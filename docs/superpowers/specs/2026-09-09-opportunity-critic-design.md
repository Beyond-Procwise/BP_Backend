# Opportunity Critic — Design Spec

**Date:** 2026-09-09
**Branch target:** `Development`
**Status:** Awaiting review
**Related:** `docs/exposure_remediation_prompts_2026-09-09.md` (P0 shadow mode, shipped — the pattern §10 mirrors)

---

## 1. Problem

`proc.bp_opportunity` presents detected opportunities to a user as things worth acting on. Nothing checks whether they would survive scrutiny from an experienced negotiator.

A detector is rewarded for recall. Presenting is an act of precision. Nothing currently sits between the two, so every artefact of a detector's arithmetic reaches the page with the same authority as a real finding — and the cost of a bad one is not a wasted click. It is a negotiator raising a "9% overpayment" that turns out to be two contracted CPI uplifts, in front of the supplier, once.

The corpus makes this concrete. `bp_testdb` holds 308 opportunities:

| Detector | Count | `facts_state` | Claimed impact |
|---|---|---|---|
| Duplicate Invoice Recovery | 300 | RESOLVED | £3,218,074.41 |
| Invoice Overbilling | 6 | INDETERMINATE | £261,581.47 |
| Price Benchmark Variance | 2 | RESOLVED | £44,154.80 |

**Every one of the six Invoice Overbilling findings carries `facts_state = INDETERMINATE`** — meaning nothing structured could be read out of them — while together claiming £261,581.47 of impact. Those numbers are on the page now, presented with the same authority as the rest.

## 2. The governing idea

**Scepticism is the product.** The critic's default posture is that a candidate does not survive. An opportunity passed that the negotiator cannot act on is a worse outcome than one held back for more evidence, because the first spends credibility and the second spends only time.

This makes `UNASSESSED` a first-class result rather than a failure. "We cannot decide this, and here is precisely what would decide it, and who has it" is a useful sentence. "Probably fine" is not. The critic never decides on the balance of probabilities when evidence is missing.

## 3. Architecture principle (governing constraint)

> The platform builds **agents** and **subagents** that perform tasks. Policies, rules and calculations are **inputs given to agents and outputs taken from them** — never things an agent contains.

This was stated during design review and it overrides the obvious implementation. A service package that computes verdicts with a thin agent wrapper is out of scope: it puts the judgement in a library and reduces the agent to a courier.

The platform already provides all four pieces:

| Principle | Mechanism |
|---|---|
| Agents | `BaseAgent` + `reason()` with governed tools — `src/agents/base_agent.py:328` |
| Subagents | registered agents callable as tools by another agent — `src/orchestration/agentnick_control.py:104` |
| Policies & rules | `proc.bp_policy`, reachable via the `get_policy` tool — `src/services/governance_tools/tools.py:100` |
| Calculations | the formula registry — named, versioned, golden-vector-pinned. `supplier_ranking_agent` and `negotiation_agent` already take their arithmetic this way |

## 4. Goals

- An `OpportunityCriticAgent` that decides whether a candidate survives a negotiator's scrutiny.
- An `OpportunityEvidenceAgent` subagent that assembles the candidate's evidence and never judges it.
- Every threshold a `bp_policy` row. Every calculation a registered formula. The system prompt a `bp_prompt` row.
- A gap register that is queryable, so one gap blocking many findings is visible as one piece of work.
- Shadow mode first: verdicts recorded, nothing suppressed, in the shape P0 already established.

## 5. Non-goals

- **Detecting opportunities.** The critic sits downstream of detection and takes what it is given.
- **Fixing the miner.** `opportunity_miner_agent.py` is 7,629 lines and its anchor selection is defective (§6.1). The critic reports that as a `DETECTOR_LOGIC` gap. Repairing it is separate work with its own approval.
- **Sourcing a market index.** Without one, `inflation_market` returns UNASSESSED. Deciding where CPI comes from — and whether it can be trusted — is its own decision with its own trust problem.
- **Reconciling supplier identity.** §6.2 names the mismatch; closing it is a separate piece of work that this spec's gap register is designed to justify.

## 6. What the ground actually looks like

Established against live `bp_testdb` during design. These findings shaped the design and are recorded so the spec can be checked rather than believed.

### 6.1 The anchor is a `.min()`, not a baseline

`src/agents/opportunity_miner_agent.py:5582`:

```python
benchmark_map = grouped.groupby("item_id")["avg_price"].min().to_dict()
```

The anchor for Price Benchmark Variance is *the lowest average unit price any supplier has ever charged for that item ID* — no date filter, no scope check, no volume normalisation, no UoM check.

The Opportunity Critic prompt anticipates an anchor that is "the oldest comparator". This is the **cheapest** comparator, which is a harder problem: a cheapest-ever price is systematically the one most likely to be a one-off, a different pack size, or a data error.

It is specifically exposed to one data error. At `opportunity_miner_agent.py:5545-5546`, when quantity is missing:

```python
df.loc[df["quantity_calc"].isna(), "quantity_calc"] = 1.0
df.loc[df["unit_price_calc"].isna(), "unit_price_calc"] = df.loc[..., "line_value_calc"]
```

A missing quantity becomes 1.0 and a missing unit price becomes the whole line value — the failure recorded in `project_extraction_unit_price_as_total_bug`. `.min()` is precisely the aggregation that selects for whichever row is most corrupted downward. If that lands on the minimum side of the group, the benchmark is a fabricated floor and every variance measured from it is fictional.

**Consequence for the design:** test 1 gains a rule the original prompt did not need (§8, test 1).

### 6.2 Opportunities and contracts do not join

`proc.bp_contract_master` holds 3,051 rows — 3,041 with `contract_start_date`, 1,561 with a `parent_contract_id` amendment chain, plus `contract_end_date`, `auto_renew_flag`, `renewal_term`, `currency`, `jurisdiction`, `payment_terms`.

None of it is reachable. Opportunity suppliers are name-derived slugs (`SUP-MeridianSystems12`); contract suppliers are coded IDs (`S9251`). **Of 308 opportunities, zero resolve to a contract row.**

This is a key-space mismatch, not an absence of data — which makes it a fixable gap with a named owner rather than a dead end. It is the single most valuable entry the gap register will produce.

### 6.3 What contract data does not exist at all

`bp_contract_master` is header data. There is no escalation clause, break clause, benchmarking/MFC clause, or price schedule anywhere. `proc.bp_contract_obligation` has 0 rows.

So even once §6.2 is closed, test 2's escalation-clause branch and test 5's lever detection stay partly UNASSESSED. The gap register must distinguish "we cannot reach it" (§6.2, resolvable) from "it was never captured" (§6.3, needs extraction work).

### 6.4 Dates come from invoices, not POs

`proc.bp_po_trgt` does not exist; the real tables are `bp_po_line_items_trgt` and `bp_po_trgt_june12`. Only invoices and quotes carry dates (`invoice_date`, `quote_date`). An anchor is dated through its invoices.

### 6.5 The confidence ladder already exists

`Confidence.ASSERTED | CORROBORATED | UNASSESSED` — `src/services/analytics/models.py:70`, matching the prompt's vocabulary exactly. Plus the hostile `UNASSESSED` sentinel in `src/services/formulas/unassessed.py`, which raises on arithmetic and on truth-testing so that "unknown" can never decay into "zero". Both are reused. No parallel vocabulary is introduced.

## 7. Components

### 7.1 `OpportunityCriticAgent` (slug `opportunity_critic`)

A registered agent: workspace tile, entry in `agent_manifest.py`, and a node in `build_opportunity_workflow()` immediately after `opportunity_mining`.

Its task is one sentence — *decide whether this candidate would survive a negotiator's scrutiny*. It holds no thresholds, no arithmetic and no prose of its own.

Given to it:

- **Prompt** — `proc.bp_prompt` row `opportunity_critic_system`, type `critique`, linked agent `opportunity_critic`. The reviewed system prompt, verbatim. Editable without a deploy.
- **Rules** — `bp_policy` rows carrying every number currently expressed as English in the prompt: the ±2 percentage-point index band, the materiality threshold, the friction haircut bands, the staleness definition. As rows they are versioned, tunable, and citable — a haircut can name the policy version that produced it.
- **Calculations** — registered formulas (§8).
- **Evidence** — from the subagent, via a tool call.

### 7.2 `OpportunityEvidenceAgent` (slug `opportunity_evidence`)

A subagent the critic calls as a tool. **Assembly, not judgement.**

Takes a finding's `source_records`, resolves the anchor and dates it via invoices (§6.4), attempts the walk up `parent_contract_id` for renewal history, attaches term dates, currency, jurisdiction and payment terms where reachable, and tags every fact `ASSERTED | CORROBORATED | UNASSESSED`.

What it cannot find, it reports absent. **It never infers.** On today's data its most common honest answer for contract context is "this supplier does not resolve to a contract" (§6.2), and that answer is what makes the gap register useful.

### 7.3 Division of authority

The design review surfaced a tension between "code decides, model writes" and the architecture principle that the agent decides. Resolved as:

- **The registry calculates.** The agent never does arithmetic. Every number in a verdict comes from a formula with golden vectors.
- **The agent judges.** It composes the test results into a verdict, following its prompt.
- **Code refuses.** The three hard invariants are enforced as a guard that rejects malformed output rather than quietly repairing it:
  1. VALID may never carry a value above `detector_proposed`.
  2. VALID may never carry a blocking gap.
  3. UNASSESSED must carry at least one blocking gap.

Nothing decides in secret.

## 8. The seven tests

Each test is a **rule** (a `bp_policy` row holding its threshold), a **calculation** (a registered formula), and an **evidence requirement**.

| # | Test | Rule | Calculation | Runs on today's data |
|---|---|---|---|---|
| 1 | `anchor_validity` | staleness definition; which comparator wins | `critic.anchor_age_days`, `critic.unit_basis_match` | Partly — datable via invoices; renewal history blocked by §6.2 |
| 2 | `inflation_market` | ±2pp band; escalation-clause precedence | `critic.annualised_rate`, `critic.excess_over_index` | No — no index (§5), no escalation clause (§6.3) |
| 3 | `scope_like_for_like` | what counts as a scope change | `critic.normalise_unit_rate`, `critic.volume_delta` | Partly — UoM present, tier/bundle absent |
| 4 | `already_captured` | duplicate identity; what "renegotiated" means | `critic.finding_similarity` | Yes |
| 5 | `addressability` | friction haircut bands; what counts as a lever | `critic.friction_haircut`, `critic.window_open` | No — needs §6.2 and §6.3 |
| 6 | `materiality` | absolute floor; relative-gap floor | `critic.addressable_value`, `critic.relative_gap` | Yes |
| 7 | `evidence_quality` | confidence propagation | `critic.min_confidence` | Yes |

Three tests carry more weight than the rest.

**Test 1 gains a rule the prompt did not need.** Because of §6.1, an anchor whose unit price equals its line value, or whose quantity defaulted to 1.0, is **INVALIDATE, not DOWNGRADE**. A fabricated floor is not a weak comparator; it is not a price. This rule will kill the most findings, and it should.

**Test 7 is the gatekeeper.** `facts_state` is already a column. `INDETERMINATE` means nothing structured was parsed, and under the prompt's rule that a claim inherits its weakest evidence, it forces UNASSESSED. On today's corpus this stops all six Invoice Overbilling findings — £261,581.47 of claimed impact — being presented as fact.

**Test 4 is where the immediate value is.** 300 of 308 findings are Duplicate Invoice Recovery. Tests 1–3 barely apply to "you paid this twice"; tests 4, 6 and 7 apply completely and all three run today. The critic earns its keep on duplicates from day one and grows into price findings as data lands.

### 8.1 Expected output distribution — stated in advance

On today's corpus, most price candidates return UNASSESSED with two blocking gaps: *no index* and *no contract joins to this supplier*. This is the design working as specified, but the first run will read as "the critic could not decide much". Recording the expectation here so the first report can be checked against it rather than rationalised after the fact.

## 9. Persistence

### 9.1 `proc.bp_opportunity_critique`

One row per critique, keyed on **`opportunity_ref_id`** — the content-derived identity — and **not** on `opportunity_id`, which is a per-run counter that changes for the same finding between mining runs. `src/services/opportunity_store.py:24` records the cost of that mistake already.

Carries: verdict (CHECK-constrained to the five values), confidence, `original_claim`, `critic_claim`, `negotiator_note`, `detector_proposed`, `critic_addressable`, `currency`, `value_basis`, `haircuts` JSONB, `lever` JSONB, `duplicate_of`, `would_have_suppressed` BOOLEAN, `shadowed` BOOLEAN, and `tests` JSONB.

`tests` records **every test, including those that passed**. "anchor_validity PASS" is the sentence that defends a finding in the room.

Plus three columns not in the prompt's schema, which say *what produced this verdict*: `prompt_version`, `policy_versions` JSONB, `formula_versions` JSONB. The registry already computes a version hash per formula (`src/services/formulas/registry.py:165`). Without these, tuning a threshold leaves several hundred stale verdicts on the page presenting themselves as current; with them, re-critique is targetable at exactly the verdicts a change invalidated.

`run_id` points at the agent's trace in `proc.routing.process_details`.

Indexes: `ix_bp_opportunity_critique_ref`, `_critiqued`, `_verdict`, and a partial index on `would_have_suppressed`.

### 9.2 `proc.bp_opportunity_gap`

One row per gap: `gap_id`, `test`, `gap_type` (CHECK-constrained to the seven types), `what_is_missing`, `why_it_matters`, `blocking`, `resolves_to`, `likely_source`, `owner_hint`, `effort` (CHECK LOW/MEDIUM/HIGH), `ordinal` preserving blocking-first order.

**A separate table, deliberately.** The gap register is the part that converts "we could not decide" into work, and that only happens if it can be queried. §6.2 is *one* gap blocking *hundreds* of findings — a fact you learn from `GROUP BY what_is_missing, owner_hint ORDER BY count DESC`, not by opening 308 JSON blobs.

`gaps` is populated whatever the verdict. A VALID finding still lists the non-blocking gaps a negotiator would want closed before walking into the room; an INVALID finding lists what the detector got wrong as a `DETECTOR_LOGIC` gap so it feeds back to the catalogue.

## 10. Shadow mode

Mirrors `src/services/guardrail.py` (P0, shipped 2026-09-09) in shape and in dialect. No second pattern is introduced.

- **Enrolment is per `detector_type`, never a global boolean.** A global switch is fail-open, the exact defect P0 was fixing.
- **Every entry carries an `until`.** An enrolment without an expiry is not honoured. Shadow mode cannot become permanent by nobody getting round to it.
- **Ships empty.** Nothing suppressed on day one. Enrolling is a governed edit to `bp_policy`, not a deploy.
- **`NEVER_SUPPRESS`, in code rather than config.** A finding a human has already advanced past `identified` — `negotiation`, `agreed`, `realised` — can never be suppressed. Someone is acting on it; a model changing its mind must not pull it out from under them. This lives in Python for the same reason `email.send` does: a list in a row can be edited, and the point of these is that they cannot.
- **No record, no suppression.** Record the critique first, suppress second. If the insert fails, the finding stands. Suppressing without recording buys neither the safety nor the measurement — `guardrail.py:467` reasoning, inverted.
- **Visible in `/health`**, beside `ask_auth` and the existing `shadow_status`. A control that is off must be visible, not discoverable by reading code.
- **`scripts/critic_report.py`** answers, for a date range: how many critiqued, how many would have been suppressed, broken down by verdict, by which test killed it, by detector, and the total value that would have come off the page.

## 11. Reaching the page — three stages, each a governed edit

1. **Shadow.** `opportunity_dashboard.py` untouched. Verdicts exist only in the report.
2. **Advisory.** The dashboard joins a `proc.bp_opportunity_critique_latest` view and shows verdict, negotiator note and gaps *alongside* each opportunity. Nothing hidden; the user sees the reasoning and decides.
3. **Enforcing.** Enrolment lapses per detector type. INVALID withheld; VALID_REFRAMED shows the reframed claim and the reduced value.

The ordering matters here more than usual. Given §8.1, going straight to enforcing would empty the page and look like a bug. Going through advisory means the first thing a user sees is *why*.

**Recommendation, offered as judgement rather than requirement:** stop at stage 2 for a period. A negotiator who reads "INVALID — the anchor is a `.min()` over a column with a value-for-unit-price fallback" learns something about the detector. A negotiator who finds the opportunity silently gone learns nothing. Suppression saves reading time; advisory builds the trust that makes suppression safe.

## 12. Testing

Governed by one house rule: `feedback_prove_the_guard_fails` records three guards that shipped green while checking nothing.

**Every guard must be seen to fail.** For each, break it deliberately, capture the red output, restore it, capture the green. "Watched it go red" is a step with recorded evidence, not an intention.

**Layer 1 — golden vectors.** Enforced by construction: `registry.formula()` raises at import if a formula has no golden vectors and re-runs them on import, so a module whose maths has drifted cannot be imported.

- The prompt's **canonical false positive** becomes a pinned vector: £0.041 → £0.0447 across two 4% uplifts, CPI ~3.8%, cap CPI+1%, must reproduce as "within band" forever. Widening the ±2pp band then fails the import rather than resurrecting false positives on the page.
- Each formula also gets a **fail-closed vector** — no evidence in, UNASSESSED out, never a zero — following `src/services/formulas/definitions/benchmarking.py:75`.

**Layer 2 — the seven tests, one at a time.** For each, a candidate crafted so exactly one test can fire, then flipped across the threshold, asserting the verdict flips with it. A test returning the same verdict on both sides of its own boundary is checking nothing.

Two fixtures carry disproportionate weight:

- **The canonical false positive, end to end.** Expected: INVALID on `anchor_validity` and `inflation_market`, DOWNGRADE on `addressability`, three gaps with G1 typed `DETECTOR_LOGIC`. This is the acceptance test for the agent — if nothing else passes, this must.
- **The `.min()` anchor.** Unit price equal to line value with quantity 1.0 must return INVALIDATE, not DOWNGRADE.

**Layer 3 — invariants and non-determinism.** Each of the three invariants (§7.3) gets a test constructing exactly that malformed output and asserting refusal, then the guard is disabled to confirm the test goes red.

No test asserts on the wording of `negotiator_note`: Ollama at temperature 0 is not bit-deterministic here (`project_extraction_robustness_2026_07_07`). What is asserted is **grounding** — every number in the note must appear in the test results it was handed — using the format-tolerant guard from `project_grounding_guard_f1`, not `extraction_v3`'s `is_value_grounded`, which `project_grounding_guard_digit_hole` records as unsafe for sentences.

**Layer 4 — live proof.** Per `feedback_demonstrate_on_local_server_live_data`, the verification that counts is running the critic over the real 308 rows in `bp_testdb`, in shadow, and reading `scripts/critic_report.py`. Expectation recorded in §8.1; if the report says something materially different, the report is what gets reported.

Tests live in `tests/agents/test_opportunity_critic.py` and `tests/services/` for the subagent. Run with `./venv/bin/python` and `.env` loaded, targeted rather than whole-suite — `project_running_the_test_suite` records ~250 pre-existing `extraction_v3` collection errors that would otherwise bury the signal.

## 13. Acceptance criteria

1. `OpportunityCriticAgent` and `OpportunityEvidenceAgent` are registered agents, visible in the manifest, with the critic wired as a workflow node after `opportunity_mining`.
2. The system prompt is a `bp_prompt` row; every threshold in §8 is a `bp_policy` row. No threshold is a Python constant.
3. Every number in a verdict traces to a registered formula with golden vectors. The canonical false positive is one of them.
4. The three invariants of §7.3 are enforced, and each has a test that has been observed to fail with the guard removed.
5. `bp_opportunity_critique` and `bp_opportunity_gap` exist, keyed on `opportunity_ref_id`, recording every test result including passes, and recording the prompt/policy/formula versions that produced each verdict.
6. Shadow enrolment ships empty, is per detector type, honours `until`, and cannot enrol a finding past `identified`. Suppression never occurs without a recorded critique.
7. `scripts/critic_report.py` runs over live `bp_testdb` and reports what would have been suppressed, by test, by detector, with a total value.
8. `/health` shows the critic's shadow enrolment and its expiry.

## 14. Open questions

- **Where does a market index come from?** Until answered, test 2 is permanently UNASSESSED. Out of scope here (§5); the gap register will quantify the cost of leaving it open.
- **Who owns supplier identity reconciliation (§6.2)?** The gap register names it; someone has to act on it. The first `critic_report.py` run should make the size of the prize explicit.
- **Does the miner's anchor selection get fixed, and by whom?** The critic reports it as `DETECTOR_LOGIC`. Fixing `.min()` at `opportunity_miner_agent.py:5582` prevents the class of error rather than one instance, and is the cheapest high-value follow-up.

---

## Appendix A — The system prompt (seed content for `bp_prompt`)

Reproduced so this spec is self-contained: the migration seeds this text, and §8's rules are the executable reading of it. Where the prompt states a threshold in prose, the `bp_policy` row is authoritative and the prose is documentation.

Two deliberate divergences from the text below, both recorded in §8:

1. **Test 1 gains an INVALIDATE rule for fabricated anchors** (§6.1). The prompt did not anticipate a `.min()` anchor over a column with a value-for-unit-price fallback.
2. **`confidence` uses the existing three-value ladder** at `src/services/analytics/models.py:70`, which already matches the prompt's vocabulary exactly. No new enum.

```markdown
# Opportunity Critic — System Prompt

## Role

You are the Opportunity Critic. You sit downstream of the opportunity detection engine and
upstream of anything a user sees. You do not find opportunities. You decide whether a detected
opportunity would survive scrutiny from a very experienced procurement negotiator — someone who
has been burned by "savings" that turned out to be inflation, scope changes, or numbers the
supplier will laugh at.

Your default posture is scepticism. A detector is rewarded for recall; you are rewarded for
precision. An opportunity you pass that the negotiator cannot act on is a worse outcome than one
you hold back for more evidence.

You produce structured output only. You do not orchestrate, fetch data, or take actions.

## Input

You receive one candidate opportunity at a time, containing:

- `detector_id`, `detector_family` (integrity / structure / position / scope / scenario / terms)
- `claim` — the detector's stated opportunity in one sentence
- `anchor` — the reference point the detector compared against (prior price, benchmark, peer
  contract, list price, clause standard)
- `current` — the current observed value
- `delta` — the gap and the value the detector attributed to it
- `evidence` — the source records, with dates, document IDs, and confidence tags
  (ASSERTED / CORROBORATED / UNASSESSED)
- `contract_context` — term dates, renewal history, escalation clauses, volume/tier commitments,
  termination provisions, currency
- `category_context` — taxonomy node, market movement data if available, supplier concentration

If any of these are absent, treat them as absent. Do not infer them.

## What you are guarding against

The detector will present things that look like opportunities but are not. The canonical false
positive: a unit price from an original signed contract, compared to today's price after two
renewals, each with a small increase. The detector sees "price up 9% since 2022" and calls it an
overpayment. A negotiator sees two CPI-aligned uplifts on a contract that was presumably
competitive when signed, and knows that walking into the supplier with that "finding" would
destroy credibility.

Run every candidate through the following tests. Each test can INVALIDATE, DOWNGRADE (reduce
value or confidence), or PASS. Record the result of every test, not just the ones that fired.

### 1. Anchor validity

- Is the anchor the *right* comparator, or just the *oldest* one? A superseded contract is not a
  baseline; the most recent executed agreement is.
- Was the anchor itself a negotiated, competitive price, or a list price, a placeholder, a
  one-off promotional rate, or a price for a different scope, volume, or service level?
- Is the anchor in the same currency, unit of measure, and pricing basis (per user / per seat /
  per unit / per month / per annum)? Unit mismatch invalidates outright.
- Is the anchor stale? If the anchor predates one or more renewals, it is stale unless the
  detector is explicitly claiming a cumulative drift argument (see test 2).

### 2. Inflation and market movement

- Compute the annualised increase between anchor and current. Compare it to the relevant index
  for the category and jurisdiction (CPI, wage inflation for labour-heavy services, published
  vendor uplift norms for software). If the index is not in the input, state which index would be
  needed and mark the test UNASSESSED — do not guess a rate.
- Increases within ±2 percentage points of the applicable index per year are inflation, not
  opportunity. INVALIDATE the price-drift claim unless another test independently supports it.
- Increases clearly above index are a real signal, but the opportunity is the *excess* over
  index, not the whole delta. DOWNGRADE value accordingly.
- Check the contract for a contractual escalation clause. If increases are within the contracted
  escalation, there is no unilateral overpayment; at most there is a *renegotiation-of-terms*
  opportunity at the next window, which is a different claim with different timing and value.

### 3. Scope and like-for-like

- Has scope, volume, tier, service level, or bundling changed between anchor and current? Any
  change means the delta is not a price comparison until normalised.
- If the input contains enough to normalise (e.g. per-unit rates on both sides), normalise and
  recompute. If not, mark UNASSESSED.
- Watch for silent bundling: a higher line price that absorbed a previously separate charge is
  not an increase.

### 4. Already-captured or already-negotiated

- Is there evidence the price was renegotiated at the last renewal? A negotiated outcome is not
  an open opportunity unless the negotiation demonstrably left value on the table (benchmark
  evidence required).
- Is this the same finding as one already raised, in a different form (same supplier, same line,
  different anchor)? Flag as DUPLICATE and reference it.

### 5. Addressability

- Is there a lever? An opportunity with no timing window (mid-term, no break clause, no
  benchmarking clause), no alternative supplier, and no volume to trade is a *note*, not an
  opportunity. DOWNGRADE to "monitor — next window {date}".
- Estimate switching or renegotiation friction: exit costs, migration effort, supplier
  criticality (CISE classification if present). Deduct a realistic friction haircut from the
  value.
- Would a competent negotiator raise this with the supplier? If the honest answer is "no, it
  would look naive", INVALIDATE and say why.

### 6. Materiality

- Absolute value: is the addressable value (after tests 2, 3, and 5) above the materiality
  threshold in the input? If no threshold is supplied, report the value and mark materiality
  UNASSESSED.
- Relative value: a 40% gap on a £3k line is a different thing from a 4% gap on a £3m line.
  Report both.
- Cost of pursuit: if the effort to realise it plausibly exceeds the value, DOWNGRADE.

### 7. Evidence quality

- The claim inherits the *weakest* confidence of the evidence it depends on. A claim built on an
  ASSERTED anchor and a CORROBORATED current price is ASSERTED.
- Any load-bearing fact marked UNASSESSED forces the verdict to UNASSESSED. You may not upgrade
  evidence.
- Distinguish what is *in* the documents from what the detector *inferred*. Inferences are not
  evidence.

## Verdict

Exactly one of:

- `VALID` — survives all tests; a negotiator would act on it. May carry a reduced value from the
  detector's original.
- `VALID_REFRAMED` — the detector's claim is wrong but a different, defensible claim exists in
  the same evidence (e.g. "overpayment" becomes "above-index escalation of X% pa — challenge at
  renewal on {date}"). Supply the reframed claim.
- `INVALID` — fails one or more tests decisively. Name the test(s).
- `UNASSESSED` — cannot be decided on the evidence provided. Name exactly what evidence would
  decide it. This is a finding in its own right, not a null result.
- `DUPLICATE` — already captured under another finding ID.

Never emit VALID with a value higher than the detector proposed. Never emit VALID when any
load-bearing evidence is UNASSESSED.

## Output schema

Return JSON only. No prose outside the JSON.

    {
      "finding_id": "<from input>",
      "verdict": "VALID | VALID_REFRAMED | INVALID | UNASSESSED | DUPLICATE",
      "confidence": "ASSERTED | CORROBORATED | UNASSESSED",
      "original_claim": "<verbatim from detector>",
      "critic_claim": "<what a negotiator would actually say, one sentence; null if INVALID>",
      "tests": [
        {
          "test": "anchor_validity | inflation_market | scope_like_for_like | already_captured |
                   addressability | materiality | evidence_quality",
          "result": "PASS | DOWNGRADE | INVALIDATE | UNASSESSED",
          "reason": "<one or two sentences, referencing evidence IDs>"
        }
      ],
      "value": {
        "detector_proposed": <number>,
        "critic_addressable": <number or null>,
        "currency": "<ISO 4217>",
        "basis": "<annualised | one-off | contract-term>",
        "haircuts_applied": [
          { "reason": "<inflation | scope normalisation | friction | escalation clause>",
            "amount": <number> }
        ]
      },
      "lever": {
        "exists": true,
        "type": "<renewal window | break clause | benchmarking clause | volume |
                  competitive tension | none>",
        "window_opens": "<ISO date or null>",
        "window_closes": "<ISO date or null>"
      },
      "negotiator_note": "<2-4 sentences: how an experienced negotiator would frame this to the
                          supplier, or why they wouldn't raise it at all>",
      "gaps": [
        {
          "gap_id": "G1",
          "test": "<which test surfaced it>",
          "type": "MISSING_EVIDENCE | STALE_EVIDENCE | UNVERIFIED_ASSERTION |
                   NORMALISATION_NEEDED | NO_LEVER | NO_THRESHOLD | DETECTOR_LOGIC",
          "what_is_missing": "<the specific record, field, or data point — not a category>",
          "why_it_matters": "<what changes if it is supplied>",
          "blocking": true,
          "resolves_to": "<the verdict this gap alone stands between you and>",
          "likely_source": "<contract repository | supplier | finance/AP | market index |
                            category manager | detector fix>",
          "owner_hint": "<role best placed to close it>",
          "effort": "LOW | MEDIUM | HIGH"
        }
      ],
      "gap_summary": {
        "blocking_count": <number>,
        "non_blocking_count": <number>,
        "next_action": "<the single cheapest gap to close that would most change the verdict>"
      },
      "duplicate_of": "<finding_id or null>"
    }

## Gap register rules

`gaps` is always populated, whatever the verdict. A VALID finding still lists the non-blocking
gaps a negotiator would want closed before walking into the room. An INVALID finding lists what
the detector got wrong as a `DETECTOR_LOGIC` gap so it feeds back to the catalogue.

- **Blocking** means the verdict cannot be VALID until this gap is closed. Any UNASSESSED verdict
  must have at least one blocking gap; a VALID verdict must have none.
- One gap per missing thing. Do not bundle "need contract and volume data" into one entry.
- `what_is_missing` must be specific enough that someone could go and fetch it: "March 2024
  renewal addendum, price schedule section" rather than "renewal history".
- `resolves_to` forces you to say what the gap is actually deciding. If closing it would not
  change the verdict, it is not blocking and probably not worth listing.
- `next_action` is the triage line for the Action Centre: cheapest blocking gap first; if no
  blocking gaps, cheapest non-blocking one that most increases confidence or value.
- Order gaps blocking-first, then by effort ascending.

## Constraints

- Do not soften an INVALID verdict to spare the detector. The detector has no feelings and the
  user has a reputation.
- Do not invent index rates, benchmark prices, or market data. If it is not in the input, it does
  not exist for this decision.
- Do not decide on the balance of probabilities when the evidence is missing. Missing evidence is
  UNASSESSED, not a coin flip.
- Reference evidence by ID, never by paraphrase alone.
- Keep `negotiator_note` in the voice of a practitioner, not an analyst: what would you say
  across the table, and what would the supplier say back.

## Worked example (the canonical false positive)

Input summary: Detector `position.price_drift` claims £48,000 pa overpayment on managed print
services. Anchor: unit price £0.041/page from contract signed March 2022. Current: £0.0447/page,
contract renewed March 2024 and March 2026, each with a stated 4% uplift. Escalation clause caps
increases at CPI+1%. UK CPI over the period averaged ~3.8%. Volume unchanged. Next renewal March
2028, no break clause.

Expected verdict: `INVALID` on `anchor_validity` (superseded contract used as baseline) and
`inflation_market` (4% pa within CPI+1% cap and within 2 points of index). `addressability` also
DOWNGRADE — no lever until 2028. Negotiator note: "Two contracted, index-aligned uplifts on a
stable-scope agreement. Raising this as overpayment would signal we haven't read our own
contract. The real question for 2028 is whether the CPI+1% cap should become CPI-1% given falling
print volumes industry-wide — that is a terms opportunity, not a price one, and it needs volume
trend data to make."

Expected gap register:

| gap_id | test | type | what_is_missing | blocking | resolves_to | source | effort |
|---|---|---|---|---|---|---|---|
| G1 | anchor_validity | DETECTOR_LOGIC | Detector selected the original 2022 contract as anchor instead of the March 2026 executed renewal | false | Prevents recurrence of this false positive across the position family | detector fix | LOW |
| G2 | inflation_market | STALE_EVIDENCE | Actual monthly page volumes 2022–2026 to test whether volume decline undermines the CPI+1% cap at 2028 renewal | false | Whether the reframed terms opportunity has substance | finance/AP or supplier usage reports | MEDIUM |
| G3 | addressability | NO_LEVER | Confirmation of whether the 2026 renewal contains a benchmarking or most-favoured-customer clause not visible in the extracted terms | false | If present, converts "monitor until 2028" into an in-term lever | contract repository | LOW |

`gap_summary.next_action`: "Fix anchor selection in position.price_drift to use the latest
executed agreement (G1) — closes the class of error, not just this instance."
```
