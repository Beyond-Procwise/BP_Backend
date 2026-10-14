# ADR 0002 — Registry-as-code formula layer

- **Status:** Accepted (implementation landed 2026-09-05, unmerged on `Development`)
- **Date:** 2026-09-05
- **Supersedes:** nothing. (Numbered 0002 because ADR 0001 was taken by a concurrent workstream mid-session.)
- **Related:** [`formula-inventory.md`](../formula-inventory.md),
  [`formula-registry-gap-report.md`](../formula-registry-gap-report.md),
  [`model-inventory.md`](../model-inventory.md),
  `docs/remediation/00_seam_map.md` §B3 (the GPSS decision)

---

## Context

This system computes 61 numbers from domain inputs across 25 modules: relationship
log-odds, supplier composites, benchmark variances, outlier verdicts, risk scores,
counter prices, opportunity impacts. Before this change, all 61 shared four properties:

1. **No contract.** Inputs were `dict`, `DataFrame`, `Series` or loose floats. No unit
   was declared, no range enforced. `risk_score` was consumed on a 0–1 scale in one
   module, a 0–100 scale in another and a coerce-either scale in a third, and nothing
   could complain.
2. **No version.** A formula's arithmetic could be edited and nothing recorded that it
   had been, so a number stored last month could not be traced to what produced it.
3. **No evaluation record.** `proc.bp_agent_actions` recorded *decisions*
   (`promote_held` 1,897 rows, `discrepancy` 794) and never an evaluation.
4. **Defaults where there should have been refusals.** Six sites returned a number for
   a missing input — most sharply `risk_intelligence_service`, where an unmeasured
   supplier defaults to a *perfect* performer on a risk model.

Two of these had already caused real, documented damage. Using a completeness figure
as an evidence-quality proxy made the promotion gate mathematically unreachable by a
perfect match (F = 75.5 against a gate of 80), so no invoice could promote and no deal
could form. And `fillna(0)` on an unmeasured criterion ranked a supplier whose payment
terms we had simply never read as though they had offered the worst terms on the table.

## Decision

Build `src/services/formulas/` — an in-memory, code-defined registry:

- `@formula(name, version, inputs, output, effective_from, gpss_version, owner, purpose, golden, kind)`
- Typed contracts: each input is a `Term` with a unit, a range and an optional
  `concept_code`; the `Output` declares type and unit.
- Every call validates before evaluating. A contract failure yields `UNASSESSED` and a
  `Finding` — never zero, `None` or a default.
- Every evaluation writes an `EvaluationRecord`: name, version hash, resolved inputs,
  provenance IDs, output, tri-state confidence, `evaluated_at`.
- Golden vectors live in the decorator call, beside the formula. They run at
  registration, so a drifted formula's module **does not import**.
- Callers use `evaluate(name, ctx)` / `evaluate_many(name, contexts)`. Batch writes one
  aggregated record.
- `model_inventory()` generates `docs/model-inventory.md` from the registry.

### Decisions inside that decision

**D1 — `UNASSESSED` raises rather than being falsy.** `bool(UNASSESSED)` and all
arithmetic on it raise `UnassessedError`. A falsy sentinel is exactly what lets
`result or 0` turn "unknown" into "zero", which is the failure mode this layer exists to
prevent. The cost is that a caller who ignores the tri-state gets an exception; that is
the intended cost, and it surfaces in tests rather than in a report.

**D2 — behaviour preservation beats contract strictness, for now.** The spec requires
both "contract failure returns UNASSESSED" and "the refactor must reproduce current
outputs exactly". Where today's code tolerates a missing input and returns a default,
those cannot both hold. Resolution: a term the current body tolerates is declared
**optional**, so the body still runs and returns the identical number, and the tolerance
is listed as a candidate for deliberate tightening with its numeric consequence.
`UNASSESSED` fires only on a genuine violation — a required term absent, a value outside
its declared range, a unit mismatch, an undeclared key.

**D3 — golden vectors pin defects as well as behaviour.** Where a formula is wrong, the
vector records the wrong answer with a note saying so (`risk.predictive_supplier_score`
pins that an unmeasured supplier scores *safer* than a measured one;
`quote.weighting_score` pins that no data scores 0.0). A defect that is pinned cannot be
fixed by accident, and fixing it deliberately becomes a version bump with visible
before/after numbers.

**D4 — `SET` formulas are a kind, not a batching hint.** Min-max normalisation and
ratio-to-cheapest are properties of the population. Declaring them `SET` means a later
"optimisation" into a per-row loop fails loudly instead of silently changing every
ranking. `evaluate_many` on a `SET` formula is one evaluation over the whole set; a
contract violation anywhere refuses the whole set, because a population statistic
computed from a partly-invalid population is not a smaller truth, it is a wrong one.

**D5 — `gpss_version` is accepted and never populated.** The specification names the
parameter, so the decorator takes it. **There is no GPSS dictionary in this project** —
`docs/remediation/00_seam_map.md` §B3 established that on 2026-08-07 and resolved the
vocabulary question in favour of `concept_code`, derived at import time from
`extraction_schemas/*.yaml`, explicitly declining to name a column after a standard not
in use so that no future reader would assume authority behind it. Populating
`gpss_version` would recreate exactly that shadow vocabulary. A test asserts the field
stays empty until a real dictionary exists. If GPSS is later adopted, `concept_code` is
the column it maps into.

**D6 — the audit sink is pluggable, and the default is not the database.** The spec says
every evaluation writes a record; write amplification says a synchronous INSERT per
evaluation would be a denial-of-service against the audit table rather than an audit
trail (12,408 invoices are 77M candidate pairs before blocking). Every evaluation *does*
write a record; where it goes is the sink's business. Default is a bounded in-process
ring buffer plus a debug log line; `DbAuditSink` writes to `proc.bp_agent_actions`;
`NullAuditSink` is available for a hot path that has opted out deliberately.

**D7 — the confidence ladder is `OBSERVED > ASSERTED > UNVERIFIED`.** A result is capped
at the weakest confidence among its inputs, so "LLM-sourced inputs cap the result at
ASSERTED" falls out of the ordering rather than needing a special case. Unstated
provenance yields `UNVERIFIED`, not `OBSERVED`: silence is not evidence.

## Consequences

**Good.** 47 formulas were named, versioned and contract-checked by this change (a
concurrent workstream has since added three more, which is the layer being used as
intended); their golden vectors run on every import and every commit; the audit spine
can answer "what produced this number"; the model inventory is generated rather than
maintained; five previously invisible defects are now pinned, documented and impossible
to fix by accident. `docs/model-inventory.md` carries the live counts.

**Costs.** A caller that ignores `UNASSESSED` now raises. Registration is import-time
work (~10 ms). `evaluate` adds a validation pass and a record per call; on the migrated
batch paths that is one record per sweep rather than per item, but it is not free.
`_body_source` hashing means a docstring edit changes the version hash — deliberate: a
comment claiming the maths does X when it does Y is a real defect, and the hash noticing
is a feature, not noise.

**Not free of judgement.** Two thirds of the registered formulas have no call site
outside their own module yet — they are registered and pinned, but their callers still
call the underlying function directly. `model-inventory.md` names every one of them at
the top, regenerated from the registry, so the remaining work stays visible rather than
being assumed done.

**A shared failure mode.** One golden vector failing takes down the import for every
consumer of the registry, not just the module that owns it. That happened once during
delivery, when a concurrent workstream committed a formula mid-edit. The blast radius is
the price of the gate being real; it is worth knowing before adding the fiftieth
formula.

## Deliberately deferred

Each item below is out of scope **and** has a stated trigger. The trigger matters more
than the item: "we might need it later" is how a registry becomes a framework.

| Deferred | Why not now | Build it when |
|---|---|---|
| **DB-backed catalogue** (`proc.bp_formula`) | Code is already the source of truth, and it is reviewed, diffed and versioned by git. A DB row is a second source that can disagree with the code, which is the shadow-vocabulary failure in another costume. | Someone outside this repo needs to enumerate the formulas without reading it — a compliance export, a second service, or a UI that lists models. Read-only projection first; never a second definition. |
| **Resolver with applicability rules** | Every call site today knows exactly which formula it wants. A resolver would be indirection with nothing on the other side. | Two formulas legitimately compete for one call site — e.g. a category-specific benchmark alongside the general one — and the choice depends on data rather than on the caller. |
| **Tenant overrides** | **There is no tenant dimension in this schema.** Not deferred by preference; there is nothing to key an override on. (Recorded independently in `project_ask_path_authorization`.) | A tenant column exists on the data these formulas read. At that point RLS on any persisted formula metadata becomes a prerequisite, not a follow-up. |
| **Expression language** | Formulas here are Python: `excel_round`, MAD, log-odds fusion, DataFrame normalisation. An expression language would be strictly less capable and would need its own evaluator, its own tests and its own security story. | A non-engineer must author a formula without a deploy. Note the real prerequisite is not the language but the approval workflow below. |
| **Approval workflow** | Formula changes go through code review today, which is an approval workflow with better tooling than anything built here would have. | Formulas become editable outside git — i.e. only after the catalogue and the expression language, and never before. |
| **Contract-driven unit conversion** | The declared unit is metadata plus an enforced range; a bare float cannot be checked, only a `Quantity` can. For the scale collisions that actually occur here (0–1 vs 0–100), the range *is* the unit check, and it works. | A formula needs to accept the same quantity in two units — genuinely mixed-currency inputs, say — rather than refusing the mismatch. |
| **Migrating all 61 inventory formulas** | 47 were registered by this change; the rest are one-line helpers or are embedded inside detector bodies with their own guards. Extracting them changes numbers. | Each is extracted with its own before/after snapshot, as `supplier.composite_score` and `opportunity.finding_weight_factor` were. |

## Schema migration (proposed, NOT run)

The evaluation record currently rides in `proc.bp_agent_actions.details` (jsonb). That
works and needs no migration. It is not, however, queryable: "every evaluation of
`benchmark.adjusted_price` at version 1.0.0 in April" is a jsonb scan.

**Nothing below has been run.** It requires explicit approval.

```sql
-- Proposed. NOT APPLIED. Additive: no existing column changes, no data moves.
CREATE TABLE IF NOT EXISTS proc.bp_formula_evaluation (
    evaluation_id   bigserial PRIMARY KEY,
    created_at      timestamptz  NOT NULL DEFAULT now(),
    evaluated_at    timestamptz  NOT NULL,      -- transaction time
    as_of           timestamptz  NULL,          -- valid time, when the formula takes one
    formula_name    text         NOT NULL,
    formula_version text         NOT NULL,
    version_hash    text         NOT NULL,      -- source + contract + vectors
    status          text         NOT NULL,      -- 'ok' | 'unassessed'
    confidence      text         NULL,          -- observed | asserted | unverified
    batch_size      integer      NOT NULL DEFAULT 1,
    inputs          jsonb        NOT NULL,
    output          jsonb        NULL,
    provenance_ids  text[]       NOT NULL DEFAULT '{}',
    findings        jsonb        NOT NULL DEFAULT '[]',
    duration_ms     numeric      NULL,
    trace_id        text         NULL,
    deal_id         text         NULL,
    document_id     text         NULL
);

CREATE INDEX IF NOT EXISTS ix_bp_formula_evaluation_name_version
    ON proc.bp_formula_evaluation (formula_name, formula_version, evaluated_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_formula_evaluation_deal
    ON proc.bp_formula_evaluation (deal_id) WHERE deal_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_bp_formula_evaluation_unassessed
    ON proc.bp_formula_evaluation (formula_name, evaluated_at DESC)
    WHERE status = 'unassessed';
```

Notes on the proposal:

- `bp_` prefix and `ix_bp_*` index names follow the existing convention.
- `evaluated_at` and `as_of` are separate columns on purpose: transaction time and valid
  time are different questions, and conflating them is what made
  `PredictiveRiskModel.evaluate` unreproducible in the first place.
- **No RLS clause.** There is no tenant dimension to key one on. If a tenant column is
  ever added to the data these formulas read, RLS on this table is a prerequisite of that
  change, not a follow-up to it.
- Retention is unaddressed. A per-evaluation table under a pairwise scorer grows fast;
  a partitioning or retention policy should land in the same change, not after it.

## Alternatives considered

**Extend `linking_engine.PROFILES` instead of building a new registry.** That table is a
genuine registry — `register_profile` / `register_signal`, four profiles across three
modules — and it was the strongest prior art here. Rejected because it registers *signal
profiles for one scoring model*, not formulas in general: it has no place for a version,
a contract, an owner or a vector, and 50 of the 61 inventory formulas do not fit its
shape at all. The new registry wraps it (`linking.relationship_confidence` delegates to
`score_link`) rather than competing with it.

**Wait for `worktree-conformance-phase1` to merge and extend its `detector_registry`.**
That branch maps detector slugs to handler methods. Rejected because it registers
*which detector runs*, and this registry versions *the maths inside* one. They compose;
the new package touches no file that branch modifies, so neither conflicts with the
other.

**Copy each formula's maths into its definition module.** Rejected outright. A second
copy of the arithmetic is a second thing to keep right, which is the problem this layer
exists to solve. Every registered formula delegates to the live implementation, so there
is exactly one place each number is computed.
