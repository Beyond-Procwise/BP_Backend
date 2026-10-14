# Contract link signals — design

Date: 2026-10-08. Status: approved; revision 2 after an integration review (section 10). Follows `2026-10-02-contract-structures-design.md`.

## 1. Why

`contract_hierarchy` scores a contract document against its candidate parent on five signals:
declared reference, expected parent type, supplier, term containment, title overlap. A live test
matrix on `bp_testdb` (2026-10-08, six child types x six situations) found:

- SOW, call-off and order form behave correctly in every situation.
- A variation, addendum or CCN is **not proposed a parent** when its title does not repeat the
  parent's words, even when its reference resolves to the parent, the supplier is the same and
  its term sits inside the parent's. "Addendum No. 1" against "Statement of Work Helix Migration"
  scores 36.0 (title read as CONFLICT); a long CCN title scores 62.9 against the 65.0 review floor.
- Only the supplier side of the parties is compared; the buyer is ignored.
- Schedule, SLA and termination notice are never proposed a parent.

The reference prompt Nick supplied ranks signals **by child type**: for an amendment, text overlap
is the *weakest* signal (its rank 7). The scorer weighs every child type identically. That is the
gap. This spec adds signals and per-role weighting **inside the existing architecture**.

## 2. Architecture rules this design obeys

1. **`linking_engine.py` is not modified.** Every golden vector for `invoice_po`, `quote_po` and
   the five graph profiles stays byte-identical. New behaviour is new *profile entries* and new
   *signal comparators* registered with `register_signal` / `register_profile`, exactly as
   `contract_succession` and `contract_coverage` do.
2. **A signal is a spec row**: `id, cluster, tier, weight, appl, cap, kind, reads`. Tier 1 = weight 5,
   cap 0.45; tier 2 = weight 3, cap 0.70; tier 3 = weight 2, cap 0.90 (the values the five existing
   profiles already use). The reference prompt's Tier 1/2/3 map to these. Its Tier 4/5 (file-name
   stems, envelope ids, embedding similarity, template likeness) are **not scored**: the prompt's
   own rule is "never link on these alone", and nothing here may reach auto-link anyway.
3. **Correlated evidence is merged by `reads`.** Two signals that read a common field are unioned
   into one cluster by `remap_clusters` and dampened by the engine's existing `_dampen`
   (x0.85 for two active signals, x0.70 for three or more). Signals that are correlated but read
   different fields (currency, payment terms, governing law all come from one template) are given
   the same *declared* cluster so the same dampening applies.
4. **Declared unmeasured.** All weights are declared, not fitted: no labelled sample of true
   contract parent links exists (every stored `parent_contract_id` dangles). Every new profile is
   added to `UNCALIBRATED_PROFILES` in `edge_writer.py`, so none can reach `auto_link`. This spec
   claims no accuracy figure.
5. **Propose-only is unchanged.** Results are written by `contract_links.propose_parent_links` to
   `proc.bp_extraction_discrepancy`; only `confirm()` sets `parent_contract_id`.
6. **No fabrication.** A value that is absent is `MISSING` (score 0.5, reliability 0, no effect).
   Money is compared only in one currency; no FX rate is ever invented.
7. **Vocabulary lives in `proc.bp_document_type`** (columns `role`, `default_parent_type`,
   `status`, `requires_parent_evidence`...). Structure changes are additive, idempotent migration
   pairs applied to **both** databases, never edits to a shipped migration.

## 3. Scope

**In (this spec):** per-role profiles; buyer, start-order, value, commercial-terms, signatory and
cost-centre signals; vocabulary changes; link type recorded on each proposal; tests.

**Out, deliberately:**
- **Wording signals** ("supersedes", "amends and restates", "incorporated into", section-level
  references, project/PO codes). They need new extraction fields in `contract.yaml` (regex primary
  per the extraction direction), new columns, and a re-read of stored contracts. A second spec.
- **Renewal.** `contract_succession` already scores renewals (adjacent terms, same supplier).
  Not rebuilt here.
- **Amendment sequence** (No. 1, 2, 3 ordering). A variation is currently never a parent of another
  variation (`_wanted_parent_types`); sequence needs sibling reasoning that does not exist yet.
- **Calibration.** Needs a labelled sample; none exists.

## 4. Profiles

> **Revision 1 (2026-10-08, found while planning).** Measured against the real engine, adding eight
> signals the obvious way *regresses* today's results: coverage `C` divides by the weight of every
> signal in the profile whether or not the data exists, so an exact-reference SOW falls 96.9 -> 75.7
> and a same-supplier SOW with no reference falls 75.6 -> 61.4, under the 65.0 floor, so it would
> stop being proposed at all. Three design changes follow; they supersede the first draft.
>
> 1. **Applicability per pair.** An optional signal joins a pair's profile only when *both*
>    documents carry the data it reads. This is the engine's own applicability idea (`appl`, `q`)
>    applied per pair; the engine is untouched. Absent data costs nothing. Variant profiles are
>    registered lazily under a derived name and the result reports the base profile name.
> 2. **Corroborators add; they do not subtract.** Where a field legitimately differs between a
>    parent and child (payment terms, signatories, cost centres) a mismatch is *neutral* (s = 0.5,
>    status WEAK), never CONFLICT. Only currency, governing law, buyer and value (where the child
>    exceeds the parent) can conflict, at tier 2/3 with no score cap.
> 3. **`start_order` is dropped.** The prompt's "child effective before parent exists" means
>    before the parent was *executed*; no execution-date column exists, and the existing
>    `term_containment` deliberately grades an early start WEAK (signature lag). Ruling A is
>    therefore moot until an execution date is extracted (wording-signals spec).
>
> Also: `expected_structure` is left out of the amendment and attachment profiles. It is always
> MISSING for them by design, and a signal that cannot be observed only lowers coverage.
> `payment_terms` is left out of the amendment profile: changing terms is what an amendment does.

Profile selection by child role (read from the vocabulary, as `_is_variation` already does):

| Child role | Profile | Child types |
|---|---|---|
| `role.variation` | `contract_amendment` (new) | variation, addendum, CCN |
| `role.attachment` | `contract_attachment` (new) | schedule, SLA |
| anything else with a default parent | `contract_hierarchy` (existing, extended) | SOW, call-off, order form |

Optional signals (joined only when applicable, per revision 1):

| Signal | Tier / weight / cap | Cluster | Reads | OK | Differs |
|---|---|---|---|---|---|
| `buyer` | 2 / 3 / 0.70 | `identity` | `buyer_org_id` | equal 1.0 | CONFLICT 0.0 |
| `value_rollup` | 3 / 2 / 0.90 | `commercial` | `total_contract_value`, `currency` | child <= parent: neutral 0.5 WEAK | child > parent: CONFLICT; different currency: not applicable |
| `currency` | 3 / 2 / 0.90 | `terms` | `currency` | equal 1.0 | CONFLICT |
| `payment_terms` | 3 / 2 / 0.90 | `terms` | `payment_terms` | equal 1.0 | neutral 0.5 WEAK |
| `governing_law` | 3 / 2 / 0.90 | `terms` | `governing_law`, `jurisdiction` | equal 1.0 | CONFLICT |
| `signatory` | 3 / 2 / 0.90 | `people` | `contract_signatory_name`, `buyer_signatory_name` | any shared name 1.0 | neutral 0.5 WEAK |
| `cost_centre` | 3 / 2 / 0.90 | `category` | `cost_centre_id`, `business_unit_id`, `spend_category` | any shared 1.0 | neutral 0.5 WEAK |

A value that fits under its parent's is *necessary, not sufficient* (a small value fits under any
parent), hence neutral rather than positive. `value_rollup` and `currency` both read `currency`, so
`remap_clusters` merges them automatically.

Profiles:

- **contract_hierarchy** - existing five signals unchanged, plus `buyer`, `value_rollup`,
  `currency`, `payment_terms`, `governing_law`, `signatory`, `cost_centre`. With no optional data
  present a pair scores exactly as today (96.9 with a reference, 75.6 without).
- **contract_amendment** - `declared_reference`, `supplier`, `term_containment`; optional `buyer`,
  `currency`, `governing_law`, `signatory`. No title, no structure, no value, no payment terms.
  Measured: a resolving reference + same supplier + contained term + a generic title scores 65.9
  (review), 78.4 with a buyer match. With no reference it scores ~20-33: an amendment is identified
  by what it amends, and supplier alone cannot say which contract that is. Intended.
- **contract_attachment** - `declared_reference`, `supplier`, `term_containment`, `title_overlap`;
  optional `buyer`, `currency`, `payment_terms`, `governing_law`. Parent types: any contract-family
  structure whose role is `role.master` or `role.framework`. Chosen in code, **not** by setting
  `default_parent_type`: that column holds one value, a schedule sits under several kinds of
  agreement, and the column is also read by the upload gate and pinned by a seed-drift test.

## 5. Vocabulary migration

`2026-10-08_contract_link_vocabulary.sql` + rollback, additive and idempotent, applied to
`bp_testdb` by the implementer and to `bp_sqldb` by the controller after review. `seed.py` is
edited in lockstep (the seed-vs-table drift test compares them column for column, aliases as an
ordered list).

- **No change to `schedule`, `sla` or `termination_notice`** (see section 4: attachment parents
  are chosen in code). Termination notice stays out of child scoring: it names the agreement it
  ends, which is a wording signal (out of scope).
- New types **`doctype.dpa`, `doctype.side_letter`, `doctype.renewal`, `doctype.guaranty`**, inserted
  `status = 'proposed'` with no pipeline, like `doctype.policy_document`. A proposed type never
  resolves or routes, so they classify nothing until Nick confirms each.

## 6. Link type on each proposal

The proposal notes gain a link type derived from the child's profile: `child_of` (hierarchy),
`amends` (amendment), `attaches_to` (attachment). It is text in the existing notes and in a
`link_type` key on the returned `details`; no new table and no schema change.

## 7. Testing

- The 2026-10-08 matrix becomes `tests/services/test_contract_link_matrix.py` (live-DB gated by
  `PROCWISE_TEST_LIVE_DB=1`, like the existing contract tests), with realistic titles added.
- Every new comparator has a pure unit test (OK / CONFLICT / MISSING), and **every guard is shown
  failing** by breaking the production code and watching the test go red.
- Engine golden vectors run unchanged and must stay byte-identical.
- The two defects found: an amendment with a resolving reference, same supplier and contained term
  but a generic title is now proposed (score >= 65); an SLA is proposed its master agreement.
- A scoped regression proves a variation's former wrong-title case no longer fails.

## 8. Risks

- **Weights are declared.** A wrong weight changes which band a real document lands in. Nothing
  auto-links; every result is a human-confirmed proposal; bands stay at the shared 92/80/65/45.
- **Variant profile names** (`contract_hierarchy+buyer+currency...`) are registered at run time,
  at most 2^7 per profile. The result carries the base `profile` name, and the base names are in
  `UNCALIBRATED_PROFILES`.
- **Corroborators can lift a no-reference pair into the warning band.** Several agreeing optional
  signals could move a same-supplier SOW from 75.6 to the 80+ band. It is still a proposal a
  person confirms; the matrix test records the observed ceiling.

## 9. Rulings

A. (moot, see revision 1.3) start-before-parent as a conflict: dropped until an execution date exists.
B. Termination notice stays out of child scoring: **assumed yes**.
C. The four new document types go in as `proposed`: **assumed yes**.

## 10. Revision 2: integration with the rest of the product

Checked against the code on 2026-10-08, after Nick asked whether the design accounts for the
current maths, agents and wider capabilities.

**What the consumers actually do**

- `contract_links.confirm()` writes `parent_contract_id` for every link type, as approved on
  2026-10-02. Its main reader, triage (`triage/loader.py`), loads the whole contract FAMILY
  through it: the seed contract's parent and every child pointing at it. A SOW, amendment or
  attachment under its agreement is exactly what that needs, so the column is a family pointer,
  not an "amends" pointer. Only a comment (`triage/model.py:105`) said otherwise; it is
  corrected.
- Confirmations are already audited: `DecisionEngine._accept_contract_parent` writes a
  `proc.bp_decision` row through `_record_human_action`.
- No agent reads the contract hierarchy today. The opportunity critic reads
  `parent_contract_id` as a plain field.
- The hierarchy profile has never produced graph edges (`pass_runner.PASS_ORDER` runs supplier
  identity, item equivalence, succession and coverage only). Unchanged by this design.
- `scripts/graph_resolution/calibrate.py` tunes a profile's `p0`/`alpha` from labelled pairs,
  choosing the best separation with zero false auto-links. Confirmed and dismissed proposals are
  such labels; the corpus has about one proposal so far, so there is nothing to fit yet.

**Three changes**

1. **Variants follow their base profile.** `applicability.score_pair` re-reads the base
   profile's current `p0`/`alpha`/`floor` and re-registers the variant on every call. The first
   draft cached a copy at first use, so a recalibrated base would have left every pair with an
   optional signal scoring on stale values.
2. **The two proposal thresholds are governed.** `MIN_SCORE` (65) and `SEPARATION` (8) move from
   constants to `promotion_thresholds.contract_parent_min_score` / `contract_parent_separation` in
   `proc.bp_policy`, beside the product's other link thresholds. Values unchanged. A missing key
   RAISES, so the migration must reach both databases before the code ships.
3. **The confirmed link type is audited.** `child_of` / `amends` / `attaches_to` is written into
   the confirmation's `proc.bp_decision.facts` and returned by the action. No new column: no
   reader needs it yet, and the record now exists for the first one that does (precedence: an
   amendment's terms override, an attachment's do not).

**Still out, until something needs them:** graph edges for confirmed parent links; a dedicated
link-type column; a calibration loop fed by confirmations; the reference prompt as an AI judge
for review-band pairs (second spec, with the wording signals).
