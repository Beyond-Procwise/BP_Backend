# Contract link signals — design

Date: 2026-10-08. Status: awaiting review. Follows `2026-10-02-contract-structures-design.md`.

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

`contract_links.propose_parent_links` already knows each child's role through the vocabulary
(`_is_variation`). It will pick the profile by role:

| Child role | Profile | Child types |
|---|---|---|
| `role.variation` | `contract_amendment` (new) | variation, addendum, CCN |
| `role.attachment` | `contract_attachment` (new) | schedule, SLA |
| everything else with a default parent | `contract_hierarchy` (existing, extended) | SOW, call-off, order form |

### 4.1 contract_hierarchy (existing; amended)

Existing five signals unchanged. Added:

| Signal | Tier / weight | Cluster | Reads | Rule |
|---|---|---|---|---|
| `buyer` | 2 / 3 | `identity` (with `supplier`) | `buyer_org_id` | equal after normalisation = OK; both present and different = CONFLICT; either absent = MISSING |
| `start_order` | 1 / 5, cap 0.45 | `temporal` | `contract_start_date` | child starts on/after parent start = OK; before = CONFLICT (an impossible date: the prompt's "child effective before parent exists"); absent = MISSING |
| `value_rollup` | 3 / 2 | `commercial` | `total_contract_value`, `currency` | same currency and child <= parent = OK; child > parent = CONFLICT; currency differs or either value absent = MISSING (never converted) |
| `currency` | 3 / 2 | `terms` | `currency` | equal = OK; differ = CONFLICT |
| `payment_terms` | 3 / 2 | `terms` | `payment_terms` | normalised equal = OK; differ = CONFLICT |
| `governing_law` | 3 / 2 | `terms` | `governing_law`, `jurisdiction` | equal = OK; differ = CONFLICT |
| `signatory` | 3 / 2 | `people` | `contract_signatory_name`, `buyer_signatory_name` | any named signatory shared across the pair = OK; both sides have names and none shared = WEAK (0.4, not CONFLICT: signatories legitimately change) |
| `cost_centre` | 3 / 2 | `category` | `cost_centre_id`, `business_unit_id`, `spend_category` | any shared = OK; both present and none shared = CONFLICT |

`start_order` and `term_containment` both read `contract_start_date`; `remap_clusters` merges
them, so a date fact is not counted twice. `currency` is read by both `value_rollup` and
`currency`; they merge too, by the same mechanism.

### 4.2 contract_amendment (new)

The amendment amends a document; it does not repeat its title. So: **no `title_overlap`**, and no
`value_rollup` (an amendment legitimately raises value). Signals: `declared_reference` (1/5),
`supplier` (1/5), `buyer` (2/3), `start_order` (1/5), `term_containment` (2/3), `currency`,
`payment_terms`, `governing_law` (3/2, `terms`), `signatory` (3/2). `expected_structure` stays but
reads MISSING for a variation (it amends any structure), as today.

Candidate parents for this profile are unchanged: every contract-family structure that is not
itself a variation.

### 4.3 contract_attachment (new)

Schedule and SLA attach to a parent agreement. Signals: `declared_reference` (1/5), `supplier`
(1/5), `buyer` (2/3), `term_containment` (2/3), `title_overlap` (2/3, retained: an SLA for a
named service shares that service's words), `currency`, `payment_terms`, `governing_law`
(3/2, `terms`). Parent types: master agreement, service agreement, framework agreement, SOW,
call-off, consulting agreement.

## 5. Vocabulary migration

`2026-10-08_contract_link_vocabulary.sql` + rollback, additive and idempotent, applied to
`bp_testdb` by the implementer and to `bp_sqldb` by the controller after review.

- `doctype.schedule`, `doctype.sla`: `default_parent_type` set so `is_child` is true.
- `doctype.termination_notice`: **left out** of child scoring. A termination notice names the
  agreement it ends, which is a wording signal (out of scope); proposing it a parent on supplier
  and dates alone would be wrong.
- New types **`doctype.dpa`, `doctype.side_letter`, `doctype.renewal`, `doctype.guaranty`** inserted
  with `status = 'proposed'`. Under the vocabulary's existing rule a proposed type never resolves,
  so they classify nothing until Nick confirms each.

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

- **Weights are declared.** A wrong weight changes which band a real document lands in. Mitigation:
  nothing auto-links; every result is a human-confirmed proposal; bands stay at the shared 92/80/65/45.
- **`start_order` at tier 1** caps a score at 0.45 on a child that starts before its parent. Real
  contracts are sometimes signed after their start date (back-dating). Mitigation: it only caps
  to review/weak, never blocks; flagged for Nick as ruling A below.
- **More signals with MISSING data.** Corpus fields such as `payment_terms` may be mostly empty;
  MISSING signals lower coverage `C`. This is measured on `bp_testdb` before and after (section 7).

## 9. Rulings requested

A. Should a child that starts before its parent be a tier-1 conflict (cap 0.45) or tier 2 (cap 0.70)?
   Recommendation: tier 1, as the prompt treats an impossible date as grounds for review.
B. Confirm termination notice stays out of child scoring until the wording signals exist.
C. Confirm the four new document types go in as `proposed`.
