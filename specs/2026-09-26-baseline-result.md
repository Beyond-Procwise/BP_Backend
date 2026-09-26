# Baseline: what AgentNick's extraction accuracy actually is

**Date:** 2026-09-26 (revised after independent review)
**Produced by:** `scripts/build_truth_set.py`
**Spec:** `specs/2026-09-26-honest-measurement-design.md`

## The number

```
examples 311   from corpus 175   recovered 133   unrecoverable 3   malformed 0

accuracy 79.2%   coverage 65.1%
  verified 7,191   unsupported 1,883   unverifiable 4,864
  fields emitted 13,938
```

**79.2% accuracy at 65.1% coverage**, replacing a recorded 0.847.

An earlier revision of this document reported 69.4% at 89.7%. An independent
review found errors running in both directions; the corrections and what each
was worth are below. **The first figure was wrong and should not be quoted.**

## How this differs from the 0.847 it replaces

The old figure scored AgentNick against `auto_collected_examples.jsonl` — 311
examples harvested from **its own prior outputs**, kept only where its own
confidence was 0.90 or above. It measured whether the model agreed with itself,
on the subset where it had already been sure. A model that reproduced every one
of its past mistakes would have scored 1.0.

The new figure checks each extracted value against the text of the document it
came from: `verified` when the value is present, `unsupported` when the document
is available and the value is not in it, `unverifiable` when there is nothing to
check against. Only the first two count toward accuracy.

Three numbers are published together and none is optional:

- **accuracy** — of what could be checked, how much was right;
- **coverage** — how much could be checked at all. Accuracy alone can be driven
  to 1.0 by making more fields unverifiable;
- **fields emitted** — neither of the above can see a field the model never
  produced, so a model answering only where it is confident would otherwise
  score 100% on both.

## What the review corrected

**Verification of small numbers carried almost no evidence.** A negative control
— substituting an arbitrary *wrong* quantity into real documents — verified a
fabricated quantity of 5 against 95.5% of them, because every page contains a 5
somewhere. That rule produced 55% of all verified fields. A bare whole number
under 1,000, matched without a decimal point or thousands separator, is now
`unverifiable` rather than `verified`. This is the spec's own principle applied
to the checker: a match that carries no evidence is not a verification.

**Comma-grouped thousands were mis-parsed and millions silently dropped.**
`"1,234"` parsed as 1.234, and `"1,234,567"` raised and was swallowed. 158 real
money values were scored wrong while printed plainly on the page. Negative
values never matched at all.

**Pipeline-minted surrogate keys were scored as extraction errors.** No document
contains `SUP-THRIVESTUDIOSLLC` or `QTE-2026-00487-1`; the pipeline constructs
them. They were 1,235 of 3,826 `unsupported` verdicts — 32% of every "wrong"
answer — blaming the model for values it could not have read. Now derived.
**This means supplier correctness is no longer measured here at all**: whether
the right supplier was identified is a resolution question answered against
`proc.bp_supplier`, not a grounding one.

**Currency codes were failed against documents printing the symbol.** `GBP`
against a page showing `£` — about 200 false negatives.

**`region` was wrongly excused.** It holds `West Sussex` in this corpus, an
address component printed on the page. 280 field instances are now measured.

Net effect: accuracy 69.4% → 79.2% as false negatives were removed; coverage
89.7% → 65.1% as coincidental matches stopped counting as verifications.

## The document-type gap, and why it cannot yet be acted on

| doc_type | accuracy | coverage |
| --- | --- | --- |
| `Invoice` | 68.6% | 58.5% |
| `Purchase_Order` | 74.3% | 73.7% |
| `Quote` | 62.5% | 69.2% |
| `invoice` | 88.9% | 63.3% |
| `purchase_order` | 90.4% | 66.1% |
| `quote` | 89.6% | 63.5% |

The capitalised batch scores 15–25 points lower. An earlier revision of this
document attributed that to the collection path that produces capitalised
`doc_type`, and recommended investigating it.

**That conclusion was not available from this data and has been withdrawn.** The
two variables are perfectly collinear: every capitalised row is a row whose
source text was *re-derived by today's parser*, and every lowercase row is one
whose source text was *captured at extraction time*. "That path extracts worse"
and "re-parsed text differs from the text the model actually saw" are
indistinguishable here. The batches also emit different field sets — sixteen
fields appear only in the capitalised one — so part of the gap is composition
rather than quality.

Separating them needs source text captured at extraction time for both paths.
Until then the gap is a question, not a finding.

## What this still does not measure

- **Whether a verified value is in the right field.** A supplier name present on
  the page verifies even if it was filed as the buyer. This answers "is this
  value on the page", not "is this the right value for this field". The known
  supplier/buyer swap would pass.
- **Fields the model omitted.** Counted as emitted-vs-not, never as an error.
- **Anything derived.** 4,864 field instances, now including the surrogate keys.

The first of these means the true accuracy is **lower** than 79.2%. The others
are simply outside what grounding can answer.
