# Baseline: what AgentNick's extraction accuracy actually is

**Date:** 2026-09-26
**Produced by:** `scripts/build_truth_set.py`
**Spec:** `specs/2026-09-26-honest-measurement-design.md`

## The number

```
examples 311   from corpus 175   recovered 133   unrecoverable 3   malformed 0

accuracy 69.4%   coverage 89.7%
  verified 8,678   unsupported 3,826   unverifiable 1,434
```

**69.4% accuracy at 89.7% coverage**, replacing a recorded 0.847.

## How this differs from the 0.847 it replaces

The old figure scored AgentNick against `auto_collected_examples.jsonl` — 311
examples harvested from **its own prior outputs**, kept only where its own
confidence was 0.90 or above. It measured whether the model agreed with itself,
on the subset where it had already been sure. A model that reproduced every one
of its past mistakes would have scored 1.0.

The new figure checks each extracted value against the text of the document it
came from. A field is `verified` when the value is present, `unsupported` when
the document is available and the value is not in it, and `unverifiable` when
there is nothing to check against — no source text, or a field the pipeline
computes rather than reads. Only the first two count toward accuracy.

Coverage is published beside accuracy and is not optional. Accuracy alone can be
driven to 1.0 by making more fields unverifiable, so a score without its
coverage says nothing.

## The finding worth acting on

| doc_type | accuracy | coverage |
| --- | --- | --- |
| `Invoice` | 56.2% | 75.7% |
| `Purchase_Order` | 64.4% | 96.3% |
| `Quote` | 61.1% | 98.0% |
| `invoice` | 76.2% | 89.8% |
| `purchase_order` | 78.4% | 92.4% |
| `quote` | 77.2% | 89.3% |

The same three document types appear twice, in two casings, because two
collection paths write this file. That was visible before as untidiness. It is
not untidiness.

**The capitalised batch is 15–20 points less accurate than the lowercase one.**
It is also the batch that stored no `source_text` — all 136 examples with an
empty source came from it. Until this run those examples scored as agreement
with themselves, so the weaker half of the corpus was the invisible half.

Which path produces the capitalised rows, and why it is less accurate, is the
next thing to find out. The two are not interchangeable and should stop being
averaged together.

## Recovery

133 of the 136 examples with no stored source text were recovered by fetching
the document and re-parsing it. Three could not be, and remain `unverifiable`
rather than dropped.

The first run recovered **zero**, while the S3 download and the PDF conversion
both succeeded — `ParsedDocument` exposes `full_text`, and the recovery helper
read `text`, which is always `None`. The failure was silent in exactly the way
this whole sub-project exists to prevent, and is now pinned by
`test_default_recover_reads_full_text_not_text`.

## Against the prediction

The design committed in advance to roughly 175 verifiable, roughly 136
unverifiable, and an accuracy meaningfully below 0.847, with the note that a
score at or above 0.847 should be read as a bug rather than good news.

Accuracy came in at 69.4%, comfortably below. The unverifiable estimate was
pessimistic: recovery reclaimed 133 of the 136, so coverage reached 89.7% rather
than the ~56% implied. The prediction held where it mattered.

## What this does not measure

- **Fields the document does not contain as text.** 1,434 field instances are
  unverifiable — derived values, and the 3 unrecoverable documents. Grounding
  cannot judge these; a human or a rule can.
- **Whether a verified value is in the right field.** A supplier name that
  appears in the document verifies even if it was filed as the buyer. This
  check answers "is this value on the page", not "is this the right value for
  this field". The known supplier/buyer swap would pass it.
- **Fields the model omitted.** Absence is recorded, not scored as an error.

Each is a reason the true accuracy is **lower** than 69.4%, not higher.
