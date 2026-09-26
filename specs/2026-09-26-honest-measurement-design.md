# Honest measurement for AgentNick

**Status:** design, awaiting review
**Date:** 2026-09-26
**Scope:** sub-project 1 of 5 in the AgentNick orchestration programme

---

## Why this exists

The evaluation corpus, `src/data/training/auto_collected_examples.jsonl`, is 311
examples harvested by `process_monitor_watcher.py` from **AgentNick's own past
outputs**. Three properties make it unusable as a measure of accuracy:

- **Confidence range is 0.90–1.00.** Only high-confidence outputs were kept, so
  the set excludes by construction every case the model found hard.
- **136 of 311 (44%) have an empty `source_text`.** Those cannot be checked
  against the document they came from.
- **No field records human verification.** Nothing anywhere says a person
  confirmed a value.

So the eval gate scores the model against its own prior answers. A model that
reproduces its past mistakes perfectly scores 1.0. The 0.847 figure carried in
project notes is a self-agreement number, not an accuracy one.

This matters beyond the number. Fine-tuning on this corpus would train the model
to be more confidently itself — the same failure already recorded against the
grounding-rejection logs, which are poisoned negatives because they record what
the guard refused rather than what was wrong. Same shape, different file.

Nothing else in the programme — judge loop, tone work, orchestration, training —
can be shown to help until there is something true to measure against.

## What this is not

Not a runtime change. Nothing here alters what the extraction pipeline does to a
live document. It builds a labelled set and a score, beside the pipeline.

Not fine-tuning. That is sub-project 5, and it stays blocked until this one
produces a corpus worth training on.

## The core decision: verification has three outcomes

`src/services/extraction_v3/grounding.py::is_value_grounded` returns a boolean.
Its documented behaviour includes *"the document is unavailable (cannot verify →
do not block)"* — it returns `True`.

For a runtime guard that is the right default. A missing PDF should not block a
user's extraction. **For a labeller it is fatal**: "I could not check" becomes
"it is correct", which is exactly how a corpus comes to say everything is fine.

The labeller is therefore a separate component with three verdicts:

| Verdict | Meaning | Counts toward accuracy |
| --- | --- | --- |
| `verified` | The source text is available and the value appears in it | Yes, as correct |
| `unsupported` | The source text is available and the value does **not** appear in it | Yes, as wrong |
| `unverifiable` | No source text, or the field cannot be grounded by construction | **No — excluded and reported separately** |

`unsupported` rather than `contradicted`: the labeller can tell that a value is
absent from the document, which is what matters. Deciding that the document says
something *different* means knowing which span should have held the field, and
that is a second, harder judgement. Claiming it would overstate what the check
actually knows.

Every score is published as a pair: **accuracy and coverage**. Accuracy is over
verifiable fields only; coverage is the fraction of fields that were verifiable.
A score without its coverage is meaningless, because it can be raised to 1.0 by
making more fields unverifiable.

### Why not reuse the existing rule

`is_value_grounded` treats the digit-signature of a value (≥3 digits appearing
anywhere in the document) as grounding. Project notes already record this as
unsafe for sentences. For labelling, an invoice total of `1234.56` must not be
called verified because `123` appears in a postcode. The labeller requires the
full normalised value, anchored at a token boundary.

## Components

Four units, each usable and testable alone.

### `src/services/truth/verify.py`

```
verify_field(field: str, value, source_text: str | None) -> Verdict
```

Pure. No I/O, no database, no model. One field, one source text, one verdict
carrying the matched span when there is one, and a reason when there is not.

Normalisation before comparison, and each rule stated rather than assumed:

- currency symbols and thousand separators stripped from numbers
- dates compared against every rendering the source might use (the document says
  `15/03/2024`; the extraction says `2024-03-15`)
- whitespace collapsed, case folded for text
- match must be anchored at a token boundary, never a bare substring

### `src/services/truth/derived.py`

The list of fields that cannot be grounded because the document does not contain
them as text: `tax_percent` where only an amount is printed, totals the pipeline
computes, `converted_amount_usd`, anything reformatted rather than copied.

This list is a **domain judgement, not a technical one**, and is the part of this
design most in need of review. A field wrongly on it is excused from measurement
forever; a field wrongly off it is reported as a failure the model cannot avoid.

### `src/services/truth/build_set.py`

Walks the corpus and writes `src/data/training/verified_examples.jsonl`: per
field, the verdict, the matched span, and the rule that decided it. Provenance
is recorded so any label can be argued with.

Where `source_text` is empty it attempts recovery from the original document in
S3, and records whether recovery succeeded. It never invents source text and
never drops an example — an unrecoverable one is retained as `unverifiable`, so
the coverage figure stays honest.

### `src/services/truth/baseline.py`

Scores a model tag against the labelled set. Prints accuracy and coverage per
document type and per field, so a regression can be located rather than merely
noticed.

## Data flow

```
auto_collected_examples.jsonl ─┐
                               ├─→ build_set ─→ verified_examples.jsonl ─→ baseline ─→ accuracy + coverage
S3 document text (recovery) ───┘        │
                                        ├─ verify.py  (one field → verdict)
                                        └─ derived.py (is this groundable at all?)
```

## Error handling

The rule throughout: **a failure to check is never a pass.**

- Source text unavailable → `unverifiable`, never `verified`.
- S3 unreachable → the example stays `unverifiable` and the run reports how many
  were affected; it does not fail, and it does not quietly shrink the set.
- A field absent from the extraction is not the same as a wrong one; absence is
  reported in its own column rather than scored as an error.
- Malformed JSON in a corpus row is reported with its line number and skipped,
  with the skip count in the summary.

## Making the training stub honest

`src/training/pipeline.py::_train_model` imports the training libraries, logs
`"Training model %s"`, creates the output directory and returns it **having
trained nothing**. `_merge_adapters` and `_convert_gguf` follow the same shape.
A caller receives a path that looks like a successful result.

They will raise `NotImplementedError` naming what is missing. Real LoRA training
is sub-project 5; until then the honest behaviour is to refuse rather than to
return an empty adapter that reads as success.

## Testing

The labeller is tested against cases where the answer is known independently:

- a value present verbatim → `verified`
- a value absent from the source → `unsupported`
- a value present **only as a digit substring** of something else → must be
  `unsupported`, not `verified` (this is the existing rule's failure, written
  down as a test)
- a date in the source as `15/03/2024`, extracted as `2024-03-15` → `verified`
- no source text → `unverifiable`, and never `verified`
- a derived field with no source text → `unverifiable`, by the derived list

Then the test that matters, over the real 311: **the baseline must move.** If
excluding unverifiable fields from the numerator does not change the score, the
labeller is not doing anything and the run should be treated as failed.

## Success criteria

1. Every field in the corpus carries a verdict and the rule that produced it.
2. Accuracy is never reported without coverage beside it.
3. The recorded baseline is replaced by an accuracy-and-coverage pair, and the
   difference between the two is explained.
4. `_train_model` refuses instead of returning a directory.
5. A person can take any single label and see why it was decided.

## Expected outcome, stated in advance so it can be wrong

Roughly 175 examples verifiable and 136 unverifiable before S3 recovery, and a
real accuracy meaningfully below 0.847. If the figure comes out at or above
0.847 the labeller is almost certainly still counting unverifiable fields as
correct, and that is a bug rather than good news.

## Open question for review

The derived-field list in `derived.py`. Which fields does the business consider
legitimately underivable from document text — and is `tax_percent` one of them
when the document prints only a tax amount and a subtotal?
