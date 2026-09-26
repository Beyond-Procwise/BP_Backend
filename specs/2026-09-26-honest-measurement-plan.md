# Honest Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace a self-agreement score with a real accuracy figure, by labelling each extracted field as verified, unsupported or unverifiable against the document it came from.

**Architecture:** Four small units under `src/services/truth/`. `verify.py` decides one field against one source text and is pure. `derived.py` says which fields cannot be grounded at all. `build_set.py` walks the corpus, recovers missing source text from S3, and writes a labelled set with per-field provenance. `baseline.py` scores a model tag and always prints coverage beside accuracy. Nothing in the live extraction path changes.

**Tech Stack:** Python 3.12, pytest, existing `src/services/extraction/parser.py` for document text, `src/services/egress.py` for the S3 client.

**Spec:** `specs/2026-09-26-honest-measurement-design.md`

## Global Constraints

- Local only. No third-party cloud APIs beyond the project's own S3 bucket.
- A failure to check is never a pass. Unavailable source → `unverifiable`, never `verified`.
- Accuracy is never reported without coverage beside it.
- No value is ever invented to fill a gap. An absent field is recorded absent.
- Run tests with `./venv/bin/python -m pytest`, with `CUDA_VISIBLE_DEVICES=""` and `OLLAMA_HOST=127.0.0.1:1` so no test reaches the GPU or a live model.
- New tables, if any, take the `bp_` prefix. This plan adds none.

## Review Focus

Five input classes the spec implies but that no obvious happy-path test exercises. Each has its test assigned to the task that owns the code.

1. **A number that appears only inside a longer number.** Extracted `1234.56`, document contains `91234.567`. Substring matching says verified; it must say unsupported. (Task 1)
2. **European decimal notation in the source.** Document prints `1.234,56`, extraction holds `1234.56`. Must verify, or every EU invoice reads as a hallucination. (Task 1)
3. **Zero and empty string are values, not absences.** `tax_amount: 0` on a zero-rated invoice must be checked, not skipped as missing. (Task 1)
4. **A date rendered differently.** Source `15/03/2024`, extraction `2024-03-15`. Must verify. (Task 1)
5. **Source text present but empty after normalisation** (a scanned page with no text layer). Must be `unverifiable`, not `unsupported` — the model may well be right and there is nothing to check against. (Task 3)

---

### Task 1: Field verification, three-valued

**Files:**
- Create: `src/services/truth/__init__.py`
- Create: `src/services/truth/verify.py`
- Test: `tests/services/truth/test_verify.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Verdict` (a frozen dataclass with `outcome: str`, `rule: str`, `span: str | None`), and `verify_field(field: str, value: Any, source_text: str | None) -> Verdict`. `outcome` is one of `"verified"`, `"unsupported"`, `"unverifiable"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/services/truth/test_verify.py
import pytest
from src.services.truth.verify import verify_field


def test_a_value_present_verbatim_is_verified():
    v = verify_field("invoice_id", "INV-2024-001", "Invoice No: INV-2024-001\nDate: 01/02/2024")
    assert v.outcome == "verified"


def test_a_value_absent_from_the_source_is_unsupported():
    v = verify_field("invoice_id", "INV-9999", "Invoice No: INV-2024-001")
    assert v.outcome == "unsupported"


def test_no_source_text_is_unverifiable_never_verified():
    for empty in (None, "", "   "):
        v = verify_field("invoice_id", "INV-2024-001", empty)
        assert v.outcome == "unverifiable", empty


def test_a_number_inside_a_longer_number_is_not_verified():
    # Review Focus 1. Substring matching calls this verified; it is not.
    v = verify_field("invoice_amount", 1234.56, "Total due 91234.567 after adjustment")
    assert v.outcome == "unsupported"


def test_european_decimal_notation_verifies():
    # Review Focus 2.
    v = verify_field("invoice_amount", 1234.56, "Gesamtbetrag: 1.234,56 EUR")
    assert v.outcome == "verified"


def test_thousand_separators_and_currency_symbols_verify():
    v = verify_field("invoice_amount", 2400.0, "Subtotal: £2,400.00")
    assert v.outcome == "verified"


def test_zero_is_a_value_and_is_checked():
    # Review Focus 3.
    present = verify_field("tax_amount", 0, "VAT (0%): 0.00")
    assert present.outcome == "verified"
    absent = verify_field("tax_amount", 0, "VAT (20%): 480.00")
    assert absent.outcome == "unsupported"


def test_a_date_in_another_rendering_verifies():
    # Review Focus 4.
    v = verify_field("invoice_date", "2024-03-15", "Date: 15/03/2024")
    assert v.outcome == "verified"


def test_the_verdict_says_which_rule_decided_it():
    v = verify_field("invoice_id", "INV-2024-001", "Invoice No: INV-2024-001")
    assert v.rule and isinstance(v.rule, str)
    assert v.span == "INV-2024-001"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:1 PYTHONPATH=src ./venv/bin/python -m pytest tests/services/truth/test_verify.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.truth'`

- [ ] **Step 3: Write the implementation**

```python
# src/services/truth/verify.py
"""Does this extracted value actually appear in the document it came from?

Three outcomes, not two. `src/services/extraction_v3/grounding.py` answers this
question with a bool and returns True when the document is unavailable -- right
for a runtime guard that must not block a user over a missing PDF, fatal here,
where "could not check" would silently become "correct" and inflate every score
built on it.

It also treats a >=3 digit signature appearing anywhere as grounding. This does
not: an invoice total of 1234.56 is not verified by a postcode containing 123.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

VERIFIED = "verified"
UNSUPPORTED = "unsupported"
UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class Verdict:
    outcome: str
    rule: str
    span: Optional[str] = None


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _candidate_numbers(text: str) -> set[str]:
    """Every number in the text, normalised to a plain decimal string.

    Handles both 1,234.56 and the European 1.234,56 -- the separator that
    appears LAST is the decimal point.
    """
    out: set[str] = set()
    for raw in re.findall(r"\d[\d.,]*\d|\d", text):
        last_comma, last_dot = raw.rfind(","), raw.rfind(".")
        if last_comma > last_dot:
            plain = raw.replace(".", "").replace(",", ".")
        else:
            plain = raw.replace(",", "")
        try:
            out.add(f"{float(plain):.4f}")
        except ValueError:
            continue
    return out


def _date_renderings(value: str) -> set[str]:
    try:
        d = datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return set()
    return {
        d.strftime(fmt)
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y",
                    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y")
    }


def verify_field(field: str, value: Any, source_text: Optional[str]) -> Verdict:
    if value is None:
        return Verdict(UNVERIFIABLE, "value-absent")
    if not source_text or not source_text.strip():
        return Verdict(UNVERIFIABLE, "no-source-text")

    haystack = _norm_text(source_text)

    # Numbers compare as numbers. A textual match would miss 2,400.00 == 2400.0
    # and would accept 1234.56 inside 91234.567.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        target = f"{float(value):.4f}"
        if target in _candidate_numbers(source_text):
            return Verdict(VERIFIED, "number-match", str(value))
        return Verdict(UNSUPPORTED, "number-absent")

    text = str(value).strip()
    if not text:
        return Verdict(UNVERIFIABLE, "value-empty")

    # A bare number arriving as a string is still a number.
    if re.fullmatch(r"-?\d[\d.,]*", text):
        try:
            target = f"{float(text.replace(',', '')):.4f}"
            if target in _candidate_numbers(source_text):
                return Verdict(VERIFIED, "number-match", text)
            return Verdict(UNSUPPORTED, "number-absent")
        except ValueError:
            pass

    for rendering in _date_renderings(text):
        if _norm_text(rendering) in haystack:
            return Verdict(VERIFIED, "date-rendering", rendering)

    needle = _norm_text(text)
    # Anchored at a token boundary: "Ltd" must not verify inside "Ultditch".
    if re.search(rf"(?<![\w]){re.escape(needle)}(?![\w])", haystack):
        return Verdict(VERIFIED, "text-match", text)

    return Verdict(UNSUPPORTED, "text-absent")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:1 PYTHONPATH=src ./venv/bin/python -m pytest tests/services/truth/test_verify.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add src/services/truth/ tests/services/truth/
git commit -o src/services/truth/__init__.py src/services/truth/verify.py tests/services/truth/test_verify.py \
  -m "feat(truth): field verification with three outcomes, not two"
```

---

### Task 2: The derived-field list

**Files:**
- Create: `src/services/truth/derived.py`
- Test: `tests/services/truth/test_derived.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `is_derived(field: str) -> bool`, and `DERIVED_FIELDS: frozenset[str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/services/truth/test_derived.py
from src.services.truth.derived import is_derived, DERIVED_FIELDS


def test_a_field_the_pipeline_computes_is_derived():
    assert is_derived("converted_amount_usd")
    assert is_derived("exchange_rate_to_usd")


def test_a_field_copied_from_the_page_is_not_derived():
    assert not is_derived("invoice_id")
    assert not is_derived("supplier_id")
    assert not is_derived("invoice_amount")


def test_unknown_fields_are_not_derived():
    # Defaulting to derived would excuse every new field from measurement.
    assert not is_derived("some_field_added_next_year")


def test_the_list_is_explicit_and_reviewable():
    assert isinstance(DERIVED_FIELDS, frozenset)
    assert DERIVED_FIELDS, "an empty list means nothing is excused -- state it deliberately"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:1 PYTHONPATH=src ./venv/bin/python -m pytest tests/services/truth/test_derived.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/services/truth/derived.py
"""Fields the document does not contain as text, so grounding cannot judge them.

A field wrongly ON this list is excused from measurement forever. A field wrongly
OFF it is reported as a failure the model had no way to avoid. So the list is
short, explicit, and defaults to NOT derived: a field nobody has classified is
measured, which fails loudly rather than quietly.

tax_percent is deliberately NOT here. Where a document prints "VAT (20%)" the
figure is on the page and should be verified. Where it prints only an amount and
a subtotal, verification will report it unsupported -- which is the honest
answer, and the signal that the pipeline computed it rather than read it.
OPEN FOR REVIEW: the business may rule otherwise.
"""
from __future__ import annotations

DERIVED_FIELDS: frozenset[str] = frozenset({
    # Currency conversion happens in the pipeline against an FX table.
    "converted_amount_usd",
    "exchange_rate_to_usd",
    # Identity and bookkeeping the pipeline assigns.
    "deal_id",
    "deal_name",
    "document_id",
    "created_date",
    "created_by",
    "last_modified_by",
    "last_modified_date",
    "confidence_score",
    "accuracy_score",
    # Routing/classification decided by the pipeline, not printed on the page.
    "doc_type",
    "region",
})


def is_derived(field: str) -> bool:
    return field in DERIVED_FIELDS
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:1 PYTHONPATH=src ./venv/bin/python -m pytest tests/services/truth/test_derived.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add src/services/truth/derived.py tests/services/truth/test_derived.py
git commit -o src/services/truth/derived.py tests/services/truth/test_derived.py \
  -m "feat(truth): name the fields grounding cannot judge, and default to measuring"
```

---

### Task 3: Label one extraction record

**Files:**
- Create: `src/services/truth/label.py`
- Test: `tests/services/truth/test_label.py`

**Interfaces:**
- Consumes: `verify_field`, `Verdict` from Task 1; `is_derived` from Task 2.
- Produces: `label_record(extracted: dict, source_text: str | None) -> dict` returning `{"fields": {field: {"value":…, "outcome":…, "rule":…}}, "counts": {"verified": int, "unsupported": int, "unverifiable": int}}`. Header and line-item fields are both walked; a line-item field is keyed `line_items[i].field`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/services/truth/test_label.py
from src.services.truth.label import label_record


SRC = "Invoice No: INV-2024-001\nFrom: TechNova Ltd\nSubtotal: £2,400.00\nDate: 15/03/2024"


def test_header_fields_are_each_given_a_verdict():
    rec = {"header": {"invoice_id": "INV-2024-001", "supplier_id": "TechNova Ltd",
                      "invoice_amount": 2400.00, "invoice_date": "2024-03-15"}}
    out = label_record(rec, SRC)
    assert out["counts"]["verified"] == 4
    assert out["fields"]["invoice_id"]["outcome"] == "verified"


def test_a_derived_field_is_unverifiable_even_with_source_text():
    rec = {"header": {"converted_amount_usd": 3100.00}}
    out = label_record(rec, SRC)
    assert out["fields"]["converted_amount_usd"]["outcome"] == "unverifiable"
    assert out["fields"]["converted_amount_usd"]["rule"] == "derived-field"


def test_a_hallucinated_value_is_unsupported():
    rec = {"header": {"supplier_id": "Nonexistent Holdings Ltd"}}
    out = label_record(rec, SRC)
    assert out["fields"]["supplier_id"]["outcome"] == "unsupported"


def test_line_item_fields_are_keyed_by_index():
    rec = {"header": {}, "line_items": [{"item_description": "Laptop Dell XPS 15"}]}
    out = label_record(rec, "Laptop Dell XPS 15  2  £1,200.00")
    assert out["fields"]["line_items[0].item_description"]["outcome"] == "verified"


def test_source_text_with_no_words_is_unverifiable_not_unsupported():
    # Review Focus 5: a scanned page with no text layer. The model may well be
    # right; there is simply nothing to check against.
    rec = {"header": {"invoice_id": "INV-2024-001"}}
    out = label_record(rec, "   \n\n \t ")
    assert out["fields"]["invoice_id"]["outcome"] == "unverifiable"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:1 PYTHONPATH=src ./venv/bin/python -m pytest tests/services/truth/test_label.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/services/truth/label.py
"""One extraction record in, one verdict per field out."""
from __future__ import annotations

from typing import Any, Optional

from src.services.truth.derived import is_derived
from src.services.truth.verify import UNVERIFIABLE, Verdict, verify_field


def _judge(field: str, value: Any, source_text: Optional[str]) -> Verdict:
    if is_derived(field):
        return Verdict(UNVERIFIABLE, "derived-field")
    return verify_field(field, value, source_text)


def label_record(extracted: dict, source_text: Optional[str]) -> dict:
    fields: dict[str, dict] = {}

    for field, value in (extracted.get("header") or {}).items():
        v = _judge(field, value, source_text)
        fields[field] = {"value": value, "outcome": v.outcome, "rule": v.rule}

    for i, line in enumerate(extracted.get("line_items") or []):
        if not isinstance(line, dict):
            continue
        for field, value in line.items():
            v = _judge(field, value, source_text)
            fields[f"line_items[{i}].{field}"] = {
                "value": value, "outcome": v.outcome, "rule": v.rule}

    counts = {"verified": 0, "unsupported": 0, "unverifiable": 0}
    for f in fields.values():
        counts[f["outcome"]] = counts.get(f["outcome"], 0) + 1
    return {"fields": fields, "counts": counts}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:1 PYTHONPATH=src ./venv/bin/python -m pytest tests/services/truth/test_label.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add src/services/truth/label.py tests/services/truth/test_label.py
git commit -o src/services/truth/label.py tests/services/truth/test_label.py \
  -m "feat(truth): label every field of an extraction, line items included"
```

---

### Task 4: Build the labelled set

**Files:**
- Create: `src/services/truth/build_set.py`
- Test: `tests/services/truth/test_build_set.py`

**Interfaces:**
- Consumes: `label_record` from Task 3.
- Produces: `build(corpus_path: str, out_path: str, recover: Callable[[str], str | None] | None = None) -> dict` returning a summary `{"examples": int, "with_source": int, "recovered": int, "unrecoverable": int, "counts": {...}}`. `recover` takes a `file_path` and returns document text or `None`; injected so tests never touch S3.

- [ ] **Step 1: Write the failing tests**

```python
# tests/services/truth/test_build_set.py
import json
from src.services.truth.build_set import build


def _corpus(tmp_path, rows):
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def test_examples_without_source_are_recovered_when_possible(tmp_path):
    rows = [{"doc_type": "Invoice", "pk": 1, "file_path": "documents/a.pdf",
             "source_text": "", "extracted": {"header": {"invoice_id": "INV-1"}}}]
    out = tmp_path / "out.jsonl"
    summary = build(_corpus(tmp_path, rows), str(out),
                    recover=lambda fp: "Invoice No: INV-1")
    assert summary["recovered"] == 1
    assert summary["counts"]["verified"] == 1


def test_an_unrecoverable_example_is_kept_as_unverifiable_not_dropped(tmp_path):
    rows = [{"doc_type": "Invoice", "pk": 2, "file_path": "documents/b.pdf",
             "source_text": "", "extracted": {"header": {"invoice_id": "INV-2"}}}]
    out = tmp_path / "out.jsonl"
    summary = build(_corpus(tmp_path, rows), str(out), recover=lambda fp: None)
    assert summary["examples"] == 1, "an example must never be silently dropped"
    assert summary["unrecoverable"] == 1
    assert summary["counts"]["unverifiable"] == 1


def test_a_malformed_row_is_reported_and_skipped(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text('{"broken": ')
    out = tmp_path / "out.jsonl"
    summary = build(str(p), str(out), recover=lambda fp: None)
    assert summary["malformed"] == 1
    assert summary["examples"] == 0


def test_every_labelled_row_carries_its_provenance(tmp_path):
    rows = [{"doc_type": "Invoice", "pk": 3, "file_path": "documents/c.pdf",
             "source_text": "Invoice No: INV-3", "extracted": {"header": {"invoice_id": "INV-3"}}}]
    out = tmp_path / "out.jsonl"
    build(_corpus(tmp_path, rows), str(out), recover=lambda fp: None)
    row = json.loads(out.read_text().strip())
    assert row["pk"] == 3
    assert row["fields"]["invoice_id"]["rule"] == "text-match"
    assert row["source"] == "corpus"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:1 PYTHONPATH=src ./venv/bin/python -m pytest tests/services/truth/test_build_set.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/services/truth/build_set.py
"""Walk the harvested corpus and write a labelled set with per-field provenance.

Never drops an example. One that cannot be checked is kept and counted as
unverifiable, because a set that quietly shrinks to the checkable cases is how
coverage goes unnoticed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

from src.services.truth.label import label_record

logger = logging.getLogger(__name__)

Recover = Callable[[str], Optional[str]]


def _default_recover(file_path: str) -> Optional[str]:
    """Fetch the document's text from S3. Imported lazily so the builder stays
    testable, and importable, on a machine with no AWS credentials."""
    try:
        from src.services.extraction import parser
        parsed = parser.parse(file_path)
        return getattr(parsed, "text", None)
    except Exception as exc:  # noqa: BLE001 - recovery is best effort by design
        logger.info("could not recover source for %s: %s", file_path, exc)
        return None


def build(corpus_path: str, out_path: str, recover: Optional[Recover] = None) -> dict:
    recover = recover or _default_recover
    summary = {"examples": 0, "with_source": 0, "recovered": 0,
               "unrecoverable": 0, "malformed": 0,
               "counts": {"verified": 0, "unsupported": 0, "unverifiable": 0}}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, encoding="utf-8") as fh, open(out_path, "w", encoding="utf-8") as out:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                summary["malformed"] += 1
                logger.warning("%s:%d is not JSON: %s", corpus_path, lineno, exc)
                continue

            source = (row.get("source_text") or "").strip()
            origin = "corpus"
            if source:
                summary["with_source"] += 1
            else:
                source = (recover(row.get("file_path") or "") or "").strip()
                if source:
                    summary["recovered"] += 1
                    origin = "recovered"
                else:
                    summary["unrecoverable"] += 1
                    origin = "none"

            labelled = label_record(row.get("extracted") or {}, source or None)
            summary["examples"] += 1
            for k, v in labelled["counts"].items():
                summary["counts"][k] = summary["counts"].get(k, 0) + v

            out.write(json.dumps({
                "pk": row.get("pk"),
                "doc_type": row.get("doc_type"),
                "file_path": row.get("file_path"),
                "source": origin,
                "fields": labelled["fields"],
                "counts": labelled["counts"],
            }) + "\n")
    return summary
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:1 PYTHONPATH=src ./venv/bin/python -m pytest tests/services/truth/test_build_set.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add src/services/truth/build_set.py tests/services/truth/test_build_set.py
git commit -o src/services/truth/build_set.py tests/services/truth/test_build_set.py \
  -m "feat(truth): build the labelled set, keeping what cannot be checked"
```

---

### Task 5: Accuracy with coverage beside it

**Files:**
- Create: `src/services/truth/baseline.py`
- Test: `tests/services/truth/test_baseline.py`

**Interfaces:**
- Consumes: the labelled set written by Task 4.
- Produces: `score(labelled_path: str) -> dict` returning `{"accuracy": float|None, "coverage": float, "verified": int, "unsupported": int, "unverifiable": int, "by_doc_type": {...}}`, and `format_report(result: dict) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/services/truth/test_baseline.py
import json
import pytest
from src.services.truth.baseline import score, format_report


def _labelled(tmp_path, rows):
    p = tmp_path / "labelled.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def _row(doc_type, outcomes):
    fields = {f"f{i}": {"value": i, "outcome": o, "rule": "x"}
              for i, o in enumerate(outcomes)}
    counts = {"verified": outcomes.count("verified"),
              "unsupported": outcomes.count("unsupported"),
              "unverifiable": outcomes.count("unverifiable")}
    return {"pk": 1, "doc_type": doc_type, "fields": fields, "counts": counts}


def test_accuracy_counts_only_verifiable_fields(tmp_path):
    rows = [_row("Invoice", ["verified", "verified", "unsupported", "unverifiable"])]
    r = score(_labelled(tmp_path, rows))
    assert r["accuracy"] == pytest.approx(2 / 3)
    assert r["coverage"] == pytest.approx(3 / 4)


def test_unverifiable_fields_never_count_as_correct(tmp_path):
    # The whole point: a set that cannot be checked does not score 1.0.
    rows = [_row("Invoice", ["unverifiable"] * 10)]
    r = score(_labelled(tmp_path, rows))
    assert r["accuracy"] is None, "no verifiable field means no accuracy, not perfect accuracy"
    assert r["coverage"] == 0.0


def test_the_report_always_shows_coverage_next_to_accuracy(tmp_path):
    rows = [_row("Invoice", ["verified", "unverifiable"])]
    text = format_report(score(_labelled(tmp_path, rows)))
    assert "accuracy" in text.lower() and "coverage" in text.lower()


def test_results_are_broken_down_by_document_type(tmp_path):
    rows = [_row("Invoice", ["verified"]), _row("Quote", ["unsupported"])]
    r = score(_labelled(tmp_path, rows))
    assert r["by_doc_type"]["Invoice"]["accuracy"] == 1.0
    assert r["by_doc_type"]["Quote"]["accuracy"] == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:1 PYTHONPATH=src ./venv/bin/python -m pytest tests/services/truth/test_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/services/truth/baseline.py
"""Accuracy over what could be checked, and how much that was.

Accuracy alone can be driven to 1.0 by making more fields unverifiable, so it is
never returned or printed without its coverage. Where nothing was verifiable the
accuracy is None -- not 1.0, and not 0.0, both of which are claims this data
cannot support.
"""
from __future__ import annotations

import json
from collections import defaultdict


def _acc(verified: int, unsupported: int):
    checked = verified + unsupported
    return (verified / checked) if checked else None


def score(labelled_path: str) -> dict:
    tally = {"verified": 0, "unsupported": 0, "unverifiable": 0}
    per_type: dict[str, dict] = defaultdict(
        lambda: {"verified": 0, "unsupported": 0, "unverifiable": 0})

    with open(labelled_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            counts = row.get("counts") or {}
            dt = row.get("doc_type") or "unknown"
            for k in tally:
                tally[k] += int(counts.get(k, 0))
                per_type[dt][k] += int(counts.get(k, 0))

    total = sum(tally.values())
    return {
        "accuracy": _acc(tally["verified"], tally["unsupported"]),
        "coverage": ((tally["verified"] + tally["unsupported"]) / total) if total else 0.0,
        **tally,
        "by_doc_type": {
            dt: {"accuracy": _acc(c["verified"], c["unsupported"]),
                 "coverage": ((c["verified"] + c["unsupported"]) / sum(c.values()))
                             if sum(c.values()) else 0.0,
                 **c}
            for dt, c in sorted(per_type.items())
        },
    }


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def format_report(result: dict) -> str:
    lines = [
        f"accuracy {_pct(result['accuracy'])}  coverage {_pct(result['coverage'])}"
        f"   (verified {result['verified']}, unsupported {result['unsupported']},"
        f" unverifiable {result['unverifiable']})",
        "",
        f"{'document type':20} {'accuracy':>10} {'coverage':>10}",
    ]
    for dt, c in result["by_doc_type"].items():
        lines.append(f"{dt:20} {_pct(c['accuracy']):>10} {_pct(c['coverage']):>10}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:1 PYTHONPATH=src ./venv/bin/python -m pytest tests/services/truth/test_baseline.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add src/services/truth/baseline.py tests/services/truth/test_baseline.py
git commit -o src/services/truth/baseline.py tests/services/truth/test_baseline.py \
  -m "feat(truth): report accuracy and coverage together, or not at all"
```

---

### Task 6: Stop the training pipeline reporting success it did not earn

**Files:**
- Modify: `src/training/pipeline.py` (`_train_model`, `_merge_adapters`, `_convert_gguf`)
- Test: `tests/training/test_pipeline_refuses.py`

**Interfaces:**
- Consumes: nothing.
- Produces: no new callable. The three functions raise `NotImplementedError` instead of returning a path.

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_pipeline_refuses.py
import pytest
from src.training.pipeline import _train_model, _merge_adapters


def test_training_refuses_rather_than_returning_an_empty_adapter(tmp_path):
    """It logged "Training model", made a directory and returned it, having
    trained nothing. A caller could ship that as a fine-tuned model."""
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"a": 1}\n')

    class Cfg:
        use_unsloth = False
        base_model = "AgentNick:extract"
        output_dir = tmp_path / "out"

    with pytest.raises(NotImplementedError, match="not implemented"):
        _train_model(Cfg(), dataset)


def test_merging_adapters_refuses_too(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    class Cfg:
        adapter_path = adapter
        output_dir = tmp_path / "merged"

    with pytest.raises(NotImplementedError, match="not implemented"):
        _merge_adapters(Cfg())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:1 PYTHONPATH=src ./venv/bin/python -m pytest tests/training/test_pipeline_refuses.py -v`
Expected: FAIL — `DID NOT RAISE NotImplementedError` (the functions currently return a path)

- [ ] **Step 3: Change the three functions to refuse**

Replace the body of `_train_model` after its dataset check with:

```python
    raise NotImplementedError(
        "LoRA fine-tuning is not implemented. This function used to log "
        "'Training model', create the output directory and return it, so a "
        "caller received a path that looked like a trained adapter and was the "
        "base model. Refusing is the honest answer until training is real. See "
        "specs/2026-09-26-honest-measurement-design.md."
    )
```

Replace the body of `_merge_adapters` after its adapter-path check with:

```python
    raise NotImplementedError(
        "Adapter merging is not implemented; it created an output directory and "
        "returned it without merging anything."
    )
```

Replace the body of `_convert_gguf` with the same shape:

```python
    raise NotImplementedError(
        "GGUF conversion is not implemented; it reported a path it never wrote."
    )
```

- [ ] **Step 4: Run the test to verify it passes, and check nothing else depended on the lie**

Run: `CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:1 PYTHONPATH=src ./venv/bin/python -m pytest tests/training/ tests/test_agent_endpoints.py -v`
Expected: the two new tests PASS. If an existing test fails because it relied on the old return value, that test was asserting the fake succeeded — read it, and fix the test rather than restoring the behaviour.

- [ ] **Step 5: Commit**

```bash
git add src/training/pipeline.py tests/training/test_pipeline_refuses.py
git commit -o src/training/pipeline.py tests/training/test_pipeline_refuses.py \
  -m "fix(training): refuse rather than report a fine-tune that never happened"
```

---

### Task 7: Run it on the real corpus and record the baseline

**Files:**
- Create: `scripts/build_truth_set.py`
- Create: `specs/2026-09-26-baseline-result.md`

**Interfaces:**
- Consumes: `build` (Task 4), `score` and `format_report` (Task 5).
- Produces: `src/data/training/verified_examples.jsonl` and a written baseline.

- [ ] **Step 1: Write the runner**

```python
# scripts/build_truth_set.py
"""Label the harvested corpus and print the baseline.

Read-only against S3; writes only the labelled set.
"""
import sys
from src.services.truth.build_set import build
from src.services.truth.baseline import score, format_report

CORPUS = "src/data/training/auto_collected_examples.jsonl"
OUT = "src/data/training/verified_examples.jsonl"


def main() -> int:
    summary = build(CORPUS, OUT)
    print(f"examples {summary['examples']}  from corpus {summary['with_source']}"
          f"  recovered {summary['recovered']}  unrecoverable {summary['unrecoverable']}"
          f"  malformed {summary['malformed']}")
    print()
    print(format_report(score(OUT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=src ./venv/bin/python scripts/build_truth_set.py`
Expected: 311 examples, roughly 175 from corpus, and an accuracy figure **below 0.847**.

- [ ] **Step 3: Check the prediction, and treat a high score as a bug**

If accuracy comes out at or above 0.847, stop. The spec says this is almost
certainly unverifiable fields still counting as correct. Re-read
`baseline.score` and confirm `unverifiable` is excluded from the denominator
before recording anything.

- [ ] **Step 4: Write the result down**

Create `specs/2026-09-26-baseline-result.md` recording: the accuracy, the
coverage, the per-document-type table, how many examples needed recovery and how
many were unrecoverable, and one sentence on how this differs from the 0.847
figure it replaces.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_truth_set.py specs/2026-09-26-baseline-result.md
git commit -o scripts/build_truth_set.py specs/2026-09-26-baseline-result.md \
  -m "feat(truth): record a baseline that measures accuracy rather than self-agreement"
```

> The labelled set itself (`src/data/training/verified_examples.jsonl`) is a
> build artifact of a corpus that will be regenerated. Do not commit it.
