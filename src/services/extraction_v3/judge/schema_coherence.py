"""Schema-coherence judge — final pass on the assembled record.

Asks the LLM "does this look like a coherent {invoice|PO|quote|contract}?"
Returns verdict 'coherent' | 'incoherent' + optional issue list. The
verdict is advisory: orchestrator demotes record confidence on 'incoherent'
but does not mutate the record.

Catches cross-field consistency issues that single-field substring checks
miss — e.g. supplier_name and invoice_id from different vendors,
nonsensical date ordering across fields.
"""
from __future__ import annotations
import json
import logging
from src.services.ollama_client import ollama_generate
from src.services.extraction_v3.judge.contracts import (
    CoherenceInput, CoherenceOutput, CoherenceIssue, InvariantResultSummary,
)

log = logging.getLogger(__name__)


def _build_prompt(input_obj: CoherenceInput) -> str:
    inv_block = "\n".join(
        f"  - {r.name}: {'PASS' if r.passed else 'FAIL'}"
        + (f" — {r.message}" if r.message else "")
        for r in input_obj.invariant_results
    )
    rec_json = json.dumps(input_obj.extracted_record, indent=2, default=str)
    return f"""You are a procurement-document extraction judge.

Reviewing an assembled {input_obj.doc_type} record. Tell me whether all
fields look INTERNALLY consistent — does this look like a real, coherent
{input_obj.doc_type}, or do some fields look like they came from a different
document or are nonsensical together?

Examples of incoherent records:
- supplier_name says one company but invoice_id format strongly suggests another
- requested_by is a person whose name doesn't appear elsewhere in the record
- dates that don't make sense (invoice_date after due_date by years)
- amounts that don't add up (caught by invariants below — also flag if you see something the invariants missed)

Record:
{rec_json}

Invariant results:
{inv_block or "  (no invariants ran)"}

Respond with ONLY a single JSON object:
{{"verdict": "coherent" | "incoherent", "issues": [{{"field": "<name>", "issue": "<one sentence>"}}, ...]}}

If everything looks consistent, return verdict="coherent" and issues=[].
"""


def _parse_response(raw: str) -> CoherenceOutput | None:
    if not raw:
        return None
    start = raw.find("{")
    if start < 0:
        return None
    depth, end, in_str, esc = 0, -1, False, False
    for i in range(start, len(raw)):
        c = raw[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        data = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None
    try:
        return CoherenceOutput(**data)
    except Exception:
        return None


def call_coherence_judge(
    doc_type: str,
    extracted_record: dict,
    invariant_results: list[InvariantResultSummary] | None = None,
) -> CoherenceOutput | None:
    """Run the schema-coherence judge. Returns CoherenceOutput or None on failure.

    Verdict is ADVISORY — caller decides what to do with 'incoherent'.
    """
    if not extracted_record:
        return None

    input_obj = CoherenceInput(
        doc_type=doc_type,
        extracted_record=extracted_record,
        invariant_results=invariant_results or [],
    )
    prompt = _build_prompt(input_obj)

    raw = ollama_generate(prompt, num_predict=512, temperature=0.0, retries=1, timeout=30)
    if raw is None:
        return None

    return _parse_response(raw)
