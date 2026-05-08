"""Grounded last-resort judge — anti-hallucination guarantee.

Fires only for required fields where Layer-2 produced zero candidates.
The judge MUST return a value that is a verbatim substring of doc_full_text;
the orchestrator post-validates this.

Critical safety contract: a value not satisfying ``value == evidence_text``
AND ``evidence_text in doc_full_text`` is REJECTED. There is no fallback.
If the LLM cannot cite, the field stays NULL → review queue.

Model name note: the returned Candidate uses ``model="qa_roberta"`` because
``ExtractorName`` is a closed Literal and the grounded judge is semantically
the closest to an extractive QA model (it extracts a verbatim span from the
document, which is exactly what a QA model does). Adding "judge_grounded"
to the Literal would touch the schemas and break the Candidate type — deferred
to a later plan task.
"""
from __future__ import annotations
import json
import logging
from src.services.ollama_client import ollama_generate
from src.services.extraction_v3.judge.contracts import (
    GroundedInput, GroundedOutput, GroundedConstraints,
)
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.yaml_schema.loader import FieldSpec

log = logging.getLogger(__name__)

MAX_VALUE_LENGTH = 64
MAX_DOC_TEXT_FOR_PROMPT = 16000  # cap how much full_text we send to keep prompt size bounded


def _build_prompt(input_obj: GroundedInput) -> str:
    """Construct the strict grounded prompt."""
    labels = ", ".join(f'"{l}"' for l in input_obj.field_canonical_labels[:5])
    return f"""You are a procurement-document extraction judge.

Field to extract: "{input_obj.field}" (type: {input_obj.field_type})
The field is typically labeled as: {labels}

The deterministic extractors did not find this field. Look at the document
text below and find the value, but you MUST quote it verbatim — your "value"
and "evidence_text" must BOTH be a substring of the document text. If you
cannot find the value as a literal substring, return null for both.

Document text:
\"\"\"{input_obj.doc_full_text[:MAX_DOC_TEXT_FOR_PROMPT]}\"\"\"

Constraints:
- value and evidence_text must be IDENTICAL strings
- both must appear as a verbatim substring of the document text above
- maximum length: {input_obj.constraints.max_length} characters
- if the value is not present in the text, return value=null and evidence_text=null

Respond with ONLY a single JSON object on one line, no prose:
{{"value": <string or null>, "evidence_text": <string or null>, "rationale": "<one sentence>"}}
"""


def _parse_response(raw: str) -> GroundedOutput | None:
    """Extract first {{...}} from raw, validate as GroundedOutput."""
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
        return GroundedOutput(**data)
    except Exception:
        return None


def call_grounded_last_resort(
    field: FieldSpec,
    doc_full_text: str,
) -> Candidate | None:
    """Run the grounded last-resort LLM call. Returns a Candidate or None.

    SAFETY: The returned Candidate's value is GUARANTEED to be a substring of
    doc_full_text. If the LLM violates this, the function returns None.
    No exceptions, no fallbacks — null on any deviation.
    """
    if not doc_full_text or not doc_full_text.strip():
        return None

    input_obj = GroundedInput(
        field=field.name,
        field_type=field.type,
        field_canonical_labels=field.canonical_labels or [field.name],
        doc_full_text=doc_full_text,
        constraints=GroundedConstraints(
            must_be_verbatim_substring_of_doc_full_text=True,
            max_length=MAX_VALUE_LENGTH,
        ),
    )
    prompt = _build_prompt(input_obj)

    raw = ollama_generate(prompt, num_predict=512, temperature=0.0, retries=1, timeout=20)
    if raw is None:
        return None

    parsed = _parse_response(raw)
    if parsed is None:
        return None

    # Null or empty → return None
    if parsed.is_null():
        return None
    if not parsed.value or not parsed.evidence_text:
        return None

    # SAFETY CHECK 1: value must equal evidence_text
    if parsed.value.strip() != parsed.evidence_text.strip():
        log.warning(
            "grounded judge: value != evidence_text (value=%r, evidence=%r) — REJECTING",
            parsed.value, parsed.evidence_text,
        )
        return None

    # SAFETY CHECK 2: evidence_text MUST be a substring of doc_full_text
    if parsed.evidence_text not in doc_full_text:
        log.warning(
            "grounded judge: evidence_text NOT in doc_full_text (evidence=%r) — HALLUCINATION REJECTED",
            parsed.evidence_text,
        )
        return None

    # SAFETY CHECK 3: length bound
    if len(parsed.value) > MAX_VALUE_LENGTH:
        log.warning("grounded judge: value too long (%d chars) — REJECTING", len(parsed.value))
        return None

    # All checks pass — emit Candidate. We don't have layout info (the LLM
    # doesn't return bbox), so use placeholder bbox (0,0,0,0) and page=0.
    # The orchestrator's downstream consumers know judge-emitted candidates
    # don't carry layout precision.
    return Candidate(
        field=field.name,
        value=parsed.value.strip(),
        page=0,
        bbox=(0.0, 0.0, 0.0, 0.0),
        evidence_text=parsed.evidence_text.strip(),
        model="qa_roberta",  # closest Literal for an extractive span judge; see module docstring
        confidence=0.7,  # mid-confidence; orchestrator can demote based on caller's review
    )
