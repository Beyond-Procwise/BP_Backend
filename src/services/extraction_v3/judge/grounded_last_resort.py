"""Grounded last-resort judge — anti-hallucination guarantee.

Fires only for required fields where Layer-2 produced zero candidates.
The judge MUST return a value that is a verbatim substring of doc_full_text;
the orchestrator post-validates this.

Critical safety contract: a value not satisfying ``value == evidence_text``
AND ``evidence_text in doc_full_text`` is REJECTED. There is no fallback.
If the LLM cannot cite, the field stays NULL → review queue.

Supports two backends selected by EXTRACTION_V3_JUDGE_MODEL:
  - "qwen" (default): Qwen2.5-VL-7B-Instruct with image + text prompt.
  - "ollama": legacy Ollama text-only (rollback path).

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
import os
import re

from src.services.extraction_v3.judge.contracts import (
    GroundedInput, GroundedOutput, GroundedConstraints,
)
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.yaml_schema.loader import FieldSpec

log = logging.getLogger(__name__)

MAX_VALUE_LENGTH = 64
MAX_DOC_TEXT_FOR_PROMPT = 16000  # cap how much full_text we send to keep prompt size bounded

_WS_RE = re.compile(r"\s+")

# Letter-spaced OCR run: 2+ single-character tokens separated by single
# spaces. Real-world examples from docling: "I N V O I C E N O : 1 3 2 5 4 8"
# (long run, caught by either threshold) AND "N O" / "1 s t" (short runs).
# The earlier ≥4 threshold missed the latter, leaving "N O" stranded between
# collapsed neighbours. Run boundaries (whitespace/EOS) on both sides keep
# this from collapsing ordinary 2-letter words like "It is" — those don't
# match \S\s\S because the inner char is followed by a letter, not space.
# Used by the L1 pattern_extractor (collapsed view) AND the grounding gate.
_LETTER_SPACED_RE = re.compile(r"(?:(?<=^)|(?<=\s))((?:\S\s){1,}\S)(?=\s|$)")


def _norm_ws(s: str) -> str:
    """Collapse runs of whitespace to single spaces and strip ends.
    Used for grounding the LLM's evidence_text against doc_full_text: OCR
    and table extraction routinely emit multiple spaces, NBSPs, or stray
    newlines that the LLM normalises away in its response. Without this,
    the substring check rejects correct extractions on cosmetic whitespace."""
    return _WS_RE.sub(" ", s).strip()


def _collapse_letter_spacing(s: str) -> str:
    """Compact ``I N V O I C E`` runs to ``INVOICE`` for grounding only.

    Some PDFs emit letter-spaced headings/lines that the L1 regex and the
    judge's image-derived evidence can't ground against verbatim. This
    transform restores word-level continuity without modifying the source.
    Conservative: ≥4 single-char tokens separated by single spaces.
    """
    if not s:
        return s
    return _LETTER_SPACED_RE.sub(lambda m: m.group(1).replace(" ", ""), s)


# Label prefixes the LLM sometimes carries into the value when asked for
# the verbatim span. Stripping these is type-agnostic and safe: each is a
# common procurement document label that is never the field value itself.
# Order matters slightly — longer phrases first so we don't half-strip.
_LABEL_PREFIX_RE = re.compile(
    r"^(?:"
    r"invoice\s*(?:number|no\.?|#)|"
    r"inv\s*(?:no\.?|#)|"
    r"quote\s*(?:number|no\.?|reference|ref|#)|"
    r"quot(?:e|ation)\s*(?:no\.?|#)|"
    r"po\s*(?:number|no\.?|#)|"
    r"purchase\s*order\s*(?:number|no\.?|#)|"
    r"document\s*(?:number|no\.?|#)|"
    r"reference\s*(?:number|no\.?|#)|"
    r"ref\s*(?:no\.?|#)|"
    r"order\s*(?:number|no\.?|#)|"
    r"(?:invoice|quote|po|order|issue|document|prepared|billing)\s*date|"
    r"date(?:\s+of\s+(?:invoice|quote|issue))?|"
    r"date\s+issued|"
    r"total(?:\s+(?:amount|due|incl(?:\.|usive)?(?:\s+(?:of\s+)?tax)?))?|"
    r"grand\s*total|"
    r"sub\s*total|"
    r"amount\s+due|"
    r"supplier(?:\s+name)?|"
    r"vendor(?:\s+name)?|"
    r"sold\s+by|"
    r"sold\s+to|"
    r"bill\s+to|"
    r"ship\s+to|"
    r"buyer(?:\s+name)?|"
    r"customer(?:\s+name)?|"
    r"recipient|"
    r"send\s+to|"
    r"from|to"
    r")\b[\s:.,#\-]*",
    re.IGNORECASE,
)


def _strip_leading_label(value: str) -> str:
    """Remove a leading procurement label (e.g. "Invoice No. ", "Quote Number: ")
    from a candidate value, if one is present. No-op if no label matches.
    This is the last line of defence against an LLM that returns the whole
    labelled span as the value when the field type is a bare identifier."""
    if not value:
        return value
    stripped = _LABEL_PREFIX_RE.sub("", value).strip()
    return stripped or value


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
- value and evidence_text must be IDENTICAL strings (the exact span you
  found in the document, e.g. "Quote No. QUT104680" or just "QUT104680")
- the string must appear as a verbatim substring of the document text above
- DO NOT normalise dates, currencies, or formatting — copy the raw token
- DO NOT invent surrounding labels that aren't in the document
- maximum length: {input_obj.constraints.max_length} characters
- if the value is not present in the text, return value=null and evidence_text=null

Respond with ONLY a single JSON object on one line, no prose:
{{"value": <string or null>, "evidence_text": <string or null>, "rationale": "<one sentence>"}}
"""


def _build_qwen_prompt(input_obj: GroundedInput) -> str:
    """Prompt optimised for Qwen2.5-VL: shorter, image-aware."""
    labels = ", ".join(f'"{l}"' for l in input_obj.field_canonical_labels[:5])
    return f"""You are a procurement-document extraction assistant.

I need to extract the field "{input_obj.field}" (type: {input_obj.field_type}) from this document.
Common labels for this field: {labels}

The image above is the source document. Find the value for "{input_obj.field}" and quote it \
VERBATIM — it MUST be a literal substring of the text visible in the image.

If the field is not in the document, return null for both value and evidence_text.

Constraints:
- value and evidence_text MUST be identical strings (the exact span you see
  in the image, e.g. "Quote No. QUT104680" or just "QUT104680")
- DO NOT normalise dates, currencies, or formatting — copy the raw token
- DO NOT invent labels that aren't directly next to the value in the image
- maximum length: {input_obj.constraints.max_length} characters

Reply with ONLY a single JSON object, no prose:
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


def _apply_safety_checks(
    parsed: GroundedOutput,
    doc_full_text: str,
    field_name: str,
) -> Candidate | None:
    """Apply the safety checks and return a Candidate or None.

    Anti-hallucination contract (transitive grounding):
      - value is a literal substring of evidence_text
      - evidence_text is a literal substring of doc_full_text
    Together these imply ``value in doc_full_text`` — i.e. the LLM cannot
    return any token the document doesn't actually contain. The earlier
    "value == evidence_text" rule was over-strict: real procurement fields
    naturally sit inside a labelled phrase ("Quote No. QUT104680", "Date:
    02/01/2019"), and demanding identity forced the judge to choose
    between returning the bare value (rejected) or the whole phrase
    (would land in the column as junk). The transitive form preserves the
    guarantee without crippling recall.
    """
    # Null or empty → return None
    if parsed.is_null():
        return None
    if not parsed.value or not parsed.evidence_text:
        return None

    value = parsed.value.strip()
    evidence = parsed.evidence_text.strip()
    if not value or not evidence:
        return None

    # SAFETY CHECK 1: value MUST be a literal substring of evidence_text.
    # Cheapest check; also catches the LLM emitting unrelated value/evidence.
    if value not in evidence:
        log.warning(
            "grounded judge: value NOT in evidence_text (value=%r, evidence=%r) — REJECTING",
            value, evidence,
        )
        return None

    # SAFETY CHECK 2: evidence_text MUST be present in doc_full_text under
    # text normalisation. The no-fabrication backstop is preserved — any
    # token in evidence that doesn't appear in the doc would survive
    # collapse and fail the substring check. We try progressively more
    # forgiving normalisations:
    #   1. raw byte-for-byte
    #   2. whitespace collapsed (handles docling's table double-spaces)
    #   3. letter-spacing collapsed THEN whitespace collapsed (handles PDFs
    #      where docling emits ``I N V O I C E N O : 1 3 2 5 4 8``)
    # All three preserve every non-whitespace token; nothing the LLM
    # invented can slip through.
    if evidence not in doc_full_text:
        ev_norm = _norm_ws(evidence)
        doc_norm = _norm_ws(doc_full_text)
        if ev_norm not in doc_norm:
            # Last resort: collapse all whitespace AND letter-spacing on
            # both sides. After this, "INVOICE NO: 132548" reduces to
            # "INVOICENO:132548" which is found in the letter-spaced
            # source "I N V O I C E N O : 1 3 2 5 4 8 ..." → "INVOICENO:1325481stOct...".
            # Safe: removing whitespace can't introduce content the doc
            # doesn't contain — every non-whitespace char in evidence
            # must still appear in the doc.
            ev_squeezed = _WS_RE.sub("", _collapse_letter_spacing(evidence))
            doc_squeezed = _WS_RE.sub("", _collapse_letter_spacing(doc_full_text))
            if ev_squeezed not in doc_squeezed:
                log.warning(
                    "grounded judge: evidence_text NOT in doc_full_text (evidence=%r) — HALLUCINATION REJECTED",
                    evidence,
                )
                return None

    # SAFETY CHECK 3: length bound on the value itself (not the evidence,
    # which is allowed to be the wider phrase around the value).
    if len(value) > MAX_VALUE_LENGTH:
        log.warning("grounded judge: value too long (%d chars) — REJECTING", len(value))
        return None

    # Last-mile cleanup: when the LLM returns the labelled span as `value`
    # ("Invoice No. INV610366"), strip the leading label so the column gets
    # the bare identifier. The original labelled string is preserved as
    # evidence so provenance still cites the document phrase. Safe because
    # the stripped value remains a substring of evidence (still in doc).
    cleaned_value = _strip_leading_label(value)
    if cleaned_value != value and cleaned_value:
        log.info(
            "grounded judge: stripped label from value (raw=%r, clean=%r)",
            value, cleaned_value,
        )
        value = cleaned_value

    return Candidate(
        field=field_name,
        value=value,
        page=0,
        bbox=(0.0, 0.0, 0.0, 0.0),
        evidence_text=evidence,
        model="qa_roberta",  # closest Literal for an extractive span judge; see module docstring
        confidence=0.7,
    )


def _call_ollama_grounded(
    field: FieldSpec, doc_full_text: str
) -> Candidate | None:
    """Run grounded last-resort via Ollama."""
    from src.services.ollama_client import ollama_generate

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
    return _apply_safety_checks(parsed, doc_full_text, field.name)


def _call_qwen_grounded(
    field: FieldSpec,
    doc_full_text: str,
    file_path: str | None = None,
) -> Candidate | None:
    """Run grounded last-resort via Qwen2.5-VL with optional image input."""
    from src.services.extraction_v3.judge.qwen_vl import qwen_vl_extract
    from src.services.extraction_v3.judge.schema_coherence import _rasterize_first_page

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
    prompt = _build_qwen_prompt(input_obj)

    image = _rasterize_first_page(file_path) if file_path else None
    if image is None:
        # No image — use a blank white placeholder
        from PIL import Image as _PILImage
        image = _PILImage.new("RGB", (32, 32), color=(255, 255, 255))
        log.debug("grounded judge[qwen]: no image for %s — using blank placeholder", file_path)

    try:
        raw = qwen_vl_extract(image, prompt, max_new_tokens=256)
    except RuntimeError as exc:
        log.error("grounded judge[qwen]: Qwen load failed — %s; falling back to Ollama", exc)
        return _call_ollama_grounded(field, doc_full_text)

    if not raw:
        return None

    parsed = _parse_response(raw)
    if parsed is None:
        return None

    return _apply_safety_checks(parsed, doc_full_text, field.name)


def call_grounded_last_resort(
    field: FieldSpec,
    doc_full_text: str,
    file_path: str | None = None,
) -> Candidate | None:
    """Run the grounded last-resort LLM call. Returns a Candidate or None.

    SAFETY: The returned Candidate's value is GUARANTEED to be a substring of
    doc_full_text. If the LLM violates this, the function returns None.
    No exceptions, no fallbacks — null on any deviation.

    Args:
        field: FieldSpec for the field to extract.
        doc_full_text: Full document text (substring guarantee source of truth).
        file_path: Path to source file for image rasterization (Qwen path).
    """
    if not doc_full_text or not doc_full_text.strip():
        return None

    judge_model = os.getenv("EXTRACTION_V3_JUDGE_MODEL", "qwen").lower()

    if judge_model == "ollama":
        log.debug("grounded judge: using Ollama (EXTRACTION_V3_JUDGE_MODEL=ollama)")
        return _call_ollama_grounded(field, doc_full_text)
    else:
        log.debug("grounded judge: using Qwen2.5-VL (EXTRACTION_V3_JUDGE_MODEL=%s)", judge_model)
        return _call_qwen_grounded(field, doc_full_text, file_path=file_path)
