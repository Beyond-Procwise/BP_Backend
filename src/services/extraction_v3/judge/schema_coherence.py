"""Schema-coherence judge — final pass on the assembled record.

Supports two judge backends selected by EXTRACTION_V3_JUDGE_MODEL:
  - "qwen" (default): Qwen2.5-VL-7B-Instruct with image+JSON input.
  - "ollama": legacy Ollama text-only judge (rollback path).

The Qwen path adds a visual grounding layer: the model sees the rasterized
first page of the document alongside the extracted JSON, so it can spot
visual inconsistencies that text-only can't catch.

Verdict is ADVISORY — orchestrator demotes record confidence on
'incoherent' but does not mutate the record.

Substring guarantee on corrected_value: the Qwen prompt instructs the model
that corrected_value MUST be a verbatim substring of document text. We post-
validate this and strip any corrected_value that is not a substring of the
full document text before returning CoherenceOutput.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from src.services.extraction_v3.judge.contracts import (
    CoherenceInput, CoherenceOutput, CoherenceIssue, InvariantResultSummary,
)

log = logging.getLogger(__name__)

_JUDGE_MODEL = os.getenv("EXTRACTION_V3_JUDGE_MODEL", "qwen").lower()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_ollama_prompt(input_obj: CoherenceInput) -> str:
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

Record:
{rec_json}

Invariant results:
{inv_block or "  (no invariants ran)"}

Respond with ONLY a single JSON object:
{{"verdict": "coherent" | "incoherent", "issues": [{{"field": "<name>", "issue": "<one sentence>"}}, ...]}}

If everything looks consistent, return verdict="coherent" and issues=[].
"""


def _build_qwen_prompt(input_obj: CoherenceInput) -> str:
    inv_block = "\n".join(
        f"  - {r.name}: {'PASS' if r.passed else 'FAIL'}"
        + (f" — {r.message}" if r.message else "")
        for r in input_obj.invariant_results
    )
    rec_json = json.dumps(input_obj.extracted_record, indent=2, default=str)
    return f"""You are reviewing an extracted {input_obj.doc_type} record. The image is the source document. \
The JSON below is what we extracted. For each field, mark it as 'correct', 'incorrect', or 'uncertain'.

Extracted record:
{rec_json}

Invariant check results:
{inv_block or "  (none)"}

Reply with ONLY a single JSON object on one line — no prose, no markdown:
{{"verdict": "coherent" | "incoherent", "issues": [{{"field": "<name>", "issue": "<one sentence>", \
"corrected_value": "<verbatim substring from image text>"}}]}}

Rules:
- If verdict is "coherent" and no issues, return issues=[].
- corrected_value MUST be a verbatim substring of text visible in the image.
  If you cannot find a verbatim substring for a correction, OMIT corrected_value entirely.
- Do NOT invent values — only report what is literally visible in the image.
"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(raw: str, doc_full_text: str | None = None) -> CoherenceOutput | None:
    """Extract first {{...}} from raw and validate as CoherenceOutput.

    If doc_full_text is provided, strips any corrected_value that is not a
    verbatim substring of it (substring guarantee enforcement).
    """
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

    # Build issues list, applying substring guarantee to corrected_value.
    issues = []
    for item in data.get("issues", []):
        if not isinstance(item, dict):
            continue
        corrected = item.get("corrected_value")
        if corrected is not None and doc_full_text is not None:
            if corrected not in doc_full_text:
                log.warning(
                    "schema_coherence: corrected_value %r not in full_text — stripping",
                    corrected,
                )
                item.pop("corrected_value", None)
        # CoherenceIssue only has field + issue (no corrected_value in the Pydantic model)
        try:
            issues.append(CoherenceIssue(
                field=item.get("field", ""),
                issue=item.get("issue", ""),
            ))
        except Exception:
            pass

    try:
        return CoherenceOutput(
            verdict=data.get("verdict", "coherent"),
            issues=issues,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Qwen path: rasterize first page + call Qwen2.5-VL
# ---------------------------------------------------------------------------

def _rasterize_first_page(file_path: str | None):
    """Return PIL.Image of the first page of a PDF, or None on failure."""
    if not file_path:
        return None
    p = Path(file_path)
    if not p.exists():
        return None

    suffix = p.suffix.lower()
    try:
        if suffix in (".pdf",):
            from pdf2image import convert_from_path
            images = convert_from_path(str(p), dpi=96, first_page=1, last_page=1)
            if images:
                img = images[0].convert("RGB")
                w, h = img.size
                if max(w, h) > 1024:
                    from PIL import Image as _PIL
                    scale = 1024 / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), _PIL.LANCZOS)
                return img
        elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
            from PIL import Image
            return Image.open(p).convert("RGB")
        else:
            # DOCX / etc. — no rasterization available; return None gracefully
            return None
    except Exception as exc:
        log.warning("schema_coherence: rasterize failed for %s: %s", file_path, exc)
        return None


def _call_qwen_coherence(
    input_obj: CoherenceInput,
    file_path: str | None = None,
    doc_full_text: str | None = None,
) -> CoherenceOutput | None:
    """Run the Qwen2.5-VL coherence judge with image input."""
    from src.services.extraction_v3.judge.qwen_vl import qwen_vl_extract

    image = _rasterize_first_page(file_path)
    if image is None:
        log.warning(
            "schema_coherence[qwen]: could not rasterize first page of %s — "
            "falling back to text-only Qwen prompt", file_path
        )

    prompt = _build_qwen_prompt(input_obj)

    try:
        if image is not None:
            raw = qwen_vl_extract(image, prompt, max_new_tokens=512)
        else:
            # No image available — run Qwen in text-only mode by using a
            # blank white 1×1 placeholder image rather than building a
            # separate text-only path. Qwen handles sparse images gracefully.
            from PIL import Image as _PILImage
            blank = _PILImage.new("RGB", (32, 32), color=(255, 255, 255))
            raw = qwen_vl_extract(blank, prompt, max_new_tokens=512)
    except RuntimeError as exc:
        log.error("schema_coherence[qwen]: Qwen load failed — %s", exc)
        return None

    if not raw:
        return None

    return _parse_response(raw, doc_full_text=doc_full_text)


# ---------------------------------------------------------------------------
# Ollama path (legacy / rollback)
# ---------------------------------------------------------------------------

def _call_ollama_coherence(input_obj: CoherenceInput) -> CoherenceOutput | None:
    from src.services.ollama_client import ollama_generate
    prompt = _build_ollama_prompt(input_obj)
    raw = ollama_generate(prompt, num_predict=512, temperature=0.0, retries=1, timeout=30)
    if raw is None:
        return None
    return _parse_response(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_coherence_judge(
    doc_type: str,
    extracted_record: dict,
    invariant_results: list[InvariantResultSummary] | None = None,
    file_path: str | None = None,
    doc_full_text: str | None = None,
) -> CoherenceOutput | None:
    """Run the schema-coherence judge. Returns CoherenceOutput or None on failure.

    Verdict is ADVISORY — caller decides what to do with 'incoherent'.

    Args:
        doc_type: 'invoice', 'purchase_order', 'quote', 'contract'.
        extracted_record: Dict of {field_name: value} already committed.
        invariant_results: Optional list of invariant check results.
        file_path: Path to the source document (used for image rasterization
            when EXTRACTION_V3_JUDGE_MODEL=qwen).
        doc_full_text: Full text of the document for substring validation of
            corrected_value fields from the Qwen response.
    """
    if not extracted_record:
        return None

    input_obj = CoherenceInput(
        doc_type=doc_type,
        extracted_record=extracted_record,
        invariant_results=invariant_results or [],
    )

    judge_model = os.getenv("EXTRACTION_V3_JUDGE_MODEL", "qwen").lower()

    if judge_model == "ollama":
        log.debug("schema_coherence: using Ollama judge (EXTRACTION_V3_JUDGE_MODEL=ollama)")
        return _call_ollama_coherence(input_obj)
    else:
        # "qwen" or any unrecognised value → Qwen (default)
        log.debug("schema_coherence: using Qwen2.5-VL judge (EXTRACTION_V3_JUDGE_MODEL=%s)", judge_model)
        return _call_qwen_coherence(input_obj, file_path=file_path, doc_full_text=doc_full_text)
