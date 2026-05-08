"""Tiebreaker judge: picks among disagreeing Candidate values for the same field.

Calls local Ollama with the TiebreakerInput contract serialized as JSON.
Post-validates that the LLM's chosen index is in [0, len(candidates)) or None.
Returns the chosen Candidate or None.
"""
from __future__ import annotations
import json
import logging
from typing import Optional
from src.services.ollama_client import ollama_generate
from src.services.extraction_v3.judge.contracts import (
    TiebreakerInput, TiebreakerOutput, TiebreakerCandidate,
)
from src.services.extraction_v3.schemas.candidate import Candidate

log = logging.getLogger(__name__)


def _build_prompt(input_obj: TiebreakerInput) -> str:
    """Construct a strict prompt asking the LLM to pick a candidate index."""
    cand_block = "\n".join(
        f"  {i}. value={c.value!r} model={c.model} confidence={c.confidence:.2f} "
        f"evidence={c.evidence!r}"
        for i, c in enumerate(input_obj.candidates)
    )
    return f"""You are a procurement-document extraction judge.

Multiple extractors disagree on the value for field "{input_obj.field}" (type: {input_obj.field_type}).
Pick ONE candidate index, or return null if NONE of them is correct.

Candidates:
{cand_block}

Surrounding document context (use this to disambiguate):
\"\"\"{input_obj.context_text}\"\"\"

Respond with ONLY a single JSON object on one line, no prose:
{{"chosen_candidate_index": <int or null>, "rationale": "<one sentence>"}}

The chosen_candidate_index must be in [0, {len(input_obj.candidates) - 1}] or null.
"""


def _parse_response(raw: str) -> TiebreakerOutput | None:
    """Extract the first {} JSON object from raw, validate as TiebreakerOutput."""
    if not raw:
        return None
    # Find first { ... } substring (LLM might wrap in markdown / commentary)
    start = raw.find("{")
    if start < 0:
        return None
    # Find matching closing brace
    depth = 0
    end = -1
    in_str = False
    esc = False
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
    blob = raw[start:end]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    try:
        return TiebreakerOutput(**data)
    except Exception:
        return None


def call_tiebreaker_judge(
    candidates: list[Candidate],
    field: str,
    field_type: str,
    parsed_full_text: str,
) -> Optional[Candidate]:
    """Run the tiebreaker LLM call. Returns the chosen Candidate or None.

    Post-validation: the LLM's chosen index must be in
    [0, len(candidates)). On any failure (bad JSON, out-of-range index,
    null), returns None.
    """
    if len(candidates) < 2:
        # Nothing to break — either no candidate or one is unambiguously selected
        return candidates[0] if candidates else None

    # Build context_text from surrounding ~200 chars of each candidate's evidence
    ctx_pieces = []
    for c in candidates:
        idx = parsed_full_text.find(c.evidence_text)
        if idx >= 0:
            ctx_pieces.append(parsed_full_text[max(0, idx - 100):idx + len(c.evidence_text) + 100])
    context_text = "\n---\n".join(ctx_pieces[:3])  # cap to 3 chunks to stay short

    input_obj = TiebreakerInput(
        field=field,
        field_type=field_type,
        candidates=[
            TiebreakerCandidate(
                value=c.value, model=c.model, confidence=c.confidence,
                evidence=c.evidence_text, page=c.page, bbox=c.bbox,
            ) for c in candidates
        ],
        context_text=context_text,
    )
    prompt = _build_prompt(input_obj)

    raw = ollama_generate(prompt, num_predict=512, temperature=0.0)
    if raw is None:
        return None

    result = _parse_response(raw)
    if result is None or result.chosen_candidate_index is None:
        return None

    idx = result.chosen_candidate_index
    if idx < 0 or idx >= len(candidates):
        # Out-of-range — reject (don't blindly trust the LLM)
        log.warning(
            "tiebreaker LLM returned out-of-range index %s for %d candidates",
            idx, len(candidates),
        )
        return None

    return candidates[idx]
