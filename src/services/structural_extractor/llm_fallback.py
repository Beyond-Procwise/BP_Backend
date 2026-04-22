"""Strict grounded LLM fallback for extraction.

Calls the LLM only for fields not recovered by structural/NLU layers, then
verifies every returned value is a literal substring of the source document
— ungrounded (hallucinated) values are dropped.
"""
from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """You are an extraction engine. Below is a procurement document.
Extract ONLY these fields: {fields_needed}.

Return ONLY valid JSON in this format:
{{"field_name": {{"value": "<exact substring from document>", "anchor": "<5-10 words from document around the value>"}}}}

If a field is not present in the document, OMIT it from the output (do not return null).
Never invent, calculate, or paraphrase values. Every value MUST appear verbatim in the document text.

{prior_context}
Document text:
{doc_text}

JSON:"""


def _call_llm(prompt: str) -> str:
    """Calls AgentNick via services.ollama_client. Returns raw text."""
    try:
        from services.ollama_client import ollama_generate  # type: ignore
    except ImportError:
        from src.services.ollama_client import ollama_generate
    return ollama_generate(prompt=prompt, temperature=0, num_predict=4096) or ""


def verify_grounded(value: str, doc_text: str) -> bool:
    """Substring verification — normalized (whitespace/case insensitive)."""
    if not value or not doc_text:
        return False
    norm_v = re.sub(r"\s+", "", str(value).lower())
    norm_d = re.sub(r"\s+", "", doc_text.lower())
    return norm_v in norm_d


def _build_prompt(
    doc_text: str,
    fields_needed: list[str],
    prior_attempts: list,
    attempt_no: int,
) -> str:
    prior_lines: list[str] = []
    if prior_attempts:
        prior_lines.append(
            "Prior extraction attempts (fields already confirmed — do not re-extract):"
        )
        for attempt in prior_attempts:
            for fname, fval in attempt.items():
                prior_lines.append(f"  - {fname}: {fval}")
    prior_ctx = "\n".join(prior_lines) + "\n" if prior_lines else ""
    if attempt_no >= 6:
        prior_ctx += (
            "IMPORTANT: Do NOT invent values. Only extract what literally appears "
            "in the document below.\n"
        )
    return _PROMPT_TEMPLATE.format(
        fields_needed=", ".join(fields_needed),
        prior_context=prior_ctx,
        doc_text=doc_text[:10000],
    )


def extract_fields_with_llm(
    doc_text: str,
    fields_needed: list[str],
    prior_attempts: list,
    attempt_no: int = 5,
) -> dict[str, str]:
    """Returns {field: value} for fields that were (a) returned by the LLM
    AND (b) grounded in ``doc_text`` (i.e. literally present)."""
    prompt = _build_prompt(doc_text, fields_needed, prior_attempts, attempt_no)
    raw = _call_llm(prompt)
    # Extract JSON object
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    out: dict[str, str] = {}
    for fname, entry in data.items():
        if fname not in fields_needed:
            continue
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        if value is None or value == "":
            continue
        if verify_grounded(str(value), doc_text):
            out[fname] = str(value)
        else:
            log.debug("LLM returned ungrounded value for %s: %r", fname, value)
    return out
