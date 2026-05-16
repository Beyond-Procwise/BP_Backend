"""L1 — runs PatternRegistry over a ParsedDocument's full_text.

Produces Candidate records with source='regex'. Every candidate carries a
Span whose `text` is the literal source substring (the substring grounding
invariant holds by construction).
"""
from __future__ import annotations

from typing import Any

from src.services.extraction.pattern_registry import (
    CompiledPattern,
    PatternRegistry,
    get_registry,
)
from src.services.extraction.types import Candidate, Span


def _locate_span(parsed: Any, hit_text: str, fallback_page: int = 1) -> Span:
    """Find (page, bbox) for the source text. Token-based when ParsedDocument
    has tokens; otherwise a page-1, zero-bbox fallback (still grounded — the
    text is the literal substring of full_text)."""
    pages = getattr(parsed, "pages", None) or []
    for page in pages:
        for tok in getattr(page, "tokens", None) or []:
            if hit_text == tok.text or hit_text in tok.text:
                return Span(page=getattr(page, "index", fallback_page),
                            bbox=tuple(tok.bbox), text=hit_text)
    return Span(page=fallback_page, bbox=(0.0, 0.0, 0.0, 0.0), text=hit_text)


def _emit_pattern_hits(
    text: str, pat: CompiledPattern, parsed: Any
) -> list[Candidate]:
    """Walk every anchor match in `text`; for each, search the post-anchor
    window for the value regex. Emit a Candidate per successful match."""
    out: list[Candidate] = []
    for am in pat.anchor_re.finditer(text):
        window_start = am.end()
        window_end = window_start + pat.max_span_after_anchor_chars
        window = text[window_start:window_end]
        vm = pat.value_re.search(window)
        if not vm:
            continue
        # group(1) when present (captures the value); else group(0)
        value = vm.group(1) if vm.lastindex else vm.group(0)
        if not value:
            continue
        # span.text is exactly what we captured — substring of `text` by construction
        out.append(Candidate(
            field=pat.field,
            value=value.strip(),
            span=_locate_span(parsed, value.strip()),
            source="regex",
            pattern_name=pat.name,
            confidence=pat.prior_confidence,
        ))
    return out


def run_pattern_extractor(parsed: Any, doc_type: str) -> list[Candidate]:
    """Run all patterns for every field in the registry against parsed.full_text.

    Higher-prior patterns are tried first per field (registry orders them).
    Multiple matches per field are kept and ranked downstream by L3 tiebreaker.
    """
    registry: PatternRegistry = get_registry(doc_type)
    text = parsed.full_text or ""
    out: list[Candidate] = []
    for field in registry.fields():
        for pat in registry.patterns_for(field):
            hits = _emit_pattern_hits(text, pat, parsed)
            if hits:
                out.extend(hits)
                # First successful pattern per field is sufficient unless caller
                # wants a tiebreaker pool — keep all hits for now; L3 picks one.
                break
    return out
