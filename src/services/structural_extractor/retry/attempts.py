import time

from src.services.structural_extractor import derivation_rules  # noqa: F401 — registers rules
from src.services.structural_extractor.derivation import resolve_all
from src.services.structural_extractor.extractors import amounts as ext_amounts
from src.services.structural_extractor.extractors import dates as ext_dates
from src.services.structural_extractor.extractors import ids as ext_ids
from src.services.structural_extractor.extractors import line_items as ext_lines
from src.services.structural_extractor.extractors import parties as ext_parties
from src.services.structural_extractor.extractors import payment_terms as ext_pt
from src.services.structural_extractor.retry.state import AttemptOutput, RetryState
from src.services.structural_extractor.types import ExtractedValue


def run_attempt_1(state: RetryState) -> AttemptOutput:
    """Structural extraction + derivation. Returns AttemptOutput."""
    t0 = time.monotonic()
    doc = state.doc
    doc_type = state.doc_type
    extracted: dict = {}
    extracted.update(ext_ids.extract_ids(doc, doc_type))
    extracted.update(ext_dates.extract_dates(doc, doc_type))
    extracted.update(ext_parties.extract_parties(doc, doc_type))
    extracted.update(ext_amounts.extract_amounts(doc, doc_type))
    extracted.update(ext_pt.extract_payment_terms(doc, doc_type))
    line_items = ext_lines.extract_line_items(doc, doc_type)
    # Apply derivation on the extracted set
    extracted = resolve_all(extracted, doc_type)
    residual = sorted(state.target_fields - extracted.keys())
    latency = int((time.monotonic() - t0) * 1000)
    return AttemptOutput(
        attempt=1, source="structural", extracted=extracted,
        line_items=line_items if line_items else None,
        validation_failures=[], residual_unresolved=residual, latency_ms=latency,
    )


def run_attempt_nlu(state: RetryState, attempt_no: int) -> AttemptOutput:
    """Attempts 2-4 stub: run structural extraction again + NLU candidate augmentation.

    Full NLU integration deferred to Phase 15; this establishes the API.
    """
    t0 = time.monotonic()
    sources = {2: "nlu_ner", 3: "nlu_table", 4: "nlu_layout"}
    source = sources.get(attempt_no, "nlu_ner")
    # For now, re-run structural as a baseline; real NLU augmentation is Phase 15
    base = run_attempt_1(state)
    return AttemptOutput(
        attempt=attempt_no, source=source, extracted=base.extracted,
        line_items=base.line_items, validation_failures=[],
        residual_unresolved=base.residual_unresolved,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


def run_attempt_llm(state: RetryState, attempt_no: int) -> AttemptOutput:
    """Attempts 5-10: grounded LLM arbiter for residual unresolved fields."""
    t0 = time.monotonic()
    doc = state.doc
    residual = list(state.unresolved - state.accepted_header.keys())
    if not residual:
        return AttemptOutput(
            attempt=attempt_no, source="llm_fallback", extracted={},
            line_items=None, validation_failures=[], residual_unresolved=[],
            latency_ms=0,
        )
    from src.services.structural_extractor.llm_fallback import extract_fields_with_llm
    prior_attempts = [a.extracted for a in state.attempts]
    llm_out = extract_fields_with_llm(
        doc_text=doc.full_text,
        fields_needed=residual,
        prior_attempts=[
            {k: v.value for k, v in ev_dict.items()} for ev_dict in prior_attempts
        ],
        attempt_no=attempt_no,
    )
    extracted: dict[str, ExtractedValue] = {}
    for fname, fval in llm_out.items():
        extracted[fname] = ExtractedValue(
            value=fval, provenance="extracted",  # LLM output verified as source substring
            anchor_text=fval, anchor_ref=None,
            source="llm_fallback", confidence=0.85, attempt=attempt_no,
        )
    still_unresolved = sorted(set(residual) - extracted.keys())
    return AttemptOutput(
        attempt=attempt_no, source="llm_fallback", extracted=extracted,
        line_items=None, validation_failures=[], residual_unresolved=still_unresolved,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
