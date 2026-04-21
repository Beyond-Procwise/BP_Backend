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
