import logging

from src.services.structural_extractor.discovery.layout_fingerprint import layout_signature
from src.services.structural_extractor.discovery.schema import fields_for
from src.services.structural_extractor.retry.attempts import (
    run_attempt_1,
    run_attempt_llm,
    run_attempt_nlu,
)
from src.services.structural_extractor.retry.merge import merge_attempt_into_state
from src.services.structural_extractor.retry.state import RetryState
from src.services.structural_extractor.types import ExtractionResult

log = logging.getLogger(__name__)


def run_retry_loop(doc, doc_type: str, max_attempts: int = 10) -> ExtractionResult:
    target = set(fields_for(doc_type))
    state = RetryState(
        doc=doc, doc_type=doc_type, target_fields=target,
        unresolved=set(target),
    )
    for attempt_no in range(1, max_attempts + 1):
        if not state.unresolved:
            break
        if attempt_no == 1:
            out = run_attempt_1(state)
        elif attempt_no in (2, 3, 4):
            out = run_attempt_nlu(state, attempt_no)
        else:
            out = run_attempt_llm(state, attempt_no)
        merge_attempt_into_state(state, out)
        log.info(
            "Retry attempt %d: source=%s, +%d fields, residual=%d",
            attempt_no, out.source, len(out.extracted), len(state.unresolved),
        )
    sig = layout_signature(doc)
    return ExtractionResult(
        header=state.accepted_header,
        line_items=state.accepted_line_items or [],
        parsed_text=doc.full_text,
        unresolved_fields=sorted(state.unresolved),
        attempts=len(state.attempts),
        pattern_id_used=None,
        layout_signature=sig,
        process_monitor_id=None,
        doc_type=doc_type,
    )
