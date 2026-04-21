from src.services.structural_extractor.retry.state import AttemptOutput, RetryState


def merge_attempt_into_state(state: RetryState, output: AttemptOutput) -> None:
    """Merge an attempt's output into the state.

    Conflict rule:
    - If field already accepted, keep earlier (earlier-attempt wins).
    - If new, add and remove from unresolved.
    - Line items: later attempts with more items replace earlier.
    """
    state.attempts.append(output)
    for field_name, ev in output.extracted.items():
        if field_name not in state.accepted_header:
            state.accepted_header[field_name] = ev
            state.unresolved.discard(field_name)
        # Else: earlier-attempt value wins (simple rule v1)
    # Line items: later attempts with more items replace earlier
    if output.line_items:
        if state.accepted_line_items is None or len(output.line_items) > len(state.accepted_line_items):
            state.accepted_line_items = output.line_items
