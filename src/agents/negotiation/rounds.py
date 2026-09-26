"""Book-keeping for a multi-round negotiation.

Which entries belong to which round, which quotes survived to the end, how a
human's HITL answer is read, and what the overall outcome was once every
supplier has stopped replying.

Every function here was a method on NegotiationAgent that never touched `self`.
They are unchanged apart from losing that argument.
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Dict, List, Optional, Sequence, Tuple, cast

from agents.base_agent import AgentContext


def bucket_entries_by_round(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group negotiation batch entries by round to enforce sequential execution."""

    buckets: Dict[Tuple[Optional[int], Optional[Any]], Dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_round = None
        for key in ("round", "round_number", "round_no"):
            candidate = entry.get(key)
            if candidate is not None:
                raw_round = candidate
                break

        numeric_sort: Optional[int] = None
        display_round: Optional[Any] = None
        if raw_round is not None:
            try:
                numeric_sort = int(float(raw_round))
                display_round = numeric_sort
            except Exception:
                display_round = raw_round
        else:
            numeric_sort = 1

        bucket_key = (numeric_sort, display_round)
        bucket = buckets.get(bucket_key)
        if not bucket:
            bucket = {
                "round": display_round,
                "numeric_sort": numeric_sort,
                "entries": [],
            }
            buckets[bucket_key] = bucket
        bucket["entries"].append(entry)

    ordered = sorted(
        buckets.values(),
        key=lambda bucket: (
            0,
            bucket["numeric_sort"],
        )
        if bucket.get("numeric_sort") is not None
        else (
            1,
            str(bucket.get("round") or "").lower(),
        ),
    )
    return ordered


def drain_pending_email_tasks(context: AgentContext, round_number: Optional[int]
) -> List[Dict[str, Any]]:
    """Retrieve and clear pending email tasks for a negotiation round."""

    try:
        task_map = getattr(context, "_pending_email_round_tasks")
    except AttributeError:
        return []

    if not task_map:
        return []

    try:
        lock = getattr(context, "_pending_email_round_lock")
    except AttributeError:
        lock = None

    try:
        round_key = int(round_number) if round_number is not None else None
    except Exception:
        round_key = None

    if round_key is None:
        return []

    if lock:
        with lock:
            return list(task_map.pop(round_key, []))
    return list(task_map.pop(round_key, []))


def extract_hitl_decisions(context: AgentContext,
    shared_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Collect explicit HITL decisions provided in the inbound payload."""

    decisions: Dict[str, Any] = {}

    def _merge(source: Optional[Dict[str, Any]]) -> None:
        if not isinstance(source, dict):
            return
        for key in ("hitl_decisions", "hitl_approvals", "hitl_review", "hitl"):
            value = source.get(key)
            if isinstance(value, dict):
                for round_key, decision_value in value.items():
                    decisions[str(round_key)] = decision_value

    if isinstance(context.input_data, dict):
        _merge(context.input_data)
        nested_shared = context.input_data.get("shared_context")
        if isinstance(nested_shared, dict):
            _merge(nested_shared)
    _merge(shared_context)
    return decisions


def normalise_hitl_value(value: Any) -> Tuple[str, Optional[str]]:
    """Normalise arbitrary decision tokens into approved/pending/rejected."""

    reason: Optional[str] = None
    candidate = value
    if isinstance(candidate, dict):
        reason = cast(Optional[str], candidate.get("reason") or candidate.get("notes"))
        candidate = candidate.get("status") or candidate.get("decision")

    if isinstance(candidate, bool):
        return ("approved" if candidate else "rejected", reason)

    token = str(candidate).strip().lower() if candidate is not None else ""
    if token in {"approved", "approve", "ok", "okay", "yes", "true", "allow", "proceed"}:
        return "approved", reason
    if token in {"rejected", "reject", "no", "false", "deny", "denied", "blocked"}:
        return "rejected", reason
    if token in {"pending", "awaiting", "hold", "review"}:
        return "pending", reason
    return "pending", reason


def compile_final_quotes(negotiation_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compile supplier offers and counters for downstream evaluation."""

    quotes: List[Dict[str, Any]] = []
    active_suppliers = negotiation_state.get("active_suppliers", {})
    if not isinstance(active_suppliers, dict):
        return quotes

    for supplier_id, supplier_state in active_suppliers.items():
        if not isinstance(supplier_state, dict):
            continue
        entry_payload = supplier_state.get("entry") or {}
        decisions = supplier_state.get("decisions") or []
        latest_decision = decisions[-1] if decisions else {}
        responses = supplier_state.get("responses") or []
        latest_response = responses[-1] if responses else {}

        currency = (
            latest_decision.get("currency")
            or entry_payload.get("currency")
            or entry_payload.get("currency_code")
        )

        quotes.append(
            {
                "supplier_id": supplier_id,
                "supplier_offer": entry_payload.get("current_offer")
                or entry_payload.get("price"),
                "counter_offer": latest_decision.get("counter_price"),
                "strategy": latest_decision.get("strategy"),
                "currency": currency,
                "rounds_completed": max(0, int(supplier_state.get("current_round", 1)) - 1),
                "latest_response": latest_response,
            }
        )

    return quotes


def run_async_task(coro: Awaitable[Any]) -> Any:
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "asyncio.run() cannot be called" not in str(exc):
            raise
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def extract_message_from_response(response: Dict[str, Any]) -> Optional[str]:
    """Extract plain text content from a supplier response."""

    for key in ("message", "message_text", "body", "text", "content"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def determine_overall_status(negotiation_state: Dict[str, Any]) -> str:
    """Summarise the overall negotiation status for reporting."""

    total = negotiation_state.get("total_suppliers", 0) or 0
    completed = len(negotiation_state.get("completed_suppliers", set()))
    failed = len(negotiation_state.get("failed_suppliers", {}))

    if total > 0 and completed == total:
        return "ALL_COMPLETED"
    if total > 0 and failed == total:
        return "ALL_FAILED"
    if total > 0 and completed + failed == total:
        return "PARTIALLY_COMPLETED"
    return "IN_PROGRESS"


def build_stop_message(status: str, reason: str, round_no: int) -> str:
    status_text = status.capitalize()
    reason_text = reason or "No further action required."
    return f"Negotiation {status_text} after round {round_no}: {reason_text}"


def build_decision_log(supplier: Optional[str],
    session_reference: Optional[str],
    price: Optional[float],
    target_price: Optional[float],
    decision: Dict[str, Any],
) -> str:
    base = (
        f"Strategy={decision.get('strategy')} counter={decision.get('counter_price')}"
        f" target={target_price} current={price} supplier={supplier} reference={session_reference}."
    )
    plays = decision.get("play_recommendations") or []
    if plays:
        top_snippets: List[str] = []
        for play in plays[:3]:
            if not isinstance(play, dict):
                continue
            lever = play.get("lever") or play.get("category")
            description = play.get("play") or play.get("description")
            if not description:
                continue
            if lever:
                top_snippets.append(f"{lever}: {description}")
            else:
                top_snippets.append(str(description))
        if top_snippets:
            base = f"{base} Plays={' | '.join(top_snippets)}."
    rationale = decision.get("rationale")
    if rationale:
        return f"{base} {rationale}"
    return base
