"""Reading the situation well enough to decide what to do about it.

Whether a price is an outlier against the ones we have seen, what a supplier's
message signals, whether there is any point continuing, and what this agent is
actually mandated to offer. Deciding, not saying: the wording lives in prose.py.

Every function here was a method on NegotiationAgent that never touched `self`.
They are unchanged apart from losing that argument.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import AgentContext
from agents.negotiation.config import (
    AUTHORITY_AGENT_KEY,
    FINAL_OFFER_PATTERNS,
    LLM_ENABLED,
    LLM_MODEL,
    MARKET_ESCALATION_THRESHOLD,
    MARKET_REVIEW_THRESHOLD,
    MAX_SUPPLIER_REPLIES,
    MAX_TERM_DAYS,
    MAX_VOLUME_LIMIT,
)
from agents.negotiation.models import NegotiationPositions

logger = logging.getLogger(__name__)


def extract_negotiation_signals(*,
    supplier_message: Optional[str],
    snippets: List[str],
    market: Dict[str, Any],
    performance: Dict[str, Any],
) -> Dict[str, Any]:
    text = " ".join([t for t in ([supplier_message] + snippets) if t])[:8000]
    signals: Dict[str, Any] = {
        "finality_hint": False,
        "capacity_tight": False,
        "moq": None,
        "payment_terms_hint": None,
        "delivery_flex": None,
        "alt_part_offered": False,
        "tone": "neutral",
        "concession_band_pct": None,
    }

    lowered = text.lower() if text else ""
    if any(p in lowered for p in ("capacity", "backlog", "constrained")):
        signals["capacity_tight"] = True
    if "moq" in lowered:
        m = re.search(r"moq[^0-9]*([0-9]{2,})", lowered)
        if m:
            try:
                signals["moq"] = int(m.group(1))
            except Exception:
                pass
    if any(x in lowered for x in ("net-30", "net30", "net 30", "net-45", "net 45", "early payment")):
        signals["payment_terms_hint"] = "tradeable"
    if any(x in lowered for x in ("expedite", "split shipment", "partial", "air freight")):
        signals["delivery_flex"] = "possible"
    if any(x in lowered for x in ("alternate", "alternative", "equivalent", "substitute", "brand b")):
        signals["alt_part_offered"] = True
    if any(x in lowered for x in ("cannot go lower", "final", "last price", "our best price", "rock bottom")):
        signals["finality_hint"] = True
        signals["tone"] = "firm"

    if LLM_ENABLED and text:
        try:  # pragma: no cover - optional dependency
            from services.ollama_client import ollama_generate  # type: ignore

            prompt = (
                "Extract JSON with keys: tone (firm/flexible/neutral), finality_hint (bool), "
                "capacity_tight (bool), moq (int or null), payment_terms_hint (tradeable/fixed/null), "
                "delivery_flex (possible/unlikely/null), concession_band_pct (float or null). Only return JSON.\n\n"
                f"Text:\n{text}"
            )
            # Use the project-standard wrapper: handles timeout (default 600s),
            # retries, and the GPU semaphore — avoids an indefinite hang.
            content = ollama_generate(prompt, model=LLM_MODEL, temperature=0.1) or ""
            if "{" in content and "}" in content:
                content = content[content.find("{") : content.rfind("}") + 1]
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    for key in signals:
                        if key in parsed and parsed[key] is not None:
                            signals[key] = parsed[key]
        except Exception:
            logger.debug("LLM signal extraction skipped/failed", exc_info=True)

    signals["market"] = market or {}
    signals["performance"] = performance or {}
    return signals


def detect_final_offer(supplier_message: Optional[str], supplier_snippets: List[str]
) -> Optional[str]:
    texts: List[str] = []
    if supplier_message:
        texts.append(supplier_message)
    texts.extend(snippet for snippet in supplier_snippets if snippet)
    for text in texts:
        lowered = text.lower()
        for pattern in FINAL_OFFER_PATTERNS:
            if pattern in lowered:
                return f"Supplier indicated final offer via phrase '{pattern}'."
    return None


def should_continue(state: Dict[str, Any],
    supplier_reply_registered: bool,
    final_offer_reason: Optional[str],
) -> Tuple[bool, str, str]:
    status = state.get("status", "ACTIVE")
    if status in {"COMPLETED", "EXHAUSTED"}:
        return False, status, f"Session already {status.lower()}."
    if final_offer_reason:
        return False, "COMPLETED", final_offer_reason
    replies = int(state.get("supplier_reply_count", 0))
    if replies >= MAX_SUPPLIER_REPLIES():
        return False, "EXHAUSTED", "Supplier reply cap reached."
    if state.get("awaiting_response") and not supplier_reply_registered:
        return False, "AWAITING_SUPPLIER", "Awaiting supplier response."
    return True, "ACTIVE", ""


def build_positions_from_decision(decision: Dict[str, Any],
    price: Optional[float],
    target_price: Optional[float],
    round_no: int,
) -> NegotiationPositions:
    """Build NegotiationPositions object from decision data."""

    positions_dict = decision.get("positions", {})
    if isinstance(positions_dict, dict):
        history = positions_dict.get("history", [])
    else:
        history = []

    return NegotiationPositions(
        start=decision.get("start_position") or positions_dict.get("start"),
        desired=decision.get("desired_position")
        or positions_dict.get("desired")
        or target_price,
        no_deal=decision.get("no_deal_position") or positions_dict.get("no_deal"),
        supplier_offer=price,
        history=history if isinstance(history, list) else [],
    )


def respect_positions(counter: Optional[float], positions: NegotiationPositions
) -> Optional[float]:
    if counter is None:
        return None
    try:
        candidate = float(counter)
    except (TypeError, ValueError):
        return None

    if positions.desired is not None:
        try:
            candidate = max(candidate, float(positions.desired))
        except (TypeError, ValueError):
            pass
    if positions.no_deal is not None:
        try:
            candidate = min(candidate, float(positions.no_deal))
        except (TypeError, ValueError):
            pass
    if positions.start is not None:
        try:
            candidate = min(candidate, float(positions.start))
        except (TypeError, ValueError):
            pass

    return round(candidate, 2)


def resolve_authority_block(context: AgentContext
) -> Optional[Dict[str, Any]]:
    """This agent's mandate for this run.

    The orchestrator resolves it for the negotiation and supplier_interaction
    workflows and injects it at ``input_data["authority"]``. The live
    inbound-reply route has no orchestrator, so the agent resolves it itself
    rather than read a missing block as permission -- which is the whole
    reason `resolve_authority` fails closed.
    """

    injected = (context.input_data or {}).get("authority")
    if isinstance(injected, dict):
        block = injected.get(AUTHORITY_AGENT_KEY)
        if isinstance(block, dict):
            return block

    try:
        from src.engines.policy_engine import PolicyEngine
        from src.services.db import get_conn
        from src.services.governance_tools.authority import resolve_authority

        resolved = resolve_authority(
            PolicyEngine(connection_factory=get_conn), [AUTHORITY_AGENT_KEY]
        )
        return resolved.get(AUTHORITY_AGENT_KEY)
    except Exception:
        logger.exception(
            "authority resolution failed for the negotiation agent; no price "
            "will be proposed on this round"
        )
        from src.services.governance_tools.authority import ungoverned_block

        return ungoverned_block(
            AUTHORITY_AGENT_KEY, "the policy that sets its value limit could not be read"
        )


def detect_outliers(*,
    supplier_offer: Optional[float],
    target_price: Optional[float],
    walkaway_price: Optional[float],
    market_floor: Optional[float],
    volume_units: Optional[float],
    term_days: Optional[int],
) -> Dict[str, Any]:
    alerts: List[str] = []
    requires_review = False
    human_override = False
    review_recommendation: Optional[str] = None
    rationale_notes: List[str] = []

    def _percentage_gap(reference: Optional[float], offer: Optional[float]) -> Optional[float]:
        try:
            if reference is None or offer is None or reference <= 0:
                return None
            return (reference - offer) / reference
        except (TypeError, ValueError):
            return None

    market_gap = _percentage_gap(market_floor, supplier_offer)
    walkaway_gap = _percentage_gap(walkaway_price, supplier_offer)

    if market_gap is not None and market_gap >= MARKET_REVIEW_THRESHOLD():
        requires_review = True
        review_recommendation = "query_for_human_review"
        alerts.append(
            f"Supplier offer is {market_gap * 100:.1f}% below market reference {market_floor:.2f}."
        )
        rationale_notes.append("Requested price is materially below market benchmarks; seek justification.")
        if market_gap >= MARKET_ESCALATION_THRESHOLD():
            human_override = True
            rationale_notes.append(
                "Supplier offer breaches escalation threshold relative to market floor."
            )

    if walkaway_gap is not None and walkaway_gap >= MARKET_REVIEW_THRESHOLD():
        requires_review = True
        review_recommendation = review_recommendation or "query_for_human_review"
        alerts.append(
            f"Supplier offer is {walkaway_gap * 100:.1f}% below walk-away price {walkaway_price:.2f}."
        )
        rationale_notes.append(
            "Requested price undercuts internal walk-away guardrail; confirm intent before proceeding."
        )
        if walkaway_gap >= MARKET_ESCALATION_THRESHOLD():
            human_override = True
            rationale_notes.append(
                "The requested price is more than 20% below our walk-away price; escalation required."
            )

    if volume_units is not None and volume_units > MAX_VOLUME_LIMIT():
        requires_review = True
        review_recommendation = review_recommendation or "query_for_human_review"
        alerts.append(
            f"Requested volume {volume_units:.0f} exceeds configured limit {MAX_VOLUME_LIMIT():.0f}."
        )
        rationale_notes.append("Request supplier rationale for above-capacity volume.")
        if volume_units > MAX_VOLUME_LIMIT() * 1.5:
            human_override = True
            rationale_notes.append("Volume exceeds escalation ceiling; seek human approval.")

    if term_days is not None and term_days > MAX_TERM_DAYS():
        requires_review = True
        review_recommendation = review_recommendation or "query_for_human_review"
        alerts.append(
            f"Requested payment term {term_days} days exceeds policy limit {MAX_TERM_DAYS()} days."
        )
        rationale_notes.append("Payment term exceeds policy; confirm via human review.")
        if term_days > MAX_TERM_DAYS() * 2:
            human_override = True
            rationale_notes.append("Payment term far exceeds tolerance; human intervention required.")

    message = None
    if human_override:
        for note in rationale_notes[::-1]:
            if "escalation required" in note.lower():
                message = note
                break
        message = message or "Escalation required before proceeding."
    elif requires_review and rationale_notes:
        message = rationale_notes[-1]

    return {
        "alerts": alerts,
        "requires_review": requires_review,
        "human_override": human_override,
        "recommendation": review_recommendation,
        "message": message,
    }
