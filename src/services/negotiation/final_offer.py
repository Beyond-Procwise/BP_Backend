"""What to do when a supplier says an offer is final: a recommendation, and a reply to approve.

Final-offer wording used to end the negotiation and nothing more -- no email,
nothing raised for a person -- so the supplier's last word went unanswered.

Pure and deterministic. The limit is the one the counter plan already uses: the
walk-away when there is one, otherwise the target. Nothing is assumed: with no
readable offer or no limit on record the answer is to escalate, and the reason
says which is missing. The reply is fixed wording; only an offer within the
limit is accepted in it, and neither the limit nor the gap is ever told to the
supplier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
from typing import Any, Dict, Optional

from src.services.draft_assurance.frame import SIGN_OFF

ACCEPT = "accept"
ESCALATE = "escalate"


@dataclass(frozen=True)
class FinalOfferAdvice:
    recommendation: str
    reason: str
    offer: Optional[float]
    currency: Optional[str]
    limit: Optional[float]
    limit_kind: Optional[str]
    over_by: Optional[float]
    over_pct: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _positive(value: Optional[float]) -> Optional[float]:
    try:
        number = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return number if number is not None and number > 0 else None


def _money(amount: float, currency: Optional[str]) -> str:
    return f"{amount:,.2f} {currency}" if currency else f"{amount:,.2f}"


def advise(*, offer: Optional[float], target: Optional[float], walkaway: Optional[float],
           currency: Optional[str], hold_reason: Optional[str] = None) -> FinalOfferAdvice:
    """Accept a final offer that is within the limit on record; otherwise a person decides.

    ``hold_reason`` is why the agent may not commit to this on its own (its governed
    authority was withheld, or a review was required). An offer within the limit is
    then still escalated: being a good price does not make it the agent's to accept.
    """

    offer_value = _positive(offer)
    walkaway_value, target_value = _positive(walkaway), _positive(target)
    if walkaway_value is not None:
        limit, limit_kind = walkaway_value, "walk-away"
    elif target_value is not None:
        limit, limit_kind = target_value, "target"
    else:
        limit, limit_kind = None, None

    def _advice(recommendation: str, reason: str, over_by=None, over_pct=None) -> FinalOfferAdvice:
        return FinalOfferAdvice(recommendation, reason, offer_value, currency, limit, limit_kind, over_by, over_pct)

    if offer_value is None:
        return _advice(ESCALATE, "The supplier called their offer final, but its price could not be read "
                                 "from the reply. A person needs to read it and decide.")
    if limit is None:
        return _advice(ESCALATE, f"The supplier's final offer is {_money(offer_value, currency)}. There is no target "
                                 "or walk-away price on record to judge it against, so a person needs to decide.")
    if offer_value <= limit:
        within = (f"The supplier's final offer of {_money(offer_value, currency)} is within the "
                  f"{limit_kind} of {_money(limit, currency)}")
        hold = (hold_reason or "").strip().rstrip(".")
        if hold:
            return _advice(ESCALATE, f"{within}, but it needs a person's sign-off: {hold}.")
        return _advice(ACCEPT, f"{within}.")
    over_by = round(offer_value - limit, 2)
    over_pct = round(over_by / limit * 100, 1)
    return _advice(
        ESCALATE,
        f"The supplier's final offer of {_money(offer_value, currency)} is {_money(over_by, currency)} "
        f"({over_pct}%) above the {limit_kind} of {_money(limit, currency)}. A person needs to decide "
        "whether to accept it or walk away.",
        over_by, over_pct,
    )


def reply_html(advice: FinalOfferAdvice, *, contact_name: Optional[str]) -> str:
    """The reply to the supplier: an acceptance, or a holding reply that commits to nothing."""

    name = (contact_name or "").strip()
    greeting = f"Dear {name}," if name else "Hello,"
    if advice.recommendation == ACCEPT and advice.offer is not None:
        paragraphs = [
            f"Thank you for your final offer. We accept the price of {_money(advice.offer, advice.currency)}.",
            "We will be in touch shortly with the next steps.",
        ]
    elif advice.offer is not None:
        paragraphs = [
            f"Thank you for your final offer of {_money(advice.offer, advice.currency)}.",
            "We are reviewing it internally and will come back to you with our decision.",
        ]
    else:
        paragraphs = [
            "Thank you for your reply and for setting out your final offer.",
            "We are reviewing it internally and will come back to you with our decision.",
        ]
    parts = [greeting, *paragraphs, *SIGN_OFF.split("\n")]
    return "".join(f"<p>{escape(part)}</p>" for part in parts)
