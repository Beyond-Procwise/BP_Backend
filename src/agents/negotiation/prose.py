"""How a negotiation message actually reads.

The opening, the acknowledgment, the position, the asks and the close, plus the
five "plays" that weave a lever into a sentence a supplier will answer. This is
wording, not strategy: what to ask for is decided elsewhere and arrives here as
arguments.

Every function here was a method on NegotiationAgent that never touched `self`.
They are unchanged apart from losing that argument.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Imported rather than redefined so the hints cannot drift between the agent and
# the advisor, which both read them.
from services.negotiation_advice.ranking import TRADE_OFF_HINTS


def append_playbook_recommendations(summary: str,
    plays: List[Dict[str, Any]],
    playbook_context: Dict[str, Any],
) -> str:
    """Append structured playbook recommendations to the negotiation summary."""

    if not plays:
        return summary

    lines: List[str] = [summary.rstrip()]

    recommendation_lines: List[str] = []
    lever_hints: List[str] = []
    for play in plays[:4]:
        if not isinstance(play, dict):
            continue
        lever = str(play.get("lever", "")).strip()
        description = str(play.get("play", "")).strip()
        if description:
            if lever:
                recommendation_lines.append(f"- {lever}: {description}")
            else:
                recommendation_lines.append(f"- {description}")
        if lever:
            lookup_key = lever.strip().title()
            hint = TRADE_OFF_HINTS.get(lookup_key)
            if hint:
                lever_hints.append(f"- {lookup_key}: {hint}")

    if recommendation_lines:
        lines.append("\nRecommended plays:")
        lines.extend(recommendation_lines)

    if lever_hints:
        lines.append("\nTrade-off considerations:")
        lines.extend(lever_hints[:3])

    return "\n".join(lines)


def determine_negotiation_tone(round_no: int, signals: Dict[str, Any], strategy: str
) -> str:
    """Determine the appropriate negotiation tone."""

    if strategy in ("accept", "decline"):
        return "decisive"

    if signals.get("finality_hint"):
        return "collaborative-firm"

    if signals.get("capacity_tight"):
        return "understanding-assertive"

    if round_no == 1:
        return "exploratory-confident"
    elif round_no == 2:
        return "focused-collaborative"
    else:
        return "closure-oriented"


def craft_opening(round_no: int,
    supplier_message: Optional[str],
    signals: Dict[str, Any],
    tone: str,
) -> str:
    """Craft a rapport-building opening."""

    # Acknowledge their response if this isn't the first round
    if round_no > 1 and supplier_message:
        if signals.get("finality_hint"):
            return (
                "Thank you for taking the time to provide such a detailed response. "
                "I appreciate your transparency about your position, and I'd like to explore "
                "whether there's a way we can structure this to work for both parties."
            )
        elif signals.get("capacity_tight"):
            return (
                "I appreciate you sharing the challenges around capacity and lead times. "
                "Understanding your operational constraints helps us find creative solutions "
                "that work within those parameters."
            )
        else:
            return (
                "Thank you for your continued engagement on this. I've reviewed your proposal "
                "and believe we're making good progress toward an agreement that benefits both sides."
            )

    # First round opening
    if tone == "exploratory-confident":
        return (
            "Thank you for submitting your proposal. I've had a chance to review the details "
            "and would like to discuss how we might align this with our project requirements "
            "and budget parameters."
        )

    return "I'd like to continue our discussion on finding the right structure for this partnership."


def weave_commercial_play(plays: List[Dict[str, Any]], round_no: int, tone: str
) -> str:
    """Weave commercial plays into natural language."""

    play_texts = [p.get("play", "") for p in plays]

    # Volume-based plays
    if any("volume" in p.lower() or "tier" in p.lower() for p in play_texts):
        return (
            "I'd like to explore how we can structure volume commitments to unlock "
            "better unit economics for both sides. If we can commit to tier-based volumes—"
            "say, 250 units initially with pathways to 500+—there may be room to optimize "
            "the pricing structure while giving you better demand visibility."
        )

    # Early payment plays
    if any("early payment" in p.lower() or "net-15" in p.lower() for p in play_texts):
        return (
            "One area where we can create immediate value is payment terms. "
            "If we can accelerate payment to net-15, would that open up opportunities "
            "for a 2-3% discount? This improves your cash flow while reducing our total cost."
        )

    # Bundling plays
    if any("bundle" in p.lower() or "consolidate" in p.lower() for p in play_texts):
        return (
            "We're also looking at how we might consolidate spend across multiple categories "
            "or adjacent products. If we can bundle this with other requirements coming through "
            "our pipeline, there could be meaningful volume synergies that benefit both parties."
        )

    # Generic commercial
    return (
        "I believe there's room to structure the commercial terms in a way that creates "
        "value for both organizations—whether through volume commitments, payment optimization, "
        "or longer-term price stability."
    )


def weave_operational_play(plays: List[Dict[str, Any]], round_no: int, tone: str
) -> str:
    """Weave operational plays into natural language."""

    play_texts = [p.get("play", "") for p in plays]

    # Delivery/lead time plays
    if any("delivery" in p.lower() or "lead time" in p.lower() for p in play_texts):
        return (
            "On the operational side, delivery timing is critical for our project schedule. "
            "If you can guarantee delivery within 2 weeks or offer split shipments "
            "(perhaps 30% upfront, balance within 4 weeks), that would significantly de-risk "
            "our production timeline and justify the investment."
        )

    # Planning/forecasting plays
    if any("forecast" in p.lower() or "planning" in p.lower() for p in play_texts):
        return (
            "We're happy to share our demand forecasts and collaborate on supply planning "
            "to give you better visibility. This integrated approach typically helps both sides "
            "reduce buffer stock and improve fulfillment rates."
        )

    # SLA/service level plays
    if any("sla" in p.lower() or "priority" in p.lower() for p in play_texts):
        return (
            "To ensure this partnership meets both our operational needs, I'd like to define "
            "clear service levels around delivery performance and fulfillment rates. "
            "Having those guardrails helps us plan effectively and holds both parties accountable."
        )

    # Generic operational
    return (
        "From an operational standpoint, we're looking for reliability and responsiveness "
        "in fulfillment. If we can align on clear delivery commitments and planning cadences, "
        "that creates a strong foundation for the partnership."
    )


def weave_strategic_play(plays: List[Dict[str, Any]], round_no: int, tone: str
) -> str:
    """Weave strategic plays into natural language."""

    play_texts = [p.get("play", "") for p in plays]

    # Innovation/co-development plays
    if any("innovation" in p.lower() or "co-develop" in p.lower() for p in play_texts):
        return (
            "Beyond the immediate transaction, I see potential for deeper collaboration "
            "on product development and innovation. If we can align on a shared roadmap "
            "for the next 12-24 months, there may be opportunities to co-invest in capabilities "
            "that benefit both our organizations."
        )

    # ESG/sustainability plays
    if any("esg" in p.lower() or "sustainab" in p.lower() for p in play_texts):
        return (
            "Sustainability is increasingly important to our stakeholders. "
            "If we can incorporate ESG metrics and carbon reduction targets into our partnership, "
            "that strengthens the strategic case and may unlock internal budget flexibility."
        )

    # Long-term commitment plays
    if any("long-term" in p.lower() or "multi-year" in p.lower() for p in play_texts):
        return (
            "We're thinking about this as a multi-year partnership rather than a one-off transaction. "
            "If you're open to a longer-term commitment with price stability mechanisms, "
            "we can structure this in a way that gives you revenue predictability while securing "
            "our supply chain."
        )

    # Generic strategic
    return (
        "Strategically, we're looking to build partnerships that go beyond transactional relationships. "
        "If we can align on shared objectives and longer-term value creation, "
        "that opens up different ways to structure the commercial terms."
    )


def weave_risk_play(plays: List[Dict[str, Any]], round_no: int, tone: str
) -> str:
    """Weave risk plays into natural language."""

    play_texts = [p.get("play", "") for p in plays]

    # Warranty plays
    if any("warranty" in p.lower() for p in play_texts):
        return (
            "Given the mission-critical nature of these components, warranty coverage is important. "
            "If you can extend warranty to 2-3 years at no additional cost, "
            "that reduces our total cost of ownership and makes the business case stronger."
        )

    # Service credit plays
    if any("service credit" in p.lower() or "sla penalt" in p.lower() for p in play_texts):
        return (
            "To manage performance risk, I'd like to incorporate service-level agreements "
            "with appropriate credits if delivery or quality targets aren't met. "
            "This ensures we have recourse mechanisms without damaging the relationship."
        )

    # Dual-sourcing plays
    if any("dual-sourc" in p.lower() or "alternative" in p.lower() for p in play_texts):
        return (
            "From a risk management perspective, we're evaluating dual-sourcing strategies "
            "to protect against supply disruption. If you can offer competitive terms and strong SLAs, "
            "you'd be well-positioned as our primary supplier with the volumes that come with that."
        )

    # Generic risk
    return (
        "We need to ensure appropriate risk mitigation through warranty coverage, "
        "performance guarantees, and clear remediation processes. "
        "Building these protections into the agreement benefits both parties."
    )


def weave_relational_play(plays: List[Dict[str, Any]], round_no: int, tone: str
) -> str:
    """Weave relational plays into natural language."""

    return (
        "Looking beyond this specific engagement, we value suppliers who can grow with us "
        "and become trusted partners. If this initial project goes well, there are "
        "significant opportunities for expanded business across our organization. "
        "That long-term potential should factor into how we structure the initial terms."
    )


def craft_generic_value_proposition(decision: Dict[str, Any], round_no: int
) -> str:
    """Fallback value proposition when playbook unavailable."""

    asks = decision.get("asks", [])
    if not asks:
        return ""

    if round_no <= 2:
        return (
            "To make this work within our budget parameters, I'd like to explore "
            "areas where we can create mutual value—whether through volume commitments, "
            "payment terms optimization, or longer-term partnership structures."
        )
    else:
        return (
            "As we work toward closure, I believe there are still opportunities "
            "to optimize the total package through creative structuring of terms, "
            "payment schedules, and service commitments."
        )


def craft_collaborative_asks(*,
    decision: Dict[str, Any],
    round_no: int,
    signals: Dict[str, Any],
    tone: str,
) -> str:
    """Frame asks as collaborative opportunities."""

    asks = decision.get("asks", [])
    lead_time_request = decision.get("lead_time_request")

    if not asks and not lead_time_request:
        return ""

    # Frame based on tone
    if tone in ("decisive", "closure-oriented"):
        intro = "To finalize this agreement, there are a few specific elements I'd like to confirm:"
    else:
        intro = "To help us structure the best possible outcome, I'd like to explore a few areas:"

    formatted_asks: List[str] = []

    # Convert asks into questions/proposals rather than demands
    for ask in asks[:4]:  # Limit to top 4
        ask_text = str(ask).strip()

        # Rephrase common asks to be more collaborative
        if "volume" in ask_text.lower() or "tier" in ask_text.lower():
            formatted_asks.append(
                "• What volume thresholds would unlock better unit economics? "
                "We're confident we can hit meaningful tiers if the pricing justifies it."
            )
        elif "payment" in ask_text.lower() and ("early" in ask_text.lower() or "discount" in ask_text.lower()):
            formatted_asks.append(
                "• Would accelerated payment (net-15) create value for you? "
                "We'd be happy to explore if that opens up pricing flexibility."
            )
        elif "lead time" in ask_text.lower() or "delivery" in ask_text.lower():
            formatted_asks.append(
                "• Can you confirm the delivery timeline and whether there's flexibility "
                "for partial shipments if that helps manage both our schedules?"
            )
        elif "warranty" in ask_text.lower():
            formatted_asks.append(
                "• What warranty coverage is included, and is there room to extend that "
                "as part of the overall package?"
            )
        elif "breakdown" in ask_text.lower() or "cost" in ask_text.lower():
            formatted_asks.append(
                "• Would you be open to sharing a high-level cost breakdown? "
                "That transparency helps us justify the investment internally."
            )
        elif "alternative" in ask_text.lower() or "spec" in ask_text.lower():
            formatted_asks.append(
                "• Are there alternative specifications or components that could reduce cost "
                "while still meeting our performance requirements?"
            )
        else:
            # Generic reframe
            formatted_asks.append(f"• {ask_text}")

    # Add lead time if specified
    if lead_time_request and not any("lead time" in fa.lower() for fa in formatted_asks):
        formatted_asks.append(
            f"• Regarding timing: {lead_time_request.lower()} would be ideal for our project schedule."
        )

    if not formatted_asks:
        return ""

    return intro + "\n\n" + "\n".join(formatted_asks)


def craft_closing(round_no: int, strategy: str, tone: str) -> str:
    """Create a closing that maintains momentum."""

    if strategy == "accept":
        return (
            "If we can align on these final points, I'm ready to move forward quickly "
            "and get the paperwork in motion. Looking forward to your thoughts."
        )

    if strategy == "decline":
        return (
            "I appreciate your engagement throughout this process. Given where we've landed, "
            "I'll need to take this back to my team for further discussion. "
            "I'll circle back if our parameters change."
        )

    if round_no >= 3:
        return (
            "I'm hopeful we can find common ground here. This represents our best position "
            "given all the factors at play. I'd appreciate your thoughts on whether we can "
            "make this work, and I'm happy to jump on a call if that would be helpful "
            "to talk through any remaining sticking points."
        )

    if round_no == 2:
        return (
            "I believe we're getting close to something that works for both sides. "
            "Let me know your thoughts on this structure, and we can refine from there. "
            "Happy to discuss any concerns or questions you might have."
        )

    # Round 1 or default
    return (
        "I'd welcome your perspective on this proposal. If you have questions or "
        "want to discuss any of these points in more detail, I'm happy to set up "
        "a quick call. Looking forward to your response."
    )


def normalise_supplier_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    mapping = {
        "transactional": "Transactional",
        "leverage": "Leverage",
        "strategic": "Strategic",
        "bottleneck": "Bottleneck",
    }
    for key, canonical in mapping.items():
        if key in lowered:
            return canonical
    return None


def normalise_negotiation_style(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    mapping = {
        "competitive": "Competitive",
        "collaborative": "Collaborative",
        "principled": "Principled",
        "accommodating": "Accommodating",
        "compromising": "Compromising",
    }
    for key, canonical in mapping.items():
        if key in lowered:
            return canonical
    return None


def craft_opening_simple(round_no: int, supplier_message: Optional[str], signals: Dict[str, Any]
) -> str:
    """Craft a simple opening."""
    if round_no > 1 and supplier_message:
        return (
            "Thank you for your response. I've reviewed your proposal and would like to discuss "
            "how we can align on the details."
        )
    return (
        "Thank you for your proposal. I'd like to discuss the terms to ensure we can reach an "
        "agreement that works for both parties."
    )


def craft_value_proposition_simple(playbook_context: Dict[str, Any], round_no: int
) -> str:
    """Craft simple value proposition from playbook."""
    plays = playbook_context.get("plays", [])[:2]

    if not plays:
        return ""

    statements: List[str] = []
    for play in plays:
        if not isinstance(play, dict):
            continue

        description = play.get("play", "")
        lowered = description.lower()
        if "volume" in lowered or "tier" in lowered:
            statements.append(
                "We'd like to explore volume commitments that could unlock better pricing for both sides."
            )
        elif "payment" in lowered and "early" in lowered:
            statements.append(
                "We can offer accelerated payment terms if that helps with pricing flexibility."
            )
        elif "warranty" in lowered:
            statements.append(
                "Extended warranty coverage would strengthen the business case on our side."
            )

    return " ".join(statements[:2]) if statements else ""


def craft_asks_simple(decision: Dict[str, Any]) -> str:
    """Craft simple asks section."""
    asks = decision.get("asks", [])
    if not asks:
        return ""

    asks_list = [f"• {ask}" for ask in asks[:4] if ask]
    if not asks_list:
        return ""

    return "To help structure the best outcome:\n" + "\n".join(asks_list)


def craft_closing_simple(round_no: int, strategy: str) -> str:
    """Craft simple closing."""
    if strategy == "accept":
        return (
            "If we can align on these final points, I'm ready to move forward. Looking forward to your thoughts."
        )

    if round_no >= 3:
        return (
            "I'm hopeful we can find common ground. Please let me know your thoughts, and I'm happy to discuss further if needed."
        )

    return (
        "I'd welcome your feedback on this proposal. Happy to discuss any questions you might have."
    )
