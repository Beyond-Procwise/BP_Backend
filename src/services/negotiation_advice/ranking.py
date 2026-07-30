"""Play ranking — which negotiation plays to run, and why.

Moved verbatim out of ``NegotiationAgent`` (a ~540KB module) so the advisor can
rank plays without instantiating an agent or holding an ``AgentContext``. The
scoring is unchanged: this file is a relocation, not a rewrite, and the bodies
below were extracted mechanically rather than retyped.

The split is by dependency, not by taste. Everything here is pure — it takes the
already-resolved supplier type, style, lever priorities and signal dicts, and
returns plays. The parts that read an ``AgentContext`` stay on the agent:
``_normalise_supplier_type``, ``_normalise_negotiation_style``,
``_extract_policy_guidance`` and ``_resolve_lever_priorities``.

Scoring contract, preserved exactly:
    base_score = 1.0 + (idx * 0.01)   # idx = position within its lever's list
    total      = base + policy + performance + market
    sort by (-score, lever, play), then take the first `limit`.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# NOTE the depth: PLAYBOOK_PATH resolved from src/agents/ used `parent.parent`.
# From src/services/negotiation_advice/ the same file is `parents[2]`. Get this
# wrong and load_playbook() returns {} and every play silently disappears.
PLAYBOOK_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "reference_data" / "negotiation_playbook.json"
)

LEVER_CATEGORIES = {
    "COMMERCIAL",
    "OPERATIONAL",
    "RISK",
    "STRATEGIC",
    "RELATIONAL",
}

TRADE_OFF_HINTS = {
    "Commercial": "May require volume commitments, stepped pricing, or altered cash flow.",
    "Operational": "May reduce supplier flexibility or need shared planning resources.",
    "Risk": "Could introduce legal negotiation overhead or stricter enforcement costs.",
    "Strategic": "Requires executive sponsorship and potential co-investment or exclusivity.",
    "Relational": "Demands governance time and tighter alignment of internal stakeholders.",
}

_playbook_cache: Optional[Dict[str, Any]] = None


def load_playbook(path: Optional[Path] = None) -> Dict[str, Any]:
    """The playbook, with its lever keys normalised to the expected casing.

    Body copied from ``NegotiationAgent._load_playbook``. The agent's
    per-instance ``_playbook_cache`` becomes a module-level cache; passing an
    explicit ``path`` bypasses it, so loading an alternative file cannot poison
    the shared one.
    """
    global _playbook_cache
    source = Path(path) if path else PLAYBOOK_PATH
    use_cache = path is None
    if use_cache and _playbook_cache is not None:
        return _playbook_cache
    try:
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        # Normalise lever keys to expected casing for faster lookups.
        for supplier_type, entry in data.items():
            styles = entry.get("styles")
            if not isinstance(styles, dict):
                continue
            for style_key, style_entry in list(styles.items()):
                if not isinstance(style_entry, dict):
                    continue
                normalised_style_entry: Dict[str, Any] = {}
                for lever_key, plays in style_entry.items():
                    lever_name = normalise_lever_category(lever_key)
                    if not lever_name:
                        continue
                    normalised_style_entry[lever_name] = plays
                styles[style_key] = normalised_style_entry
        if use_cache:
            _playbook_cache = data
        return data
    except FileNotFoundError:
        logger.warning("Negotiation playbook file missing at %s", source)
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to load negotiation playbook from %s", source)
    if use_cache:
        _playbook_cache = {}
    return {}


def normalise_lever_category(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    cleaned = re.sub(r"[^a-zA-Z]+", " ", text).strip().upper()
    if not cleaned:
        return None
    for category in LEVER_CATEGORIES:
        if cleaned == category or category in cleaned.split():
            return category.title()
    return None


def _score_policy_alignment(
    lever: str, guidance: Dict[str, Set[str]]
) -> Tuple[float, List[str]]:
    notes: List[str] = []
    score = 0.0
    if lever in guidance.get("required", set()):
        score += 0.6
        notes.append("Required by policy")
    if lever in guidance.get("preferred", set()):
        score += 0.3
        notes.append("Policy prefers this lever")
    if lever in guidance.get("discouraged", set()):
        score -= 0.3
        notes.append("Policy discourages this lever")
    if lever in guidance.get("restricted", set()):
        score -= 0.7
        notes.append("Policy restricts this lever")
    return score, notes


def _score_supplier_performance(
    lever: str, performance: Dict[str, Any]
) -> Tuple[float, List[str]]:
    score = 0.0
    notes: List[str] = []
    if not performance:
        return score, notes

    def _coerce_float_metric(*keys: str) -> Optional[float]:
        for key in keys:
            value = performance.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    on_time = _coerce_float_metric("on_time_delivery", "on_time", "delivery_score", "otif")
    if on_time is not None:
        if on_time < 0.9 and lever in {"Operational", "Risk"}:
            score += 0.4 if lever == "Operational" else 0.2
            notes.append("On-time delivery below 90%")
        elif on_time > 0.97 and lever == "Operational":
            score -= 0.1
            notes.append("Delivery reliability already strong")

    defect_rate = _coerce_float_metric("quality_incidents", "defect_rate", "return_rate")
    if defect_rate is not None and defect_rate > 0:
        if lever == "Risk":
            score += 0.3
        notes.append("Quality issues detected")

    esg_score = _coerce_float_metric("esg_score", "sustainability_score")
    if esg_score is not None and esg_score < 0.6 and lever == "Strategic":
        score += 0.2
        notes.append("ESG performance lagging")

    collaboration_score = _coerce_float_metric("relationship_score", "collaboration_index")
    if collaboration_score is not None and collaboration_score < 0.6 and lever == "Relational":
        score += 0.3
        notes.append("Relationship maturity low")

    innovation_score = _coerce_float_metric("innovation_score", "co_innovation_index")
    if innovation_score is not None and innovation_score < 0.5 and lever == "Strategic":
        score += 0.25
        notes.append("Innovation potential needs reinforcement")

    return score, notes


def _score_market_context(
    lever: str, market: Dict[str, Any]
) -> Tuple[float, List[str]]:
    score = 0.0
    notes: List[str] = []
    if not market:
        return score, notes

    supply_risk = market.get("supply_risk") or market.get("supply_risk_level")
    if isinstance(supply_risk, str) and supply_risk.strip().lower() in {"high", "elevated", "tight"}:
        if lever == "Risk":
            score += 0.3
        notes.append("Market supply risk elevated")

    demand_trend = market.get("demand_trend") or market.get("demand")
    if isinstance(demand_trend, str) and demand_trend.strip().lower() in {"rising", "high"}:
        if lever == "Commercial":
            score += 0.2
        notes.append("Demand is rising")

    inflation = market.get("inflation") or market.get("price_trend")
    if isinstance(inflation, str) and inflation.strip().lower() in {"inflationary", "increasing"}:
        if lever == "Commercial":
            score += 0.25
        notes.append("Prices trending upward")

    capacity = market.get("capacity") or market.get("capacity_constraints")
    if isinstance(capacity, str) and capacity.strip().lower() in {"limited", "constrained"}:
        if lever in {"Operational", "Strategic"}:
            score += 0.2
        notes.append("Capacity constraints present")

    esg_pressure = market.get("esg_pressure") or market.get("regulatory_focus")
    if isinstance(esg_pressure, str) and esg_pressure.strip().lower() in {"high", "tightening"}:
        if lever == "Strategic":
            score += 0.2
        notes.append("ESG expectations increasing")

    return score, notes


def _compose_play_rationale(
    descriptor: Optional[str],
    style: Optional[str],
    lever: str,
    policy_notes: List[str],
    performance_notes: List[str],
    market_notes: List[str],
) -> str:
    segments: List[str] = []
    if descriptor:
        segments.append(str(descriptor))
    if style:
        segments.append(f"Supports {style.lower()} posture on the {lever.lower()} lever.")
    else:
        segments.append(f"Targets the {lever.lower()} lever.")
    if policy_notes:
        segments.append("Policy: " + "; ".join(policy_notes))
    if performance_notes:
        segments.append("Performance: " + "; ".join(performance_notes))
    if market_notes:
        segments.append("Market: " + "; ".join(market_notes))
    return " ".join(segments)


def rank_plays(
    supplier_type: Optional[str],
    negotiation_style: Optional[str],
    *,
    lever_priorities: Optional[Sequence[str]] = None,
    policy_guidance: Optional[Dict[str, Set[str]]] = None,
    supplier_performance: Optional[Dict[str, Any]] = None,
    market_context: Optional[Dict[str, Any]] = None,
    playbook: Optional[Dict[str, Any]] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """Rank the plays for a (supplier type, style) pair.

    The scoring half of ``NegotiationAgent._resolve_playbook_context``, copied
    verbatim. The early returns match it exactly, including the empty
    ``{"plays": [], "lever_priorities": []}`` shape for an unknown supplier type
    and the descriptor-bearing shape for an unknown style — the agent's callers
    already branch on those.
    """
    playbook = playbook if playbook is not None else load_playbook()
    if not playbook:
        return {"plays": [], "lever_priorities": []}

    if not supplier_type or supplier_type not in playbook:
        return {"plays": [], "lever_priorities": []}

    supplier_entry = playbook.get(supplier_type, {})
    styles = supplier_entry.get("styles", {})
    if not negotiation_style or negotiation_style not in styles:
        return {
            "plays": [],
            "descriptor": supplier_entry.get("descriptor"),
            "examples": supplier_entry.get("examples", []),
            "lever_priorities": [],
            "style": negotiation_style,
            "supplier_type": supplier_type,
        }

    lever_priorities = list(lever_priorities or [])
    if not lever_priorities:
        lever_priorities = list(styles[negotiation_style].keys())

    policy_guidance = policy_guidance if isinstance(policy_guidance, dict) else {}
    if not isinstance(supplier_performance, dict):
        supplier_performance = {}
    if not isinstance(market_context, dict):
        market_context = {}

    plays: List[Dict[str, Any]] = []
    style_plays = styles[negotiation_style]
    for lever in lever_priorities:
        lever_plays = style_plays.get(lever)
        if not isinstance(lever_plays, list):
            continue
        for idx, play_text in enumerate(lever_plays):
            if not isinstance(play_text, str) or not play_text.strip():
                continue
            base_score = 1.0 + (idx * 0.01)
            policy_score, policy_notes = _score_policy_alignment(lever, policy_guidance)
            performance_score, performance_notes = _score_supplier_performance(
                lever, supplier_performance
            )
            market_score, market_notes = _score_market_context(lever, market_context)
            total_score = base_score + policy_score + performance_score + market_score
            rationale = _compose_play_rationale(
                supplier_entry.get("descriptor"),
                negotiation_style,
                lever,
                policy_notes,
                performance_notes,
                market_notes,
            )
            plays.append(
                {
                    "supplier_type": supplier_type,
                    "style": negotiation_style,
                    "lever": lever,
                    "play": play_text.strip(),
                    "score": round(total_score, 4),
                    "policy_alignment": policy_notes,
                    "performance_signals": performance_notes,
                    "market_signals": market_notes,
                    "rationale": rationale,
                    "trade_offs": TRADE_OFF_HINTS.get(
                        lever, "Monitor implementation impact across stakeholders."
                    ),
                }
            )

    plays.sort(key=lambda item: (-item["score"], item.get("lever", ""), item.get("play", "")))
    top_plays = plays[:limit]

    return {
        "plays": top_plays,
        "descriptor": supplier_entry.get("descriptor"),
        "examples": supplier_entry.get("examples", []),
        "style": negotiation_style,
        "supplier_type": supplier_type,
        "lever_priorities": lever_priorities,
    }
