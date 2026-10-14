"""Negotiation maths, registered.

Counter pricing, Kraljic classification, play ranking and play readiness.
Delegates to ``src.agents.negotiation_agent`` and
``src.services.negotiation_advice``.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.agents.negotiation_agent import compute_decision as _compute_decision
from src.services.negotiation_advice import classification as _cls
from src.services.negotiation_advice import grounding as _gnd
from src.services.negotiation_advice import ranking as _rnk

from ..contract import (
    COUNT, CURRENCY_CODE, DAYS, GBP, LABEL, MONEY, PERCENT, RATIO, RECORD, ROWS,
    SECONDS, TEXT, Output, Term,
)
from ..registry import GoldenVector, formula

_OWNER = "negotiation"
_FROM = date(2026, 9, 5)

_PB = {
    "Leverage": {
        "descriptor": "Contested market, material spend.",
        "examples": ["e-auction"],
        "styles": {
            "Competitive": {
                "Commercial": ["Benchmark the price against the market",
                               "Run a competitive e-auction"],
                "Risk": ["Ask for a supply continuity plan"],
            }
        },
    }
}


@formula(
    "negotiation.counter_plan",
    version="1.0.0",
    owner=_OWNER,
    purpose="What to counter at, and whether to counter at all, given the round and the gap",
    effective_from=_FROM,
    inputs=[
        Term("current_offer", MONEY, "the supplier's live offer", minimum=0.0),
        Term("target_price", MONEY, "our target", minimum=0.0),
        Term("round", COUNT, "negotiation round, 1-based", minimum=1, required=False),
        Term("max_rounds", COUNT, "rounds before holding", minimum=1, required=False),
        Term("walkaway_price", MONEY, "no-deal threshold", minimum=0.0, required=False),
        Term("currency", CURRENCY_CODE, required=False),
        Term("ask_early_pay_disc", RATIO, "early-payment discount to ask for",
             minimum=0.0, maximum=1.0, required=False),
        Term("ask_lead_time_keep", LABEL, required=False),
        Term("supplier_message_text", TEXT,
             "the supplier's reply, scanned for final-offer language", required=False),
        Term("offer_prev", MONEY, "their previous offer", minimum=0.0, required=False),
    ],
    output=Output("dict", MONEY,
                  "decision, counter_price, asks, lead_time_request, message, log"),
    notes=(
        "`compute_decision` hardwires aggressiveness 0.75, leverage 0.6, urgency 0.3, "
        "risk_buffer_pct 0.06, min_abs_buffer 3.0, step_pct_of_gap 0.12 and "
        "ask_early_pay_disc 0.02, overriding the dataclass defaults. They are not "
        "governed and not configurable at the call site (gap report F22). Round "
        "behaviour: R1 anchors at x0.88 above a 10% gap else midpoint; R2 captures "
        "60% of the gap; R3+ enforces target + max(3.0, target x 6%)."
    ),
    golden=[
        GoldenVector(
            inputs={"current_offer": 100.0, "target_price": 80.0, "round": 1,
                    "max_rounds": None, "walkaway_price": None, "currency": None,
                    "ask_early_pay_disc": None, "ask_lead_time_keep": None,
                    "supplier_message_text": None, "offer_prev": None},
            expected={"decision": "counter", "counter_price": 88.0},
            note="R1 anchor, 25% gap -> x0.88",
        ),
        GoldenVector(
            inputs={"current_offer": 100.0, "target_price": 95.0, "round": 1,
                    "max_rounds": None, "walkaway_price": None, "currency": None,
                    "ask_early_pay_disc": None, "ask_lead_time_keep": None,
                    "supplier_message_text": None, "offer_prev": None},
            expected={"decision": "counter", "counter_price": 97.5},
            note="R1 narrow gap -> midpoint",
        ),
        GoldenVector(
            inputs={"current_offer": 100.0, "target_price": 80.0, "round": 2,
                    "max_rounds": None, "walkaway_price": None, "currency": None,
                    "ask_early_pay_disc": None, "ask_lead_time_keep": None,
                    "supplier_message_text": None, "offer_prev": 105.0},
            expected={"decision": "counter", "counter_price": 88.0},
            note="R2 assertive push captures 60% of the remaining gap",
        ),
        GoldenVector(
            inputs={"current_offer": 100.0, "target_price": 80.0, "round": 3,
                    "max_rounds": None, "walkaway_price": None, "currency": None,
                    "ask_early_pay_disc": None, "ask_lead_time_keep": None,
                    "supplier_message_text": None, "offer_prev": 102.0},
            expected={"decision": "counter", "counter_price": 84.8},
            note="R3+ buffer: target + max(3.0, 80 x 0.06) = 84.8",
        ),
        GoldenVector(
            inputs={"current_offer": 100.0, "target_price": 80.0, "round": 4,
                    "max_rounds": None, "walkaway_price": None, "currency": None,
                    "ask_early_pay_disc": None, "ask_lead_time_keep": None,
                    "supplier_message_text": None, "offer_prev": None},
            expected={"decision": "hold", "counter_price": 100.0},
            note="past max_rounds -- hold, do not concede further",
        ),
        GoldenVector(
            inputs={"current_offer": 79.0, "target_price": 80.0, "round": 2,
                    "max_rounds": None, "walkaway_price": None, "currency": None,
                    "ask_early_pay_disc": None, "ask_lead_time_keep": None,
                    "supplier_message_text": "this is our final offer",
                    "offer_prev": None},
            expected={"decision": "accept", "counter_price": 79.0, "finality": True},
        ),
        GoldenVector(
            inputs={"current_offer": 120.0, "target_price": 80.0, "round": 2,
                    "max_rounds": None, "walkaway_price": None, "currency": None,
                    "ask_early_pay_disc": None, "ask_lead_time_keep": None,
                    "supplier_message_text": "this is our final offer",
                    "offer_prev": None},
            expected={"decision": "decline", "counter_price": None, "finality": True},
            note="final offer above the threshold -- escalate, never split the difference",
        ),
        GoldenVector(
            inputs={"current_offer": 0.0, "target_price": 80.0, "round": 1,
                    "max_rounds": None, "walkaway_price": None, "currency": None,
                    "ask_early_pay_disc": None, "ask_lead_time_keep": None,
                    "supplier_message_text": None, "offer_prev": None},
            expected={"decision": "clarify", "counter_price": None},
            note="no usable price -- ask, do not guess",
        ),
    ],
)
def counter_plan(current_offer, target_price, round=None, max_rounds=None,
                 walkaway_price=None, currency=None, ask_early_pay_disc=None,
                 ask_lead_time_keep=None, supplier_message_text=None, offer_prev=None):
    payload: Dict[str, Any] = {
        "current_offer": current_offer,
        "target_price": target_price,
    }
    if round is not None:
        payload["round"] = int(round)
    if max_rounds is not None:
        payload["max_rounds"] = int(max_rounds)
    if walkaway_price is not None:
        payload["walkaway_price"] = walkaway_price
    if currency is not None:
        payload["currency"] = currency
    if ask_early_pay_disc is not None:
        payload["ask_early_pay_disc"] = ask_early_pay_disc
    if ask_lead_time_keep is not None:
        payload["ask_lead_time_keep"] = bool(ask_lead_time_keep)
    return _compute_decision(payload, supplier_message_text or "", offer_prev)


@formula(
    "negotiation.kraljic_quadrant",
    version="1.0.0",
    owner=_OWNER,
    purpose="Kraljic quadrant and suggested negotiation style for a deal",
    effective_from=_FROM,
    inputs=[
        Term("signals", RECORD,
             "deal_value, alternative_supplier_count, risk_score, is_preferred, "
             "price_variance_pct"),
        Term("thresholds", RECORD, "high_spend and many_alternatives overrides",
             required=False),
    ],
    output=Output("dict", LABEL,
                  "quadrant, reasons, confidence, style, style_reasons, indeterminate"),
    notes=(
        "Default bars `high_spend = 98,175` and `many_alternatives = 93` are the live "
        "deal-value p90 and per-deal median alternative count. They are distribution "
        "parameters compiled into source: when the corpus moves they go stale silently "
        "and nothing measures the drift (gap report D-10). `many_alternatives = 93` is "
        "also defined a second time as `signals.THIN_MARKET_ALTERNATIVES` (D-11). "
        "Note bp_supplier.supplier_type is NOT a Kraljic axis and is deliberately unused."
    ),
    golden=[
        GoldenVector(
            inputs={"signals": {"deal_value": 250000.0, "alternative_supplier_count": 150,
                                "risk_score": 40.0, "is_preferred": False,
                                "price_variance_pct": 8.0},
                    "thresholds": None},
            expected={"quadrant": "Leverage", "style": "Competitive",
                      "quadrant_confidence": 1.0, "indeterminate": False},
        ),
        GoldenVector(
            inputs={"signals": {"deal_value": 4000.0, "alternative_supplier_count": 12,
                                "risk_score": 70.0}, "thresholds": None},
            expected={"quadrant": "Bottleneck", "style": "Principled"},
        ),
        GoldenVector(
            inputs={"signals": {"deal_value": 250000.0, "alternative_supplier_count": 12,
                                "is_preferred": True}, "thresholds": None},
            expected={"quadrant": "Strategic", "style": "Collaborative"},
        ),
        GoldenVector(
            inputs={"signals": {"deal_value": 4000.0,
                                "alternative_supplier_count": 150}, "thresholds": None},
            expected={"quadrant": "Transactional", "style": "Competitive"},
        ),
        GoldenVector(
            inputs={"signals": {"deal_value": None, "alternative_supplier_count": None},
                    "thresholds": None},
            expected={"quadrant": None, "indeterminate": True, "quadrant_confidence": 0.0},
            note="no spend and no market -- says so rather than guessing a quadrant",
        ),
    ],
)
def kraljic_quadrant(signals: Mapping[str, Any],
                     thresholds: Optional[Mapping[str, Any]] = None) -> dict:
    return _cls.classify(dict(signals or {}), dict(thresholds) if thresholds else None)


@formula(
    "negotiation.threshold_confidence",
    version="1.0.0",
    owner=_OWNER,
    purpose="How far a value sits from a classification bar, as a 0.5-1.0 confidence",
    effective_from=_FROM,
    inputs=[
        Term("value", RATIO, "the measured value", required=False),
        Term("bar", RATIO, "the threshold it is being tested against", required=False),
    ],
    output=Output("float", RATIO, "0.5 at the bar, rising to 1.0 away from it"),
    golden=[
        GoldenVector(inputs={"value": 250000.0, "bar": 98175.0}, expected=1.0),
        GoldenVector(inputs={"value": 98175.0, "bar": 98175.0}, expected=0.5,
                     note="exactly on the bar is a coin toss, and says so"),
        GoldenVector(inputs={"value": 10.0, "bar": 0.0}, expected=0.5,
                     note="a zero bar cannot discriminate"),
    ],
)
def threshold_confidence(value=None, bar=None) -> float:
    return _cls._confidence(float(value or 0.0), float(bar or 0.0))


@formula(
    "negotiation.play_rank",
    version="1.0.0",
    owner=_OWNER,
    purpose="Rank playbook plays for a (supplier type, style) pair against live signals",
    effective_from=_FROM,
    inputs=[
        Term("supplier_type", LABEL, "Kraljic quadrant", required=False),
        Term("negotiation_style", LABEL, required=False),
        Term("lever_priorities", ROWS, required=False),
        Term("policy_guidance", RECORD, "required/preferred/discouraged/restricted lever sets",
             required=False),
        Term("supplier_performance", RECORD, required=False),
        Term("market_context", RECORD, required=False),
        Term("playbook", RECORD, "the playbook to rank within; loaded from disk if omitted",
             required=False),
        Term("limit", COUNT, "how many plays to return", minimum=1, required=False),
    ],
    output=Output("dict", RATIO, "ranked plays with scores, rationale and trade-offs"),
    notes=(
        "Score = 1.0 + index x 0.01 + policy + performance + market. The index term is "
        "a tie-break that preserves playbook order, not a judgement."
    ),
    golden=[
        GoldenVector(
            inputs={"supplier_type": "Leverage", "negotiation_style": "Competitive",
                    "lever_priorities": None,
                    "policy_guidance": {"required": {"Commercial"}},
                    "supplier_performance": {"on_time_delivery": 0.8},
                    "market_context": {"supply_risk": "elevated"},
                    "playbook": _PB, "limit": None},
            expected={"plays": [
                {"play": "Run a competitive e-auction", "score": 1.61, "lever": "Commercial"},
                {"play": "Benchmark the price against the market", "score": 1.6},
                {"play": "Ask for a supply continuity plan", "score": 1.5, "lever": "Risk"},
            ]},
            note="policy +0.6 on Commercial; market +0.3 lands only on the Risk lever",
        ),
        GoldenVector(
            inputs={"supplier_type": "Nope", "negotiation_style": "Competitive",
                    "lever_priorities": None, "policy_guidance": None,
                    "supplier_performance": None, "market_context": None,
                    "playbook": _PB, "limit": None},
            expected={"plays": [], "lever_priorities": []},
            note="unknown supplier type yields no plays, not generic ones",
        ),
    ],
)
def play_rank(supplier_type=None, negotiation_style=None, lever_priorities=None,
              policy_guidance=None, supplier_performance=None, market_context=None,
              playbook=None, limit=None):
    kwargs: Dict[str, Any] = {
        "lever_priorities": lever_priorities,
        "policy_guidance": policy_guidance,
        "supplier_performance": supplier_performance,
        "market_context": market_context,
        "playbook": playbook,
    }
    if limit is not None:
        kwargs["limit"] = int(limit)
    return _rnk.rank_plays(supplier_type, negotiation_style, **kwargs)


@formula(
    "negotiation.policy_alignment_score",
    version="1.0.0",
    owner=_OWNER,
    purpose="How far governed policy pushes for or against one negotiation lever",
    effective_from=_FROM,
    inputs=[
        Term("lever", LABEL, "Commercial | Operational | Risk | Relational | Strategic"),
        Term("guidance", RECORD, "required / preferred / discouraged / restricted lever sets",
             required=False),
    ],
    output=Output("tuple", RATIO, "(score nudge, human-readable notes)"),
    notes="Nudges are +0.6 required, +0.3 preferred, -0.3 discouraged, -0.7 restricted.",
    golden=[
        GoldenVector(inputs={"lever": "Commercial", "guidance": {"required": {"Commercial"}}},
                     expected=(0.6, ["Required by policy"])),
        GoldenVector(
            inputs={"lever": "Commercial",
                    "guidance": {"restricted": {"Commercial"}, "preferred": {"Commercial"}}},
            expected=(-0.39999999999999997,
                      ["Policy prefers this lever", "Policy restricts this lever"]),
            note="contradictory guidance nets out rather than erroring",
        ),
        GoldenVector(inputs={"lever": "Commercial", "guidance": {}}, expected=(0.0, [])),
    ],
)
def policy_alignment_score(lever: str, guidance=None):
    return _rnk._score_policy_alignment(lever, dict(guidance or {}))


@formula(
    "negotiation.supplier_performance_score",
    version="1.0.0",
    owner=_OWNER,
    purpose="How far a supplier's measured performance argues for one lever",
    effective_from=_FROM,
    inputs=[
        Term("lever", LABEL),
        Term("performance", RECORD, "on_time_delivery, defect_rate, esg_score, ...",
             required=False),
    ],
    output=Output("tuple", RATIO, "(score nudge, notes)"),
    notes=(
        "An empty performance dict yields (0.0, []) -- the honest 'no signal' outcome. "
        "`negotiation_advice.signals.supplier_performance_dict` deliberately omits "
        "unknown keys rather than defaulting them, so an unmeasured supplier produces "
        "no nudge instead of an invented one."
    ),
    golden=[
        GoldenVector(inputs={"lever": "Operational", "performance": {"on_time_delivery": 0.8}},
                     expected=(0.4, ["On-time delivery below 90%"])),
        GoldenVector(inputs={"lever": "Operational", "performance": {"on_time_delivery": 0.99}},
                     expected=(-0.1, ["Delivery reliability already strong"])),
        GoldenVector(inputs={"lever": "Operational", "performance": {}}, expected=(0.0, [])),
    ],
)
def supplier_performance_score(lever: str, performance=None):
    return _rnk._score_supplier_performance(lever, dict(performance or {}))


@formula(
    "negotiation.market_context_score",
    version="1.0.0",
    owner=_OWNER,
    purpose="How far market conditions argue for one lever",
    effective_from=_FROM,
    inputs=[
        Term("lever", LABEL),
        Term("market", RECORD, "supply_risk, demand_trend, inflation, capacity, esg_pressure",
             required=False),
    ],
    output=Output("tuple", RATIO, "(score nudge, notes)"),
    golden=[
        GoldenVector(inputs={"lever": "Risk", "market": {"supply_risk": "elevated"}},
                     expected=(0.3, ["Market supply risk elevated"])),
        GoldenVector(inputs={"lever": "Commercial", "market": {}}, expected=(0.0, [])),
    ],
)
def market_context_score(lever: str, market=None):
    return _rnk._score_market_context(lever, dict(market or {}))


@formula(
    "negotiation.play_readiness",
    version="1.0.0",
    owner=_OWNER,
    purpose="Whether a play's precondition actually holds on this deal's evidence",
    effective_from=_FROM,
    inputs=[
        Term("play", RECORD, "the play being tested"),
        Term("signals", RECORD, "the deal's own evidence", required=False),
    ],
    output=Output("dict", LABEL, "the play plus state (ready|groundwork|not_applicable)"),
    notes=(
        "A tri-state that already existed before this registry did, and the closest "
        "prior art in the codebase to UNASSESSED. 'Leverage competitor quotes' is "
        "strong with a second quote and damaging without one -- a buyer who bluffs a "
        "competing quote they do not have loses standing. A known sole-source deal "
        "returns not_applicable rather than groundwork: there is no second supplier to "
        "go and find, so advising a competitive event would be noise."
    ),
    golden=[
        GoldenVector(
            inputs={"play": {"play": "Leverage competitor quotes to pressure pricing",
                             "score": 1.0},
                    "signals": {"quote_supplier_count": 3, "alternative_supplier_count": 5}},
            expected={"state": "ready", "family": "competitive_tension"},
        ),
        GoldenVector(
            inputs={"play": {"play": "Leverage competitor quotes to pressure pricing",
                             "score": 1.0},
                    "signals": {"quote_supplier_count": 1, "alternative_supplier_count": 1}},
            expected={"state": "not_applicable"},
        ),
        GoldenVector(
            inputs={"play": {"play": "Request a refund for non-compliant charges",
                             "score": 1.0},
                    "signals": {"invoice_total": 1200.0, "po_total": 1000.0}},
            expected={"state": "ready", "family": "overbilling"},
        ),
        GoldenVector(
            inputs={"play": {"play": "Hold a supplier day", "score": 1.0}, "signals": {}},
            expected={"state": "ready", "family": None},
            note="a play with no testable precondition is ready by default",
        ),
    ],
)
def play_readiness(play: Mapping[str, Any], signals=None) -> dict:
    return _gnd.assess(dict(play), dict(signals or {}))


# ---------------------------------------------------------------------------
# PROVISIONAL: the ZOPA and package-optimiser maths.
#
# These four are registered so they are visible, versioned and diffable --- not
# because their behaviour is validated. Their golden vectors pin *arithmetic*,
# never *correctness*: each computes against inputs that Track B does not
# supply (see docs/adr/0001-negotiation-track-b-deferral.md). A vector here
# says "this is what the code does today", not "this is the right answer".
# They must not be cited as evidence that a negotiation number is sound.
#
# The methods are pure --- they touch `self` only to reach `_coerce_float` and
# `_validate_buyer_max`, both of which are themselves pure --- so they are
# bound to an uninitialised instance rather than requiring a live agent.
# ---------------------------------------------------------------------------

_PROVISIONAL = (
    "PROVISIONAL -- vectors pin current arithmetic, not validated behaviour. "
)


def _bare_agent():
    """A NegotiationAgent with no __init__: enough for the pure methods."""
    from src.agents.negotiation_agent import NegotiationAgent

    return NegotiationAgent.__new__(NegotiationAgent)


@formula(
    "negotiation.zopa_estimate",
    version="2.0.0",
    owner=_OWNER,
    purpose="Buyer's ceiling, the supplier's cost floor if any evidence exists, and an entry counter",
    effective_from=_FROM,
    inputs=[
        Term("price", MONEY, "the supplier's live offer", minimum=0.0, required=False),
        Term("target", MONEY, "our target", minimum=0.0, required=False),
        Term("history", RECORD, "min_accepted_price for this supplier", required=False),
        Term("benchmarks", RECORD, "p10 or low", required=False),
        Term("should_cost", MONEY, "a costed floor, if one exists", minimum=0.0,
             required=False),
        Term("signals", RECORD, "capacity_tight, tone, concession_band_pct",
             required=False),
    ],
    output=Output("dict", MONEY,
                  "buyer_max, supplier_floor, supplier_floor_basis, entry_counter, findings"),
    notes=(
        _PROVISIONAL
        + "v2.0.0 REMOVED the `price * 0.85` fallback floor (authorised behaviour "
        "change, 2026-09-05): with no should-cost, no benchmark and no history the "
        "floor is None and a finding is emitted, where it previously invented a "
        "number from the supplier's own offer. Nothing in this repository produces "
        "a should_cost, so the None branch is the ordinary case. `entry_counter` "
        "remains heuristic: price x (1 - clamp(concession, 0.03, 0.12)), default "
        "concession 0.05, all four constants ungoverned."
    ),
    replaces=(),
    golden=[
        GoldenVector(
            inputs={"price": 100.0, "target": 80.0, "history": None,
                    "benchmarks": None, "should_cost": None, "signals": None},
            expected={"buyer_max": 80.0, "supplier_floor": None,
                      "supplier_floor_basis": None},
            note="PROVISIONAL: no cost evidence -> no floor, not 85.0",
        ),
        GoldenVector(
            inputs={"price": 100.0, "target": 80.0, "history": None,
                    "benchmarks": None, "should_cost": 62.0, "signals": None},
            expected={"supplier_floor": 62.0, "supplier_floor_basis": "should_cost"},
            note="PROVISIONAL: a costed floor is used and named",
        ),
        GoldenVector(
            inputs={"price": 100.0, "target": 80.0, "history": None,
                    "benchmarks": {"p10": 70.0}, "should_cost": None, "signals": None},
            expected={"supplier_floor": 70.0, "supplier_floor_basis": "benchmark_p10"},
            note="PROVISIONAL: benchmark p10 is a point estimate with no sample size",
        ),
    ],
)
def zopa_estimate(price=None, target=None, history=None, benchmarks=None,
                  should_cost=None, signals=None) -> dict:
    return _bare_agent()._estimate_zopa(
        price=price, target=target, history=dict(history or {}),
        benchmarks=dict(benchmarks or {}), should_cost=should_cost,
        signals=dict(signals or {}),
    )


@formula(
    "negotiation.outlier_rails",
    version="1.0.0",
    owner=_OWNER,
    purpose="Whether an offer breaches a review or escalation rail on price, volume or term",
    effective_from=_FROM,
    inputs=[
        Term("supplier_offer", MONEY, minimum=0.0, required=False),
        Term("target_price", MONEY, minimum=0.0, required=False),
        Term("walkaway_price", MONEY, "our no-deal threshold", minimum=0.0,
             required=False),
        Term("market_floor", MONEY, "lowest market reference", minimum=0.0,
             required=False),
        Term("volume_units", COUNT, minimum=0.0, required=False),
        Term("term_days", DAYS, "requested payment term", minimum=0, required=False),
    ],
    output=Output("dict", LABEL,
                  "requires_review, human_override, review_recommendation, alerts"),
    notes=(
        _PROVISIONAL
        + "Both price rails test (reference - offer)/reference, so they fire when the "
        "supplier's offer is BELOW the reference -- an error/fraud check, not an "
        "authority limit. Nothing here stops US emitting a counter or an acceptance "
        "above the walk-away; `plan_counter` clamps to target_price, never to "
        "walkaway_price (audit 1.6). Constants NEG_MARKET_REVIEW_PCT 0.2, "
        "NEG_MARKET_ESCALATION_PCT 0.4, NEG_MAX_VOLUME_LIMIT 1000, NEG_MAX_TERM_DAYS "
        "120, with x1.5 and x2 escalation multipliers. The escalation log line reads "
        "'more than 20% below our walk-away price' while the constant tested is 0.4."
    ),
    golden=[
        GoldenVector(
            inputs={"supplier_offer": 50.0, "target_price": 80.0,
                    "walkaway_price": 90.0, "market_floor": 100.0,
                    "volume_units": None, "term_days": None},
            expected={"requires_review": True, "human_override": True},
            note="PROVISIONAL: 50% below market floor clears both rails",
        ),
        GoldenVector(
            inputs={"supplier_offer": 95.0, "target_price": 80.0,
                    "walkaway_price": None, "market_floor": 100.0,
                    "volume_units": None, "term_days": None},
            expected={"requires_review": False, "human_override": False},
            note="PROVISIONAL: a 5% gap is inside the rails",
        ),
        GoldenVector(
            inputs={"supplier_offer": 95.0, "target_price": 80.0,
                    "walkaway_price": None, "market_floor": None,
                    "volume_units": 1500.0, "term_days": 200},
            expected={"requires_review": True, "human_override": False},
            note=("PROVISIONAL: both breach their review rail, neither escalates. "
                  "Volume 1500 is exactly 1000 x 1.5 and the test is strict >, so "
                  "the escalation boundary is unreachable at the round number a "
                  "buyer is most likely to enter."),
        ),
        GoldenVector(
            inputs={"supplier_offer": 95.0, "target_price": 80.0,
                    "walkaway_price": None, "market_floor": None,
                    "volume_units": 1600.0, "term_days": 250},
            expected={"requires_review": True, "human_override": True},
            note="PROVISIONAL: past the boundary, both escalate",
        ),
        GoldenVector(
            inputs={"supplier_offer": 200.0, "target_price": 80.0,
                    "walkaway_price": 90.0, "market_floor": 100.0,
                    "volume_units": None, "term_days": None},
            expected={"requires_review": False},
            note=("PROVISIONAL AND WRONG-LOOKING BY DESIGN: an offer at 2.2x the "
                  "walk-away raises nothing, because these rails only look "
                  "downward. Pinned so the gap is visible, not because it is right."),
        ),
    ],
)
def outlier_rails(supplier_offer=None, target_price=None, walkaway_price=None,
                  market_floor=None, volume_units=None, term_days=None) -> dict:
    out = _bare_agent()._detect_outliers(
        supplier_offer=supplier_offer, target_price=target_price,
        walkaway_price=walkaway_price, market_floor=market_floor,
        volume_units=volume_units, term_days=term_days,
    )
    return out


@formula(
    "negotiation.batna_strength",
    version="1.0.0",
    owner=_OWNER,
    purpose="The buyer's walk-away position from alternative quotes and trading history",
    effective_from=_FROM,
    inputs=[
        Term("alternative_quotes", COUNT,
             "qualified alternative supplier quotes held", minimum=0, required=False),
        Term("supplier_history_count", COUNT,
             "prior orders placed with this supplier", minimum=0, required=False),
    ],
    output=Output("dict", LABEL, "strength, score, confidence, narrative, findings"),
    notes=(
        "Salvaged from NegotiationStrategyEngine._build_batna and select_strategy "
        "before that module was deleted (it was constructed at every boot and "
        "reachable from nothing). Two corrections in the move: a missing count is "
        "UNASSESSED rather than 0, and no-BATNA scores 0.0 rather than selecting "
        "the engine's most aggressive strategy (STRATEGY_ANCHORING, target_discount "
        "0.15). Bars: 2 alternatives for strong, 5 orders for an established "
        "relationship -- both inherited from the engine and ungoverned."
    ),
    golden=[
        GoldenVector(
            inputs={"alternative_quotes": 2, "supplier_history_count": 0},
            expected={"strength": "strong", "score": 1.0},
        ),
        GoldenVector(
            inputs={"alternative_quotes": 1, "supplier_history_count": 0},
            expected={"strength": "moderate", "score": 0.5},
        ),
        GoldenVector(
            inputs={"alternative_quotes": 0, "supplier_history_count": 7},
            expected={"strength": "weak", "score": 0.2},
        ),
        GoldenVector(
            inputs={"alternative_quotes": 0, "supplier_history_count": 0},
            expected={"strength": "none", "score": 0.0},
            note="nowhere else to go is the weakest position, not a licence to anchor",
        ),
        GoldenVector(
            inputs={"alternative_quotes": None, "supplier_history_count": None},
            expected={"strength": "UNASSESSED"},
            note="never looked is not looked-and-found-none",
        ),
    ],
)
def batna_strength(alternative_quotes=None, supplier_history_count=None) -> dict:
    from src.services.negotiation.leverage import assess_batna

    out = assess_batna(alternative_quotes=alternative_quotes,
                       supplier_history_count=supplier_history_count)
    return {
        "strength": str(out.strength),
        "score": str(out.score) if not out.is_assessed else out.score,
        "confidence": out.confidence.value,
        "narrative": out.narrative,
        "findings": list(out.findings),
    }
