"""Compose an advice payload, and apply one conversational turn.

Buyer-stated facts are merged over measured signals for classification only.
The measured value stays in `signals`; the stated value is echoed in
`stated_facts` so the UI can label its provenance and the buyer can withdraw it.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.services.db import get_conn
from src.services.negotiation_advice.classification import (
    classify, default_thresholds, _style_for, THRESHOLD_POLICY_SLUG,
)
from src.services.negotiation_advice.grounding import apply_states
from src.services.formulas import ensure_registered, evaluate
from src.services.negotiation_advice.ranking import rank_plays  # noqa: F401 (public re-export)

#: What `classify` itself returns when the deal's signals cannot place it. Used
#: as the explicit fallback when the CONTRACT refuses, so a refused evaluation
#: and an indeterminate deal reach the rest of this module in the same shape --
#: neither of which is a quadrant, and neither of which may be guessed.
_INDETERMINATE = {
    "quadrant": None, "quadrant_reasons": ["classification inputs were not usable"],
    "quadrant_confidence": 0.0, "style": None, "style_reasons": [],
    "indeterminate": True,
}
from src.services.negotiation_advice.signals import (
    gather_signals, market_context_dict, supplier_performance_dict,
)
from src.services.negotiation_advice.store import (
    active_facts, load_advice, save_advice, state_fact, withdraw_fact,
)

log = logging.getLogger(__name__)

_DEFAULT_LIMIT = 6
_MORE_LIMIT = 12


def load_thresholds(conn) -> dict:
    """Governed thresholds, degrading to seeded defaults on any failure.

    PolicyEngine must be given a connection factory. A bare ``PolicyEngine()``
    has no way to reach proc.bp_policy, loads zero rows, and returns None for
    every slug — no exception, so the fallback below would have swallowed it and
    the governed thresholds seeded by the migration would never have been read.
    ``connection_factory=get_conn`` is the same construction the governance tools
    use, and it loads all twelve policies.
    """
    try:
        from src.engines.policy_engine import PolicyEngine
        policy = PolicyEngine(connection_factory=get_conn).get_policy(
            THRESHOLD_POLICY_SLUG
        )
        if policy:
            details = policy.get("policy_details") or policy.get("details")
            if isinstance(details, dict):
                return {**default_thresholds(), **details}
    except Exception:
        log.debug("threshold policy lookup failed; using defaults",
                  exc_info=True)
    return default_thresholds()


_WHY_RULES = (
    "Explain in ONE short sentence why this negotiation play suits this "
    "supplier, using ONLY the facts given. Do not invent figures, terms, "
    "dates or supplier behaviour. Do not suggest a different play."
)


def explain_play(play: dict, signals: dict, *, generate=None) -> str:
    """One grounded sentence on why this play fits, else the fixed rationale.

    The model phrases the reason; it never selects or reorders plays — those stay
    deterministic so the advice is auditable. A failure degrades to the
    deterministic rationale: the advice is useful without prose.
    """
    fallback = str(play.get("rationale") or "")
    evidence = "; ".join(
        f"{e.get('label')}: {e.get('value')}" for e in (play.get("evidence") or [])
    )
    prompt = (
        f"{_WHY_RULES}\n\n"
        f"Supplier: {signals.get('supplier_name') or 'unknown'}\n"
        f"Lever: {play.get('lever')}\n"
        f"Play: {play.get('play')}\n"
        f"Evidence: {evidence or 'none'}\n"
        f"Deal value: {signals.get('deal_value')} "
        f"{signals.get('currency') or ''}\n"
    )
    try:
        if generate is None:
            from src.services.deal_summary import _SUMMARY_MODEL
            from src.services.ollama_client import ollama_generate

            def generate(**kw):
                return ollama_generate(kw.pop("prompt"), **kw)

            # think=False is mandatory: AgentNick is a reasoning model and
            # returns an empty `response` without it.
            text = generate(prompt=prompt, model=_SUMMARY_MODEL,
                            temperature=0.0, num_predict=120, timeout=60,
                            retries=1, think=False)
        else:
            text = generate(prompt=prompt, model=None, temperature=0.0,
                            num_predict=120, timeout=60, retries=1,
                            think=False)
    except Exception:
        log.debug("play explanation failed; using deterministic rationale",
                  exc_info=True)
        return fallback
    if not text or not str(text).strip():
        return fallback
    return str(text).strip()


def build_advice(deal_id: str, *, conn=None, created_by: Optional[str] = None,
                 overrides: Optional[dict] = None,
                 stated: Optional[dict] = None,
                 lever: Optional[str] = None,
                 limit: int = _DEFAULT_LIMIT) -> Optional[dict]:
    if conn is None:
        with get_conn() as own:
            return build_advice(deal_id, conn=own, created_by=created_by,
                                overrides=overrides, stated=stated,
                                lever=lever, limit=limit)

    measured = gather_signals(conn.cursor(), deal_id)
    if measured is None:
        return None

    stated = dict(stated or {})
    overrides = dict(overrides or {})

    # Stated facts steer classification; measured values are left intact.
    effective = dict(measured)
    effective.update(stated)

    ensure_registered()
    verdict = evaluate("negotiation.kraljic_quadrant", {
        "signals": effective, "thresholds": load_thresholds(conn),
    }).or_else(_INDETERMINATE)
    quadrant = overrides.get("quadrant") or verdict["quadrant"]
    quadrant_source = "buyer" if overrides.get("quadrant") else "computed"

    style_reasons = verdict.get("style_reasons") or []
    if overrides.get("quadrant") and not overrides.get("style"):
        # The buyer moved the quadrant; the computed style belonged to the old
        # one. Leaving it would pair e.g. Bottleneck with Competitive — the
        # opposite of what a thin supply market calls for.
        style, style_reasons = _style_for(quadrant, effective)
        style_source = "computed"
    else:
        style = overrides.get("style") or verdict["style"]
        style_source = "buyer" if overrides.get("style") else "computed"

    plays: list = []
    if quadrant and style:
        ranked = evaluate("negotiation.play_rank", {
            "supplier_type": quadrant, "negotiation_style": style,
            "lever_priorities": [lever] if lever else None,
            "policy_guidance": None,
            "supplier_performance": supplier_performance_dict(effective),
            "market_context": market_context_dict(effective),
            "playbook": None, "limit": limit,
        }).or_else({"plays": []})
        plays = apply_states(ranked.get("plays") or [], effective)
        for p in plays:
            p["why"] = explain_play(p, effective)

    saved = save_advice(
        conn, deal_id=deal_id, supplier_id=measured.get("supplier_id"),
        quadrant=quadrant, quadrant_source=quadrant_source,
        quadrant_confidence=verdict.get("quadrant_confidence"),
        style=style, style_source=style_source, signals=measured, plays=plays,
        created_by=created_by,
    )

    return {
        "advice_id": saved.get("advice_id"),
        "deal_id": deal_id,
        "supplier_id": measured.get("supplier_id"),
        "supplier_name": measured.get("supplier_name"),
        "quadrant": quadrant,
        "quadrant_source": quadrant_source,
        "quadrant_reasons": verdict.get("quadrant_reasons") or [],
        "quadrant_confidence": verdict.get("quadrant_confidence"),
        "style": style,
        "style_source": style_source,
        "style_reasons": style_reasons,
        "indeterminate": bool(verdict.get("indeterminate")) and not quadrant,
        "signals": measured,
        "stated_facts": stated,
        "plays": plays,
    }


def apply_turn(deal_id: str, message: dict, *, conn=None,
               created_by: Optional[str] = None) -> Optional[dict]:
    if conn is None:
        with get_conn() as own:
            return apply_turn(deal_id, message, conn=own, created_by=created_by)

    action = str((message or {}).get("action") or "").strip()
    existing = load_advice(conn, deal_id)
    advice_id = (existing or {}).get("advice_id")

    if advice_id is None and action in ("state_fact", "withdraw_fact"):
        # Nothing says the buyer viewed the deal before correcting it. Without a
        # row there is no advice_id to attach the fact to and the turn used to
        # drop it in silence, so build the advice this fact belongs to first.
        seed = build_advice(deal_id, conn=conn, created_by=created_by)
        if seed is None:
            return None
        advice_id = seed.get("advice_id")

    if action == "state_fact" and advice_id:
        state_fact(conn, advice_id=advice_id,
                   fact_key=message.get("fact_key"),
                   fact_value=message.get("fact_value"),
                   stated_by=created_by or "buyer")
    elif action == "withdraw_fact" and advice_id:
        withdraw_fact(conn, advice_id=advice_id,
                      fact_key=message.get("fact_key"))

    stated = active_facts(conn, advice_id) if advice_id else {}
    stated = {k: _coerce(v) for k, v in (stated or {}).items()}

    overrides = {}
    if action == "override":
        for key in ("quadrant", "style"):
            if message.get(key):
                overrides[key] = message[key]

    lever = message.get("lever") if action == "set_lever" else None
    limit = _MORE_LIMIT if action == "more_plays" else _DEFAULT_LIMIT

    out = build_advice(deal_id, conn=conn, created_by=created_by,
                       overrides=overrides, stated=stated, lever=lever,
                       limit=limit)
    if out is None:
        return None

    if action == "compare_style" and message.get("style") and out["quadrant"]:
        ensure_registered()
        other = evaluate("negotiation.play_rank", {
            "supplier_type": out["quadrant"], "negotiation_style": message["style"],
            "lever_priorities": None, "policy_guidance": None,
            "supplier_performance": supplier_performance_dict(out["signals"]),
            "market_context": market_context_dict(out["signals"]),
            "playbook": None, "limit": _DEFAULT_LIMIT,
        }).or_else({"plays": []})
        out["comparison"] = {
            "style": message["style"],
            "plays": apply_states(other.get("plays") or [], out["signals"]),
        }
    return out


def _coerce(value: Any) -> Any:
    """Stated facts arrive as text; numbers must compare as numbers."""
    if value is None or isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text or text.lower() in ("none", "null"):
        return None
    try:
        return float(text) if "." in text else int(text)
    except ValueError:
        return text
