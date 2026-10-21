"""Procurement policy check (SC-007): the maths and the rules, with no model in the loop.

``evaluate`` is a pure function. Facts come in as a dict; the governed policy comes in as data
the caller already fetched; a verdict comes out. The approval limit, its currency and the
role that approves are read from the policy and nowhere else, so a model cannot misremember
them. A rule with no governed policy is reported as not assessed; it is never reported as
passed.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

COMPLIANT = "COMPLIANT"
CONDITIONS = "COMPLIANT_WITH_CONDITIONS"
NON_COMPLIANT = "NON_COMPLIANT"
INSUFFICIENT = "INSUFFICIENT_INFO"

_WORDS = {COMPLIANT: "compliant", CONDITIONS: "compliant with conditions",
          NON_COMPLIANT: "not compliant", INSUFFICIENT: "insufficient information"}


def _plain(name: str) -> str:
    return name.replace("_", " ")


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def _money(x: float, currency: str) -> str:
    text = f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"
    return f"{text} {currency}"


def _fingerprint(policy: Dict[str, Any]) -> str:
    blob = json.dumps(policy.get("details"), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _threshold(rules: Dict[str, Any]) -> Optional[float]:
    for key, value in rules.items():
        if key.startswith("default_threshold") and _num(value) is not None:
            return float(value)
    return None


def _insufficient(needed: List[str], why: str, evaluated_at: str, policy: Optional[dict] = None) -> Dict[str, Any]:
    out = {"verdict": INSUFFICIENT, "verdict_text": _WORDS[INSUFFICIENT], "evaluated_at": evaluated_at,
           "needed": needed, "summary": why, "rules_evaluated": [], "violations": [],
           "required_approvals": [], "required_actions": [f"Provide: {', '.join(needed)}"] if needed else [],
           "not_assessed": [], "needs_review": [], "assumptions": [], "waivers_applied": [],
           "split_signal": {"performed": False, "fired": False, "reason_not_performed": why},
           "policy": policy, "facts_used": {}}
    out["reply_text"] = f"Insufficient information. {why}"
    return out


def evaluate(request: Dict[str, Any], governance: Dict[str, Any], config: Dict[str, Any],
             *, now: Optional[str] = None) -> Dict[str, Any]:
    evaluated_at = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    slug = (config.get("governed_policies") or {}).get("approval_limit", "approval_threshold_policy")
    policy = governance.get("approval_policy")
    if not governance.get("available") or not policy:
        return _insufficient([], "The governed approval policy could not be read, so no verdict is given.",
                             evaluated_at)
    rules = ((policy.get("details") or {}).get("rules")) or {}
    limit = _threshold(rules)
    policy_currency = str(rules.get("currency") or "")
    if limit is None or not policy_currency:
        return _insufficient([], "The approval policy has no usable limit, so no verdict is given.", evaluated_at)
    role = (policy.get("details") or {}).get("required_role") or "an approver"
    meta = {"slug": slug, "name": policy.get("policy_name"), "fingerprint": _fingerprint(policy),
            "threshold": limit, "currency": policy_currency, "required_role": role}
    rule_name = _plain(slug)

    needed: List[str] = []
    total = _num(request.get("total_commitment"))
    if total is None or total < 0:
        needed.append("total_commitment")
    currency = str(request.get("currency") or "").strip().upper()
    if not currency:
        needed.append("currency")
    fx = 1.0
    assumptions: List[str] = []
    if currency and currency != policy_currency:
        rate = _num(request.get("fx_rate_to_policy_currency"))
        if rate is None or rate <= 0:
            needed.append("fx_rate_to_policy_currency")
        else:
            fx = rate
            assumptions.append(f"The exchange rate {rate} {currency} to {policy_currency} was supplied by the caller, not looked up.")
    if needed:
        return _insufficient(needed, "A figure needed for the approval limit is missing: " + ", ".join(needed) + ".",
                             evaluated_at, meta)

    amount = round(total * fx, 2)
    amount = int(amount) if float(amount).is_integer() else amount
    violations: List[Dict[str, str]] = []
    approvals: List[Dict[str, str]] = []
    actions: List[str] = []
    evaluated = []

    above = amount > limit
    evaluated.append({"rule": rule_name, "outcome": "fail" if above else "pass",
                      "evidence": f"{_money(amount, policy_currency)} against a limit of {_money(limit, policy_currency)}"})
    if above:
        approvals.append({"role": role, "reason_rule": rule_name})

    # Split orders: planned orders count as well as placed ones.
    t = config.get("thresholds") or {}
    pct = float(t.get("split_near_threshold_pct", 10)); min_orders = int(t.get("split_min_orders", 3))
    orders = [o for o in (request.get("orders") or []) if isinstance(o, dict) and _num(o.get("amount")) is not None]
    amounts = [float(o["amount"]) * fx for o in orders]
    placed = any(o.get("source") == "history" for o in orders)
    performed = bool(request.get("history_checked")) or placed
    sig = {"performed": performed, "fired": False, "reason_not_performed": ""}
    if not performed:
        sig["reason_not_performed"] = "Order history was not checked, so earlier orders could not be included."
    if len(amounts) >= 2:
        each_under = all(a <= limit for a in amounts)
        band = [a for a in amounts if limit * (1 - pct / 100) <= a <= limit]
        if (each_under and sum(amounts) > limit) or len(band) >= min_orders:
            sig["fired"] = True
            sig["combined"] = round(sum(amounts), 2)
            severity = "blocking" if placed else "warning"
            violations.append({
                "rule": "split orders", "severity": severity,
                "message": (f"{len(amounts)} orders totalling {_money(sum(amounts), policy_currency)} sit at or under the "
                            f"limit of {_money(limit, policy_currency)} one by one."),
                "remediation": "Submit them as one request and have the combined value approved."})
            actions.append("Have the combined value approved as one request.")
    evaluated.append({"rule": "split orders", "outcome": "fail" if sig["fired"] else ("pass" if performed else "not_performed"),
                      "evidence": f"{len(amounts)} orders considered"})
    if orders and not any("date" in o for o in orders):
        assumptions.append("No order dates were given, so every listed order was treated as inside the window.")

    req, appr = request.get("requester"), request.get("approver")
    if req and appr and req == appr:
        violations.append({"rule": "separation of duties", "severity": "blocking",
                           "message": "The requester is also the approver.",
                           "remediation": "Route the request to a different approver."})
    if req or appr:
        evaluated.append({"rule": "separation of duties",
                          "outcome": "fail" if (req and appr and req == appr) else ("pass" if req and appr else "not_performed"),
                          "evidence": "requester and approver compared"})

    slugs = [str(s).lower() for s in governance.get("slugs") or []]
    keywords = config.get("rule_policy_keywords") or {}
    not_assessed: List[str] = []; needs_review: List[str] = []
    for rule in config.get("ungoverned_rules") or []:
        name = _plain(rule)
        has_policy = any(k in s for k in keywords.get(rule, [rule]) for s in slugs)
        if has_policy:
            needs_review.append(name)
            evaluated.append({"rule": name, "outcome": "governed_but_not_evaluated",
                              "evidence": "A governed policy exists but this checker does not evaluate it."})
            actions.append(f"Have {name} reviewed by hand against its policy.")
        else:
            not_assessed.append(name)
            evaluated.append({"rule": name, "outcome": "not_governed", "evidence": "No governed policy covers this rule."})

    waivers: List[str] = []
    if request.get("waiver_id"):
        assumptions.append(f"The waiver {request['waiver_id']} was not applied: there is no waiver policy to verify it against.")

    if any(v["severity"] == "blocking" for v in violations):
        verdict = NON_COMPLIANT
    elif approvals or violations or needs_review:
        verdict = CONDITIONS
    else:
        verdict = COMPLIANT

    lines = [f"{_WORDS[verdict].capitalize()}.",
             f"Total commitment {_money(amount, policy_currency)} against the {rule_name} limit of "
             f"{_money(limit, policy_currency)}."]
    if above:
        lines.append(f"It is above the limit, so approval is required from the {role} role.")
    else:
        lines.append("It is within the limit, so this policy needs no further approval.")
    for v in violations:
        lines.append(f"{v['severity'].capitalize()}: {v['message']} {v['remediation']}")
    if not_assessed:
        lines.append("Not assessed (no governed policy exists): " + ", ".join(not_assessed) + ".")
    if needs_review:
        lines.append("Needs a manual review (a policy exists but is not evaluated here): " + ", ".join(needs_review) + ".")
    if not sig["performed"]:
        lines.append("Split-order check: " + sig["reason_not_performed"])
    lines.extend(assumptions)
    lines.append(f"Policy: {rule_name}, fingerprint {meta['fingerprint']}.")
    summary = " ".join(lines[:3]) + (" " + lines[3] if len(lines) > 4 and violations else "")
    return {"verdict": verdict, "verdict_text": _WORDS[verdict], "evaluated_at": evaluated_at, "policy": meta,
            "facts_used": {"total_commitment": total, "currency": currency, "amount_in_policy_currency": amount},
            "rules_evaluated": evaluated, "violations": violations, "required_approvals": approvals,
            "required_actions": actions, "split_signal": sig, "waivers_applied": waivers,
            "not_assessed": not_assessed, "needs_review": needs_review, "assumptions": assumptions,
            "needed": [], "summary": summary + (" Rules not assessed are listed." if not_assessed else ""),
            "reply_text": "\n".join(lines)}
