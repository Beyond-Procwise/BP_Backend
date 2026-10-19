"""Approval requirements, read from the policy engine and nowhere else (ruled 2026-10-19).

``get_approval_chain`` answers "what does policy require for this document?" from two policy shapes:
  * ``approval_threshold`` (exists today, ApprovalThresholdPolicy): one amount, above it escalates to ``required_role``.
  * ``approval_chain`` (does not exist yet; this is the shape it is read in): ordered steps by document type, business unit,
    currency and amount band, in ``rules.chains``.
It states what it could not decide as ``gaps`` instead of guessing: another currency is not converted (a silent re-denomination of
a control is the defect to avoid), two different thresholds are not picked between, a chain that matches nothing is not invented.
The hard-coded ``approval_route`` table is the screen's copy and is deliberately not a second source. An unreadable policy store
raises (UNAVAILABLE); it never reads as "no rule".

``evaluate_approval_status`` stays UNSUPPORTED: saying a document is *satisfied* needs a rule nobody has stated and the role each
approver held, which is not recorded.
"""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import Field

from .common import _P, _prov
from .context import CallerContext
from .envelope import GatewayFault, Raw
from .gateway import Backends

_THRESHOLD_ACTION = "spend.approve"
_CHAIN_SLUG = "approval_chain"
_DOC_ALIASES = {"purchase_order": "po", "purchase order": "po"}


class GetApprovalChainParams(_P):
    doc_type: str = Field(min_length=1, max_length=40)
    amount: Decimal = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    business_unit: str = Field(min_length=1, max_length=80)
    as_of: Optional[dt.date] = None


class EvaluateApprovalParams(_P):
    doc_type: str = Field(min_length=1, max_length=40)
    doc_id: str = Field(min_length=1, max_length=80)


def _engine(backends: Optional[Backends]) -> Any:
    eng = backends.extra.get("policy_engine") if backends else None
    if eng is None:
        from src.services import rbac
        eng = rbac.policy_engine()
    if eng is None:
        raise GatewayFault("UNAVAILABLE", "the policy engine is not available", retryable=True)
    return eng


def _doc(value: str) -> str:
    v = value.strip().lower()
    return _DOC_ALIASES.get(v, v)


def _threshold(engine: Any, p: GetApprovalChainParams, gaps: List[str], used: List[str]) -> Optional[Dict[str, Any]]:
    found = [pol for pol in engine.policies_for_action(_THRESHOLD_ACTION)
             if isinstance(((pol.get("details") or {}).get("rules") or {}).get("default_threshold_gbp"), (int, float, Decimal))]
    if not found:
        gaps.append("no approval-threshold policy applies to spend approvals")
        return None
    values = {Decimal(str(pol["details"]["rules"]["default_threshold_gbp"])) for pol in found}
    if len(values) > 1:
        gaps.append("more than one approval-threshold policy applies with different thresholds, so none was picked")
        used.extend(str(pol.get("policyId")) for pol in found)
        return None
    pol = found[0]
    used.append(str(pol.get("policyId")))
    rules, details = pol["details"]["rules"], pol["details"]
    limit, currency = values.pop(), str(rules.get("currency") or "GBP").upper()
    out = {"policy": pol.get("policyName"), "currency": currency, "threshold": float(limit) if limit % 1 else int(limit),
           "required_role": details.get("required_role"), "outcome": None,
           "on_above": rules.get("on_above"), "on_at_or_below": rules.get("on_at_or_below"),
           "applies_to": details.get("applies_to"),
           "note": "this policy applies to spend approvals and does not distinguish document type or business unit"}
    if p.currency.upper() != currency:
        gaps.append(f"the amount is in {p.currency.upper()} but the threshold is written in {currency}; it is not converted, so no outcome is given")
    else:
        out["outcome"] = rules.get("on_above") if p.amount > limit else rules.get("on_at_or_below")
    return out


def _chain(engine: Any, p: GetApprovalChainParams, gaps: List[str], used: List[str]) -> Optional[List[Dict[str, Any]]]:
    pol = engine.get_policy(_CHAIN_SLUG)
    if not pol:
        gaps.append("no approval-chain policy exists, so who signs and in what order is not defined")
        return None
    used.append(str(pol.get("policyId")))
    chains = ((pol.get("details") or {}).get("rules") or {}).get("chains")
    if not isinstance(chains, list):
        gaps.append("the approval-chain policy is malformed (rules.chains is not a list), so it was not used")
        return None
    doc, unit, cur = _doc(p.doc_type), p.business_unit.strip().lower(), p.currency.upper()
    best, best_score, tie = None, -1, False
    for c in chains:
        if not isinstance(c, dict):
            continue
        cdoc, cunit = _doc(str(c.get("doc_type", "*"))), str(c.get("business_unit", "*")).strip().lower()
        if str(c.get("currency", "")).upper() != cur or cdoc not in (doc, "*") or cunit not in (unit, "*"):
            continue
        score = (cdoc == doc) + (cunit == unit)
        if score > best_score:
            best, best_score, tie = c, score, False
        elif score == best_score:
            tie = True
    if best is None:
        gaps.append(f"the approval-chain policy has no chain for document type {p.doc_type!r}, business unit {p.business_unit!r} in {cur}")
        return None
    if tie:
        gaps.append("more than one approval chain matches equally, so none was picked")
        return None
    steps = []
    for band in best.get("steps") or []:
        lo, hi = Decimal(str(band.get("min_amount", 0))), band.get("max_amount")
        hi = None if hi is None else Decimal(str(hi))
        if (p.amount > lo or (lo == 0 and p.amount == 0)) and (hi is None or p.amount <= hi):
            steps.append({"approvers": list(band.get("approvers") or []), "min_amount": float(lo),
                          "max_amount": None if hi is None else float(hi)})
    if len(steps) != 1:
        gaps.append("the chain's amount bands do not give exactly one match for this amount, so it was not used")
        return None
    return [{"step": i + 1, "approver": a} for i, a in enumerate(steps[0]["approvers"])]


def get_approval_chain(ctx: CallerContext, p: GetApprovalChainParams, backends: Optional[Backends]) -> Raw:
    engine = _engine(backends)                       # a PolicyStoreUnavailable from here becomes UNAVAILABLE in the gateway
    gaps: List[str] = []
    used: List[str] = []
    if p.as_of is not None and p.as_of != dt.date.today():
        gaps.append("policy history is not kept; this is the policy in force now, not on the date asked")
    threshold = _threshold(engine, p, gaps, used)
    chain = _chain(engine, p, gaps, used)
    data = {"doc_type": p.doc_type, "amount": float(p.amount), "currency": p.currency.upper(), "business_unit": p.business_unit,
            "threshold": threshold, "chain": chain, "gaps": gaps}
    return Raw(data=data, provenance=[_prov("bp_policy", used)])


def evaluate_approval_status(ctx: CallerContext, p: EvaluateApprovalParams, backends: Optional[Backends]) -> Raw:
    raise GatewayFault("UNSUPPORTED", "not supported yet; waiting on an approval-chain policy (who must sign, in what order) and a record "
                       "of which role each approver held when they signed; neither exists today. An LLM does not decide this. "
                       "Logged to the function backlog.")
