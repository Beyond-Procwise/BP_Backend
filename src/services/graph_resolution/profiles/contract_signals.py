# src/services/graph_resolution/profiles/contract_signals.py
"""Corroborating signals for contract parent links.

These are OPTIONAL signals: applicability.score_pair adds one to a pair's profile
only when its comparator can be evaluated on that pair. Two rules shape them.

  * Corroborators add, they do not subtract. A SOW may legitimately name different
    payment terms, signatories or cost centres than its master agreement, so a
    mismatch there is NEUTRAL (0.5, "WEAK": no contribution, full coverage). Only
    currency, governing law, buyer, and a child that EXCEEDS its parent's value
    can CONFLICT -- and at tier 2/3 no conflict caps the score.
  * Necessary is not sufficient. A child's value fitting under the parent's proves
    nothing (a small value fits under any parent), so that case is neutral too.

Money is compared only inside one currency. No FX rate is ever invented.

Weights are DECLARED, not measured: no labelled sample of true parent links
exists (every stored parent_contract_id dangles).
"""
from __future__ import annotations

from typing import Optional

from src.services import linking_engine as _le
from . import contract_hierarchy as _ch


def _has(v) -> bool:
    return _ch._norm_ref(v) not in _ch._PLACEHOLDERS


def _eq(a, b) -> bool:
    return _ch._norm_ref(a) == _ch._norm_ref(b)


def _money(v) -> Optional[float]:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def cmp_buyer(src, tgt) -> tuple[float, str]:
    a, b = src.get("buyer_org_id"), tgt.get("buyer_org_id")
    if not (_has(a) and _has(b)):
        return 0.5, "MISSING"
    return (1.0, "OK") if _eq(a, b) else (0.0, "CONFLICT")


def cmp_currency(src, tgt) -> tuple[float, str]:
    a, b = src.get("currency"), tgt.get("currency")
    if not (_has(a) and _has(b)):
        return 0.5, "MISSING"
    return (1.0, "OK") if _eq(a, b) else (0.0, "CONFLICT")


def cmp_payment_terms(src, tgt) -> tuple[float, str]:
    a, b = src.get("payment_terms"), tgt.get("payment_terms")
    if not (_has(a) and _has(b)):
        return 0.5, "MISSING"
    return (1.0, "OK") if _eq(a, b) else (0.5, "WEAK")


def cmp_governing_law(src, tgt) -> tuple[float, str]:
    """Like with like: a governing law against a governing law, else a
    jurisdiction against a jurisdiction. A law on one side and only a
    jurisdiction on the other are different facts and say nothing.

    Picking each side's first stated field independently compared the order
    form OF-2026-0211's jurisdiction ('United Kingdom') with its framework's
    governing law ('England and Wales') and called it a CONFLICT, though both
    documents state the same jurisdiction (live, bp_testdb, 2026-10-08).
    """
    for field in ("governing_law", "jurisdiction"):
        a, b = src.get(field), tgt.get(field)
        if _has(a) and _has(b):
            return (1.0, "OK") if _eq(a, b) else (0.0, "CONFLICT")
    return 0.5, "MISSING"


def cmp_value_rollup(src, tgt) -> tuple[float, str]:
    c, p = _money(src.get("total_contract_value")), _money(tgt.get("total_contract_value"))
    cs, ct = src.get("currency"), tgt.get("currency")
    if c is None or p is None or not (_has(cs) and _has(ct)) or not _eq(cs, ct):
        return 0.5, "MISSING"
    return (0.5, "WEAK") if c <= p else (0.0, "CONFLICT")


_SIGNATORY_FIELDS = ("contract_signatory_name", "buyer_signatory_name")


def cmp_signatory(src, tgt) -> tuple[float, str]:
    """Each party's signatory against the SAME party's on the other document.

    The supplier's signatory on one and the buyer's on the other are different
    roles, so a name matching across them is not corroboration. A match on either
    party is OK; parties that are comparable but all differ are neutral
    (signatories legitimately change); no comparable party is MISSING.
    """
    compared = matched = False
    for field in _SIGNATORY_FIELDS:
        a, b = src.get(field), tgt.get(field)
        if _has(a) and _has(b):
            compared = True
            matched = matched or _eq(a, b)
    if not compared:
        return 0.5, "MISSING"
    return (1.0, "OK") if matched else (0.5, "WEAK")


_COST_FIELDS = ("cost_centre_id", "business_unit_id", "spend_category")


def cmp_cost_centre(src, tgt) -> tuple[float, str]:
    pairs = [(src.get(f), tgt.get(f)) for f in _COST_FIELDS
             if _has(src.get(f)) and _has(tgt.get(f))]
    if not pairs:
        return 0.5, "MISSING"
    return (1.0, "OK") if any(_eq(a, b) for a, b in pairs) else (0.5, "WEAK")


for _kind, _fn in (("csh_buyer", cmp_buyer), ("csh_value", cmp_value_rollup),
                   ("csh_currency", cmp_currency), ("csh_payment", cmp_payment_terms),
                   ("csh_law", cmp_governing_law), ("csh_signatory", cmp_signatory),
                   ("csh_cost", cmp_cost_centre)):
    _le.register_signal(_kind, (lambda f: lambda s, t, sl, tl: f(s, t))(_fn))


def _spec(id_, cluster, tier, weight, cap, kind, reads):
    return {"id": id_, "cluster": cluster, "tier": tier, "weight": weight,
            "appl": 1.0, "cap": cap, "kind": kind, "reads": reads}


BUYER = _spec("buyer", "identity", 2, 3, 0.70, "csh_buyer", ["buyer_org_id"])
VALUE_ROLLUP = _spec("value_rollup", "commercial", 3, 2, 0.90, "csh_value",
                     ["total_contract_value", "currency"])
CURRENCY = _spec("currency", "terms", 3, 2, 0.90, "csh_currency", ["currency"])
PAYMENT_TERMS = _spec("payment_terms", "terms", 3, 2, 0.90, "csh_payment", ["payment_terms"])
GOVERNING_LAW = _spec("governing_law", "terms", 3, 2, 0.90, "csh_law",
                      ["governing_law", "jurisdiction"])
SIGNATORY = _spec("signatory", "people", 3, 2, 0.90, "csh_signatory", list(_SIGNATORY_FIELDS))
COST_CENTRE = _spec("cost_centre", "category", 3, 2, 0.90, "csh_cost", list(_COST_FIELDS))

HIERARCHY_OPTIONAL = [BUYER, VALUE_ROLLUP, CURRENCY, PAYMENT_TERMS, GOVERNING_LAW,
                      SIGNATORY, COST_CENTRE]
#: An amendment changes terms and value; it is not corroborated by repeating them.
AMENDMENT_OPTIONAL = [BUYER, CURRENCY, GOVERNING_LAW, SIGNATORY]
ATTACHMENT_OPTIONAL = [BUYER, CURRENCY, PAYMENT_TERMS, GOVERNING_LAW]
