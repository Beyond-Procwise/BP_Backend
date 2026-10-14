# src/services/graph_resolution/profiles/contract_renewal.py
"""Which agreement does this renewal renew?

A renewal is an amendment in every respect but one: its term FOLLOWS the term of
the agreement it renews. contract_amendment reads "starts after the parent ended"
as a CONFLICT, which is right for a variation of a live contract and is the
definition of a renewal. Measured live 2026-10-11: RN-2026-0313 named
SA-2026-0310, with the same supplier, buyer and governing law, and scored 47.2 --
below the proposal floor -- on that one signal.

So this is contract_amendment with the term signal replaced. Everything said there
about title, structure, value and payment terms holds here unchanged, and
contract_links applies the same gate: no resolving reference, no proposal.

Weights are DECLARED, not measured, exactly like contract_hierarchy.
"""
from __future__ import annotations

from src.services import linking_engine as _le
from ..applicability import score_pair
from . import contract_hierarchy as _ch
from . import contract_signals as _cs

PROFILE = "contract_renewal"


def _cmp_term_succession(src, tgt) -> tuple[float, str]:
    """Does the renewal's term continue the parent's?

    No adjacency window: a renewal is signed before, on or after the expiry it
    extends, and the reference -- not the gap -- says which agreement it renews.
    The one thing dates can rule out is a renewal that ended before its parent
    began.
    """
    cs, ce = (_ch._to_date(src.get("contract_start_date")),
              _ch._to_date(src.get("contract_end_date")))
    ps = _ch._to_date(tgt.get("contract_start_date"))
    if cs is None or ps is None:
        return 0.5, "MISSING"
    if ce is not None and ce < ps:
        return 0.0, "CONFLICT"       # ended before the parent began
    if cs >= ps:
        return 1.0, "OK"
    return 0.6, "WEAK"               # starts before the agreement it renews


_le.register_signal("crn_term", lambda s, t, sl, tl: _cmp_term_succession(s, t))

_BY_ID = {s["id"]: s for s in _ch.SIGNALS}
#: Same cluster, tier, weight and cap as term_containment: one comparator differs.
_TERM = {**_BY_ID["term_containment"], "id": "term_succession", "kind": "crn_term"}
SIGNALS = [_BY_ID["declared_reference"], _BY_ID["supplier"], _TERM]
_PARAMS = dict(_ch._PARAMS)

_le.register_profile(PROFILE, {**_PARAMS, "signals": SIGNALS})


def score(src: dict, tgt: dict) -> dict:
    return score_pair(PROFILE, SIGNALS, _cs.AMENDMENT_OPTIONAL, src, tgt)
