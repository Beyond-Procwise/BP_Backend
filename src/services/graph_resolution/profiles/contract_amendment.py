# src/services/graph_resolution/profiles/contract_amendment.py
"""Which contract does this variation, addendum or CCN amend?

The hierarchy profile weighs a title overlap, which is right for a SOW (it repeats
its master agreement's subject) and wrong for an amendment: "Addendum No. 1" shares
no words with the contract it amends, and scored 36.0 (title read as CONFLICT) even
with a resolving reference, the same supplier and a contained term. An amendment is
identified by what it names, so this profile has no title signal.

Also left out, deliberately:
  * expected_structure -- a variation amends ANY structure, so it is always MISSING
    for these types, and a signal that cannot be observed only lowers coverage;
  * value_rollup and payment_terms -- changing value and terms is what an
    amendment does, so repeating them is not corroboration.

Weights are DECLARED, not measured, exactly like contract_hierarchy.
"""
from __future__ import annotations

from src.services import linking_engine as _le
from ..applicability import score_pair
from . import contract_hierarchy as _ch
from . import contract_signals as _cs

PROFILE = "contract_amendment"

_BY_ID = {s["id"]: s for s in _ch.SIGNALS}
SIGNALS = [_BY_ID["declared_reference"], _BY_ID["supplier"], _BY_ID["term_containment"]]
_PARAMS = dict(_ch._PARAMS)

_le.register_profile(PROFILE, {**_PARAMS, "signals": SIGNALS})


def score(src: dict, tgt: dict) -> dict:
    return score_pair(PROFILE, SIGNALS, _cs.AMENDMENT_OPTIONAL, src, tgt)
