# src/services/graph_resolution/profiles/contract_attachment.py
"""Which agreement does this schedule or SLA attach to?

A schedule or SLA has no force on its own: it is cited by an agreement. Its title
usually shares the subject of that agreement (an SLA for a named service), so the
title signal is kept here, unlike for an amendment. expected_structure is left
out: a schedule sits under SEVERAL kinds of agreement, the vocabulary's
default_parent_type holds only one value, and a signal that is always MISSING only
lowers coverage. Candidate parent types are chosen in contract_links.

Weights are DECLARED, not measured.
"""
from __future__ import annotations

from src.services import linking_engine as _le
from ..applicability import score_pair
from . import contract_hierarchy as _ch
from . import contract_signals as _cs

PROFILE = "contract_attachment"

_BY_ID = {s["id"]: s for s in _ch.SIGNALS}
SIGNALS = [_BY_ID["declared_reference"], _BY_ID["supplier"],
           _BY_ID["term_containment"], _BY_ID["title_overlap"]]
_PARAMS = dict(_ch._PARAMS)

_le.register_profile(PROFILE, {**_PARAMS, "signals": SIGNALS})


def score(src: dict, tgt: dict) -> dict:
    return score_pair(PROFILE, SIGNALS, _cs.ATTACHMENT_OPTIONAL, src, tgt)
