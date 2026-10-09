"""Run-time classification of a multi-policy enforcement verdict (pure: no DB, model or clock).

Rulings: a block always wins and a standing rule never beats it; notify policies
never take part in conflicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from services.agent_policy.conflict_detect import deciders_of, pair_key, same_source


@dataclass
class LiveConflict:
    kind: Optional[str] = None        # None | "block_record" | "auto" | "human"
    involved: List[Dict[str, Any]] = field(default_factory=list)   # hits in any conflicting pair
    pairs: List[Tuple[str, str]] = field(default_factory=list)      # (key, key), sorted
    required: List[Dict[str, Any]] = field(default_factory=list)   # approve hits still needing approval
    last_level_only: Set[str] = field(default_factory=set)          # keys routed to their last level
    rules: List[Dict[str, Any]] = field(default_factory=list)       # standing rules applied (auto)


def conflicting(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    pa, pb = a["policy"], b["policy"]
    if a["id"] == b["id"] or same_source(pa, pb):
        return False                    # tiered policies from one source are not conflicts
    return a["outcome"] != b["outcome"] or deciders_of(pa) != deciders_of(pb)


def classify(verdict) -> LiveConflict:
    deciding = [h for h in verdict.blocks + verdict.approvals if not h.get("unreadable")]
    pairs, involved = [], []
    for i, a in enumerate(deciding):
        for b in deciding[i + 1:]:
            if conflicting(a, b):
                pairs.append(tuple(sorted((str(a["id"]), str(b["id"])))))
                for h in (a, b):
                    if not any(x is h for x in involved):
                        involved.append(h)
    if not pairs:
        return LiveConflict()
    if any(h["outcome"] == "block" for h in involved):
        # design ruling: Not allowed still blocks; the case is for the record and a policy fix
        return LiveConflict("block_record", involved, pairs)
    rules = {}
    for h in involved:
        for r in h["policy"].get("conflicts") or []:
            if isinstance(r, dict) and r.get("with") and r.get("prevails"):
                rules[pair_key(str(h["id"]), str(r["with"]))] = r
    if all(pair_key(*p) in rules for p in pairs):
        losers = {(set(p) - {rules[pair_key(*p)]["prevails"]}).pop() for p in pairs
                  if rules[pair_key(*p)]["prevails"] in p}
        required = [h for h in verdict.approvals if str(h["id"]) not in losers]
        if required and len(losers) == len({k for p in pairs for k in p}) - 1:
            return LiveConflict("auto", involved, pairs, required, set(),
                                [rules[pair_key(*p)] for p in pairs])
    return LiveConflict("human", involved, pairs, list(verdict.approvals), {str(h["id"]) for h in involved})
