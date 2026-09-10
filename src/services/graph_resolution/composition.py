"""Route correlated signals into one cluster.

The engine already discounts correlated evidence: `_dampen` is applied per
cluster. So the honest way to stop two signals double-counting one observation
is not new arithmetic -- it is putting them in the same cluster and letting the
existing dampening do its job.
"""
from __future__ import annotations

from typing import Dict, List

from .observations import intersects


def remap_clusters(signal_specs: List[dict],
                   observations_by_signal: Dict[str, frozenset]) -> Dict[str, str]:
    """signal_id -> cluster, merging any signals that share an observation.

    Union-find over "shares at least one observation". A merged group takes the
    declared cluster of its lowest-sorted member, so the result is deterministic
    regardless of the order signals were declared in.
    """
    ids = [s["id"] for s in signal_specs]
    declared = {s["id"]: s["cluster"] for s in signal_specs}
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            oa = observations_by_signal.get(a, frozenset())
            ob = observations_by_signal.get(b, frozenset())
            if oa and ob and intersects(oa, ob):
                union(a, b)

    return {i: declared[find(i)] for i in ids}
