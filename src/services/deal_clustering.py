# src/services/deal_clustering.py  (part 1 — primitives; orchestrator added in Task 8)
"""Batch -> proposed sourcing-event clusters. Pure and write-free: no DB handle, no
I/O, so it is fully testable on fixtures and a bad run can never corrupt assigned deals.
"""
from __future__ import annotations

from itertools import combinations

from src.services.requirement_similarity import rivalry_score
from src.services.version_collapse import collapse_versions  # noqa: F401 (re-exported for callers)

THRESHOLD = 0.70   # complete-linkage bar; a tunable starting value (spec §Validation)


def pairwise_matrix(bids: list[dict], lines: dict, scorer=rivalry_score) -> dict:
    """Correlation + evidence for every unordered bid pair. Two bids from the SAME
    supplier are never rivals (R3) and are excluded before scoring."""
    matrix: dict = {}
    for a, b in combinations(bids, 2):
        sa, sb = a.get("supplier_id"), b.get("supplier_id")
        if sa is not None and sb is not None and sa == sb:
            continue   # R3: same supplier -> versions/duplicates, never rivalry
        res = scorer(a, b, lines.get(a["quote_id"], []), lines.get(b["quote_id"], []))
        matrix[frozenset((a["quote_id"], b["quote_id"]))] = res
    return matrix


def _corr(matrix: dict, qa: str, qb: str) -> float:
    res = matrix.get(frozenset((qa, qb)))
    return res["correlation"] if res else 0.0


def complete_linkage(bids: list[dict], matrix: dict, threshold: float = THRESHOLD) -> list[list[dict]]:
    """Agglomerative clustering under COMPLETE linkage: merge two clusters only when
    EVERY cross-pair clears the threshold. Single linkage was measured and rejected —
    one 0.626 pair chained IT-MSA and Platform into a six-supplier blob."""
    clusters = [[b] for b in bids]
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if all(_corr(matrix, x["quote_id"], y["quote_id"]) >= threshold
                       for x in clusters[i] for y in clusters[j]):
                    clusters[i] = clusters[i] + clusters[j]
                    del clusters[j]
                    changed = True
                    break
            if changed:
                break
    return clusters


def cluster_confidence(cluster: list[dict], matrix: dict) -> float:
    """Confidence = min pairwise correlation across members x100 (spec schema comment).
    A singleton has no pair; callers treat it as a single-bid event, not scored here."""
    if len(cluster) < 2:
        return 100.0
    pairs = [_corr(matrix, a["quote_id"], b["quote_id"])
             for a, b in combinations(cluster, 2)]
    return round(min(pairs) * 100.0, 1)
