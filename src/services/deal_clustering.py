# src/services/deal_clustering.py  (part 1 — primitives; orchestrator added in Task 8)
"""Batch -> proposed sourcing-event clusters. Pure and write-free: no DB handle, no
I/O, so it is fully testable on fixtures and a bad run can never corrupt assigned deals.
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Optional

from src.services.requirement_similarity import rivalry_score
from src.services.version_collapse import base_reference, collapse_versions  # noqa: F401 (collapse_versions re-exported for callers)
from src.services.linking_engine import score_link

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


def awarded_po(bid: dict, pos: list[dict], po_lines: dict, bid_lines: list,
               min_score: float = 80.0, scorer=score_link) -> Optional[str]:
    """The PO this bid won, by CONTINUITY scoring (quote_po: same supplier AND price —
    exact unit-price match is correct for an award). NOT supplier-name string matching,
    which loses SUP-GomezGoodAndCross vs 'Gomez, Good and Cross Trading Ltd' and any null
    supplier. Returns the best PO's id at/above min_score, else None."""
    best_id, best_f = None, 0.0
    for po in pos:
        link = scorer(bid, po, "quote_po", bid_lines, po_lines.get(po["po_id"], []))
        if link["F"] >= min_score and link["F"] > best_f:
            best_id, best_f = po["po_id"], link["F"]
    return best_id


def award_veto(bid_a: dict, bid_b: dict, awards: dict) -> bool:
    """True when two correlated bids each anchor a DISTINCT PO — repeat buying, not
    rivalry. A competition has exactly one award; separate POs+invoices per bid is a
    structural fact that vetoes rivalry (pairwise, over these specific quotes)."""
    pa, pb = awards.get(bid_a["quote_id"]), awards.get(bid_b["quote_id"])
    return pa is not None and pb is not None and pa != pb


# src/services/deal_clustering.py  (part 2 — cluster_batch orchestrator, Task 8)
#
# Ties version-collapse, rivalry clustering, award detection, PO/invoice
# attachment, evidence, HITL review flags and orphans together. Pure and
# write-free: reads its arguments, returns a dict, touches no DB/IO.
_REVIEW_BAND = 80.0     # linking_engine._BAND_WARN
_NEAR = 0.05            # "within 0.05 of the threshold" review trigger


def _norm_ref(v) -> Optional[str]:
    """Normalize a reference for identifier matching: lowercase, strip
    non-alphanumerics (mirrors linking_engine._norm_id)."""
    if v is None:
        return None
    s = re.sub(r"[^a-z0-9]", "", str(v).lower())
    return s or None


_RFQ_TOKEN_RE = re.compile(r"\b(?:[A-Z0-9]+-)*RFQ-[A-Z0-9]+(?:-[A-Z0-9]+)*\b")


def _canon_rfq(ref: str) -> str:
    """Canonical RFQ id, anchored on the RFQ stem so a full reference
    (PROC-2025-RFQ-FRT-021) and a layout-detached bare token (RFQ-FRT-021)
    identify the same competition."""
    s = re.sub(r"[^a-z0-9]", "", str(ref).lower())
    idx = s.find("rfq")
    return s[idx:] if idx >= 0 else s


def extract_rfq_reference(text: str) -> Optional[str]:
    """The RFQ reference a document cites, from its parsed text. Deterministic
    token scan (L1 regex, per the extraction direction): any hyphenated token
    containing the RFQ stem. Multiple DISTINCT references -> None (ambiguous;
    never guess). Absent -> None."""
    if not text:
        return None
    tokens = _RFQ_TOKEN_RE.findall(text)
    if not tokens:
        return None
    canons = {_canon_rfq(t) for t in tokens}
    if len(canons) != 1:
        return None
    return max(tokens, key=len)  # prefer the fullest form seen


def apply_rfq_linkage(bids: list[dict], matrix: dict) -> None:
    """Tier-1 linked identifier: two bids citing the SAME RFQ are rivals in the
    same sourcing event by declaration of the documents themselves — decisive
    where present (spec §Data flow: "linked identifiers ... decisive where
    present"). Their pair correlation is floored at 0.95 so complete linkage
    cannot reject them on fuzzy-text grounds; the shared reference is recorded
    on the pair evidence. Different or absent references change nothing —
    absence of the identifier is not evidence against rivalry. Runs BEFORE the
    award-exclusivity veto, which still outranks it."""
    by_id = {b["quote_id"]: b for b in bids}
    for key, res in matrix.items():
        qa, qb = tuple(key)
        ra = by_id.get(qa, {}).get("rfq_reference")
        rb = by_id.get(qb, {}).get("rfq_reference")
        if ra and rb and _canon_rfq(ra) == _canon_rfq(rb):
            res["correlation"] = max(res["correlation"], 0.95)
            res["rfq_shared"] = max((ra, rb), key=len)


def _explicit_award(bid: dict, pos: list[dict], po_lines: dict) -> Optional[str]:
    """The PO that cites this bid as the quote it was raised against — a tier-1
    LINKED identifier, decisive where present and checked before any score-based
    inference (spec: declared linkage outranks inference). Returns the po_id.

    Two carriers, HEADER FIRST. A PO states its award once in the header
    ("Reference: Against BAFO quote SDP-Q-44120") — that is the grain the
    documents actually use, and it is what ``quote_reference`` holds. The
    per-line ``quote_number`` is the legacy carrier, kept for data that has it
    (it is 0/316 filled on the current corpus, which is why award detection
    never fired before this column existed).

    Both sides are version-collapsed: PO-2024-0145 cites "ORB-Q-6612 (V3)" while
    the collapsed bid is "ORB-Q-6612", and _norm_ref alone would compare
    orbq6612v3 against orbq6612 and miss.
    """
    target = _norm_ref(base_reference(bid.get("base_reference") or ""))
    if target is None:
        return None
    for po in pos:
        if _norm_ref(base_reference(po.get("quote_reference") or "")) == target:
            return po["po_id"]
    for po in pos:
        for line in po_lines.get(po["po_id"], []) or []:
            if _norm_ref(base_reference(line.get("quote_number") or "")) == target:
                return po["po_id"]
    return None


def _fmt_name(cluster: list[dict], quote_lines: dict) -> str:
    """Human-readable proposed name from the shared requirement description."""
    first = cluster[0]["quote_id"]
    lines = quote_lines.get(first) or [{}]
    desc = (lines[0].get("item_description") or "Sourcing event")[:60]
    return f"{desc} — {len(cluster)} bidders"


def cluster_batch(*, quotes, quote_lines, purchase_orders, po_lines, invoices, declared=None) -> dict:
    """Batch -> proposed sourcing events. Pure: reads its arguments, writes nothing."""
    declared = declared or []
    declared_pks = {pk for grp in declared for (_dt, pk) in grp}

    bids_all = collapse_versions(quotes)
    # Everything a human already fixed is set aside; inference operates only on the rest.
    bids = [b for b in bids_all if b["quote_id"] not in declared_pks]

    matrix = pairwise_matrix(bids, quote_lines)

    # Tier-1 linked identifier: a shared RFQ reference is decisive rivalry
    # linkage; applied before the award veto so structure still outranks it.
    apply_rfq_linkage(bids, matrix)

    # Award map: an explicit LINKED identifier (PO line's quote_number == this bid's
    # base_reference) is decisive and checked first; only fall back to continuity
    # scoring when no PO carries the reference (spec §declared linkage outranks
    # inference). Batch quotes carry no po_id, so continuity alone tops out ~F=35 —
    # min_score is lowered to 60 as a fallback net, not the primary signal.
    awards = {}
    for b in bids:
        bid_lines = quote_lines.get(b["quote_id"], [])
        awards[b["quote_id"]] = (_explicit_award(b, purchase_orders, po_lines)
                                  or awarded_po(b, purchase_orders, po_lines, bid_lines, min_score=60.0))

    # Apply the veto by zeroing correlation on any vetoed pair, so complete linkage cannot
    # merge them (correlation proposes; award structure rules out — spec §Award exclusivity).
    for key, res in matrix.items():
        qa, qb = tuple(key)
        if award_veto({"quote_id": qa}, {"quote_id": qb}, awards):
            res["correlation"] = 0.0

    clusters = complete_linkage(bids, matrix)

    proposals, ungrouped = [], []
    members_with_lines = members_total = 0

    # Declared groups become fixed proposals (never re-clustered).
    for grp in declared:
        members = [{"doc_type": dt, "doc_pk": pk, "base_reference": None,
                    "role": "competing_quote" if dt == "quote" else dt,
                    "match_score": None, "match_evidence": None} for (dt, pk) in grp]
        proposals.append({"proposed_name": "Declared grouping", "confidence": 100.0,
                          "declared": True, "review_required": False, "review_reasons": [],
                          "flags": [], "members": members})

    for cluster in clusters:
        if len(cluster) < 2:
            b = cluster[0]
            # R4: a bid alone -> route to a human with its best rejected match.
            best = max(((q, matrix[frozenset((b["quote_id"], q))]["correlation"])
                        for q in (x["quote_id"] for x in bids if x is not b)
                        if frozenset((b["quote_id"], q)) in matrix),
                       key=lambda t: t[1], default=None)
            reason = ("correlated but award-vetoed"
                      if best and awards.get(b["quote_id"]) and awards.get(best[0])
                      and awards[b["quote_id"]] != awards[best[0]]
                      else "no correlated bid above threshold")
            ungrouped.append({"doc_type": "quote", "doc_pk": b["quote_id"], "reason": reason,
                              "best_match": ({"quote_id": best[0], "correlation": best[1]}
                                             if best else None)})
            continue

        conf = cluster_confidence(cluster, matrix)
        anchor = min(cluster, key=lambda b: b["quote_id"])  # deterministic anchor
        members, review_reasons, flags = [], [], []
        for b in cluster:
            role = "anchor_quote" if b is anchor else "competing_quote"
            ev = (matrix.get(frozenset((anchor["quote_id"], b["quote_id"])))
                  if b is not anchor else None)
            members.append({"doc_type": "quote", "doc_pk": b["quote_id"],
                            "base_reference": b["base_reference"], "role": role,
                            "match_score": (round(ev["correlation"] * 100, 2) if ev else None),
                            "match_evidence": ev})
            members_total += 1
            if quote_lines.get(b["quote_id"]):
                members_with_lines += 1
            else:
                flags.append(f"missing line items on {b['quote_id']}")
            if b.get("supplier_id") is None:
                # Flagged so the reviewer sees the supplier is unidentified — but this
                # alone does not demand review (a null-supplier bid, e.g. Swift on
                # Freight, can still sit in a high-confidence cluster).
                flags.append(f"unresolved supplier on {b['quote_id']}")

        # Attach the awarded PO (explicit quote_number, else continuity) + its invoices
        # (deterministic po_id match).
        po_id = next((awards[b["quote_id"]] for b in cluster if awards.get(b["quote_id"])), None)
        if po_id:
            members.append({"doc_type": "po", "doc_pk": po_id, "base_reference": None,
                            "role": "po", "match_score": None, "match_evidence": None})
            # Dedupe: re-extraction leaves multiple raw rows per document, so the
            # batch fetch can supply the same invoice twice; each attaches once.
            seen_invoices: set = set()
            for inv in invoices:
                if inv.get("po_id") == po_id and inv["invoice_id"] not in seen_invoices:
                    seen_invoices.add(inv["invoice_id"])
                    members.append({"doc_type": "invoice", "doc_pk": inv["invoice_id"],
                                    "base_reference": None, "role": "invoice",
                                    "match_score": None, "match_evidence": None})

        # Review triggers (spec §Human-in-the-loop). ONLY confidence-below-band or a
        # cross-pair sitting near the clustering threshold demand human review; a
        # null/unresolved supplier or a missing-line member is surfaced via `flags`
        # instead (spec: "flagged so the user sees the supplier is unidentified"),
        # so a solid cluster with one unresolved-supplier bid (Freight/Swift) is not
        # forced into the review queue.
        if conf < _REVIEW_BAND:
            review_reasons.append(f"confidence {conf} below {int(_REVIEW_BAND)}")
        for a, b in combinations(cluster, 2):
            c = _corr(matrix, a["quote_id"], b["quote_id"])
            if abs(c - THRESHOLD) <= _NEAR:
                review_reasons.append(f"pair {a['quote_id']}~{b['quote_id']} near threshold ({c})")

        proposals.append({"proposed_name": _fmt_name(cluster, quote_lines),
                          "confidence": conf, "declared": False,
                          "review_required": bool(review_reasons), "review_reasons": review_reasons,
                          "flags": flags, "members": members})

    # Orphan POs: awarded, no anchoring bid in the batch (Caldwell). Never absorbed.
    grouped_pos = {m["doc_pk"] for p in proposals for m in p["members"] if m["doc_type"] == "po"}
    for po in purchase_orders:
        if po["po_id"] not in grouped_pos and po["po_id"] not in awards.values():
            ungrouped.append({"doc_type": "po", "doc_pk": po["po_id"],
                              "reason": "awarded PO with no anchoring bid in batch",
                              "best_match": None})

    return {"proposals": proposals, "ungrouped": ungrouped,
            "members_with_lines": members_with_lines, "members_total": members_total}
