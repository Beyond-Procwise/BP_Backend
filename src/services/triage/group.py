"""One cause, one finding (spec §6). Pure.

Order matters: duplicates first (they can absorb quantity and over-billing), then
quantity groups and price groups, then the PO running total, which becomes an effect
of whatever explains it and is its own finding only for the part nothing explains.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from .model import SCORED, Finding, Result, Severity, pct_change
from .score import score_result

_EXPLAINS_OVERAGE = ("duplicate", "quantity", "unit_price", "uniform_uplift")
_GROUPED = frozenset({"duplicate", "quantity", "unit_price", "cumulative_total"})


def _uplift_clusters(rs: list[Result], within: Decimal) -> list[list[Result]]:
    keyed = sorted(((pct_change(r), r) for r in rs if pct_change(r) is not None),
                   key=lambda t: t[0])
    clusters: list[list[Result]] = []
    current: list[Result] = []
    start = None
    for p, r in keyed:
        if current and p - start <= within:
            current.append(r)
        else:
            if current:
                clusters.append(current)
            current, start = [r], p
    if current:
        clusters.append(current)
    clusters.extend([r] for r in rs if pct_change(r) is None)
    return clusters


def group(results: list[Result], cfg) -> list[Finding]:
    live = [r for r in results if r.outcome in SCORED
            and r.severity is not None and r.severity > Severity.S0]
    findings: list[Finding] = []

    dup_by_invoice: dict[str, Finding] = {}
    for r in live:
        if r.rule_id == "duplicate":
            f = Finding(r.deal_id, "duplicate", [r], r.cause_key)
            findings.append(f)
            dup_by_invoice[r.claim_doc] = f

    # Keyed on the PO alone: the quantity invoiced against a PO is one fact, whichever
    # invoice (or credit note) happens to be the latest, so a later document must not
    # re-key the finding a person may already be working on.
    by_po = defaultdict(list)
    for r in live:
        if r.rule_id == "quantity":
            by_po[r.po_id].append(r)
    for po_id, rs in by_po.items():
        dups = sorted({r.claim_doc for r in rs if r.claim_doc in dup_by_invoice})
        if dups:
            dup_by_invoice[dups[0]].effects.extend(rs)
        else:
            findings.append(Finding(rs[0].deal_id, "quantity", rs, f"{po_id}|quantity"))

    by_invoice = defaultdict(list)
    for r in live:
        if r.rule_id == "unit_price":
            by_invoice[r.claim_doc].append(r)
    for inv_id, rs in by_invoice.items():
        for cluster in _uplift_clusters(rs, cfg["uplift_same_pct_within"]):
            if len(cluster) >= cfg["uplift_min_lines"]:
                # The cluster's starting percentage keeps two clusters on one invoice
                # (say +3% and +10%) from sharing a fingerprint.
                start_pct = pct_change(cluster[0])
                key = f"{inv_id}|uplift|{start_pct:.1f}" if start_pct is not None else \
                    f"{inv_id}|uplift|{cluster[0].claim_line}"
                findings.append(Finding(cluster[0].deal_id, "uniform_uplift", cluster, key))
            else:
                findings.extend(Finding(r.deal_id, "unit_price", [r], r.cause_key)
                                for r in cluster)

    for r in live:
        if r.rule_id != "cumulative_total":
            continue
        related = [f for f in findings if f.rule_id in _EXPLAINS_OVERAGE
                   and any(c.po_id == r.po_id for c in f.causes)]
        explained = sum((f.exposure for f in related), Decimal("0"))
        allowance = Decimal(str(r.tolerance.get("allowance", "0")))
        if related and abs(r.exposure) <= explained + allowance:
            max(related, key=lambda f: (f.severity, f.exposure)).effects.append(r)
            continue
        if related:
            r.exposure = abs(r.exposure) - explained
            # The explained part is already money at stake on those findings: the
            # Action Centre sums claim - auth, so this one must show only the remainder.
            if r.claim_amount is not None and r.auth_amount is not None:
                r.claim_amount = r.auth_amount + r.exposure
            r.note = f"{r.note}; {explained} of the overage is explained by findings on its lines"
            score_result(r, cfg)
        findings.append(Finding(r.deal_id, "cumulative_total", [r], r.cause_key))

    findings.extend(Finding(r.deal_id, r.rule_id, [r], r.cause_key)
                    for r in live if r.rule_id not in _GROUPED)
    return findings
