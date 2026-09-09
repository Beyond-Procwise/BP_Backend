"""A parent for the documents that never named one.

The promotion path scores a document against the single purchase order it cites.
A document that cites nothing is scored against nothing: it is held as
``no_parent_reference``, or — where some other path already promoted it — it sits
in _trgt linked to no order at all, so no three-way match runs against it and its
spend hangs off no purchase. Measured on bp_testdb: 1,964 invoices in _trgt carry
no purchase-order reference at all.

A document citing an order we do not hold (194 more) is deliberately NOT in that
population. It has stated which order it belongs to, and the usual reason we
cannot find it is that the order has not been ingested yet — not that the
document meant a different one. Proposing an alternative there would be
overruling a citation, which is the one thing this module must never do.

This module puts those documents in front of the scorer, and puts the scorer's
verdicts through the resolution layer, because one at a time is not enough to
answer the question. Several unreferenced invoices from one supplier compete for
that supplier's orders, and a document can end up proposed for its second choice
because its first is better spent elsewhere. The margin that comes back is what
routing acts on: it says whether the evidence chose an order or merely tolerated
one.

**The order's value is reported, not enforced.** It was enforced first — an order
absorbing only what it authorised, which is the resolution layer's capacity model
and the reason this looked like its first real user. Measured against documents
whose true parent is known, that withheld 108 of 253 true parents: an invoice
that alone bills more than its order, or a set that together outruns it, lost its
candidate and got no proposal at all. That is the wrong trade, and this codebase
has already ruled on it — ``three_way_match`` raises over-billing as a finding and
never holds the document, because a buyer needs to SEE the over-billing.
Suppressing the link is worse than suppressing the promotion: an unlinked invoice
is not over-billing anything, it is invisible.

So the value question is answered ON the proposal (``claim``,
``order_remaining``, ``within_order_value``) and left out of the solve. What that
costs is the ability to prefer an order with room over one without, for a
document that has both — and no document in this corpus has two candidates above
the floor, so that benefit was unmeasurable while the cost was measured. Capacity
remains right where a resource genuinely runs out; a purchase order and its
invoices is not one, because over-billing is a finding rather than an
impossibility.

**Nothing here writes a link.** A proposal is a suggestion carrying its own
margin, for a person to accept or reject. Inferring a parent and stamping it onto
the document would be manufacturing the very reference the document does not
carry, and the margin is exactly the number that says whether the evidence chose
an order or merely tolerated one.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.services.agent_actions import PHASE_CONSOLIDATION, record_action
from src.services.db import get_conn
from src.services.linking_engine import (
    _DOC,
    _PO,
    _PO_NORM_SQL,
    _rows,
    _to_float,
    score_link,
)
from src.services.resolution import (
    CandidateEdge,
    CardinalityRule,
    ResolutionRequest,
    resolve,
)
from src.services.resolution.model import DEGENERACY_FLOOR

log = logging.getLogger(__name__)

# The floor a candidate must clear to be worth showing anyone -- and it is NOT
# the promotion path's review floor, because this is not the promotion path's
# scoring problem.
#
# A document with no reference is missing the profile's heaviest signal by
# definition, so every score here is lower than the same match would earn with
# its citation. Measured on bp_testdb, blanking the reference on 400 invoices
# that DO name their order and re-scoring each against its supplier's whole
# order set:
#
#   true order      n=320   min 22.1   median 56.9   max 60.6
#   wrong order     n=305   min  0.3   median  1.7   max 21.4
#
# The two populations do not overlap: the weakest true pair outscores the
# strongest wrong one. The ranking is also perfect -- the true order came first
# for all 320 -- so what the floor decides is not WHICH order but whether to say
# anything at all. It sits in the empty band between the two, 18.6 above every
# wrong pair and below 99% of true ones (317/320 kept, 0/305 wrong admitted).
#
# REVIEW_MIN (65) would have been above BOTH populations: nothing could ever have
# been proposed, and the whole pass would have returned an empty list that looked
# like a clean bill of health. That is why this number is measured rather than
# borrowed, and why the calibration is pinned by a test.
#
# It is calibrated on this corpus and must be re-derived on another -- most of
# all where converted_amount_usd is populated, which it is not here, so the
# amount signal contributes nothing to either population above.
PROPOSAL_MIN_SCORE = float(os.getenv("PROPOSE_MIN_LINK_SCORE", "40"))

# Every candidate pair is a variable in the model, so a supplier with 40 orders
# and 30 unreferenced invoices would be 1,200 of them. Each document competes for
# its best few orders instead, which is linear in the document count. The best
# candidate is never dropped, so a document whose orders do not compete is
# unaffected by the bound.
MAX_CANDIDATES_PER_DOC = int(os.getenv("PROPOSE_MAX_CANDIDATES", "5"))

# An invoice may bill against one purchase order; an order carries many invoices.
_PROFILE_ID = "unreferenced_doc_po"
_RULE = CardinalityRule(_PROFILE_ID, "N:1", None, 1)

# The band inside which a claim counts as fitting the order. It is the one
# three_way_match already allows a purchase order to be over-billed by, so this
# module never calls a document over-claiming that the match itself would accept.
_CAPACITY_TOL_PCT = 0.005
_CAPACITY_TOL_MIN = 0.01


@dataclass(frozen=True)
class LinkProposal:
    """One suggested parent, and how forced the suggestion was.

    A proposal is never a promotion, and that is structural rather than a rule
    anyone enforces: the missing reference costs the heaviest signal in the
    profile, so the best score measured on a known-correct pair here is 60.6
    against an auto-promotion gate of 80.

    ``routing`` is what a queue acts on: ``suggested`` where the evidence
    separates this order from every alternative, ``contested`` where it does not
    and a person is choosing between near-equals rather than confirming a finding.
    """

    doc_type: str
    doc_pk: str
    po_id: str
    F: float
    margin: float
    margin_normalised: float
    routing: str
    alternatives: tuple[str, ...]
    claim: Optional[float] = None            # what this document would bill
    order_remaining: Optional[float] = None  # what the order has left, same unit
    within_order_value: Optional[bool] = None


# Rejections are the only part of this whole surface worth storing. The proposal is
# derived and recomputed on every pass, so persisting it would only create something
# that can go stale; a person saying "not that order" is a decision, and nothing else
# in the system can re-derive it.
#
# The pair is the unit. "Not this order" is not "this document belongs nowhere" — the
# runner-up is exactly what should be shown next — so a rejection suppresses one
# (document, order) pair and nothing more.
REJECTION_DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.bp_link_rejection (
    rejection_id BIGSERIAL PRIMARY KEY,
    doc_type     TEXT NOT NULL,
    doc_pk       TEXT NOT NULL,
    po_id        TEXT NOT NULL,
    rejected_by  TEXT NOT NULL,
    note         TEXT,
    rejected_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- One row per pair. Two people reaching the same conclusion, or one person
    -- clicking twice, is not two rejections.
    UNIQUE (doc_type, doc_pk, po_id)
);

CREATE INDEX IF NOT EXISTS ix_bp_link_rejection_doc
    ON proc.bp_link_rejection (doc_type, doc_pk);
"""

_REJECT_INSERT = """
INSERT INTO proc.bp_link_rejection (doc_type, doc_pk, po_id, rejected_by, note)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (doc_type, doc_pk, po_id) DO NOTHING
"""


def _ensure_rejection_table(cur) -> None:
    cur.execute(REJECTION_DDL)


def rejected_pairs(cur, doc_type: str = "invoice") -> set:
    """Every (doc_type, doc_pk, po_id) a person has refused.

    Read ONCE for a whole pass. Reading it per document or per supplier is the
    mistake this module already made with orders and line items.
    """
    # Fenced by a savepoint, not merely wrapped in try/except. The table does not
    # exist until somebody rejects something for the first time, and in Postgres a
    # failed statement aborts the WHOLE transaction: a bare except would return an
    # empty set and then every later read in this pass would fail with "current
    # transaction is aborted". The savepoint is what makes the absence survivable.
    try:
        cur.execute("SAVEPOINT link_rejection_read")
    except Exception:  # noqa: BLE001 — no transaction to fence (autocommit); read plainly
        log.debug("could not open a savepoint for the rejection read", exc_info=True)
        return _rejections_or_empty(cur, doc_type)
    try:
        rows = _rows(cur, "select doc_type, doc_pk, po_id from proc.bp_link_rejection "
                          "where doc_type = %s", (doc_type,))
    except Exception:  # noqa: BLE001 — the store not existing yet is not a failure
        log.debug("could not read link rejections", exc_info=True)
        cur.execute("ROLLBACK TO SAVEPOINT link_rejection_read")
        return set()
    cur.execute("RELEASE SAVEPOINT link_rejection_read")
    return {(str(r["doc_type"]), str(r["doc_pk"]), str(r["po_id"])) for r in rows}


def _rejections_or_empty(cur, doc_type: str) -> set:
    try:
        rows = _rows(cur, "select doc_type, doc_pk, po_id from proc.bp_link_rejection "
                          "where doc_type = %s", (doc_type,))
    except Exception:  # noqa: BLE001
        log.debug("could not read link rejections", exc_info=True)
        return set()
    return {(str(r["doc_type"]), str(r["doc_pk"]), str(r["po_id"])) for r in rows}


@dataclass(frozen=True)
class ProposalRun:
    """One whole pass: what it proposes, and what it looked at to get there.

    The counts are not decoration. A pass over bp_testdb proposes nothing at all,
    and an empty list on its own reads as "every invoice has an order" when the
    truth is that 1,964 have none and not one of them resembles an order their
    supplier holds. ``considered`` is what lets a screen tell those two apart.
    """

    proposals: tuple[LinkProposal, ...]
    considered: dict

    def __iter__(self):
        return iter(self.proposals)

    def __len__(self):
        return len(self.proposals)


def _log_odds(f: float) -> float:
    """The engine's F score as log-odds, for the resolver's objective. Monotone
    in F, so the resolver's ranking is the scorer's ranking."""
    p = min(max(f / 100.0, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


# The native amount a document states, per type. converted_amount_usd is the
# common unit where it exists -- on bp_testdb it exists on 4 of 5,041 purchase
# orders and 10 of 12,408 invoices, while these columns and the currency are on
# every row. Read only the converted figure and the value evidence is blank on
# 99.9% of the documents it is meant to describe.
_NATIVE_AMOUNT = {"invoice": "invoice_amount", "quote": "total_amount"}
_USD = "USD"


def _order_capacity(po: dict) -> tuple[Optional[float], Optional[str]]:
    """What the order has left to absorb, and the unit that figure is in.

    A purchase order whose value we do not hold constrains nobody: reading a
    missing total as zero would silently disqualify every document from every
    order we happen to have an incomplete row for.

    One unit per order, decided by the order. A converted figure is preferred
    where the order carries one, because it is the one unit a mixed-currency set
    of documents can all be stated in; otherwise the order's own currency is the
    unit, and only documents stating that currency can be compared against it.
    Nothing here converts anything.
    """
    if "remaining_capacity" in po:
        return _to_float(po.get("remaining_capacity")), (po.get("capacity_unit") or _USD)
    usd = _to_float(po.get("converted_amount_usd"))
    if usd is not None:
        return usd, _USD
    native = _to_float(po.get("total_amount"))
    currency = str(po.get("currency") or "").strip().upper()
    if native is not None and currency:
        return native, currency
    return None, None


def _claim(doc: dict, unit: Optional[str], doc_type: str) -> Optional[float]:
    """What this document would draw from an order measured in ``unit``.

    None where the two cannot be compared without inventing an exchange rate.
    That is not a bound of zero and not a disqualification — it is the absence of
    a bound, and the document competes on evidence alone.
    """
    if unit is None:
        return None
    if unit == _USD:
        return _to_float(doc.get("converted_amount_usd"))
    if str(doc.get("currency") or "").strip().upper() != unit:
        return None
    return _to_float(doc.get(_NATIVE_AMOUNT.get(doc_type, "total_amount")))


def _tolerance(capacity: float) -> float:
    return max(_CAPACITY_TOL_MIN, _CAPACITY_TOL_PCT * abs(capacity))


def propose_links(documents: list[dict], purchase_orders: list[dict],
                  doc_lines: Optional[dict] = None, po_lines: Optional[dict] = None,
                  *, doc_type: str = "invoice", profile: Optional[str] = None,
                  min_score: Optional[float] = None,
                  max_candidates: Optional[int] = None,
                  rejected: Optional[set] = None,
                  scorer: Optional[Callable] = None) -> list[LinkProposal]:
    """Score every document against every order and resolve the whole set at once.

    Pure: no database, no writes. ``documents`` and ``purchase_orders`` are rows
    as the tables hold them, ``doc_lines`` / ``po_lines`` map a primary key to its
    line items, and an order may carry ``remaining_capacity`` where the caller has
    already taken off what is billed against it.
    """
    pk = _DOC[doc_type]["pk"]
    profile = profile or _DOC[doc_type]["profile"]
    floor = PROPOSAL_MIN_SCORE if min_score is None else float(min_score)
    cap_n = MAX_CANDIDATES_PER_DOC if max_candidates is None else int(max_candidates)
    doc_lines, po_lines = doc_lines or {}, po_lines or {}
    rejected = rejected or set()
    # Resolved at call time, not bound as a default, so the whole-corpus path can
    # be driven by a stated evidence table in a test the way propose_links itself is.
    scorer = scorer or score_link

    orders = sorted(purchase_orders, key=lambda p: str(p.get("po_id") or ""))
    scored: dict[str, list[tuple[float, str, Optional[float], Optional[float]]]] = {}

    for doc in sorted(documents, key=lambda d: str(d.get(pk) or "")):
        doc_pk = str(doc.get(pk) or "")
        if not doc_pk:
            continue
        candidates: list[tuple[float, str, Optional[float], Optional[float]]] = []
        for po in orders:
            po_id = str(po.get("po_id") or "")
            if not po_id:
                continue
            if (doc_type, doc_pk, po_id) in rejected:
                # A person has already said no to this pair. It is not a candidate, not
                # an alternative, and not a tie-break — it is simply not on the table.
                continue
            capacity, unit = _order_capacity(po)
            claim = _claim(doc, unit, doc_type)
            f = float(scorer(doc, po, profile, doc_lines.get(doc_pk, []),
                             po_lines.get(po_id, [])).get("F", 0.0) or 0.0)
            if f < floor:
                continue
            candidates.append((f, po_id, claim, capacity))
        # Best first, canonical order breaking ties, then bounded.
        candidates.sort(key=lambda c: (-c[0], c[1]))
        if candidates:
            scored[doc_pk] = candidates[:cap_n]

    edges = []
    for doc_pk, candidates in sorted(scored.items()):
        for f, po_id, _claim_amt, _capacity in candidates:
            edges.append(CandidateEdge(
                source_id=doc_pk, target_id=po_id, log_odds=_log_odds(f),
                confidence=f / 100.0, profile_id=_PROFILE_ID, consumes={},
            ))
    if not edges:
        return []

    result = resolve(ResolutionRequest(
        request_id=f"link_proposals:{doc_type}",
        edges=tuple(edges), capacities=(), rules=(_RULE,),
        profile_registry_version=f"link_proposals/{profile}",
    ))

    proposals = []
    for link in result.links:
        candidates = scored[link.source_id]
        f, _po, claim, capacity = next(c for c in candidates if c[1] == link.target_id)
        fits = None
        if claim is not None and capacity is not None:
            fits = claim <= capacity + _tolerance(capacity)
        proposals.append(LinkProposal(
            doc_type=doc_type, doc_pk=link.source_id, po_id=link.target_id, F=f,
            margin=link.margin, margin_normalised=link.margin_normalised,
            routing=("contested" if link.margin_normalised < DEGENERACY_FLOOR
                     else "suggested"),
            alternatives=tuple(po_id for _f, po_id, _c, _cap in candidates
                               if po_id != link.target_id),
            claim=claim, order_remaining=capacity, within_order_value=fits,
        ))
    proposals.sort(key=lambda p: p.doc_pk)
    return proposals


# ---------------------------------------------------------------------------
# Reading the two populations out of the database
# ---------------------------------------------------------------------------
def unparented_documents(cur, doc_type: str = "invoice") -> list[dict]:
    """Documents carrying no usable purchase-order reference.

    Both tiers, because the two populations are the same problem wearing
    different clothes: a held _stg document nobody could score, and a _trgt
    document some other path promoted without ever linking it. The second is the
    larger of the two and the one nothing looks at today.

    The blank-reference test is repeated in Python rather than left to the SQL
    because that is the rule, and a reference of whitespace -- which is how an
    unread field comes back from extraction -- is not a reference.
    """
    cfg = _DOC[doc_type]
    pk = cfg["pk"]
    out: dict[str, dict] = {}
    for table in (cfg["trgt"], cfg["stg"]):
        for row in _rows(cur, f"select * from {table} "
                              f"where po_id is null or btrim(po_id) = ''"):
            ref = row.get("po_id")
            if ref is not None and str(ref).strip():
                continue
            out.setdefault(str(row.get(pk) or ""), row)
    out.pop("", None)
    return [out[k] for k in sorted(out)]


def candidate_orders(cur, supplier_id: str) -> list[dict]:
    """That supplier's purchase orders, each with what it has left to absorb."""
    return orders_by_supplier(cur, [supplier_id]).get(supplier_id, [])


def orders_by_supplier(cur, supplier_ids: list[str]) -> dict[str, list[dict]]:
    """Every named supplier's orders, in a fixed number of reads.

    Read one supplier at a time this cost 3,042 queries and 64.6s over bp_testdb,
    which is not a thing a screen can wait for. The suppliers do not interact —
    each one's documents only ever compete for that supplier's own orders — but
    the READS do not have to be per-supplier to keep that true.

    An order that has already been billed to its limit has nothing to offer an
    unreferenced invoice, and one billed halfway can only take half. Only invoices
    that actually reference an order are counted against it — an unreferenced one
    has not been billed against anything yet, which is the whole reason it is here.
    """
    ids = sorted({str(s).strip() for s in supplier_ids if str(s or "").strip()})
    if not ids:
        return {}
    orders = _rows(cur, f"select * from {_PO['trgt']} where supplier_id = any(%s)",
                   (ids,))
    if not orders:
        return {}
    invoiced = _invoices_by_order(cur, [str(o.get("po_id") or "") for o in orders])
    out: dict[str, list[dict]] = {}
    for po in orders:
        _apply_remaining(po, invoiced)
        out.setdefault(str(po.get("supplier_id") or "").strip(), []).append(po)
    return out


def _apply_remaining(po: dict, invoiced: dict) -> None:
    """What this order has left, in its own unit, and what could not be counted."""
    total, unit = _order_capacity(po)
    if total is None:
        return
    billed, uncounted = 0.0, 0
    for inv in invoiced.get(_norm(po.get("po_id")), []):
        amount = _claim(inv, unit, "invoice")
        if amount is None:
            # Billed in a currency this order is not stated in. Converting would
            # mean inventing a rate, so it is not subtracted -- and the order says
            # so, because a remaining figure that quietly omits part of the
            # billing overstates what is left.
            uncounted += 1
            continue
        billed += amount
    po["capacity_unit"] = unit
    po["remaining_capacity"] = max(0.0, total - billed)
    po["billing_not_counted"] = uncounted


def _invoices_by_order(cur, po_ids: list[str]) -> dict[str, list[dict]]:
    """Invoices referencing each order, deduped across the two tiers.

    A document staged and promoted is one document, not two, so the same invoice
    must not be billed twice against the order it references. Aggregation happens
    in Python rather than in SQL because how much an invoice bills an order
    depends on the unit that order is stated in, which is a per-order question.
    """
    if not po_ids:
        return {}
    keys = [_norm(p) for p in po_ids]
    cond = _PO_NORM_SQL.format(col="po_id")
    seen: dict[str, dict] = {}
    for table in ("proc.bp_invoice_trgt", "proc.bp_invoice_stg"):
        for row in _rows(cur, f"select * from {table} where {cond} = any(%s)", (keys,)):
            seen.setdefault(str(row.get("invoice_id") or ""), row)
    out: dict[str, list[dict]] = {}
    for row in seen.values():
        out.setdefault(_norm(row.get("po_id")), []).append(row)
    return out


def _norm(po_id) -> str:
    from src.services.linking_engine import _norm_po

    return _norm_po(po_id) or ""


# ---------------------------------------------------------------------------
# The whole pass
# ---------------------------------------------------------------------------
def propose_parent_links(conn: Any = None, doc_type: str = "invoice",
                         min_score: Optional[float] = None) -> ProposalRun:
    """Every unreferenced document's proposed parent, supplier by supplier.

    Read-only. Grouped by supplier because a candidate cannot cross a supplier —
    an invoice is not billed by a company that holds no order — so resolving the
    groups separately is the same answer as resolving them together, on problems
    small enough to stay fast.
    """
    if conn is None:
        with get_conn() as own:
            return _propose(own, doc_type, min_score)
    return _propose(conn, doc_type, min_score)


def _propose(conn, doc_type: str, min_score: Optional[float]) -> list[LinkProposal]:
    cur = conn.cursor()
    docs = unparented_documents(cur, doc_type)

    considered = {"documents": len(docs), "without_supplier": 0,
                  "supplier_holds_no_order": 0, "scored": 0}

    by_supplier: dict[str, list[dict]] = {}
    for doc in docs:
        supplier = (doc.get("supplier_id") or "").strip()
        if not supplier:
            # Nothing to narrow the candidates by. Scoring one document against
            # every order in the corpus is not a proposal, it is a guess.
            considered["without_supplier"] += 1
            continue
        by_supplier.setdefault(supplier, []).append(doc)

    if not by_supplier:
        return ProposalRun((), considered)

    # Everything the whole pass needs, read once rather than once per supplier.
    cfg = _DOC[doc_type]
    pk = cfg["pk"]
    orders = orders_by_supplier(cur, list(by_supplier))
    rejected = rejected_pairs(cur, doc_type)
    doc_lines = _lines_for(cur, cfg["lines_trgt"], cfg["lines_stg"], pk,
                           [str(d.get(pk)) for g in by_supplier.values() for d in g])
    po_lines = _lines_for(cur, "proc.bp_po_line_items_stg",
                          "proc.bp_po_line_items_stg", "po_id",
                          [str(o.get("po_id")) for g in orders.values() for o in g])

    out: list[LinkProposal] = []
    for supplier, group in sorted(by_supplier.items()):
        group_orders = orders.get(supplier)
        if not group_orders:
            considered["supplier_holds_no_order"] += len(group)
            continue
        considered["scored"] += len(group)
        out.extend(_propose_for_supplier(cur, doc_type, group, min_score,
                                         orders=group_orders, doc_lines=doc_lines,
                                         po_lines=po_lines, rejected=rejected))
    return ProposalRun(tuple(out), considered)


def _lines_for(cur, trgt_table: str, stg_table: str, key: str,
               keys: list[str]) -> dict[str, list[dict]]:
    """Line items for a whole group in one read per tier, keyed by parent."""
    out: dict[str, list[dict]] = {}
    if not keys:
        return out
    for table in dict.fromkeys((trgt_table, stg_table)):
        try:
            rows = _rows(cur, f"select * from {table} where {key} = any(%s)", (keys,))
        except Exception:  # noqa: BLE001 — a tier that does not exist is not a failure
            log.debug("could not read line items from %s", table, exc_info=True)
            continue
        for row in rows:
            out.setdefault(str(row.get(key) or ""), []).append(row)
        if out:
            break
    return out


# ---------------------------------------------------------------------------
# What a person can do about a proposal
# ---------------------------------------------------------------------------
def _propose_for_supplier(cur, doc_type: str, group: list[dict],
                          min_score: Optional[float] = None,
                          orders: Optional[list] = None,
                          doc_lines: Optional[dict] = None,
                          po_lines: Optional[dict] = None,
                          rejected: Optional[set] = None) -> list[LinkProposal]:
    """One supplier's unreferenced documents against that supplier's orders.

    The orders and line items are passed in by the whole-corpus pass, which reads
    them once for everybody; asked for a single document they are fetched here.
    """
    cfg = _DOC[doc_type]
    pk = cfg["pk"]
    supplier = (group[0].get("supplier_id") or "").strip() if group else ""
    if not supplier:
        return []
    if orders is None:
        orders = candidate_orders(cur, supplier)
    if not orders:
        return []
    if doc_lines is None:
        doc_lines = _lines_for(cur, cfg["lines_trgt"], cfg["lines_stg"], pk,
                               [str(d.get(pk)) for d in group])
    if po_lines is None:
        po_lines = _lines_for(cur, "proc.bp_po_line_items_stg",
                              "proc.bp_po_line_items_stg", "po_id",
                              [str(o.get("po_id")) for o in orders])
    if rejected is None:
        rejected = rejected_pairs(cur, doc_type)
    return propose_links(group, orders, doc_lines, po_lines,
                         doc_type=doc_type, min_score=min_score, rejected=rejected)


def _proposals_for_document(cur, doc_type: str, doc: dict) -> list[LinkProposal]:
    """This document's proposals, decided in the company of its rivals.

    Re-derived here rather than taken from the caller: a proposal is only worth
    what the evidence says at the moment it is acted on, and a score that
    travelled out to a browser and back is not evidence.
    """
    pk = _DOC[doc_type]["pk"]
    supplier = (doc.get("supplier_id") or "").strip()
    if not supplier:
        return []
    group = [d for d in unparented_documents(cur, doc_type)
             if (d.get("supplier_id") or "").strip() == supplier]
    return [p for p in _propose_for_supplier(cur, doc_type, group)
            if p.doc_pk == str(doc.get(pk) or "")]


def confirm_parent_link(doc_type: str, doc_pk: str, po_id: str,
                        reviewer: Optional[str] = None, note: Optional[str] = None,
                        conn: Any = None) -> dict:
    """A person accepts a proposed parent, and the reference is written.

    Only an order this engine actually scored above the floor for this document
    can be confirmed — the winner or any of the alternatives it was shown
    alongside, because on a contested proposal the person is choosing between
    near-equals and the engine has no business insisting on its own tie-break.
    That check is the lock on this door: without it the endpoint would be a way
    to write any reference onto any document, which is the fabrication the
    proposals exist to avoid.

    A document that already names an order is never re-linked. The reference it
    carries is a fact, and nothing here overrules the document.

    The link is written on the reviewer's authority, and the action row records
    whose it was, what the evidence said, and how forced the choice was — so a
    link a person asserted never becomes indistinguishable from one the document
    itself stated. Note that stamping po_id on _trgt is what the deal trigger
    watches, so a confirmed invoice joins its order's deal from here.
    """
    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                result = _confirm(own, doc_type, doc_pk, po_id, reviewer, note)
                own.commit()
                return result
            except Exception:
                own.rollback()
                raise
    return _confirm(conn, doc_type, doc_pk, po_id, reviewer, note)


def _decidable(cur, doc_type: str, doc_pk: str, po_id: str):
    """The check both decisions make before either writes anything.

    Returns (proposal, None) when this person may answer for this pair, or
    (None, refusal) when they may not. Accepting and refusing a proposal have to
    rest on exactly the same question — is this order one the engine actually put
    in front of a person for this document — or the two doors have different locks.
    """
    cfg = _DOC[doc_type]
    pk = cfg["pk"]

    doc = None
    for table in (cfg["trgt"], cfg["stg"]):
        for row in _rows(cur, f"select * from {table} where {pk} = %s", (doc_pk,)):
            if str(row.get(pk) or "") == str(doc_pk):
                doc = row
                break
        if doc is not None:
            break
    if doc is None:
        return None, {"status": "not_found", "detail": f"{doc_type} {doc_pk} not found"}

    existing = doc.get("po_id")
    if existing is not None and str(existing).strip():
        return None, {"status": "refused",
                      "detail": f"{doc_type} {doc_pk} already references "
                                f"{str(existing).strip()}"}

    for p in _proposals_for_document(cur, doc_type, doc):
        if p.po_id == po_id or po_id in p.alternatives:
            return p, None
    return None, {"status": "refused",
                  "detail": f"{po_id} was not proposed as a parent for {doc_type} {doc_pk}"}


def reject_parent_link(doc_type: str, doc_pk: str, po_id: str,
                       reviewer: Optional[str] = None, note: Optional[str] = None,
                       conn: Any = None) -> dict:
    """A person says this is not the order, and the engine stops asking.

    The other half of the decision, and the half that makes the queue finishable.
    Without it a proposal somebody disagreed with returned on every load and the
    only way to make it stop was to accept it — which is how a queue teaches people
    to accept things.

    It suppresses one (document, order) PAIR. The document may still be proposed a
    different order next pass, which is exactly what should happen: "not this one"
    is not "this belongs nowhere", and the runner-up is the next thing to look at.

    Nothing about the document is written. The only fact recorded is that a named
    person refused this pairing, in ``proc.bp_link_rejection`` — and because that
    is a row rather than an event, undoing one is a delete rather than an
    archaeology exercise.
    """
    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                result = _reject(own, doc_type, doc_pk, po_id, reviewer, note)
                own.commit()
                return result
            except Exception:
                own.rollback()
                raise
    return _reject(conn, doc_type, doc_pk, po_id, reviewer, note)


def _reject(conn, doc_type: str, doc_pk: str, po_id: str, reviewer, note) -> dict:
    if doc_type not in _DOC:
        return {"status": "error", "detail": f"unknown doc_type {doc_type}"}
    cur = conn.cursor()

    chosen, refusal = _decidable(cur, doc_type, doc_pk, po_id)
    if refusal is not None:
        return refusal

    _ensure_rejection_table(cur)
    cur.execute(_REJECT_INSERT,
                (doc_type, str(doc_pk), po_id, reviewer or "human_review", note))

    record_action(
        phase=PHASE_CONSOLIDATION, action_type="parent_link_rejected",
        doc_type=doc_type, doc_pk=str(doc_pk), agent=reviewer or "human_review",
        status="ok", confidence=chosen.F,
        summary=f"human-rejected {doc_type} {doc_pk} -> {po_id} "
                f"(F={chosen.F}, margin={chosen.margin})",
        details={"rejected_by": reviewer or "human_review", "note": note,
                 "po_id": po_id, "F": chosen.F, "margin": chosen.margin,
                 "routing": chosen.routing,
                 "alternatives": list(chosen.alternatives)},
        conn=conn)
    return {"status": "rejected", "doc_type": doc_type, "doc_pk": doc_pk,
            "po_id": po_id, "rejected_by": reviewer or "human_review"}


def _confirm(conn, doc_type: str, doc_pk: str, po_id: str, reviewer, note) -> dict:
    if doc_type not in _DOC:
        return {"status": "error", "detail": f"unknown doc_type {doc_type}"}
    cfg = _DOC[doc_type]
    pk = cfg["pk"]
    cur = conn.cursor()

    chosen, refusal = _decidable(cur, doc_type, doc_pk, po_id)
    if refusal is not None:
        return refusal

    for table in (cfg["trgt"], cfg["stg"]):
        cur.execute(f"update {table} set po_id = %s "
                    f"where {pk} = %s and (po_id is null or btrim(po_id) = '')",
                    (po_id, doc_pk))

    record_action(
        phase=PHASE_CONSOLIDATION, action_type="parent_link_confirmed",
        doc_type=doc_type, doc_pk=str(doc_pk), agent=reviewer or "human_review",
        status="ok", confidence=chosen.F,
        summary=f"human-confirmed {doc_type} {doc_pk} -> {po_id} "
                f"(F={chosen.F}, margin={chosen.margin})",
        details={"confirmed_by": reviewer or "human_review", "note": note,
                 "po_id": po_id, "F": chosen.F, "margin": chosen.margin,
                 "margin_normalised": chosen.margin_normalised,
                 "routing": chosen.routing, "proposed_po_id": chosen.po_id,
                 "alternatives": list(chosen.alternatives)},
        conn=conn)
    return {"status": "linked", "doc_type": doc_type, "doc_pk": doc_pk,
            "po_id": po_id, "F": chosen.F, "margin": chosen.margin,
            "routing": chosen.routing, "confirmed_by": reviewer or "human_review"}
