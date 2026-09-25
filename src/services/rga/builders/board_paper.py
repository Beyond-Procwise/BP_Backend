"""The board paper: one deal, laid out as the SpendIQ Pipeline's Approve-stage paper (2026-09-25).

User ruling: "use the style already in the pipeline under approvals". That paper
(engine.js dealBoardPaperHTML) is per deal: a masthead, an executive overview with four tiles
(total value, total budget, benefit, strategy), the recommendation and what the Board is asked
to do, a one-sentence statement, then Background, Risks, Benefits and Approvals. This builder
measures exactly what that paper reads off the deal record, and states as NOT RECORDED what the
paper states as not recorded -- no invented budget, strategy, forum, decision list or signature:

  * Total budget -- no budget line is recorded against a deal.
  * Strategy -- no deal is linked to a named plan (its category is not one).
  * Board forum, and the numbered decisions the Board is asked to take -- written against a
    negotiation mandate no endpoint records.
  * Approval decisions -- who MUST sign is derived (services/approval_route.py, the screen's
    rules); who HAS signed is recorded nowhere.

The category behind the approval route arrives in the scope, from the screen, because the
source carries none: it is the category a person confirmed in SpendIQ, and the paper says so.

Figures are in the deal's own currency -- one deal, one currency, no conversion. Bids in
another currency are not compared. Supplier names ride on figure labels only (the post-check
reads labels as references); the composer is shown bids by rank, never by name, because most
supplier names in this corpus carry digits a model copies into prose (see
builders/supplier_criticality_review.py).
"""
from __future__ import annotations

import re
from decimal import Decimal
from typing import Dict, List, Optional

from src.services import approval_route
from src.services.analytics.models import Confidence
from src.services.rga.builders.exec_procurement_summary import _fetch
from src.services.rga.factpack import FactBuilder, register
from src.services.rga.models import FormatHint

REPORT_TYPE_ID = "board_paper"
SECTION_ORDER = ["overview", "recommendation", "background", "risks", "benefits", "approvals"]

_DEAL = """
SELECT deal_name, supplier_id, supplier_name, quote_count, po_count, invoice_count,
       quote_total, po_total, invoice_total, currency, three_way_match, cycle_days_quote_to_po
  FROM proc.bp_deal_overview
 WHERE deal_id = %s
"""

# Each supplier's latest bid on the deal.
_BIDS = """
SELECT DISTINCT ON (q.supplier_id)
       q.supplier_id,
       COALESCE((SELECT MAX(s.supplier_name) FROM proc.bp_supplier s
                  WHERE s.supplier_id = q.supplier_id),
                (SELECT MAX(o.supplier_name) FROM proc.bp_deal_overview o
                  WHERE o.supplier_id = q.supplier_id), q.supplier_id),
       q.total_amount, q.currency
  FROM proc.bp_quote_trgt q
 WHERE q.deal_id = %s AND q.total_amount IS NOT NULL AND q.supplier_id IS NOT NULL
 ORDER BY q.supplier_id, q.quote_date DESC NULLS LAST, q.quote_id DESC
"""

_CHECKS = """
SELECT COALESCE(category, 'other'), severity, COUNT(*)::int
  FROM proc.bp_detection_finding
 WHERE deal_id = %s AND status = 'open'
 GROUP BY 1, 2
"""

_OPPS = """
SELECT COUNT(*)::int, SUM(financial_impact_gbp)::numeric
  FROM proc.bp_opportunity
 WHERE deal_id = %s AND retired_at IS NULL
"""

#: The fixed measures, in fact-id order (F0001 onwards) for every deal, so an edited or
#: hand-written paper's references always mean the same thing. Named suppliers, approval
#: steps and check groups follow them.
FIXED_LABELS = [
    "Total value", "Total budget", "Identified benefit (GBP)", "Strategy", "Board forum",
    "Documents on file", "Quotes on file", "Purchase orders on file", "Invoices on file",
    "Suppliers who bid", "Spread from the lowest to the highest bid",
    "Recoverable by taking the lowest bid", "Three-way match rate for this deal",
    "Quote-to-PO cycle", "Open checks", "Critical checks", "Warning checks",
    "Opportunities raised on this deal", "Approvers required for this deal",
    "Decisions the Board is asked to take", "Approval decisions recorded",
]

_PLACEMENT = {"required": "required", "not-required": "not required at this value",
              "untestable": "cannot be tested"}
_ORDINAL = ["", "second ", "third ", "fourth ", "fifth ", "sixth ", "seventh ", "eighth ",
            "ninth ", "tenth "]


def _money_label(label: str, currency: Optional[str]) -> str:
    return f"{label} ({currency})" if currency else label


def _int(fb: FactBuilder, label: str, value, derivation: str, unit: Optional[str] = None,
         confidence: Confidence = Confidence.ASSERTED) -> None:
    fb.add(label=label, value=Decimal(value), derivation=derivation, confidence=confidence,
           format_hint=FormatHint.INT, unit=unit)


def build(fb: FactBuilder) -> None:
    deal_id = fb.scope["deal_id"]
    rows = _fetch(_DEAL, (deal_id,))
    deal = rows[0] if rows else None
    (name, supplier_id, _supplier_name, n_q, n_po, n_inv, q_tot, po_tot, inv_tot, ccy,
     three_way, cycle) = deal if deal else (None,) * 12
    missing = "this deal is not on the record, so there is nothing measured to state"

    # ---- the four tiles ---------------------------------------------------------------
    value = next((v for v in (inv_tot, po_tot, q_tot) if v is not None), None)
    basis = ("invoiced" if inv_tot is not None else "ordered" if po_tot is not None
             else "quoted")
    if deal and value is not None:
        fb.add(label=_money_label("Total value", ccy), value=Decimal(value),
               derivation=f"board_paper.total_value[{basis}]", confidence=Confidence.CORROBORATED,
               format_hint=FormatHint.MONEY_EXACT, currency=ccy)
    else:
        fb.unmeasured(label=_money_label("Total value", ccy), derivation="board_paper.total_value",
                      reason=missing if not deal else
                      "no quote, order or invoice on the deal carries an amount")
    fb.unmeasured(label="Total budget", derivation="board_paper.budget",
                  reason="no approved budget line is recorded against this deal")

    opp_n, opp_value = (_fetch(_OPPS, (deal_id,)) or [(0, None)])[0]
    if opp_value is not None:
        fb.add(label="Identified benefit (GBP)", value=Decimal(opp_value),
               derivation="board_paper.opportunity_value", confidence=Confidence.CORROBORATED,
               format_hint=FormatHint.MONEY_EXACT, currency="GBP")
    else:
        fb.unmeasured(label="Identified benefit (GBP)", derivation="board_paper.opportunity_value",
                      reason="no opportunity has been raised on this deal, so no benefit is "
                             "stated -- which is not the same as a benefit of nothing")
    fb.unmeasured(label="Strategy", derivation="board_paper.strategy",
                  reason="the deal is not linked to a named plan; its category is not one")
    fb.unmeasured(label="Board forum", derivation="board_paper.forum",
                  reason="no forum, meeting date or paper reference is recorded for this deal")

    # ---- the evidence -----------------------------------------------------------------
    counts = [("Documents on file", (n_q or 0) + (n_po or 0) + (n_inv or 0), "documents"),
              ("Quotes on file", n_q, "quotes"), ("Purchase orders on file", n_po, "orders"),
              ("Invoices on file", n_inv, "invoices")]
    for label, n, unit in counts:
        if deal:
            _int(fb, label, n or 0, f"board_paper.{unit}", unit)
        else:
            fb.unmeasured(label=label, derivation=f"board_paper.{unit}", reason=missing)

    bids = sorted(_fetch(_BIDS, (deal_id,)), key=lambda b: (Decimal(b[2]), b[0]))
    _int(fb, "Suppliers who bid", len(bids), "board_paper.bidders", "suppliers")
    one_currency = bool(bids) and len({b[3] for b in bids}) == 1
    bid_ccy = bids[0][3] if one_currency else None
    if len(bids) > 1 and one_currency and bids[0][2] > 0:
        low, high = Decimal(bids[0][2]), Decimal(bids[-1][2])
        fb.add(label="Spread from the lowest to the highest bid", value=(high - low) / low * 100,
               derivation="board_paper.bid_spread", confidence=Confidence.CORROBORATED,
               format_hint=FormatHint.PCT)
        own = next((Decimal(b[2]) for b in bids if b[0] == supplier_id), None)
        if own is not None:
            fb.add(label=_money_label("Recoverable by taking the lowest bid", bid_ccy),
                   value=own - low, derivation="board_paper.recoverable",
                   confidence=Confidence.CORROBORATED, format_hint=FormatHint.MONEY_EXACT,
                   currency=bid_ccy)
        else:
            fb.unmeasured(label=_money_label("Recoverable by taking the lowest bid", bid_ccy),
                          derivation="board_paper.recoverable",
                          reason="the chosen supplier's own bid is not on file to compare")
    else:
        why = ("only one supplier bid, so there is no competing bid to measure against"
               if len(bids) <= 1 else
               "the bids are in different currencies and are not compared")
        fb.unmeasured(label="Spread from the lowest to the highest bid",
                      derivation="board_paper.bid_spread", reason=why)
        fb.unmeasured(label=_money_label("Recoverable by taking the lowest bid", bid_ccy or ccy),
                      derivation="board_paper.recoverable", reason=why)

    if three_way is None:
        fb.unmeasured(label="Three-way match rate for this deal",
                      derivation="board_paper.three_way_match",
                      reason="the three-way match has not been run against this deal")
    else:
        fb.add(label="Three-way match rate for this deal", value=Decimal(100 if three_way else 0),
               derivation="board_paper.three_way_match", confidence=Confidence.CORROBORATED,
               format_hint=FormatHint.PCT)
    if cycle is None:
        fb.unmeasured(label="Quote-to-PO cycle", derivation="board_paper.cycle_days",
                      reason="the deal does not record both a quote date and an order date")
    else:
        _int(fb, "Quote-to-PO cycle", cycle, "board_paper.cycle_days", "days",
             Confidence.CORROBORATED)

    checks = _fetch(_CHECKS, (deal_id,))
    by_sev: Dict[str, int] = {}
    by_kind: Dict[str, int] = {}
    for kind, sev, n in checks:
        by_sev[sev] = by_sev.get(sev, 0) + n
        by_kind[kind] = by_kind.get(kind, 0) + n
    _int(fb, "Open checks", sum(by_sev.values()), "board_paper.open_checks", "checks")
    _int(fb, "Critical checks", by_sev.get("critical", 0), "board_paper.critical_checks", "checks")
    _int(fb, "Warning checks", by_sev.get("warning", 0), "board_paper.warning_checks", "checks")
    _int(fb, "Opportunities raised on this deal", opp_n or 0, "board_paper.opportunities",
         "opportunities")

    # ---- approvals ----------------------------------------------------------------------
    category = fb.scope.get("category")
    route = None
    if category:
        route = approval_route.route(category, Decimal(value) if value is not None else None, ccy)
        _int(fb, "Approvers required for this deal",
             sum(1 for r in route.rows if r.placement == "required"),
             f"board_paper.approval_route[{route.matrix}] @ category confirmed in SpendIQ",
             "approvers")
    else:
        fb.unmeasured(label="Approvers required for this deal",
                      derivation="board_paper.approval_route",
                      reason="no category is resolved for this deal, and the approval route "
                             "is derived from category and value")
    fb.unmeasured(label="Decisions the Board is asked to take",
                  derivation="board_paper.board_decisions",
                  reason="the decisions a board paper asks for are written against a "
                         "negotiation mandate, and none is recorded for this deal")
    fb.unmeasured(label="Approval decisions recorded", derivation="board_paper.signatures",
                  reason="no approval decision is recorded against a deal, so the paper can "
                         "show who must sign but not who has")

    # ---- variable: bids, approval steps, check groups ---------------------------------
    for rank, (sid, sname, amount, bccy) in enumerate(bids, start=1):
        fb.add(label=_money_label(f"Bid from {sname or sid}", bccy), value=Decimal(amount),
               derivation=f"board_paper.bid[rank {rank}][{sid}]",
               confidence=Confidence.CORROBORATED, format_hint=FormatHint.MONEY_EXACT,
               currency=bccy)
        # The exact gap to the lowest bid. Displayed amounts are compact (£1.2M), so three bids
        # a few per cent apart all read the same; live, the model called them identical.
        if one_currency:
            fb.add(label=_money_label(f"Above the lowest bid — {sname or sid}", bccy),
                   value=Decimal(amount) - Decimal(bids[0][2]),
                   derivation=f"board_paper.bid_gap[rank {rank}][{sid}]",
                   confidence=Confidence.CORROBORATED, format_hint=FormatHint.MONEY_EXACT,
                   currency=bccy)
    if route is not None:
        for step, r in enumerate(route.rows, start=1):
            _int(fb, f"Approval step — {r.who}: {_PLACEMENT[r.placement]}. {r.why}", step,
                 f"board_paper.approval_step[{route.matrix}][{r.who}][{r.placement}]", "step")
    for kind in sorted(by_kind, key=lambda k: (-by_kind[k], k)):
        _int(fb, f"Open checks — {kind}", by_kind[kind], f"board_paper.checks[{kind}]", "checks")


_BID = re.compile(r"^board_paper\.(bid|bid_gap)\[rank (\d+)\]")
_STEP = re.compile(r"^board_paper\.approval_step\[[^\]]*\]\[([^\]]*)\]\[([^\]]*)\]")


def composer_label(entry) -> str:
    """How the composer is shown a figure. Bids by rank, never by supplier name (names carry
    digits it copies into prose); an approval step by who and placement, without the rule text,
    whose amounts ('> £250k') it would copy too. The page prints the full label."""
    m = _BID.match(entry.derivation)
    if m:
        rank = int(m.group(2))
        word = _ORDINAL[rank - 1] if rank <= len(_ORDINAL) else "next "
        if m.group(1) == "bid_gap":
            return f"Above the lowest bid — the {word}lowest bidder ({entry.currency})"
        return f"Bid from the {word}lowest bidder ({entry.currency})"
    m = _STEP.match(entry.derivation)
    if m:
        return f"Approval step — {m.group(1)}: {_PLACEMENT[m.group(2)]}"
    return entry.label


COMPOSER_NOTE = (
    "This is a board paper, not a bullet dump: plain British English, third person, full "
    "sentences. The sections follow the Pipeline's board paper -- overview, recommendation, "
    "background, risks, benefits, approvals. The approval route says who must sign, not who "
    "has. Bids are shown to you by rank; their suppliers are named on the figure cards, so "
    "never write a supplier's name in a sentence and never number anything yourself. The "
    "budget, strategy, forum and the decisions the Board is asked to take are not recorded: "
    "say so plainly, once, and recommend nothing on them. Use at most one findings list in "
    "the whole paper. Displayed amounts are exact; state them as placed.")

def _find(facts, derivation_prefix: str):
    return next((f for f in facts if f.derivation.startswith(derivation_prefix)), None)


def deal_note(facts) -> str:
    """This deal's recommendations and plain statements, for the composer -- derived from the
    record the way the Pipeline paper derives them (engine.js dealRecommendations), so the
    paper recommends what the record supports and nothing else. Live 2026-09-25, left to
    reason from the figures, the model called three different bids identical, said one bid
    when three suppliers bid, and said approvals had been completed when none is recorded.
    Figures appear only as placeholders: a digit here is a digit it would copy into prose."""
    ref = lambda f: "{{" + f.fact_id + "}}"
    rec, state = [], []
    crit = _find(facts, "board_paper.critical_checks")
    if crit is not None and crit.value:
        rec.append(f"Resolve the {ref(crit)} critical checks open against this deal before "
                   "anything else.")
    tw = _find(facts, "board_paper.three_way_match")
    if tw is not None and tw.value is not None and tw.value == 0:
        rec.append("Reconcile the quote, purchase order and invoice: they do not three-way "
                   "match.")
    elif tw is not None and tw.value is not None:
        state.append("The quote, purchase order and invoice three-way match.")
    else:
        state.append("The three-way match has not been run against this deal.")
    rcv = _find(facts, "board_paper.recoverable")
    if rcv is not None and rcv.value:
        rec.append(f"Taking the lowest bid would recover {ref(rcv)}.")
    elif rcv is not None and rcv.value is not None:
        state.append("The chosen supplier's bid is already the lowest bid.")
    bidders = _find(facts, "board_paper.bidders")
    if bidders is not None and bidders.value and bidders.value > 1:
        state.append(f"{ref(bidders)} suppliers bid, so there is a competing benchmark. The "
                     "bids differ: compare them only through the spread and each bid's "
                     "figure above the lowest bid.")
    elif bidders is not None and bidders.value == 1:
        rec.append("Only one supplier bid, so there is no competing benchmark.")
    else:
        state.append("No bid is on file for this deal.")
    if not rec:
        rec.append("Nothing measured on this deal is asking for a decision -- say so, as a "
                   "result read off the figures.")
    state.append("No approval decision is recorded: never say anyone has approved, signed or "
                 "cleared this deal, or that it has passed through its approvals.")
    state.append("Do not recommend that the Board approves or rejects the deal; that decision "
                 "is the Board's, and no mandate is recorded.")
    return ("RECOMMEND (write these as the recommendation, and no others):\n  - "
            + "\n  - ".join(rec)
            + "\nSTATE AS THEY ARE:\n  - " + "\n  - ".join(state))


def composer_note(pack) -> str:
    return COMPOSER_NOTE + "\n" + deal_note(pack.facts)


# Sentences that say an approval happened. None is recorded for any deal, and live the model
# wrote "the deal has progressed through the required approval steps" even when told not to.
# Narrow on purpose: "has not been approved" and "the route is not completed" are true and pass.
_APPROVAL_CLAIMS = [
    re.compile(r"\b(progressed|passed|gone|moved)\s+through\b[^.]*\bapprov", re.I),
    re.compile(r"\b(has|have|had)\s+been\s+(formally\s+)?(approved|signed off|cleared)\b", re.I),
    re.compile(r"\b(was|were)\s+(formally\s+)?(approved|signed off|cleared)\b", re.I),
    re.compile(r"\bapprovals?\b[^.]*\b(were|was|have been|has been)\s+"
               r"(completed|obtained|given|granted)\b", re.I),
]


def prose_faults(ast) -> List[str]:
    """The composer's sentences that claim an approval the record does not hold -- each a
    fault that sends the draft back for one corrected attempt (compose_report)."""
    from src.services.rga.models import NarrativeBlock
    faults = []
    for section in ast.sections:
        for block in section.blocks:
            if isinstance(block, NarrativeBlock):
                for sentence in re.split(r"(?<=[.!?])\s+", block.text):
                    if any(p.search(sentence) for p in _APPROVAL_CLAIMS):
                        faults.append(
                            f'the sentence "{sentence[:160]}" says an approval happened, but no '
                            "approval decision is recorded for any deal: say who must sign, and "
                            "that no approval decision is recorded")
    return faults


# Registered last: the composer's note and label view are defined after the builder.
register(REPORT_TYPE_ID, section_order=SECTION_ORDER, title="Board paper",
         composer_note=composer_note, composer_label=composer_label,
         prose_rules=prose_faults)(build)
