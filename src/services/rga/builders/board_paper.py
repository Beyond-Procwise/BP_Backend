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
from src.services.version_collapse import QUOTE_BASE_SQL, QUOTE_VERSION_SQL

REPORT_TYPE_ID = "board_paper"
SECTION_ORDER = ["overview", "recommendation", "background", "risks", "benefits", "approvals"]

_DEAL = """
SELECT deal_name, supplier_id, supplier_name, quote_count, po_count, invoice_count,
       quote_total, po_total, invoice_total, currency, value_reconciled,
       cycle_days_quote_to_po, three_way_matched,
       -- How much of the delivery check could actually be done, so the rate
       -- below has an honest denominator and the share nothing can prove is
       -- stated rather than quietly excluded (design section 13.3).
       (SELECT coalesce(sum(g.lines_assessed), 0) FROM proc.bp_goods_receipt_trgt g
         WHERE g.deal_id = o.deal_id)                        AS lines_assessed,
       (SELECT coalesce(sum(g.lines_unverifiable), 0) FROM proc.bp_goods_receipt_trgt g
         WHERE g.deal_id = o.deal_id)                        AS lines_unverifiable,
       (SELECT count(*) FROM proc.bp_extraction_discrepancy x
          JOIN proc.bp_invoice_trgt i ON i.invoice_id = x.doc_pk_candidate
         WHERE x.doc_type = 'invoice' AND x.status = 'open'
           AND x.issue_type IN ('billed_not_received','nothing_received')
           AND i.deal_id = o.deal_id)                        AS open_gaps
  FROM proc.bp_deal_overview o
 WHERE o.deal_id = %s
"""

# Each bid on the deal at its latest VERSION -- one per supplier quote (a supplier's separate
# lots are separate bids; a quote with no resolved supplier is still a bid, named by its own
# reference). "Latest" used to mean latest by date, so a mis-dated V1 could stand in for V3,
# and a V3 with no readable total fell back to V2's price; such a bid now has no standing price.
_BIDS = f"""
SELECT bid_supplier, bid_name, total_amount, currency FROM (
  SELECT DISTINCT ON (COALESCE(q.supplier_id, ''), {QUOTE_BASE_SQL('q.quote_id')})
         COALESCE(q.supplier_id, 'quote ' || {QUOTE_BASE_SQL('q.quote_id')}) AS bid_supplier,
         COALESCE((SELECT MAX(s.supplier_name) FROM proc.bp_supplier s
                    WHERE s.supplier_id = q.supplier_id),
                  q.supplier_id, {QUOTE_BASE_SQL('q.quote_id')}) AS bid_name,
         q.total_amount, q.currency
    FROM proc.bp_quote_trgt q
   WHERE q.deal_id = %s
   ORDER BY COALESCE(q.supplier_id, ''), {QUOTE_BASE_SQL('q.quote_id')},
            {QUOTE_VERSION_SQL('q.quote_id')} DESC, q.quote_id DESC
) latest
 WHERE total_amount IS NOT NULL
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
    "Recoverable by taking the lowest bid", "Quote, PO and invoice reconciled",
    "Quote-to-PO cycle", "Open checks", "Critical checks", "Warning checks",
    "Opportunities raised on this deal", "Approvers required for this deal",
    "Decisions the Board is asked to take", "Approval decisions recorded",
    # Appended, not placed beside "Quote, PO and invoice reconciled" where they
    # belong by subject: this list is in FACT-ID order, and inserting would
    # renumber every measure after it, so an already-written paper's F0013
    # would silently come to mean something else.
    "Goods billed were received",
    "Purchase-order lines checked against a delivery note",
    "Purchase-order lines no delivery note can prove",
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
     reconciled, cycle, received, n_assessed, n_unverifiable,
     n_gaps) = deal if deal else (None,) * 16
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
    _int(fb, "Suppliers who bid", len({b[0] for b in bids}), "board_paper.bidders", "suppliers")
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

    if reconciled is None:
        fb.unmeasured(label="Quote, PO and invoice reconciled",
                      derivation="board_paper.value_reconciled",
                      reason="no reconciliation result has been recorded for this deal")
    else:
        fb.add(label="Quote, PO and invoice reconciled", value=Decimal(100 if reconciled else 0),
               derivation="board_paper.value_reconciled", confidence=Confidence.CORROBORATED,
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

    # The second reconciliation line, and the one a board actually asks about.
    # "Quote, PO and invoice reconciled" above says the AMOUNTS agree; this says
    # the goods arrived. NULL is NOT ASSESSED and must read that way: a deal
    # nobody sent a delivery note for has not failed the check, and rendering it
    # as 0% would put a failure in front of a board that no evidence supports.
    #
    # It is a RATE OVER THE LINES THAT COULD BE CHECKED, not the deal's boolean
    # rendered as a percentage. The first version printed Decimal(100 if x else 0)
    # with a PCT hint, so a deal where one line passed, two failed and one could
    # not be assessed published as "0.0%" -- which is not what the data says.
    # The two counts beneath it carry the denominator.
    assessed = int(n_assessed or 0)
    if received is None or assessed <= 0:
        fb.unmeasured(label="Goods billed were received",
                      derivation="board_paper.three_way_matched",
                      reason="no delivery note on this deal records a line that could be "
                             "compared, so what was billed has not been checked against "
                             "what arrived")
    else:
        clean = max(assessed - int(n_gaps or 0), 0)
        fb.add(label="Goods billed were received",
               value=(Decimal(clean) / Decimal(assessed) * 100),
               derivation="board_paper.three_way_matched",
               confidence=Confidence.CORROBORATED, format_hint=FormatHint.PCT)

    # The denominator, stated. Section 13.3 of the design: two fifths of
    # purchase-order lines can never be covered by a delivery note, and the
    # board paper must show that as a denominator rather than quietly excluding
    # it from the rate above.
    if deal:
        _int(fb, "Purchase-order lines checked against a delivery note",
             assessed, "board_paper.lines_assessed", "lines",
             Confidence.CORROBORATED)
        _int(fb, "Purchase-order lines no delivery note can prove",
             int(n_unverifiable or 0), "board_paper.lines_unverifiable", "lines",
             Confidence.CORROBORATED)
    else:
        fb.unmeasured(label="Purchase-order lines checked against a delivery note",
                      derivation="board_paper.lines_assessed",
                      reason="this deal is not on the record")
        fb.unmeasured(label="Purchase-order lines no delivery note can prove",
                      derivation="board_paper.lines_unverifiable",
                      reason="this deal is not on the record")

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
    tw = _find(facts, "board_paper.value_reconciled")
    if tw is not None and tw.value is not None and tw.value == 0:
        rec.append("Reconcile the quote, purchase order and invoice: their values do "
                   "not agree.")
    elif tw is not None and tw.value is not None:
        state.append("The quote, purchase order and invoice values reconcile.")
    else:
        state.append("No reconciliation result has been recorded for this deal.")
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
