"""The supplier criticality review, measured from what the platform can trace (2026-09-24).

The brief builds this report from a supplier criticality scoring system ("CISE") and supplier
risk signals. Neither exists here, and this module will not stand one in:

1. **No criticality score is stated.** The third-party risk register (proc.bp_tprm_supplier)
   is seeded data -- 120 rows, every one "Material outsourcing", and not one of its names
   matches a supplier on a deal. Presenting it as criticality would be a figure nobody
   measured. "Supplier criticality rating" is therefore recorded as not assessed, with the
   reason, so a board sees exactly where the gap is (user ruling 2026-09-24).
2. **No single-source figure.** Single sourcing is "the only supplier for something", and the
   corpus has no category dimension to say what that something is.
3. **No contract coverage.** The contracts on record (proc.bp_contract_master) number their
   suppliers in a scheme no crosswalk links to the suppliers on deals, so no deal's spend can
   be shown to sit under a contract.

What IS measured: how concentrated the period's spend is, who the five largest suppliers are
and what they were paid, and the open findings raised against their deals. Amounts are
converted one currency at a time (never summed across currencies, never at par), exactly as
the executive summary does -- see builders/exec_procurement_summary.py for the corpus facts
behind the period predicate and the currency rule.

Supplier names carry digits in this corpus ("Ashcroft Associates 10"). They appear only as the
LABELS of figures, which the post-check reads as references, never as prose or table text a
model writes.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from decimal import Decimal
from typing import Dict, List, Tuple

from src.services.analytics.models import Confidence
from src.services.rga.builders.exec_procurement_summary import _PERIOD, _fetch, _rates
from src.services.rga.factpack import FactBuilder, register
from src.services.rga.models import FindingCode, FormatHint, Severity

REPORT_TYPE_ID = "supplier_criticality_review"

SECTION_ORDER = ["summary", "concentration", "top_suppliers", "findings", "gaps"]

TOP_N = 5
#: The share of spend "Suppliers making up four-fifths of spend" counts up to.
CONCENTRATION_SHARE = Decimal("0.8")

_SUPPLIERS = f"""
SELECT COUNT(DISTINCT supplier_id)::int
  FROM proc.bp_deal_overview
 WHERE {_PERIOD}
"""

# Per row, so each amount is converted from its own currency.
_AMOUNTS = f"""
SELECT supplier_id, supplier_name, invoice_total, currency
  FROM proc.bp_deal_overview
 WHERE {_PERIOD}
   AND invoice_total IS NOT NULL
   AND supplier_id IS NOT NULL
 ORDER BY supplier_id
"""

_FINDINGS = f"""
SELECT o.supplier_id, COUNT(*)::int
  FROM proc.bp_detection_finding f
  JOIN proc.bp_deal_overview o ON o.deal_id = f.deal_id
 WHERE f.status = 'open'
   AND {_PERIOD.replace('deal_date', 'o.deal_date').replace('first_activity_date', 'o.first_activity_date')}
 GROUP BY o.supplier_id
"""


def _per_supplier(rows, display) -> Tuple[Dict[str, Decimal], Dict[str, str], int, List[str]]:
    """Each supplier's spend in the display currency, their name, and what was excluded."""
    grouped: "OrderedDict[str, list]" = OrderedDict()
    names: Dict[str, str] = {}
    for supplier_id, name, amount, currency in rows:
        grouped.setdefault(supplier_id, []).append((amount, currency))
        # The name as the deal overview spells it; the greatest, so it is stable.
        names[supplier_id] = max(names.get(supplier_id) or "", name or supplier_id)
    spend: Dict[str, Decimal] = {}
    excluded, missing = 0, []
    for supplier_id, pairs in grouped.items():
        result = display.total(pairs)
        excluded += result.excluded
        missing += [c for c in result.excluded_currencies if c not in missing]
        if result.value is not None:
            spend[supplier_id] = result.value
    return spend, names, excluded, missing


_RANK_WORDS = {1: "the largest supplier", 2: "the second largest supplier",
               3: "the third largest supplier", 4: "the fourth largest supplier",
               5: "the fifth largest supplier"}
_RANKED = re.compile(r"^supplier_review\.supplier_(spend|share|open_findings)\[rank (\d)\]")


def composer_label(entry) -> str:
    """How the composer is shown a figure: a named supplier's figures by rank, never by name.

    Live, 2026-09-24, the model copied 'Lighthouse Associates 13' into prose in three runs of
    three even when told not to, and a typed number in prose refuses the report. A name it is
    never shown is a name it cannot copy. The page still prints the real name -- from the
    fact's own label -- beside the figure, and 'the second largest supplier received X' is
    true of the figure it places.
    """
    m = _RANKED.match(entry.derivation)
    if not m:
        return entry.label
    who = _RANK_WORDS[int(m.group(2))]
    if m.group(1) == "spend":
        return f"Spend with {who} ({entry.currency})"
    if m.group(1) == "share":
        return f"Share of spend — {who}"
    return f"Open findings — {who}"


# Told to the composer with every run of this report. Live, 2026-09-24: shown suppliers by
# rank, the model then numbered them itself -- a "Rank" column of 1, 2, 3 -- and the report
# was refused for typed numbers in table cells.
COMPOSER_NOTE = (
    "Suppliers are shown to you by rank; their names are printed on their figure cards. Show "
    "a supplier with a figure card or a chart of their figures, and in sentences say 'the "
    "largest supplier', 'the second largest supplier' and so on -- never write a supplier's "
    "name in a sentence. Never number anything yourself: no rank column and no row numbers "
    "in a table -- a table cell is a figure id or words. The criticality rating, "
    "single sourcing and contract coverage were not measured: say so plainly, and recommend "
    "nothing on them.")


@register(REPORT_TYPE_ID, section_order=SECTION_ORDER, composer_note=COMPOSER_NOTE,
          composer_label=composer_label, title="Supplier criticality review")
def build(fb: FactBuilder) -> None:
    """Measure the supplier review for ``fb.scope`` (period_start, period_end, currency).

    The first nine facts are the same measures in the same order for every period, so an
    edited or hand-written report's references (F0001..F0009) always mean the same thing;
    the named suppliers follow.
    """
    start, end = fb.scope["period_start"], fb.scope["period_end"]
    target = fb.scope.get("currency", "GBP")
    period = (start, end)

    # F0001 -- how many suppliers were paid in the period.
    (suppliers,), = _fetch(_SUPPLIERS, period)
    fb.add(label="Suppliers transacted with", value=Decimal(suppliers),
           derivation="supplier_review.suppliers", confidence=Confidence.ASSERTED,
           format_hint=FormatHint.INT, unit="suppliers")

    rows = _fetch(_AMOUNTS, period)
    rates, fetched_at, unavailable = _rates()
    spend: Dict[str, Decimal] = {}
    names: Dict[str, str] = {}
    total = None
    complete = True
    note = "rates unavailable"
    if rows and rates and not unavailable:
        from src.services.analytics.currency import DisplayCurrency

        display = DisplayCurrency(target=target, rates=rates, fetched_at=fetched_at)
        note = display.rate_note()
        spend, names, excluded, missing = _per_supplier(rows, display)
        complete = excluded == 0
        if spend:
            total = sum(spend.values(), Decimal(0))
        if not complete:
            fb.finding(code=FindingCode.PARTIAL_CURRENCY_CONVERSION, severity=Severity.HIGH,
                       detail=(f"{excluded} deal(s) excluded from the {target} figures for "
                               f"want of a rate: {', '.join(missing)}"),
                       blocks_release=False)
    # A partial conversion understates totals AND skews every share, so all of them drop to
    # UNASSESSED together -- a share of an understated total presented as measured is wrong.
    money_confidence = Confidence.CORROBORATED if complete else Confidence.UNASSESSED

    # F0002 -- the period's spend.
    if total is None:
        fb.unmeasured(label=f"Invoiced spend ({target})", derivation="supplier_review.spend",
                      reason="no amount in the period could be stated in the display "
                             "currency, so no total and no shares are given")
    else:
        fb.add(label=f"Invoiced spend ({target})", value=total,
               derivation=f"supplier_review.spend @ {note}", confidence=money_confidence,
               format_hint=FormatHint.MONEY, currency=target)

    ranked = sorted(spend.items(), key=lambda kv: (-kv[1], kv[0]))

    # F0003..F0005 -- concentration.
    if total:
        top = ranked[:TOP_N]
        fb.add(label="Largest supplier's share of spend",
               value=ranked[0][1] / total * 100, derivation="supplier_review.largest_share",
               confidence=money_confidence, format_hint=FormatHint.PCT)
        fb.add(label="Top five suppliers' share of spend",
               value=sum((v for _, v in top), Decimal(0)) / total * 100,
               derivation="supplier_review.top_five_share",
               confidence=money_confidence, format_hint=FormatHint.PCT)
        running, count = Decimal(0), 0
        for _, value in ranked:
            running += value
            count += 1
            if running >= total * CONCENTRATION_SHARE:
                break
        fb.add(label="Suppliers making up four-fifths of spend", value=Decimal(count),
               derivation="supplier_review.suppliers_to_four_fifths",
               confidence=money_confidence, format_hint=FormatHint.INT, unit="suppliers")
    else:
        reason = "there is no stated spend in the period to divide"
        for label, derivation in (("Largest supplier's share of spend", "largest_share"),
                                  ("Top five suppliers' share of spend", "top_five_share"),
                                  ("Suppliers making up four-fifths of spend",
                                   "suppliers_to_four_fifths")):
            fb.unmeasured(label=label, derivation=f"supplier_review.{derivation}", reason=reason)

    # F0006 -- open findings on the period's deals (a count; measurable without rates).
    open_by_supplier = {sid: n for sid, n in _fetch(_FINDINGS, period)}
    fb.add(label="Open findings against the period's deals",
           value=Decimal(sum(open_by_supplier.values())),
           derivation="supplier_review.open_findings", confidence=Confidence.ASSERTED,
           format_hint=FormatHint.INT, unit="findings")

    # F0007..F0009 -- what cannot be measured here, said plainly (see the module docstring).
    fb.unmeasured(label="Supplier criticality rating",
                  derivation="supplier_review.criticality",
                  reason="no supplier criticality assessment exists on the platform; the "
                         "third-party risk register holds sample entries that match none of "
                         "the suppliers bought from, so no rating is given")
    fb.unmeasured(label="Single-source dependence",
                  derivation="supplier_review.single_source",
                  reason="purchases carry no category, so it cannot be said which goods or "
                         "services depend on a single supplier")
    fb.unmeasured(label="Spend covered by a contract",
                  derivation="supplier_review.contract_coverage",
                  reason="contracts on record number their suppliers differently from the "
                         "deals, and nothing links the two, so coverage cannot be measured")

    # F0010 onwards -- the five largest suppliers, each named on its figures' labels. The rank
    # rides in the derivation, which is how the composer is shown them (composer_label).
    for rank, (supplier_id, value) in enumerate(ranked[:TOP_N], start=1):
        name = names.get(supplier_id) or supplier_id
        fb.add(label=f"Spend with {name} ({target})", value=value,
               derivation=f"supplier_review.supplier_spend[rank {rank}][{supplier_id}] @ {note}",
               confidence=money_confidence, format_hint=FormatHint.MONEY, currency=target)
        fb.add(label=f"Share of spend — {name}", value=value / total * 100,
               derivation=f"supplier_review.supplier_share[rank {rank}][{supplier_id}]",
               confidence=money_confidence, format_hint=FormatHint.PCT)
        fb.add(label=f"Open findings — {name}",
               value=Decimal(open_by_supplier.get(supplier_id, 0)),
               derivation=f"supplier_review.supplier_open_findings[rank {rank}][{supplier_id}]",
               confidence=Confidence.ASSERTED, format_hint=FormatHint.INT, unit="findings")
