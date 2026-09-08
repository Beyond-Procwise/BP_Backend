"""Where an analytic question leaves the search path and is answered exactly.

"What are the top 10 suppliers by spend?" is not a search. It has one right
answer, and every part of it — the figures, the period, the currency, the share
each supplier holds, what to look at next — is arithmetic over rows this system
already holds. Sent down the retrieval path it becomes a block of key=value
text handed to a model to format, which is how it came to be ranked on raw
mixed-currency amounts.

This module is the seam. It decides which questions belong to the analytic
layer, assembles the answer from the four parts that layer is built from, and
hands back the same payload shape ``POST /workflows/ask`` already returns.

Two rules it holds to:

  * **Take only what it can answer exactly.** A count, a policy question, a
    spend total with no ranking in it: all stay on the path they were on. The
    flag is off by default, so nothing changes until it is turned on.
  * **Never make the ask bar worse.** A fetch that raises returns ``None`` and
    the old path answers, exactly as it would have. The reader loses the better
    answer, never the answer.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence

from src.services.agent_actions import record_action
from src.services.analytics.currency import DisplayCurrency
from src.services.analytics.insight import write_insight
from src.services.analytics.next_steps import AllowList, select_next_steps
from src.services.analytics.period import Period, fiscal_year_to_date, prior_year_equivalent
from src.services.analytics.render import render_analytic_answer
from src.services.analytics.supplier_spend import (
    Lens,
    SupplierSpendRow,
    build_supplier_spend_ranking,
)

logger = logging.getLogger(__name__)

# The actions this layer serves, and the lens each one is. A step whose action
# is not here is declined rather than answered with something else: a chip that
# quietly answers a different question from the one it offered is worse than a
# chip that hands the question back to the old path.
ACTION_LENSES = {
    "analytic.supplier_spend_ranking": Lens.RANKING,
    "analytic.supplier_concentration": Lens.CONCENTRATION,
    "analytic.supplier_spend_trend": Lens.TREND,
}

DEFAULT_TOP_N = 10
# A table nobody reads, and one query away from a page that never renders.
MAX_TOP_N = 50

_SUPPLIER = re.compile(r"\b(supplier|vendor|manufacturer)s?\b", re.I)
_EXPENDITURE = re.compile(r"\b(spend|spent|spending|invoiced|billed|cost)\w*\b", re.I)
_RANKING = re.compile(r"\b(top|biggest|largest|highest|leading|rank|ranked|ranking|most)\b", re.I)
# "How many suppliers do we have" is a count. It reads as a ranking to every
# pattern above and is answered by a single number, not a table.
_COUNT = re.compile(r"\b(how many|count of|number of)\b", re.I)
_TOP_N = re.compile(r"\btop\s+(\d{1,4})\b", re.I)


def is_supplier_spend_ranking(query: str) -> bool:
    """True for the one question this layer answers today."""
    text = (query or "").strip()
    if not text or _COUNT.search(text):
        return False
    if not _SUPPLIER.search(text):
        return False
    if not _RANKING.search(text):
        return False
    # "Biggest supplier" is a spend question without saying so; "top 10
    # suppliers by discrepancy count" is not, and says so.
    return bool(_EXPENDITURE.search(text)) or bool(
        re.search(r"\b(biggest|largest|top|leading)\b", text, re.I))


def requested_top_n(query: str) -> int:
    """How many rows the reader asked for, bounded to a table that can be read."""
    match = _TOP_N.search(query or "")
    if not match:
        return DEFAULT_TOP_N
    return max(1, min(MAX_TOP_N, int(match.group(1))))


@dataclass(frozen=True)
class SpendData:
    """What the database has to say about the period, fetched once."""

    rows: Sequence[SupplierSpendRow]
    supplier_count: int
    invoice_count: int
    available_data: FrozenSet[str]
    prior_rows: Sequence[SupplierSpendRow] = field(default_factory=tuple)


def fetch_live(period: Period, prior: Period) -> SpendData:
    """The period, the year before it, and which next steps lead anywhere."""
    from src.services.analytics import repository
    from src.services.db import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            rows = repository.fetch_supplier_spend(cur, period)
            suppliers, invoices = repository.fetch_population(cur, period)
            prior_rows = repository.fetch_supplier_spend(cur, prior)
            available = repository.available_data(cur, period, prior)
    return SpendData(rows=rows, prior_rows=prior_rows, supplier_count=suppliers,
                     invoice_count=invoices, available_data=available)


def _settings() -> Any:
    from config.settings import settings

    return settings


def analytic_answer(
    query: str,
    *,
    display: Optional[DisplayCurrency] = None,
    display_currency: Optional[str] = None,
    persona: str = "default",
    action_id: Optional[str] = None,
    enabled: Optional[bool] = None,
    fetch: Optional[Callable[[Period, Period], SpendData]] = None,
    writer: Optional[Callable[[str], Optional[str]]] = None,
    audit: Callable[..., None] = record_action,
    today: Optional[date] = None,
) -> Optional[Dict[str, Any]]:
    """The answer to an analytic question, or ``None`` — meaning "not mine".

    ``None`` is returned for a question this layer does not answer, while the
    flag is off, and for any failure along the way. The caller carries on down
    the path it was already on, so the worst case is the answer the ask bar
    gives today.
    """
    settings = None
    if enabled is None:
        try:
            settings = _settings()
            enabled = bool(getattr(settings, "analytic_answer_v2_enabled", False))
        except Exception:
            logger.debug("analytic answer flag unreadable; staying off", exc_info=True)
            return None
    if not enabled:
        return None
    # A dispatched step names the answer it wants. Its label is not a question
    # and must never be parsed as one — that round trip is what lost the
    # subject of the old follow-up chips.
    lens = ACTION_LENSES.get(action_id) if action_id else (
        Lens.RANKING if is_supplier_spend_ranking(query) else None)
    if lens is None:
        return None

    try:
        if settings is None:
            try:
                settings = _settings()
            except Exception:
                settings = None
        fy_start = int(getattr(settings, "analytic_fiscal_year_start_month", 4) or 4)
        threshold = Decimal(str(getattr(settings, "analytic_concentration_threshold_pct", 30)))

        period = fiscal_year_to_date(today or date.today(), fy_start_month=fy_start)
        prior = prior_year_equivalent(period)
        data = (fetch or fetch_live)(period, prior)

        if display is None:
            # Resolved here rather than at the call site: it fetches the live
            # rate batch, and every question that is not this one must not pay
            # for a currency it will never state.
            from src.services.analytics import repository

            display = repository.display_currency_from_request(display_currency)

        answer = build_supplier_spend_ranking(
            rows=data.rows,
            prior_rows=data.prior_rows,
            prior_period=prior,
            display=display,
            period=period,
            population_count=data.supplier_count,
            invoice_count=data.invoice_count,
            answer_id=str(uuid.uuid4()),
            refreshed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            top_n=requested_top_n(query),
            concentration_threshold_pct=threshold,
            lens=lens,
        )
        answer = write_insight(answer, persona=persona, generate=writer, audit=audit)
        # Only the actions this layer can actually answer. The ladder reaches
        # composition, contract coverage, relationship owner and risk exposure,
        # and nothing is built behind any of them — a chip that dispatches into
        # nothing is the dead end the whole mechanism exists to avoid. This is
        # also where a real entitlement gate goes when there is one.
        steps = select_next_steps(answer, persona=persona,
                                  entitlements=AllowList(set(ACTION_LENSES)),
                                  available_data=data.available_data)
        answer = answer.model_copy(update={"next_steps": steps})
    except Exception:
        # The ask bar answered this question before this layer existed, and it
        # still can. A failure here is a worse answer, never no answer.
        logger.exception("analytic answer failed; falling back to the search path")
        return None

    return {
        "answer": render_analytic_answer(answer),
        # The old path asked a model for three questions and printed them under
        # every answer. Next steps replace them, and inventing one here would
        # put the thing this work removed back on the screen.
        "follow_ups": [],
        "retrieved_documents": [],
        "next_steps": [step.model_dump() for step in steps],
        "analytic_answer": answer.model_dump(mode="json"),
    }
