"""The ``corpus_facts`` lookup through the gateway: same intents, same fact keys, but read as the person asking.

``corpus_facts.detect_intent`` (rules, no model) still picks the intent. Here each intent maps to the one gateway function that
answers it, and the result is shaped as the legacy facts dict so ``corpus_facts.render_facts`` builds the same prompt text.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from .context import CallerContext
from .gateway import Gateway

logger = logging.getLogger(__name__)

INTENT_FUNCTIONS: Dict[str, str] = {
    "policies": "list_policies",
    "deals": "list_deals",
    "findings": "get_findings_overview",
    "spend": "get_spend_summary",
    "quotes": "get_quote_overview",
    "purchase_orders": "get_purchase_order_overview",
    "invoices": "get_invoice_overview",
    "suppliers": "get_supplier_overview",
}

_UNAVAILABLE = {"corpus_lookup_unavailable": [{
    "note": "the live record counts could not be read right now; do not state totals or counts, and say they could not be checked"}]}


def _spend_facts(data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """The spend summary as the legacy ``spend`` facts. A supplier billed in two currencies is two rows, never one summed figure."""
    return {
        "invoiced_spend_by_supplier": [{"supplier_name": r["supplier_name"], "invoiced_amount": r["amount"], "currency": r["currency"]}
                                       for r in data["suppliers"]],
        "totals_by_currency": [{"currency": c, "invoiced_amount": v} for c, v in data["totals_by_currency"].items()],
    }


def facts_via_gateway(gateway: Gateway, ctx: CallerContext, intent: Optional[str]) -> Dict[str, Any]:
    """The facts for one intent, or ``{}`` when there is no intent. A failed read says so; it is not an empty answer."""
    if not intent:
        return {}
    fn = INTENT_FUNCTIONS[intent]
    params: Dict[str, Any] = {}
    if fn == "get_spend_summary":
        params = {"period_start": dt.date(1900, 1, 1), "period_end": dt.date.today()}
    result = gateway.call(ctx, fn, params)
    if not result.ok:
        logger.warning("corpus facts via gateway: %s failed (%s)", fn, result.error.code if result.error else "?")
        return dict(_UNAVAILABLE)
    return _spend_facts(result.data) if fn == "get_spend_summary" else dict(result.data)
