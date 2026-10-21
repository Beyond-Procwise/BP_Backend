"""savings_tracker: what the value ledger says has been saved, read-only.

The figures come from ``value_summary_service.ledger_totals``, the one rule every screen uses for
"saved", so an agent never states a number the dashboard would not. This tool logs nothing: a
saving is recorded by a person through the ledger routes, with evidence.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

# Only the settled kinds are "saved"; a claim still open is shown apart and never added in.
_NOTE = ("Saved = avoided + recovered + realised, from confirmed ledger entries only. Claimed money is not yet "
         "recovered and is not in Saved. This tool records nothing; a person logs a saving with evidence.")


def _load() -> List[Dict[str, Any]]:
    from src.services import value_summary_service as vss
    from src.services.db import get_conn

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(vss._LEDGER_SQL)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def track(since: Any = None, until: Any = None, *,
          load: Optional[Callable[[], List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    from src.services import value_summary_service as vss

    for name, v in (("since", since), ("until", until)):
        try:
            bad = v is not None and vss._as_date(v) is None
        except (TypeError, ValueError):
            bad = True
        if bad:
            return {"found": False, "note": f"{name} must be a date like 2026-01-31."}
    try:
        rows = (load or _load)()
    except Exception as exc:  # noqa: BLE001 - an unreadable ledger is an outage, not zero savings
        log.warning("savings_tracker: ledger unreadable: %s", exc)
        return {"found": False, "available": False,
                "note": "The value ledger could not be read just now. This does not mean nothing has been saved."}
    totals = vss.ledger_totals(rows, since=since, until=until)
    return {"found": True, "entries": len(rows), "currency": "GBP", "totals": totals,
            "window": {"since": str(since) if since else None, "until": str(until) if until else None},
            "note": _NOTE if rows else "The ledger has no entries yet. " + _NOTE}
