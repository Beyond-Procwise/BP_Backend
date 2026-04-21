import logging
import time

import requests

from src.services.structural_extractor.derivation import rule

log = logging.getLogger(__name__)
_CACHE: dict[str, tuple[float, float]] = {}  # ccy -> (rate, timestamp)
_CACHE_TTL = 3600  # 1h


def _fetch_rate_live(ccy: str) -> float | None:
    try:
        r = requests.get(f"https://open.er-api.com/v6/latest/{ccy}", timeout=5)
        if r.status_code == 200:
            data = r.json()
            rate = data.get("rates", {}).get("USD")
            if rate is not None:
                return float(rate)
    except Exception:
        log.debug("FX fetch failed for %s", ccy, exc_info=True)
    return None


def _get_rate(ccy: str) -> float | None:
    now = time.monotonic()
    cached = _CACHE.get(ccy)
    if cached is not None:
        rate, ts = cached
        if now - ts < _CACHE_TTL:
            return rate
    rate = _fetch_rate_live(ccy)
    if rate is not None:
        _CACHE[ccy] = (rate, now)
    elif cached is not None:
        # Fallback to stale cache if available
        return cached[0]
    return rate


@rule("xrate_lookup", "exchange_rate_to_usd", ["currency"])
def _xrate(inputs):
    ccy = str(inputs["currency"] or "")
    if not ccy:
        return None
    if ccy == "USD":
        return 1.0
    return _get_rate(ccy)


@rule("convert_to_usd_inv", "converted_amount_usd", ["invoice_total_incl_tax", "exchange_rate_to_usd"])
def _conv_inv(inputs):
    try:
        return round(float(inputs["invoice_total_incl_tax"]) * float(inputs["exchange_rate_to_usd"]), 2)
    except Exception:
        return None


@rule("convert_to_usd_po", "converted_amount_usd", ["total_amount_incl_tax", "exchange_rate_to_usd"])
def _conv_po(inputs):
    try:
        return round(float(inputs["total_amount_incl_tax"]) * float(inputs["exchange_rate_to_usd"]), 2)
    except Exception:
        return None
