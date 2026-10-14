import logging
import time

from src.services import egress

from src.services.structural_extractor.derivation import rule

log = logging.getLogger(__name__)
_CACHE: dict[str, tuple[float, float]] = {}  # ccy -> (rate, timestamp)
_CACHE_TTL = 3600  # 1h


def _fetch_json(ccy: str) -> dict | None:
    try:
        r = egress.get(
            f"https://open.er-api.com/v6/latest/{ccy}",
            purpose=egress.Purpose.FX_RATES,
            timeout=5,
        )
        if r is not None and r.status_code == 200:
            return r.json()
    except Exception:
        log.debug("FX fetch failed for %s", ccy, exc_info=True)
    return None


def _fetch_rate_live(ccy: str) -> float | None:
    data = _fetch_json(ccy)
    if data:
        rate = data.get("rates", {}).get("USD")
        if rate is not None:
            return float(rate)
    return None


def fetch_usd_quoted_rates() -> dict[str, float] | None:
    """Fetch the full USD-base rates table (currency -> units per 1 USD).

    Used by GET /fx/rates (dashboard currency selector). Shares the same
    live source/endpoint as ``_fetch_rate_live`` above (open.er-api.com) via
    ``_fetch_json`` — this just reads the whole ``rates`` dict from the
    USD-base response instead of picking out a single currency.
    """
    data = _fetch_json("USD")
    if not data:
        return None
    rates = data.get("rates")
    if not isinstance(rates, dict) or not rates:
        return None
    try:
        return {str(k): float(v) for k, v in rates.items()}
    except Exception:
        log.debug("FX rates parse failed", exc_info=True)
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
