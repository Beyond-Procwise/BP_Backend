import re
from datetime import date, datetime, timedelta

from src.services.structural_extractor.derivation import rule

NET_RE = re.compile(r"Net\s*(\d+)", re.IGNORECASE)
WITHIN_RE = re.compile(r"within\s+(\d+)\s*days?", re.IGNORECASE)


def _parse_days(payment_terms_text: str) -> int | None:
    if not payment_terms_text:
        return None
    m = NET_RE.search(payment_terms_text)
    if m:
        return int(m.group(1))
    m = WITHIN_RE.search(payment_terms_text)
    if m:
        return int(m.group(1))
    return None


def _as_date(v):
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v).date()
        except ValueError:
            pass
    return None


@rule("due_date_from_terms", "due_date", ["invoice_date", "payment_terms"])
def _due_from_terms(inputs):
    inv_date = _as_date(inputs["invoice_date"])
    days = _parse_days(str(inputs["payment_terms"]))
    if inv_date is None or days is None:
        return None
    return inv_date + timedelta(days=days)


@rule("due_date_default", "due_date", ["invoice_date"])
def _due_default(inputs):
    inv_date = _as_date(inputs["invoice_date"])
    if inv_date is None:
        return None
    return inv_date + timedelta(days=90)
