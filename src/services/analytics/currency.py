"""The currency an analytic answer reports in — chosen on screen, not here.

Display currency is a control on Procurement Home's top bar, backed by a shared
controller in the UI repo (``src/lib/currency/displayCurrency.js``) that the
SpendIQ dashboard reads through as well. One selection, one currency, every
surface. An analytic answer that picked its own base would be the single
surface disagreeing with the rest of the product, so the selection travels with
the request and lands here.

This is the server-side half of that controller, and it enforces the same three
rules, each of which was a real way to report a wrong number:

  1. **Convert each currency once, from its own native amount.** Converting GBP
     to GBP is then an exact identity. Going via a stored USD scalar instead
     silently lost ~5.5% on every sterling figure.
  2. **A currency with no rate is excluded and counted, never assumed 1:1.**
     Nothing here invents a rate, and there is no hardcoded fallback. When
     nothing can be converted the total is ``None`` and the caller must drop the
     figure — never render it as zero, which would say this organisation spent
     nothing.
  3. **A manual override announces itself**, so a what-if can never be mistaken
     for a live-rate figure.

Rates are USD-quoted — units of that currency per 1 USD — exactly as
``proc.bp_fx_rates`` stores them, ``GET /fx/rates`` serves them
(``src/api/routers/fx.py``) and the client reads them. Both halves therefore
convert at the same numbers from the same batch.

Nothing here writes a rate, and no extracted document amount is ever modified:
conversion is a presentation of stored figures, and the native amount stays the
record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping, Optional, Sequence

# Report as billed, without conversion — the selector's third option.
NATIVE = "native"

# What the app reports in until the user says otherwise. Mirrors
# DEFAULT_CURRENCY in the client controller.
DEFAULT_CURRENCY = "GBP"


def _as_rate(value: Any) -> Optional[Decimal]:
    """A rate is only usable if it is a positive, finite number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        rate = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    if not rate.is_finite() or rate <= 0:
        return None
    return rate


def _as_amount(value: Any) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return amount if amount.is_finite() else None


@dataclass(frozen=True)
class ConversionResult:
    """A total, and an honest account of what it left out.

    ``value`` is ``None`` when nothing could be converted. ``excluded`` counts
    the rows dropped for want of a rate, and ``excluded_currencies`` names them,
    so the answer can say which currencies are missing from the figure instead
    of quietly understating it.
    """

    value: Optional[Decimal]
    excluded: int
    excluded_currencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class DisplayCurrency:
    """The currency selection in force for one answer.

    ``target`` is an ISO code or :data:`NATIVE`. ``rates`` is the USD-quoted
    batch. ``manual`` holds the user's display-only overrides, which win over
    the live rate for the currencies they name.
    """

    target: str = DEFAULT_CURRENCY
    rates: Mapping[str, Any] = field(default_factory=dict)
    manual: Mapping[str, Any] = field(default_factory=dict)
    fetched_at: Optional[datetime] = None
    stale: bool = False

    # -- state -------------------------------------------------------------

    @property
    def is_native(self) -> bool:
        return not self.target or self.target == NATIVE

    @property
    def rates_unavailable(self) -> bool:
        return not self.rates

    def is_manual(self, currency: Optional[str]) -> bool:
        return bool(currency) and str(currency).upper() in {
            str(k).upper() for k in self.manual
        }

    # -- rates -------------------------------------------------------------

    def effective_rate(self, currency: Optional[str]) -> Optional[Decimal]:
        """The rate actually in force: a manual override if set, else the live one."""
        if not currency:
            return None
        code = str(currency).upper()
        for key, value in self.manual.items():
            if str(key).upper() == code:
                rate = _as_rate(value)
                if rate is not None:
                    return rate
        # One USD per USD is the definition of a USD-quoted table, not a rate we
        # are guessing, and it must hold with the table absent — otherwise a
        # manual GBP rate could not convert a USD amount, which is exactly when
        # a manual rate is wanted.
        if code == "USD":
            return Decimal(1)
        for key, value in self.rates.items():
            if str(key).upper() == code:
                return _as_rate(value)
        return None

    # -- conversion --------------------------------------------------------

    def convert_amount(self, amount: Any, currency: Optional[str]) -> Optional[Decimal]:
        """One amount in the display currency, or None if it cannot be stated there."""
        if self.is_native:
            return None
        parsed = _as_amount(amount)
        if parsed is None:
            return None
        source = self.effective_rate(currency)
        target = self.effective_rate(self.target)
        if source is None or target is None:
            return None
        # Identity when the rates are the same, which is the whole reason this
        # is done per currency rather than through a common scalar.
        if source == target:
            return parsed
        return parsed / source * target

    def total(self, rows: Iterable[tuple[Any, Optional[str]]]) -> ConversionResult:
        """Sum ``(amount, currency)`` pairs into one figure in the display currency."""
        pairs: Sequence[tuple[Any, Optional[str]]] = list(rows)
        if self.is_native:
            return ConversionResult(value=None, excluded=len(pairs))

        total = Decimal(0)
        converted = 0
        excluded = 0
        missing: list[str] = []
        for amount, currency in pairs:
            value = self.convert_amount(amount, currency)
            if value is None:
                excluded += 1
                code = str(currency).upper() if currency else "unknown"
                if code not in missing:
                    missing.append(code)
                continue
            total += value
            converted += 1
        return ConversionResult(
            value=total if converted else None,
            excluded=excluded,
            excluded_currencies=tuple(missing),
        )

    # -- disclosure --------------------------------------------------------

    def rate_note(self) -> str:
        """Where the rate in force came from. Shown next to every converted figure.

        The year is carried, where the client's chip omits it: an answer can be
        copied or exported out of the session that produced it, and a bare
        "07 Sep" would then date a figure to no particular year.
        """
        if self.is_manual(self.target):
            rate = self.effective_rate(self.target)
            return f"manual rate · 1 USD = {rate} {self.target}"
        if self.rates_unavailable or self.fetched_at is None:
            return "rates unavailable"
        when = self.fetched_at.strftime("%d %b %Y, %H:%M")
        return f"rates as of {when} ({'stale' if self.stale else 'live'})"
