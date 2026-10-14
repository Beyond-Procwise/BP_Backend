"""What a contract is worth, and in what currency, read from its own words.

WHY THIS EXISTS. `total_contract_value` and `currency` are pattern-less -- nine and
four `canonical_labels` between them, no `patterns` -- and were NULL on all 7 live
contract rows while **every one of those documents states a value**:

    The total estimated value of this Framework Agreement over its term is GBP 750,000.
    The total cost of the Services will be £25,000.
    The total charges for this Order Form are GBP 48,000, invoiced monthly in arrears
        at GBP 4,000 per month.

The fourth instance of one structural hole (parties, signatories, term, value) and
the one that costs money: a contract whose value is NULL contributes nothing to any
spend or savings figure.

THE TRAP HERE IS THE SECOND NUMBER. That third sentence states a total AND a rate;
the real Marketing Agreement states two instalments beside its total, and a
spending threshold ("if an expense is over £500") two sentences later. Taking the
wrong one is worse than taking none -- it is a plausible figure that silently
misreports the contract. So the value is the FIRST money token after a phrase that
actually says "total", and the search stops at the end of that sentence.

A CURRENCY MARKER IS REQUIRED. "The total charges are 48,000" yields nothing: a
bare number after "total" could be a headcount, and a money figure whose currency
nobody stated is a thing this product has already been burned by.

Amount and currency are parsed by `extraction_v2.parsers`, which this module
checked for the two-venv trap before relying on them -- unlike the date reader's
first version, which called `dateparser` and so produced nothing under pytest while
working in production.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..types import Candidate, Span
from .contract_parties import CONFIDENCE, _squeeze

VALUE_FIELD = "total_contract_value"
CURRENCY_FIELD = "currency"

#: ISO codes and symbols seen in this corpus. A code must stand as its own word so
#: "CHF" inside a part number is not a currency.
_CODES = ("GBP", "USD", "EUR", "JPY", "INR", "CHF", "AUD", "CAD", "SEK", "NOK",
          "DKK", "PLN", "ZAR", "SGD", "HKD", "NZD")
_SYMBOLS = "£$€¥₹"

#: A money token: a currency marker on either side of a number. The marker is
#: mandatory -- see the docstring.
_MONEY = re.compile(
    r"(?P<pre>[" + _SYMBOLS + r"]|\b(?:" + "|".join(_CODES) + r")\b)\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
    r"|(?P<num2>\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)\s*"
    r"(?P<post>[" + _SYMBOLS + r"]|\b(?:" + "|".join(_CODES) + r")\b)"
)

#: A labelled total: the `canonical_labels` contract.yaml already declares.
_VALUE_LABEL = re.compile(
    r"(?:^|[\s\n])(?i:maximum\s+contract\s+value|total\s+contract\s+value|"
    r"contract\s+value|agreement\s+value|total\s+value|contract\s+amount|"
    r"total\s+amount|consideration|not\s+to\s+exceed)\s*[:–]\s*"
)

#: Prose that introduces a total. "total" is mandatory in every alternative --
#: that word is the only thing separating the contract's worth from a rate, an
#: instalment or a spending threshold.
_VALUE_PROSE = re.compile(
    r"(?i)\btotal\s+(?:estimated\s+|aggregate\s+|maximum\s+)?"
    r"(?:contract\s+|agreement\s+|order\s+form\s+)?"
    r"(?:value|cost|costs|charges|charge|price|fees|fee|consideration|amount)\b"
)


@dataclass(frozen=True)
class ContractValue:
    """The contract's worth. Both fields, or neither."""

    value: Optional[str] = None
    currency: Optional[str] = None
    text: Optional[str] = None


def _parse_money(token: re.Match) -> Optional[tuple[str, str, str]]:
    """``(amount, currency, literal)`` for a money token, or None.

    Goes through the pipeline's own parsers, so this module can never emit an
    amount or a code the pipeline would reject.
    """
    from src.services.extraction_v2.parsers.amounts import parse_amount
    from src.services.extraction_v2.parsers.currency import parse_currency

    num = token.group("num") or token.group("num2")
    marker = token.group("pre") or token.group("post")
    if not num or not marker:
        return None
    amount = parse_amount(num)
    code = parse_currency(marker)
    if amount is None or code is None:
        return None
    return (str(amount), str(code), token.group(0))


def _first_money_in_sentence(text: str, pos: int, window: int = 110) -> Optional[tuple[str, str, str]]:
    """The first money token after `pos`, within this sentence.

    Bounded by the sentence end as well as by `window`: a window that crosses a
    full stop reads the next clause's number, and "The total value is stated in
    Schedule 1. The deposit is GBP 5,000." must yield nothing.
    """
    span = text[pos:pos + window]
    end = re.search(r"\.(?:\s|\n|$)", span)
    if end:
        span = span[:end.end()]
    m = _MONEY.search(span)
    return _parse_money(m) if m else None


def read_value(full_text: str) -> ContractValue:
    """The contract's total value and currency, or neither."""
    if not full_text:
        return ContractValue()

    for pattern in (_VALUE_LABEL, _VALUE_PROSE):
        found: list[tuple[str, str, str]] = []
        for m in pattern.finditer(full_text):
            got = _first_money_in_sentence(full_text, m.end())
            if got:
                found.append(got)
        if not found:
            continue
        # One contract, one value. Several that AGREE are one fact stated twice;
        # several that disagree are a misread, and a plausible wrong number is
        # worse than none.
        distinct = {(amount, code) for amount, code, _lit in found}
        if len(distinct) > 1:
            return ContractValue()
        amount, code, literal = found[0]
        return ContractValue(value=amount, currency=code, text=literal)

    return ContractValue()


def value_candidates(full_text: str) -> list[Candidate]:
    """`total_contract_value` and `currency`, from the same money token.

    Both or neither, on purpose: an amount without its currency is the shape of a
    figure that later gets read in the wrong one.
    """
    v = read_value(full_text)
    if not v.value or not v.currency:
        return []
    span = Span(page=0, bbox=(0.0, 0.0, 0.0, 0.0), text=v.text or v.value)
    return [
        Candidate(field=VALUE_FIELD, value=v.value, span=span, source="regex",
                  pattern_name="contract_total_clause", confidence=CONFIDENCE),
        Candidate(field=CURRENCY_FIELD, value=v.currency, span=span, source="regex",
                  pattern_name="contract_total_clause", confidence=CONFIDENCE),
    ]


__all__ = ["ContractValue", "read_value", "value_candidates",
           "VALUE_FIELD", "CURRENCY_FIELD"]


# ---------------------------------------------------------------------------
# Correcting rows stored before anything read a contract's value.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValueCorrection:
    value: Optional[str]
    currency: Optional[str]
    changed: bool
    reason: str


def decide_value_correction(
    *,
    full_text: str,
    stored_value: Optional[str],
    stored_currency: Optional[str],
    provenance_source: Optional[str],
) -> ValueCorrection:
    """Should this row's value change, and to what?

    Three rules, as for the term. The sweep never produced a contract value
    either -- the field is pattern-less, which is why it was NULL -- so there is
    no wrong value of its to clear, only an absent one to fill. Where this reader
    finds nothing a stored value STAYS: the context layer may have grounded a
    shape these patterns do not cover, and absence of a read is not evidence of a
    wrong number.

    1. a human-confirmed value is never touched;
    2. a row with no stored text cannot be re-read;
    3. what the document states wins -- including over a stored figure, because a
       document's own total outranks whatever an earlier read picked up.
    """
    from .contract_parties import HUMAN_SOURCE

    if (provenance_source or "").lower() == HUMAN_SOURCE:
        return ValueCorrection(stored_value, stored_currency, False,
                               "a human confirmed this value; nothing overrules that")
    if not (full_text or "").strip():
        return ValueCorrection(stored_value, stored_currency, False,
                               "no stored text to re-read; left as found")

    v = read_value(full_text)
    if not v.value:
        return ValueCorrection(stored_value, stored_currency, False,
                               "the document states no total this reader recognises; "
                               "left as found rather than cleared")
    changed = (v.value != stored_value) or (v.currency != stored_currency)
    return ValueCorrection(
        v.value, v.currency, changed,
        "read from the document's own total"
        + ("" if changed else " and already stored correctly"),
    )


__all__ += ["ValueCorrection", "decide_value_correction"]
