"""Read the price a supplier states in their own reply, or nothing.

A reply reaches the negotiation with whatever figures its stored row carries.
On the mailbox route nothing read the email, so the row carried none and the
negotiation ran with "supplier offer missing" against a reply that stated a
price in plain words.

This reader is deliberately narrow. It reads only what the supplier wrote in
this message -- the quoted copy of our own email underneath carries our figures,
not theirs -- and it returns a price only when those words state exactly one
money amount. No amount, or several different ones (a unit price beside a
total), is left for a person: a wrong price is worse than a missing one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Set, Tuple

#: How the figure was read; stored with it so it is never mistaken for a confirmed one.
METHOD = "regex_single_currency_amount"

_SYMBOLS = {"£": "GBP", "$": "USD", "€": "EUR"}
_CODES = ("GBP", "USD", "EUR")

_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?"
_AMOUNT = re.compile(
    rf"(?P<sym>[£$€])\s?(?P<a>{_NUMBER})(?![\d.,]*\d)"
    rf"|(?P<code1>{'|'.join(_CODES)})\s?(?P<b>{_NUMBER})(?![\d.,]*\d)"
    rf"|(?<![\d.,])(?P<c>{_NUMBER})\s?(?P<code2>{'|'.join(_CODES)})\b",
    re.IGNORECASE,
)

_REPLY_HEADER = re.compile(r"^On\b.{0,200}\bwrote:\s*$", re.IGNORECASE | re.DOTALL)
_ORIGINAL_MESSAGE = re.compile(r"^-{2,}\s*(Original Message|Forwarded message)\s*-{2,}", re.IGNORECASE)
_HEADER_LINE = re.compile(r"^(From|Sent|Date|To|Subject):\s", re.IGNORECASE)


@dataclass(frozen=True)
class ReplyOffer:
    price: Decimal
    currency: str
    method: str = METHOD


def own_words(text: Optional[str]) -> str:
    """The part of a reply its sender wrote: nothing quoted, nothing below the quote header."""

    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept: List[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(">"):
            continue
        if _ORIGINAL_MESSAGE.match(stripped):
            break
        # "On <date> <someone> wrote:" -- mail clients wrap it over two lines.
        following = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if stripped.lower().startswith("on ") and _REPLY_HEADER.match(f"{stripped} {following}"):
            break
        # Outlook's quoted header block: "From: ..." directly followed by "Sent:"/"Date:".
        if stripped.lower().startswith("from:") and _HEADER_LINE.match(following) and not following.lower().startswith("from:"):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def _amounts(text: str) -> Set[Tuple[Decimal, str]]:
    found: Set[Tuple[Decimal, str]] = set()
    for match in _AMOUNT.finditer(text):
        number = match.group("a") or match.group("b") or match.group("c")
        symbol = match.group("sym")
        code = match.group("code1") or match.group("code2")
        currency = _SYMBOLS[symbol] if symbol else code.upper()
        try:
            value = Decimal(number.replace(",", ""))
        except InvalidOperation:
            continue
        if value > 0:
            found.add((value, currency))
    return found


def read_offer(text: Optional[str]) -> Optional[ReplyOffer]:
    """The one money amount the supplier's own words state, or ``None``."""

    amounts = _amounts(own_words(text))
    if len(amounts) != 1:
        return None
    value, currency = next(iter(amounts))
    return ReplyOffer(price=value, currency=currency)
