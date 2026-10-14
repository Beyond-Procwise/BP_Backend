"""The volume a line is for, read from its own description.

A services quote bills "Service Desk (24x7, 2,400 users) — annual" as quantity 1, unit year,
£540,000. The billing quantity is right, but what is being bought — 2,400 users — stays in the
words, so a rival's "3,000 users" line at £584,000 reads as dearer when per user it is cheaper.
This lifts that volume into two fields of its own (volume, volume_unit). The billing quantity
and unit are never touched: they are what the document printed.

Deliberately narrow, because a wrong volume is worse than none:
  - only countable units from a fixed list (users, seats, licences, devices, ...), so a product
    name like "Compact 24–27 Unit" or a size like "A4/A3" is never read as a volume;
  - the number must stand on its own ("2,400 users", "240 seats"), not be part of a range
    ("13–14 users") or a code;
  - a description naming two different volumes is ambiguous and yields none.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional

# Countable things a service or licence line is sized by -> the one name the page shows.
VOLUME_UNITS = {
    "user": "users", "users": "users",
    "seat": "seats", "seats": "seats",
    "licence": "licences", "licences": "licences", "license": "licences", "licenses": "licences",
    "device": "devices", "devices": "devices",
    "endpoint": "endpoints", "endpoints": "endpoints",
    "site": "sites", "sites": "sites",
    "location": "locations", "locations": "locations",
    "mailbox": "mailboxes", "mailboxes": "mailboxes",
    "employee": "employees", "employees": "employees",
    "server": "servers", "servers": "servers",
    "desk": "desks", "desks": "desks",
    "agent": "agents", "agents": "agents",
}

_NUM = r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?"
_VOLUME_RE = re.compile(
    # Not preceded by a digit, a range dash or a separator that would make it part of
    # something bigger ("13–14 users", "v2.400 users", "ITM-240 seats").
    r"(?<![\d.,–\-/:#])(" + _NUM + r")\s*(" + "|".join(sorted(VOLUME_UNITS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def volume_from_description(description: Optional[str]) -> tuple[Optional[Decimal], Optional[str]]:
    """(volume, unit) named in a line's description, or (None, None)."""
    if not description:
        return None, None
    found = set()
    for m in _VOLUME_RE.finditer(str(description)):
        try:
            n = Decimal(m.group(1).replace(",", ""))
        except InvalidOperation:
            continue
        if n <= 0:
            continue
        found.add((n, VOLUME_UNITS[m.group(2).lower()]))
    if len(found) != 1:
        return None, None
    return found.pop()


def add_line_volumes(line_items: list[dict]) -> list[dict]:
    """Each line with volume / volume_unit set from its description where it names one.
    A line that already carries a volume keeps it."""
    out = []
    for li in line_items:
        row = dict(li)
        if row.get("volume") is None:
            v, u = volume_from_description(row.get("item_description"))
            if v is not None:
                row["volume"], row["volume_unit"] = v, u
        out.append(row)
    return out
