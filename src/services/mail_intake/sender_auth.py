"""Record SPF, DKIM, DMARC and alignment. Do not recompute them.

SPF validates the connecting IP address. Once a message has been forwarded
that address belongs to the forwarder, so recomputing SPF downstream yields a
confident number about the wrong thing. DKIM survives forwarding better but
breaks on any mailing list that rewrites the body.

So this module reads the verdict the receiving MTA wrote, and records **who
asserted it** alongside. On the relay path that asserter is our own boundary
MTA and the verdict is trustworthy. On the Graph and IMAP paths it is the
customer's mail system, and an Authentication-Results header can be forged by
anything upstream of the machine that wrote the topmost one — hence
`trusted_asserter`.

A missing header is `unknown`, never `pass`. Spec §4.1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from email.utils import parseaddr
from typing import Optional

UNKNOWN = "unknown"

_METHOD = re.compile(r"\b(spf|dkim|dmarc)\s*=\s*([a-z]+)", re.IGNORECASE)


@dataclass(frozen=True)
class AuthResult:
    spf: str
    dkim: str
    dmarc: str
    envelope_from: Optional[str]
    header_from: Optional[str]
    alignment: str
    asserted_by: Optional[str]


def _domain(address: Optional[str]) -> Optional[str]:
    if not address or "@" not in address:
        return None
    return address.rsplit("@", 1)[1].strip().lower() or None


def _first(bundle, name: str) -> Optional[str]:
    values = bundle.headers.get(name.lower()) or []
    return values[0] if values else None


def evaluate(bundle, *, trusted_asserter: Optional[str] = None) -> AuthResult:
    """Read the authentication verdicts carried by the message."""
    header = _first(bundle, "authentication-results")

    asserted_by = None
    verdicts = {"spf": UNKNOWN, "dkim": UNKNOWN, "dmarc": UNKNOWN}

    if header:
        asserted_by = header.split(";", 1)[0].strip() or None
        for method, value in _METHOD.findall(header):
            verdicts[method.lower()] = value.lower()
        if trusted_asserter and asserted_by != trusted_asserter:
            # Hearsay: recorded, but not treated as a verdict.
            verdicts = {"spf": UNKNOWN, "dkim": UNKNOWN, "dmarc": UNKNOWN}

    header_from = parseaddr(_first(bundle, "from") or "")[1] or None
    envelope_raw = _first(bundle, "return-path") or _first(bundle, "sender")
    envelope_from = parseaddr(envelope_raw or "")[1] or None

    header_domain = _domain(header_from)
    envelope_domain = _domain(envelope_from)
    if header_domain and envelope_domain:
        alignment = "aligned" if header_domain == envelope_domain else "misaligned"
    else:
        alignment = UNKNOWN

    return AuthResult(
        spf=verdicts["spf"],
        dkim=verdicts["dkim"],
        dmarc=verdicts["dmarc"],
        envelope_from=envelope_from,
        header_from=header_from,
        alignment=alignment,
        asserted_by=asserted_by,
    )
