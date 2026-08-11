"""The single outbound HTTP transport.

Every other module is forbidden from importing an HTTP client directly; the
boundary is enforced by ``tests/test_egress_import_boundary.py``, which freezes
the modules that already do and fails on any new one. This is the module they
are all meant to come through.

WHAT THIS IS FOR

The audit found twenty-one of twenty-three outbound paths reaching the network
with no policy evaluation of any kind, and no way to answer "what left the
boundary on 12 March" for any of them. Fixing that needs one place where a
request can be described before it is made, so the description can be checked
and recorded. Not a place where a request is checked *afterwards* — by then it
has gone.

WHAT A CALLER MUST DECLARE

  purpose      why this call is being made, from ``Purpose``. Not free text: a
               purpose that can be spelled two ways cannot be aggregated, and a
               policy cannot be written against it.
  destination  the host being addressed, recorded whether or not the call
               succeeds. A failed call still disclosed a hostname and, usually,
               the shape of a query.

Both are required positionally. A caller that has not thought about why it is
making a request has not finished writing it.

WHAT THIS DOES NOT DO YET, STATED PLAINLY

  * It does not consult a policy engine. There is no policy model keyed on
    (destination, purpose, classification) to consult — building one is the next
    piece of work, and ``_evaluate`` is the seam it will plug into. Until then
    this records rather than refuses, and the docstring says so rather than
    letting the presence of a "gate" imply a decision is being made.
  * It does not classify payloads. There is no classification registry.
  * It does not write ``bp_egress_event``. Recording currently goes to the log,
    because a table that exists but is written from one of twenty-three paths
    would make coverage look better than it is. The record shape here is the one
    that table will take.

So: this is the seam, with the SSRF guard and the audit line real, and the
policy evaluation honestly absent. It is worth having before the policy model
exists precisely because it is what makes adding one a single change rather
than twenty-three.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from urllib.parse import urljoin, urlsplit

import requests

logger = logging.getLogger(__name__)


class Purpose(str, Enum):
    """Why an outbound call is being made.

    A closed vocabulary on purpose. Free text cannot be aggregated ("supplier
    research" / "supplier_research" / "research"), cannot be counted per
    thousand rows, and cannot have a policy written against it.
    """

    SUPPLIER_RESEARCH = "supplier_research"      # public web, supplier profile
    FX_RATES = "fx_rates"                        # currency reference data
    MODEL_INFERENCE = "model_inference"          # prompt to a model host
    VECTOR_INDEX = "vector_index"                # embedding read/write
    MAILBOX = "mailbox"                          # IMAP/Graph mailbox access
    NOTIFICATION = "notification"                # outbound mail
    HEALTHCHECK = "healthcheck"                  # liveness, carries no payload


@dataclass(frozen=True)
class EgressDenied(Exception):
    """Raised when a request is refused before anything leaves the host."""

    reason: str
    destination: str
    purpose: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.purpose} -> {self.destination} refused: {self.reason}"


# ---------------------------------------------------------------------------
# Destination safety
# ---------------------------------------------------------------------------

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_MAX_REDIRECTS = 5


def _resolve(host: str) -> list[str]:
    """Every address ``host`` resolves to. Empty on failure.

    Separate so a test can supply a resolution without a DNS server, and so the
    failure mode is one place: no addresses means no proof the host is external.
    """
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except Exception:  # noqa: BLE001
        return []
    return [info[4][0] for info in infos]


def _is_global(addr: str) -> bool:
    """True only for an address routable on the public internet.

    ``ipaddress.is_global`` tracks the IANA registry, which a hand-written list
    of prefixes would not. IPv4-mapped IPv6 is unwrapped first: the mapped form
    is global by the v6 rules while the address it names is not.
    """
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped
    return bool(ip.is_global)


def check_destination(url: str, *, allowed_ports: frozenset[int] | None = None,
                      require_global: bool = True) -> tuple[bool, str]:
    """(allowed, reason) for one URL.

    ``require_global`` is False for the hosts that are deliberately internal —
    the model daemon on localhost, a Qdrant instance inside the VPC. Those still
    pass through this module so the call is described and recorded; what varies
    is whether reaching a private address is the point or the attack.
    """
    if not url or not isinstance(url, str):
        return False, "empty url"
    try:
        parts = urlsplit(url)
    except Exception:  # noqa: BLE001
        return False, "unparseable url"

    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        return False, f"scheme {parts.scheme!r} is not http/https"
    if parts.username or parts.password:
        return False, "credentials in the authority"

    try:
        host, port = parts.hostname, parts.port
    except ValueError:
        return False, "invalid port"
    if not host:
        return False, "no host"
    if allowed_ports is not None and port is not None and port not in allowed_ports:
        return False, f"port {port} is not permitted for this purpose"

    if not require_global:
        return True, ""

    try:
        ipaddress.ip_address(host)
        addresses = [host]
    except ValueError:
        addresses = _resolve(host)
    if not addresses:
        return False, f"{host} did not resolve"
    bad = [a for a in addresses if not _is_global(a)]
    if bad:
        return False, f"{host} resolves to non-global address(es) {bad}"
    return True, ""


# ---------------------------------------------------------------------------
# The seam a policy engine will plug into
# ---------------------------------------------------------------------------

def _evaluate(purpose: Purpose, destination: str) -> tuple[bool, str]:
    """Whether this call is permitted.

    Deliberately trivial today, and named so the absence is visible in a stack
    trace rather than hidden behind an approving-sounding wrapper. There is no
    policy model keyed on (destination, purpose, classification) to consult; when
    there is, this is the one function that changes and every caller inherits it.

    Returning True here is not a decision that the call is safe. It is the
    absence of a decision, which is what the audit found and what this module
    exists to make fixable in one place.
    """
    return True, "no destination policy is configured"


def _record(*, purpose: Purpose, destination: str, method: str,
            outcome: str, detail: str = "") -> None:
    """One line per outbound call, whatever the outcome.

    A refused or failed call still disclosed a hostname and usually the shape of
    a query, so it is recorded on the same footing as a successful one. This is
    the shape ``bp_egress_event`` will take: purpose, destination, decision.
    """
    logger.info(
        "egress purpose=%s destination=%s method=%s outcome=%s%s",
        purpose.value, destination, method, outcome,
        f" detail={detail}" if detail else "",
    )


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

def request(
    method: str,
    url: str,
    *,
    purpose: Purpose,
    allowed_ports: frozenset[int] | None = None,
    require_global: bool = True,
    follow_redirects: bool = False,
    timeout: int = 15,
    **kwargs: Any,
) -> Optional[requests.Response]:
    """Make one outbound HTTP call, described and recorded.

    Returns the response, or None if it was refused or failed — callers in this
    codebase uniformly treat an outbound failure as "no data", and raising would
    turn a network problem into an outage in paths that currently degrade.
    ``EgressDenied`` is raised only when a caller asks for it via
    :func:`request_or_raise`.

    Redirects are NOT delegated to ``requests`` by default. A redirect it follows
    is a request this function never described, checked or recorded, so each hop
    is made here and re-checked.
    """
    destination = urlsplit(url).hostname or "?"

    permitted, why = _evaluate(purpose, destination)
    if not permitted:
        _record(purpose=purpose, destination=destination, method=method,
                outcome="denied", detail=why)
        return None

    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        ok, reason = check_destination(
            current, allowed_ports=allowed_ports, require_global=require_global
        )
        if not ok:
            _record(purpose=purpose, destination=urlsplit(current).hostname or "?",
                    method=method, outcome="refused", detail=reason)
            return None

        try:
            response = requests.request(
                method, current, timeout=timeout, allow_redirects=False, **kwargs
            )
        except Exception as exc:  # noqa: BLE001
            _record(purpose=purpose, destination=destination, method=method,
                    outcome="error", detail=type(exc).__name__)
            return None

        if follow_redirects and response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")
            if not location:
                break
            current = urljoin(current, location)
            continue

        _record(purpose=purpose, destination=destination, method=method,
                outcome=f"http_{response.status_code}")
        return response

    _record(purpose=purpose, destination=destination, method=method,
            outcome="refused", detail=f"more than {_MAX_REDIRECTS} redirects")
    return None


def get(url: str, *, purpose: Purpose, **kwargs: Any) -> Optional[requests.Response]:
    return request("GET", url, purpose=purpose, **kwargs)


def post(url: str, *, purpose: Purpose, **kwargs: Any) -> Optional[requests.Response]:
    return request("POST", url, purpose=purpose, **kwargs)


def request_or_raise(method: str, url: str, *, purpose: Purpose,
                     **kwargs: Any) -> requests.Response:
    """As :func:`request`, but raises ``EgressDenied`` instead of returning None.

    For callers where a silent None would be read as "no data found" rather than
    "we were not allowed to look" — a distinction that matters when the answer
    is shown to a buyer.
    """
    response = request(method, url, purpose=purpose, **kwargs)
    if response is None:
        raise EgressDenied(
            reason="refused, denied or failed — see the egress log line",
            destination=urlsplit(url).hostname or "?",
            purpose=purpose.value,
        )
    return response
