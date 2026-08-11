"""Backend-executed web tools for AgentNick supplier research (keyless).

The local AgentNick model has no internet; these functions are exposed to it as
tools. AgentNick decides what to search/read; the backend performs the network
call and returns results. Both fail open (empty result on any error) so a network
problem never breaks the research loop.

WHY ``fetch_url`` IS GUARDED AND ``web_search`` IS NOT
-----------------------------------------------------
The URL handed to ``fetch_url`` is chosen by a language model
(``research.py``: ``fetch_url(str(args.get("url", "")))``), and the model's
context is seeded from supplier names that were themselves extracted from
customer documents. That is a chain from "text inside a PDF" to "host this
process connects to", which makes this the only server-side request forgery
surface in the codebase. ``web_search`` posts to one fixed provider and cannot
be pointed anywhere.

What the guard refuses, and why each one is here rather than theoretical:

  * Any scheme but http/https — ``file://`` reads the disk, ``gopher://``
    speaks to a Redis or memcached port.
  * Any port but 80/443 — a public hostname on :11434 is the local model
    daemon, on :5432 the database. A supplier's About page is on 80 or 443.
  * Any host that resolves to a non-global address — loopback, RFC1918,
    link-local (169.254.169.254 is the EC2 credential endpoint), unique-local,
    reserved. Checked AFTER resolution, because a public name with a private
    A record defeats any check on the literal string.
  * Redirects — ``requests`` follows them by default, and a followed redirect
    is a request this function never inspected. Following is turned off and
    each hop is re-checked here.

KNOWN LIMITATION, stated rather than papered over: this validates the addresses
a hostname resolves to at check time, then hands the URL to ``requests``, which
resolves again. A name that answers with a public address and then a private one
(DNS rebinding) is not stopped by this. Closing that needs the connection pinned
to the checked address, which means a custom transport adapter. The window is
narrow and the remaining exposure is a single GET whose body is capped and
returned to a model, not to a caller — but it is not zero, and it should not be
described as if it were.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urljoin, urlsplit

import requests

log = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (compatible; ProcWiseSupplierResearch/1.0)"
_FETCH_TIMEOUT = 10
_FETCH_CAP = 8000
_SEARCH_TIMEOUT = 12

_ALLOWED_SCHEMES = frozenset({"http", "https"})
# A supplier's website is served on 80 or 443. Anything else on a public
# hostname is a service, and reaching a service is not research.
_ALLOWED_PORTS = frozenset({80, 443})
_MAX_REDIRECTS = 5


def _resolve(host: str) -> list[str]:
    """Every address ``host`` resolves to, v4 and v6. Empty on failure.

    Separate from the check below so a test can supply a resolution without a
    DNS server, and so the failure mode is one place: no addresses means no
    proof the host is external, which the caller treats as a refusal.
    """
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except Exception:  # noqa: BLE001 - any resolver failure is "unknown"
        return []
    return [info[4][0] for info in infos]


def _is_global(addr: str) -> bool:
    """True only for an address that is routable on the public internet.

    ``ipaddress.is_global`` already covers loopback, private, link-local,
    multicast and reserved ranges, and it is maintained with the IANA registry
    rather than by a hand-written list of prefixes that would drift. IPv4-mapped
    IPv6 (``::ffff:127.0.0.1``) is unwrapped first, because the mapped form is
    global by the v6 rules while the address it names is not.
    """
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped
    return bool(ip.is_global)


def _check_url(url: str) -> tuple[bool, str]:
    """(allowed, reason). Reason is logged on refusal, never returned to the model."""
    if not url or not isinstance(url, str):
        return False, "empty url"

    try:
        parts = urlsplit(url)
    except Exception:  # noqa: BLE001 - a URL too malformed to split
        return False, "unparseable url"

    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        return False, f"scheme {parts.scheme!r} is not http/https"

    # Credentials in the authority are never needed to read a public page, and
    # they are how a URL is made to look like one host while addressing another.
    if parts.username or parts.password:
        return False, "credentials in the authority"

    try:
        host = parts.hostname
        port = parts.port
    except ValueError:
        return False, "invalid port"
    if not host:
        return False, "no host"

    if port is not None and port not in _ALLOWED_PORTS:
        return False, f"port {port} is not 80/443"

    # A literal address needs no resolution; a name does. Both end up in the
    # same check, so an attacker gains nothing by choosing one form.
    try:
        ipaddress.ip_address(host)
        addresses = [host]
    except ValueError:
        addresses = _resolve(host)

    if not addresses:
        return False, f"{host} did not resolve"

    # EVERY address must be global. A name with one public and one private
    # answer is a name that reaches the private one on the next connection.
    bad = [a for a in addresses if not _is_global(a)]
    if bad:
        return False, f"{host} resolves to non-global address(es) {bad}"

    return True, ""


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Return [{title, url, snippet}] for a query (DuckDuckGo via ddgs)."""
    if not query or not query.strip():
        return []
    try:
        from ddgs import DDGS
        out: list[dict] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query.strip(), max_results=max_results):
                url = r.get("href") or r.get("url") or ""
                if url:
                    out.append({
                        "title": (r.get("title") or "")[:200],
                        "url": url,
                        "snippet": (r.get("body") or r.get("snippet") or "")[:400],
                    })
        return out
    except Exception as exc:  # noqa: BLE001 - fail open
        log.warning("web_search failed for %r: %s", query, exc)
        return []


def _html_to_text(resp) -> str:
    ctype = resp.headers.get("Content-Type", "")
    if "html" not in ctype and "text" not in ctype:
        return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
    except Exception:  # noqa: BLE001 - fall back to raw text
        text = " ".join(resp.text.split())
    return text[:_FETCH_CAP]


def fetch_url(url: str) -> str:
    """Fetch a page and return cleaned visible text (capped). Empty on refusal or error.

    Every hop — the URL supplied and each redirect target — passes ``_check_url``
    before a request is issued for it. Redirect following is taken off
    ``requests`` deliberately: a redirect it follows on our behalf is a request
    this function never saw, and the guard would be checking only the first URL
    of a chain the model does not control the end of.

    A refusal returns "" and logs the reason. The reason is not returned: the
    caller feeds this straight back to a model as tool output, and telling it
    which addresses are refused is telling it what to probe for.
    """
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        ok, reason = _check_url(current)
        if not ok:
            log.warning("fetch_url refused %r: %s", current, reason)
            return ""
        try:
            resp = requests.get(
                current,
                headers={"User-Agent": _UA},
                timeout=_FETCH_TIMEOUT,
                allow_redirects=False,
            )
        except Exception as exc:  # noqa: BLE001 - fail open
            log.warning("fetch_url failed for %r: %s", current, exc)
            return ""

        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("Location")
            if not location:
                return ""
            # Relative Locations are legal and common; resolve against the hop
            # we actually made, then re-check the result like any other URL.
            current = urljoin(current, location)
            continue

        try:
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch_url failed for %r: %s", current, exc)
            return ""
        return _html_to_text(resp)

    log.warning("fetch_url refused %r: more than %d redirects", url, _MAX_REDIRECTS)
    return ""
