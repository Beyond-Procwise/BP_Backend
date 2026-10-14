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

The destination rules, the redirect handling and the audit line live in
``services.egress`` — scheme, port, post-resolution address check, per-hop
re-validation. They were written here first, when this was the only
model-driven request in the codebase. Keeping a second copy would mean two
places to fix when the rule changes, and the copy that did not get fixed would
be the one still reachable, so this now calls through the shared transport and
supplies only what is specific to it: the purpose (supplier research) and the
ports a supplier's website is actually served on.

KNOWN LIMITATION, stated rather than papered over: egress validates the
addresses a hostname resolves to at check time, then hands the URL to
``requests``, which resolves again. A name that answers with a public address
and then a private one (DNS rebinding) is not stopped by this. Closing it needs
the connection pinned to the checked address, which means a custom transport
adapter. The window is narrow and the remaining exposure is a single GET whose
body is capped and returned to a model, not to a caller — but it is not zero,
and it should not be described as if it were.
"""

from __future__ import annotations

import logging

from src.services.egress import Purpose, get as egress_get

log = logging.getLogger(__name__)

_FETCH_CAP = 8000
_SEARCH_TIMEOUT = 12

# A supplier's website is served on 80 or 443. Anything else on a public
# hostname is a service, and reaching a service is not research.
_ALLOWED_PORTS = frozenset({80, 443})


_UA = "Mozilla/5.0 (compatible; ProcWiseSupplierResearch/1.0)"


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

    The destination checks, the redirect handling and the audit line all live in
    ``services.egress`` now, not here. They were written here first, when this
    was the only model-driven request in the codebase; keeping a second copy
    would mean two places to fix when the rule changes, and the copy that did
    not get fixed would be the one still reachable.

    A refusal returns "" and is logged by egress. The reason is not returned:
    the caller feeds this straight back to a model as tool output, and telling
    it which addresses are refused is telling it what to probe for.
    """
    response = egress_get(
        url,
        purpose=Purpose.SUPPLIER_RESEARCH,
        allowed_ports=_ALLOWED_PORTS,
        follow_redirects=True,
        timeout=10,
        headers={"User-Agent": _UA},
    )
    if response is None:
        return ""
    try:
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - fail open
        log.warning("fetch_url: %s returned an error status: %s", url, exc)
        return ""
    return _html_to_text(response)
