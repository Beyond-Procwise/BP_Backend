"""Backend-executed web tools for AgentNick supplier research (keyless).

The local AgentNick model has no internet; these functions are exposed to it as
tools. AgentNick decides what to search/read; the backend performs the network
call and returns results. Both fail open (empty result on any error) so a network
problem never breaks the research loop.
"""
from __future__ import annotations

import logging

import requests

log = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (compatible; ProcWiseSupplierResearch/1.0)"
_FETCH_TIMEOUT = 10
_FETCH_CAP = 8000
_SEARCH_TIMEOUT = 12


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


def fetch_url(url: str) -> str:
    """Fetch a page and return cleaned visible text (capped). Empty on error."""
    if not url or not url.lower().startswith(("http://", "https://")):
        return ""
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
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
    except Exception as exc:  # noqa: BLE001 - fail open
        log.warning("fetch_url failed for %r: %s", url, exc)
        return ""
