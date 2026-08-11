"""fetch_url is a model-driven request. Prove it cannot be pointed inward.

The URL passed to ``fetch_url`` comes straight out of an LLM tool call
(``supplier_enrichment/research.py``: ``fetch_url(str(args.get("url", "")))``).
The model chooses the host. That makes this function the one place in the
codebase where an attacker who can influence a document — a supplier name, a
line of PDF text — can influence an outbound HTTP request.

Every test here asserts on whether the transport was REACHED, not on what it
returned. A guard that logs and proceeds is a guard that failed.

The literal addresses below are the ones that actually matter on this
deployment, not textbook examples:
  * 169.254.169.254 — EC2 instance metadata (IAM credentials)
  * 127.0.0.1:11434 — the local Ollama daemon
  * 10.100.10.180   — the live bp_sqldb host
"""
from __future__ import annotations

import pytest

from src.services import egress
from src.services.supplier_enrichment import web_tools


class _Transport:
    """Records every URL the guard let through."""

    def __init__(self, *, status: int = 200, headers: dict | None = None,
                 text: str = "ok"):
        self.calls: list[str] = []
        self.kwargs: list[dict] = []
        self._status = status
        self._headers = headers or {"Content-Type": "text/html"}
        self._text = text

    def __call__(self, method, url, **kwargs):
        self.calls.append(url)
        self.kwargs.append(kwargs)
        return _Response(self._status, self._headers, self._text, url)


class _Response:
    def __init__(self, status, headers, text, url):
        self.status_code = status
        self.headers = headers
        self.text = text
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")


@pytest.fixture
def transport(monkeypatch):
    """Patched at the egress layer, not at web_tools.

    fetch_url no longer owns the destination checks — they live in
    services.egress, which is the only module permitted to hold an HTTP client.
    These tests still drive fetch_url, because what matters is that the
    model-supplied URL cannot reach an internal address through the function the
    model actually calls; where the check physically lives is an implementation
    detail that should be free to move.
    """
    t = _Transport()
    monkeypatch.setattr(egress.requests, "request", t)
    return t


# --------------------------------------------------------------------------
# The addresses that must never be reached
# --------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
    "http://169.254.170.2/v2/credentials",          # ECS task metadata
    "http://127.0.0.1:11434/api/tags",              # local Ollama
    "http://localhost:11434/api/tags",
    "http://10.100.10.180:5432/",                   # live bp_sqldb
    "http://192.168.1.1/",
    "http://172.16.0.1/",
    "http://[::1]/",                                # IPv6 loopback
    "http://0.0.0.0/",
    "http://[fd00::1]/",                            # IPv6 unique-local
])
def test_internal_addresses_are_never_requested(transport, url):
    assert web_tools.fetch_url(url) == ""
    assert transport.calls == [], (
        f"SSRF: fetch_url issued a request to {url!r}. "
        f"Transport saw: {transport.calls}"
    )


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "gopher://127.0.0.1:11211/",
    "ftp://internal.example/",
    "http://user:pass@169.254.169.254/",            # credentials in authority
])
def test_non_http_schemes_and_authority_tricks_are_refused(transport, url):
    assert web_tools.fetch_url(url) == ""
    assert transport.calls == []


def test_a_hostname_resolving_to_a_private_address_is_refused(
    transport, monkeypatch
):
    """DNS is the bypass a literal-IP blocklist misses.

    ``metadata.evil.example`` is a public name whose A record points at the
    instance-metadata address. Blocking the literal string 169.254.169.254 does
    nothing here; the check has to happen after resolution.
    """
    monkeypatch.setattr(
        egress, "_resolve", lambda host: ["169.254.169.254"]
    )
    assert web_tools.fetch_url("http://metadata.evil.example/") == ""
    assert transport.calls == []


def test_redirects_are_not_delegated_to_the_transport(transport, monkeypatch):
    """``requests`` follows redirects by default, and a followed redirect is a
    request the guard never saw.

    This asserts on the KWARG rather than on a simulated hop: a fake transport
    that does not itself follow redirects would let this pass either way, which
    would make the test a guard that checks nothing. The only way the caller can
    stay responsible for each hop is to turn the library's own following off.
    """
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])
    web_tools.fetch_url("https://acme-supplies.example/")
    assert transport.kwargs, "transport was never reached"
    assert transport.kwargs[0].get("allow_redirects") is False, (
        "fetch_url delegated redirect-following to requests; a 302 to "
        "169.254.169.254 would then be issued without passing the guard"
    )


def test_a_redirect_into_the_private_range_is_not_followed(monkeypatch):
    """The public host answers 302 to the metadata service.

    Because the transport is told not to follow (test above), the hop is the
    guard's to make — and it must refuse this one and return the empty string
    rather than the redirected body.
    """
    hops = []

    def _get(method, url, **kwargs):
        hops.append(url)
        if "evil.example" in url:
            return _Response(
                302,
                {"Location": "http://169.254.169.254/latest/meta-data/",
                 "Content-Type": "text/html"},
                "", url,
            )
        return _Response(200, {"Content-Type": "text/html"}, "SECRET", url)

    monkeypatch.setattr(egress.requests, "request", _get)
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])
    assert web_tools.fetch_url("https://evil.example/redirect") == ""
    assert not any("169.254" in h for h in hops), (
        f"SSRF via redirect: transport reached {hops}"
    )


def test_a_redirect_to_another_public_page_is_followed(monkeypatch):
    """The guard must not break ordinary redirects — http->https, or a
    trailing-slash canonicalisation, are the common case on supplier sites."""
    hops = []

    def _get(method, url, **kwargs):
        hops.append(url)
        if url.endswith("/about"):
            return _Response(
                301,
                {"Location": "https://acme-supplies.example/about/",
                 "Content-Type": "text/html"},
                "", url,
            )
        return _Response(200, {"Content-Type": "text/html"}, "ACME Ltd", url)

    monkeypatch.setattr(egress.requests, "request", _get)
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])
    assert "ACME" in web_tools.fetch_url("https://acme-supplies.example/about")
    assert hops == ["https://acme-supplies.example/about",
                    "https://acme-supplies.example/about/"]


def test_a_redirect_loop_terminates(monkeypatch):
    def _get(method, url, **kwargs):
        return _Response(302, {"Location": "https://acme-supplies.example/loop",
                               "Content-Type": "text/html"}, "", url)

    monkeypatch.setattr(egress.requests, "request", _get)
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])
    assert web_tools.fetch_url("https://acme-supplies.example/loop") == ""


def test_a_non_standard_port_on_a_public_host_is_refused(transport, monkeypatch):
    """Port 11434/5432/6379 on a public name is a proxy for reaching a service,
    not a web page. Supplier research needs 80 and 443 and nothing else."""
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])
    assert web_tools.fetch_url("http://example.com:11434/api/tags") == ""
    assert transport.calls == []


# --------------------------------------------------------------------------
# The guard must not break the legitimate case
# --------------------------------------------------------------------------

def test_an_ordinary_public_supplier_page_still_fetches(transport, monkeypatch):
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])
    out = web_tools.fetch_url("https://acme-supplies.example/about")
    assert transport.calls == ["https://acme-supplies.example/about"]
    assert out  # non-empty text returned


def test_an_unresolvable_host_is_refused_rather_than_attempted(
    transport, monkeypatch
):
    """No resolution means no proof the host is external. Refuse, don't try."""
    monkeypatch.setattr(egress, "_resolve", lambda host: [])
    assert web_tools.fetch_url("https://nx.invalid/") == ""
    assert transport.calls == []
