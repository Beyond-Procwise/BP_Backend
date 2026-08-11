"""The outbound transport describes, checks and records every call.

These assert the three properties the chokepoint has to have to be worth
building: it cannot be pointed inward, it cannot be called without saying why,
and nothing leaves without a line describing it.

What is deliberately NOT asserted: that a policy engine approved the call.
There isn't one. ``_evaluate`` returns True with a reason string saying so, and
a test that asserted "the gate allowed it" would read as coverage of a decision
that is not being made.
"""
from __future__ import annotations

import logging

import pytest

from src.services import egress
from src.services.egress import Purpose


class _Resp:
    def __init__(self, status=200, headers=None, text="ok"):
        self.status_code = status
        self.headers = headers or {"Content-Type": "text/html"}
        self.text = text


@pytest.fixture
def transport(monkeypatch):
    calls = []

    def _request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _Resp()

    monkeypatch.setattr(egress.requests, "request", _request)
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])
    return calls


# --------------------------------------------------------------------------
# Cannot be pointed inward
# --------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",
    "http://127.0.0.1:11434/api/tags",
    "http://10.100.10.180:5432/",
    "http://[::1]/",
    "file:///etc/passwd",
    "http://user:pass@example.com/",
])
def test_an_unsafe_destination_never_reaches_the_transport(transport, url):
    assert egress.get(url, purpose=Purpose.SUPPLIER_RESEARCH) is None
    assert transport == [], f"egress issued a request to {url}"


def test_an_internal_destination_is_allowed_when_the_purpose_says_so(transport):
    """The model daemon is on localhost by design. It still comes through here
    so the call is described and recorded; what changes is that reaching a
    private address is the point rather than the attack."""
    monkey = egress.get(
        "http://127.0.0.1:11434/api/tags",
        purpose=Purpose.MODEL_INFERENCE,
        require_global=False,
    )
    assert monkey is not None
    assert transport and transport[0][1] == "http://127.0.0.1:11434/api/tags"


def test_a_redirect_is_rechecked_rather_than_delegated(monkeypatch):
    hops = []

    def _request(method, url, **kwargs):
        hops.append(url)
        assert kwargs.get("allow_redirects") is False, (
            "redirect following was delegated to requests, so a hop would be "
            "made that egress never checked or recorded"
        )
        if "evil" in url:
            return _Resp(302, {"Location": "http://169.254.169.254/"})
        return _Resp()

    monkeypatch.setattr(egress.requests, "request", _request)
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])

    assert egress.get("https://evil.example/r", purpose=Purpose.SUPPLIER_RESEARCH,
                      follow_redirects=True) is None
    assert not any("169.254" in h for h in hops)


# --------------------------------------------------------------------------
# Cannot be called without saying why
# --------------------------------------------------------------------------

def test_purpose_is_required(transport):
    with pytest.raises(TypeError):
        egress.get("https://example.com/")  # type: ignore[call-arg]


def test_purpose_is_a_closed_vocabulary():
    """Free text cannot be aggregated or policed. 'supplier research' and
    'supplier_research' would be two purposes that mean one thing."""
    with pytest.raises(ValueError):
        Purpose("whatever-the-caller-felt-like")


# --------------------------------------------------------------------------
# Nothing leaves without a record
# --------------------------------------------------------------------------

def test_a_successful_call_is_recorded(transport, caplog):
    with caplog.at_level(logging.INFO, logger="src.services.egress"):
        egress.get("https://acme.example/about", purpose=Purpose.SUPPLIER_RESEARCH)
    line = "\n".join(caplog.messages)
    assert "purpose=supplier_research" in line
    assert "destination=acme.example" in line
    assert "outcome=http_200" in line


def test_a_refused_call_is_recorded_too(transport, caplog):
    """A refused call still disclosed a hostname to the resolver and, usually,
    the shape of a query. It belongs in the record on the same footing."""
    with caplog.at_level(logging.INFO, logger="src.services.egress"):
        egress.get("http://169.254.169.254/", purpose=Purpose.SUPPLIER_RESEARCH)
    line = "\n".join(caplog.messages)
    assert "outcome=refused" in line
    assert "destination=169.254.169.254" in line


def test_a_failed_call_is_recorded_without_leaking_the_exception_text(
    monkeypatch, caplog
):
    def _boom(method, url, **kwargs):
        raise ConnectionError("connect to 10.0.0.5 failed: no route")

    monkeypatch.setattr(egress.requests, "request", _boom)
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])

    with caplog.at_level(logging.INFO, logger="src.services.egress"):
        assert egress.get("https://acme.example/", purpose=Purpose.FX_RATES) is None
    line = "\n".join(caplog.messages)
    assert "outcome=error" in line and "ConnectionError" in line
    assert "10.0.0.5" not in line, "the record repeated an internal address"


# --------------------------------------------------------------------------
# Failure shape
# --------------------------------------------------------------------------

def test_request_returns_none_rather_than_raising(transport):
    """Callers in this codebase treat an outbound failure as "no data"; raising
    would turn a network problem into an outage in paths that degrade today."""
    assert egress.get("http://127.0.0.1/", purpose=Purpose.FX_RATES) is None


def test_transport_errors_can_be_re_raised_with_their_type_intact(monkeypatch):
    """Retry loops branch on the KIND of failure.

    ollama_client backs off differently for a ReadTimeout than a
    ConnectionError. Collapsing both into None would leave that code looking
    correct while taking one path forever, so the original exception has to
    reach it unchanged.
    """
    import requests as _rq

    def _boom(method, url, **kwargs):
        raise _rq.exceptions.ReadTimeout("too slow")

    monkeypatch.setattr(egress.requests, "request", _boom)
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])

    with pytest.raises(_rq.exceptions.ReadTimeout):
        egress.get("https://acme.example/", purpose=Purpose.MODEL_INFERENCE,
                   raise_transport_errors=True)


def test_a_refused_destination_still_returns_none_under_raise_transport_errors(
    transport,
):
    """A refusal is not a transport failure. Raising it into a retry loop would
    make the caller retry something that will refuse identically every time."""
    assert egress.get("http://169.254.169.254/", purpose=Purpose.MODEL_INFERENCE,
                      raise_transport_errors=True) is None
    assert transport == []


def test_request_or_raise_distinguishes_refused_from_empty(transport):
    """For callers where a silent None reads as "nothing found" rather than
    "we were not allowed to look"."""
    with pytest.raises(egress.EgressDenied) as exc:
        egress.request_or_raise("GET", "http://169.254.169.254/",
                                purpose=Purpose.SUPPLIER_RESEARCH)
    assert exc.value.purpose == "supplier_research"
