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


# --------------------------------------------------------------------------
# SDK clients
#
# The point of these is the ASYMMETRY. aws_client records every request;
# vector_client and graph_client record only that a client was made. Asserting
# that difference is what stops a reader of an egress log assuming uniform
# coverage.
# --------------------------------------------------------------------------

def test_an_aws_client_records_every_request_not_just_its_construction(caplog):
    client = egress.aws_client(
        "s3", purpose=Purpose.OBJECT_STORAGE, region_name="eu-west-1",
        aws_access_key_id="x", aws_secret_access_key="y",
    )
    with caplog.at_level(logging.INFO, logger="src.services.egress"):
        try:
            client.list_buckets()
        except Exception:
            pass          # offline / bad creds: the hook fires before the send
    line = "\n".join(caplog.messages)
    assert "purpose=object_storage" in line
    assert "outcome=aws_request" in line
    assert "amazonaws.com" in line


def test_constructing_an_aws_client_alone_records_nothing(caplog):
    """Construction is not a request. A line here would inflate the record with
    events that put no bytes on the wire."""
    with caplog.at_level(logging.INFO, logger="src.services.egress"):
        egress.aws_client("s3", purpose=Purpose.OBJECT_STORAGE,
                          region_name="eu-west-1",
                          aws_access_key_id="x", aws_secret_access_key="y")
    assert "aws_request" not in "\n".join(caplog.messages)


def test_a_vector_client_records_its_construction_and_says_what_it_cannot_see(
    monkeypatch, caplog,
):
    """qdrant_client has no request hook. One line stands for every upsert and
    search that follows, and the record says so rather than implying each call
    was seen."""
    import qdrant_client
    monkeypatch.setattr(qdrant_client, "QdrantClient",
                        lambda **kw: object())
    with caplog.at_level(logging.INFO, logger="src.services.egress"):
        egress.vector_client(purpose=Purpose.VECTOR_INDEX,
                             url="https://abc.eu-central-1-0.aws.cloud.qdrant.io:6333",
                             api_key="k")
    line = "\n".join(caplog.messages)
    assert "outcome=client_created" in line
    assert "destination=abc.eu-central-1-0.aws.cloud.qdrant.io" in line
    assert "per-call recording unavailable" in line


def test_a_graph_client_carries_the_same_caveat(monkeypatch, caplog):
    # neo4j is not installed in every environment (it is absent from this venv,
    # which is why the four modules that import it do so lazily). Skip rather
    # than pretend the driver is there.
    neo4j = pytest.importorskip("neo4j")
    monkeypatch.setattr(neo4j.GraphDatabase, "driver",
                        staticmethod(lambda uri, auth=None, **kw: object()))
    with caplog.at_level(logging.INFO, logger="src.services.egress"):
        egress.graph_client(purpose=Purpose.GRAPH, uri="bolt://localhost:7687")
    line = "\n".join(caplog.messages)
    assert "outcome=client_created" in line
    assert "per-call recording unavailable" in line


def test_the_sdk_factories_return_a_real_client(monkeypatch):
    """A wrapper that changed behaviour would be rejected by callers, and a
    factory nobody uses records nothing."""
    client = egress.aws_client("s3", purpose=Purpose.OBJECT_STORAGE,
                               region_name="eu-west-1",
                               aws_access_key_id="x", aws_secret_access_key="y")
    assert hasattr(client, "list_buckets") and hasattr(client, "put_object")


# --------------------------------------------------------------------------
# RAG_EXTERNAL_ENABLED
#
# The audit's 2J finding was that a tenant cannot disable external enrichment,
# and I3 was that the one switch which does exist is read at a single line in a
# single router while four importable entry points to the same egress ignore it.
# So the check lives in the factory, and these assert it cannot be walked past.
# --------------------------------------------------------------------------

def test_the_switch_is_read_in_the_factory_not_by_callers(monkeypatch):
    """Any caller of vector_client gets the disabled stand-in — none of them has
    to remember to check first."""
    monkeypatch.setenv("RAG_EXTERNAL_ENABLED", "0")
    client = egress.vector_client(purpose=Purpose.VECTOR_INDEX,
                                  url="https://x.cloud.qdrant.io:6333")
    assert isinstance(client, egress._DisabledVectorClient)


def test_disabled_reads_return_empty_so_the_product_degrades(monkeypatch):
    """A search finding nothing is what "we hold no searchable documents" looks
    like. Raising here would turn a switched-off feature into a 500."""
    monkeypatch.setenv("RAG_EXTERNAL_ENABLED", "0")
    client = egress.vector_client(purpose=Purpose.VECTOR_INDEX,
                                  url="https://x.cloud.qdrant.io:6333")
    assert client.search(collection_name="c", query_vector=[0.1]) == []
    assert client.retrieve(collection_name="c", ids=[1]) == []
    assert client.count(collection_name="c") == 0
    points, offset = client.scroll(collection_name="c")
    assert points == [] and offset is None


def test_disabled_writes_raise_rather_than_no_op(monkeypatch):
    """A silent no-op upsert would let a document report as ingested and never
    be findable — data loss wearing the costume of success."""
    monkeypatch.setenv("RAG_EXTERNAL_ENABLED", "0")
    client = egress.vector_client(purpose=Purpose.VECTOR_INDEX,
                                  url="https://x.cloud.qdrant.io:6333")
    for call in ("upsert", "delete", "delete_payload", "create_collection"):
        with pytest.raises(egress.EgressDisabled):
            getattr(client, call)(collection_name="c")


def test_an_unknown_method_raises_rather_than_silently_succeeding(monkeypatch):
    """Returning a Mock-like object for anything unrecognised would make a call
    nobody anticipated appear to work."""
    monkeypatch.setenv("RAG_EXTERNAL_ENABLED", "0")
    client = egress.vector_client(purpose=Purpose.VECTOR_INDEX,
                                  url="https://x.cloud.qdrant.io:6333")
    with pytest.raises(egress.EgressDisabled):
        client.some_method_added_next_year(collection_name="c")


def test_a_disabled_write_is_recorded(monkeypatch, caplog):
    monkeypatch.setenv("RAG_EXTERNAL_ENABLED", "0")
    client = egress.vector_client(purpose=Purpose.VECTOR_INDEX,
                                  url="https://x.cloud.qdrant.io:6333")
    with caplog.at_level(logging.INFO, logger="src.services.egress"):
        with pytest.raises(egress.EgressDisabled):
            client.upsert(collection_name="c")
    line = "\n".join(caplog.messages)
    assert "outcome=disabled" in line
    assert "RAG_EXTERNAL_ENABLED=0" in line


def test_the_switch_defaults_to_on(monkeypatch):
    """Turning it off for an existing deployment without being asked would
    silently empty every search result."""
    monkeypatch.delenv("RAG_EXTERNAL_ENABLED", raising=False)
    assert egress.external_rag_enabled() is True


@pytest.mark.parametrize("value,expected", [
    ("0", False), ("false", False), ("False", False),
    ("1", True), ("true", True), ("", True),
])
def test_the_switch_parses_the_usual_spellings(monkeypatch, value, expected):
    monkeypatch.setenv("RAG_EXTERNAL_ENABLED", value)
    assert egress.external_rag_enabled() is expected
