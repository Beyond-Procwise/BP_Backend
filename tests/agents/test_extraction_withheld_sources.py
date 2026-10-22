"""1.3e: the extraction agent refuses caller-named withheld keys and prefixes before touching S3.

A caller can name a source three ways: a list of keys (the document picker), a hand-typed key or
prefix, or a prefix. Each is checked with the shared guard BEFORE the reference is resolved or any
object is listed or fetched. The agent here is a bare instance with a fake S3 client that fails the
test if it is ever used, and a resolution step that records being reached.
"""
from __future__ import annotations

import inspect
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from src.agents import data_extraction_agent as mod
from src.services.mail_intake.store import QUARANTINE_PREFIX, RAW_MIME_PREFIX


class _Reached(Exception):
    """Raised by the fake resolution step: the guard let the reference through."""


class _FakeS3:
    def head_object(self, **kw):
        pytest.fail("S3 head_object called before the withheld guard")

    def list_objects_v2(self, **kw):
        pytest.fail("S3 list_objects_v2 called before the withheld guard")

    def get_object(self, **kw):
        pytest.fail("S3 get_object called before the withheld guard")


def _builder():
    fn = mod.DataExtractionAgent._process_documents
    params = inspect.signature(fn).parameters
    assert {"s3_object_keys", "s3_prefix", "s3_object_key"} <= set(params), "the source forms moved"
    return fn


@pytest.fixture
def agent(monkeypatch):
    a = mod.DataExtractionAgent.__new__(mod.DataExtractionAgent)
    a.settings = SimpleNamespace(s3_bucket_name="bucket", s3_prefixes=["Invoice/"])
    a.resolved_with = []

    @contextmanager
    def _borrow():
        yield _FakeS3()

    def _resolve(typed, workflow_id=None, agent_name=None):
        a.resolved_with.append(typed)
        raise _Reached(typed)
    monkeypatch.setattr(a, "_borrow_s3_client", _borrow)
    monkeypatch.setattr(a, "_resolve_typed_reference", _resolve)
    monkeypatch.setattr(a, "_log_workflow_event", lambda **kw: None)
    return a


WITHHELD_CASES = [
    pytest.param({"s3_object_keys": [QUARANTINE_PREFIX + "owner/x.pdf"]}, id="picker-list-quarantine"),
    pytest.param({"s3_object_keys": ["documents/ok.pdf", RAW_MIME_PREFIX + "msg.eml"]}, id="picker-list-raw-second"),
    pytest.param({"s3_object_keys": QUARANTINE_PREFIX + "owner/x.pdf"}, id="typed-key-quarantine"),
    pytest.param({"s3_object_keys": "mail-intake/"}, id="typed-ancestor"),
    pytest.param({"s3_object_key": RAW_MIME_PREFIX + "msg.eml"}, id="single-key-raw"),
    pytest.param({"s3_prefix": "mail-intake/"}, id="prefix-ancestor"),
    pytest.param({"s3_prefix": QUARANTINE_PREFIX}, id="prefix-quarantine"),
    pytest.param({"s3_prefix": "docs/../" + QUARANTINE_PREFIX}, id="prefix-traversal"),
]


@pytest.mark.parametrize("kwargs", WITHHELD_CASES)
def test_a_withheld_source_is_refused_before_s3_or_resolution(agent, kwargs):
    with pytest.raises(ValueError, match="withheld"):
        _builder()(agent, **kwargs)
    assert agent.resolved_with == []


def test_a_benign_typed_reference_reaches_resolution_whole(agent):
    """The typed string is checked as ONE path, not as characters: a normal reference passes the guard
    and is resolved exactly as typed (an earlier draft iterated the string and refused every slash)."""
    with pytest.raises(_Reached):
        _builder()(agent, s3_object_keys="documents/workspace/Invoice A.pdf")
    assert agent.resolved_with == ["documents/workspace/Invoice A.pdf"]


def test_a_benign_prefix_is_not_refused(agent):
    """A configured sweep prefix passes the guard; the next thing the builder does is list it, which
    the fake S3 reports as the test's own failure, so stop there."""
    with pytest.raises(BaseException) as info:
        _builder()(agent, s3_prefix="Invoice/")
    assert "list_objects_v2 called" in str(info.value), info.value
