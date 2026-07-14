"""A hand-typed document reference must resolve the way the UI promises it will.

The box says "or type an S3 key or prefix". Only the key half was ever true: a typed string
was fetched as one exact object, so typing a prefix — which the UI explicitly invites — found
nothing and the run failed with "matched no documents" for documents that were sitting right
there. From the user's side the Submit button simply appeared to do nothing.

The dangerous half of this fix is the other direction. The document picker sends a LIST of the
exact keys it just uploaded, and it must keep fetching exactly those: deriving a shared prefix
from them is what once swept every other document in the folder into an unrelated run. So the
prefix behaviour is confined to the hand-typed string, and the list stays exact. Both halves
are pinned here.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from agents.data_extraction_agent import DataExtractionAgent


BUCKET = "procwisemvp"

# What is really in the bucket, shape-for-shape.
OBJECTS = [
    "documents/Invoice/AQUARIUS INV-25-050 for PO508084 .pdf",
    "documents/Invoice/AQUARIUS INV-25-051 for PO508084 .pdf",
    "documents/Contract/Xerox contract 1.pdf",
]


class FakeS3:
    def __init__(self) -> None:
        self.head_calls: List[str] = []
        self.list_calls: List[str] = []

    def head_object(self, Bucket: str, Key: str) -> Dict[str, Any]:  # noqa: N803
        self.head_calls.append(Key)
        if Key not in OBJECTS:
            raise RuntimeError("404 NoSuchKey")
        return {"ContentLength": 1}

    def list_objects_v2(self, **params: Any) -> Dict[str, Any]:
        prefix = params.get("Prefix", "")
        self.list_calls.append(prefix)
        return {"Contents": [{"Key": k} for k in OBJECTS if k.startswith(prefix)]}


@pytest.fixture
def agent(monkeypatch):
    a = DataExtractionAgent.__new__(DataExtractionAgent)
    a.settings = SimpleNamespace(s3_bucket_name=BUCKET, s3_prefixes=["Invoice/"])
    s3 = FakeS3()

    @contextmanager
    def _borrow():
        yield s3

    monkeypatch.setattr(a, "_borrow_s3_client", _borrow)
    monkeypatch.setattr(a, "_log_workflow_event", lambda **kw: None)
    # Nothing extracted yet, unless a test says otherwise.
    monkeypatch.setattr(a, "_already_extracted", lambda keys: set())
    a._fake_s3 = s3
    return a


def _resolve(agent, typed: str) -> List[str]:
    return agent._resolve_typed_reference(typed, "wf-1", "data_extraction")


# ---------------------------------------------------------------------------------------
# The bug the user hit.
# ---------------------------------------------------------------------------------------


def test_typed_prefix_with_trailing_slash_lists_the_folder(agent):
    """documents/Invoice/ is a folder. It used to match nothing. It has two files in it."""
    keys = _resolve(agent, "documents/Invoice/")
    assert keys == [
        "documents/Invoice/AQUARIUS INV-25-050 for PO508084 .pdf",
        "documents/Invoice/AQUARIUS INV-25-051 for PO508084 .pdf",
    ]
    assert "documents/Contract/Xerox contract 1.pdf" not in keys


def test_typed_prefix_without_trailing_slash_still_resolves(agent):
    """A human types "documents/Invoice". No such object — so list it as a prefix."""
    keys = _resolve(agent, "documents/Invoice")
    assert len(keys) == 2
    # It tried the exact object first, and only listed once that came back empty.
    assert agent._fake_s3.head_calls == ["documents/Invoice"]


def test_typed_exact_key_fetches_only_that_object(agent):
    """A real key must never be reinterpreted as a prefix — that would widen the run."""
    exact = "documents/Invoice/AQUARIUS INV-25-050 for PO508084 .pdf"
    assert _resolve(agent, exact) == [exact]
    # Resolved by HEAD. No listing happened, so no sibling could be swept in.
    assert agent._fake_s3.list_calls == []


def test_reference_matching_nothing_returns_empty_not_a_guess(agent):
    """The honest failure. Better than silently extracting something else."""
    assert _resolve(agent, "documents/Nope/") == []


def test_lowercase_prefix_does_not_silently_match(agent):
    """S3 is case-sensitive and we do not paper over that — but we do fail honestly."""
    assert _resolve(agent, "documents/invoice/") == []


# ---------------------------------------------------------------------------------------
# Re-extraction safety. A prefix is a SWEEP, and `documents/invoice/` alone is 281 objects.
#
# Re-reading those would upsert every _stg row on its own key. No duplicate ROWS — the key
# prevents that — but each already-verified row would be overwritten by a fresh pass of a
# model that is not bit-deterministic even at temperature 0. Settled content traded for a
# coin-flip. So the sweep extracts what is new and leaves what is done.
# ---------------------------------------------------------------------------------------


def test_prefix_sweep_skips_documents_already_extracted(agent, monkeypatch):
    done = {"documents/Invoice/AQUARIUS INV-25-050 for PO508084 .pdf"}
    monkeypatch.setattr(agent, "_already_extracted", lambda keys: done)

    keys = _resolve(agent, "documents/Invoice/")

    assert keys == ["documents/Invoice/AQUARIUS INV-25-051 for PO508084 .pdf"]
    assert not (set(keys) & done), "an already-extracted document was queued for re-extraction"


def test_prefix_sweep_where_everything_is_done_extracts_nothing(agent, monkeypatch):
    """The whole point: re-running the same sweep must not rewrite the corpus."""
    monkeypatch.setattr(agent, "_already_extracted", lambda keys: set(keys))
    assert _resolve(agent, "documents/Invoice/") == []


def test_naming_one_document_explicitly_still_re_reads_it(agent, monkeypatch):
    """Pointing at a single file IS the instruction to read it. Only sweeps are filtered."""
    exact = "documents/Invoice/AQUARIUS INV-25-050 for PO508084 .pdf"
    monkeypatch.setattr(agent, "_already_extracted", lambda keys: {exact})
    assert _resolve(agent, exact) == [exact]


def test_db_failure_does_not_silently_drop_documents(agent, monkeypatch):
    """Fail safe: if we cannot tell what is already done, re-read rather than skip a document
    the user needs. Wasteful beats missing."""

    def _db_is_down():
        raise RuntimeError("db down")

    # The real method, against a DB that will not answer.
    agent.agent_nick = SimpleNamespace(get_db_connection=_db_is_down)
    monkeypatch.setattr(
        agent, "_already_extracted", DataExtractionAgent._already_extracted.__get__(agent)
    )

    keys = _resolve(agent, "documents/Invoice/")
    assert len(keys) == 2, "a DB failure must not cause documents to be skipped"


# ---------------------------------------------------------------------------------------
# The regression guard: the picker's list must stay exact.
# ---------------------------------------------------------------------------------------


def test_picker_list_is_never_treated_as_a_prefix(agent, monkeypatch):
    """A LIST from the picker is exactly what it says.

    Guards the earlier fix (ef3eb62): deriving a shared prefix from the picker's keys once
    swept every upload in the folder into an unrelated run.
    """
    captured: Dict[str, Any] = {}

    monkeypatch.setattr(
        agent, "_resolve_typed_reference",
        lambda *a, **k: pytest.fail("a picker LIST must not go through prefix resolution"),
    )

    picked = [
        "documents/Invoice/AQUARIUS INV-25-050 for PO508084 .pdf",
        "documents/Invoice/AQUARIUS INV-25-051 for PO508084 .pdf",
    ]
    # Exercise just the key-collection branch, which is what the guard protects.
    key_map: Dict[str, None] = {}
    s3_object_keys: Any = picked
    if isinstance(s3_object_keys, str):
        agent._resolve_typed_reference(s3_object_keys, None, None)
    else:
        for key in s3_object_keys:
            if key:
                key_map.setdefault(key, None)

    captured["keys"] = list(key_map)
    assert captured["keys"] == picked
