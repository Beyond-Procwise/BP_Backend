"""The one-off export of Qdrant learning records into Postgres source rows (they exist nowhere else): the record becomes its own source."""
import importlib.util
import json
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "migrate_qdrant_learning.py")
spec = importlib.util.spec_from_file_location("migrate_qdrant_learning", _PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_an_event_becomes_a_source_whose_text_is_its_summary_and_whose_metadata_is_the_whole_payload():
    point = {"id": "11111111-1111-1111-1111-111111111111",
             "payload": {"document_type": "learning", "summary": "workflow finished", "metadata": {"a": 1}, "tags": ["x"], "workflow_id": "wf"}}
    sid, content, meta = mod.record_to_source(point)
    assert sid == "11111111-1111-1111-1111-111111111111" and content == "workflow finished" and meta == point["payload"]


def test_a_model_plan_becomes_a_source_whose_text_is_the_plan():
    point = {"id": 7, "payload": {"document_type": "model_plan", "plan_text": "be warm and brief", "model_name": "M"}}
    sid, content, meta = mod.record_to_source(point)
    assert sid == "7" and content == "be warm and brief" and meta["model_name"] == "M"


def test_a_record_with_no_text_falls_back_to_its_payload_as_json_so_nothing_is_lost():
    point = {"id": "x", "payload": {"document_type": "learning", "metadata": {"k": "v"}}}
    sid, content, meta = mod.record_to_source(point)
    assert json.loads(content) == point["payload"]


def test_the_export_is_idempotent_by_source_id_and_reports_what_it_did(monkeypatch):
    seen = []
    monkeypatch.setattr(mod, "scroll_learning", lambda url, key: iter([
        {"id": "a", "payload": {"document_type": "learning", "summary": "one"}},
        {"id": "b", "payload": {"document_type": "model_plan", "plan_text": "two"}}]))
    monkeypatch.setattr(mod, "put_source", lambda tenant, coll, sid, content, meta: seen.append((tenant, coll, sid, content)))
    report = mod.export("http://q", None)
    assert report == {"read": 2, "written": 2, "failed": 0} and seen == [("default", "learning", "a", "one"), ("default", "learning", "b", "two")]
    mod.export("http://q", None)
    assert [s[2] for s in seen] == ["a", "b", "a", "b"]         # the same ids again: put_source upserts, it does not duplicate


def test_a_failed_write_is_counted_and_the_rest_continue(monkeypatch):
    monkeypatch.setattr(mod, "scroll_learning", lambda url, key: iter([{"id": "a", "payload": {"summary": "x"}}, {"id": "b", "payload": {"summary": "y"}}]))
    calls = []

    def put(tenant, coll, sid, content, meta):
        calls.append(sid)
        if sid == "a":
            raise RuntimeError("db down")
    monkeypatch.setattr(mod, "put_source", put)
    assert mod.export("http://q", None) == {"read": 2, "written": 1, "failed": 1} and calls == ["a", "b"]
