"""Render: obligations in, manifest out. Pure; no database."""
from __future__ import annotations

import copy
import logging

import pytest

from services.prompt_engine import render as RD
from services.prompt_engine import templates as T

VERSION = {
    "prompt_key": "claims-triage", "version": 14, "syntax": "v1",
    "template": "{{> brand-voice@3}}\nTriage the claim for {{customer.name}} in {{region}}.",
    "variables": [
        {"name": "region", "type": "enum", "values": ["EU", "UK"]},
        {"name": "customer", "type": "object"},
    ],
    "settings": {
        "output_schema": {"$ref": "claims_triage_v2"},
        "params": {"temperature": 0.2},
        "tested_models": ["agentnick"],
        "tool_descriptions": {"lookup_policy": "Look a policy up by number.",
                              "refund": "Issue a refund."},
    },
}
FRAGMENTS = {
    ("brand-voice", 3): {"template": "Be plain and brief.", "variables": []},
    ("gdpr-disclaimer", 2): {"template": "Personal data is handled under GDPR.", "variables": []},
    ("escalation-pack", 5): {"template": "Escalate anything over the limit.", "variables": []},
}
PACKS = {"escalation-pack": 5}


def _fragment(key, version):
    return FRAGMENTS.get((key, version))


def _pack(name):
    return PACKS.get(name)


def _request(**over):
    base = {
        "prompt_id": "claims-triage", "pointer": "live",
        "vars": {"region": "EU", "customer": {"name": "Ada", "ssn": "123-45-6789"}},
        "allowed_tools": ["lookup_policy"],
        "obligations": [],
        "trace": {"run_id": "run-1", "step_id": "step-1", "attempt": 1, "decision_id": "d-1"},
    }
    base.update(over)
    return base


def _compose(**over):
    return RD.compose(VERSION, _request(**over), fragment=_fragment, pack=_pack)


def test_the_text_the_settings_and_a_manifest_come_back():
    body, manifest = _compose()

    assert body["instructions"] == "Be plain and brief.\nTriage the claim for Ada in EU."
    assert body["output_schema"] == {"$ref": "claims_triage_v2"}
    assert body["params"] == {"temperature": 0.2}
    assert body["tested_models"] == ["agentnick"]
    assert manifest["prompt_id"] == "claims-triage" and manifest["version"] == 14
    assert manifest["fragments"] == {"brand-voice": 3}
    assert manifest["obligations_applied"] == []
    assert manifest["token_count"] > 0
    assert manifest["hash"].startswith("sha256:") and len(manifest["hash"]) == 71


def test_the_same_request_gives_the_same_bytes_and_the_same_hash():
    first = _compose()
    request = _request()
    request["vars"] = {"customer": {"ssn": "123-45-6789", "name": "Ada"}, "region": "EU"}
    second = RD.compose(copy.deepcopy(VERSION), request, fragment=_fragment, pack=_pack)

    assert RD.canonical(first[0]) == RD.canonical(second[0])
    assert first[1]["hash"] == second[1]["hash"]


def test_different_wording_gives_a_different_hash():
    assert _compose()[1]["hash"] != _compose(vars={"region": "UK", "customer": {"name": "Ada"}})[1]["hash"]


def test_the_trace_does_not_change_the_hash():
    other = _compose(trace={"run_id": "run-2", "step_id": "s", "attempt": 2, "decision_id": "d-2"})
    assert other[1]["hash"] == _compose()[1]["hash"]


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
def test_only_allowed_tools_are_described():
    body, _ = _compose()
    assert body["tool_descriptions"] == [
        {"name": "lookup_policy", "description": "Look a policy up by number."}]
    assert "refund" not in RD.canonical(body)


def test_no_allowed_tools_means_no_descriptions():
    assert _compose(allowed_tools=[])[0]["tool_descriptions"] == []


# --------------------------------------------------------------------------
# Obligations
# --------------------------------------------------------------------------
def test_append_adds_the_fragment_and_records_it():
    body, manifest = _compose(obligations=["append:gdpr-disclaimer@2"])

    assert body["instructions"].endswith("\n\nPersonal data is handled under GDPR.")
    assert manifest["fragments"] == {"brand-voice": 3, "gdpr-disclaimer": 2}
    assert manifest["obligations_applied"] == ["append:gdpr-disclaimer@2"]


def test_include_pack_adds_the_pack_at_the_version_in_use_and_records_which():
    body, manifest = _compose(obligations=["include_pack:escalation-pack"])

    assert body["instructions"].endswith("\n\nEscalate anything over the limit.")
    assert manifest["fragments"]["escalation-pack"] == 5


def test_redact_removes_the_value_before_anything_is_written():
    version = dict(VERSION, template="Customer: {{customer}} / {{customer.ssn}} in {{region}}.")
    body, manifest = RD.compose(version, _request(obligations=["redact:customer.ssn"]),
                                fragment=_fragment, pack=_pack)

    assert "123-45-6789" not in RD.canonical(body)
    assert "123-45-6789" not in RD.canonical(manifest)
    assert body["instructions"] == 'Customer: {"name":"Ada","ssn":"[redacted]"} / [redacted] in EU.'
    assert manifest["obligations_applied"] == ["redact:customer.ssn"]


def test_redacting_a_whole_variable_masks_all_of_it():
    version = dict(VERSION, template="{{customer}} {{customer.name}} {{region}}")
    body, _ = RD.compose(version, _request(obligations=["redact:customer"]),
                         fragment=_fragment, pack=_pack)
    assert body["instructions"] == "[redacted] [redacted] EU"


@pytest.mark.parametrize("unknown", [
    "allow:everything", "append:gdpr-disclaimer", "append:gdpr-disclaimer@latest",
    "redact:", "REDACT:customer.ssn", "include_pack:", "", 7,
    "redact:customer.__class__", "append:gdpr-disclaimer@2; redact:x",
])
def test_an_obligation_outside_the_vocabulary_refuses_the_whole_render(unknown):
    with pytest.raises(RD.ObligationUnknown) as e:
        _compose(obligations=["redact:customer.ssn", unknown])
    assert e.value.code == "OBLIGATION_UNKNOWN"


def test_an_appended_fragment_or_pack_that_is_not_there_refuses():
    for obligation in ("append:gdpr-disclaimer@9", "include_pack:no-such-pack"):
        with pytest.raises(T.FragmentMissing):
            _compose(obligations=[obligation])


# --------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------
def test_a_missing_or_mistyped_or_extra_value_refuses_with_no_text():
    with pytest.raises(T.VarMissing):
        _compose(vars={"region": "EU"})
    with pytest.raises(T.VarType):
        _compose(vars={"region": "US", "customer": {"name": "Ada"}})
    with pytest.raises(T.VarUndeclared):
        _compose(vars={"region": "EU", "customer": {"name": "Ada"}, "tier": "plus"})


def test_wording_imported_as_it_was_is_returned_as_it_was():
    legacy = {"prompt_key": "joshi", "version": 1, "syntax": "legacy",
              "template": "Answer {question} with {{anything}}.", "variables": [], "settings": {}}
    body, manifest = RD.compose(legacy, _request(prompt_id="joshi", vars={}, allowed_tools=[]),
                                fragment=_fragment, pack=_pack)

    assert body["instructions"] == "Answer {question} with {{anything}}."
    assert manifest["fragments"] == {}


def test_values_are_refused_for_wording_that_declares_none():
    legacy = {"prompt_key": "joshi", "version": 1, "syntax": "legacy",
              "template": "Answer.", "variables": [], "settings": {}}
    with pytest.raises(T.VarUndeclared):
        RD.compose(legacy, _request(prompt_id="joshi"), fragment=_fragment, pack=_pack)


# --------------------------------------------------------------------------
# Nothing a customer wrote reaches a log
# --------------------------------------------------------------------------
def test_nothing_logged_carries_wording_or_values(caplog):
    with caplog.at_level(logging.DEBUG):
        _compose(obligations=["append:gdpr-disclaimer@2", "redact:customer.ssn"])
        with pytest.raises(T.VarType):
            _compose(vars={"region": "MARKER-REGION", "customer": {"name": "MARKER-NAME"}})

    for marker in ("Ada", "123-45-6789", "Triage the claim", "Be plain", "MARKER"):
        assert marker not in caplog.text
