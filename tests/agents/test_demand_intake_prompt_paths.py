"""The demand-intake agent gets its prompt from the registry when the registry
holds a typed version in use, and exactly as before when it does not."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import src.agents.demand_intake_agent as mod
from services.prompt_engine import convert, registry, templates
from src.agents.demand_intake_agent import (DemandIntakeAgent, DemandIntakeUnavailable,
                                            _PLACEHOLDERS)

LEGACY = ('You pull out demand values.\n\nValid categories: {categories}\nFields: {fields}\n'
          'Known: {known}\nAsked: {asked_field}\nText: {text}\nAfter {today}; money in {currency}.\n'
          'Use "March" style months.\nReply with only JSON:\n'
          '{"fields": {"path": {"value": "...", "confidence": "high|medium|low"}}}')
TYPED, DECLARED = convert.braces_to_typed(LEGACY, _PLACEHOLDERS)
CONTEXT = {
    'categories': ['IT Services', 'Facilities'], 'fields': 'title: text\nvalue: text',
    'known': 'profile.entity', 'asked_field': 'intake.go_live',
    'text': 'We need the SD-WAN renewal live by March, budget about 240k.',
    'today': '2026-10-04', 'currency': 'GBP',
}


def _agent():
    agent = DemandIntakeAgent(MagicMock())
    agent.resolve_prompt = MagicMock(return_value=LEGACY)
    return agent


def _model():
    return patch.object(DemandIntakeAgent, 'call_ollama', return_value={'response': '{"fields": {}}'})


def _typed_in_use(monkeypatch, rendered=None):
    seen = {}
    monkeypatch.setattr(mod.prompt_client, "in_use", lambda key: {
        "prompt_key": key, "version": 3, "syntax": "v1", "template": TYPED, "variables": DECLARED})

    def render(key, **kw):
        seen.update(kw, key=key)
        text = templates.render_text(TYPED, DECLARED, kw["vars"], {})
        return {"instructions": rendered or text, "manifest": {"hash": "sha256:x", "version": 3}}

    monkeypatch.setattr(mod.prompt_client, "render", render)
    return seen


# --------------------------------------------------------------------------
# The conversion changes no wording
# --------------------------------------------------------------------------
def test_converting_and_converting_back_gives_the_original_wording():
    assert convert.typed_to_braces(TYPED, _PLACEHOLDERS) == LEGACY
    assert [d["name"] for d in DECLARED] == list(_PLACEHOLDERS)


@pytest.mark.parametrize("context", [
    CONTEXT,
    {},                                                              # nothing known
    {**CONTEXT, 'categories': [], 'known': None},
    {**CONTEXT, 'text': 'Quote {"a": 1} and }} and {{text}} and {{> voice@1}} please'},
    {**CONTEXT, 'text': 'Línea uno\nline two — £240k'},
    {**CONTEXT, 'fields': 12, 'known': ['a', '', 'b']},
])
def test_the_typed_template_fills_to_the_same_text_character_for_character(context):
    older = DemandIntakeAgent._render(LEGACY, context)
    newer = templates.render_text(
        TYPED, DECLARED, {n: DemandIntakeAgent._one(context.get(n)) for n in _PLACEHOLDERS}, {})
    assert newer == older


def test_the_one_place_they_differ_is_a_value_that_names_a_later_blank():
    """The older fill replaces blanks one after another, so a category called
    "{text}" had the requester's words substituted into it. The typed fill puts
    each value in once and never reads it again. That is the intended behaviour:
    these values are data, never instructions."""
    context = {**CONTEXT, 'categories': ['{text}']}
    older = DemandIntakeAgent._render(LEGACY, context)
    newer = templates.render_text(
        TYPED, DECLARED, {n: DemandIntakeAgent._one(context.get(n)) for n in _PLACEHOLDERS}, {})

    assert "Valid categories: We need the SD-WAN" in older
    assert "Valid categories: {text}\n" in newer


def test_wording_that_already_has_double_braces_is_not_converted():
    with pytest.raises(convert.NotConvertible):
        convert.braces_to_typed("Use {{this}} and {text}.", ["text"])


# --------------------------------------------------------------------------
# Which path runs is the registry's decision
# --------------------------------------------------------------------------
def test_with_nothing_typed_in_use_the_prompt_is_filled_exactly_as_before(monkeypatch):
    monkeypatch.setattr(mod.prompt_client, "in_use", lambda key: None)
    monkeypatch.setattr(mod.prompt_client, "render",
                        lambda *a, **k: pytest.fail("rendered through the engine"))
    agent = _agent()
    with _model() as model:
        agent.extract(CONTEXT)

    assert model.call_args.kwargs["prompt"] == DemandIntakeAgent._render(LEGACY, CONTEXT)


def test_imported_wording_in_use_also_takes_the_older_path(monkeypatch):
    monkeypatch.setattr(mod.prompt_client, "in_use", lambda key: {
        "version": 1, "syntax": "legacy", "template": LEGACY, "variables": []})
    agent = _agent()
    with _model() as model:
        agent.extract(CONTEXT)
    assert model.call_args.kwargs["prompt"] == DemandIntakeAgent._render(LEGACY, CONTEXT)


def test_with_a_typed_version_in_use_the_engine_renders_it_and_the_model_gets_that(monkeypatch):
    seen = _typed_in_use(monkeypatch)
    agent = _agent()
    with _model() as model:
        agent.extract({**CONTEXT, "run_id": "run-42"})

    assert model.call_args.kwargs["prompt"] == DemandIntakeAgent._render(LEGACY, CONTEXT)
    assert seen["key"] == mod.PROMPT_NAME and seen["version"] == 3
    assert seen["vars"]["categories"] == "IT Services | Facilities"
    assert seen["trace"]["run_id"] == "run-42"
    assert seen["trace"]["step_id"] == "demand_intake.extract"
    agent.resolve_prompt.assert_not_called()


def test_the_example_guard_reads_the_same_wording_on_both_paths(monkeypatch):
    older = DemandIntakeAgent._instruction_examples(LEGACY)
    _typed_in_use(monkeypatch)
    template, _ = _agent()._governed_prompt(CONTEXT)

    assert template == LEGACY
    assert DemandIntakeAgent._instruction_examples(template) == older and "march" in older


@pytest.mark.parametrize("failure", [
    mod.prompt_client.PromptUnavailable("did not complete in 30s"),
    templates.VarType("wrong type for: text (expects string)"),
    registry.VersionNotFound("no version 3"),
])
def test_if_the_prompt_cannot_be_rendered_the_model_is_not_called(monkeypatch, failure):
    monkeypatch.setattr(mod.prompt_client, "in_use", lambda key: {
        "version": 3, "syntax": "v1", "template": TYPED, "variables": DECLARED})

    def refuse(*a, **k):
        raise failure

    monkeypatch.setattr(mod.prompt_client, "render", refuse)
    agent = _agent()
    with _model() as model:
        with pytest.raises(DemandIntakeUnavailable):
            agent.extract(CONTEXT)
    model.assert_not_called()
    agent.resolve_prompt.assert_not_called()


def test_if_the_registry_cannot_be_read_it_refuses_rather_than_use_other_wording(monkeypatch):
    def down(key):
        raise mod.prompt_client.PromptUnavailable("reading prompt did not complete in 30s")

    monkeypatch.setattr(mod.prompt_client, "in_use", down)
    agent = _agent()
    with _model() as model:
        with pytest.raises(DemandIntakeUnavailable):
            agent.extract(CONTEXT)
    model.assert_not_called()
    agent.resolve_prompt.assert_not_called()
