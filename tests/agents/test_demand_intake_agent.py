"""Demand Intake extraction: the model reads the request and fills named fields, nothing else.

Two properties this file exists to hold:

  · The INSTRUCTION is governed. The browser sends the values to fill the prompt with; it does
    not send the prompt. If the governed row is missing the agent refuses, because the only
    alternative is executing an instruction a caller supplied.
  · Everything the model returns is SHAPED before it leaves. The browser validates each path
    against its own configuration, but it should never have to defend itself against a reply
    that is not even the right shape.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.demand_intake_agent import DemandIntakeAgent, DemandIntakeUnavailable

_TEMPLATE = ('Categories: {categories}\nFields: {fields}\nKnown: {known}\n'
             'Asked: {asked_field}\nText: {text}\nToday: {today}\nCurrency: {currency}\n'
             'Reply with only JSON:\n{"fields": {"path": {"value": "...", "confidence": "high"}}}')

_CONTEXT = {
    'categories': ['IT Services', 'Facilities'],
    'fields': 'title: text\nvalue: text\nintake.go_live: text',
    'known': 'profile.entity',
    'asked_field': 'intake.go_live',
    'text': 'We need the SD-WAN renewal live by March, budget about 240k.',
    'today': '2026-10-04',
    'currency': 'GBP',
}


def _agent(template=_TEMPLATE):
    agent = DemandIntakeAgent(MagicMock())
    agent.resolve_prompt = MagicMock(return_value=template)
    return agent


def _replies(payload):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return patch.object(DemandIntakeAgent, 'call_ollama', return_value={'response': text})


def test_refuses_when_the_governed_prompt_is_missing():
    # A browser-supplied instruction is not governed. Refusing is the whole point: the intake
    # conversation already fails soft and asks its next question regardless.
    with _replies({'fields': {}}):
        with pytest.raises(DemandIntakeUnavailable):
            _agent(template=None).extract(_CONTEXT)


def test_the_prompt_it_sends_is_the_governed_one_with_the_values_in_it():
    with _replies({'fields': {}}) as call:
        _agent().extract(_CONTEXT)
    sent = call.call_args.kwargs['prompt']
    assert 'IT Services | Facilities' in sent
    assert 'We need the SD-WAN renewal live by March' in sent
    assert '2026-10-04' in sent and 'GBP' in sent
    assert 'intake.go_live' in sent
    # and no placeholder survived
    assert '{categories}' not in sent and '{text}' not in sent


def test_it_asks_for_json_from_a_model_that_will_answer():
    # think=False is required: AgentNick is a reasoning model and returns an empty response
    # without it (Model Routing Policy). format=json is what makes the reply parseable.
    with _replies({'fields': {}}) as call:
        _agent().extract(_CONTEXT)
    assert call.call_args.kwargs['format'] == 'json'
    assert call.call_args.kwargs['think'] is False


def test_it_returns_the_fields_the_model_found():
    reply = {'fields': {'title': {'value': 'SD-WAN renewal', 'confidence': 'high'},
                        'value': {'value': 240000, 'confidence': 'medium'}}}
    with _replies(reply):
        out = _agent().extract(_CONTEXT)
    assert out['fields']['title'] == {'value': 'SD-WAN renewal', 'confidence': 'high'}
    assert out['fields']['value'] == {'value': 240000, 'confidence': 'medium'}
    assert out['governed'] is True


def test_a_bare_value_is_accepted_as_a_value_of_unknown_confidence():
    # Models answer {"title": "x"} as often as {"title": {"value": "x"}}. Dropping the first
    # shape would make the feature look broken half the time.
    with _replies({'fields': {'title': 'SD-WAN renewal'}}):
        out = _agent().extract(_CONTEXT)
    assert out['fields']['title'] == {'value': 'SD-WAN renewal', 'confidence': 'low'}


def test_an_invented_confidence_becomes_low_rather_than_travelling():
    with _replies({'fields': {'title': {'value': 'x', 'confidence': 'extremely high'}}}):
        out = _agent().extract(_CONTEXT)
    assert out['fields']['title']['confidence'] == 'low'


def test_an_empty_value_is_not_a_value():
    with _replies({'fields': {'title': {'value': '   ', 'confidence': 'high'},
                              'value': {'value': None, 'confidence': 'high'}}}):
        out = _agent().extract(_CONTEXT)
    assert out['fields'] == {}


def test_a_list_value_survives_because_criteria_is_a_list():
    with _replies({'fields': {'criteria': {'value': ['99.95% SLA', '15% cheaper'],
                                           'confidence': 'medium'}}}):
        out = _agent().extract(_CONTEXT)
    assert out['fields']['criteria']['value'] == ['99.95% SLA', '15% cheaper']


def test_a_reply_that_is_not_json_yields_no_fields_rather_than_an_exception():
    # The requester's session is worth more than one model call.
    with _replies('I think the answer is probably March'):
        out = _agent().extract(_CONTEXT)
    assert out['fields'] == {}


def test_a_model_that_raises_yields_no_fields():
    with patch.object(DemandIntakeAgent, 'call_ollama', side_effect=RuntimeError('ollama down')):
        out = _agent().extract(_CONTEXT)
    assert out['fields'] == {}
    assert out['failed'] is True


def test_a_reply_wrapped_in_something_else_is_still_read():
    # Some replies come back {"fields": {...}}, some come back as the fields themselves.
    with _replies({'title': {'value': 'SD-WAN renewal', 'confidence': 'high'}}):
        out = _agent().extract(_CONTEXT)
    assert out['fields']['title']['value'] == 'SD-WAN renewal'


def test_the_text_is_not_allowed_to_be_the_instruction():
    """A requester who types an instruction is still only a requester.

    The text is interpolated into the governed prompt, so this cannot be prevented by escaping —
    what IS guaranteed is that no field the configuration did not name survives, and the browser
    validates every path. This test pins the server half: a path the model invents comes back
    labelled like any other and carries no authority of its own.
    """
    with _replies({'fields': {'__proto__': {'value': 'x'},
                              'approval.granted': {'value': True, 'confidence': 'high'}}}):
        out = _agent().extract(_CONTEXT)
    assert '__proto__' not in out['fields']
    assert out['fields']['approval.granted']['value'] is True


def test_a_swallowed_transport_failure_is_reported_as_a_failure():
    """call_ollama does NOT raise when Ollama refuses — it returns {"response": "", "error": …}.

    Measured against the live model on 2026-10-04: with the GPU full, Ollama answered
    "model requires more system memory (18.4 GiB) than is available (2.8 GiB)" and this agent
    returned {"fields": {}} with no failure flag — so "the model never ran" was indistinguishable
    from "the model found nothing in the text". The screen tells the requester which of those
    happened, so the distinction has to survive.
    """
    with patch.object(DemandIntakeAgent, 'call_ollama',
                      return_value={'response': '', 'error': 'model requires more system memory'}):
        out = _agent().extract(_CONTEXT)
    assert out['fields'] == {}
    assert out['failed'] is True


def test_a_model_that_genuinely_found_nothing_is_not_a_failure():
    # The other half: an empty answer to a sentence with nothing in it is a correct answer.
    with _replies({'fields': {}}):
        out = _agent().extract(_CONTEXT)
    assert out['fields'] == {}
    assert out.get('failed') is False


def test_a_reply_that_is_not_json_is_a_failure_too():
    # The model answered, but not with the one thing it was asked for.
    with _replies('I think the answer is probably March'):
        assert _agent().extract(_CONTEXT)['failed'] is True
