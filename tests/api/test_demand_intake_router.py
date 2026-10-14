"""The intake extraction endpoint: what it requires, what it refuses, what it never echoes."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import demand_intake as dr
from src.agents.demand_intake_agent import DemandIntakeUnavailable

_CONTEXT = {'categories': ['IT Services'], 'fields': 'title: text', 'known': 'none',
            'asked_field': 'title', 'text': 'We need an SD-WAN renewal by March.',
            'today': '2026-10-04', 'currency': 'GBP'}


class _P:
    subject = 'sub-1'


@pytest.fixture
def client(monkeypatch):
    seen = []

    def fake_extract(agent_nick, context):
        seen.append(context)
        return {'fields': {'title': {'value': 'SD-WAN renewal', 'confidence': 'high'}},
                'governed': True}

    monkeypatch.setattr(dr, 'extract_fields', fake_extract)
    app = FastAPI()
    app.state.agent_nick = object()
    app.include_router(dr.router)
    app.dependency_overrides[dr.require_user] = lambda: _P()
    client = TestClient(app)
    client.seen = seen
    return client


def test_it_returns_the_fields_the_agent_found(client):
    response = client.post('/demand/intake/extract', json={'context': _CONTEXT})
    assert response.status_code == 200
    body = response.json()
    assert body['fields']['title'] == {'value': 'SD-WAN renewal', 'confidence': 'high'}
    assert body['governed'] is True


def test_the_context_reaches_the_agent_as_sent(client):
    client.post('/demand/intake/extract', json={'context': _CONTEXT})
    assert client.seen[0]['text'] == 'We need an SD-WAN renewal by March.'
    assert client.seen[0]['categories'] == ['IT Services']


def test_a_missing_governed_prompt_is_a_503_that_says_so(client, monkeypatch):
    def refuse(agent_nick, context):
        raise DemandIntakeUnavailable('the prompt demand_intake_extract is not installed, '
                                      'so extraction is not authorised')
    monkeypatch.setattr(dr, 'extract_fields', refuse)
    response = client.post('/demand/intake/extract', json={'context': _CONTEXT})
    assert response.status_code == 503
    assert 'not installed' in response.json()['detail']


def test_no_agent_nick_is_a_503_rather_than_a_500(client, monkeypatch):
    app = FastAPI()
    app.include_router(dr.router)
    app.dependency_overrides[dr.require_user] = lambda: _P()
    assert TestClient(app).post('/demand/intake/extract',
                                json={'context': _CONTEXT}).status_code == 503


def test_a_request_with_no_text_is_refused_before_a_model_is_woken(client):
    # The screen should never ask for an extraction of nothing, and a model call costs seconds.
    response = client.post('/demand/intake/extract',
                           json={'context': dict(_CONTEXT, text='   ')})
    assert response.status_code == 400
    assert client.seen == []


def test_a_request_with_no_context_at_all_is_a_422(client):
    assert client.post('/demand/intake/extract', json={}).status_code == 422


def test_an_enormous_text_is_refused_rather_than_sent(client):
    response = client.post('/demand/intake/extract',
                           json={'context': dict(_CONTEXT, text='x' * 20001)})
    assert response.status_code == 413
    assert client.seen == []


def test_the_reply_never_carries_a_prompt_back(client, monkeypatch):
    """Whatever the agent returns, the response is fields and provenance — never the instruction.

    The prompt is governed server-side precisely so it is not a thing clients hold; echoing it
    would hand every caller the text to tamper with next time.
    """
    monkeypatch.setattr(dr, 'extract_fields', lambda nick, ctx: {
        'fields': {'title': {'value': 'x', 'confidence': 'low'}},
        'governed': True, 'prompt': 'You read a procurement request…'})
    body = client.post('/demand/intake/extract', json={'context': _CONTEXT}).json()
    assert 'prompt' not in body
