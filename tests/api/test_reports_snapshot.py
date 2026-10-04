"""The builder-pages endpoint: the same rules as the deck, and a pack_key on the way in."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import reports as rr

_SNAP = {'reportTitle': 'Executive Intelligence Report', 'style': {'key': 'deck-x'},
         'rows': [{'id': 'r-1', 'blocks': ['b-1']}],
         'blocks': {'b-1': {'id': 'b-1', 'type': 'layout', 'layoutId': 't_prose'}}}

_JOBS = {
    'j-ok': {'job_id': 'j-ok', 'status': 'released', 'has_snapshot': True, 'run_id': 'r-1'},
    'j-blocked': {'job_id': 'j-blocked', 'status': 'blocked', 'has_snapshot': False,
                  'run_id': 'r-2'},
}


class _P:
    subject = 'sub-1'


@pytest.fixture
def client(monkeypatch):
    gates = []
    monkeypatch.setattr(rr, 'gate', lambda action, *a, **k: gates.append(action))
    monkeypatch.setattr(rr.job_store, 'get', lambda job_id: _JOBS.get(job_id))
    monkeypatch.setattr(rr.job_store, 'snapshot',
                        lambda job_id: _SNAP if job_id == 'j-ok' else None)
    # Signed off already: the sign-off rules have their own tests, and these are about the pages.
    monkeypatch.setattr(rr.signoff, 'state', lambda job: {'state': 'not_required'})
    app = FastAPI()
    app.include_router(rr.router)
    app.dependency_overrides[rr.require_user] = lambda: _P()
    client = TestClient(app)
    client.gates = gates
    return client


def test_a_released_job_serves_its_pages(client):
    response = client.get('/reports/jobs/j-ok/snapshot')
    assert response.status_code == 200
    body = response.json()
    assert body['style'] == {'key': 'deck-x'}
    assert body['blocks']['b-1']['type'] == 'layout'


def test_reading_the_pages_is_gated_as_reading_the_report(client):
    client.get('/reports/jobs/j-ok/snapshot')
    assert 'report.read' in client.gates


def test_a_job_that_is_not_released_is_a_409_like_the_deck(client):
    """The status decides, not the presence of bytes — _serve's own rule."""
    response = client.get('/reports/jobs/j-blocked/snapshot')
    assert response.status_code == 409


def test_a_released_job_with_no_pages_says_so(client, monkeypatch):
    monkeypatch.setattr(rr.job_store, 'snapshot', lambda job_id: None)
    response = client.get('/reports/jobs/j-ok/snapshot')
    assert response.status_code == 404
    assert 'pages' in response.json()['detail']


def test_a_job_nobody_filed_is_a_404(client):
    assert client.get('/reports/jobs/j-nope/snapshot').status_code == 404


def test_pages_are_held_until_sign_off_exactly_as_the_deck_is(client, monkeypatch):
    """Otherwise the builder pages are a way around the hold: the same report, readable."""
    monkeypatch.setattr(rr.signoff, 'state', lambda job: {'state': 'awaiting'})
    monkeypatch.setattr(rr.signoff, 'may_sign_off', lambda principal: False)
    assert client.get('/reports/jobs/j-ok/snapshot').status_code == 409


def test_a_refused_report_serves_no_pages_to_anybody(client, monkeypatch):
    monkeypatch.setattr(rr.signoff, 'state',
                        lambda job: {'state': 'refused', 'reason': 'the figures are stale'})
    assert client.get('/reports/jobs/j-ok/snapshot').status_code == 409


def test_generate_takes_a_pack_key_and_files_it_with_the_job(client, monkeypatch):
    filed = {}

    def create(report_type, **kw):
        filed.update(kw)
        return {'job_id': 'j-new', 'status': 'queued'}, True

    monkeypatch.setattr(rr.job_store, 'create', create)
    monkeypatch.setattr(rr.job_runner, 'submit', lambda job_id: None)
    response = client.post('/reports/generate',
                           json={'report_type': 'exec_procurement_summary',
                                 'period_start': '2026-04-01', 'period_end': '2026-09-30',
                                 'pack_key': 'deck-x'})
    assert response.status_code == 202
    assert filed.get('pack_key') == 'deck-x'
    # and NOT in the scope: the scope is hashed into the Fact Pack's id and the dedup key, and a
    # style changes none of a report's facts
    assert 'pack_key' not in (filed.get('scope') or {})


def test_generate_without_a_pack_key_is_unchanged(client, monkeypatch):
    filed = {}

    def create(report_type, **kw):
        filed.update(kw)
        return {'job_id': 'j-new', 'status': 'queued'}, True

    monkeypatch.setattr(rr.job_store, 'create', create)
    monkeypatch.setattr(rr.job_runner, 'submit', lambda job_id: None)
    response = client.post('/reports/generate',
                           json={'report_type': 'exec_procurement_summary',
                                 'period_start': '2026-04-01', 'period_end': '2026-09-30'})
    assert response.status_code == 202
    assert filed.get('pack_key') is None


def test_the_listing_says_which_reports_have_pages(client):
    """The UI offers "Open as pages" on this flag alone, so the status read must carry it.

    Beside has_page, and for the same reason the store computes it rather than selecting the
    column: a status poll must never haul the drawing itself.
    """
    public = rr._view(dict(_JOBS['j-ok']))
    assert public['has_snapshot'] is True
    assert 'snapshot' not in public          # the flag, never the drawing


def test_a_report_drawn_before_pages_existed_has_none(client):
    public = rr._view({'job_id': 'j-old', 'status': 'released', 'run_id': 'r-9'})
    assert public['has_snapshot'] is False
