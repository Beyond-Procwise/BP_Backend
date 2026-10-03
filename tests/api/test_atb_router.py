"""The ATB import API: what it gates, what it refuses, and what it does not leak."""
import contextlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.endpoint_gate import NotPermitted
from api.routers import atb as ar
from src.services.atb.pptx_import.import_pack import ImportRefused, ImportResult

PACK = {'key': 'k', 'name': 'K.pptx', 'format': {'kind': 'deck', 'width_in': 13.333,
                                                 'height_in': 7.5},
        'writing': {'locale': 'en-US', 'locale_contested': True, 'locale_suggested': 'en-GB'},
        'colours': {'ink': '#172033'}}
PACK_ROW = {'pack_id': 'p-1', 'pack_key': 'k', 'version': 1, 'source_file': 'K.pptx',
            'slide_count': 85, 'status': 'candidate', 'created_at': '2026-10-02T00:00:00Z',
            'created_by': 'sub-1', 'approved_by': None, 'format': PACK['format'],
            'tokens': PACK, 'evidence': {'values': {}}}
LAYOUT_ROW = {'layout_id': 'l-1', 'pack_id': 'p-1', 'layout_key': 'imported_abc1234567',
              'proposed_name': 'full-width table', 'name': None, 'status': 'candidate',
              'slide_refs': [4, 5], 'regions': [], 'slots': {}, 'example_fill': {},
              'example_source': {'file': 'K.pptx', 'slide': 4}, 'problems': []}


class _P:
    subject = 'sub-1'


@pytest.fixture
def client(monkeypatch):
    gates = []
    monkeypatch.setattr(ar, 'gate', lambda action, *a, **k: gates.append(action))
    monkeypatch.setattr(ar, 'get_conn', lambda: contextlib.nullcontext('CONN'))
    monkeypatch.setattr(ar.store, 'packs', lambda conn, **k: [PACK_ROW])
    monkeypatch.setattr(ar.store, 'pack', lambda conn, pack_id: PACK_ROW if pack_id == 'p-1' else None)
    monkeypatch.setattr(ar.store, 'layouts', lambda conn, **k: [LAYOUT_ROW])
    # A candidate by default: rejecting one is triage. The approved case is set per test.
    monkeypatch.setattr(ar.store, 'layout_status',
                        lambda conn, layout_id: LAYOUT_ROW['status'] if layout_id == 'l-1' else None)
    monkeypatch.setattr(ar.store, 'rename_layout', lambda *a, **k: 1)
    monkeypatch.setattr(ar.store, 'set_layout_status', lambda *a, **k: 1)
    monkeypatch.setattr(ar.store, 'set_pack_status', lambda *a, **k: 1)
    monkeypatch.setattr(ar.store, 'define_rating_scale', lambda *a, **k: None)
    app = FastAPI()
    app.include_router(ar.router)
    app.dependency_overrides[ar.require_user] = lambda: _P()
    client = TestClient(app)
    client.gates = gates
    return client


def _upload(client, name='pack.pptx', body=b'PK\x03\x04 not really'):
    return client.post('/atb/import',
                       files={'file': (name, body, 'application/vnd.openxmlformats-officedocument'
                                                   '.presentationml.presentation')})


def test_an_import_is_gated_as_a_write(client, monkeypatch):
    monkeypatch.setattr(ar, 'import_pack', lambda *a, **k: ImportResult(
        pack_key='k', version=1, pack=PACK, layouts=[], single_use=[], evidence={},
        pack_id='p-1'))
    assert _upload(client).status_code == 200
    assert client.gates == ['style_pack.write']


def test_an_import_refusal_is_a_400_naming_the_reason(client, monkeypatch):
    def refuse(*a, **k):
        raise ImportRefused('the file could not be opened as a presentation: bad zip')
    monkeypatch.setattr(ar, 'import_pack', refuse)
    response = _upload(client)
    assert response.status_code == 400
    assert 'could not be opened' in response.json()['detail']


def test_only_a_pptx_is_read_for_its_style(client):
    assert _upload(client, name='pack.pdf').status_code == 415


def test_the_import_reports_the_single_use_structures_and_the_contested_locale(client, monkeypatch):
    monkeypatch.setattr(ar, 'import_pack', lambda *a, **k: ImportResult(
        pack_key='k', version=2, pack=PACK,
        layouts=[{'id': 'imported_1', 'proposed_name': 'full-width table', 'slide_refs': [4, 5],
                  'problems': []}],
        single_use=[{'slides': [9], 'structure': '6-up cards'}], evidence={}, pack_id='p-1',
        problems=[{'kind': 'unresolved', 'region': 'chart1', 'slides': [10], 'why': 'no renderer'}]))
    body = _upload(client).json()
    assert body['version'] == 2
    assert body['single_use'] == [{'slides': [9], 'structure': '6-up cards'}]
    assert body['locale_contested'] is True
    assert body['locale_suggested'] == 'en-GB'
    assert body['problems'][0]['kind'] == 'unresolved'


def test_approving_a_pack_is_gated_as_configure_not_write(client):
    assert client.post('/atb/packs/p-1/approve').status_code == 200
    assert client.gates == ['style_pack.approve']


def test_approving_a_layout_is_gated_as_configure(client):
    assert client.post('/atb/layouts/l-1/approve').status_code == 200
    assert client.gates == ['style_pack.approve']


def test_rejecting_a_candidate_and_renaming_are_only_writes(client):
    """Triage. Whoever imported a deck should be able to discard the arrangements that were
    never templates, without an Admin."""
    assert client.post('/atb/layouts/l-1/reject').status_code == 200
    assert client.post('/atb/layouts/l-1', json={'name': 'Eight recommendations'}).status_code == 200
    assert client.gates == ['style_pack.write', 'style_pack.write']


def test_rejecting_an_APPROVED_layout_takes_the_authority_that_approved_it(client, monkeypatch):
    """Withdrawing an approval changes what every report built on that layout looks like, which
    is the whole reason approving is `configure`. As a blanket `write` — a reversible class with
    no policy row — any Buyer could undo an Admin's decision."""
    monkeypatch.setattr(ar.store, 'layout_status', lambda conn, layout_id: 'approved')
    assert client.post('/atb/layouts/l-1/reject').status_code == 200
    assert client.gates == ['style_pack.approve']


def test_a_refusal_from_the_gate_is_a_403(client, monkeypatch):
    def refuse(action, *a, **k):
        raise NotPermitted('role Viewer may not perform configure')
    monkeypatch.setattr(ar, 'gate', refuse)
    assert client.post('/atb/packs/p-1/approve').status_code == 403


def test_a_layout_cannot_be_renamed_to_nothing(client):
    assert client.post('/atb/layouts/l-1', json={'name': '  '}).status_code == 400


def test_a_column_cannot_be_promoted_to_a_scale_missing_a_label(client):
    response = client.post('/atb/packs/p-1/rating-scales', json={
        'name': 'hml', 'chips': {'High': {'bg': '#F9E1E1', 'ink': '#B42D2D'}},
        'promote': {'layout_key': 'imported_abc1234567', 'column': 'risk',
                    'labels': ['High', 'Low']}})
    assert response.status_code == 400
    assert 'Low' in response.json()['detail']


def test_a_scale_whose_labels_are_all_defined_is_accepted(client, monkeypatch):
    promotions = []
    monkeypatch.setattr(ar.store, 'promote_column',
                        lambda conn, **k: promotions.append(k) or 1)
    response = client.post('/atb/packs/p-1/rating-scales', json={
        'name': 'hml', 'chips': {'High': {'bg': '#eee', 'ink': '#222'},
                                 'Low': {'bg': '#efe', 'ink': '#232'}},
        'promote': {'layout_key': 'imported_abc1234567', 'column': 'risk',
                    'labels': ['High', 'Low']}})
    assert response.status_code == 200
    assert response.json()['labels'] == ['High', 'Low']
    # The promotion is APPLIED, not merely checked. The first version validated it and threw it
    # away, returning 200 as though the column had become a rating column.
    assert promotions == [{'pack_id': 'p-1', 'layout_key': 'imported_abc1234567',
                           'column': 'risk', 'scale': 'hml'}]
    assert response.json()['promoted'] == {'layout_key': 'imported_abc1234567', 'column': 'risk'}


def test_a_chip_that_is_not_a_colour_is_refused(client):
    response = client.post('/atb/packs/p-1/rating-scales',
                           json={'name': 'hml', 'chips': {'High': {'bg': 'navy-ish'}}})
    assert response.status_code == 400
    assert 'navy-ish' in response.json()['detail']


def test_a_promotion_that_matches_no_column_is_a_404(client, monkeypatch):
    monkeypatch.setattr(ar.store, 'promote_column', lambda conn, **k: 0)
    response = client.post('/atb/packs/p-1/rating-scales', json={
        'name': 'hml', 'chips': {'High': {'bg': '#eee'}},
        'promote': {'layout_key': 'imported_abc1234567', 'column': 'nope', 'labels': ['High']}})
    assert response.status_code == 404


def test_an_unknown_pack_is_a_404(client):
    assert client.get('/atb/packs/nope').status_code == 404
    assert client.get('/atb/packs/nope/evidence').status_code == 404


def test_no_response_names_a_route_or_a_table(client):
    for path in ('/atb/packs', '/atb/packs/p-1', '/atb/packs/p-1/evidence', '/atb/layouts'):
        body = client.get(path).text
        assert '/atb/' not in body, path
        assert 'bp_style_pack' not in body, path
        assert 'bp_page_layout' not in body, path
        assert '[withheld]' not in body, path


def test_the_listing_carries_the_contested_locale_flag(client):
    packs = client.get('/atb/packs').json()['packs']
    assert packs[0]['locale_contested'] is True
    assert 'tokens' not in packs[0], 'a listing is a listing'


def test_one_pack_carries_its_tokens_and_its_layouts(client):
    body = client.get('/atb/packs/p-1').json()
    assert body['tokens']['colours']['ink'] == '#172033'
    assert body['layouts'][0]['layout_key'] == 'imported_abc1234567'
    assert body['layouts'][0]['example_source']['slide'] == 4


# ------------------------------------------------------------------ review finding I6
def test_approving_something_that_does_not_exist_is_a_404(client, monkeypatch):
    """Every write returned 200 for any id. An importing pack also reached `approved` that way,
    stepping around the status filter every read path depends on."""
    monkeypatch.setattr(ar.store, 'set_pack_status', lambda *a, **k: 0)
    monkeypatch.setattr(ar.store, 'set_layout_status', lambda *a, **k: 0)
    monkeypatch.setattr(ar.store, 'rename_layout', lambda *a, **k: 0)
    assert client.post('/atb/packs/nope/approve').status_code == 404
    assert client.post('/atb/layouts/nope/approve').status_code == 404
    assert client.post('/atb/layouts/nope/reject').status_code == 404
    assert client.post('/atb/layouts/nope', json={'name': 'x'}).status_code == 404
