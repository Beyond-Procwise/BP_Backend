"""Storage, and the write order AUTOCOMMIT forces.

These run against the live database, which is what the importing→candidate sequence has to be
proven against: get_conn() is AUTOCOMMIT, so there is no transaction to roll back and the only
protection is a status no read path serves.
"""
import os
import uuid

import pytest

from src.services.atb.pptx_import import store

pytestmark = pytest.mark.skipif(
    os.environ.get('PROCWISE_TEST_LIVE_DB') != '1',
    reason='set PROCWISE_TEST_LIVE_DB=1 to run these against the database')

PACK = {'key': 'sample', 'name': 'Sample.pptx',
        'format': {'kind': 'deck', 'width_in': 13.333, 'height_in': 7.5},
        'colours': {'ink': '#172033'}, 'rating_scales': {}, 'writing': {'locale': 'en-GB'}}
LAYOUT = {'id': 'imported_abc1234567', 'proposed_name': '4-up cards', 'name': '4-up cards',
          'regions': [{'id': 'title', 'component': 'title_block'}], 'slots': {},
          'slide_refs': [1, 2], 'example_fill': {}, 'example_source': {}, 'problems': []}


@pytest.fixture
def conn():
    from dotenv import load_dotenv
    load_dotenv()
    from src.services.db import get_conn
    with get_conn() as connection:
        yield connection


@pytest.fixture
def key():
    """A fresh pack_key per test, so these never collide with each other or with real data."""
    return 'test_' + uuid.uuid4().hex[:12]


def _insert(conn, key, version=1, pack=None):
    return store.insert_importing(
        conn, pack_key=key, version=version, source_file='a.pptx', source_sha256='x',
        slide_count=2, pack=pack or PACK, evidence={'values': {}}, user='tester')


def test_an_importing_pack_is_never_served(conn, key):
    pack_id = _insert(conn, key)
    assert all(p['pack_id'] != pack_id for p in store.packs(conn))
    assert store.pack(conn, pack_id) is None
    store.mark_candidate(conn, pack_id)
    assert any(p['pack_id'] == pack_id for p in store.packs(conn))
    assert store.pack(conn, pack_id)['status'] == 'candidate'


def test_an_importing_pack_is_visible_when_asked_for_explicitly(conn, key):
    pack_id = _insert(conn, key)
    found = store.packs(conn, include_importing=True)
    assert any(p['pack_id'] == pack_id for p in found)


def test_versions_increment_per_key(conn, key):
    assert store.next_version(conn, key) == 1
    _insert(conn, key, version=1)
    assert store.next_version(conn, key) == 2


def test_layouts_are_served_by_pack_and_by_status(conn, key):
    pack_id = _insert(conn, key)
    layout_id = store.insert_layout(conn, pack_id=pack_id, layout=LAYOUT)
    store.mark_candidate(conn, pack_id)
    assert [l['layout_id'] for l in store.layouts(conn, pack_id=pack_id)] == [layout_id]
    assert store.layouts(conn, pack_id=pack_id, status='approved') == []
    store.set_layout_status(conn, layout_id, 'approved', 'nick')
    approved = store.layouts(conn, pack_id=pack_id, status='approved')
    assert [l['layout_id'] for l in approved] == [layout_id]
    assert approved[0]['approved_by'] == 'nick'
    assert approved[0]['approved_at'] is not None


def test_a_layout_of_an_importing_pack_is_never_served(conn, key):
    pack_id = _insert(conn, key)
    store.insert_layout(conn, pack_id=pack_id, layout=LAYOUT)
    assert store.layouts(conn) == [] or all(
        l['pack_id'] != pack_id for l in store.layouts(conn))


def test_a_reimport_inherits_names_rating_scales_and_rejections(conn, key):
    # Review Focus 4: a human renamed everything; re-measuring must not lose that.
    pack_id = _insert(conn, key)
    layout_id = store.insert_layout(conn, pack_id=pack_id, layout=LAYOUT)
    other = dict(LAYOUT, id='imported_def7654321', proposed_name='full-width table',
                 name='full-width table')
    other_id = store.insert_layout(conn, pack_id=pack_id, layout=other)
    store.mark_candidate(conn, pack_id)

    store.rename_layout(conn, layout_id, 'Eight headline recommendations', 'nick')
    store.set_layout_status(conn, other_id, 'rejected', 'nick')
    store.define_rating_scale(conn, pack_id, 'hml',
                              {'High': {'bg': '#F9E1E1', 'ink': '#B42D2D'}}, 'nick')

    carried = store.inherited(conn, key)
    assert carried['names'][LAYOUT['id']] == 'Eight headline recommendations'
    assert 'hml' in carried['rating_scales']
    assert carried['rating_scales']['hml']['High']['ink'] == '#B42D2D'
    assert other['id'] in carried['rejected']


def test_nothing_is_inherited_from_a_key_that_has_never_been_imported(conn, key):
    carried = store.inherited(conn, key)
    assert carried == {'names': {}, 'rating_scales': {}, 'locale': None, 'rejected': []}


def test_a_hand_defined_scale_is_recorded_as_the_human_s(conn, key):
    pack_id = _insert(conn, key)
    store.mark_candidate(conn, pack_id)
    store.define_rating_scale(conn, pack_id, 'hml', {'High': {'bg': '#eee', 'ink': '#222'}},
                              'nick')
    row = store.pack(conn, pack_id)
    assert row['tokens']['rating_scales']['hml']['High']['bg'] == '#eee'
    assert row['evidence']['values']['rating_scales.hml']['defined_by'] == 'user'


def test_approval_records_who_and_when(conn, key):
    pack_id = _insert(conn, key)
    store.mark_candidate(conn, pack_id)
    store.set_pack_status(conn, pack_id, 'approved', 'nick')
    row = store.pack(conn, pack_id)
    assert row['status'] == 'approved'
    assert row['approved_by'] == 'nick'
    assert row['approved_at'] is not None


def test_a_pages_own_slide_title_is_not_inherited_as_though_a_human_chose_it(conn, key):
    """A PAGE's default name is its slide's title, and `inherited` cannot tell a default from a
    rename — so a corrected title would be overwritten by the old one forever.

    Scenario: v1 of the deck has slide 11 titled "Market intelligance" (a typo). It is imported.
    The typo is fixed and the deck re-imported; the body is unchanged, so the geometry hash — and
    therefore the layout_key — is the same. Before this, carried['names'] handed the typo back and
    the page kept it while rendering the corrected words.
    """
    pack_id = _insert(conn, key)
    page = dict(LAYOUT, id='imported_page111', kind='page',
                proposed_name='3-up chart + full-width panel',
                name='Market intelligance: a two-speed market',
                slide_refs=[11],
                example_fill={'slots': {'title': {'text': 'Market intelligance: a two-speed market'}}})
    store.insert_layout(conn, pack_id=pack_id, layout=page)
    store.mark_candidate(conn, pack_id)

    assert page['id'] not in store.inherited(conn, key)['names']


def test_a_page_a_human_actually_renamed_is_still_inherited(conn, key):
    """The other half: the rename is the one part §5b says nobody can automate."""
    pack_id = _insert(conn, key)
    page = dict(LAYOUT, id='imported_page222', kind='page',
                proposed_name='3-up chart + full-width panel',
                name='Market intelligence: a two-speed market', slide_refs=[11],
                example_fill={'slots': {'title': {'text': 'Market intelligence: a two-speed market'}}})
    layout_id = store.insert_layout(conn, pack_id=pack_id, layout=page)
    store.mark_candidate(conn, pack_id)
    store.rename_layout(conn, layout_id, 'The market page Nick renamed', 'nick')

    assert store.inherited(conn, key)['names'][page['id']] == 'The market page Nick renamed'


def test_a_page_named_after_its_slide_number_is_not_inherited_either(conn, key):
    """The fallback is a default too: "Slide 75" must not travel to a v2 whose slide 75 has words."""
    pack_id = _insert(conn, key)
    page = dict(LAYOUT, id='imported_page333', kind='page', proposed_name='full-width panel',
                name='Slide 75', slide_refs=[75], example_fill={'slots': {}})
    store.insert_layout(conn, pack_id=pack_id, layout=page)
    store.mark_candidate(conn, pack_id)

    assert page['id'] not in store.inherited(conn, key)['names']


def test_a_templates_name_is_inherited_exactly_as_before(conn, key):
    """The page rule must not touch templates: their default IS the structure, and the structure
    is part of the key, so inheriting it is harmless and losing it is not."""
    pack_id = _insert(conn, key)
    store.insert_layout(conn, pack_id=pack_id, layout=dict(LAYOUT, name='3-up cards'))
    store.mark_candidate(conn, pack_id)

    assert store.inherited(conn, key)['names'][LAYOUT['id']] == '3-up cards'
