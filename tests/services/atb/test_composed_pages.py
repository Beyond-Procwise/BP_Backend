"""Composed pages: the slides a deck used ONCE, kept as pages rather than thrown away.

The ruling being carried out is §6a of the step 1 design: "emit the 8 templates now; the 26 wait
for step 2 and are then imported as composed pages". A quadrant is not a template; it is a page
someone arranged, and the distinction is stored rather than inferred.
"""
import os
import uuid

import pytest

from src.services.atb.pptx_import import store


@pytest.fixture
def conn():
    if os.environ.get('PROCWISE_TEST_LIVE_DB') != '1':
        pytest.skip('set PROCWISE_TEST_LIVE_DB=1')
    from dotenv import load_dotenv
    load_dotenv()
    from src.services.db import get_conn
    with get_conn() as connection:
        yield connection


def _pack(conn, key):
    """A candidate pack to hang layouts off, with the minimum the table requires."""
    store.insert_importing(conn, pack_key=key, version=1, source_file=key + '.pptx',
                           source_sha256='x', slide_count=3, pack={'key': key},
                           evidence={}, user='test')
    pack_id = [p for p in store.packs(conn, include_importing=True)
               if p['pack_key'] == key][0]['pack_id']
    store.mark_candidate(conn, pack_id)
    return pack_id


LAYOUT = {'id': 'imported_t1', 'proposed_name': '3-up cards', 'name': '3-up cards',
          'slide_refs': [4, 9], 'regions': [], 'slots': {}, 'example_fill': {},
          'example_source': {}, 'problems': []}


def test_a_layout_is_a_template_unless_it_says_otherwise(conn):
    """Every row that exists today predates `kind`, so the default must leave them templates."""
    key = 'kindtest-%s' % uuid.uuid4().hex[:8]
    pack_id = _pack(conn, key)
    store.insert_layout(conn, pack_id=pack_id, layout=LAYOUT)
    rows = store.layouts(conn, pack_id=pack_id)
    assert [r['kind'] for r in rows] == ['template']


def test_a_page_is_stored_as_a_page(conn):
    key = 'kindtest-%s' % uuid.uuid4().hex[:8]
    pack_id = _pack(conn, key)
    page = dict(LAYOUT, id='imported_p1', kind='page', slide_refs=[11])
    store.insert_layout(conn, pack_id=pack_id, layout=page)
    assert [r['kind'] for r in store.layouts(conn, pack_id=pack_id)] == ['page']


def test_the_two_kinds_are_asked_for_separately(conn):
    """The Layout picker reads templates and nothing else — see the browser's half of this."""
    key = 'kindtest-%s' % uuid.uuid4().hex[:8]
    pack_id = _pack(conn, key)
    store.insert_layout(conn, pack_id=pack_id, layout=LAYOUT)
    store.insert_layout(conn, pack_id=pack_id,
                        layout=dict(LAYOUT, id='imported_p1', kind='page', slide_refs=[11]))
    templates = store.layouts(conn, pack_id=pack_id, kind='template')
    pages = store.layouts(conn, pack_id=pack_id, kind='page')
    assert [r['layout_key'] for r in templates] == ['imported_t1']
    assert [r['layout_key'] for r in pages] == ['imported_p1']
    assert len(store.layouts(conn, pack_id=pack_id)) == 2       # no filter: both


def test_the_database_refuses_a_kind_nobody_defined(conn):
    key = 'kindtest-%s' % uuid.uuid4().hex[:8]
    pack_id = _pack(conn, key)
    with pytest.raises(Exception):
        store.insert_layout(conn, pack_id=pack_id,
                            layout=dict(LAYOUT, id='imported_x', kind='quadrant'))


REFERENCE = os.environ.get(
    'ATB_REFERENCE_PACK',
    os.path.expanduser('~/Downloads/Infrastructure-Procurement-Strategy-Pack.pptx'))


def _imported():
    from src.services.atb.pptx_import.import_pack import import_pack
    with open(REFERENCE, 'rb') as handle:
        return import_pack(handle.read(), os.path.basename(REFERENCE), 'test')


@pytest.mark.skipif(not os.path.exists(REFERENCE), reason='needs ATB_REFERENCE_PACK')
def test_the_deck_contributes_both_templates_and_pages():
    """Measured before this was designed: 8 reusable clusters and 26 single-use ones, and all 26
    build and validate as layouts."""
    result = _imported()
    assert len(result.layouts) == 8
    assert len(result.pages) == 26
    assert {l.get('kind', 'template') for l in result.layouts} == {'template'}
    assert {p['kind'] for p in result.pages} == {'page'}
    # the old field keeps its shape, so nothing reading it breaks
    assert len(result.single_use) == 26


@pytest.mark.skipif(not os.path.exists(REFERENCE), reason='needs ATB_REFERENCE_PACK')
def test_a_page_is_named_by_what_the_slide_says():
    result = _imported()
    names = {p['name'] for p in result.pages}
    assert 'Market intelligence: a two-speed market' in names
    assert 'Price outlook, next 12 months' in names
    # and NOT by where its boxes sit
    assert not any(n.startswith('3-up ') or n.startswith('6-up ') for n in names)
    # the structure is still recorded, for the review screen to show as the proposal
    structures = {p['proposed_name'] for p in result.pages}
    assert any('-up ' in s for s in structures)


@pytest.mark.skipif(not os.path.exists(REFERENCE), reason='needs ATB_REFERENCE_PACK')
def test_every_page_has_a_name_even_the_slide_with_no_title():
    """Review focus 2: one of the deck's 26 single-use slides carries no title band text."""
    result = _imported()
    assert all((p['name'] or '').strip() for p in result.pages)
    fallbacks = [p['name'] for p in result.pages if p['name'].startswith('Slide ')]
    assert len(fallbacks) == 1, fallbacks




def _import_with_carried(monkeypatch, carried):
    """Run a real import with a CARRIED decision, writing nothing.

    inherited() is only consulted when a connection is supplied (import_pack:78), which is right —
    there is nothing to inherit from without the database. So a connection is supplied and every
    store call that would WRITE is stubbed: what is under test is the name precedence, not the
    writes, and this file's other tests cover the writes.
    """
    from src.services.atb.pptx_import import import_pack as ip
    monkeypatch.setattr(ip.store, 'inherited', lambda conn, pack_key: carried)
    monkeypatch.setattr(ip.store, 'packs', lambda conn, **k: [])
    monkeypatch.setattr(ip.store, 'next_version', lambda conn, key: 1)
    monkeypatch.setattr(ip.store, 'insert_importing', lambda conn, **k: 'pack-probe')
    monkeypatch.setattr(ip.store, 'insert_layout', lambda conn, **k: 'layout-probe')
    monkeypatch.setattr(ip.store, 'mark_candidate', lambda conn, pack_id: None)
    with open(REFERENCE, 'rb') as handle:
        data = handle.read()
    return ip.import_pack(data, os.path.basename(REFERENCE), 'test', conn=object())


def _first_single_use_key():
    from src.services.atb.pptx_import import import_pack as ip
    from src.services.atb.pptx_import.cluster import group
    from src.services.atb.pptx_import.read import read_deck
    from src.services.atb.pptx_import.emit import layout_key
    with open(REFERENCE, 'rb') as handle:
        deck = read_deck(handle.read())
    key = ip.key_for(os.path.basename(REFERENCE))
    return layout_key([c for c in group(deck) if not c.reused][0], key)


@pytest.mark.skipif(not os.path.exists(REFERENCE), reason='needs ATB_REFERENCE_PACK')
def test_a_name_a_human_gave_a_page_outranks_the_slide_title(monkeypatch):
    """Review focus 1. §5b: a name a human supplied carries forward across re-imports, and it is
    "the one part of this nobody can automate". A page's DEFAULT name now comes from the slide, so
    the inherited name has to beat the slide title too — not just the structure string."""
    renamed = _first_single_use_key()
    result = _import_with_carried(monkeypatch, {'names': {renamed: 'The page Nick renamed'},
                                                'rating_scales': {}, 'locale': None,
                                                'rejected': []})
    page = [p for p in result.pages if p['id'] == renamed][0]
    assert page['name'] == 'The page Nick renamed'
    # and the structure is still recorded as the proposal
    assert '-up ' in page['proposed_name']


@pytest.mark.skipif(not os.path.exists(REFERENCE), reason='needs ATB_REFERENCE_PACK')
def test_a_rejected_page_stays_rejected_across_a_re_import(monkeypatch):
    """The same carried decision, for the other half of §5b."""
    gone = _first_single_use_key()
    result = _import_with_carried(monkeypatch, {'names': {}, 'rating_scales': {}, 'locale': None,
                                                'rejected': [gone]})
    assert gone not in {p['id'] for p in result.pages}
    assert len(result.pages) == 25


def test_a_page_name_falls_back_to_the_slide_number():
    from src.services.atb.pptx_import.emit import page_name
    assert page_name({'slots': {}}, [11]) == 'Slide 11'
    assert page_name({'slots': {'title': {'text': '   '}}}, [11]) == 'Slide 11'
    assert page_name({'slots': {'title': {'text': 'A real title'}}}, [11]) == 'A real title'


def test_a_long_slide_title_is_trimmed_rather_than_rejected():
    from src.services.atb.pptx_import.emit import page_name
    long_title = 'A ' + 'very ' * 40 + 'long title'
    name = page_name({'slots': {'title': {'text': long_title}}}, [3])
    assert len(name) <= 80
    assert name.startswith('A very')


@pytest.mark.skipif(not os.path.exists(REFERENCE), reason='needs ATB_REFERENCE_PACK')
def test_two_pages_with_the_same_title_are_still_told_apart():
    """Review focus 3. Names are not unique and must not be treated as identity: the KEY is."""
    result = _imported()
    keys = [p['id'] for p in result.pages]
    assert len(set(keys)) == len(keys)
