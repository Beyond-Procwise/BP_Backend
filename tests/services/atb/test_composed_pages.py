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
