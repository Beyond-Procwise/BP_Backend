"""The storing half of import_pack: version, inheritance, and the candidate flip."""
import os
import uuid

import pytest

from src.services.atb.pptx_import import store
from src.services.atb.pptx_import.import_pack import import_pack

pytestmark = pytest.mark.skipif(
    os.environ.get('PROCWISE_TEST_LIVE_DB') != '1',
    reason='set PROCWISE_TEST_LIVE_DB=1 to run these against the database')

SLIDE = [(0.5, 0.35, 12.33, 0.6, 'A reused title', 30, '#172033'),
         (0.5, 1.0, 12.33, 0.4, 'the basis', 14, '#56627A'),
         (0.5, 1.5, 6.0, 1.0, 'reused body text', 11.5, '#172033')]


@pytest.fixture
def conn():
    from dotenv import load_dotenv
    load_dotenv()
    from src.services.db import get_conn
    with get_conn() as connection:
        yield connection


@pytest.fixture
def name():
    return 'test-import-%s.pptx' % uuid.uuid4().hex[:10]


def test_an_import_lands_as_a_candidate_with_its_layouts(conn, name, build_deck):
    result = import_pack(build_deck([SLIDE] * 12), name, 'tester', conn=conn)
    assert result.pack_id
    row = store.pack(conn, result.pack_id)
    assert row['status'] == 'candidate'
    assert row['slide_count'] == 12
    assert row['source_sha256']
    stored = store.layouts(conn, pack_id=result.pack_id)
    assert len(stored) == len(result.layouts) == 1
    assert stored[0]['status'] == 'candidate'


def test_a_second_import_of_the_same_name_is_a_new_version(conn, name, build_deck):
    first = import_pack(build_deck([SLIDE] * 12), name, 'tester', conn=conn)
    second = import_pack(build_deck([SLIDE] * 12), name, 'tester', conn=conn)
    assert (first.version, second.version) == (1, 2)
    assert first.pack_id != second.pack_id


def test_a_reimport_keeps_the_name_a_human_gave_a_layout(conn, name, build_deck):
    first = import_pack(build_deck([SLIDE] * 12), name, 'tester', conn=conn)
    layout_id = store.layouts(conn, pack_id=first.pack_id)[0]['layout_id']
    store.rename_layout(conn, layout_id, 'Eight headline recommendations', 'nick')

    second = import_pack(build_deck([SLIDE] * 12), name, 'tester', conn=conn)
    assert second.layouts[0]['name'] == 'Eight headline recommendations'
    assert second.layouts[0]['proposed_name'] != 'Eight headline recommendations'


def test_a_reimport_does_not_resurrect_a_rejected_layout(conn, name, build_deck):
    first = import_pack(build_deck([SLIDE] * 12), name, 'tester', conn=conn)
    layout_id = store.layouts(conn, pack_id=first.pack_id)[0]['layout_id']
    store.set_layout_status(conn, layout_id, 'rejected', 'nick')

    second = import_pack(build_deck([SLIDE] * 12), name, 'tester', conn=conn)
    assert second.layouts == []


def test_a_reimport_keeps_a_hand_defined_rating_scale(conn, name, build_deck):
    first = import_pack(build_deck([SLIDE] * 12), name, 'tester', conn=conn)
    store.define_rating_scale(conn, first.pack_id, 'hml',
                              {'High': {'bg': '#F9E1E1', 'ink': '#B42D2D'}}, 'nick')
    second = import_pack(build_deck([SLIDE] * 12), name, 'tester', conn=conn)
    assert second.pack['rating_scales']['hml']['High']['ink'] == '#B42D2D'


def test_the_version_diff_names_what_moved(conn, name, build_deck):
    import_pack(build_deck([SLIDE] * 12), name, 'tester', conn=conn)
    revised = [(0.5, 0.35, 12.33, 0.6, 'A reused title', 34, '#172033'),
               (0.5, 1.0, 12.33, 0.4, 'the basis', 14, '#56627A'),
               (0.5, 1.5, 6.0, 1.0, 'reused body text', 11.5, '#172033')]
    second = import_pack(build_deck([revised] * 12), name, 'tester', conn=conn)
    assert second.diff['type_scale_pt']['title'] == {'from': 30.0, 'to': 34.0}
