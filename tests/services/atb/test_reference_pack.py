"""The known-answer test: run the importer on the reference deck and watch it rediscover the pack
that was typed by hand from the build brief.

The deck is a client document and is NOT in the repo. Set ATB_REFERENCE_PACK to it, or leave it
where it was dropped; without it these skip and say so.

Every difference from the hand-authored `consulting-navy-16x9.json` is either absent or a
CORRECTION WITH EVIDENCE. Three are expected not to match and the tests assert the disagreement
rather than the value.
"""
import os

import pytest

from src.services.atb.pptx_import.contract import validate_layout, validate_pack
from src.services.atb.pptx_import.import_pack import import_pack

REFERENCE = os.environ.get(
    'ATB_REFERENCE_PACK',
    os.path.expanduser('~/Downloads/Infrastructure-Procurement-Strategy-Pack.pptx'))

pytestmark = pytest.mark.skipif(
    not os.path.exists(REFERENCE),
    reason='set ATB_REFERENCE_PACK to the Infrastructure Procurement Strategy Pack')


@pytest.fixture(scope='module')
def result():
    with open(REFERENCE, 'rb') as handle:
        return import_pack(handle.read(), os.path.basename(REFERENCE), 'test')


def test_rediscovers_the_hand_authored_palette(result):
    colours = result.pack['colours']
    assert colours['ink'] == '#172033'
    assert colours['muted'] == '#56627A'
    assert colours['panel'] == '#F3F5F8'
    assert {colours['accent'], colours['accent_2']} == {'#2350C8', '#0F6E78'}
    assert colours['rule'] == '#D5DBE5'


def test_rediscovers_the_type_scale_and_the_fonts(result):
    scale = result.pack['type_scale_pt']
    assert scale['title'] == 30.0
    assert scale['body'] in (11.5, 12.0)
    assert scale['table'] == 10.0
    assert scale['footer'] == 9.0
    assert scale['subtitle'] == 14.0
    assert result.pack['fonts']['heading']['family'] == 'Cambria'
    assert result.pack['fonts']['body']['family'] == 'Calibri'


def test_rediscovers_the_grid_and_corrects_the_body_top_i_authored(result):
    grid = result.pack['grid']
    assert grid['margin_in'] == 0.5
    assert grid['title_top_in'] == 0.35
    assert grid['footer_top_in'] == 7.02
    # The hand-authored pack says 1.65in. The file says 1.5in, in 99 shapes.
    assert grid['body_top_in'] == 1.5
    assert result.pack['chapter_chip_in'] == 0.32


def test_the_deck_has_no_single_gutter_and_the_pack_says_so(result):
    # The hand-authored pack says 0.3in. That came from the brief, not from the file: the deck's
    # equal-width bands sit at 0.1, 0.15, 0.3, 0.45, 0.65 and 0.8in, and 0.3 is its FOURTH most
    # common gap. The same finding as the grid not being strictly twelve columns — so the modal
    # value is reported and the disagreement is disclosed rather than averaged into a design rule.
    recorded = result.evidence['values']['grid.gutter_in']
    assert result.pack['grid']['gutter_in'] == recorded['value']
    assert 'THE DECK USES SEVERAL' in recorded['why']
    distribution = dict((v, n) for v, n in recorded['distribution'])
    assert len(distribution) > 4
    assert 0.3 in distribution


def test_the_format_is_the_deck_size(result):
    assert result.pack['format'] == {'kind': 'deck', 'width_in': 13.333, 'height_in': 7.5}


def test_the_declared_language_is_contested(result):
    writing = result.pack['writing']
    assert writing['locale'] == 'en-US'
    assert writing['locale_contested'] is True
    assert writing['locale_suggested'] == 'en-GB'


def test_the_series_palette_is_a_superset_of_the_hand_authored_one(result):
    hand_authored = {'#172033', '#0F6E78', '#B42D2D', '#9A5B00', '#2350C8', '#1F7A5A'}
    assert hand_authored <= set(result.pack['series_palette'])


def test_the_theme_is_recorded_and_ignored(result):
    ignored = result.evidence['ignored']
    assert ignored['theme.fonts']['value']['majorFont'] == 'Calibri Light'
    assert ignored['theme.colours']['value']['accent1'] == '#4472C4'


def test_emits_eight_reusable_layouts_covering_fifty_nine_slides(result):
    assert len(result.layouts) == 8, [l['proposed_name'] for l in result.layouts]
    assert sum(len(l['slide_refs']) for l in result.layouts) == 59
    assert len(result.single_use) == 26


def test_the_biggest_layout_is_the_full_width_table_on_twenty_one_slides(result):
    biggest = max(result.layouts, key=lambda l: len(l['slide_refs']))
    assert len(biggest['slide_refs']) == 21
    assert 'table' in biggest['proposed_name']


def test_the_pack_and_every_layout_pass_the_contract(result):
    assert validate_pack(result.pack) == []
    for layout in result.layouts:
        assert validate_layout(layout) == [], layout['id']


def test_every_layout_keeps_a_labelled_example_from_its_own_slide(result):
    for layout in result.layouts:
        source = layout['example_source']
        assert source['file'] == os.path.basename(REFERENCE)
        assert source['slide'] == layout['slide_refs'][0]
        assert layout['example_fill']['slots'], layout['id']


def test_the_import_is_deterministic(result):
    with open(REFERENCE, 'rb') as handle:
        again = import_pack(handle.read(), os.path.basename(REFERENCE), 'test')
    assert again.pack == result.pack
    assert again.layouts == result.layouts
