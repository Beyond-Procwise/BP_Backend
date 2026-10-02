from src.services.atb.pptx_import.evidence import Evidence
from src.services.atb.pptx_import.grid import chapter_chip_in, grid
from src.services.atb.pptx_import.read import read_deck

# The reference deck's own geometry: 0.5in margins, a 12.33in content width, the title at 0.35,
# the body at 1.5, the footnote at 7.02, four cards 2.86in wide at a 0.3in gutter, and a 0.32in
# square chapter chip at 0.5, 0.47.
GRIDDED = [(0.5, 0.35, 12.33, 0.6, 'title', 30, '#172033'),
           (0.5, 0.47, 0.32, 0.32, '', 12, '#172033'),
           (0.9, 0.47, 0.32, 0.32, '', 12, '#172033')] + \
          [(0.5 + i * (2.86 + 0.3), 1.5, 2.86, 2.0, 'card %d' % i, 12, '#172033')
           for i in range(4)] + \
          [(0.5, 7.02, 11.73, 0.3, 'a footnote', 9, '#56627A')]


def test_measures_the_margin_and_the_bands(build_deck):
    ev = Evidence()
    g = grid(read_deck(build_deck([GRIDDED] * 8)), ev)
    assert g['cols'] == 12
    assert g['margin_in'] == 0.5
    assert g['title_top_in'] == 0.35
    assert g['body_top_in'] == 1.5
    assert g['footer_top_in'] == 7.02


def test_a_decoration_does_not_decide_the_title_band(build_deck):
    # Two 0.32in chips per slide against one title: counted naively the chip's 0.47 wins and every
    # imported layout puts its title a sixth of an inch too low.
    ev = Evidence()
    g = grid(read_deck(build_deck([GRIDDED] * 8)), ev)
    assert g['title_top_in'] == 0.35


def test_measures_the_gutter_between_cards_in_a_row(build_deck):
    # 0.25in, NOT the 0.3in default: a gutter census that collected nothing would fall back to the
    # default and this test could not tell the difference.
    ev = Evidence()
    cards = [(0.5, 0.35, 12.33, 0.6, 'title', 30, '#172033')] + \
            [(0.5 + i * (2.0 + 0.25), 1.5, 2.0, 2.0, 'card %d' % i, 12, '#172033')
             for i in range(5)]
    g = grid(read_deck(build_deck([cards] * 8)), ev)
    assert g['gutter_in'] == 0.25
    assert ev.as_dict()['values']['grid.gutter_in']['gaps'] == 32   # four gaps a slide, eight slides


def test_finds_the_chapter_chip(build_deck):
    ev = Evidence()
    assert chapter_chip_in(read_deck(build_deck([GRIDDED] * 8)), ev) == 0.32


def test_an_oblong_in_the_title_band_is_not_a_chapter_chip(build_deck):
    # A small NON-square shape beside the chip: without the squareness test the mode can land on
    # the oblong's width and every imported layout gets the wrong chip size.
    ev = Evidence()
    with_oblong = GRIDDED + [(1.4, 0.47, 0.50, 0.20, '', 12, '#172033'),
                             (2.0, 0.47, 0.50, 0.20, '', 12, '#172033'),
                             (2.6, 0.47, 0.50, 0.20, '', 12, '#172033')]
    assert chapter_chip_in(read_deck(build_deck([with_oblong] * 8)), ev) == 0.32


def test_a_deck_with_no_chip_reports_none(build_deck):
    plain = [(0.5, 0.35, 12.33, 0.6, 'title', 30, '#172033'),
             (0.5, 1.5, 6.0, 1.0, 'body', 12, '#172033')]
    assert chapter_chip_in(read_deck(build_deck([plain] * 8)), Evidence()) is None


def test_the_content_width_is_recorded_against_the_page_less_its_margins(build_deck):
    ev = Evidence()
    grid(read_deck(build_deck([GRIDDED] * 8)), ev)
    recorded = ev.as_dict()['values']['grid.content_width_in']
    assert recorded['value'] == 12.33
    assert '12.33' in recorded['why']
    # Corroboration, not the measurement: eight full-width titles span it, while 32 cards at
    # 2.86in are the MODAL width. A rule that took the mode would call a card the content width.
    assert recorded['shapes'] == 8


def test_a_deck_with_nothing_full_width_says_the_margin_may_be_wrong(build_deck):
    ev = Evidence()
    narrow = [(2.0, 0.35, 4.0, 0.6, 'title', 30, '#172033'),
              (2.0, 1.5, 4.0, 1.0, 'body', 12, '#172033')]
    grid(read_deck(build_deck([narrow] * 8)), ev)
    recorded = ev.as_dict()['values']['grid.content_width_in']
    assert recorded['shapes'] == 0
    assert 'margin may be wrong' in recorded['why']


def test_a_portrait_deck_still_yields_a_grid(build_deck):
    ev = Evidence()
    portrait = [(0.59, 0.59, 7.09, 0.5, 'title', 16, '#172033'),
                (0.59, 1.4, 7.09, 1.0, 'body', 10, '#172033'),
                (0.59, 10.9, 7.09, 0.3, 'footnote', 8, '#56627A')]
    g = grid(read_deck(build_deck([portrait] * 8, width_in=8.27, height_in=11.69)), ev)
    assert g['cols'] == 12
    assert g['margin_in'] == 0.59
    assert g['footer_top_in'] == 10.9
