from src.services.atb.pptx_import.evidence import Evidence
from src.services.atb.pptx_import.read import read_deck
from src.services.atb.pptx_import.typescale import fonts, type_scale


def _deck(build_deck, sizes_by_y, slides=8):
    """build_deck names the font by band: Cambria above 1in, Calibri below — as the reference
    deck does."""
    return read_deck(build_deck([[(0.5, y, 6.0, 0.5, 'text at %spt' % pt, pt, '#172033')
                                  for y, pt in sizes_by_y]] * slides))


def test_the_largest_size_in_the_title_band_is_the_title(build_deck):
    ev = Evidence()
    scale = type_scale(_deck(build_deck, [(0.35, 30), (1.5, 11.5), (2.5, 10), (3.5, 9)]), ev)
    assert scale['title'] == 30.0
    assert scale['subtitle'] == 11.5
    assert scale['footer'] == 9.0


def test_a_big_size_outside_the_title_band_is_not_the_title(build_deck):
    # A 40pt figure in the middle of a page is a KPI value, not the page's title.
    ev = Evidence()
    scale = type_scale(_deck(build_deck, [(0.35, 30), (3.0, 40), (1.5, 11.5), (5.0, 9)]), ev)
    assert scale['title'] == 30.0


def test_the_body_is_the_most_used_size_not_the_largest_one(build_deck):
    # Three body runs per slide at 11.5pt against one at 16pt: a rule that took the largest
    # candidate would call 16pt the body. Every other fixture here gives each size the same count,
    # where "most used" and "largest" cannot be told apart.
    ev = Evidence()
    deck = read_deck(build_deck([[(0.5, 0.35, 12.33, 0.6, 'title', 30, '#172033'),
                                  (0.5, 1.2, 6.0, 0.4, 'a heading', 16, '#172033'),
                                  (0.5, 1.8, 6.0, 0.4, 'body one', 11.5, '#172033'),
                                  (0.5, 2.4, 6.0, 0.4, 'body two', 11.5, '#172033'),
                                  (0.5, 3.0, 6.0, 0.4, 'body three', 11.5, '#172033'),
                                  (0.5, 7.0, 6.0, 0.3, 'a footnote', 9, '#56627A')]] * 8))
    scale = type_scale(deck, ev)
    assert scale['body'] == 11.5
    assert scale['subtitle'] == 16.0, 'the larger size between body and title is the subtitle'


def test_sizes_within_point_six_merge(build_deck):
    ev = Evidence()
    scale = type_scale(_deck(build_deck, [(0.35, 30), (1.5, 11.5), (2.5, 11.0), (3.5, 9)]), ev)
    assert 11.0 not in scale.values() or 11.5 not in scale.values()
    merged = [i for i in ev.as_dict()['incidental'] if i['kind'] == 'type_size']
    assert merged and 'within 0.6pt' in merged[0]['why']


def test_a_serif_family_gets_a_serif_fallback_and_says_it_invented_it(build_deck):
    ev = Evidence()
    out = fonts(_deck(build_deck, [(0.35, 30), (1.5, 11.5)]), ev)
    assert out['heading']['family'] == 'Cambria'
    # NOT `'serif' in fallback`: the sans stack ends in "sans-serif", so that assertion is true
    # for both stacks and proved nothing.
    assert 'Georgia' in out['heading']['fallback']
    assert 'sans-serif' not in out['heading']['fallback']
    assert out['body']['family'] == 'Calibri'
    assert out['body']['fallback'].endswith('sans-serif')
    assert 'Georgia' not in out['body']['fallback']
    assert ev.as_dict()['values']['fonts.heading']['invented'] is True


def test_the_theme_fonts_are_recorded_and_not_used(build_deck):
    ev = Evidence()
    out = fonts(_deck(build_deck, [(0.35, 30), (1.5, 11.5)]), ev)
    assert out['heading']['family'] != 'Calibri Light'
    ignored = ev.as_dict()['ignored']['theme.fonts']
    # The stock template's own major font — whatever it is, the measured heading family is not it.
    assert ignored['value']['majorFont'] == 'Calibri'
    assert out['heading']['family'] not in ignored['value'].values()
    assert 'not what the deck looks like' in ignored['why']


def test_a_deck_that_states_no_sizes_still_yields_the_required_roles(build_deck):
    from src.services.atb.pptx_import.read import Box, Deck, Run, Shape
    runs = (Run(text='no size', size_pt=None, font='Calibri', colour=None, bold=False, lang=None),)
    shape = Shape(kind='text', box=Box(0.5, 1.5, 3, 0.4), runs=runs, fill=None, line=None,
                  table=None, slide=1, name='t')
    deck = Deck(width_in=13.333, height_in=7.5, slides=((shape,),),
                theme={'colours': {}, 'fonts': {}}, chart_series_colours=(), run_langs={})
    scale = type_scale(deck, Evidence())
    for role in ('title', 'subtitle', 'body', 'table', 'footer'):
        assert scale[role] > 0
