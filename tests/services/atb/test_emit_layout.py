import pytest

from src.services.atb.pptx_import.cluster import group
from src.services.atb.pptx_import.contract import PackInvalid, validate_layout
from src.services.atb.pptx_import.emit import build_layout, layout_key, proposed_name
from src.services.atb.pptx_import.evidence import Evidence
from src.services.atb.pptx_import.read import Box, Deck, Run, Shape

PACK = {'grid': {'cols': 12, 'margin_in': 0.5, 'gutter_in': 0.3, 'title_top_in': 0.35,
                 'body_top_in': 1.5, 'footer_top_in': 7.02},
        'type_scale_pt': {'title': 30, 'subtitle': 14, 'card_title': 13, 'body': 11.5,
                          'table': 10, 'footer': 9},
        'format': {'kind': 'deck', 'width_in': 13.333, 'height_in': 7.5}}


def _text(x, y, w, h, words, slide=1):
    runs = (Run(text=words, size_pt=12, font='Calibri', colour='#172033', bold=False,
                lang='en-GB'),)
    return Shape(kind='text', box=Box(x, y, w, h), runs=runs, fill=None, line=None,
                 table=None, slide=slide, name='TextBox')


def _deck(slides):
    return Deck(width_in=13.333, height_in=7.5, slides=tuple(slides),
                theme={'colours': {}, 'fonts': {}}, chart_series_colours=(), run_langs={})


def _two_slide_deck(body_words='Dual-track the hypervisor'):
    slides = [(_text(0.5, 0.35, 12.33, 0.6, 'Eight headline recommendations', i),
               _text(0.5, 1.0, 12.33, 0.4, 'What the board is asked to note', i),
               _text(0.5, 1.5, 6.0, 1.0, body_words, i)) for i in (1, 2)]
    return _deck(slides)


def test_a_layout_passes_the_contract():
    deck = _two_slide_deck()
    layout = build_layout(group(deck)[0], deck, PACK, Evidence(), 'Pack.pptx')
    assert validate_layout(layout) == []


def test_it_keeps_one_example_from_its_first_member_slide():
    deck = _two_slide_deck()
    layout = build_layout(group(deck)[0], deck, PACK, Evidence(), 'Pack.pptx')
    assert layout['example_source'] == {'file': 'Pack.pptx', 'slide': 1}
    assert layout['example_fill']['slots']['title']['text'] == 'Eight headline recommendations'
    assert layout['example_fill']['slots']['subtitle']['text'] == \
        'What the board is asked to note'


def test_the_body_words_reach_the_example_fill():
    deck = _two_slide_deck('Right-sized bridge renewal covering migration time')
    layout = build_layout(group(deck)[0], deck, PACK, Evidence(), 'Pack.pptx')
    assert 'Right-sized bridge renewal' in str(layout['example_fill'])


def test_a_table_example_carries_its_rows_keyed_by_column_id():
    rows = (('Area', 'Effect'), ('Freight', 'one rate card'), ('IT', 'retender'))
    slides = [(_text(0.5, 0.35, 12.33, 0.6, 'A table page', i),
               Shape(kind='table', box=Box(0.5, 1.5, 12.33, 4.0), runs=(), fill=None,
                     line=None, table=rows, slide=i, name='Table')) for i in (1, 2)]
    deck = _deck(slides)
    layout = build_layout(group(deck)[0], deck, PACK, Evidence(), 'Pack.pptx')
    table_slot = next(n for n, s in layout['slots'].items() if s['type'] == 'table')
    filled = layout['example_fill']['slots'][table_slot]['rows']
    assert filled[0] == {'area': 'Freight', 'effect': 'one rate card'}
    assert len(filled) == 2


def test_the_name_starts_as_the_proposed_name_and_both_are_kept():
    deck = _two_slide_deck()
    layout = build_layout(group(deck)[0], deck, PACK, Evidence(), 'Pack.pptx')
    assert layout['name'] == layout['proposed_name']
    assert layout['name'], 'the contract requires a non-empty name'


def test_the_proposed_name_describes_the_structure():
    slides = [(_text(0.5, 0.35, 12.33, 0.6, 't', i),
               _text(0.5, 1.5, 2.8, 1.0, 'a', i), _text(3.6, 1.5, 2.8, 1.0, 'b', i),
               _text(6.7, 1.5, 2.8, 1.0, 'c', i)) for i in (1, 2)]
    deck = _deck(slides)
    assert '3-up' in proposed_name(group(deck)[0])


def test_a_table_page_is_named_a_table():
    rows = (('A',), ('b',))
    slides = [(_text(0.5, 0.35, 12.33, 0.6, 't', i),
               Shape(kind='table', box=Box(0.5, 1.5, 12.33, 4.0), runs=(), fill=None,
                     line=None, table=rows, slide=i, name='Table')) for i in (1, 2)]
    assert 'table' in proposed_name(group(_deck(slides))[0])


def test_a_slide_with_an_empty_body_is_named_title_only():
    slides = [(_text(0.5, 0.35, 12.33, 0.6, 't', i),) for i in (1, 2)]
    assert proposed_name(group(_deck(slides))[0]) == 'title only'


def test_the_id_is_stable_across_two_runs_of_the_same_deck():
    deck = _two_slide_deck()
    first = build_layout(group(deck)[0], deck, PACK, Evidence(), 'Pack.pptx')
    second = build_layout(group(deck)[0], deck, PACK, Evidence(), 'Pack.pptx')
    assert first['id'] == second['id']
    assert first == second


def test_two_different_structures_get_different_ids():
    one = group(_two_slide_deck())[0]
    rows = (('A',), ('b',))
    other_slides = [(_text(0.5, 0.35, 12.33, 0.6, 't', i),
                     Shape(kind='table', box=Box(0.5, 1.5, 12.33, 4.0), runs=(), fill=None,
                           line=None, table=rows, slide=i, name='Table')) for i in (1, 2)]
    other = group(_deck(other_slides))[0]
    assert layout_key(one) != layout_key(other)


def test_an_id_is_lower_snake_case_as_the_contract_requires():
    import re
    key = layout_key(group(_two_slide_deck())[0])
    assert re.match(r'^[a-z][a-z0-9_]*$', key), key


def test_a_layout_that_fails_the_contract_raises(monkeypatch):
    # Pointed straight at the validator. The first version of this test passed a pack with a 0pt
    # title, which divides by zero inside the slot measurement and never reached validate_layout
    # at all — it was green with the check deleted.
    from src.services.atb.pptx_import import emit

    deck = _two_slide_deck()
    cluster = group(deck)[0]
    monkeypatch.setattr(emit, 'regions_and_slots', lambda *a, **k: (
        [{'id': 'x', 'component': 'hologram', 'box_in': {'x': 0, 'y': 0, 'w': 1, 'h': 1}}],
        {'x': {'type': 'text', 'fill': 'agent', 'max_chars': 10}},
        [],
    ))
    with pytest.raises(PackInvalid) as exc:
        build_layout(cluster, deck, PACK, Evidence(), 'Pack.pptx')
    assert 'hologram' in str(exc.value)
