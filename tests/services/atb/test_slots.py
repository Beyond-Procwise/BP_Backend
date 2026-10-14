from src.services.atb.pptx_import.cluster import Cluster, group
from src.services.atb.pptx_import.contract import COMPONENTS, SLOT_TYPES
from src.services.atb.pptx_import.evidence import Evidence
from src.services.atb.pptx_import.read import Box, Deck, Shape
from src.services.atb.pptx_import.slots import LOOSE_FIT_IN, regions_and_slots

PACK = {'grid': {'cols': 12, 'margin_in': 0.5, 'gutter_in': 0.3, 'title_top_in': 0.35,
                 'body_top_in': 1.5, 'footer_top_in': 7.02},
        'type_scale_pt': {'title': 30, 'subtitle': 14, 'card_title': 13, 'body': 11.5,
                          'table': 10, 'footer': 9},
        'format': {'kind': 'deck', 'width_in': 13.333, 'height_in': 7.5}}


def _shape(x, y, w, h, kind='text', slide=1, table=None):
    return Shape(kind=kind, box=Box(x, y, w, h), runs=(), fill=None, line=None,
                 table=table, slide=slide, name='s')


def _deck(slides):
    return Deck(width_in=13.333, height_in=7.5, slides=tuple(slides),
                theme={'colours': {}, 'fonts': {}}, chart_series_colours=(), run_langs={})


def _one_cluster(slides):
    deck = _deck(slides)
    return group(deck)[0], deck


def test_a_table_on_every_member_becomes_a_table_slot():
    rows = (('Area', 'Effect'), ('Freight', 'one rate card'))
    slides = [(_shape(0.5, 1.5, 12.33, 4.2, kind='table', slide=i, table=rows),) for i in (1, 2)]
    cluster, deck = _one_cluster(slides)
    regions, slots, problems = regions_and_slots(cluster, deck, PACK, Evidence())
    table_regions = [r for r in regions if r['component'] == 'table']
    assert len(table_regions) == 1
    slot = slots[table_regions[0]['id']]
    assert slot['type'] == 'table'
    assert [c['id'] for c in slot['columns']] == ['area', 'effect']
    assert [c['label'] for c in slot['columns']] == ['Area', 'Effect']
    assert slot['max_rows'] == 1
    assert problems == []


def test_a_chart_row_keeps_its_geometry_and_says_it_cannot_draw_yet():
    # Revised ruling (ledger, Task 6): `chart` IS a legal component, so the region and slot are
    # emitted and the gap is declared, rather than the row being thrown away.
    slides = [(_shape(0.5, 1.5, 6.0, 3.0, kind='chart', slide=i),) for i in (1, 2)]
    cluster, deck = _one_cluster(slides)
    regions, slots, problems = regions_and_slots(cluster, deck, PACK, Evidence())
    chart = [r for r in regions if r['component'] == 'chart']
    assert len(chart) == 1
    assert chart[0]['box_in'] == {'x': 0.5, 'y': 1.5, 'w': 6.0, 'h': 3.0}
    assert slots[chart[0]['id']] == {'type': 'chart', 'fill': 'bind'}
    assert [p['kind'] for p in problems] == ['unresolved']
    assert 'no chart renderer' in problems[0]['why']


def test_members_that_disagree_leave_the_region_unresolved_and_emit_nothing():
    a = _shape(0.5, 1.5, 12.33, 4.2, kind='table', slide=1, table=(('A',), ('b',)))
    b = _shape(0.5, 1.5, 12.33, 4.2, kind='chart', slide=2)
    cluster = Cluster(signature=((1, 'MIXED'),), slides=(1, 2), rows=((a, b),))
    regions, slots, problems = regions_and_slots(cluster, _deck([(a,), (b,)]), PACK, Evidence())
    assert not [r for r in regions if r['component'] in ('table', 'chart')]
    kinds = [p['kind'] for p in problems]
    assert 'unresolved' in kinds
    assert sorted(problems[kinds.index('unresolved')]['slides']) == [1, 2]


def _cluster_across(slides_rows):
    """One cluster whose members are separate SLIDES, which is what a loose fit compares."""
    return Cluster(signature=((1, 'SHAPE'),), slides=tuple(range(1, len(slides_rows) + 1)),
                   rows=slides_rows[0], rows_by_slide=tuple(slides_rows))


def test_a_member_whose_box_is_off_by_more_than_the_tolerance_is_a_loose_fit():
    wide = ((_shape(0.5, 1.5, 12.33, 4.2, slide=1),),)
    narrow = ((_shape(0.5, 1.5, 11.0, 4.2, slide=2),),)
    cluster = _cluster_across([wide, narrow])
    _, _, problems = regions_and_slots(cluster, _deck([wide[0], narrow[0]]), PACK, Evidence())
    loose = [p for p in problems if p['kind'] == 'loose_fit']
    # Two members that disagree put the median between them, so BOTH are loose — the honest
    # reading of a bimodal cluster: neither box is the shape.
    assert {slide for problem in loose for slide in problem['slides']} == {1, 2}
    assert 'off the median' in loose[0]['why']
    assert str(LOOSE_FIT_IN) in loose[0]['why']


def test_the_median_of_three_members_is_a_real_box_and_only_the_outlier_is_loose():
    wide = ((_shape(0.5, 1.5, 12.33, 4.2, slide=1),),)
    same = ((_shape(0.5, 1.5, 12.33, 4.2, slide=2),),)
    narrow = ((_shape(0.5, 1.5, 11.0, 4.2, slide=3),),)
    cluster = _cluster_across([wide, same, narrow])
    regions, _, problems = regions_and_slots(cluster, _deck([wide[0], same[0], narrow[0]]),
                                             PACK, Evidence())
    loose = [p for p in problems if p['kind'] == 'loose_fit']
    assert {slide for problem in loose for slide in problem['slides']} == {3}
    assert [r for r in regions if r['component'] == 'paragraph'][0]['box_in']['w'] == 12.33


def test_a_four_up_card_row_spans_the_whole_row_not_one_card():
    # The region for a band of cards is the band. Taking the median of the four cards' boxes gave
    # something one card wide sitting in the middle of the row, and then called every card an
    # outlier against it — 114 meaningless problems on the reference deck.
    row = tuple(_shape(0.5 + i * 3.2, 1.5, 2.9, 2.0, slide=1) for i in range(4))
    cluster = Cluster(signature=((4, 'SHAPE'),), slides=(1, 2), rows=(row,),
                      rows_by_slide=((row,), (row,)))
    regions, slots, problems = regions_and_slots(cluster, _deck([row, row]), PACK, Evidence())
    cards = [r for r in regions if r['component'] == 'text_card'][0]
    assert cards['box_in']['x'] == 0.5
    assert cards['box_in']['w'] == round(0.5 + 3 * 3.2 + 2.9 - 0.5, 3)
    assert not [p for p in problems if p['kind'] == 'loose_fit']


def test_a_member_inside_the_tolerance_is_not_a_loose_fit():
    a = _shape(0.5, 1.5, 12.33, 4.2, slide=1)
    b = _shape(0.5, 1.5, 12.25, 4.2, slide=2)
    cluster = Cluster(signature=((1, 'SHAPE'),), slides=(1, 2), rows=((a, b),))
    _, _, problems = regions_and_slots(cluster, _deck([(a,), (b,)]), PACK, Evidence())
    assert not [p for p in problems if p['kind'] == 'loose_fit']


def test_several_equal_columns_become_a_list_slot():
    row = tuple(_shape(0.5 + i * 3.2, 1.5, 2.9, 2.0, slide=1) for i in range(4))
    cluster = Cluster(signature=((4, 'SHAPE'),), slides=(1, 2), rows=(row,))
    regions, slots, _ = regions_and_slots(cluster, _deck([row, row]), PACK, Evidence())
    cards = [r for r in regions if r['component'] == 'text_card']
    assert cards
    slot = slots[cards[0]['id']]
    assert slot['type'] == 'list'
    assert slot['max'] == 4
    assert slot['item']['heading']['max_chars'] > 0
    assert slot['item']['body']['max_chars'] > 0


def test_every_layout_carries_a_title_and_a_subtitle_slot_with_a_length():
    slides = [(_shape(0.5, 1.5, 6.0, 1.0, slide=i),) for i in (1, 2)]
    cluster, deck = _one_cluster(slides)
    regions, slots, _ = regions_and_slots(cluster, deck, PACK, Evidence())
    assert regions[0]['component'] == 'title_block'
    assert slots['title']['max_words'] > 0
    assert slots['subtitle']['max_chars'] > 0


def test_every_agent_prose_slot_carries_a_length():
    row = tuple(_shape(0.5 + i * 3.2, 1.5, 2.9, 2.0, slide=1) for i in range(3))
    cluster = Cluster(signature=((3, 'SHAPE'),), slides=(1, 2), rows=(row, (_shape(0.5, 4.0, 12.33, 1.0),)))
    _, slots, _ = regions_and_slots(cluster, _deck([row, row]), PACK, Evidence())
    for name, slot in slots.items():
        if slot.get('fill') == 'agent' and slot['type'] in ('text', 'rich_text', 'callout'):
            assert slot.get('max_chars') or slot.get('max_words'), name


def test_every_region_carries_a_measured_box_except_the_page_chrome():
    slides = [(_shape(0.5, 1.5, 6.0, 1.0, slide=i),) for i in (1, 2)]
    cluster, deck = _one_cluster(slides)
    regions, _, _ = regions_and_slots(cluster, deck, PACK, Evidence())
    for region in regions:
        assert 'grid' not in region
        if region['component'] != 'source_footer':
            assert 'box_in' in region


def test_two_tables_on_one_page_get_two_region_ids():
    # The contract rejects duplicate region ids, and a fixed id per role collides the moment a
    # layout has two rows of the same kind. No other fixture here has two.
    rows = (('A', 'B'), ('c', 'd'))
    top = (_shape(0.5, 1.5, 12.33, 2.0, kind='table', slide=1, table=rows),)
    bottom = (_shape(0.5, 4.0, 12.33, 2.0, kind='table', slide=1, table=rows),)
    cluster = Cluster(signature=((1, 'TABLE'), (1, 'TABLE')), slides=(1, 2), rows=(top, bottom))
    regions, slots, _ = regions_and_slots(cluster, _deck([top, bottom]), PACK, Evidence())
    table_ids = [r['id'] for r in regions if r['component'] == 'table']
    assert len(table_ids) == 2
    assert len(set(table_ids)) == 2, 'two tables, two region ids'
    assert all(i in slots for i in table_ids)


def test_every_region_names_a_contract_component_and_every_slot_a_contract_type():
    row = tuple(_shape(0.5 + i * 3.2, 1.5, 2.9, 2.0, slide=1) for i in range(4))
    table = (_shape(0.5, 4.0, 12.33, 2.0, kind='table', slide=1, table=(('A', 'B'), ('c', 'd'))),)
    cluster = Cluster(signature=((4, 'SHAPE'), (1, 'TABLE')), slides=(1, 2), rows=(row, table))
    regions, slots, _ = regions_and_slots(cluster, _deck([row, row]), PACK, Evidence())
    for region in regions:
        assert region['component'] in COMPONENTS, region
    for name, slot in slots.items():
        assert slot['type'] in SLOT_TYPES, name
    assert len({r['id'] for r in regions}) == len(regions), 'region ids must be unique'
