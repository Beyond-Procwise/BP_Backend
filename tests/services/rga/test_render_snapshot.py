"""The third drawing of one AST: the report, as pages in the builder."""
import json
from decimal import Decimal

import pytest

from src.services.rga.layout_fit import Catalogue, LayoutOption, SlotOption
from src.services.rga.models import (ChartBlock, ChartSeries, Confidence, FactEntry, FactPack,
                                     FormatHint, MetricBlock, NarrativeBlock, Origin, ReportAST,
                                     Section)
from src.services.rga.render import snapshot as renderer
from src.services.rga.style import resolve_style_brief

_TYPE = 'exec_procurement_summary'


def _text(name, **kw):
    return SlotOption(name=name, type='text', fill='agent', **kw)


LAYOUTS = (
    LayoutOption(layout_key='t_prose', name='Title + panel', kind='template',
                 slots=(_text('title', max_words=19), _text('prose1', max_chars=240)),
                 region_ids=('title', 'prose1')),
    LayoutOption(layout_key='t_cards', name='4-up cards', kind='template',
                 slots=(_text('title', max_words=19),
                        SlotOption(name='cards1', type='list', fill='agent', item_min=1,
                                   item_max=4, heading_max=34, body_max=57)),
                 region_ids=('title', 'cards1')),
    LayoutOption(layout_key='t_chart', name='Chart + panel', kind='template',
                 slots=(_text('title', max_words=19), _text('prose1', max_chars=240),
                        SlotOption(name='chart1', type='chart', fill='bind')),
                 region_ids=('title', 'chart1', 'prose1')),
)
CAT = Catalogue(pack_key='deck-x', pack_id='p-1', format_kind='deck', layouts=LAYOUTS)


def _fact(fact_id, value, **kw):
    unassessed = value is None
    return FactEntry(fact_id=fact_id, label=kw.pop('label', fact_id), value=value,
                     unit=kw.pop('unit', None), currency=kw.pop('currency', 'GBP'),
                     format_hint=kw.pop('format_hint', FormatHint.MONEY_EXACT),
                     confidence=Confidence.UNASSESSED if unassessed else Confidence.CORROBORATED,
                     origin=Origin.OBSERVED,
                     provenance_id='prov-1', derivation='sum of invoice lines', **kw)


def _pack(facts):
    return FactPack(pack_id='pk-1', report_type_id=_TYPE, scope={},
                    as_of='2026-09-30', generated_at='2026-10-04T00:00:00Z',
                    generated_by='test', facts=facts)


def _snap(ast, pack=None, title='Executive Intelligence Report'):
    return renderer.snapshot_dict(ast, pack or _pack([_fact('F0001', Decimal('240000'))]),
                                  title=title, catalogue=CAT)


def test_each_section_becomes_one_page_block_in_its_own_row():
    ast = ReportAST(sections=[
        Section(id='s1', title='What we found', blocks=[NarrativeBlock(text='Concentrated.')]),
        Section(id='s2', title='The numbers', blocks=[MetricBlock(fact_ref='F0001')]),
    ])
    snap = _snap(ast)
    assert len(snap['rows']) == 2
    ids = [r['blocks'][0] for r in snap['rows']]
    kinds = [snap['blocks'][i]['type'] for i in ids]
    assert kinds == ['layout', 'layout']
    assert [snap['blocks'][i]['layoutId'] for i in ids] == ['t_prose', 't_cards']


def test_the_page_carries_the_words_in_the_slots_the_fit_chose():
    ast = ReportAST(sections=[Section(id='s1', title='What we found',
                                      blocks=[NarrativeBlock(text='Concentrated.')])])
    page = list(_snap(ast)['blocks'].values())[0]
    assert page['fill']['slots']['title']['text'] == 'What we found'
    assert page['fill']['slots']['prose1']['text'] == 'Concentrated.'


def test_a_figure_travels_as_a_token_and_the_fact_travels_beside_it():
    ast = ReportAST(sections=[Section(id='s2', title='The numbers',
                                      blocks=[MetricBlock(fact_ref='F0001')])])
    snap = _snap(ast)
    page = list(snap['blocks'].values())[0]
    assert page['fill']['slots']['cards1']['items'][0]['heading'] == '{{f:F0001}}'
    fact = snap['facts']['F0001']
    assert fact['value'] == 240000.0 and fact['currency'] == 'GBP'
    assert fact['format_hint'] == 'money_exact'
    assert fact['provenance'] == 'corroborated'
    # confidence and origin travel separately: badge_text forbids collapsing them
    assert fact['confidence'] == 'CORROBORATED' and fact['origin'] == 'OBSERVED'
    # THE AUTHORITATIVE SPELLING travels with the fact. FactEntry.display is the one place a figure
    # is turned into words, and its docstring says a figure printed one way and checked against
    # another spelling traces to nothing — so the browser must not re-derive it.
    assert fact['display'] == FactEntry(
        fact_id='F0001', label='F0001', value=Decimal('240000'), currency='GBP',
        format_hint=FormatHint.MONEY_EXACT, confidence=Confidence.CORROBORATED,
        origin=Origin.OBSERVED, provenance_id='prov-1', derivation='d').display


def test_an_unassessed_fact_travels_as_absent_not_as_zero():
    """Review focus 2. value=None is not a zero: the page must draw an em-dash."""
    pack = _pack([_fact('F0001', None)])
    ast = ReportAST(sections=[Section(id='s2', title='The numbers',
                                      blocks=[MetricBlock(fact_ref='F0001')])])
    snap = renderer.snapshot_dict(ast, pack, title='T', catalogue=CAT)
    assert 'F0001' in snap['facts']
    assert snap['facts']['F0001']['value'] is None
    assert snap['facts']['F0001']['provenance'] == 'unassessed'
    # and its display is the product's own empty-amount spelling, not an empty string
    assert snap['facts']['F0001']['display']


def test_the_snapshot_names_the_pack_and_does_not_embed_it():
    """Review focus 3. A pack withdrawn after the run must not live on inside the report."""
    snap = _snap(ReportAST(sections=[Section(id='s1', title='T',
                                             blocks=[NarrativeBlock(text='x')])]))
    assert snap['style'] == {'key': 'deck-x'}
    assert 'colours' not in json.dumps(snap)


def test_one_layout_draws_two_pages_with_their_own_fills():
    """Review focus 5. A layout is reusable; two sections may share one."""
    ast = ReportAST(sections=[
        Section(id='s1', title='First', blocks=[NarrativeBlock(text='One.')]),
        Section(id='s2', title='Second', blocks=[NarrativeBlock(text='Two.')]),
    ])
    snap = _snap(ast)
    pages = [snap['blocks'][r['blocks'][0]] for r in snap['rows']]
    assert [p['layoutId'] for p in pages] == ['t_prose', 't_prose']
    assert pages[0]['id'] != pages[1]['id']
    assert pages[0]['fill']['slots']['prose1']['text'] == 'One.'
    assert pages[1]['fill']['slots']['prose1']['text'] == 'Two.'


def test_a_chart_section_hosts_a_real_graph_block_in_the_chart_region():
    chart = ChartBlock(chart_type='bar',
                       series=[ChartSeries(label='Spend', fact_refs=['F0001'])])
    ast = ReportAST(sections=[Section(id='s3', title='Spend by category',
                                      blocks=[chart, NarrativeBlock(text='Concentrated.')])])
    snap = _snap(ast)
    page = next(b for b in snap['blocks'].values() if b['type'] == 'layout')
    hosted_id = page['regionBlocks']['chart1']
    hosted = snap['blocks'][hosted_id]
    assert hosted['type'] == 'graph'
    assert hosted['chartType'] == 'bar'
    # the hosted block is in the blocks map but NOT in any row: it lives in the region
    assert all(hosted_id not in r['blocks'] for r in snap['rows'])


def test_the_build_stamp_records_which_pack_and_which_facts_drew_it():
    snap = _snap(ReportAST(sections=[Section(id='s1', title='T',
                                             blocks=[NarrativeBlock(text='x')])]))
    assert snap['build']['pack_id'] == 'pk-1'
    assert snap['build']['style_pack'] == 'deck-x'
    assert len(snap['build']['pack_hash']) > 0


def test_the_report_is_not_locked_and_not_a_template_so_it_can_be_edited():
    snap = _snap(ReportAST(sections=[Section(id='s1', title='T',
                                             blocks=[NarrativeBlock(text='x')])]))
    assert snap['locked'] is False and snap['isTemplate'] is False
    assert snap['factOverrides'] == {}


def test_render_returns_an_artefact_the_post_check_can_read():
    ast = ReportAST(sections=[Section(id='s1', title='T', blocks=[NarrativeBlock(text='x')])])
    pack = _pack([_fact('F0001', Decimal('240000'))])
    brief = resolve_style_brief(_TYPE)
    art = renderer.render(ast, pack, brief, title='T', catalogue=CAT)
    assert art.media_type == 'application/json'
    assert art.renderer == 'snapshot'
    assert art.pack_id == 'pk-1' and art.pack_hash == pack.hash
    # a blank field in the reproducibility record is "a post-check failure, not a shrug"
    assert art.recorded() is True
    assert json.loads(art.content.decode('utf-8'))['reportTitle'] == 'T'


def test_a_section_nothing_can_draw_stops_the_whole_report():
    from src.services.rga.layout_fit import NoFit
    ast = ReportAST(sections=[Section(id='s9', title=' '.join(['word'] * 40),
                                      blocks=[NarrativeBlock(text='x')])])
    with pytest.raises(NoFit):
        _snap(ast)
