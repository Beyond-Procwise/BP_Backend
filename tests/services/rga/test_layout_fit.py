"""The catalogue: what one approved pack can hold, as the fit sees it."""
import pytest

from src.services.rga.layout_fit import NoPack, catalogue

_PACK = {'pack_id': 'p-1', 'pack_key': 'deck-x', 'version': 2, 'status': 'approved',
         'format': {'kind': 'deck', 'width_in': 13.333, 'height_in': 7.5}}
_OLDER = dict(_PACK, pack_id='p-0', version=1)
_CANDIDATE = {'pack_id': 'p-9', 'pack_key': 'deck-x', 'version': 3, 'status': 'candidate',
              'format': {'kind': 'deck'}}

_LAYOUT = {
    'layout_key': 'imported_t1', 'pack_id': 'p-1', 'name': '3-up chart + panel',
    'proposed_name': '3-up chart + panel', 'kind': 'template', 'status': 'approved',
    'regions': [{'id': 'title', 'component': 'title_block'},
                {'id': 'chart1', 'component': 'chart'},
                {'id': 'prose1', 'component': 'prose_panel'}],
    'slots': {
        'title': {'type': 'text', 'fill': 'agent', 'max_words': 19, 'required': True},
        'prose1': {'type': 'text', 'fill': 'agent', 'max_chars': 240},
        'cards1': {'type': 'list', 'fill': 'agent', 'min': 2, 'max': 8,
                   'item': {'heading': {'type': 'text', 'max_chars': 34},
                            'body': {'type': 'text', 'max_chars': 57}}},
        'rows1': {'type': 'table', 'fill': 'agent',
                  'columns': [{'id': 'area', 'label': 'Area', 'max_chars': 32},
                              {'id': 'value', 'label': 'Value', 'max_chars': 32}]},
        'chart1': {'type': 'chart', 'fill': 'bind'},
        'sources': {'type': 'sources', 'fill': 'auto'},
    },
}


class _Conn:
    """Stands in for the store, which is the only thing this module reads."""

    def __init__(self, packs, layouts):
        self._packs, self._layouts = packs, layouts

    def packs(self):
        return list(self._packs)

    def layouts(self, pack_id=None, status=None, kind=None):
        out = [l for l in self._layouts
               if (pack_id is None or l['pack_id'] == pack_id)
               and (status is None or l['status'] == status)
               and (kind is None or l['kind'] == kind)]
        return out


@pytest.fixture
def store(monkeypatch):
    def install(packs, layouts):
        conn = _Conn(packs, layouts)
        monkeypatch.setattr('src.services.rga.layout_fit.store.packs',
                            lambda c, **k: conn.packs())
        monkeypatch.setattr('src.services.rga.layout_fit.store.layouts',
                            lambda c, **k: conn.layouts(**k))
        return conn
    return install


def test_the_catalogue_is_the_newest_approved_version_of_that_pack(store):
    store([_OLDER, _PACK, _CANDIDATE], [_LAYOUT])
    cat = catalogue(object(), 'deck-x')
    assert cat.pack_id == 'p-1'          # version 2, approved — not the candidate version 3
    assert cat.format_kind == 'deck'
    assert [l.layout_key for l in cat.layouts] == ['imported_t1']


def test_a_pack_nobody_approved_is_not_a_catalogue(store):
    store([_CANDIDATE], [_LAYOUT])
    with pytest.raises(NoPack):
        catalogue(object(), 'deck-x')


def test_a_pack_key_nobody_imported_is_not_a_catalogue(store):
    store([_PACK], [_LAYOUT])
    with pytest.raises(NoPack):
        catalogue(object(), 'deck-nope')


def test_it_reads_the_slot_limits_the_layout_states(store):
    store([_PACK], [_LAYOUT])
    slots = {s.name: s for s in catalogue(object(), 'deck-x').layouts[0].slots}
    assert slots['title'].type == 'text' and slots['title'].max_words == 19
    assert slots['prose1'].max_chars == 240
    assert slots['cards1'].item_min == 2 and slots['cards1'].item_max == 8
    assert slots['cards1'].heading_max == 34 and slots['cards1'].body_max == 57
    assert [c['id'] for c in slots['rows1'].columns] == ['area', 'value']


def test_it_carries_who_fills_each_slot_because_only_the_agent_s_are_assignable(store):
    store([_PACK], [_LAYOUT])
    slots = {s.name: s for s in catalogue(object(), 'deck-x').layouts[0].slots}
    assert slots['title'].fill == 'agent'
    assert slots['chart1'].fill == 'bind'        # a region host, never prose
    assert slots['sources'].fill == 'auto'       # the platform's, not the report's


# ---------------------------------------------------------------------------
# The fit: which layout draws a section, or why none can.
# ---------------------------------------------------------------------------
from src.services.rga.layout_fit import LayoutOption, NoFit, SlotOption, fit          # noqa: E402
from src.services.rga.models import (ChartBlock, ChartSeries, FindingListBlock,       # noqa: E402
                                     MetricBlock, NarrativeBlock, Section, TableBlock)


def _text(name, **kw):
    return SlotOption(name=name, type='text', fill='agent', **kw)


def _list(name, item_min=2, item_max=8, heading_max=34, body_max=57):
    return SlotOption(name=name, type='list', fill='agent', item_min=item_min,
                      item_max=item_max, heading_max=heading_max, body_max=body_max)


TITLE_AND_PROSE = LayoutOption(
    layout_key='t_prose', name='Title + panel', kind='template',
    slots=(_text('title', max_words=19), _text('prose1', max_chars=240),
           SlotOption(name='sources', type='sources', fill='auto')),
    region_ids=('title', 'prose1', 'footer'))

TITLE_AND_CARDS = LayoutOption(
    layout_key='t_cards', name='4-up cards', kind='template',
    slots=(_text('title', max_words=19), _list('cards1', item_min=2, item_max=4)),
    region_ids=('title', 'cards1'))

CHART_AND_PANEL = LayoutOption(
    layout_key='t_chart', name='Chart + panel', kind='template',
    slots=(_text('title', max_words=19), _text('prose1', max_chars=240),
           SlotOption(name='chart1', type='chart', fill='bind')),
    region_ids=('title', 'chart1', 'prose1'))

TABLE = LayoutOption(
    layout_key='t_table', name='Full-width table', kind='template',
    slots=(_text('title', max_words=19),
           SlotOption(name='rows1', type='table', fill='agent',
                      columns=({'id': 'area', 'label': 'Area'},
                               {'id': 'value', 'label': 'Value'}))),
    region_ids=('title', 'rows1'))

ALL = (TITLE_AND_PROSE, TITLE_AND_CARDS, CHART_AND_PANEL, TABLE)


def _section(id_, title, blocks):
    return Section(id=id_, title=title, blocks=blocks)


def test_prose_goes_on_a_layout_with_a_text_slot():
    s = _section('s1', 'What we found', [NarrativeBlock(text='Spend is concentrated.')])
    f = fit(s, ALL)
    assert f.layout_key == 't_prose'
    assert f.text['title'] == 'What we found'
    assert f.text['prose1'] == 'Spend is concentrated.'


def test_metrics_become_the_items_of_a_list_with_the_figure_as_the_heading():
    s = _section('s2', 'The numbers', [MetricBlock(fact_ref='spend.total'),
                                       MetricBlock(fact_ref='spend.addressable')])
    f = fit(s, ALL)
    assert f.layout_key == 't_cards'
    assert f.lists['cards1'] == [('{{f:spend.total}}', 'spend.total'),
                                 ('{{f:spend.addressable}}', 'spend.addressable')]


def test_a_chart_is_recorded_for_its_region_to_host_and_never_written_as_prose():
    chart = ChartBlock(chart_type='bar',
                       series=[ChartSeries(label='Spend', fact_refs=['spend.total'])])
    s = _section('s3', 'Spend by category', [chart, NarrativeBlock(text='It is concentrated.')])
    f = fit(s, ALL)
    assert f.layout_key == 't_chart'
    assert f.charts == {'chart1': chart}
    assert 'chart' not in ' '.join(f.text.values()).lower()


def test_a_table_goes_on_the_table_layout_with_its_columns_in_order():
    s = _section('s4', 'By area', [TableBlock(columns=['Area', 'Value'],
                                              rows=[['Network', 'spend.network']])])
    f = fit(s, ALL)
    assert f.layout_key == 't_table'
    cols, rows = f.tables['rows1']
    assert cols == ['Area', 'Value']
    assert rows == [['Network', '{{f:spend.network}}']]


def test_a_section_that_fits_no_layout_names_itself():
    """Review focus 1. A person must be told WHICH section had nowhere to go."""
    s = _section('s5', 'Six charts', [ChartBlock(chart_type='bar', series=[]),
                                      ChartBlock(chart_type='line', series=[])])
    with pytest.raises(NoFit) as caught:
        fit(s, ALL)
    assert caught.value.section_id == 's5'
    assert 'chart' in caught.value.reason


def test_more_blocks_than_slots_is_a_refusal_not_a_truncation():
    """Review focus 4. Nine metrics into a list that takes at most four."""
    s = _section('s6', 'Nine numbers',
                 [MetricBlock(fact_ref=f'm.{i}') for i in range(9)])
    with pytest.raises(NoFit) as caught:
        fit(s, ALL)
    assert caught.value.section_id == 's6'
    assert '4' in caught.value.reason          # the limit it could not meet


def test_a_title_longer_than_the_slot_allows_is_a_refusal():
    s = _section('s7', ' '.join(['word'] * 25), [NarrativeBlock(text='Short.')])
    with pytest.raises(NoFit):
        fit(s, ALL)


def test_prose_longer_than_the_slot_allows_is_a_refusal_not_a_trim():
    s = _section('s8', 'Long', [NarrativeBlock(text='x' * 400)])
    with pytest.raises(NoFit):
        fit(s, ALL)


def test_the_first_fitting_layout_in_catalogue_order_wins_so_a_rerun_draws_the_same_page():
    s = _section('s9', 'What we found', [NarrativeBlock(text='Short.')])
    assert fit(s, ALL).layout_key == 't_prose'
    assert fit(s, (TITLE_AND_CARDS,) + ALL).layout_key == 't_prose'   # cards cannot take prose


def test_a_layout_with_no_title_slot_is_never_chosen():
    titleless = LayoutOption(layout_key='t_none', name='No title', kind='template',
                             slots=(_text('prose1', max_chars=240),), region_ids=('prose1',))
    s = _section('s10', 'Needs a masthead', [NarrativeBlock(text='Short.')])
    with pytest.raises(NoFit):
        fit(s, (titleless,))


# -- the FindingListBlock ruling (Task 2, ledgered): refs are carried, not resolved here ------

def test_a_finding_list_claims_a_list_slot_and_carries_its_refs_for_the_renderer():
    s = _section('s11', 'What needs attention',
                 [FindingListBlock(finding_refs=['pk-1-PC001', 'pk-1-PC002'])])
    f = fit(s, ALL)
    assert f.layout_key == 't_cards'
    assert f.findings['cards1'] == ['pk-1-PC001', 'pk-1-PC002']
    assert f.lists == {}            # not resolved here: the fit has no Fact Pack


def test_metrics_and_findings_need_their_own_list_slots():
    """One list slot cannot hold both: they are resolved differently, and interleaving a raw
    finding id with a figure is how an id reaches a board paper."""
    s = _section('s12', 'Both',
                 [MetricBlock(fact_ref='spend.total'), MetricBlock(fact_ref='spend.other'),
                  FindingListBlock(finding_refs=['pk-1-PC001', 'pk-1-PC002'])])
    with pytest.raises(NoFit) as caught:
        fit(s, ALL)                 # every layout here has at most ONE list slot
    assert caught.value.section_id == 's12'


def test_two_list_slots_take_the_metrics_and_the_findings_separately():
    both = LayoutOption(layout_key='t_two_lists', name='Two lists', kind='template',
                        slots=(_text('title', max_words=19), _list('cards1'), _list('cards2')),
                        region_ids=('title', 'cards1', 'cards2'))
    s = _section('s13', 'Both',
                 [MetricBlock(fact_ref='spend.total'), MetricBlock(fact_ref='spend.other'),
                  FindingListBlock(finding_refs=['pk-1-PC001', 'pk-1-PC002'])])
    f = fit(s, (both,))
    assert f.lists['cards1'] == [('{{f:spend.total}}', 'spend.total'),
                                 ('{{f:spend.other}}', 'spend.other')]
    assert f.findings['cards2'] == ['pk-1-PC001', 'pk-1-PC002']
