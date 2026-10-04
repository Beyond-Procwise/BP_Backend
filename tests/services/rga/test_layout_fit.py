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
