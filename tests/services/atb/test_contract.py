import os
import re

import pytest

from src.services.atb.pptx_import.contract import (
    COMPONENTS, FILL_MODES, PROSE_TYPES, REQUIRED_COLOURS, REQUIRED_TYPE_SCALE, SLOT_TYPES,
    STYLE_FORMATS, validate_layout, validate_pack)

UI = os.environ.get('BEYOND_PROCWISE_UI',
                    os.path.expanduser('~/PycharmProjects/beyond_procwise_ui'))
STYLE_JS = os.path.join(UI, 'src/modules/SpendIQ/atb/validateStyle.js')
LAYOUT_JS = os.path.join(UI, 'src/modules/SpendIQ/atb/validateLayout.js')


def _pack(**over):
    pack = {'kind': 'atb_style', 'schema_version': 1, 'key': 'k', 'name': 'K',
            'format': {'kind': 'deck', 'width_in': 13.333, 'height_in': 7.5},
            'colours': {c: '#172033' for c in REQUIRED_COLOURS},
            'type_scale_pt': {r: 10 for r in REQUIRED_TYPE_SCALE},
            'fonts': {'body': {'family': 'Calibri', 'fallback': 'Arial, sans-serif'}},
            'series_palette': ['#172033'], 'rating_scales': {}, 'grid': {'cols': 12},
            'writing': {'locale': 'en-GB'}}
    pack.update(over)
    return pack


def _layout(**over):
    layout = {'id': 'imported_abc123', 'version': 1, 'name': '4-up cards',
              'formats': ['deck'],
              'regions': [{'id': 'title', 'component': 'title_block',
                           'box_in': {'x': 0.5, 'y': 0.35, 'w': 12.33, 'h': 0.6}}],
              'slots': {'title': {'type': 'text', 'fill': 'agent', 'max_words': 12}}}
    layout.update(over)
    return layout


def test_a_minimal_pack_is_valid():
    assert validate_pack(_pack()) == []


def test_rejects_a_pack_missing_a_required_colour():
    errors = validate_pack(_pack(colours={'ink': '#172033'}))
    assert any('muted' in e for e in errors)


def test_rejects_an_empty_series_palette():
    assert any('series_palette' in e for e in validate_pack(_pack(series_palette=[])))


def test_rejects_a_font_with_no_fallback():
    errors = validate_pack(_pack(fonts={'body': {'family': 'Calibri'}}))
    assert any('fallback' in e for e in errors)


def test_rejects_a_deck_with_no_page_size():
    assert any('width_in' in e for e in validate_pack(_pack(format={'kind': 'deck'})))


def test_an_a4_pack_needs_no_page_size():
    assert validate_pack(_pack(format={'kind': 'a4-portrait'})) == []


def test_a_minimal_layout_is_valid():
    assert validate_layout(_layout()) == []


def test_rejects_a_layout_with_no_name():
    assert any('name' in e for e in validate_layout(_layout(name='')))


def test_rejects_an_id_that_is_not_lower_snake_case():
    assert any('lower_snake_case' in e for e in validate_layout(_layout(id='Imported-ABC')))


def test_rejects_an_unknown_component():
    bad = _layout(regions=[{'id': 'x', 'component': 'hologram', 'box_in':
                            {'x': 0, 'y': 0, 'w': 1, 'h': 1}}])
    assert any('hologram' in e for e in validate_layout(bad))


def test_accepts_chart_which_the_ui_contract_allows():
    ok = _layout(regions=[{'id': 'c', 'component': 'chart',
                           'box_in': {'x': 0.5, 'y': 1.5, 'w': 6, 'h': 3}}],
                 slots={'c': {'type': 'chart', 'fill': 'bind'}})
    assert validate_layout(ok) == []


def test_rejects_a_region_stating_both_a_grid_and_a_box():
    bad = _layout(regions=[{'id': 'x', 'component': 'paragraph',
                            'grid': {'col': 1, 'row': 1, 'w': 12},
                            'box_in': {'x': 0, 'y': 0, 'w': 1, 'h': 1}}])
    assert any('exactly one' in e for e in validate_layout(bad))


def test_rejects_duplicate_region_ids():
    region = {'id': 'same', 'component': 'paragraph', 'box_in': {'x': 0, 'y': 0, 'w': 1, 'h': 1}}
    assert any('duplicates' in e for e in validate_layout(_layout(regions=[region, dict(region)])))


def test_rejects_agent_prose_with_no_length():
    bad = _layout(slots={'title': {'type': 'text', 'fill': 'agent'}})
    assert any('max_words or max_chars' in e for e in validate_layout(bad))


def test_a_bound_slot_needs_no_length():
    ok = _layout(slots={'figure': {'type': 'chart', 'fill': 'bind'}})
    assert validate_layout(ok) == []


@pytest.mark.skipif(not os.path.exists(STYLE_JS) or not os.path.exists(LAYOUT_JS),
                    reason='set BEYOND_PROCWISE_UI to the UI checkout to run the drift check')
def test_does_not_drift_from_the_javascript_contract():
    """The JS files are the original — their own headers say BP_Backend vendors these rules. This
    reads the constants back out of them, so the two cannot disagree silently."""
    def arrays(path):
        source = open(path, encoding='utf-8').read()

        def arr(name):
            match = re.search(name + r"\s*=\s*\[([^\]]*)\]", source)
            assert match, f'{name} not found in {os.path.basename(path)}'
            return [v.strip().strip("'\"") for v in match.group(1).split(',') if v.strip()]
        return arr

    style_arr = arrays(STYLE_JS)
    layout_arr = arrays(LAYOUT_JS)
    assert style_arr('REQUIRED_COLOURS') == list(REQUIRED_COLOURS)
    assert style_arr('REQUIRED_TYPE_SCALE') == list(REQUIRED_TYPE_SCALE)
    assert style_arr('STYLE_FORMATS') == list(STYLE_FORMATS)
    assert layout_arr('COMPONENTS') == list(COMPONENTS)
    assert layout_arr('SLOT_TYPES') == list(SLOT_TYPES)
    assert layout_arr('FILL_MODES') == list(FILL_MODES)
    assert layout_arr('PROSE_TYPES') == list(PROSE_TYPES)


# ---------------------------------------------------------------- review findings C2 and I14
# The vendored contract was missing five rules the JavaScript enforces. Every one of them is a
# layout or pack that passes here, is stored, and is then rejected in the browser — so the review
# screen cannot draw the card it is meant to approve.

def test_a_pack_must_say_what_kind_of_document_it_is():
    assert any('atb_style' in e for e in validate_pack(_pack(kind=None)))
    assert any('schema_version' in e for e in validate_pack(_pack(schema_version=0)))


def test_the_shipped_pack_shape_is_what_the_hand_authored_packs_use():
    import json
    import os
    authored = os.path.join(UI, 'src/modules/SpendIQ/atb/styles/consulting-navy-16x9.json')
    if not os.path.exists(authored):
        import pytest
        pytest.skip('set BEYOND_PROCWISE_UI to the UI checkout')
    pack = json.load(open(authored, encoding='utf-8'))
    assert validate_pack(pack) == [], 'the vendored contract must accept a pack the UI ships'


def test_a_chart_slot_must_be_bound():
    bad = _layout(regions=[{'id': 'c', 'component': 'chart',
                            'box_in': {'x': 0.5, 'y': 1.5, 'w': 6, 'h': 3}}],
                  slots={'c': {'type': 'chart', 'fill': 'agent'}})
    assert any('fill:"bind"' in e for e in validate_layout(bad))


def test_a_list_slot_must_declare_min_max_and_an_item():
    bad = _layout(slots={'cards': {'type': 'list', 'fill': 'agent'}})
    errors = validate_layout(bad)
    assert any('numeric min and max' in e for e in errors)
    assert any('item shape' in e for e in errors)


def test_a_table_slot_must_declare_columns_with_types_and_max_rows():
    bad = _layout(slots={'rows': {'type': 'table', 'fill': 'agent', 'columns': []}})
    errors = validate_layout(bad)
    assert any('must declare columns' in e for e in errors)
    assert any('max_rows' in e for e in errors)
    untyped = _layout(slots={'rows': {'type': 'table', 'fill': 'agent', 'max_rows': 8,
                                      'columns': [{'id': 'a'}]}})
    assert any('must declare a type' in e for e in validate_layout(untyped))


def test_a_rating_slot_must_name_its_scale():
    bad = _layout(slots={'risk': {'type': 'rating', 'fill': 'agent'}})
    assert any('name the scale' in e for e in validate_layout(bad))
