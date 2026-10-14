from src.services.atb.pptx_import.contract import validate_pack
from src.services.atb.pptx_import.emit import build_pack
from src.services.atb.pptx_import.evidence import Evidence
from src.services.atb.pptx_import.read import read_deck

# As the reference deck is: the ink sets the titles AND the body (804 runs of #172033 there), and
# only the quieter things are muted. A fixture where the body colour outnumbers the ink makes the
# body colour the ink, correctly — and says nothing about the code.
SLIDE = [(0.5, 0.35, 12.33, 0.6, 'A page title', 30, '#172033'),
         (0.5, 1.5, 6.0, 1.0, 'Body text here', 11.5, '#172033'),
         (0.5, 3.0, 6.0, 1.0, 'More body text', 11.5, '#172033'),
         (0.5, 4.5, 6.0, 0.5, 'a quiet caption', 10, '#56627A'),
         (0.5, 7.02, 11.73, 0.3, 'a footnote', 9, '#56627A')]


def test_a_pack_built_from_a_deck_passes_the_contract(build_deck):
    ev = Evidence()
    pack = build_pack(read_deck(build_deck([SLIDE] * 12)), ev, key='sample', name='Sample.pptx')
    assert validate_pack(pack) == []
    assert pack['format'] == {'kind': 'deck', 'width_in': 13.333, 'height_in': 7.5}
    assert pack['key'] == 'sample'


def test_a_portrait_deck_is_a4_portrait_and_states_no_page_size(build_deck):
    # Review Focus 5: kind follows the aspect, and an a4-portrait pack states no width/height.
    ev = Evidence()
    portrait = [(0.59, 0.59, 7.09, 0.5, 'Title', 16, '#172033'),
                (0.59, 1.4, 7.09, 1.0, 'Body', 10, '#172033'),
                (0.59, 10.9, 7.09, 0.3, 'note', 8, '#56627A')]
    pack = build_pack(read_deck(build_deck([portrait] * 12, width_in=8.27, height_in=11.69)),
                      ev, key='p', name='Portrait.pptx')
    assert pack['format'] == {'kind': 'a4-portrait'}
    assert validate_pack(pack) == []


def test_the_evidence_travels_with_the_pack(build_deck):
    ev = Evidence()
    build_pack(read_deck(build_deck([SLIDE] * 12)), ev, key='k', name='K.pptx')
    values = ev.as_dict()['values']
    assert values['colours.ink']['value'] == '#172033'
    assert values['colours.ink']['runs'] >= 12
    assert values['grid.margin_in']['value'] == 0.5
    assert values['type_scale_pt.title']['value'] == 30.0


def test_the_theme_colours_are_recorded_as_ignored(build_deck):
    ev = Evidence()
    build_pack(read_deck(build_deck([SLIDE] * 12)), ev, key='k', name='K.pptx')
    ignored = ev.as_dict()['ignored']
    assert 'theme.colours' in ignored
    assert 'not what it looks like' in ignored['theme.colours']['why']


def test_an_invalid_pack_raises_rather_than_being_returned(build_deck, monkeypatch):
    # No real .pptx produces an invalid pack — the fallbacks see to that — so the guarantee under
    # test is that the contract failure PROPAGATES. One measurement is made to return nothing.
    import pytest

    from src.services.atb.pptx_import import palette
    from src.services.atb.pptx_import.contract import PackInvalid

    monkeypatch.setattr(palette, 'colours', lambda deck, ev, floor=10: {})
    with pytest.raises(PackInvalid) as exc:
        build_pack(read_deck(build_deck([SLIDE] * 12)), Evidence(), key='k', name='K.pptx')
    assert 'colours.ink is required' in str(exc.value)


def test_a_deck_of_one_empty_slide_still_yields_a_valid_pack(build_deck):
    # The fallbacks exist for exactly this: a pack must validate or nothing downstream works.
    pack = build_pack(read_deck(build_deck([[(0.5, 1.5, 3.0, 0.4, 'x', 12, '#172033')]])),
                      Evidence(), key='k', name='K.pptx')
    assert validate_pack(pack) == []
