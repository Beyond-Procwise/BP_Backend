"""Assemble the measurements into a pack the existing validators accept."""
from __future__ import annotations

from . import derived
from . import grid as grid_module
from . import palette, typescale
from .contract import PackInvalid, validate_pack
from .evidence import Evidence
from .read import Deck


def build_pack(deck: Deck, ev: Evidence, key: str, name: str) -> dict:
    colours = palette.colours(deck, ev)
    scale = typescale.type_scale(deck, ev)
    fonts = typescale.fonts(deck, ev)
    grid = grid_module.grid(deck, ev)
    chip = grid_module.chapter_chip_in(deck, ev)

    is_deck = deck.width_in > deck.height_in
    fmt: dict = {'kind': 'deck' if is_deck else 'a4-portrait'}
    if is_deck:
        fmt['width_in'] = deck.width_in
        fmt['height_in'] = deck.height_in
    if deck.theme.get('colours'):
        ev.ignored('theme.colours', deck.theme['colours'],
                   'the deck paints its own shapes; a theme colour is not what it looks like')

    pack = {
        'key': key,
        'name': name,
        'format': fmt,
        'colours': colours,
        'type_scale_pt': scale,
        'fonts': fonts,
        'grid': grid,
        'series_palette': derived.series_palette(deck, ev, colours),
        'rating_scales': derived.rating_scales(deck, ev),
        'writing': derived.writing(deck, ev),
    }
    if chip:
        pack['chapter_chip_in'] = chip

    errors = validate_pack(pack)
    if errors:
        # Never stored, never served: a half-valid pack renders a page with a missing colour and
        # nothing downstream would say why.
        raise PackInvalid('; '.join(errors))
    return pack
