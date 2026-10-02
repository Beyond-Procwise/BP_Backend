"""Assemble the measurements into a pack the existing validators accept."""
from __future__ import annotations

import hashlib

from . import derived
from . import grid as grid_module
from . import palette, typescale
from .cluster import EMPTY_ROW, Cluster
from .contract import PackInvalid, validate_layout, validate_pack
from .evidence import Evidence
from .example import example_fill
from .read import Deck
from .slots import regions_and_slots


def build_pack(deck: Deck, ev: Evidence, key: str, name: str) -> dict:
    colours = palette.colours(deck, ev)
    scale = typescale.type_scale(deck, ev)
    fonts = typescale.fonts(deck, ev, title_pt=scale.get('title'))
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
        # The shape the hand-authored packs use; without these two the browser's validateStyle
        # rejects the pack and silently falls back to the bundled layouts.
        'kind': 'atb_style',
        'schema_version': 1,
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


def proposed_name(cluster: Cluster) -> str:
    """A geometric description, never a name. Design §9.2: naming is the human's."""
    if cluster.signature == (EMPTY_ROW,):
        return 'title only'
    parts = []
    for row in cluster.signature:
        columns, kind = row[0], row[1]
        repeat = ' repeating' if len(row) > 2 else ''
        word = {'TABLE': 'table', 'CHART': 'chart'}.get(kind,
                                                        'cards' if columns > 1 else 'panel')
        parts.append(f'{columns}-up {word}{repeat}' if columns > 1
                     else f'full-width {word}{repeat}')
    return ' + '.join(parts)


def layout_key(cluster: Cluster, pack_key: str = '') -> str:
    """Stable across runs, and lower_snake_case as the contract requires.

    Derived from the signature AND the pack it belongs to, rather than from a counter or a uuid:
    importing the same deck twice has to produce the same ids (an acceptance criterion), but two
    DIFFERENT packs with the same structure must not collide — the browser keys its layout
    registry by this id, so two approved packs would otherwise offer one picker entry and render
    the other pack's geometry.
    """
    digest = hashlib.sha256(f'{pack_key}|{cluster.signature!r}'.encode()).hexdigest()[:10]
    return f'imported_{digest}'


def build_layout(cluster: Cluster, deck: Deck, pack: dict, ev: Evidence, filename: str) -> dict:
    regions, slots, problems = regions_and_slots(cluster, deck, pack, ev)
    fill, source = example_fill(cluster, deck, slots)
    name = proposed_name(cluster)
    layout = {
        'id': layout_key(cluster, pack.get('key', '')),
        'version': 1,
        # The contract requires a non-empty `name`. It starts as the proposed one so the layout is
        # valid from the moment it is built; `proposed_name` is kept beside it so the review screen
        # can show what the importer suggested against what the human called it.
        'name': name,
        'proposed_name': name,
        'formats': [pack['format']['kind']],
        'regions': regions,
        'slots': slots,
        'slide_refs': list(cluster.slides),
        'example_fill': fill,
        'example_source': {'file': filename, **source},
        'problems': problems,
        'writing_guidance': '',
        'pagination': None,
    }
    errors = validate_layout(layout)
    if errors:
        raise PackInvalid(f'layout {layout["id"]}: ' + '; '.join(errors))
    return layout
