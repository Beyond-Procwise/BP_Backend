"""The style-pack and layout contract, ported from the UI's own validators.

SOURCE OF TRUTH: beyond_procwise_ui/src/modules/SpendIQ/atb/validateStyle.js and
validateLayout.js. Both headers say "BP_Backend vendors the same rules", which is what this is.
`test_does_not_drift_from_the_javascript_contract` reads the constants back out of the JavaScript
and fails if the two disagree; it skips when the UI checkout is absent, naming the variable to set.

One addition beyond the JS (design §7): a region may state `grid` (a 12-column span) OR `box_in`
(measured inches), exactly one. Imported regions use `box_in`, because the reference deck is not
strictly on a twelve-column grid. The JS validator does not know about `box_in` yet — that is the
UI half of step 1 — so it accepts such a region and simply finds no grid to check.
"""
from __future__ import annotations

import re

STYLE_FORMATS = ('deck', 'a4-portrait')
REQUIRED_COLOURS = ('ink', 'muted', 'accent', 'panel')
REQUIRED_TYPE_SCALE = ('title', 'subtitle', 'body', 'table', 'footer')
COMPONENTS = (
    'title_block', 'kpi_card', 'kpi_card_row', 'kpi_stack', 'text_card', 'callout',
    'paragraph', 'bullet_panel', 'table', 'heatmap', 'chart', 'chevron_row',
    'arrow_map_row', 'timeline', 'gantt', 'raci', 'chip_row', 'source_footer',
    'page_number',
)
SLOT_TYPES = (
    'text', 'rich_text', 'fact_ref', 'list', 'table', 'rating', 'chart', 'chevrons',
    'timeline', 'gantt', 'raci', 'callout', 'chips', 'sources', 'appendix_rows',
)
FILL_MODES = ('agent', 'bind', 'auto', 'static')
PROSE_TYPES = ('text', 'rich_text', 'callout')

_COLOUR_RE = re.compile(
    r'^(#[0-9a-fA-F]{3}|#[0-9a-fA-F]{6}|#[0-9a-fA-F]{8}|rgba?\([^)]*\)'
    r'|transparent|inherit|currentColor)$')
_ID_RE = re.compile(r'^[a-z][a-z0-9_]*$')


class PackInvalid(Exception):
    """The derived pack or layout does not satisfy the contract. Carries the errors."""


def is_colour(value) -> bool:
    return isinstance(value, str) and bool(_COLOUR_RE.match(value.strip()))


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_pack(pack: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(pack, dict):
        return ['pack must be an object']

    # The two the first version of this port omitted. Both hand-authored packs start with them,
    # and without them the browser rejects every imported pack and falls back to the bundled ones
    # — an import that looks successful end to end and does nothing.
    if pack.get('kind') != 'atb_style':
        errors.append('kind must be "atb_style"')
    version = pack.get('schema_version')
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        errors.append('schema_version must be an integer >= 1')

    fmt = pack.get('format')
    if not isinstance(fmt, dict) or fmt.get('kind') not in STYLE_FORMATS:
        errors.append(f'format.kind must be one of {list(STYLE_FORMATS)}')
    elif fmt.get('kind') == 'deck':
        for key in ('width_in', 'height_in'):
            if not _is_num(fmt.get(key)) or fmt[key] <= 0:
                errors.append(f'format.{key} is required for a deck')

    colours = pack.get('colours')
    if not isinstance(colours, dict):
        errors.append('colours must be an object')
    else:
        for key in REQUIRED_COLOURS:
            if colours.get(key) is None:
                errors.append(f'colours.{key} is required')
            elif not is_colour(colours[key]):
                errors.append(f'colours.{key} "{colours[key]}" is not a usable CSS colour')

    scale = pack.get('type_scale_pt')
    if not isinstance(scale, dict):
        errors.append('type_scale_pt must be an object')
    else:
        for key in REQUIRED_TYPE_SCALE:
            if not _is_num(scale.get(key)) or scale[key] <= 0:
                errors.append(f'type_scale_pt.{key} must be a positive number')

    fonts = pack.get('fonts')
    if not isinstance(fonts, dict) or not fonts:
        errors.append('fonts must be a non-empty object')
    else:
        for role, font in fonts.items():
            if not isinstance(font, dict) or not font.get('family'):
                errors.append(f'fonts.{role}.family must be a non-empty string')
            elif not font.get('fallback'):
                errors.append(f'fonts.{role}.fallback is required — the named family may not '
                              'be installed')

    series = pack.get('series_palette')
    if not isinstance(series, list) or not series:
        errors.append('series_palette must be a non-empty array')
    else:
        errors += [f'series_palette[{i}] "{c}" is not a usable CSS colour'
                   for i, c in enumerate(series) if not is_colour(c)]

    scales = pack.get('rating_scales')
    if not isinstance(scales, dict):
        errors.append('rating_scales must be an object')
    else:
        for name, scale_def in scales.items():
            if not isinstance(scale_def, dict):
                errors.append(f'rating_scales.{name} must be an object')
                continue
            for label, chip in scale_def.items():
                if not isinstance(chip, dict):
                    errors.append(f'rating_scales.{name}.{label} must be an object')
                    continue
                for part in ('bg', 'ink'):
                    if part in chip and not is_colour(chip[part]):
                        errors.append(f'rating_scales.{name}.{label}.{part} '
                                      f'"{chip[part]}" is not a usable CSS colour')

    grid = pack.get('grid')
    if not isinstance(grid, dict) or grid.get('cols') != 12:
        errors.append('grid.cols must be 12')

    writing = pack.get('writing')
    if not isinstance(writing, dict) or not writing.get('locale'):
        errors.append('writing.locale is required — it decides the spelling list and number '
                      'formats')
    return errors


def validate_rating_scale(name: str, chips: dict) -> list[str]:
    """A hand-defined scale, held to the same rule as a measured one.

    Without this a scale went into the pack unchecked, so `{"High": {"bg": "navy-ish"}}` could be
    approved and would then fail validateStyle at render time — the outcome the design's §5a exists
    to avoid.
    """
    errors: list[str] = []
    if not isinstance(name, str) or not name.strip():
        errors.append('a rating scale needs a name')
    if not isinstance(chips, dict) or not chips:
        return errors + ['a rating scale needs at least one label']
    for label, chip in chips.items():
        if not isinstance(label, str) or not label.strip():
            errors.append('every label in a rating scale needs a name')
        if not isinstance(chip, dict):
            errors.append(f'rating_scales.{name}.{label} must be an object')
            continue
        if not chip.get('bg') and not chip.get('ink'):
            errors.append(f'rating_scales.{name}.{label} needs a bg or an ink')
        for part in ('bg', 'ink'):
            if part in chip and not is_colour(chip[part]):
                errors.append(f'rating_scales.{name}.{label}.{part} "{chip[part]}" is not a '
                              'usable CSS colour')
    return errors


def validate_layout(layout: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(layout, dict):
        return ['layout must be an object']

    if not isinstance(layout.get('id'), str) or not _ID_RE.match(layout.get('id') or ''):
        errors.append('id must be lower_snake_case and start with a letter')
    version = layout.get('version')
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        errors.append('version must be an integer >= 1')
    if not isinstance(layout.get('name'), str) or not layout.get('name'):
        errors.append('name must be a non-empty string')

    formats = layout.get('formats')
    if not isinstance(formats, list) or not formats:
        errors.append('formats must be a non-empty array')
    else:
        errors += [f'formats: unknown format "{f}"' for f in formats if f not in STYLE_FORMATS]

    regions = layout.get('regions')
    if not isinstance(regions, list) or not regions:
        errors.append('regions must be a non-empty array')
    else:
        seen: set[str] = set()
        for index, region in enumerate(regions):
            at = f'regions[{index}]'
            if not isinstance(region, dict):
                errors.append(f'{at} must be an object')
                continue
            region_id = region.get('id')
            if not isinstance(region_id, str) or not region_id:
                errors.append(f'{at}.id must be a non-empty string')
            elif region_id in seen:
                errors.append(f'{at}.id duplicates "{region_id}"')
            else:
                seen.add(region_id)
            if region.get('component') not in COMPONENTS:
                errors.append(f'{at}.component "{region.get("component")}" is not an '
                              'implemented component')
            has_grid, has_box = 'grid' in region, 'box_in' in region
            if has_grid and has_box:
                errors.append(f'{at} states both grid and box_in — exactly one')
            if has_grid:
                grid = region['grid']
                if not isinstance(grid, dict):
                    errors.append(f'{at}.grid must be an object')
                else:
                    if not _is_num(grid.get('col')):
                        errors.append(f'{at}.grid.col must be a number')
                    if not _is_num(grid.get('w')):
                        errors.append(f'{at}.grid.w must be a number')
                    if grid.get('row') is None:
                        errors.append(f'{at}.grid.row is required')
                    if _is_num(grid.get('col')) and _is_num(grid.get('w')) \
                            and grid['col'] + grid['w'] - 1 > 12:
                        errors.append(f'{at}.grid overflows the 12-column grid '
                                      f'(col {grid["col"]} + w {grid["w"]})')
            if has_box:
                box = region['box_in']
                if not isinstance(box, dict) or any(not _is_num(box.get(k))
                                                    for k in ('x', 'y', 'w', 'h')):
                    errors.append(f'{at}.box_in needs numeric x, y, w and h')

    slots = layout.get('slots')
    if not isinstance(slots, dict) or not slots:
        errors.append('slots must be a non-empty object')
    else:
        for name, slot in slots.items():
            at = f'slots.{name}'
            if not isinstance(slot, dict):
                errors.append(f'{at} must be an object')
                continue
            if slot.get('type') not in SLOT_TYPES:
                errors.append(f'{at}.type "{slot.get("type")}" is not a known slot type')
            if slot.get('fill') not in FILL_MODES:
                errors.append(f'{at}.fill "{slot.get("fill")}" is not a known fill mode')
            if slot.get('type') in PROSE_TYPES and slot.get('fill') == 'agent' \
                    and slot.get('max_words') is None and slot.get('max_chars') is None:
                errors.append(f'{at} is agent-written prose and must declare max_words '
                              'or max_chars')
            # The model never supplies chart data, so a chart slot is bound from facts.
            if slot.get('type') == 'chart' and slot.get('fill') != 'bind':
                errors.append(f'{at} is a chart and must be fill:"bind" — the model never '
                              'supplies chart data')
            if slot.get('type') == 'list':
                if not _is_num(slot.get('min')) or not _is_num(slot.get('max')):
                    errors.append(f'{at} is a list and must declare numeric min and max')
                if not isinstance(slot.get('item'), dict):
                    errors.append(f'{at} is a list and must declare an item shape')
            if slot.get('type') == 'table':
                columns = slot.get('columns')
                if not isinstance(columns, list) or not columns:
                    errors.append(f'{at} is a table and must declare columns')
                else:
                    for index, column in enumerate(columns):
                        if not isinstance(column, dict) or not column.get('type'):
                            errors.append(f'{at}.columns[{index}] must declare a type')
                if not _is_num(slot.get('max_rows')):
                    errors.append(f'{at} is a table and must declare max_rows so the planner '
                                  'can paginate')
            if slot.get('type') == 'rating' and not isinstance(slot.get('scale'), str):
                errors.append(f'{at} is a rating and must name the scale it uses')
    return errors
