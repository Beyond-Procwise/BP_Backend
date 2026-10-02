"""What each region is, and what goes in it.

A region's box is the MEDIAN of its members' boxes, in inches, because the reference deck is not
strictly on a twelve-column grid. A member more than LOOSE_FIT_IN off the median is reported with
its slide number rather than averaged away: a cluster that should have been two is then visible on
the review screen instead of silently blended.

A region whose members disagree about their content is `unresolved`, carrying both readings and
emitting nothing. The document-type resolver took the same ruling when two types tied — an honest
unresolved beats a coin flip.
"""
from __future__ import annotations

from statistics import median

from .cluster import Cluster
from .evidence import Evidence
from .read import Deck, Shape

LOOSE_FIT_IN = 0.15
_CHARS_PER_EM = 0.5
_LINE_HEIGHT = 1.35
_PT_PER_IN = 72
_CHARS_PER_WORD = 6.1
_TITLE_LINES = 2
_SUBTITLE_LINES = 2
_HEADING_LINES = 2


def _slug(text: str, fallback: str) -> str:
    out = ''.join(c.lower() if c.isalnum() else '_' for c in str(text)).strip('_')
    return out or fallback


def row_span(row: tuple[Shape, ...]) -> dict:
    """The box a whole row occupies — its union, not the median of its parts.

    A four-up card row's REGION is the band from the first card's left edge to the last card's
    right edge. Taking the median of the four cards' boxes returned something the size of one
    card, sitting in the middle of the row, and then reported each card as an outlier against it.
    """
    left = min(s.box.x for s in row)
    top = min(s.box.y for s in row)
    right = max(s.box.x + s.box.w for s in row)
    bottom = max(s.box.y + s.box.h for s in row)
    return {'x': round(left, 3), 'y': round(top, 3),
            'w': round(right - left, 3), 'h': round(bottom - top, 3)}


def _median_of(boxes: list[dict]) -> dict:
    return {key: round(median(b[key] for b in boxes), 3) for key in ('x', 'y', 'w', 'h')}


def _max_chars(width_in: float, height_in: float, size_pt: float, lines: int | None = None) -> int:
    per_line = max(1, int((width_in * _PT_PER_IN) / (size_pt * _CHARS_PER_EM)))
    if lines is None:
        lines = max(1, int((height_in * _PT_PER_IN) / (size_pt * _LINE_HEIGHT)))
    return per_line * max(1, lines)


def regions_and_slots(cluster: Cluster, deck: Deck, pack: dict, ev: Evidence):
    """-> (regions, slots, problems)"""
    scale = pack['type_scale_pt']
    grid = pack['grid']
    regions: list[dict] = []
    slots: dict[str, dict] = {}
    problems: list[dict] = []

    # Every slide in the reference deck has a title band, so every imported layout does.
    title_w = round(deck.width_in - 2 * grid['margin_in'], 3)
    regions.append({'id': 'title', 'component': 'title_block',
                    'box_in': {'x': grid['margin_in'], 'y': grid['title_top_in'],
                               'w': title_w, 'h': 0.6}})
    slots['title'] = {
        'type': 'text', 'fill': 'agent', 'required': True, 'style': 'assertion',
        'max_words': max(4, int(_max_chars(title_w, 0.6, scale['title'], _TITLE_LINES)
                                / _CHARS_PER_WORD)),
    }
    slots['subtitle'] = {
        'type': 'text', 'fill': 'agent', 'required': True, 'style': 'basis',
        'max_chars': _max_chars(title_w, 0.4, scale['subtitle'], _SUBTITLE_LINES),
    }

    members = cluster.rows_by_slide or (cluster.rows,)
    for index, row in enumerate(cluster.rows, 1):
        if not row:
            continue
        # The same row ACROSS the member slides. A member with fewer rows contributes nothing to
        # this one rather than shifting the median.
        spans = [(cluster.slides[i] if i < len(cluster.slides) else 0, row_span(member[index - 1]))
                 for i, member in enumerate(members) if len(member) >= index and member[index - 1]]
        box = _median_of([span for _, span in spans]) if spans else row_span(row)
        for slide_no, span in spans:
            off = max(abs(span[key] - box[key]) for key in ('x', 'y', 'w', 'h'))
            if off > LOOSE_FIT_IN:
                problems.append({
                    'kind': 'loose_fit', 'region': f'row{index}',
                    'why': f'slide {slide_no} is {round(off, 2)}in off the median box, more '
                           f'than the {LOOSE_FIT_IN}in tolerance',
                    'slides': [slide_no],
                })

        content = {s.kind for s in row} & {'table', 'chart'}
        if len(content) > 1:
            problems.append({
                'kind': 'unresolved', 'region': f'row{index}',
                'why': f'the members disagree about this region: {sorted(content)}',
                'slides': sorted({s.slide for s in row}),
            })
            continue

        if 'table' in content:
            table = next((s.table for s in row if s.table), None)
            header = list(table[0]) if table else []
            rows_in_members = [len(s.table) - 1 for s in row if s.table]
            per_column = box['w'] / max(1, len(header))
            region_id = f'rows{index}'
            regions.append({'id': region_id, 'component': 'table', 'box_in': box})
            slots[region_id] = {
                'type': 'table', 'fill': 'agent',
                'max_rows': max(rows_in_members) if rows_in_members else 1,
                'columns': [{'id': _slug(name, f'col{i}'), 'label': name, 'type': 'text',
                             'max_chars': _max_chars(per_column, 0.3, scale['table'], 2)}
                            for i, name in enumerate(header)],
            }
            continue

        if 'chart' in content:
            # `chart` is a legal component in the contract but has no renderer yet, so the
            # geometry is kept for step 2 and the gap is declared rather than left to be noticed.
            region_id = f'chart{index}'
            regions.append({'id': region_id, 'component': 'chart', 'box_in': box})
            slots[region_id] = {'type': 'chart', 'fill': 'bind'}
            problems.append({
                'kind': 'unresolved', 'region': region_id,
                'why': 'the region is a chart and there is no chart renderer yet, so it will '
                       'draw nothing until one exists',
                'slides': sorted({s.slide for s in row}),
            })
            continue

        columns = len({round(s.box.x, 1) for s in row})
        if columns > 1:
            region_id = f'cards{index}'
            per_card = box['w'] / columns
            regions.append({'id': region_id, 'component': 'text_card', 'box_in': box})
            slots[region_id] = {
                'type': 'list', 'fill': 'agent', 'min': 2, 'max': columns,
                'item': {
                    'heading': {'type': 'text',
                                'max_chars': _max_chars(per_card, 0.4,
                                                        scale.get('card_title', scale['body']),
                                                        _HEADING_LINES)},
                    'body': {'type': 'text',
                             'max_chars': _max_chars(per_card, box['h'], scale['body'])},
                },
            }
        else:
            region_id = f'prose{index}'
            regions.append({'id': region_id, 'component': 'paragraph', 'box_in': box})
            slots[region_id] = {'type': 'text', 'fill': 'agent',
                                'max_chars': _max_chars(box['w'], box['h'], scale['body'])}

    regions.append({'id': 'footer', 'component': 'source_footer'})
    slots['sources'] = {'type': 'sources', 'fill': 'auto'}
    ev.record(f'layout.{cluster.signature}',
              {'regions': len(regions), 'problems': len(problems)},
              slides=list(cluster.slides))
    return regions, slots, problems
