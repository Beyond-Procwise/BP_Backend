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
_MIN_REGION_H_IN = 0.2
_ROW_GAP_IN = 0.05


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


def _rows_per_signature_row(cluster: Cluster) -> list[list[int]]:
    """Which of the representative slide's rows each signature row stands for.

    All 1:1, except a signature row marked `repeat`, which stands for itself and every row after
    it — that is what the collapse in `cluster.signature` means.
    """
    rows = list(range(len(cluster.rows)))
    plan: list[list[int]] = []
    for position, entry in enumerate(cluster.signature):
        if len(entry) > 2 and entry[2] == 'repeat':
            plan.append(rows[position:] or [position])
            return plan
        if position < len(rows):
            plan.append([position])
    return plan or [[i] for i in rows]


def _stack_and_check(regions: list[dict], problems: list[dict], deck: Deck, pack: dict,
                     cluster: Cluster) -> None:
    """A band ends where the next band begins, and whatever is left must fit on the page.

    A row's span is the union of its shapes, so one tall panel beside a stack of small cards gave a
    region that swallowed every row below it: 7 of the reference deck's 8 layouts had overlapping
    regions and two crossed the footer line — acceptance criterion 3 failed by this module's own
    output before the browser ever saw it. Rows are stacked in document order, so clipping each to
    the next one's top is the geometry, not a patch. Anything still wrong is reported rather than
    drawn on top of its neighbour.
    """
    body = [r for r in regions if 'box_in' in r and r['id'] != 'title']
    for current, following in zip(body, body[1:]):
        box, below = current['box_in'], following['box_in']
        if box['y'] + box['h'] > below['y'] and below['y'] > box['y']:
            box['h'] = round(max(_MIN_REGION_H_IN, below['y'] - box['y'] - _ROW_GAP_IN), 3)

    margin = pack['grid']['margin_in']
    footer_line = pack['grid']['footer_top_in']
    right_edge = deck.width_in - margin
    for region in body:
        box = region['box_in']
        if round(box['x'] + box['w'], 2) > round(right_edge, 2) + 0.05:
            problems.append({'kind': 'off_page', 'region': region['id'],
                             'why': f'ends at {round(box["x"] + box["w"], 2)}in, past the '
                                    f'{round(right_edge, 2)}in right margin',
                             'slides': list(cluster.slides)})
        if round(box['y'] + box['h'], 2) > round(footer_line, 2) + 0.05:
            problems.append({'kind': 'into_footer', 'region': region['id'],
                             'why': f'ends at {round(box["y"] + box["h"], 2)}in, below the '
                                    f'{footer_line}in footer line',
                             'slides': list(cluster.slides)})
    for index, first in enumerate(body):
        for second in body[index + 1:]:
            a, b = first['box_in'], second['box_in']
            if a['x'] < b['x'] + b['w'] - 0.01 and b['x'] < a['x'] + a['w'] - 0.01 \
                    and a['y'] < b['y'] + b['h'] - 0.01 and b['y'] < a['y'] + a['h'] - 0.01:
                problems.append({'kind': 'overlap', 'region': f'{first["id"]}~{second["id"]}',
                                 'why': 'the two regions cover the same part of the page',
                                 'slides': list(cluster.slides)})


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
    # Walk the SIGNATURE, not the representative slide's rows. The signature collapses a trailing
    # repeated row into one marked `repeat`; iterating the raw rows emitted six card regions copied
    # from whichever slide happened to be first, which a renderer cannot know to repeat and a human
    # approves without knowing what they are approving.
    plan = _rows_per_signature_row(cluster)
    for index, row_indices in enumerate(plan, 1):
        row = tuple(shape for i in row_indices for shape in cluster.rows[i])
        repeats = len(row_indices) > 1
        if not row:
            continue
        # The same row ACROSS the member slides. A member with fewer rows contributes nothing to
        # this one rather than shifting the median.
        spans = []
        for i, member in enumerate(members):
            mine = [member[j] for j in row_indices if j < len(member) and member[j]]
            if not mine:
                continue
            flattened = tuple(shape for part in mine for shape in part)
            spans.append((cluster.slides[i] if i < len(cluster.slides) else 0,
                          row_span(flattened)))
        box = _median_of([span for _, span in spans]) if spans else row_span(row)
        for slide_no, span in spans:
            # x, y and w only. A row's HEIGHT is its content — a table's height is its row count —
            # so comparing it reported `slide 4 is 3.8in off` for the 21-slide table layout and
            # made 41 of the reference deck's 45 problems artefacts. The height RANGE is recorded
            # on the region instead.
            off = max(abs(span[key] - box[key]) for key in ('x', 'y', 'w'))
            if off > LOOSE_FIT_IN:
                problems.append({
                    'kind': 'loose_fit', 'region': f'row{index}',
                    'why': f'slide {slide_no} is {round(off, 2)}in off the median box, more '
                           f'than the {LOOSE_FIT_IN}in tolerance',
                    'slides': [slide_no],
                })

        heights = sorted({round(span['h'], 2) for _, span in spans})
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
            regions.append({'id': region_id, 'component': 'table', 'box_in': box,
                            **({'height_range_in': heights} if len(heights) > 1 else {})})
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
            regions.append({'id': region_id, 'component': 'text_card', 'box_in': box,
                            **({'repeat_over': region_id} if repeats else {}),
                            **({'height_range_in': heights} if len(heights) > 1 else {})})
            slots[region_id] = {
                'type': 'list', 'fill': 'agent', 'min': 2,
                'max': columns * len(row_indices) if repeats else columns,
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

    _stack_and_check(regions, problems, deck, pack, cluster)
    regions.append({'id': 'footer', 'component': 'source_footer'})
    slots['sources'] = {'type': 'sources', 'fill': 'auto'}
    ev.record(f'layout.{cluster.signature}',
              {'regions': len(regions), 'problems': len(problems)},
              slides=list(cluster.slides))
    return regions, slots, problems
