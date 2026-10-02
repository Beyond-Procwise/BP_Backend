"""Where things repeatedly sit.

`cols` is always 12 — it is the authoring convenience the hand-written layouts use, and the design
(§5) records that the reference deck is NOT strictly on twelve columns: 4-up cards are 2.9in,
3-up 3.71in, 5-up 2.45in, and several widths do not land on a span. That is why an imported region
carries measured inches instead (§7).

DECORATIONS ARE EXCLUDED from every census here. A 0.32in chapter chip sits in the title band two
to a slide on some decks, and counted naively its 0.47in top edge outvotes the title's 0.35in —
which would put the title of every imported layout a sixth of an inch too low.
"""
from __future__ import annotations

from collections import Counter

from .evidence import Evidence
from .read import Deck, Shape

COLS = 12
_DEFAULT_MARGIN = 0.5
_DEFAULT_GUTTER = 0.3
_SAME_ROW_IN = 0.45
_CHIP_MAX_IN = 0.6
_MIN_W_IN = 0.5          # anything narrower is a decoration, not a measurement
_MIN_H_IN = 0.15


def _measurable(shape: Shape) -> bool:
    return shape.box.w >= _MIN_W_IN and shape.box.h >= _MIN_H_IN


def _modal(values: list[float], default: float) -> tuple[float, int]:
    if not values:
        return default, 0
    counts = Counter(round(v, 2) for v in values)
    best = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    return best[0], best[1]


def grid(deck: Deck, ev: Evidence) -> dict:
    lefts: list[float] = []
    widths: list[float] = []
    top_band: list[float] = []
    mid_band: list[float] = []
    bottom_band: list[float] = []
    gutters: list[float] = []
    bottom_from = deck.height_in * 0.85

    for slide in deck.slides:
        shapes = [s for s in slide if _measurable(s)]
        for shape in shapes:
            lefts.append(shape.box.x)
            widths.append(shape.box.w)
            if shape.box.y < 1.0:
                top_band.append(shape.box.y)
            elif shape.box.y > bottom_from:
                bottom_band.append(shape.box.y)
            else:
                mid_band.append(shape.box.y)
        rows: dict[float, list[Shape]] = {}
        for shape in shapes:
            key = next((k for k in rows if abs(k - shape.box.y) <= _SAME_ROW_IN), shape.box.y)
            rows.setdefault(key, []).append(shape)
        for row in rows.values():
            row.sort(key=lambda s: s.box.x)
            for left, right in zip(row, row[1:]):
                gap = round(right.box.x - (left.box.x + left.box.w), 2)
                if 0.05 <= gap <= 1.0:
                    gutters.append(gap)

    margin, margin_n = _modal(lefts, _DEFAULT_MARGIN)
    gutter, gutter_n = _modal(gutters, _DEFAULT_GUTTER)
    title_top, title_n = _modal(top_band, 0.35)
    body_top, body_n = _modal(mid_band, 1.5)
    footer_top, footer_n = _modal(bottom_band, round(deck.height_in - 0.48, 2))
    # DERIVED, not measured. The modal width is a CARD width — four cards a slide outnumber one
    # full-width title — so taking the mode gives 2.86in on the reference deck's own geometry and
    # only looked right there because its full-width titles happen to be numerous. The
    # measurement's job is to corroborate: how many shapes actually span it.
    content_w = round(deck.width_in - 2 * margin, 2)
    content_n = sum(1 for w in widths if abs(w - content_w) <= 0.05)

    ev.record('grid.margin_in', margin, shapes=margin_n, why='modal left edge')
    ev.record('grid.gutter_in', gutter, gaps=gutter_n, why='modal gap between shapes in a row')
    ev.record('grid.title_top_in', title_top, shapes=title_n,
              why='modal top edge above 1in, decorations excluded')
    ev.record('grid.body_top_in', body_top, shapes=body_n, why='modal top edge in the body band')
    ev.record('grid.footer_top_in', footer_top, shapes=footer_n,
              why='modal top edge in the bottom band')
    ev.record('grid.content_width_in', content_w, shapes=content_n,
              why=f'the page ({deck.width_in}in) less two {margin}in margins is {content_w}in; '
                  f'{content_n} shapes span it'
                  + ('' if content_n else ' — NOTHING in this deck does, so the margin may be wrong'))
    return {
        'cols': COLS,
        'margin_in': margin,
        'gutter_in': gutter,
        'title_top_in': title_top,
        'body_top_in': body_top,
        'footer_top_in': footer_top,
    }


def chapter_chip_in(deck: Deck, ev: Evidence) -> float | None:
    """The small square that marks a chapter. Its own filter is the opposite of _measurable: this
    is the one place a decoration IS the measurement."""
    squares: list[float] = []
    for slide in deck.slides:
        for shape in slide:
            w, h = shape.box.w, shape.box.h
            if 0 < w <= _CHIP_MAX_IN and 0 < h <= _CHIP_MAX_IN and abs(w - h) <= 0.05 \
                    and shape.box.y < 1.0:
                squares.append(round(w, 2))
    if not squares:
        return None
    size, n = _modal(squares, 0.32)
    ev.record('chapter_chip_in', size, shapes=n, why='modal square in the title band')
    return size
