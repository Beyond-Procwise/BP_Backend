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
_SAME_WIDTH_IN = 0.1   # two shapes this close in width are one column pitch
# The footer band as a FRACTION of the page. 0.92 of a 7.5in deck is 6.9in, which sits between
# the reference deck's lowest body row (6.6in) and its footnote line (7.02in); on an 11.69in
# portrait sheet the same fraction is 10.75in, just above its 10.9in footnote. At 0.85 the band
# swallowed body rows at 6.45in and reported those as the footer. cluster.py shares this.
FOOTER_BAND_FRACTION = 0.92


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
    bottom_from = deck.height_in * FOOTER_BAND_FRACTION
    wide_enough = (deck.width_in - 2 * _DEFAULT_MARGIN) * 0.5

    for slide in deck.slides:
        shapes = [s for s in slide if _measurable(s)]
        for shape in shapes:
            lefts.append(shape.box.x)
            widths.append(shape.box.w)
            if shape.box.y < 1.0:
                top_band.append(shape.box.y)
            elif shape.box.y > bottom_from:
                # The footer line is where footer CONTENT sits, not where the page number sits.
                # The reference deck's page-number placeholder is at 7.05in on 84 slides and its
                # footnote at 7.02in; taking the mode of everything in the band gives the page
                # number's edge. Only shapes spanning at least half the content width count.
                if shape.box.w >= wide_enough:
                    bottom_band.append(shape.box.y)
            else:
                mid_band.append(shape.box.y)
        rows: dict[float, list[Shape]] = {}
        for shape in shapes:
            key = next((k for k in rows if abs(k - shape.box.y) <= _SAME_ROW_IN), shape.box.y)
            rows.setdefault(key, []).append(shape)
        for row in rows.values():
            row.sort(key=lambda s: s.box.x)
            # Only between shapes of EQUAL WIDTH. A gutter is the gap in a multi-column band; the
            # modal gap between ANY two adjacent shapes is 0.1in on the reference deck, because a
            # deck is full of tightly-packed unequal things — the same error as taking the modal
            # width for the content width.
            for left, right in zip(row, row[1:]):
                if abs(left.box.w - right.box.w) > _SAME_WIDTH_IN:
                    continue
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
    # THE DECK MAY NOT HAVE ONE GUTTER. The reference deck's equal-width bands sit at 0.1, 0.15,
    # 0.3, 0.45, 0.65 and 0.8in, which is the same finding as the grid not being strictly twelve
    # columns: each band divides the content width its own way. The modal value is reported as the
    # measurement and the whole distribution goes in the evidence, so nobody reads a single number
    # as a design rule the deck does not follow. The brief's 0.3in is the deck's fourth most
    # common gap, not its gutter.
    spread = sorted(Counter(round(g, 2) for g in gutters).items(), key=lambda kv: -kv[1])
    ev.record('grid.gutter_in', gutter, gaps=gutter_n,
              why='modal gap between two shapes of equal width in a row'
                  + ('; THE DECK USES SEVERAL: ' + ', '.join(f'{v}in x{n}' for v, n in spread[:6])
                     if len(spread) > 1 else ''),
              distribution=[[v, n] for v, n in spread])
    ev.record('grid.title_top_in', title_top, shapes=title_n,
              why='modal top edge above 1in, decorations excluded')
    ev.record('grid.body_top_in', body_top, shapes=body_n, why='modal top edge in the body band')
    ev.record('grid.footer_top_in', footer_top, shapes=footer_n,
              why='modal top edge in the bottom band, among shapes spanning at least half the '
                  'content width — a page number is furniture, not the footer line')
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
