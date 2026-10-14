"""How many layouts a deck really has.

Measured on the reference deck (85 slides): these rules give 34 groups — 8 used by more than one
slide and covering 59 of them, and 26 used once. An earlier draft of the design guessed "the
mid-teens"; it was wrong, and generalising the repeat rule from a single row to a repeating
SEQUENCE of rows (tested) changes the count not at all.

Design §6a: the 8 reused groups become TEMPLATES and the 26 single-use ones become PAGES,
because a quadrant is a page someone arranged, not a template. `reused` is the only thing that
decides which, and import_pack turns it into the stored `kind`.
"""
from __future__ import annotations

from dataclasses import dataclass

from .grid import FOOTER_BAND_FRACTION, TITLE_BAND_FRACTION
from .read import Deck, Shape

# Shared with grid.py: a band is a fraction of the page, not an absolute inch count.
TITLE_BAND_FRACTION = 0.167
SAME_ROW_IN = 0.45
SAME_COL_IN = 0.1
DECORATION_IN = 0.5
EMPTY_ROW = ('empty', 'NONE')


@dataclass(frozen=True)
class Cluster:
    signature: tuple
    slides: tuple[int, ...]
    # The representative slide's rows — what the example fill is taken from.
    rows: tuple[tuple[Shape, ...], ...]
    # EVERY member's rows, slide by slide, in the same order as `slides`. Without this a region's
    # box could only be derived from one slide, and "this member is off the median" could only
    # compare a shape against the row it is part of — which reported each card of a four-up row as
    # an outlier. Measured on the reference deck: 114 such meaningless problems.
    rows_by_slide: tuple[tuple[tuple[Shape, ...], ...], ...] = ()

    @property
    def reused(self) -> bool:
        return len(self.slides) > 1


def _body(shapes: tuple[Shape, ...], height_in: float) -> list[Shape]:
    footer = height_in * FOOTER_BAND_FRACTION
    title_band_to = height_in * TITLE_BAND_FRACTION
    return [s for s in shapes
            if title_band_to <= s.box.y <= footer
            and not (s.box.w < DECORATION_IN and s.box.h < DECORATION_IN)]


def rows_of(shapes: tuple[Shape, ...], height_in: float) -> list[list[Shape]]:
    """Body shapes grouped into rows by top edge."""
    body = sorted(_body(shapes, height_in), key=lambda s: (s.box.y, s.box.x))
    rows: list[list[Shape]] = []
    current: list[Shape] = []
    anchor: float | None = None
    for shape in body:
        if anchor is not None and abs(shape.box.y - anchor) <= SAME_ROW_IN:
            current.append(shape)
            continue
        if current:
            rows.append(current)
        current = [shape]
        anchor = shape.box.y
    if current:
        rows.append(current)
    return rows


def _row_signature(row: list[Shape]) -> tuple[int, str]:
    columns: list[float] = []
    for shape in sorted(row, key=lambda s: s.box.x):
        if not columns or abs(shape.box.x - columns[-1]) > SAME_COL_IN:
            columns.append(shape.box.x)
    kinds = {s.kind for s in row}
    # Rule 3: a table row never merges with a chart row, and neither with a row of plain shapes.
    kind = 'TABLE' if 'table' in kinds else 'CHART' if 'chart' in kinds else 'SHAPE'
    return len(columns), kind


def signature(shapes: tuple[Shape, ...], height_in: float) -> tuple:
    rows = [_row_signature(r) for r in rows_of(shapes, height_in)]
    if not rows:
        return (EMPTY_ROW,)
    out: list[tuple] = list(rows)
    collapsed = False
    # Rule 2: a trailing repeated row collapses to one copy, marked repeating. That is what makes
    # a three-row card grid and a two-row one the same layout.
    while len(out) >= 2 and out[-1][:2] == out[-2][:2]:
        out.pop()
        collapsed = True
    if collapsed:
        out[-1] = out[-1][:2] + ('repeat',)
    return tuple(out)


def group(deck: Deck) -> list[Cluster]:
    found: dict[tuple, list[int]] = {}
    rows_by_signature: dict[tuple, list[tuple]] = {}
    for index, shapes in enumerate(deck.slides, 1):
        sig = signature(shapes, deck.height_in)
        found.setdefault(sig, []).append(index)
        rows_by_signature.setdefault(sig, []).append(
            tuple(tuple(r) for r in rows_of(shapes, deck.height_in)))
    clusters = [Cluster(signature=sig, slides=tuple(slides),
                        rows=rows_by_signature[sig][0],
                        rows_by_slide=tuple(rows_by_signature[sig]))
                for sig, slides in found.items()]
    clusters.sort(key=lambda c: (-len(c.slides), c.slides[0]))
    return clusters
