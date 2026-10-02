"""The type scale, named by rank and position, and the fonts, taken from the runs.

The theme's majorFont/minorFont are recorded and NOT used: the reference deck's theme says
Calibri Light while the deck's own headings are Cambria. A .pptx names a family and nothing else,
so the fallback stack is invented here — and the evidence says invented, not measured.
"""
from __future__ import annotations

from collections import Counter

from .evidence import Evidence
from .read import Deck

SERIF_FAMILIES = frozenset({
    'cambria', 'georgia', 'times new roman', 'times', 'garamond', 'book antiqua', 'palatino',
    'palatino linotype', 'constantia', 'bookman old style', 'century schoolbook',
})

_SERIF_STACK = "Georgia, 'Times New Roman', serif"
_SANS_STACK = "'Segoe UI', system-ui, -apple-system, Arial, sans-serif"

# Largest first. title/subtitle/body/table/footer are REQUIRED_TYPE_SCALE in the contract; the
# other three are optional and filled when the deck has enough distinct sizes for them.
_ROLES = ('title', 'subtitle', 'kpi_value', 'card_title', 'body', 'table', 'small', 'footer')
_REQUIRED = ('title', 'subtitle', 'body', 'table', 'footer')
_MERGE_PT = 0.6
_TITLE_BAND_IN = 1.0
_FALLBACK_SCALE = {'title': 30.0, 'subtitle': 14.0, 'kpi_value': 24.0, 'card_title': 13.0,
                   'body': 11.0, 'table': 10.0, 'small': 9.5, 'footer': 9.0}


def _sizes(deck: Deck, title_band_only: bool = False) -> Counter:
    sizes: Counter = Counter()
    for slide in deck.slides:
        for shape in slide:
            if title_band_only and shape.box.y >= _TITLE_BAND_IN:
                continue
            for run in shape.runs:
                if run.size_pt and run.text.strip():
                    sizes[round(run.size_pt, 1)] += 1
    return sizes


def type_scale(deck: Deck, ev: Evidence) -> dict[str, float]:
    sizes = _sizes(deck)
    if not sizes:
        ev.record('type_scale_pt', dict(_FALLBACK_SCALE),
                  why='the deck states no run sizes; the platform default scale stands in')
        return dict(_FALLBACK_SCALE)

    distinct: list[float] = []
    for size in sorted(sizes, reverse=True):
        if distinct and abs(distinct[-1] - size) <= _MERGE_PT:
            smaller, larger = sorted((size, distinct[-1]))
            keep = size if sizes[size] > sizes[distinct[-1]] else distinct[-1]
            drop = larger if keep == smaller else smaller
            ev.incidental('type_size', drop,
                          f'merged into {keep}pt, within {_MERGE_PT}pt of it')
            distinct[-1] = keep
            continue
        distinct.append(size)

    # ANCHORED AT BOTH ENDS, and the body is decided by USE.
    #
    # Zipping the sizes onto the roles largest-first (the first version of this) put the SMALLEST
    # measured size on card_title and left footer with nothing, so footer fell back to the title:
    # a 30pt footnote. The anchors are what a deck actually has — a title at the top, a footnote
    # at the bottom, and a body size that most of the words are set in.
    band = _sizes(deck, title_band_only=True)
    title = max((s for s in distinct if band.get(s, 0) > 0), default=distinct[0])
    footer = min(distinct)
    ev.record('type_scale_pt.title', title, runs=sizes[title],
              why='the largest size that appears in the title band')
    ev.record('type_scale_pt.footer', footer, runs=sizes[footer],
              why='the smallest size in the deck')

    def most_used(candidates: list[float], why: str, role: str) -> float | None:
        if not candidates:
            return None
        # Ties go to the larger size: on a page the body is set larger than the table.
        pick = sorted(candidates, key=lambda s: (-sizes[s], -s))[0]
        ev.record(f'type_scale_pt.{role}', pick, runs=sizes[pick], why=why)
        return pick

    scale: dict[str, float] = {'title': title, 'footer': footer}
    body = most_used([s for s in distinct if s not in (title, footer)],
                     'the size most of the deck is set in', 'body') or title
    scale['body'] = body
    scale['table'] = most_used([s for s in distinct if footer < s < body],
                               'most-used size between the footnote and the body', 'table') or body
    above = [s for s in distinct if body < s < title]
    scale['subtitle'] = most_used(above, 'most-used size between the body and the title',
                                  'subtitle') or body
    rest = [s for s in above if s != scale['subtitle']]
    card_title = most_used(rest, 'next most-used size between the body and the title', 'card_title')
    if card_title:
        scale['card_title'] = card_title
    # kpi_value and small are optional in the contract and are NOT invented here: the components
    # that name them already fall back, and a guessed token is worse than an absent one.
    for role in _REQUIRED:
        if role not in scale:
            scale[role] = body
            ev.record(f'type_scale_pt.{role}', body, runs=0,
                      why='the deck has no distinct size for this role; it follows the body')
    return scale


def _stack_for(family: str | None) -> tuple[str, bool]:
    if family and family.strip().lower() in SERIF_FAMILIES:
        return _SERIF_STACK, True
    return _SANS_STACK, False


def fonts(deck: Deck, ev: Evidence) -> dict[str, dict[str, str]]:
    body_counts: Counter = Counter()
    heading_counts: Counter = Counter()
    for slide in deck.slides:
        for shape in slide:
            target = heading_counts if shape.box.y < _TITLE_BAND_IN else body_counts
            for run in shape.runs:
                if run.font and run.text.strip():
                    target[run.font] += 1
    body_family = body_counts.most_common(1)[0][0] if body_counts else None
    heading_family = heading_counts.most_common(1)[0][0] if heading_counts else body_family
    if body_family is None:
        body_family = heading_family

    theme_fonts = deck.theme.get('fonts') or {}
    if theme_fonts:
        ev.ignored('theme.fonts', theme_fonts,
                   'the runs name their own families; a theme font is not what the deck '
                   'looks like')

    out: dict[str, dict[str, str]] = {}
    for role, family in (('heading', heading_family), ('body', body_family)):
        stack, serif = _stack_for(family)
        out[role] = {'family': family or 'inherit', 'fallback': stack}
        ev.record(f'fonts.{role}', out[role], invented=True,
                  why='family measured from the runs; fallback stack invented '
                      f'({"serif" if serif else "sans"})')
    return out
