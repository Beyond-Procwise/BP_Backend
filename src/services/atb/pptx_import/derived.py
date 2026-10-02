"""The values the contract requires that a .pptx does not simply contain.

Four are derived and one is contested. The reference deck declares en-US on all 6,940 runs while
spelling Virtualisation, Mobilise, Optimise, utilisation, with 8 -ize uses in 85 slides. So the
declared value is recorded, the spelling is tested, and the contradiction is FLAGGED for a human
rather than resolved by guess — the same lesson as the theme.
"""
from __future__ import annotations

import re
from collections import Counter

from .evidence import Evidence
from .read import Deck

BRITISH_RE = re.compile(r'\b\w+(?:isation|ise[sd]?|ising|our|ours)\b', re.I)
# NOT `-or`: that is only an Americanism in contrast to a -our spelling, and as a suffix on its
# own it matches for, sector, vendor, major, contractor — 124 of them in the reference deck, which
# outvoted its 116 British spellings and left a plainly British deck uncontested.
AMERICAN_RE = re.compile(r'\b\w+(?:ization|ize[sd]?|izing)\b', re.I)

# Words that end -ise or -our and are not a British spelling of anything. Without these, any deck
# that says "four hours" reads as British English.
_FALSE_FRIENDS = re.compile(
    r'^(?:wise|rise|risen|promise|promises|promised|precise|concise|advertise|advertises|'
    r'advertised|exercise|exercises|exercised|surprise|surprises|surprised|franchise|'
    r'franchises|compromise|compromises|compromised|merchandise|supervise|supervises|'
    r'supervised|televise|revise|revises|revised|devise|devises|devised|arise|arises|'
    r'otherwise|likewise|clockwise|paradise|expertise|disguise|'
    r'four|fours|hour|hours|your|yours|tour|tours|pour|pours|flour|our|ours|labour|colour|'
    r'honour|favour|neighbour|behaviour|armour|humour|rumour|vapour|savour|harbour|'
    r'contour|contours|detour|detours|velour|glamour|parlour|valour|vigour|candour|'
    r'clamour|endeavour|fervour|rigour|saviour|splendour|tumour)$', re.I)

_MIN_BRITISH = 3


def series_palette(deck: Deck, ev: Evidence, colours: dict[str, str]) -> list[str]:
    """Never empty: an empty series palette fails the contract."""
    if deck.chart_series_colours:
        out = list(deck.chart_series_colours)
        ev.record('series_palette', out, charts=True,
                  why='the colours the chart series use, in first-use order')
        return out
    fallback = [colours.get(name) for name in
                ('accent', 'accent_2', 'ink', 'caution', 'positive', 'alert_ink')]
    out = list(dict.fromkeys(c for c in fallback if c)) or [colours.get('ink', '#000000')]
    ev.record('series_palette', out, charts=False,
              why="the deck holds no charts; the pack's own accents stand in, because an empty "
                  'series palette fails the contract')
    return out


def rating_scales(deck: Deck, ev: Evidence) -> dict:
    """A scale needs a small repeated vocabulary AND a fill per label.

    Without fills there is nothing to colour a chip with, so no scale is derived and the column
    stays text (design §5a). A human defines one on the review screen, and §5b carries it forward.
    """
    scales: dict[str, dict] = {}
    for slide in deck.slides:
        for shape in slide:
            if shape.kind != 'table' or not shape.table or len(shape.table) < 3:
                continue
            header, *body = shape.table
            for index, name in enumerate(header):
                values = [row[index].strip() for row in body
                          if index < len(row) and row[index].strip()]
                if len(values) < 3:
                    continue
                vocabulary = Counter(values)
                if len(vocabulary) > 5 or max(len(v) for v in values) > 12:
                    continue
                # A rating column REPEATS its vocabulary — Low / High / Low. A column of short
                # distinct values is a label column (Freight / IT / Tail), and flagging it would
                # offer to turn every category name into a rating chip.
                if len(vocabulary) == len(values):
                    continue
                # python-pptx reports no fill for a table CELL through the shape, so there is
                # nothing here to take a chip colour from. Stated rather than guessed.
                ev.incidental(
                    'rating_scale', f'{name or "column %d" % index} on slide {shape.slide}',
                    'a repeated vocabulary with no cell fill to take a chip colour from; the '
                    'column stays text until a scale is defined by hand')
    return scales


def writing(deck: Deck, ev: Evidence) -> dict:
    declared = Counter(deck.run_langs).most_common(1)
    locale = declared[0][0] if declared else 'en-GB'
    text = ' '.join(run.text for slide in deck.slides for shape in slide for run in shape.runs)
    british = [w for w in BRITISH_RE.findall(text) if not _FALSE_FRIENDS.match(w)]
    american = [w for w in AMERICAN_RE.findall(text) if not _FALSE_FRIENDS.match(w)]
    contested = bool(locale.lower().endswith('-us')
                     and len(british) >= _MIN_BRITISH
                     and len(british) > len(american))
    suggested = 'en-GB' if contested else locale
    ev.record('writing.locale', locale,
              runs=declared[0][1] if declared else 0,
              british_spellings=len(british), american_spellings=len(american),
              contested=contested,
              why='declared by the runs'
                  + ('; CONTESTED — the spelling disagrees, so a human decides' if contested
                     else ''))
    return {
        'locale': locale,
        'locale_contested': contested,
        'locale_suggested': suggested,
        'title_max_words': 12,
        'title_style': 'assertion',
        'subtitle_style': 'basis',
    }
