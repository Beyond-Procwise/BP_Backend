"""Which colours a deck really uses, and what each one is for.

Role follows USE, not the theme: the most-used text colour is the ink whatever the theme says. On
the reference deck the theme is stock Office and would have given the wrong answer for every
token.

A colour under COLOUR_FLOOR uses is incidental and never becomes a token — on a deck of 85 slides
a colour used five times is a one-off, and a token invented from it would then be applied to
everything.
"""
from __future__ import annotations

import colorsys

from .evidence import Evidence
from .read import Deck

COLOUR_FLOOR = 10

_THEME_INK_KEYS = ('dk1', 'dk2')
_PANEL_LUMINANCE = 0.85
_ACCENT_SATURATION = 0.3
_ACCENT_LUMINANCE = (0.2, 0.7)
_RULE_RATIO = 3


def _rgb(colour: str) -> tuple[int, int, int]:
    value = colour.lstrip('#')
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _luminance(colour: str) -> float:
    r, g, b = (c / 255 for c in _rgb(colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _saturation(colour: str) -> float:
    r, g, b = (c / 255 for c in _rgb(colour))
    return colorsys.rgb_to_hls(r, g, b)[2]


def _hue(colour: str) -> float:
    r, g, b = (c / 255 for c in _rgb(colour))
    return colorsys.rgb_to_hls(r, g, b)[0] * 360


def census(deck: Deck) -> dict[str, dict[str, int]]:
    """Every colour that paints something, counted as fill, as text and as line."""
    out: dict[str, dict[str, int]] = {}

    def bump(colour: str | None, role: str) -> None:
        if not colour:
            return
        out.setdefault(colour, {'fills': 0, 'runs': 0, 'lines': 0})[role] += 1

    for slide in deck.slides:
        for shape in slide:
            bump(shape.fill, 'fills')
            bump(shape.line, 'lines')
            for run in shape.runs:
                if run.text.strip():
                    bump(run.colour, 'runs')
    return out


def colours(deck: Deck, ev: Evidence, floor: int = COLOUR_FLOOR) -> dict[str, str]:
    counts = census(deck)
    totals = {colour: sum(roles.values()) for colour, roles in counts.items()}
    eligible = {c: roles for c, roles in counts.items() if totals[c] >= floor}
    for colour in counts:
        if totals[colour] < floor:
            ev.incidental('colour', colour,
                          f'used {totals[colour]} times, under the {floor}-use floor')

    out: dict[str, str] = {}
    taken: set[str] = set()

    def take(name: str, candidates: list[str], why: str) -> None:
        for colour in candidates:
            if colour in taken:
                continue
            taken.add(colour)
            out[name] = colour
            ev.record(f'colours.{name}', colour, why=why, **counts[colour])
            return

    by_runs = sorted(eligible, key=lambda c: (-eligible[c]['runs'], c))
    by_fills = sorted(eligible, key=lambda c: (-eligible[c]['fills'], c))
    # Accents and semantic colours rank by TOTAL use, not by fills. An accent earns its name by
    # how much of the deck it marks — the reference deck's teal is on 94 runs and 19 fills, its
    # blue on 45 runs and 31 fills — and a semantic colour (a red figure, an amber warning) is
    # mostly TEXT, so ranking those by fill alone picked a different amber and a different green
    # than the deck leads with.
    by_use = sorted(eligible, key=lambda c: (-sum(eligible[c].values()), c))
    # PREDOMINANTLY on lines, not exclusively. The reference deck's rule colour (#D5DBE5) paints
    # 61 lines and 8 fills, so "only on lines" found nothing and the pack came back with no rule.
    line_led = [c for c in sorted(eligible, key=lambda c: -eligible[c]['lines'])
                if eligible[c]['lines'] >= _RULE_RATIO * (eligible[c]['fills']
                                                          + eligible[c]['runs'])
                and eligible[c]['lines'] > 0]
    pale = [c for c in by_fills if eligible[c]['fills'] and _luminance(c) > _PANEL_LUMINANCE]
    saturated = [c for c in by_use
                 if _saturation(c) > _ACCENT_SATURATION
                 and _ACCENT_LUMINANCE[0] <= _luminance(c) <= _ACCENT_LUMINANCE[1]]

    take('ink', [c for c in by_runs if eligible[c]['runs']], 'most-used text colour')
    take('muted', [c for c in by_runs if eligible[c]['runs']], 'second most-used text colour')
    take('rule', line_led, f'used on lines at least {_RULE_RATIO}x as often as anywhere else')
    take('panel', pale, 'palest frequently-filled colour')
    for name, hues in (('panel_blue', (190, 260)), ('panel_teal', (150, 190)),
                       ('panel_amber', (20, 60)), ('panel_violet', (260, 300))):
        take(name, [c for c in pale if hues[0] <= _hue(c) <= hues[1]], f'pale fill, hue in {hues}')
    take('accent', saturated, 'most-used saturated colour, by total use')
    take('accent_2', saturated, 'second most-used saturated colour, by total use')
    # WHICH ACCENT IS PRIMARY IS A CLOSE CALL ON A REAL DECK, and not one to decide silently: the
    # reference deck's two are 133 and 98 uses apart. Recorded so the review screen can offer the
    # swap rather than the measurement pretending to certainty it does not have.
    if 'accent' in out and 'accent_2' in out:
        first, second = sum(counts[out['accent']].values()), sum(counts[out['accent_2']].values())
        if second and first < second * 1.5:
            ev.record('colours.accent.close_call', [out['accent'], out['accent_2']],
                      uses=[first, second],
                      why='the two accents are within half of each other; which one is primary '
                          'is a human call, offered on the review screen')
    for name, hues in (('alert_ink', (0, 20)), ('caution', (20, 60)), ('positive', (90, 160))):
        take(name, [c for c in saturated if hues[0] <= _hue(c) <= hues[1]],
             f'saturated, hue in {hues}')

    # A pack must carry an ink and a panel to validate. A deck whose text states no colour at all
    # (Review Focus 2) falls back to the theme's dark colour, then to black — and the evidence
    # says which, because a fallback presented as a measurement is a lie about the deck.
    if 'ink' not in out:
        for key in _THEME_INK_KEYS:
            theme_ink = (deck.theme.get('colours') or {}).get(key)
            if theme_ink:
                out['ink'] = theme_ink
                ev.assumed('colours.ink', theme_ink,
                           f'no run in this deck states a colour, so the theme\'s {key} stands '
                           'in — nothing here was measured from the deck itself')
                break
    if 'ink' not in out:
        out['ink'] = '#000000'
        ev.assumed('colours.ink', '#000000',
                   'no run states a colour and the theme names no dark colour; black assumed')
    if 'muted' not in out:
        out['muted'] = out['ink']
        ev.assumed('colours.muted', out['ink'],
                   'the deck uses one text colour, so muted follows the ink')
    if 'panel' not in out:
        out['panel'] = '#FFFFFF'
        ev.assumed('colours.panel', '#FFFFFF', 'the deck fills nothing pale; white assumed')
    if 'accent' not in out:
        out['accent'] = out['ink']
        ev.assumed('colours.accent', out['ink'],
                   'the deck uses no saturated colour, so the accent follows the ink')
    return out
