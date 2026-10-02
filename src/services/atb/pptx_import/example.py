"""One example fill per layout, taken from its first member slide.

WHY AT ALL: an empty fill renders a blank page, which is indistinguishable from a broken
renderer. That exact bug shipped in the UI on 2026-10-02 and was found by review, not by a test.

WHY LABELLED: these are the source document's words, and they belong to whoever wrote it. The fill
travels with {'file', 'slide'} so every screen can say "as it appeared in <file>, slide N" and
nobody can mistake it for their own figures.
"""
from __future__ import annotations

from .cluster import Cluster
from .read import Deck, Shape

_TITLE_BAND_IN = 1.25


def words_of(shape: Shape) -> str:
    return ' '.join(r.text for r in shape.runs if r.text.strip()).strip()


def example_fill(cluster: Cluster, deck: Deck, slots: dict) -> tuple[dict, dict]:
    slide_no = cluster.slides[0]
    shapes = deck.slides[slide_no - 1]
    fill: dict = {'slots': {}}

    banner = sorted((s for s in shapes if s.box.y < _TITLE_BAND_IN and words_of(s)),
                    key=lambda s: s.box.y)
    if banner:
        fill['slots']['title'] = {'text': words_of(banner[0])}
    if len(banner) > 1:
        fill['slots']['subtitle'] = {'text': words_of(banner[1])}

    for name, slot in slots.items():
        if name in ('title', 'subtitle', 'sources'):
            continue
        kind = slot.get('type')
        if kind == 'table':
            table = next((s.table for row in cluster.rows for s in row if s.table), None)
            if table and len(table) > 1:
                ids = [c['id'] for c in slot.get('columns', [])]
                fill['slots'][name] = {
                    'rows': [dict(zip(ids, row)) for row in table[1:]],
                }
        elif kind == 'list':
            items = [{'heading': words_of(s)[:80], 'body': ''}
                     for row in cluster.rows for s in row if words_of(s)]
            if items:
                fill['slots'][name] = {'items': items[:slot.get('max', 4)]}
        elif kind in ('text', 'rich_text', 'callout'):
            prose = [words_of(s) for row in cluster.rows for s in row if words_of(s)]
            if prose:
                fill['slots'][name] = {'text': prose[0]}
    return fill, {'slide': slide_no}
