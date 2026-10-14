"""One example fill per layout, taken from its first member slide.

WHY AT ALL: an empty fill renders a blank page, which is indistinguishable from a broken
renderer. That exact bug shipped in the UI on 2026-10-02 and was found by review, not by a test.

WHY LABELLED: these are the source document's words, and they belong to whoever wrote it. The fill
travels with {'file', 'slide'} so every screen can say "as it appeared in <file>, slide N" and
nobody can mistake it for their own figures.
"""
from __future__ import annotations

import re

from .cluster import Cluster
from .read import Deck, Shape
from .slots import _rows_per_signature_row

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

    # EACH REGION FROM ITS OWN ROW. Reading every row for every slot gave the same sentence in
    # prose2, prose4 and prose6 and the same leading items in every card grid — so the thumbnail a
    # human approves already differed from the page the report would print, which is the one thing
    # §8 promises it cannot do. A region id carries its signature-row number (`cards3`, `prose4`),
    # and the plan maps that back to the rows it stands for.
    plan = _rows_per_signature_row(cluster)
    for name, slot in slots.items():
        if name in ('title', 'subtitle', 'sources'):
            continue
        kind = slot.get('type')
        own = _rows_for(name, plan, cluster)
        if kind == 'table':
            table = next((s.table for s in own if s.table), None)
            if table and len(table) > 1:
                ids = [c['id'] for c in slot.get('columns', [])]
                fill['slots'][name] = {'rows': [dict(zip(ids, row)) for row in table[1:]]}
        elif kind == 'list':
            items = [{'heading': words_of(s)[:80], 'body': ''} for s in own if words_of(s)]
            if items:
                fill['slots'][name] = {'items': items[:slot.get('max', 4)]}
        elif kind in ('text', 'rich_text', 'callout'):
            prose = [words_of(s) for s in own if words_of(s)]
            if prose:
                fill['slots'][name] = {'text': prose[0]}
    return fill, {'slide': slide_no}


def _rows_for(slot_name: str, plan: list[list[int]], cluster: Cluster) -> tuple[Shape, ...]:
    """The shapes of the row(s) the named region stands for."""
    match = re.search(r'(\d+)$', slot_name)
    if not match:
        return tuple(shape for row in cluster.rows for shape in row)
    position = int(match.group(1)) - 1
    if position < 0 or position >= len(plan):
        return ()
    return tuple(shape for index in plan[position]
                 if index < len(cluster.rows) for shape in cluster.rows[index])
