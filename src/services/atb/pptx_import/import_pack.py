"""One call: bytes in, a stored candidate pack and its layouts out.

Design §6a: only clusters used by more than one slide become layouts. The single-use structures
are LISTED with their slide numbers — a quadrant is a page someone arranged, not a template, and
emitting 26 single-use layouts would fill the picker with near-identical skeletons, which is the
complaint this whole build answers. Step 2 imports them as composed pages, and this list is its
worklist.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from . import store
from .cluster import group
from .contract import PackInvalid
from .emit import build_layout, build_pack, proposed_name
from .evidence import Evidence
from .read import DeckUnreadable, read_deck

_NOTHING_INHERITED: dict = {'names': {}, 'rating_scales': {}, 'locale': None, 'rejected': []}


class ImportRefused(Exception):
    """The file was not imported, and this says why. Nothing was stored."""


@dataclass
class ImportResult:
    pack_key: str
    version: int
    pack: dict
    layouts: list[dict]
    single_use: list[dict]
    evidence: dict
    pack_id: str | None = None
    diff: dict = field(default_factory=dict)
    problems: list[dict] = field(default_factory=list)


def key_for(filename: str) -> str:
    stem = re.sub(r'\.pptx$', '', filename, flags=re.I)
    return re.sub(r'[^a-z0-9]+', '-', stem.lower()).strip('-') or 'imported-pack'


def _diff(previous: dict | None, pack: dict) -> dict:
    """What moved since the previous version of this key, so a revised deck shows its changes."""
    if not previous:
        return {}
    before = previous.get('tokens') or {}
    changed: dict = {}
    for field_name in ('colours', 'type_scale_pt', 'grid', 'fonts'):
        old, new = before.get(field_name) or {}, pack.get(field_name) or {}
        moved = {k: {'from': old.get(k), 'to': new.get(k)}
                 for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}
        if moved:
            changed[field_name] = moved
    return changed


def import_pack(data: bytes, filename: str, user: str, conn=None) -> ImportResult:
    try:
        deck = read_deck(data)
    except DeckUnreadable as exc:
        raise ImportRefused(str(exc)) from exc

    key = key_for(filename)
    ev = Evidence()
    try:
        pack = build_pack(deck, ev, key=key, name=filename)
    except PackInvalid as exc:
        raise ImportRefused(f'the derived pack is not valid: {exc}') from exc

    carried = store.inherited(conn, key) if conn is not None else dict(_NOTHING_INHERITED)
    if carried.get('rating_scales'):
        pack['rating_scales'].update(carried['rating_scales'])
    if carried.get('locale'):
        pack['writing']['locale'] = carried['locale']

    layouts: list[dict] = []
    single_use: list[dict] = []
    problems: list[dict] = []
    for cluster in group(deck):
        if not cluster.reused:
            single_use.append({'slides': list(cluster.slides),
                               'structure': proposed_name(cluster)})
            continue
        try:
            layout = build_layout(cluster, deck, pack, ev, filename)
        except PackInvalid as exc:
            problems.append({'kind': 'layout_invalid', 'region': None, 'why': str(exc),
                             'slides': list(cluster.slides)})
            continue
        if layout['id'] in carried.get('rejected', []):
            continue
        if layout['id'] in carried.get('names', {}):
            layout['name'] = carried['names'][layout['id']]
        problems.extend(layout['problems'])
        layouts.append(layout)

    # A value the deck did not give us is a PROBLEM, not just an evidence note: the review screen
    # shows problems, and a pack built entirely of assumptions looked exactly like a measured one.
    problems = [{'kind': 'assumed', 'region': a['path'], 'slides': [], 'why': a['why']}
                for a in ev.assumptions] + problems
    result = ImportResult(pack_key=key, version=1, pack=pack, layouts=layouts,
                          single_use=single_use, evidence=ev.as_dict(), problems=problems)
    if conn is None:
        return result

    previous = next((p for p in store.packs(conn, include_importing=True)
                     if p['pack_key'] == key), None)
    result.diff = _diff(previous, pack)
    result.version = store.next_version(conn, key)
    result.pack_id = store.insert_importing(
        conn, pack_key=key, version=result.version, source_file=filename,
        source_sha256=hashlib.sha256(data).hexdigest(), slide_count=len(deck.slides),
        pack=pack, evidence=result.evidence, user=user)
    for layout in layouts:
        store.insert_layout(conn, pack_id=result.pack_id, layout=layout)
    store.mark_candidate(conn, result.pack_id)
    return result
