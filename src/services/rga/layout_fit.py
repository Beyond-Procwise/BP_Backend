"""Which of a pack's layouts can draw a section, and whether any of them can.

The REPORT GENERATION AGENT composes a report as sections of blocks (prose, a metric, a table, a
chart). The ATB IMPORTER measures a customer's own PowerPoint into layouts: named slots with types
and limits, positioned in inches. This module is the only place those two vocabularies meet, and it
meets them DETERMINISTICALLY: no model chooses a layout, because a model can name a layout that
does not exist, and because `generate_report` promises that the same Fact Pack redraws to the same
artefact without a model.

It is also the only module here that reads a layout row. The renderer must not learn the store's
column names and the fit must not learn SQL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.services.atb.pptx_import import store

#: Who the importer says fills a slot. Only an agent-filled slot may take composed content:
#: 'auto' is the platform's (the sources footer) and 'bind' is a chart, which becomes a region
#: host in the builder rather than anything written.
AGENT_FILL = 'agent'
BIND_FILL = 'bind'


class NoPack(RuntimeError):
    """No approved version of that pack, so there is nothing to lay a report out on."""


@dataclass(frozen=True)
class SlotOption:
    name: str
    type: str
    fill: str
    max_words: Optional[int] = None
    max_chars: Optional[int] = None
    columns: Tuple[Dict[str, Any], ...] = ()
    item_min: Optional[int] = None
    item_max: Optional[int] = None
    heading_max: Optional[int] = None
    body_max: Optional[int] = None


@dataclass(frozen=True)
class LayoutOption:
    layout_key: str
    name: str
    kind: str
    slots: Tuple[SlotOption, ...]
    region_ids: Tuple[str, ...]

    def of_type(self, type_: str, fill: str = AGENT_FILL) -> Tuple[SlotOption, ...]:
        return tuple(s for s in self.slots if s.type == type_ and s.fill == fill)


@dataclass(frozen=True)
class Catalogue:
    pack_key: str
    pack_id: str
    format_kind: str
    layouts: Tuple[LayoutOption, ...]


def _slot(name: str, raw: Dict[str, Any]) -> SlotOption:
    item = (raw.get('item') or {}) if isinstance(raw.get('item'), dict) else {}
    heading = item.get('heading') or {}
    body = item.get('body') or {}
    return SlotOption(
        name=name,
        type=str(raw.get('type') or 'text'),
        fill=str(raw.get('fill') or AGENT_FILL),
        max_words=raw.get('max_words'),
        max_chars=raw.get('max_chars'),
        columns=tuple(raw.get('columns') or ()),
        item_min=raw.get('min'),
        item_max=raw.get('max'),
        heading_max=heading.get('max_chars'),
        body_max=body.get('max_chars'),
    )


def catalogue(conn, pack_key: str) -> Catalogue:
    """The approved layouts of the NEWEST APPROVED version of one pack.

    One pack per report: `STATE.style` in the builder is per report, so a snapshot that mixed two
    packs would be unrepresentable. The newest APPROVED version, not simply the newest: a candidate
    re-import sitting on top of an approved one must not change what a report is laid out on.
    """
    approved = [p for p in store.packs(conn)
                if p.get('pack_key') == pack_key and p.get('status') == 'approved']
    if not approved:
        raise NoPack(f'no approved style pack named {pack_key!r}')
    pack = max(approved, key=lambda p: int(p.get('version') or 0))
    rows = store.layouts(conn, pack_id=pack['pack_id'], status='approved')
    layouts = tuple(
        LayoutOption(
            layout_key=row['layout_key'],
            name=row.get('name') or row.get('proposed_name') or row['layout_key'],
            kind=row.get('kind') or 'template',
            slots=tuple(_slot(n, s or {}) for n, s in sorted((row.get('slots') or {}).items())),
            region_ids=tuple(r.get('id') for r in (row.get('regions') or []) if r.get('id')),
        )
        for row in rows
    )
    return Catalogue(pack_key=pack_key, pack_id=pack['pack_id'],
                     format_kind=((pack.get('format') or {}).get('kind') or 'deck'),
                     layouts=layouts)
