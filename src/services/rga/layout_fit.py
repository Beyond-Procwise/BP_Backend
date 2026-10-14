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


# ---------------------------------------------------------------------------
# The fit: which layout draws a section, or why none can.
# ---------------------------------------------------------------------------
from src.services.rga.models import (FACT_ID, ChartBlock, FindingListBlock,  # noqa: E402
                                     MetricBlock, NarrativeBlock, Section, TableBlock)

#: A slot name that must exist on any layout a section can be drawn on. A page without a masthead
#: is not a page of a board paper, and every layout the importer measures carries one.
TITLE_SLOT = 'title'


class NoFit(RuntimeError):
    """No layout in this pack can draw that section, and this says which section and why."""

    def __init__(self, section_id: str, reason: str):
        super().__init__(f'section {section_id}: {reason}')
        self.section_id = section_id
        self.reason = reason


@dataclass(frozen=True)
class Fitted:
    layout_key: str
    text: Dict[str, str] = field(default_factory=dict)
    lists: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    tables: Dict[str, Tuple[List[str], List[List[str]]]] = field(default_factory=dict)
    charts: Dict[str, ChartBlock] = field(default_factory=dict)
    #: slot name -> the finding ids that slot lists. NOT resolved here: the fit has no Fact Pack,
    #: and a raw finding id on a board paper is worse than a refusal, so the renderer — which has
    #: the pack — turns each ref into what it says.
    findings: Dict[str, List[str]] = field(default_factory=dict)


def _token(ref: str) -> str:
    """A figure reaches a slot as a REFERENCE, never as a number: the renderer resolves it and the
    post-check can see what the page claims to rely on."""
    return '{{f:%s}}' % ref


def _cell(value: str) -> str:
    text = str(value or '')
    if text.startswith('{{f:'):
        return text
    # A cell is a FACT ID or a literal label (TableBlock's own docstring), and an id has one shape
    # in this product: models.FACT_ID, ^F\d{4}$. Guessing by punctuation would turn a label like
    # "Q1.Network" into a token and leave every real id as a literal.
    if FACT_ID.match(text):
        return _token(text)
    return text


def _fits_text(slot: SlotOption, text: str) -> bool:
    if slot.max_words is not None and len(text.split()) > slot.max_words:
        return False
    if slot.max_chars is not None and len(text) > slot.max_chars:
        return False
    return True


def _fits_items(slot: SlotOption, count: int, headings: Sequence[str] = (),
                bodies: Sequence[str] = ()) -> bool:
    if slot.item_max is not None and count > slot.item_max:
        return False
    if slot.item_min is not None and count < slot.item_min:
        return False
    if slot.heading_max is not None and any(len(h) > slot.heading_max for h in headings):
        return False
    if slot.body_max is not None and any(len(b) > slot.body_max for b in bodies):
        return False
    return True


def _try(section: Section, layout: LayoutOption) -> Optional[Fitted]:
    """-> a Fitted, or None when this layout cannot take this section.

    Assignment is first-free of the right type, in slot-name order (the catalogue sorts them), so
    the same section always lands in the same slots.
    """
    titles = [s for s in layout.of_type('text') if s.name == TITLE_SLOT]
    if not titles or not _fits_text(titles[0], section.title):
        return None
    free_text = [s for s in layout.of_type('text') if s.name != TITLE_SLOT]
    free_list = list(layout.of_type('list'))
    free_table = list(layout.of_type('table'))
    free_chart = list(layout.of_type('chart', fill=BIND_FILL))

    out = Fitted(layout_key=layout.layout_key)
    out.text[TITLE_SLOT] = section.title

    # A list is filled ONCE with every item that belongs to it — a card grid is one slot, not one
    # slot per card. Metrics and findings take DIFFERENT list slots: they are resolved differently,
    # and interleaving a raw finding id with a figure is how an id reaches a board paper.
    metrics = [b for b in section.blocks if isinstance(b, MetricBlock)]
    if metrics:
        if not free_list:
            return None
        slot = free_list.pop(0)
        items = [(_token(m.fact_ref), m.fact_ref) for m in metrics]
        if not _fits_items(slot, len(items), [h for h, _ in items], [b for _, b in items]):
            return None
        out.lists[slot.name] = items

    refs = [r for b in section.blocks if isinstance(b, FindingListBlock) for r in b.finding_refs]
    if refs:
        if not free_list:
            return None
        slot = free_list.pop(0)
        # The count is checked here; the headings and bodies cannot be, because what each ref SAYS
        # is the renderer's to resolve. The renderer checks the lengths it then produces.
        if not _fits_items(slot, len(refs)):
            return None
        out.findings[slot.name] = refs

    for block in section.blocks:
        if isinstance(block, NarrativeBlock):
            if not free_text:
                return None
            slot = free_text.pop(0)
            if not _fits_text(slot, block.text):
                return None
            out.text[slot.name] = block.text
        elif isinstance(block, TableBlock):
            if not free_table:
                return None
            slot = free_table.pop(0)
            if slot.columns and len(block.columns) > len(slot.columns):
                return None
            out.tables[slot.name] = (list(block.columns),
                                     [[_cell(c) for c in row] for row in block.rows])
        elif isinstance(block, ChartBlock):
            if not free_chart:
                return None
            out.charts[free_chart.pop(0).name] = block
    return out


def fit(section: Section, layouts: Sequence[LayoutOption]) -> Fitted:
    """The first layout in catalogue order that can draw this section.

    FIRST, not best: the order is the layouts' creation order, which is the order the deck used
    them, and a deterministic choice is what lets the same Fact Pack redraw to the same pages.
    """
    for layout in layouts:
        fitted = _try(section, layout)
        if fitted is not None:
            return fitted
    raise NoFit(section.id, _why(section, layouts))


def _why(section: Section, layouts: Sequence[LayoutOption]) -> str:
    """The reason a person can act on: what the section needed that no layout had."""
    need: Dict[str, int] = {}
    for block in section.blocks:
        kind = {NarrativeBlock: 'text', TableBlock: 'table', ChartBlock: 'chart'}.get(type(block))
        if kind:
            need[kind] = need.get(kind, 0) + 1
    metrics = sum(1 for b in section.blocks if isinstance(b, MetricBlock))
    refs = sum(len(b.finding_refs) for b in section.blocks if isinstance(b, FindingListBlock))
    lists_needed = (1 if metrics else 0) + (1 if refs else 0)
    parts: List[str] = []
    for kind, wanted in sorted(need.items()):
        best = max((len(l.of_type(kind, BIND_FILL if kind == 'chart' else AGENT_FILL))
                    for l in layouts), default=0)
        if best < wanted:
            parts.append(f'it needs {wanted} {kind} slots and the best layout has {best}')
    if lists_needed:
        best_slots = max((len(l.of_type('list')) for l in layouts), default=0)
        if best_slots < lists_needed:
            parts.append(f'it needs {lists_needed} card lists (figures and findings are listed '
                         f'separately) and the best layout has {best_slots}')
    if metrics:
        best = max((max((s.item_max or 0) for s in l.of_type('list')) if l.of_type('list') else 0
                    for l in layouts), default=0)
        if best < metrics:
            parts.append(f'it has {metrics} figures and the largest card list takes {best}')
    if not parts:
        parts.append('no layout in this pack has a title slot that fits its heading, '
                     'or its text is longer than any slot allows')
    return '; '.join(parts)
