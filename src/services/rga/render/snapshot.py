"""The report, drawn as pages in the report builder.

The third drawing of the same composed AST that `render/pptx.py` and `render/html.py` draw. The
other two produce files; this one produces the JSON SNAPSHOT the builder already saves and opens
(`POST /spendiq/reports` stores it, `openReport()` loads it into STATE), so a generated report
arrives as editable pages laid out on the customer's own measured layouts — not as a stack of
cards, and not as a file to download.

WHAT IT DOES NOT DO. It does not choose the words: the composer did. It does not choose the layout
from the model: `layout_fit` chooses deterministically, so the same Fact Pack redraws the same
pages. It does not write a figure into text: a figure is a {{f:<fact_id>}} reference and the fact
travels beside it, which is what keeps the measured/edited/missing chips honest.
"""
from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Dict, List, Optional

from src.services.rga.layout_fit import Catalogue, Fitted, fit
from src.services.rga.models import (ChartBlock, Confidence, FactEntry, FactPack, ReportAST,
                                     canonical_hash)
from src.services.rga.render import RenderedArtefact

RENDERER = 'snapshot'
RENDERER_VERSION = '1'
MEDIA_TYPE = 'application/json'

#: What the builder opens a generated report as. `role` and `rangePreset` match baseState().
_ROLE = 'exec'
_RANGE_PRESET = 'FY'


def _coverage_line(pack: FactPack) -> str:
    """The same words as the deck's front page — see render/html._coverage_line."""
    unmeasured = sum(1 for f in pack.facts if f.confidence is Confidence.UNASSESSED)
    if not unmeasured:
        return f'all {len(pack.facts)} measures assessed'
    return f'{unmeasured} of {len(pack.facts)} measures not assessed'


def _fact_out(entry: FactEntry) -> Dict[str, Any]:
    """One fact, in the shape the layout renderer reads.

    `provenance` is REQUIRED. factProvenance() falls back to 'unknown', which draws an unmeasured
    chip — so a fact the pack measured must say so, or the report under-claims its own evidence.
    A value of None is NOT a zero: it reaches the page as an absent measure and draws an em-dash.
    """
    value = entry.value
    return {
        'value': float(value) if isinstance(value, Decimal) else value,
        'kind': 'money' if entry.currency else ('number' if value is not None else 'text'),
        'currency': entry.currency,
        'unit': entry.unit,
        'format_hint': getattr(entry.format_hint, 'value', entry.format_hint),
        'label': entry.label,
        # CONFIDENCE AND ORIGIN TRAVEL SEPARATELY, and neither is flattened into the other.
        # badge_text's docstring is explicit: "a LEGACY_UNVERIFIED figure that looks like a verified
        # one is the specific misreading it exists to prevent", and "collapsing the two fields would
        # render it as something else". `provenance` is the chip the browser draws and is the
        # lower-cased confidence — corroborated, asserted or unassessed — never "measured if it has
        # a value", which would call one source's figure a cross-checked one.
        'confidence': getattr(entry.confidence, 'value', entry.confidence),
        'origin': getattr(entry.origin, 'value', entry.origin),
        'provenance': str(getattr(entry.confidence, 'value', entry.confidence)).lower(),
        # THE AUTHORITATIVE SPELLING. FactEntry.display is the one place in this product a figure
        # becomes words, and its docstring says why: "a figure printed one way and checked against
        # another spelling traces to nothing". The two hint vocabularies do NOT line up — the RGA
        # has money_exact, which exists because a compact £1.2M makes bids a few per cent apart
        # read as identical, and the browser has no such hint — so the browser is given the
        # spelling rather than the recipe. The numeric value travels too, for the chips and for an
        # edit.
        'display': entry.display,
    }


class _Snap:
    """Builds the snapshot, keeping the builder's own id sequences."""

    def __init__(self) -> None:
        self.blocks: Dict[str, Dict[str, Any]] = {}
        self.rows: List[Dict[str, Any]] = []
        self.b_seq = 0
        self.r_seq = 0

    def _bid(self) -> str:
        self.b_seq += 1
        return f'b-{self.b_seq}'

    def _rid(self) -> str:
        self.r_seq += 1
        return f'r-{self.r_seq}'

    def graph(self, chart: ChartBlock) -> str:
        """A real graph block, for a region to host. Not placed in a row: it lives in the page."""
        bid = self._bid()
        self.blocks[bid] = {
            'id': bid, 'type': 'graph', 'span': 1,
            'chartType': chart.chart_type,
            'series': [{'label': s.label, 'factRefs': list(s.fact_refs)} for s in chart.series],
            'title': '', 'comment': '', 'commentPrompt': '', 'commentOpen': False,
            'genOpen': False,
        }
        return bid

    def page(self, fitted: Fitted, findings: Dict[str, str]) -> str:
        region_blocks = {slot: self.graph(chart) for slot, chart in fitted.charts.items()}
        slots: Dict[str, Any] = {}
        for name, text in fitted.text.items():
            slots[name] = {'text': text}
        for name, items in fitted.lists.items():
            slots[name] = {'items': [{'heading': h, 'body': b} for h, b in items]}
        for name, refs in fitted.findings.items():
            # The fit carried the ids; what each one SAYS is resolved here, where the pack is.
            slots[name] = {'items': [{'heading': findings.get(r, r), 'body': ''} for r in refs]}
        for name, (columns, rows) in fitted.tables.items():
            slots[name] = {'columns': list(columns), 'rows': [list(r) for r in rows]}
        bid = self._bid()
        self.blocks[bid] = {
            'id': bid, 'type': 'layout', 'span': 1, 'layoutId': fitted.layout_key,
            'fill': {'slots': slots}, 'regionBlocks': region_blocks,
            'comment': '', 'commentPrompt': '', 'commentOpen': False, 'genOpen': False,
        }
        self.rows.append({'id': self._rid(), 'blocks': [bid]})
        return bid


def _findings_by_ref(pack: FactPack) -> Dict[str, str]:
    """What each finding says, by its id. A finding list on a page shows the sentence, never the
    id — an id on a board paper is the kind of thing that makes a reader distrust the rest of it."""
    out: Dict[str, str] = {}
    for finding in getattr(pack, 'findings', ()) or ():
        fid = getattr(finding, 'finding_id', None)
        if fid:
            out[str(fid)] = str(getattr(finding, 'detail', '') or fid)
    return out


def snapshot_dict(ast: ReportAST, pack: FactPack, *, title: str,
                  catalogue: Catalogue) -> Dict[str, Any]:
    """The snapshot. Raises NoFit, from layout_fit, if a section cannot be drawn at all.

    A refusal stops the whole report rather than dropping a section: a board paper missing its
    recommendation is worse than one that was not produced, and the caller reports which section
    had nowhere to go.
    """
    snap = _Snap()
    findings = _findings_by_ref(pack)
    for section in ast.sections:
        snap.page(fit(section, catalogue.layouts), findings)
    facts = {e.fact_id: _fact_out(e) for e in pack.facts}
    return {
        'reportTitle': title,
        'role': _ROLE,
        'period': pack.as_of,
        'rangePreset': _RANGE_PRESET,
        'locked': False,
        'isTemplate': False,
        # A REFERENCE to the pack, never a copy of it: a pack withdrawn after this report ran must
        # not live on inside it, and atbResolveStyle already falls back when a key is gone.
        'style': {'key': catalogue.pack_key},
        'blocks': snap.blocks,
        'rows': snap.rows,
        'bSeq': snap.b_seq,
        'rSeq': snap.r_seq,
        'facts': facts,
        'factOverrides': {},
        'build': {
            'pack_id': pack.pack_id,
            'pack_hash': pack.hash,
            'style_pack': catalogue.pack_key,
            'report_type_id': pack.report_type_id,
            'as_of': pack.as_of,
            'generated_at': str(pack.generated_at),
        },
    }


def render(ast: ReportAST, pack: FactPack, brief: Any, *, title: str,
           catalogue: Optional[Catalogue] = None, **_: Any) -> RenderedArtefact:
    """A peer of the deck and page renderers, so the post-check reads this drawing too.

    The hashes are computed the way render/html.py computes them, not read off the objects: a
    ReportAST has no `hash` and StyleBrief.version is a method, and RenderedArtefact treats a blank
    field in its reproducibility record as a post-check failure rather than a shrug.
    """
    if catalogue is None:
        raise ValueError('a builder snapshot needs the style pack it is laid out on')
    snap = snapshot_dict(ast, pack, title=title, catalogue=catalogue)
    content = json.dumps(snap, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return RenderedArtefact(
        content=content,
        media_type=MEDIA_TYPE,
        renderer=RENDERER,
        renderer_version=RENDERER_VERSION,
        pack_id=pack.pack_id,
        pack_hash=pack.hash,
        style_version=brief.version(),
        ast_hash=canonical_hash(ast.model_dump(mode='json')),
        style_provenance=dict(brief.provenance),
        style_disclosure=brief.disclosure(),
        coverage_disclosure=_coverage_line(pack),
    )
