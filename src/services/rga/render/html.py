"""The printable page. Deterministic, self-contained, and A4 portrait on paper.

The second rendering of the same composed report the deck draws (``render/pptx.py``): one
AST, one Fact Pack, one Style Brief, two files. Everything the deck guarantees holds here,
and for the same reasons -- see that module's docstring for the badge rule.

WHAT "PRINTS CORRECTLY" MEANS HERE

  * ``@page`` fixes the sheet: A4 portrait, 18/16/20 mm margins, and a running footer in the
    page margin boxes -- the report, its pack and hash on the left, "page N of M" on the
    right -- so a printed page can be traced even when it is separated from the rest.
  * The cover stands alone and every section starts a new page (``break-before: page``).
  * A table never splits a row (``tr { break-inside: avoid }``) and repeats its header row on
    every page it spans (``thead { display: table-header-group }``).
  * Colours print as they look (``print-color-adjust: exact``): a warm "UNASSESSED" that
    printed as grey would read as measured.

SELF-CONTAINED, AND WHY THAT IS ALSO A SAFETY PROPERTY

No script, no stylesheet link, no web font, no remote image: the page looks the same on any
machine and prints offline. Every piece of text -- the model's prose, labels, finding details
-- goes through ``html.escape``; style values from the brief are checked before they reach
the stylesheet (a colour must be a hex colour, a font name letters and spaces). The endpoint
that serves the page adds a Content-Security-Policy on top.

THE LOCALITY RULE, ON A PAGE

Every drawing of a fact is ONE element carrying its value, its ``[Fxxxx]`` marker and its
badge. ``extract_text`` returns one chunk per ``data-chunk`` element -- the page's equivalent
of the deck's text frame -- so the post-check's per-fact test means the same thing on both.
"""
from __future__ import annotations

import html as _html
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

from src.services.analytics.models import Confidence
from src.services.rga.models import (
    PLACEHOLDER,
    ChartBlock,
    FactEntry,
    FactPack,
    FindingListBlock,
    MetricBlock,
    NarrativeBlock,
    ReportAST,
    Section,
    TableBlock,
    canonical_hash,
)
from src.services.rga.render import RenderedArtefact
from src.services.rga.render.pptx import badge_text
from src.services.rga.style import PALETTE, StyleBrief

RENDERER = "html"
# Bump when the drawing changes: a rebuild that produced different bytes under the same
# version would silently break the reproducibility promise.
RENDERER_VERSION = "1.0.0"
MEDIA_TYPE = "text/html; charset=utf-8"

_LEGACY = "LEGACY_UNVERIFIED"
_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
_FONT = re.compile(r"^[A-Za-z0-9 \-]{1,40}$")


def _e(value: Any) -> str:
    return _html.escape(str(value), quote=True)


def _colour(palette: Dict[str, str], key: str) -> str:
    value = (palette or {}).get(key)
    return value if isinstance(value, str) and _HEX.match(value) else PALETTE[key]


def _font(name: Any, fallback: str) -> str:
    stack = f"{fallback}"
    if isinstance(name, str) and _FONT.match(name):
        stack = f"'{name}', {fallback}"
    return stack


def _css_string(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def _coverage_line(pack: FactPack) -> str:
    """The same words as the deck's front page -- see render/pptx._coverage_line."""
    unmeasured = sum(1 for f in pack.facts if f.confidence is Confidence.UNASSESSED)
    if not unmeasured:
        return f"all {len(pack.facts)} measures assessed"
    return f"{unmeasured} of {len(pack.facts)} measures not assessed"


def _stamp(entry: FactEntry, show_badges: bool) -> str:
    badge = badge_text(entry, show_badges)
    return f"[{entry.fact_id}] · {badge}" if badge else f"[{entry.fact_id}]"


def _tone(entry: FactEntry) -> str:
    if entry.confidence is Confidence.UNASSESSED or entry.origin.value == _LEGACY:
        return "warn"
    return "ok"


class _Page:
    """Accumulates one section's footnotes while its blocks are drawn."""

    def __init__(self, pack: FactPack, show_badges: bool) -> None:
        self.pack = pack
        self.show_badges = show_badges
        self.notes: List[str] = []

    def note(self, entry: FactEntry) -> None:
        badge = badge_text(entry, self.show_badges)
        parts = [f"[{entry.fact_id}]"] + ([badge] if badge else []) + [
            entry.label, entry.derivation, entry.provenance_id]
        text = " · ".join(parts)
        if text not in self.notes:
            self.notes.append(text)

    def figure(self, entry: FactEntry, css: str = "fig") -> str:
        """Value, marker and badge in one element: the locality unit."""
        self.note(entry)
        return (f'<span class="{css} {_tone(entry)}" data-chunk>'
                f'<span class="val">{_e(entry.display)}</span> '
                f'<span class="stamp">{_e(_stamp(entry, self.show_badges))}</span></span>')

    def stamp(self, entry: FactEntry) -> str:
        self.note(entry)
        return (f'<span class="stamp {_tone(entry)}" data-chunk>'
                f'{_e(_stamp(entry, self.show_badges))}</span>')


def render(
    ast: ReportAST,
    pack: FactPack,
    brief: StyleBrief,
    *,
    title: str = "Executive procurement summary",
) -> RenderedArtefact:
    """Draw the page and return it with its reproducibility record."""
    palette = brief.get("report.style.palette") or {}
    body_font = _font(brief.get("report.style.font.body"), "Arial, Helvetica, sans-serif")
    mono_font = _font(brief.get("report.style.font.mono"), "'Courier New', monospace")
    show_badges = bool(brief.get("report.style.show_confidence_badges"))
    show_footnotes = bool(brief.get("report.style.show_provenance_footnotes"))

    ast_hash = canonical_hash(ast.model_dump(mode="json"))
    style_version = brief.version()
    coverage = _coverage_line(pack)

    out: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="en-GB"><head><meta charset="utf-8">',
        f"<title>{_e(title)} — {_e(pack.scope.get('period_label') or pack.pack_id)}</title>",
        "<style>",
        _stylesheet(palette, body_font, mono_font, title, pack),
        "</style></head><body>",
        _cover(pack, brief, title, ast_hash, style_version, coverage),
    ]
    for section in ast.sections:
        out.append(_section(section, pack, show_badges, show_footnotes))
    out.append("</body></html>\n")

    return RenderedArtefact(
        content="\n".join(out).encode("utf-8"),
        media_type=MEDIA_TYPE,
        renderer=RENDERER,
        renderer_version=RENDERER_VERSION,
        pack_id=pack.pack_id,
        pack_hash=pack.hash,
        style_version=style_version,
        ast_hash=ast_hash,
        style_provenance=dict(brief.provenance),
        style_disclosure=brief.disclosure(),
        coverage_disclosure=coverage,
    )


def _stylesheet(palette, body_font, mono_font, title, pack) -> str:
    c = {k: _colour(palette, k) for k in PALETTE}
    footer = _css_string(f"{title} · pack {pack.pack_id} · hash {pack.hash[:12]}")
    return f"""
@page {{
  size: A4 portrait;
  margin: 18mm 16mm 20mm;
  @bottom-left {{ content: {footer}; font: 7pt {body_font}; color: {c['mute']}; }}
  @bottom-right {{ content: "page " counter(page) " of " counter(pages); font: 7pt {body_font}; color: {c['mute']}; }}
}}
html {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
body {{ margin: 0; font: 10.5pt/1.45 {body_font}; color: {c['ink']}; background: {c['paper']}; }}
@media screen {{
  body {{ background: #EEF1F3; padding: 12mm 0; }}
  .cover, .section {{ width: 178mm; margin: 0 auto 10mm; padding: 16mm; background: {c['paper']};
    box-shadow: 0 1px 4px rgba(10,42,67,.15); }}
}}
.cover {{ break-after: page; }}
.band {{ background: {c['ocean_dark']}; color: {c['paper']}; padding: 14mm 10mm 10mm; margin-bottom: 10mm; }}
.band h1 {{ margin: 0 0 4mm; font-size: 26pt; line-height: 1.15; }}
.band .scope {{ font: 11pt {mono_font}; }}
.coverage {{ font-size: 13pt; font-weight: 700; margin: 0 0 3mm; }}
.coverage.warn {{ color: {c['warm']}; }}
.coverage.ok {{ color: {c['ok']}; }}
.disclosure, .provenance {{ color: {c['mute']}; font-size: 8.5pt; margin: 0 0 2mm; }}
.provenance {{ font-family: {mono_font}; margin-top: 30mm; }}
.section {{ break-before: page; }}
.section h2 {{ font-size: 17pt; color: {c['ocean_dark']}; margin: 0 0 6mm; padding-bottom: 2mm;
  border-bottom: 2pt solid {c['ocean']}; break-after: avoid; }}
.metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; margin: 0 0 6mm; }}
.metric {{ border: 0.75pt solid {c['rule']}; border-radius: 2mm; padding: 3mm 4mm; break-inside: avoid; }}
.metric.primary {{ grid-column: 1 / -1; }}
.metric .label {{ color: {c['mute']}; font-size: 9pt; }}
.metric .val {{ display: block; font: 700 18pt {mono_font}; color: {c['ink']}; margin: 1mm 0; }}
.metric.primary .val {{ font-size: 24pt; }}
.stamp {{ font-size: 8pt; font-weight: 700; }}
.ok {{ color: {c['ok']}; }}
.warn {{ color: {c['warm']}; }}
.fig .val {{ font-family: {mono_font}; color: {c['ink']}; }}
p.narrative {{ margin: 0 0 1mm; font-size: 11pt; }}
.stamps {{ margin: 0 0 5mm; color: {c['mute']}; }}
.stamps .stamp {{ margin-right: 4mm; }}
table {{ width: 100%; border-collapse: collapse; margin: 0 0 6mm; font-size: 9.5pt; }}
thead {{ display: table-header-group; }}
tr {{ break-inside: avoid; }}
th {{ background: {c['ocean_dark']}; color: {c['paper']}; text-align: left; padding: 2mm 3mm; }}
td {{ border-bottom: 0.75pt solid {c['rule']}; padding: 2mm 3mm; vertical-align: top; }}
.chart {{ margin: 0 0 6mm; }}
.bar-row {{ display: grid; grid-template-columns: 36% 38% 26%; align-items: center; gap: 2mm;
  margin-bottom: 1.5mm; break-inside: avoid; font-size: 9pt; }}
.bar {{ height: 4mm; background: {c['ocean']}; }}
.findings {{ margin: 0 0 6mm; padding-left: 5mm; font-size: 9.5pt; }}
.findings li {{ break-inside: avoid; margin-bottom: 1mm; }}
.findings .blocking {{ color: {c['warm']}; }}
.none {{ color: {c['mute']}; }}
.notes {{ border-top: 0.75pt solid {c['rule']}; margin-top: 6mm; padding-top: 2mm; color: {c['mute']};
  font-size: 7.5pt; list-style: none; padding-left: 0; }}
.notes li {{ break-inside: avoid; margin-bottom: 0.8mm; word-break: break-word; }}
"""


def _cover(pack, brief, title, ast_hash, style_version, coverage) -> str:
    scope = pack.scope
    scope_line = " · ".join(str(v) for v in (
        scope.get("period_label"), scope.get("currency"), f"as at {pack.as_of}") if v)
    tone = "warn" if "not assessed" in coverage else "ok"
    stamp = (f"pack {pack.pack_id} · hash {pack.hash[:12]} · style {style_version} · "
             f"ast {ast_hash[:12]} · renderer {RENDERER}/{RENDERER_VERSION}")
    return (
        '<header class="cover">'
        f'<div class="band"><h1 data-chunk>{_e(title)}</h1>'
        f'<div class="scope" data-chunk>{_e(scope_line)}</div></div>'
        f'<p class="coverage {tone}" data-chunk>{_e(coverage)}</p>'
        f'<p class="disclosure" data-chunk>{_e(brief.disclosure())}</p>'
        f'<p class="provenance" data-chunk>{_e(stamp)}</p>'
        "</header>"
    )


def _substitute(text: str, pack: FactPack) -> str:
    """``{{F0042}}`` -> the fact's display, as in the deck. An unknown reference stays
    visible as the placeholder; the post-check catches it."""
    return PLACEHOLDER.sub(
        lambda m: pack.fact(m.group(1)).display if pack.fact(m.group(1)) else m.group(0), text)


def _section(section: Section, pack: FactPack, show_badges: bool, show_footnotes: bool) -> str:
    page = _Page(pack, show_badges)
    parts: List[str] = [f'<section class="section"><h2 data-chunk>{_e(section.title)}</h2>']
    metrics: List[str] = []

    def flush_metrics() -> None:
        if metrics:
            parts.append('<div class="metrics">' + "".join(metrics) + "</div>")
            metrics.clear()

    for block in section.blocks:
        if isinstance(block, MetricBlock):
            metrics.append(_metric(block, page))
            continue
        flush_metrics()
        if isinstance(block, NarrativeBlock):
            parts.append(_narrative(block, page))
        elif isinstance(block, TableBlock):
            parts.append(_table(block, page))
        elif isinstance(block, ChartBlock):
            parts.append(_chart(block, page))
        elif isinstance(block, FindingListBlock):
            parts.append(_findings(block, pack))
    flush_metrics()

    if show_footnotes and page.notes:
        parts.append('<ol class="notes">'
                     + "".join(f"<li data-chunk>{_e(n)}</li>" for n in page.notes) + "</ol>")
    parts.append("</section>")
    return "".join(parts)


def _metric(block: MetricBlock, page: _Page) -> str:
    entry = page.pack.fact(block.fact_ref)
    if entry is None:
        return (f'<div class="metric warn" data-chunk>[unresolved fact '
                f'{_e(block.fact_ref)}]</div>')
    css = "metric primary" if block.emphasis == "primary" else "metric"
    return (f'<div class="{css}"><div class="label" data-chunk>{_e(entry.label)}</div>'
            f'{page.figure(entry)}</div>')


def _narrative(block: NarrativeBlock, page: _Page) -> str:
    """Prose, then a stamp for every fact it rests on -- quoted or not (as the deck)."""
    stamps: List[str] = []
    seen = set()
    for ref in list(block.fact_refs) + block.placeholders():
        if ref in seen:
            continue
        seen.add(ref)
        entry = page.pack.fact(ref)
        if entry is not None:
            stamps.append(page.stamp(entry))
    html_out = f'<p class="narrative" data-chunk>{_e(_substitute(block.text, page.pack))}</p>'
    if stamps:
        html_out += '<div class="stamps">' + " ".join(stamps) + "</div>"
    return html_out


def _table(block: TableBlock, page: _Page) -> str:
    head = "".join(f"<th data-chunk>{_e(c)}</th>" for c in block.columns)
    rows = []
    for row in block.rows:
        cells = []
        for raw in row:
            entry = page.pack.fact(raw)
            if entry is not None:
                # The figure, its marker and its badge in the cell: a table-only figure must
                # never read as a bare value (the deck's original bug).
                cells.append(f"<td>{page.figure(entry, css='cell')}</td>")
            else:
                cells.append(f"<td data-chunk>{_e(raw)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>")


def _chart(block: ChartBlock, page: _Page) -> str:
    """Labelled bars, as in the deck: no canvas, every number stays text."""
    entries = [(series.label, page.pack.fact(ref))
               for series in block.series for ref in series.fact_refs]
    measured = [e for _, e in entries if e is not None and e.value is not None]
    peak = max((abs(e.value) for e in measured), default=None)
    rows = []
    for label, entry in entries:
        if entry is None:
            continue
        width = 0.0
        if entry.value is not None and peak:
            width = round(float(abs(entry.value)) / float(peak) * 100, 1)
        rows.append(
            '<div class="bar-row">'
            f'<span data-chunk>{_e(label)} — {_e(entry.label)}</span>'
            f'<span class="track"><span class="bar" style="display:block;width:{width}%"></span></span>'
            f"{page.figure(entry)}</div>")
    return '<div class="chart">' + "".join(rows) + "</div>"


def _findings(block: FindingListBlock, pack: FactPack) -> str:
    wanted = set(block.finding_refs)
    listed = [f for f in pack.findings if not wanted or f.finding_id in wanted]
    if not listed:
        return '<p class="none" data-chunk>No findings raised against this pack.</p>'
    items = "".join(
        f'<li class="{"blocking" if f.blocks_release else ""}" data-chunk>'
        f"{_e(f.severity.value)} · {_e(f.code.value)} — {_e(f.detail)}</li>"
        for f in listed)
    return f'<ul class="findings">{items}</ul>'


# --------------------------------------------------------------------------
# What the post-check reads
# --------------------------------------------------------------------------


class _Chunks(HTMLParser):
    """Text of every outermost ``data-chunk`` element, whitespace collapsed."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: List[str] = []
        self._depth = 0          # nesting inside the current chunk, 0 = outside one
        self._buf: List[str] = []
        self._in_style = False

    def handle_starttag(self, tag, attrs) -> None:
        if tag == "style":
            self._in_style = True
            return
        if self._depth:
            self._depth += 1
        elif any(name == "data-chunk" for name, _ in attrs):
            self._depth = 1
            self._buf = []

    def handle_endtag(self, tag) -> None:
        if tag == "style":
            self._in_style = False
            return
        if self._depth:
            self._depth -= 1
            if self._depth == 0:
                text = " ".join("".join(self._buf).split())
                if text:
                    self.chunks.append(text)

    def handle_data(self, data) -> None:
        if self._depth and not self._in_style:
            self._buf.append(data)


def extract_text(content: bytes) -> List[str]:
    """Every string the page shows, one chunk per element the post-check reasons about."""
    parser = _Chunks()
    parser.feed(content.decode("utf-8"))
    parser.close()
    return parser.chunks
