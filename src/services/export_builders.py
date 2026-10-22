"""Build a Word note, a spreadsheet, a short deck or a chart from structured content the agent supplies.

These are the export tools the skills had marked NOT AVAILABLE. Each builder takes plain data
(title, sections, tables, slides, data points), returns bytes, and invents nothing: a figure the
agent did not pass is not in the file. Limits keep a file to what a person will read; above them the
builder refuses with a plain reason rather than truncating silently. The chart is SVG drawn here
without a plotting library, so it renders anywhere and carries no hidden data.
"""
from __future__ import annotations

import html
import io
from typing import Any, Dict, List, Mapping, Optional, Sequence

MAX_SECTIONS = 40
MAX_PARAGRAPHS = 400
MAX_TABLE_ROWS = 2000
MAX_COLUMNS = 30
MAX_SHEETS = 12
MAX_SLIDES = 30
MAX_BULLETS = 12
MAX_CHART_POINTS = 40
CHART_TYPES = ("bar", "line", "donut")


class ExportRefused(ValueError):
    """The content is outside what a builder will put in a file; the reason is the message."""


def _text(v: Any) -> str:
    return "" if v is None else str(v)


def _table(spec: Any, where: str) -> Optional[Dict[str, Any]]:
    """Normalise {columns: [...], rows: [[...]]}; None when no table was given."""
    if not spec:
        return None
    if not isinstance(spec, Mapping):
        raise ExportRefused(f"{where}: a table is an object with columns and rows")
    columns = [_text(c) for c in (spec.get("columns") or [])]
    rows = [list(r) if isinstance(r, (list, tuple)) else [r] for r in (spec.get("rows") or [])]
    if not columns:
        raise ExportRefused(f"{where}: a table needs columns")
    if len(columns) > MAX_COLUMNS:
        raise ExportRefused(f"{where}: {len(columns)} columns; the cap is {MAX_COLUMNS}")
    if len(rows) > MAX_TABLE_ROWS:
        raise ExportRefused(f"{where}: {len(rows)} rows; the cap is {MAX_TABLE_ROWS}")
    width = len(columns)
    rows = [(list(r) + [None] * width)[:width] for r in rows]
    return {"columns": columns, "rows": rows}


def _sections(sections: Any) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(sections, (list, tuple)) or not sections:
        raise ExportRefused("at least one section is needed: {heading, paragraphs, table}")
    if len(sections) > MAX_SECTIONS:
        raise ExportRefused(f"{len(sections)} sections; the cap is {MAX_SECTIONS}")
    paragraphs = 0
    for i, s in enumerate(sections, 1):
        if not isinstance(s, Mapping):
            raise ExportRefused(f"section {i} must be an object")
        paras = [_text(p) for p in (s.get("paragraphs") or []) if _text(p).strip()]
        bullets = [_text(b) for b in (s.get("bullets") or []) if _text(b).strip()]
        paragraphs += len(paras) + len(bullets)
        if paragraphs > MAX_PARAGRAPHS:
            raise ExportRefused(f"more than {MAX_PARAGRAPHS} paragraphs; split the document")
        out.append({"heading": _text(s.get("heading")).strip(), "paragraphs": paras, "bullets": bullets,
                    "table": _table(s.get("table"), f"section {i}")})
    return out


# ----------------------------------------------------------------------------- Word
def docx_bytes(title: str, sections: Any, *, notes: Optional[str] = None, prepared_for: Optional[str] = None) -> bytes:
    from docx import Document
    from docx.shared import Pt

    secs = _sections(sections)
    doc = Document()
    doc.add_heading(_text(title).strip() or "Untitled", level=0)
    if prepared_for:
        doc.add_paragraph(f"Prepared for {prepared_for}.")
    for s in secs:
        if s["heading"]:
            doc.add_heading(s["heading"], level=1)
        for p in s["paragraphs"]:
            doc.add_paragraph(p)
        for b in s["bullets"]:
            doc.add_paragraph(b, style="List Bullet")
        t = s["table"]
        if t:
            table = doc.add_table(rows=1, cols=len(t["columns"]))
            table.style = "Table Grid"
            for cell, col in zip(table.rows[0].cells, t["columns"]):
                cell.text = col
                for run in cell.paragraphs[0].runs:
                    run.font.bold = True
            for row in t["rows"]:
                cells = table.add_row().cells
                for cell, value in zip(cells, row):
                    cell.text = _text(value)
    if notes:
        p = doc.add_paragraph()
        run = p.add_run(_text(notes))
        run.font.size = Pt(9)
        run.italic = True
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ----------------------------------------------------------------------------- Excel
def xlsx_bytes(title: str, sheets: Any) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    if not isinstance(sheets, (list, tuple)) or not sheets:
        raise ExportRefused("at least one sheet is needed: {name, columns, rows}")
    if len(sheets) > MAX_SHEETS:
        raise ExportRefused(f"{len(sheets)} sheets; the cap is {MAX_SHEETS}")
    wb = Workbook()
    wb.remove(wb.active)
    used = set()
    for i, spec in enumerate(sheets, 1):
        if not isinstance(spec, Mapping):
            raise ExportRefused(f"sheet {i} must be an object")
        t = _table(spec, f"sheet {i}")
        if t is None:
            raise ExportRefused(f"sheet {i} needs columns and rows")
        name = (_text(spec.get("name")).strip() or f"Sheet{i}")[:31]
        for ch in '[]:*?/\\':
            name = name.replace(ch, " ")
        base, n = name, 2
        while name.lower() in used:
            name, n = f"{base[:28]} {n}", n + 1
        used.add(name.lower())
        ws = wb.create_sheet(title=name)
        ws.append(t["columns"])
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for row in t["rows"]:
            ws.append([_cell(v) for v in row])
        for col_idx, col in enumerate(t["columns"], 1):
            longest = max([len(_text(col))] + [len(_text(r[col_idx - 1])) for r in t["rows"][:200]])
            ws.column_dimensions[get_column_letter(col_idx)].width = min(60, max(10, longest + 2))
        ws.freeze_panes = "A2"
    wb.properties.title = _text(title)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _cell(v: Any) -> Any:
    """Numbers stay numbers so the sheet can be summed; everything else is text."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return v
    s = _text(v).strip()
    try:
        if s and s.replace(",", "").replace(".", "", 1).lstrip("-").isdigit():
            return float(s.replace(",", "")) if "." in s else int(s.replace(",", ""))
    except ValueError:
        pass
    return s


# ----------------------------------------------------------------------------- PowerPoint
def pptx_bytes(title: str, slides: Any, *, subtitle: Optional[str] = None) -> bytes:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    if not isinstance(slides, (list, tuple)) or not slides:
        raise ExportRefused("at least one slide is needed: {title, bullets, table}")
    if len(slides) > MAX_SLIDES:
        raise ExportRefused(f"{len(slides)} slides; the cap is {MAX_SLIDES}")
    prs = Presentation()
    first = prs.slides.add_slide(prs.slide_layouts[0])
    first.shapes.title.text = _text(title).strip() or "Untitled"
    if subtitle and len(first.placeholders) > 1:
        first.placeholders[1].text = _text(subtitle)
    for i, spec in enumerate(slides, 1):
        if not isinstance(spec, Mapping):
            raise ExportRefused(f"slide {i} must be an object")
        bullets = [_text(b) for b in (spec.get("bullets") or []) if _text(b).strip()]
        if len(bullets) > MAX_BULLETS:
            raise ExportRefused(f"slide {i}: {len(bullets)} bullets; the cap is {MAX_BULLETS}")
        t = _table(spec.get("table"), f"slide {i}")
        if t and len(t["rows"]) > 15:
            raise ExportRefused(f"slide {i}: a slide table holds at most 15 rows; put the rest in a spreadsheet")
        slide = prs.slides.add_slide(prs.slide_layouts[5 if t else 1])
        slide.shapes.title.text = _text(spec.get("title")).strip() or f"Slide {i}"
        if t:
            shape = slide.shapes.add_table(len(t["rows"]) + 1, len(t["columns"]), Inches(0.5), Inches(1.6),
                                           Inches(9), Inches(0.4) * (len(t["rows"]) + 1))
            table = shape.table
            for c, col in enumerate(t["columns"]):
                table.cell(0, c).text = col
            for r, row in enumerate(t["rows"], 1):
                for c, value in enumerate(row):
                    table.cell(r, c).text = _text(value)
            if bullets:
                box = slide.shapes.add_textbox(Inches(0.5), Inches(1.6) + Inches(0.4) * (len(t["rows"]) + 2), Inches(9), Inches(1.5))
                tf = box.text_frame
                tf.text = bullets[0]
                for b in bullets[1:]:
                    tf.add_paragraph().text = b
        else:
            body = slide.placeholders[1].text_frame
            body.text = bullets[0] if bullets else ""
            for b in bullets[1:]:
                p = body.add_paragraph()
                p.text = b
                p.font.size = Pt(18)
        note = _text(spec.get("notes")).strip()
        if note:
            slide.notes_slide.notes_text_frame.text = note
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


# ----------------------------------------------------------------------------- Chart (SVG)
def chart_svg(chart_type: str, title: str, data: Any, *, unit: str = "") -> bytes:
    kind = _text(chart_type).strip().lower()
    if kind not in CHART_TYPES:
        raise ExportRefused(f"chart_type must be one of {', '.join(CHART_TYPES)}")
    points = _points(data)
    if kind == "donut" and any(v < 0 for _, v in points):
        raise ExportRefused("a donut cannot show a negative value; use a bar chart")
    body = {"bar": _bar_svg, "line": _line_svg, "donut": _donut_svg}[kind](points, unit)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="480" viewBox="0 0 800 480" '
           f'font-family="Helvetica, Arial, sans-serif" font-size="13">'
           f'<rect width="800" height="480" fill="#ffffff"/>'
           f'<text x="24" y="34" font-size="20" font-weight="bold">{html.escape(_text(title))}</text>'
           f'{body}</svg>')
    return svg.encode("utf-8")


def _points(data: Any) -> List[tuple]:
    if not isinstance(data, (list, tuple)) or not data:
        raise ExportRefused("data is a list of {label, value}")
    if len(data) > MAX_CHART_POINTS:
        raise ExportRefused(f"{len(data)} points; the cap is {MAX_CHART_POINTS}")
    out = []
    for i, d in enumerate(data, 1):
        if not isinstance(d, Mapping) or "value" not in d:
            raise ExportRefused(f"point {i} needs a label and a value")
        try:
            v = float(d["value"])
        except (TypeError, ValueError):
            raise ExportRefused(f"point {i}: value {d['value']!r} is not a number") from None
        out.append((_text(d.get("label")) or f"#{i}", v))
    return out


def _fmt(v: float, unit: str) -> str:
    s = f"{v:,.0f}" if v == int(v) else f"{v:,.2f}"
    return f"{s} {unit}".strip()


def _bar_svg(points: Sequence[tuple], unit: str) -> str:
    left, top, width, height = 70, 60, 700, 340
    lo, hi = min(0.0, min(v for _, v in points)), max(0.0, max(v for _, v in points))
    span = (hi - lo) or 1.0
    zero_y = top + height * (hi / span)
    slot = width / len(points)
    bar_w = max(6.0, slot * 0.6)
    parts = [f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + width}" y2="{zero_y:.1f}" stroke="#555"/>']
    for i, (label, v) in enumerate(points):
        x = left + i * slot + (slot - bar_w) / 2
        h = abs(v) / span * height
        y = zero_y - h if v >= 0 else zero_y
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#2b6cb0"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{(y - 4) if v >= 0 else (y + h + 14):.1f}" text-anchor="middle" font-size="11">{html.escape(_fmt(v, unit))}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + height + 18}" text-anchor="middle" font-size="11">{html.escape(label[:18])}</text>')
    return "".join(parts)


def _line_svg(points: Sequence[tuple], unit: str) -> str:
    left, top, width, height = 70, 60, 700, 340
    lo, hi = min(v for _, v in points), max(v for _, v in points)
    span = (hi - lo) or 1.0
    step = width / max(1, len(points) - 1)
    coords = [(left + i * step, top + height - (v - lo) / span * height) for i, (_, v) in enumerate(points)]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    parts = [f'<polyline points="{path}" fill="none" stroke="#2b6cb0" stroke-width="2.5"/>']
    for (label, v), (x, y) in zip(points, coords):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#2b6cb0"/>')
        parts.append(f'<text x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="11">{html.escape(_fmt(v, unit))}</text>')
        parts.append(f'<text x="{x:.1f}" y="{top + height + 18}" text-anchor="middle" font-size="11">{html.escape(label[:14])}</text>')
    return "".join(parts)


def _donut_svg(points: Sequence[tuple], unit: str) -> str:
    import math
    total = sum(v for _, v in points)
    if total <= 0:
        raise ExportRefused("a donut needs values that add up to more than zero")
    cx, cy, r, ring = 260, 260, 150, 60
    palette = ["#2b6cb0", "#dd6b20", "#38a169", "#805ad5", "#d53f8c", "#718096", "#d69e2e", "#319795"]
    parts, angle = [], -math.pi / 2
    for i, (label, v) in enumerate(points):
        frac = v / total
        a2 = angle + frac * 2 * math.pi
        large = 1 if frac > 0.5 else 0
        x1, y1 = cx + r * math.cos(angle), cy + r * math.sin(angle)
        x2, y2 = cx + r * math.cos(a2), cy + r * math.sin(a2)
        xi1, yi1 = cx + (r - ring) * math.cos(a2), cy + (r - ring) * math.sin(a2)
        xi2, yi2 = cx + (r - ring) * math.cos(angle), cy + (r - ring) * math.sin(angle)
        colour = palette[i % len(palette)]
        if frac >= 0.9999:
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r - ring / 2}" fill="none" stroke="{colour}" stroke-width="{ring}"/>')
        else:
            parts.append(f'<path d="M{x1:.1f},{y1:.1f} A{r},{r} 0 {large} 1 {x2:.1f},{y2:.1f} L{xi1:.1f},{yi1:.1f} '
                         f'A{r - ring},{r - ring} 0 {large} 0 {xi2:.1f},{yi2:.1f} Z" fill="{colour}"/>')
        parts.append(f'<rect x="480" y="{80 + i * 24}" width="14" height="14" fill="{colour}"/>'
                     f'<text x="500" y="{92 + i * 24}" font-size="12">{html.escape(label[:28])}: {html.escape(_fmt(v, unit))} ({frac * 100:.1f}%)</text>')
        angle = a2
    return "".join(parts)


# ----------------------------------------------------------------------------- KPI card (SVG)
STATUS_COLOURS = {"green": "#38a169", "amber": "#d69e2e", "red": "#e53e3e", "grey": "#718096"}


def kpi_card_svg(title: str, value: Any, *, unit: str = "", movement_pct: Any = None, target: Any = None,
                 status: str = "grey", source: str = "") -> bytes:
    """A single-figure card: title, the value with its unit, the movement, the target and a status colour. The
    status is the caller's (from a stated rule), never judged here."""
    colour = STATUS_COLOURS.get(_text(status).strip().lower() or "grey")
    if colour is None:
        raise ExportRefused(f"status must be one of {', '.join(STATUS_COLOURS)}")
    if value is None or _text(value).strip() == "":
        raise ExportRefused("a KPI card needs a value")
    try:
        shown = _fmt(float(value), unit) if _text(value).replace(",", "").replace(".", "", 1).lstrip("-").isdigit() else f"{_text(value)} {unit}".strip()
    except (TypeError, ValueError):
        shown = f"{_text(value)} {unit}".strip()
    lines = []
    if movement_pct is not None:
        try:
            m = float(movement_pct)
            lines.append(f"{'+' if m >= 0 else ''}{m:.1f}% versus prior period")
        except (TypeError, ValueError):
            raise ExportRefused("movement_pct must be a number") from None
    if target is not None and _text(target).strip():
        lines.append(f"Target {_text(target)}")
    body = (f'<rect x="12" y="12" width="376" height="176" rx="12" fill="#ffffff" stroke="{colour}" stroke-width="4"/>'
            f'<text x="32" y="52" font-size="16" fill="#4a5568">{html.escape(_text(title))}</text>'
            f'<text x="32" y="110" font-size="40" font-weight="bold" fill="#1a202c">{html.escape(shown)}</text>')
    for i, line in enumerate(lines):
        body += f'<text x="32" y="{140 + i * 20}" font-size="13" fill="#4a5568">{html.escape(line)}</text>'
    if source:
        body += f'<text x="32" y="182" font-size="10" fill="#718096">Source: {html.escape(_text(source))}</text>'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200" viewBox="0 0 400 200" '
            f'font-family="Helvetica, Arial, sans-serif">{body}</svg>').encode("utf-8")
