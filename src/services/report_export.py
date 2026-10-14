"""Turn the report builder's rendered page into a file: a PDF, or an Excel workbook.

The browser already assembles a complete, self-contained print document (charts are
PNG data-URIs), so the server only needs HTML + CSS -> PDF. No headless browser.

The HTML arrives from a caller, so rendering it must not become a way to make this
server fetch things. WeasyPrint follows ``<img src>``, ``<link href>`` and CSS
``url()`` by default; the fetcher below refuses everything except ``data:`` URIs, so
a ``file:///etc/passwd`` or an ``http://169.254.169.254/`` in the page is refused,
not fetched.
"""
from __future__ import annotations

import html as _html
import io
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_HTML_BYTES = 12 * 1024 * 1024
MAX_TABLE_ROWS = 50_000
_BUCKET_DEFAULT = "procwisemvp"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class ExportRefused(ValueError):
    """The request cannot be turned into a file; the message says why."""


def _data_only_fetcher():
    # fail_on_errors: a refused URL stops the render instead of being skipped with a warning.
    from weasyprint.urls import URLFetcher
    return URLFetcher(allowed_protocols={"data"}, fail_on_errors=True, allow_redirects=False)


WATERMARK_TEXT = "PRESENTATION DATA - NOT REAL"
_WATERMARK_CSS = (
    "@page { @top-center { content: \"" + WATERMARK_TEXT + "\"; color: #b00020; font: bold 9pt sans-serif; } "
    "@bottom-center { content: \"" + WATERMARK_TEXT + "\"; color: #b00020; font: bold 9pt sans-serif; } }"
    ".rb-watermark { position: fixed; top: 38%; left: 4%; width: 92%; text-align: center; font: bold 54pt sans-serif; "
    "color: rgba(176,0,32,0.16); transform: rotate(-24deg); z-index: 9999; }")


def watermark_html() -> str:
    return f'<div class="rb-watermark">{WATERMARK_TEXT}</div>'


def watermark_css() -> str:
    return _WATERMARK_CSS


def render_pdf(html: str, css: str = "") -> bytes:
    if not html or not html.strip():
        raise ExportRefused("there is no page to export")
    if len(html.encode("utf-8")) > MAX_HTML_BYTES:
        raise ExportRefused("the page is too large to export")
    from weasyprint import CSS, HTML
    from weasyprint.urls import FatalURLFetchingError
    fetcher = _data_only_fetcher()
    try:
        sheets = [CSS(string=css, url_fetcher=fetcher)] if css and css.strip() else []
        return HTML(string=html, url_fetcher=fetcher).write_pdf(stylesheets=sheets)
    except FatalURLFetchingError as exc:
        raise ExportRefused(f"the page refers to an outside resource, which is not fetched: {exc}") from exc


def render_xlsx(tables: List[Dict[str, Any]]) -> bytes:
    """One sheet per table: ``{title, columns: [..], rows: [[..], ..]}``."""
    if not tables:
        raise ExportRefused("there are no tables to export")
    from openpyxl import Workbook
    wb = Workbook()
    wb.remove(wb.active)
    used = set()
    for i, t in enumerate(tables, 1):
        cols, rows = t.get("columns") or [], t.get("rows") or []
        if len(rows) > MAX_TABLE_ROWS:
            raise ExportRefused(f"table {i} has more than {MAX_TABLE_ROWS} rows")
        name = re.sub(r"[\[\]:*?/\\]", " ", str(t.get("title") or f"Table {i}")).strip()[:31] or f"Table {i}"
        base, n = name, 2
        while name.lower() in used:
            name = f"{base[:28]} {n}"; n += 1
        used.add(name.lower())
        ws = wb.create_sheet(name)
        ws.append([_cell(c) for c in cols])
        for r in rows:
            ws.append([_cell(v) for v in r])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _cell(v: Any) -> Any:
    # A cell starting with = + - @ is a formula in Excel; a report value must stay text.
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@"):
        return "'" + v
    return v if isinstance(v, (int, float, bool)) or v is None else str(v)


def storage_key(report_id: Optional[str], name: str, fmt: str, stamp: str) -> str:
    return f"reports/exports/{_SAFE.sub('_', str(report_id or 'adhoc'))}/{_SAFE.sub('_', name)[:80]}_{stamp}.{fmt}"


def store(content: bytes, key: str, content_type: str) -> bool:
    """Keep a copy in S3. False (not an exception) when it cannot: the download still works."""
    try:
        from src.services.egress import Purpose, aws_client
        bucket = os.getenv("S3_BUCKET_NAME", _BUCKET_DEFAULT)
        aws_client("s3", purpose=Purpose.OBJECT_STORAGE).put_object(
            Bucket=bucket, Key=key, Body=content, ContentType=content_type)
        return True
    except Exception:
        logger.warning("report export: could not store %s", key, exc_info=True)
        return False


def record_url(report_id: str, key: str) -> bool:
    """Write the S3 key to proc.bp_reports.report_url (a column nothing wrote until now)."""
    try:
        from src.services.db import get_conn
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("UPDATE proc.bp_reports SET report_url = %s WHERE id::text = %s",
                        (f"s3://{os.getenv('S3_BUCKET_NAME', _BUCKET_DEFAULT)}/{key}", str(report_id)))
            return cur.rowcount > 0
    except Exception:
        logger.warning("report export: could not record report_url for %s", report_id, exc_info=True)
        return False


# ---- A report's export is built from the SAME payload the page shows ---------------------------------

def _fmt(v: Any, fmt: str) -> str:
    if v is None:
        return "not assessed" if fmt == "percent" else "no data"
    if fmt == "percent":
        return f"{v:,.1f}%"
    if fmt == "currency":
        return f"{v:,.0f}"
    if fmt in ("count", "days", "score"):
        return f"{v:,.1f}" if fmt != "count" else f"{v:,.0f}"
    return f"{v:,.2f}"


def _esc(x: Any) -> str:
    return _html.escape(str(x), quote=True)


def _tile_rows(tile: Dict[str, Any], pretty: bool = False) -> List[List[Any]]:
    """Every tile as one flat table, whatever it is drawn as, so nothing can vanish from an export.
    ``pretty`` formats numbers for a page; a workbook keeps them as numbers."""
    r = tile.get("result") or {}
    fmt = r.get("format") or "number"
    num = (lambda v: _fmt(v, fmt) if isinstance(v, (int, float)) or v is None else v) if pretty else (lambda v: v)
    if "columns" in r:
        body = [[c if i == 0 else num(c) for i, c in enumerate(row)] for row in r["rows"]]
        return [r["columns"]] + body + ([["Total", num(r["total"])]] if r.get("total") is not None else [])
    if "labels" in r:
        header = ["", *[s["name"] for s in r["series"]]]
        return [header] + [[lab, *[num(s["data"][i]) for s in r["series"]]] for i, lab in enumerate(r["labels"])]
    if "items" in r:
        return [["Finding"]] + [[i["text"]] for i in r["items"]]
    if "value" in r:
        rows = [["Value", num(r["value"])]]
        cmp_ = tile.get("comparison")
        if cmp_:
            rows += [["Compared with", f'{cmp_["window"]["from"]} to {cmp_["window"]["to"]}'], ["Compared value", num(cmp_["value"])],
                     ["Change", num(cmp_["delta"])]]
        if r.get("assessed") is not None:
            rows.append(["Assessed", r["assessed"]])
        return rows
    return [["Status", tile.get("status")]]


def parameters_lines(tile: Dict[str, Any]) -> List[str]:
    p = tile.get("params") or {}
    per = p.get("period") or {}
    lines = [f"Metric: {', '.join(p.get('metrics') or [])}" + (f" (derived: {p['derive']})" if p.get("derive") else ""),
             f"Grouped by: {', '.join(p.get('group_by') or []) or 'none'}",
             f"Filters: {p.get('filters') or 'none'}",
             f"Period: {per.get('from')} to {per.get('to')}" + (f" (running; measured to {per.get('effective_to')})" if per.get("partial") else ""),
             f"Comparison: {p.get('comparison') or 'metric default'}" + (f"; target {p['target']}" if p.get("target") is not None else ""),
             f"Data mode: {tile.get('data_mode')}; as of {per.get('as_of')} ({per.get('anchor')})"]
    return lines


def report_appendix_html(payload: Dict[str, Any], *, title: str, generated_by: str) -> str:
    """The parameters block, then every tile in its true state with its parameters and its data."""
    mark = payload.get("marker")
    head = (f"<h1>{_esc(title)}</h1><p>Period {_esc(payload['period']['from'])} to {_esc(payload['period']['to'])} · "
            f"as of {_esc(payload['as_of'])} · generated by {_esc(generated_by)} · data: {_esc(payload['data_mode'])}</p>")
    if mark:
        head += f'<p class="rb-marker"><b>{_esc(mark)}</b> — {_esc(payload.get("org") or "")}</p>'
    out = ['<section class="rb-params">', head, "</section>"]
    for t in payload["tiles"]:
        out.append(f'<section class="rb-tile" data-tile="{_esc(t["id"])}"><h2>{_esc(t["id"])} <small>[{_esc(t["viz"])}]</small></h2>')
        if t.get("marker"):
            out.append(f'<p class="rb-marker"><b>{_esc(t["marker"])}</b></p>')
        out.append("<ul>" + "".join(f"<li>{_esc(l)}</li>" for l in parameters_lines(t)) + "</ul>")
        if t["status"] != "ok":
            out.append(f'<p class="rb-state"><b>{_esc(t["status"].replace("_", " "))}</b>: {_esc(t.get("reason") or "")}</p>')
        else:
            rows = _tile_rows(t, pretty=True)
            out.append("<table>" + "".join("<tr>" + "".join(f"<td>{_esc('' if c is None else c)}</td>" for c in r) + "</tr>"
                                            for r in rows) + "</table>")
        for c in t.get("checks") or []:
            out.append(f'<p class="rb-check">check: {_esc(c["code"])}' + (f' — {_esc(c["note"])}' if c.get("note") else "") + "</p>")
        out.append("</section>")
    return "".join(out)


APPENDIX_CSS = ("section{page-break-inside:avoid;margin:0 0 12pt} table{border-collapse:collapse;font:8pt sans-serif} "
                "td{border:1px solid #ccc;padding:2pt 5pt} h1{font:bold 16pt sans-serif} h2{font:bold 11pt sans-serif} "
                "li,p{font:8pt sans-serif} .rb-marker{color:#b00020}")


def payload_tables(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One sheet per tile (its data, with its parameters beneath it) after a parameters sheet."""
    marked = bool(payload.get("marker"))
    sheets = [{"title": "Parameters",
               # the mark is the FIRST row of the first sheet, not a line somewhere below a header
               "columns": ["NOTE", payload["marker"]] if marked else ["Item", "Value"],
               "rows": [*([["Organisation", payload.get("org")]] if marked else []),
                        ["Data mode", payload["data_mode"]], ["As of", payload["as_of"]],
                        ["Period", f"{payload['period']['from']} to {payload['period']['to']}"]]}]
    for t in payload["tiles"]:
        rows = _tile_rows(t) if t["status"] == "ok" else [["Status", t["status"].replace("_", " ")], ["Reason", t.get("reason")]]
        cols, body = (rows[0], rows[1:]) if t["status"] == "ok" else (["", ""], rows)
        sheet = {"title": str(t["id"]), "columns": [str(c) for c in cols], "rows": [list(r) for r in body]}
        sheet["rows"] += [[], ["Parameters"]] + [[l] for l in parameters_lines(t)]
        if t.get("marker"):
            sheet["rows"] = [sheet["columns"]] + sheet["rows"]
            sheet["columns"] = [t["marker"]]
        sheets.append(sheet)
    return sheets
