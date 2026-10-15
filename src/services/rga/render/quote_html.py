"""The customer's quote: one page drawn from a QuoteAST, and the PDF printed from it.

The same rules as the printable report (``render/html.py``) -- self-contained, every string
escaped, A4 on paper -- with one deliberate difference. A report draws each figure with its
``[Fxxxx]`` marker, a confidence badge and a provenance footnote, because its reader is
inside the company and is entitled to know how well a figure is known. A quote's reader is
the customer. Nothing on this page says where a figure came from; the record that does
(pack hash, AST hash, renderer version) travels beside the page in ``RenderedArtefact`` and
is never printed on it.

The renderer decides nothing. Which lines, which columns and which terms come from the
tree; every word and figure comes from the pack; the headings are fixed here. The PDF is a
print of the HTML bytes and nothing else, so the two carry the same AST hash and the
post-check that passed the page has passed the PDF.

THE CHUNKS

``extract_text`` (shared with the report page) returns one chunk per ``data-chunk``
element. A line is ONE chunk -- its whole row -- and a total is one chunk holding its label
and its amount, so the post-check can ask "is this line drawn" and "does this total read
what the fact says" of a single string.
"""
from __future__ import annotations

from typing import List

from src.services.rga.models import canonical_hash
from src.services.rga.quote_models import (
    LetterheadBlock,
    LineTableBlock,
    NotesBlock,
    PaymentBlock,
    QuoteAST,
    QuotePack,
    SectionBlock,
    SignatureBlock,
    TermsBlock,
    TotalsBlock,
    long_date,
)
from src.services.rga.render import RenderedArtefact
from src.services.rga.render.html import PAGE_CSP, _colour, _e, _font
from src.services.rga.style import PALETTE, StyleBrief

RENDERER = "quote_html"
PDF_RENDERER = "quote_pdf"
# Bump when the drawing changes: a rebuild that produced different bytes under the same
# version would silently break the reproducibility promise.
RENDERER_VERSION = "1.1.0"   # 1.1.0: letterhead, seller's passages, discount, tax
MEDIA_TYPE = "text/html; charset=utf-8"
PDF_MEDIA_TYPE = "application/pdf"

TITLE = "Quotation"
NUMBER_LABEL = "Quote number"
REFERENCE_LABEL = "Reference"
HEADINGS = {"line_no": "#", "sku": "SKU", "mpn": "Part no.", "description": "Description",
            "quantity": "Qty", "unit_of_measure": "Unit", "unit_price": "Unit price",
            "discount_pct": "Discount", "line_total": "Line total"}
_RIGHT = frozenset({"quantity", "unit_price", "discount_pct", "line_total"})
# The page embeds its one picture, the seller's logo, as a data: URI and fetches nothing.
QUOTE_CSP = PAGE_CSP + "; img-src data:"


def scope_line(pack: QuotePack) -> str:
    """The one sentence of dates and currency. Shared with the post-check, which licenses
    the day and year it prints."""
    who = pack.identity
    return (f"Dated {long_date(who.quote_date)} · Valid until {long_date(who.valid_until)} · "
            f"Prices in {who.currency}, excluding tax")


def render(ast: QuoteAST, pack: QuotePack, brief: StyleBrief) -> RenderedArtefact:
    """Draw the page and return it with its reproducibility record."""
    palette = brief.get("report.style.palette") or {}
    body_font = _font(brief.get("report.style.font.body"), "Arial, Helvetica, sans-serif")
    mono_font = _font(brief.get("report.style.font.mono"), "'Courier New', monospace")
    who = pack.identity

    head = [f'<h1 data-chunk>{_e(TITLE)}</h1>']
    if who.number:
        head.append(f'<p class="ref" data-chunk>{_e(NUMBER_LABEL)} {_e(who.number)}</p>')
    head.append(f'<p class="ref" data-chunk>{_e(REFERENCE_LABEL)} {_e(who.quote_ref)}</p>')
    to = who.account_name + (f" — {who.contact_name}" if who.contact_name else "")
    head.append(f'<p class="to" data-chunk>For {_e(to)}</p>')
    head.append(f'<p class="scope" data-chunk>{_e(scope_line(pack))}</p>')

    out: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="en-GB"><head><meta charset="utf-8">',
        f'<meta http-equiv="Content-Security-Policy" content="{QUOTE_CSP}">',
        f"<title>{_e(TITLE)} {_e(who.number or who.quote_ref)}</title>",
        "<style>", _stylesheet(palette, body_font, mono_font), "</style></head><body>",
        '<article class="quote">',
    ]
    # A letterhead heads the page wherever the layout lists it.
    out.extend(_letterhead(pack) for b in ast.blocks if isinstance(b, LetterheadBlock))
    out.append("<header>" + "".join(head) + "</header>")
    for block in ast.blocks:
        if isinstance(block, LineTableBlock):
            out.append(_lines(block, pack))
        elif isinstance(block, TotalsBlock):
            out.append(_totals(block, pack))
        elif isinstance(block, NotesBlock):
            out.append(_listed("Notes", block.note_refs, pack.note))
        elif isinstance(block, TermsBlock):
            out.append(_listed("Terms", block.term_refs, pack.term))
        elif isinstance(block, PaymentBlock):
            out.append(_listed("Payment", [block.term_ref], pack.term))
        elif isinstance(block, SignatureBlock):
            out.append(_signature(block, pack))
        elif isinstance(block, SectionBlock):
            out.append(_section(block, pack))
    if pack.seller is not None and pack.seller.footer_note:
        out.append(f'<footer><p data-chunk>{_e(pack.seller.footer_note)}</p></footer>')
    out.append("</article></body></html>\n")

    return RenderedArtefact(
        content="\n".join(out).encode("utf-8"),
        media_type=MEDIA_TYPE,
        renderer=RENDERER,
        renderer_version=RENDERER_VERSION,
        pack_id=pack.pack_id,
        pack_hash=pack.hash,
        style_version=brief.version(),
        ast_hash=canonical_hash(ast.model_dump(mode="json")),
        style_provenance=dict(brief.provenance),
        style_disclosure=brief.disclosure(),
    )


def to_pdf(page: RenderedArtefact) -> RenderedArtefact:
    """The same page as a PDF. Takes the rendered page, not a tree: there is no second
    drawing to disagree with the first."""
    if page.renderer != RENDERER:
        raise ValueError(f"a quote PDF is printed from a {RENDERER} page, not {page.renderer!r}")
    from src.services.report_export import render_pdf

    return RenderedArtefact(
        content=render_pdf(page.content.decode("utf-8")),
        media_type=PDF_MEDIA_TYPE,
        renderer=PDF_RENDERER,
        renderer_version=RENDERER_VERSION,
        pack_id=page.pack_id,
        pack_hash=page.pack_hash,
        style_version=page.style_version,
        ast_hash=page.ast_hash,
        style_provenance=dict(page.style_provenance),
        style_disclosure=page.style_disclosure,
    )


def _unresolved(ref: str) -> str:
    """Left visible, as the report page leaves ``{{F0099}}``: the post-check refuses it."""
    return f'<p class="unresolved" data-chunk>[unresolved {_e(ref)}]</p>'


def _lines(block: LineTableBlock, pack: QuotePack) -> str:
    head = "".join(
        f'<th class="{"num" if c in _RIGHT else ""}" data-chunk>{_e(HEADINGS[c])}</th>'
        for c in block.columns)
    rows = []
    for ref in block.line_refs:
        line = pack.line(ref)
        if line is None:
            rows.append(f'<tr><td colspan="{len(block.columns)}">{_unresolved(ref)}</td></tr>')
            continue
        # The newline between cells is what keeps them apart in the row's chunk.
        cells = "\n".join(
            f'<td class="{"num" if c in _RIGHT else ""}">{_e(line.cell(c))}</td>'
            for c in block.columns)
        rows.append(f"<tr data-chunk>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _totals(block: TotalsBlock, pack: QuotePack) -> str:
    rows = []
    for ref in block.fact_refs:
        entry = pack.fact(ref)
        rows.append(_unresolved(ref) if entry is None else (
            f'<p class="total" data-chunk><span>{_e(entry.label)}</span>\n'
            f'<span class="val">{_e(entry.display)}</span></p>'))
    return '<div class="totals">' + "".join(rows) + "</div>"


def _listed(heading: str, refs: List[str], find) -> str:
    items = "".join(
        f"<li>{_unresolved(r)}</li>" if find(r) is None
        else f"<li data-chunk>{_e(find(r).text)}</li>"
        for r in refs)
    return f'<section><h2 data-chunk>{_e(heading)}</h2><ul>{items}</ul></section>'


def _letterhead(pack: QuotePack) -> str:
    seller = pack.seller
    if seller is None:
        return _unresolved("letterhead")
    lines = [f'<p class="name" data-chunk>{_e(seller.legal_name)}</p>']
    if seller.trading_name:
        lines.append(f'<p data-chunk>Trading as {_e(seller.trading_name)}</p>')
    if seller.address:
        lines.append(f'<p class="addr" data-chunk>{_e(seller.address)}</p>')
    reach = [v for v in (seller.phone, seller.email, seller.web) if v]
    if reach:
        lines.append(f'<p data-chunk>{" · ".join(_e(v) for v in reach)}</p>')
    reg = [f"{label} {_e(v)}" for label, v in (("Company no.", seller.company_number),
                                               ("VAT no.", seller.vat_number)) if v]
    if reg:
        lines.append(f'<p data-chunk>{" · ".join(reg)}</p>')
    bank = [f"{label} {_e(v)}" for label, v in (
        ("Bank", seller.bank_name), ("Sort code", seller.sort_code),
        ("Account", seller.account_number), ("IBAN", seller.iban)) if v]
    if bank:
        lines.append(f'<p class="bank" data-chunk>{" · ".join(bank)}</p>')
    logo = (f'<img class="logo" alt="" src="{_e(seller.logo_data_uri)}">'
            if seller.logo_data_uri else "")
    return f'<div class="letterhead">{logo}<div>{"".join(lines)}</div></div>'


def _section(block: SectionBlock, pack: QuotePack) -> str:
    section = pack.section(block.section_ref)
    if section is None:
        return _unresolved(block.section_ref)
    body, bullets = [], []
    for para in section.paragraphs:
        # the newline after <br> keeps the two lines apart in the chunk the gate reads
        text = "<br>\n".join(_e(part) for part in para.text.split("\n"))
        if para.bullet:
            bullets.append(f"<li data-chunk>{text}</li>")
            continue
        if bullets:
            body.append(f"<ul>{''.join(bullets)}</ul>")
            bullets = []
        body.append(f"<p data-chunk>{text}</p>")
    if bullets:
        body.append(f"<ul>{''.join(bullets)}</ul>")
    return (f'<section><h2 data-chunk>{_e(section.heading)}</h2>{"".join(body)}</section>')


def _signature(block: SignatureBlock, pack: QuotePack) -> str:
    # The supplier's own name is the company profile's; with none entered it is not
    # invented here.
    name = (pack.identity.account_name if block.party == "customer"
            else pack.seller.legal_name if pack.seller else "the supplier")
    return ('<div class="sign">'
            f'<p data-chunk>Signed for {_e(name)}</p>'
            '<p class="rule" data-chunk>Name</p><p class="rule" data-chunk>Date</p></div>')


def _stylesheet(palette, body_font, mono_font) -> str:
    c = {k: _colour(palette, k) for k in PALETTE}
    return f"""
@page {{ size: A4 portrait; margin: 18mm 16mm 20mm;
  @bottom-right {{ content: "page " counter(page) " of " counter(pages); font: 7pt {body_font}; color: {c['mute']}; }} }}
html {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
body {{ margin: 0; font: 10.5pt/1.45 {body_font}; color: {c['ink']}; background: {c['paper']}; }}
@media screen {{
  body {{ background: #EEF1F3; padding: 12mm 0; }}
  .quote {{ width: 178mm; margin: 0 auto; padding: 16mm; background: {c['paper']};
    box-shadow: 0 1px 4px rgba(10,42,67,.15); }}
}}
header {{ border-bottom: 2pt solid {c['ocean']}; margin-bottom: 8mm; padding-bottom: 4mm; }}
h1 {{ margin: 0 0 2mm; font-size: 24pt; color: {c['ocean_dark']}; }}
h2 {{ font-size: 12pt; color: {c['ocean_dark']}; margin: 8mm 0 2mm; break-after: avoid; }}
header p {{ margin: 0 0 1mm; }}
.ref {{ font-family: {mono_font}; }}
.scope {{ color: {c['mute']}; font-size: 9.5pt; }}
table {{ width: 100%; border-collapse: collapse; margin: 0 0 4mm; font-size: 9.5pt; }}
thead {{ display: table-header-group; }}
tr {{ break-inside: avoid; }}
th {{ background: {c['ocean_dark']}; color: {c['paper']}; text-align: left; padding: 2mm 3mm; }}
td {{ border-bottom: 0.75pt solid {c['rule']}; padding: 2mm 3mm; vertical-align: top; }}
.num {{ text-align: right; white-space: nowrap; }}
td.num {{ font-family: {mono_font}; }}
.totals {{ margin: 0 0 6mm; }}
.total {{ display: flex; justify-content: flex-end; gap: 8mm; margin: 0; font-weight: 700; font-size: 12pt; }}
.total .val {{ font-family: {mono_font}; }}
ul {{ margin: 0; padding-left: 5mm; }}
li {{ break-inside: avoid; margin-bottom: 1mm; }}
.sign {{ margin-top: 14mm; break-inside: avoid; width: 80mm; }}
.sign .rule {{ border-top: 0.75pt solid {c['ink']}; margin: 10mm 0 0; padding-top: 1mm; color: {c['mute']}; font-size: 8.5pt; }}
.unresolved {{ color: {c['warm']}; font-weight: 700; }}
.letterhead {{ display: flex; gap: 6mm; align-items: flex-start; margin-bottom: 6mm; font-size: 9pt; color: {c['mute']}; }}
.letterhead p {{ margin: 0 0 0.6mm; }}
.letterhead .name {{ font-size: 13pt; font-weight: 700; color: {c['ink']}; }}
.letterhead .addr {{ white-space: pre-line; }}
.logo {{ max-height: 22mm; max-width: 60mm; }}
section p {{ margin: 0 0 2mm; }}
footer {{ margin-top: 10mm; font-size: 8.5pt; color: {c['mute']}; }}
"""
