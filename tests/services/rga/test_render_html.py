"""The printable page: A4 portrait, self-contained, deterministic, and held to the deck's
locality rule -- a figure, its [Fxxxx] marker and its badge travel in one element.

No database, no model. The fixtures are the same hand-written pack/brief/AST the deck's
tests use, so the two renderers are exercised on identical content.
"""
from __future__ import annotations

import re

from src.services.rga.models import (
    NarrativeBlock,
    ReportAST,
    Section,
    TableBlock,
)
from src.services.rga.render import html as page
from src.services.rga.render.pptx import badge_text


def _css(content: bytes) -> str:
    text = content.decode("utf-8")
    return re.sub(r"\s+", "", text[text.index("<style>"):text.index("</style>")])


def test_it_is_deterministic(ast, pack, brief):
    assert page.render(ast, pack, brief).content == page.render(ast, pack, brief).content


def test_it_is_self_contained(ast, pack, brief):
    text = page.render(ast, pack, brief).content.decode("utf-8").lower()
    for forbidden in ("<script", "<link", "@import", "url(", "http:", "https:"):
        assert forbidden not in text, forbidden


def test_it_prints_a4_portrait_with_a_numbered_footer(ast, pack, brief):
    css = _css(page.render(ast, pack, brief).content)
    assert "size:A4portrait" in css
    assert "counter(page)" in css and "counter(pages)" in css
    assert pack.hash[:12] in css


def test_each_section_starts_a_page(ast, pack, brief):
    art = page.render(ast, pack, brief)
    text = art.content.decode("utf-8")
    assert text.count('class="section"') == len(ast.sections)
    assert ".section{break-before:page" in _css(art.content)


def test_long_tables_repeat_their_header_and_keep_rows_whole(ast, pack, brief):
    art = page.render(ast, pack, brief)
    css = _css(art.content)
    assert "thead{display:table-header-group" in css
    assert "tr{break-inside:avoid" in css
    assert "<thead>" in art.content.decode("utf-8")


def test_every_text_is_escaped(pack, brief):
    hostile = ReportAST(sections=[Section(id="s", title='Risks <b>&"', blocks=[
        NarrativeBlock(text='Spend rose <b>sharply</b> & "fast" to {{F0001}}.',
                       fact_refs=["F0001"]),
        TableBlock(columns=["Measure <i>", "Value"], rows=[["Label <script>", "F0003"]]),
    ])])
    text = page.render(hostile, pack, brief).content.decode("utf-8")
    body = text[text.index("</style>"):]
    for raw in ("<b>", "<i>", "<script>"):
        assert raw not in body, raw
    assert "Risks &lt;b&gt;&amp;&quot;" in body
    assert "sharply&lt;/b&gt; &amp; &quot;fast&quot;" in body


def test_every_fact_drawing_is_one_chunk_with_its_badge(ast, pack, brief):
    show = bool(brief.get("report.style.show_confidence_badges"))
    chunks = page.extract_text(page.render(ast, pack, brief).content)
    for ref in sorted(ast.fact_refs()):
        entry = pack.fact(ref)
        want = [f"[{ref}]", badge_text(entry, show)]
        assert any(all(w in c for w in want) and entry.display in c for c in chunks), ref


def test_a_table_figure_carries_its_badge_in_its_own_cell(pack, brief):
    only_in_table = ReportAST(sections=[Section(id="t", title="Coverage", blocks=[
        TableBlock(columns=["Measure", "Value"], rows=[["Realised savings", "F0004"]])])])
    chunks = page.extract_text(page.render(only_in_table, pack, brief).content)
    entry = pack.fact("F0004")
    cell = [c for c in chunks if "[F0004]" in c and "Realised savings" not in c]
    assert cell and "UNASSESSED" in cell[0] and entry.display in cell[0]


def test_extract_text_never_reads_the_stylesheet(ast, pack, brief):
    chunks = page.extract_text(page.render(ast, pack, brief).content)
    assert chunks and not any("@page" in c or "counter(" in c for c in chunks)


def test_the_record_is_complete(ast, pack, brief):
    art = page.render(ast, pack, brief)
    assert art.recorded() and art.renderer == "html" and art.pack_hash == pack.hash
    assert art.media_type == "text/html; charset=utf-8"
    assert "of 4 measures not assessed" in art.coverage_disclosure
    chunks = page.extract_text(art.content)
    joined = "\n".join(chunks)
    assert art.coverage_disclosure in joined
    assert brief.disclosure() in joined
    assert f"pack {pack.pack_id}" in joined and art.ast_hash[:12] in joined
