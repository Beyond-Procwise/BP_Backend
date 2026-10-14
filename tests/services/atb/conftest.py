"""Decks built in code, so every measurement test is deterministic.

The reference deck is a client document and is not in the repo; the tests that need it read
ATB_REFERENCE_PACK and skip without it. Everything else builds exactly the deck it needs.
"""
from io import BytesIO

import pytest
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt


def _new(width_in=13.333, height_in=7.5):
    prs = Presentation()
    prs.slide_width = Inches(width_in)
    prs.slide_height = Inches(height_in)
    return prs


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _save(prs):
    out = BytesIO()
    prs.save(out)
    return out.getvalue()


@pytest.fixture
def build_deck():
    """build_deck([[(x, y, w, h, text, pt, '#RRGGBB'), ...], ...]) -> bytes"""
    def build(slides, width_in=13.333, height_in=7.5):
        prs = _new(width_in, height_in)
        for shapes in slides:
            slide = _blank(prs)
            for x, y, w, h, text, pt, ink in shapes:
                box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
                run = box.text_frame.paragraphs[0].add_run()
                run.text = text
                run.font.size = Pt(pt)
                run.font.color.rgb = RGBColor.from_string(ink.lstrip('#'))
                run.font.name = 'Cambria' if y < 1.0 else 'Calibri'
        return _save(prs)
    return build


@pytest.fixture
def filled_deck():
    """A deck whose colours are unambiguous: two text colours, two fills, one line colour."""
    def build(ink, muted, panel, accent, rule=None, slides=12):
        prs = _new()
        for _ in range(slides):
            slide = _blank(prs)
            # ink twice, muted once: "most-used text colour" has to be decided by USE, not by a
            # tiebreak. With equal counts the role test passes for the wrong reason.
            for colour, y in ((ink, 1.5), (ink, 1.7), (muted, 2.0)):
                box = slide.shapes.add_textbox(Inches(0.5), Inches(y), Inches(3), Inches(0.4))
                run = box.text_frame.paragraphs[0].add_run()
                run.text = 'x'
                run.font.size = Pt(12)
                run.font.color.rgb = RGBColor.from_string(colour.lstrip('#'))
            for colour, y in ((panel, 3.0), (accent, 4.0)):
                shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(y),
                                             Inches(3), Inches(0.4))
                shp.fill.solid()
                shp.fill.fore_color.rgb = RGBColor.from_string(colour.lstrip('#'))
                shp.line.color.rgb = RGBColor.from_string((rule or colour).lstrip('#'))
        return _save(prs)
    return build


@pytest.fixture
def grouped_deck():
    """One slide whose two text boxes are inside a group — Review Focus 1."""
    prs = _new()
    slide = _blank(prs)
    a = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(3), Inches(0.5))
    a.text_frame.paragraphs[0].add_run().text = 'inside a group'
    b = slide.shapes.add_textbox(Inches(4.0), Inches(1.5), Inches(3), Inches(0.5))
    b.text_frame.paragraphs[0].add_run().text = 'also inside'
    # python-pptx has no group() helper, so the two shape elements are moved under a new grpSp.
    spTree = slide.shapes._spTree
    grp = spTree.makeelement(qn('p:grpSp'), {})
    nv = spTree.makeelement(qn('p:nvGrpSpPr'), {})
    cnv = spTree.makeelement(qn('p:cNvPr'), {'id': '99', 'name': 'Group 99'})
    nv.append(cnv)
    nv.append(spTree.makeelement(qn('p:cNvGrpSpPr'), {}))
    nv.append(spTree.makeelement(qn('p:nvPr'), {}))
    grp.append(nv)
    grp.append(spTree.makeelement(qn('p:grpSpPr'), {}))
    for el in (a._element, b._element):
        spTree.remove(el)
        grp.append(el)
    spTree.append(grp)
    return _save(prs)
