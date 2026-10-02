# PowerPoint Style Import — Implementation Plan (step 1a, BP_Backend)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read one `.pptx` and produce a style pack, 8 reusable page layouts and an evidence
record, stored as candidates and served over an API, so a report can be laid out in a style
measured from a real deck instead of typed by hand.

**Architecture:** A new Python package `src/services/atb/pptx_import/`, one module per
measurement, each a pure function over a `Deck` read once by `read.py`. Nothing writes to the
database except `store.py`, and the write order is `importing → layouts → candidate` because
`get_conn()` is AUTOCOMMIT and a rollback would be a no-op. A FastAPI router exposes import,
read, rename and approve. The importer's output is held to the same contract the hand-authored
packs pass.

**Tech Stack:** Python 3.11, python-pptx 1.0.2 (already installed in both venvs — no new
dependency), FastAPI, psycopg, PostgreSQL (`proc` schema), pytest.

**Spec:** `specs/2026-10-02-pptx-style-import-design.md` — read it before Task 1. Section
numbers below refer to it.

**Not in this plan:** the review screen (`beyond_procwise_ui`, its own plan, written next — it
needs these endpoints to exist first) and the 26 single-use set pieces (§6a: step 2 imports those
as composed pages). This plan ends with a working, demonstrable importer and API.

## Global Constraints

- **Run tests with `./venv/bin/python -m pytest`**, not `.venv` and not system python. `venv` is
  the test environment; `.venv` is the runtime. Load `.env` for DB tests.
- **`CUDA_VISIBLE_DEVICES=""`** on every test run, and never point a test at a live Ollama port.
- **DB tests need `PROCWISE_TEST_LIVE_DB=1`**; without it pytest uses a fake connection.
- **`get_conn()` is AUTOCOMMIT.** `rollback()` is a no-op and `FOR UPDATE` locks end with the
  statement. Never write a multi-statement unit of work that assumes atomicity.
- **New tables take the `bp_` prefix; indexes are `ix_bp_<table>_<cols>`.**
- **Every API write takes `require_user`**, and `POST …/approve` gets its own `bp_policy` row —
  a `write`-class action with no policy row admits any Buyer.
- **No field in an API response may name an internal route or table.** The output-safety layer
  replaces such fields with `[withheld]`. Return ids.
- **Commit through a private index** while another session is working in this checkout:
  `HEAD_SHA=$(git rev-parse HEAD)`, `export GIT_INDEX_FILE=$(mktemp)`, `git read-tree "$HEAD_SHA"`,
  `git add -- <only your paths>`, `git commit-tree`, `git update-ref refs/heads/Development "$COMMIT" "$HEAD_SHA"`,
  `unset GIT_INDEX_FILE`, then `git reset HEAD -- <your paths>`. A bare `git commit` takes the
  whole shared index, which took 498 of another session's staged files on 2026-10-02.
- **The colour floor is 10 uses. The loose-fit tolerance is 0.15in.** Both are module constants,
  both ruled 2026-10-02.
- **The reference deck is not in the repo** (it is a client document). Tests that need it read
  `ATB_REFERENCE_PACK`, defaulting to
  `~/Downloads/Infrastructure-Procurement-Strategy-Pack.pptx`, and **skip with a message naming
  the variable** when it is absent. Every other test builds its own deck with python-pptx.

## Review Focus

Five input classes the spec implies, that no task's happy path exercises, most likely first. Each
line's test is written into the task that owns the code.

1. **Every shape nested inside a group.** PowerPoint groups freely; `Shape.shapes` is not walked
   by default. Unflattened, the census sees nothing and the pack comes back empty and fails its
   own contract. → Task 1.
2. **A deck that names only theme colours** (`<a:schemeClr val="tx1"/>`, no `srgbClr`). The
   palette must resolve them to RGB; unresolved, `colours.ink` is missing and the pack is
   invalid. → Task 2.
3. **A deck with no charts.** `series_palette` must be non-empty for the pack to validate, so the
   fallback to saturated accents has to actually fire. → Task 5.
4. **A re-import after a human renamed every layout and defined a rating scale.** §5b says those
   survive; the obvious implementation re-measures and loses them. → Task 10.
5. **A portrait or 4:3 deck.** `format.kind` follows the aspect, and an `a4-portrait` pack states
   no `width_in`/`height_in`, which every downstream geometry call must tolerate. → Task 6.

---

### Task 1: Read the deck into plain data

**Files:**
- Create: `src/services/atb/__init__.py`
- Create: `src/services/atb/pptx_import/__init__.py`
- Create: `src/services/atb/pptx_import/read.py`
- Create: `src/services/atb/pptx_import/evidence.py`
- Create: `tests/services/atb/__init__.py`
- Create: `tests/services/atb/conftest.py`
- Create: `tests/services/atb/test_read.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `read_deck(data: bytes) -> Deck`
  - `Box(x: float, y: float, w: float, h: float)` — inches, frozen dataclass
  - `Run(text: str, size_pt: float | None, font: str | None, colour: str | None, bold: bool, lang: str | None)`
  - `Shape(kind: str, box: Box, runs: tuple[Run, ...], fill: str | None, line: str | None, table: tuple[tuple[str, ...], ...] | None, slide: int, name: str)`
    where `kind` is one of `'text' | 'auto' | 'table' | 'chart' | 'picture' | 'placeholder'`
  - `Deck(width_in: float, height_in: float, slides: tuple[tuple[Shape, ...], ...], theme: dict, chart_series_colours: tuple[str, ...], run_langs: dict[str, int])`
  - `Evidence()` with `.record(path, value, **facts)`, `.incidental(kind, value, why)`,
    `.ignored(what, value, why)`, `.as_dict() -> dict`
  - `class DeckUnreadable(Exception)`
  - Test fixture `deck_bytes(builder)` in `conftest.py`

- [ ] **Step 1: Write the fixture builder**

`tests/services/atb/conftest.py`:

```python
"""Decks built in code, so every measurement test is deterministic.

The reference deck is a client document and is not in the repo; tests that need it read
ATB_REFERENCE_PACK and skip without it. Everything else builds exactly the deck it needs.
"""
from io import BytesIO

import pytest
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt


def _new(width_in=13.333, height_in=7.5):
    prs = Presentation()
    prs.slide_width = Inches(width_in)
    prs.slide_height = Inches(height_in)
    return prs


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


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
                run.font.name = 'Calibri'
        out = BytesIO()
        prs.save(out)
        return out.getvalue()
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
    # python-pptx has no group() helper; move both shape elements under a new grpSp.
    from pptx.oxml.ns import qn
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
    out = BytesIO()
    prs.save(out)
    return out.getvalue()
```

- [ ] **Step 2: Write the failing test**

`tests/services/atb/test_read.py`:

```python
import pytest

from src.services.atb.pptx_import.read import DeckUnreadable, read_deck


def test_reads_page_size_and_one_shape(build_deck):
    deck = read_deck(build_deck([[(0.5, 0.35, 12.33, 0.6, 'A title', 30, '#172033')]]))
    assert (round(deck.width_in, 3), round(deck.height_in, 3)) == (13.333, 7.5)
    assert len(deck.slides) == 1
    shape = deck.slides[0][0]
    assert shape.kind == 'text'
    assert (shape.box.x, shape.box.y, shape.box.w) == (0.5, 0.35, 12.33)
    assert shape.runs[0].text == 'A title'
    assert shape.runs[0].size_pt == 30.0
    assert shape.runs[0].colour == '#172033'
    assert shape.slide == 1


def test_flattens_groups_or_the_census_sees_nothing(grouped_deck):
    # Review Focus 1: PowerPoint groups freely. Unflattened, these two shapes are invisible.
    deck = read_deck(grouped_deck)
    texts = [r.text for s in deck.slides for sh in s for r in sh.runs]
    assert 'inside a group' in texts
    assert 'also inside' in texts
    assert all(sh.kind != 'group' for s in deck.slides for sh in s)


def test_refuses_a_file_it_cannot_open():
    with pytest.raises(DeckUnreadable) as e:
        read_deck(b'this is not a pptx')
    assert 'could not be opened' in str(e.value)


def test_refuses_a_deck_with_no_slides():
    from io import BytesIO
    from pptx import Presentation
    out = BytesIO()
    Presentation().save(out)
    with pytest.raises(DeckUnreadable) as e:
        read_deck(out.getvalue())
    assert 'no slides' in str(e.value)
```

- [ ] **Step 3: Run it and watch it fail**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/atb/test_read.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.atb'`.

- [ ] **Step 4: Write `read.py`**

```python
"""One pass over a .pptx, into plain frozen dataclasses.

Everything downstream measures THIS, never python-pptx objects, for three reasons: the
measurement modules stay pure and trivially testable, a shape's geometry is converted to inches
exactly once, and groups are flattened here so no later module can forget to walk into them.

Theme values are READ AND RECORDED, never used. On the reference deck the theme is stock Office
(Calibri Light, #4472C4) and has nothing to do with what the deck looks like — see the spec §2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from pptx import Presentation
from pptx.util import Emu

_DRAWING_NS = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
_SRGB = re.compile(r'<a:srgbClr val="([0-9A-Fa-f]{6})"')
_LANG = re.compile(r'lang="([A-Za-z-]+)"')


class DeckUnreadable(Exception):
    """The file is not a usable presentation. Carries the reason for the API to pass on."""


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class Run:
    text: str
    size_pt: float | None
    font: str | None
    colour: str | None
    bold: bool
    lang: str | None


@dataclass(frozen=True)
class Shape:
    kind: str
    box: Box
    runs: tuple[Run, ...]
    fill: str | None
    line: str | None
    table: tuple[tuple[str, ...], ...] | None
    slide: int
    name: str


@dataclass(frozen=True)
class Deck:
    width_in: float
    height_in: float
    slides: tuple[tuple[Shape, ...], ...]
    theme: dict[str, Any]
    chart_series_colours: tuple[str, ...]
    run_langs: dict[str, int]


def _inches(v) -> float:
    return 0.0 if v is None else round(Emu(int(v)).inches, 3)


def _hex(value: str | None) -> str | None:
    return None if value is None else '#' + value.upper()


def _fill_of(element) -> str | None:
    xml = element.xml
    start = xml.find('<a:solidFill>')
    if start < 0:
        return None
    m = _SRGB.search(xml, start)
    return _hex(m.group(1)) if m else None


def _line_of(element) -> str | None:
    xml = element.xml
    start = xml.find('<a:ln')
    if start < 0:
        return None
    m = _SRGB.search(xml, start)
    return _hex(m.group(1)) if m else None


def _kind_of(shape) -> str:
    if shape.has_table:
        return 'table'
    if getattr(shape, 'has_chart', False):
        return 'chart'
    name = str(getattr(shape, 'shape_type', '') or '')
    if 'PICTURE' in name:
        return 'picture'
    if 'PLACEHOLDER' in name:
        return 'placeholder'
    if shape.has_text_frame:
        return 'text'
    return 'auto'


def _runs_of(shape, langs: dict[str, int]) -> tuple[Run, ...]:
    if not shape.has_text_frame:
        return ()
    out: list[Run] = []
    for paragraph in shape.text_frame.paragraphs:
        for run in paragraph.runs:
            lang = None
            m = _LANG.search(run._r.xml)
            if m:
                lang = m.group(1)
                langs[lang] = langs.get(lang, 0) + 1
            colour = None
            try:
                if run.font.color is not None and run.font.color.type is not None:
                    rgb = run.font.color.rgb
                    colour = _hex(str(rgb)) if rgb is not None else None
            except (AttributeError, TypeError, ValueError):
                colour = None
            out.append(Run(
                text=run.text or '',
                size_pt=float(run.font.size.pt) if run.font.size else None,
                font=run.font.name,
                colour=colour,
                bold=bool(run.font.bold),
                lang=lang,
            ))
    return tuple(out)


def _table_of(shape) -> tuple[tuple[str, ...], ...] | None:
    if not shape.has_table:
        return None
    return tuple(tuple(cell.text for cell in row.cells) for row in shape.table.rows)


def _walk(shapes, slide_no: int, langs: dict[str, int], out: list[Shape]) -> None:
    for shape in shapes:
        if 'GROUP' in str(getattr(shape, 'shape_type', '') or ''):
            # Review Focus 1: a group's children are invisible unless walked. Flattened here so
            # that no measurement module has to remember to do it.
            _walk(shape.shapes, slide_no, langs, out)
            continue
        out.append(Shape(
            kind=_kind_of(shape),
            box=Box(_inches(shape.left), _inches(shape.top), _inches(shape.width), _inches(shape.height)),
            runs=_runs_of(shape, langs),
            fill=_fill_of(shape._element),
            line=_line_of(shape._element),
            table=_table_of(shape),
            slide=slide_no,
            name=str(shape.name or ''),
        ))


def _theme_of(prs) -> dict[str, Any]:
    part = prs.slide_masters[0].part.part_related_by(
        'http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme')
    xml = part.blob.decode('utf-8', 'ignore')
    colours = dict(re.findall(
        r'<a:(dk1|dk2|lt1|lt2|accent[1-6])>.*?val="([0-9A-Fa-f]{6})"', xml))
    fonts = dict(re.findall(r'<a:(majorFont|minorFont)>\s*<a:latin typeface="([^"]*)"', xml))
    return {'colours': {k: _hex(v) for k, v in colours.items()}, 'fonts': fonts}


def _chart_series_colours(prs) -> tuple[str, ...]:
    seen: list[str] = []
    for part in prs.part.package.iter_parts():
        if '/charts/chart' not in part.partname:
            continue
        xml = part.blob.decode('utf-8', 'ignore')
        for m in _SRGB.finditer(xml):
            colour = _hex(m.group(1))
            if colour not in seen:
                seen.append(colour)
    return tuple(seen)


def read_deck(data: bytes) -> Deck:
    try:
        prs = Presentation(BytesIO(data))
    except Exception as exc:
        raise DeckUnreadable(f'the file could not be opened as a presentation: {exc}') from exc
    if not len(prs.slides):
        raise DeckUnreadable('the presentation has no slides')
    langs: dict[str, int] = {}
    slides: list[tuple[Shape, ...]] = []
    for i, slide in enumerate(prs.slides, 1):
        found: list[Shape] = []
        _walk(slide.shapes, i, langs, found)
        slides.append(tuple(found))
    try:
        theme = _theme_of(prs)
    except Exception:
        theme = {'colours': {}, 'fonts': {}}
    return Deck(
        width_in=round(Emu(int(prs.slide_width)).inches, 3),
        height_in=round(Emu(int(prs.slide_height)).inches, 3),
        slides=tuple(slides),
        theme=theme,
        chart_series_colours=_chart_series_colours(prs),
        run_langs=langs,
    )
```

- [ ] **Step 5: Write `evidence.py`**

```python
"""Where every value in a pack came from.

A fact in this codebase cannot be constructed without a provenance id, and a style token should
answer to the same standard: "#172033 is the ink" is not a finding, "#172033 paints 619 fills and
804 text runs, first on slide 1" is.
"""
from __future__ import annotations

from typing import Any


class Evidence:
    def __init__(self) -> None:
        self._values: dict[str, dict[str, Any]] = {}
        self._incidental: list[dict[str, Any]] = []
        self._ignored: dict[str, dict[str, Any]] = {}

    def record(self, path: str, value: Any, **facts: Any) -> None:
        self._values[path] = {'value': value, **facts}

    def incidental(self, kind: str, value: Any, why: str) -> None:
        self._incidental.append({'kind': kind, 'value': value, 'why': why})

    def ignored(self, what: str, value: Any, why: str) -> None:
        self._ignored[what] = {'value': value, 'why': why}

    def as_dict(self) -> dict[str, Any]:
        return {
            'values': dict(self._values),
            'incidental': list(self._incidental),
            'ignored': dict(self._ignored),
        }
```

- [ ] **Step 6: Run the tests**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/atb/test_read.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Prove the group walk is load-bearing**

Change `_walk`'s group branch to `continue` without recursing, re-run, and confirm
`test_flattens_groups_or_the_census_sees_nothing` **fails**. Restore it.

- [ ] **Step 8: Commit** (private index, per Global Constraints)

```
feat(atb): read a .pptx into plain data, groups flattened

One pass, into frozen dataclasses in inches, so every measurement module is a pure
function over the same Deck and none of them can forget to walk into a group — which
is the one mistake that makes the whole census silently see nothing.

Theme values are read and recorded, never used: on the reference deck the theme is
stock Office and disagrees with what the deck looks like.
```

---

### Task 2: The colour census and its roles

**Files:**
- Create: `src/services/atb/pptx_import/palette.py`
- Create: `tests/services/atb/test_palette.py`

**Interfaces:**
- Consumes: `Deck`, `Shape`, `Run` from Task 1; `Evidence` from Task 1.
- Produces:
  - `COLOUR_FLOOR = 10`
  - `census(deck: Deck) -> dict[str, dict[str, int]]` — `{'#172033': {'fills': 619, 'runs': 804, 'lines': 563}}`
  - `colours(deck: Deck, ev: Evidence, floor: int = COLOUR_FLOOR) -> dict[str, str]`

- [ ] **Step 1: Write the failing test**

```python
from src.services.atb.pptx_import.evidence import Evidence
from src.services.atb.pptx_import.palette import census, colours


def _deck_with(ink, muted, panel, accent, rule=None):
    """A deck whose colour usage is unambiguous, built to the role rules in spec §5."""
    from io import BytesIO
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt
    from src.services.atb.pptx_import.read import read_deck

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for i in range(12):                      # over the 10-use floor
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        for colour, y in ((ink, 1.0), (muted, 2.0)):
            box = slide.shapes.add_textbox(Inches(0.5), Inches(y), Inches(3), Inches(0.4))
            run = box.text_frame.paragraphs[0].add_run()
            run.text = 'x'
            run.font.size = Pt(12)
            run.font.color.rgb = RGBColor.from_string(colour.lstrip('#'))
        for colour, y in ((panel, 3.0), (accent, 4.0)):
            shp = slide.shapes.add_shape(1, Inches(0.5), Inches(y), Inches(3), Inches(0.4))
            shp.fill.solid()
            shp.fill.fore_color.rgb = RGBColor.from_string(colour.lstrip('#'))
            shp.line.color.rgb = RGBColor.from_string((rule or colour).lstrip('#'))
    out = BytesIO()
    prs.save(out)
    return read_deck(out.getvalue())


def test_counts_a_colour_in_all_three_roles():
    deck = _deck_with('#172033', '#56627A', '#F3F5F8', '#2350C8')
    c = census(deck)
    assert c['#172033']['runs'] == 12
    assert c['#F3F5F8']['fills'] == 12


def test_names_ink_muted_panel_and_accent():
    ev = Evidence()
    out = colours(_deck_with('#172033', '#56627A', '#F3F5F8', '#2350C8'), ev)
    assert out['ink'] == '#172033'
    assert out['muted'] == '#56627A'
    assert out['panel'] == '#F3F5F8'
    assert out['accent'] == '#2350C8'


def test_a_colour_under_the_floor_is_incidental_not_a_token():
    deck = _deck_with('#172033', '#56627A', '#F3F5F8', '#2350C8', rule='#D5DBE5')
    ev = Evidence()
    out = colours(deck, ev)
    assert '#D5DBE5' in str(ev.as_dict()) or out.get('rule') == '#D5DBE5'
    # a one-off never becomes a token
    assert '#010203' not in out.values()


def test_resolves_theme_colours_or_the_pack_has_no_ink():
    # Review Focus 2: a deck that names only scheme colours. Unresolved, colours.ink is missing
    # and the pack cannot validate.
    from io import BytesIO
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from src.services.atb.pptx_import.read import read_deck

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for _ in range(12):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        box = slide.shapes.add_textbox(Inches(0.5), Inches(1), Inches(3), Inches(0.4))
        run = box.text_frame.paragraphs[0].add_run()
        run.text = 'theme coloured'
        run.font.size = Pt(12)                 # no explicit colour: inherits the theme
    out = BytesIO()
    prs.save(out)
    ev = Evidence()
    resolved = colours(read_deck(out.getvalue()), ev)
    assert resolved.get('ink'), 'a deck with only theme colours still needs an ink'
```

- [ ] **Step 2: Run it and watch it fail**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/atb/test_palette.py -v
```

Expected: FAIL — `No module named 'src.services.atb.pptx_import.palette'`.

- [ ] **Step 3: Write `palette.py`**

```python
"""Which colours a deck really uses, and what each one is for.

Role follows USE, not the theme: the most-used text colour is the ink whatever the theme says.
A colour under COLOUR_FLOOR uses is incidental and never becomes a token — on a deck of 85 slides
a colour used five times is a one-off, and a token invented from it would be applied to
everything.
"""
from __future__ import annotations

import colorsys

from .evidence import Evidence
from .read import Deck

COLOUR_FLOOR = 10

_THEME_INK_KEYS = ('dk1', 'dk2')


def _rgb(colour: str) -> tuple[int, int, int]:
    v = colour.lstrip('#')
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def _luminance(colour: str) -> float:
    r, g, b = (c / 255 for c in _rgb(colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _saturation(colour: str) -> float:
    r, g, b = (c / 255 for c in _rgb(colour))
    return colorsys.rgb_to_hls(r, g, b)[2]


def _hue(colour: str) -> float:
    r, g, b = (c / 255 for c in _rgb(colour))
    return colorsys.rgb_to_hls(r, g, b)[0] * 360


def census(deck: Deck) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}

    def bump(colour: str | None, role: str) -> None:
        if not colour:
            return
        out.setdefault(colour, {'fills': 0, 'runs': 0, 'lines': 0})[role] += 1

    for slide in deck.slides:
        for shape in slide:
            bump(shape.fill, 'fills')
            bump(shape.line, 'lines')
            for run in shape.runs:
                if run.text.strip():
                    bump(run.colour, 'runs')
    return out


def colours(deck: Deck, ev: Evidence, floor: int = COLOUR_FLOOR) -> dict[str, str]:
    counts = census(deck)
    total = {c: sum(v.values()) for c, v in counts.items()}
    eligible = {c: v for c, v in counts.items() if total[c] >= floor}
    for colour, v in counts.items():
        if total[colour] < floor:
            ev.incidental('colour', colour, f'used {total[colour]} times, under the {floor}-use floor')

    taken: set[str] = set()

    def take(name: str, candidates: list[str], why: str) -> None:
        for colour in candidates:
            if colour in taken:
                continue
            taken.add(colour)
            out[name] = colour
            ev.record(f'colours.{name}', colour, why=why, **counts[colour])
            return

    out: dict[str, str] = {}
    by_runs = sorted(eligible, key=lambda c: (-eligible[c]['runs'], c))
    by_fills = sorted(eligible, key=lambda c: (-eligible[c]['fills'], c))
    line_only = [c for c in by_fills if eligible[c]['lines'] and not eligible[c]['fills'] and not eligible[c]['runs']]

    take('ink', [c for c in by_runs if eligible[c]['runs']], 'most-used text colour')
    take('muted', [c for c in by_runs if eligible[c]['runs']], 'second most-used text colour')
    take('rule', line_only, 'used only on lines')
    pale = [c for c in by_fills if eligible[c]['fills'] and _luminance(c) > 0.85]
    take('panel', pale, 'palest frequently-filled colour')
    for name, hues in (('panel_blue', (190, 260)), ('panel_teal', (150, 190)),
                       ('panel_amber', (20, 60)), ('panel_violet', (260, 300))):
        take(name, [c for c in pale if hues[0] <= _hue(c) <= hues[1]], f'pale fill, hue in {hues}')
    saturated = [c for c in by_fills
                 if _saturation(c) > 0.3 and 0.2 <= _luminance(c) <= 0.7]
    take('accent', saturated, 'most-used saturated colour')
    take('accent_2', saturated, 'second most-used saturated colour')
    for name, hues in (('alert_ink', (0, 20)), ('caution', (20, 60)), ('positive', (90, 160))):
        take(name, [c for c in saturated if hues[0] <= _hue(c) <= hues[1]], f'saturated, hue in {hues}')

    # A pack must have an ink and a panel to validate. A deck whose text carries no explicit
    # colour at all (Review Focus 2) falls back to the theme's dark colour, recorded as such.
    if 'ink' not in out:
        for key in _THEME_INK_KEYS:
            theme_ink = (deck.theme.get('colours') or {}).get(key)
            if theme_ink:
                out['ink'] = theme_ink
                ev.record('colours.ink', theme_ink, why=f'no explicit text colour in the deck; theme {key}')
                break
    out.setdefault('ink', '#000000')
    out.setdefault('muted', out['ink'])
    out.setdefault('panel', '#FFFFFF')
    out.setdefault('accent', out['ink'])
    return out
```

- [ ] **Step 4: Run the tests**

Expected: 4 passed.

- [ ] **Step 5: Prove the floor is load-bearing**

Set `floor=0` in `test_a_colour_under_the_floor_is_incidental_not_a_token` and confirm the
incidental assertion changes behaviour; then set `COLOUR_FLOOR = 0` in the module and confirm a
test fails. Restore.

- [ ] **Step 6: Commit**

```
feat(atb): the palette is what the deck uses, not what its theme claims

Role follows use — most-used text colour is the ink, a colour seen only on lines is
the rule, pale fills are the panel family, saturated ones the accents. Under ten uses
in a deck is incidental and never becomes a token.

A deck that names only scheme colours still gets an ink, from the theme, recorded as
that rather than passed off as measured.
```

---

### Task 3: The type scale and the fonts

**Files:**
- Create: `src/services/atb/pptx_import/typescale.py`
- Create: `tests/services/atb/test_typescale.py`

**Interfaces:**
- Consumes: `Deck`, `Evidence`.
- Produces:
  - `type_scale(deck: Deck, ev: Evidence) -> dict[str, float]` with keys
    `title, subtitle, kpi_value, card_title, body, table, small, footer`
  - `fonts(deck: Deck, ev: Evidence) -> dict[str, dict[str, str]]` — `{'heading': {'family':…, 'fallback':…}, 'body': {…}}`
  - `SERIF_FAMILIES: frozenset[str]`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from src.services.atb.pptx_import.evidence import Evidence
from src.services.atb.pptx_import.read import read_deck
from src.services.atb.pptx_import.typescale import fonts, type_scale


def _sized_deck(sizes_by_y, heading_font='Cambria', body_font='Calibri'):
    from io import BytesIO
    from pptx import Presentation
    from pptx.util import Inches, Pt
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for _ in range(8):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        for y, pt in sizes_by_y:
            box = slide.shapes.add_textbox(Inches(0.5), Inches(y), Inches(6), Inches(0.5))
            run = box.text_frame.paragraphs[0].add_run()
            run.text = 'text at %spt' % pt
            run.font.size = Pt(pt)
            run.font.name = heading_font if y < 1.0 else body_font
    out = BytesIO()
    prs.save(out)
    return read_deck(out.getvalue())


def test_the_largest_size_in_the_title_band_is_the_title():
    ev = Evidence()
    scale = type_scale(_sized_deck([(0.35, 30), (1.5, 11.5), (2.5, 10), (3.5, 9)]), ev)
    assert scale['title'] == 30.0
    assert scale['body'] == 11.5
    assert scale['footer'] == 9.0


def test_sizes_within_point_six_merge():
    ev = Evidence()
    scale = type_scale(_sized_deck([(0.35, 30), (1.5, 11.5), (2.5, 11.0), (3.5, 9)]), ev)
    assert len({v for k, v in scale.items() if k in ('body', 'table')}) <= 2


def test_a_serif_family_gets_a_serif_fallback_and_says_it_invented_it():
    ev = Evidence()
    out = fonts(_sized_deck([(0.35, 30), (1.5, 11.5)]), ev)
    assert out['heading']['family'] == 'Cambria'
    assert 'serif' in out['heading']['fallback']
    assert out['body']['family'] == 'Calibri'
    assert 'sans-serif' in out['body']['fallback']
    assert any(v.get('invented') for v in ev.as_dict()['values'].values())


def test_the_theme_fonts_are_recorded_and_not_used():
    ev = Evidence()
    out = fonts(_sized_deck([(0.35, 30), (1.5, 11.5)]), ev)
    assert out['heading']['family'] != 'Calibri Light'
    assert 'theme.fonts' in ev.as_dict()['ignored']
```

- [ ] **Step 2: Run it and watch it fail**

Expected: FAIL — no module `typescale`.

- [ ] **Step 3: Write `typescale.py`**

```python
"""The type scale, named by rank and position, and the fonts, taken from the runs.

The theme's majorFont/minorFont are recorded and NOT used: the reference deck's theme says
Calibri Light and the deck's own headings are Cambria. A .pptx names a family and nothing else,
so the fallback stack is invented here — and says so.
"""
from __future__ import annotations

from collections import Counter

from .evidence import Evidence
from .read import Deck

SERIF_FAMILIES = frozenset({
    'cambria', 'georgia', 'times new roman', 'times', 'garamond', 'book antiqua', 'palatino',
    'palatino linotype', 'constantia', 'bookman old style', 'century schoolbook',
})

_SERIF_STACK = "Georgia, 'Times New Roman', serif"
_SANS_STACK = "'Segoe UI', system-ui, -apple-system, Arial, sans-serif"

# The roles a pack must define, largest first. REQUIRED_TYPE_SCALE in validateStyle.js is
# title/subtitle/body/table/footer; the other three are optional and emitted when the deck has
# enough distinct sizes to fill them.
_ROLES = ('title', 'subtitle', 'kpi_value', 'card_title', 'body', 'table', 'small', 'footer')
_MERGE_PT = 0.6
_TITLE_BAND_IN = 1.0


def _sizes(deck: Deck) -> Counter:
    sizes: Counter = Counter()
    for slide in deck.slides:
        for shape in slide:
            for run in shape.runs:
                if run.size_pt and run.text.strip():
                    sizes[round(run.size_pt, 1)] += 1
    return sizes


def _title_band_sizes(deck: Deck) -> Counter:
    sizes: Counter = Counter()
    for slide in deck.slides:
        for shape in slide:
            if shape.box.y >= _TITLE_BAND_IN:
                continue
            for run in shape.runs:
                if run.size_pt and run.text.strip():
                    sizes[round(run.size_pt, 1)] += 1
    return sizes


def type_scale(deck: Deck, ev: Evidence) -> dict[str, float]:
    sizes = _sizes(deck)
    if not sizes:
        ev.record('type_scale_pt', {}, why='the deck states no run sizes')
        return {'title': 30.0, 'subtitle': 14.0, 'body': 11.0, 'table': 10.0, 'footer': 9.0}

    # merge sizes within _MERGE_PT, keeping the most-used of each cluster
    distinct: list[float] = []
    for size in sorted(sizes, reverse=True):
        if distinct and abs(distinct[-1] - size) <= _MERGE_PT:
            if sizes[size] > sizes[distinct[-1]]:
                ev.incidental('type_size', distinct[-1], f'merged into {size}pt, within {_MERGE_PT}pt')
                distinct[-1] = size
            else:
                ev.incidental('type_size', size, f'merged into {distinct[-1]}pt, within {_MERGE_PT}pt')
            continue
        distinct.append(size)

    band = _title_band_sizes(deck)
    title = max((s for s in distinct if band.get(s, 0) > 0), default=distinct[0])
    rest = [s for s in distinct if s != title]
    scale: dict[str, float] = {'title': title}
    ev.record('type_scale_pt.title', title, runs=sizes[title], why='largest size used in the title band')
    for role, size in zip(_ROLES[1:], rest):
        scale[role] = size
        ev.record(f'type_scale_pt.{role}', size, runs=sizes[size], why='by rank below the title')
    for role in ('subtitle', 'body', 'table', 'footer'):
        scale.setdefault(role, scale.get('body', title))
    return scale


def _stack_for(family: str | None) -> tuple[str, bool]:
    if family and family.strip().lower() in SERIF_FAMILIES:
        return _SERIF_STACK, True
    return _SANS_STACK, False


def fonts(deck: Deck, ev: Evidence) -> dict[str, dict[str, str]]:
    body_counts: Counter = Counter()
    heading_counts: Counter = Counter()
    for slide in deck.slides:
        for shape in slide:
            target = heading_counts if shape.box.y < _TITLE_BAND_IN else body_counts
            for run in shape.runs:
                if run.font and run.text.strip():
                    target[run.font] += 1
                    if target is heading_counts:
                        body_counts[run.font] += 0
    body_family = body_counts.most_common(1)[0][0] if body_counts else None
    heading_family = heading_counts.most_common(1)[0][0] if heading_counts else body_family

    theme_fonts = (deck.theme.get('fonts') or {})
    if theme_fonts:
        ev.ignored('theme.fonts', theme_fonts,
                   'the runs name their own families; a theme font is not what the deck looks like')

    out: dict[str, dict[str, str]] = {}
    for role, family in (('heading', heading_family), ('body', body_family)):
        stack, serif = _stack_for(family)
        out[role] = {'family': family or 'inherit', 'fallback': stack}
        ev.record(f'fonts.{role}', out[role], invented=True,
                  why=f'family measured from the runs; fallback stack invented ({"serif" if serif else "sans"})')
    return out
```

- [ ] **Step 4: Run the tests**

Expected: 4 passed.

- [ ] **Step 5: Commit**

```
feat(atb): the type scale from the runs, the fonts from the runs

The largest size that actually appears in the title band is the title; the rest are
named by rank, with sizes inside 0.6pt merged. Fonts come from the runs and the theme's
are recorded as ignored — the reference deck's theme says Calibri Light while its
headings are Cambria.

A .pptx names a family and nothing else, so the fallback stack is invented from a serif
test on the family name, and the evidence says invented rather than measured.
```

---

### Task 4: The grid

**Files:**
- Create: `src/services/atb/pptx_import/grid.py`
- Create: `tests/services/atb/test_grid.py`

**Interfaces:**
- Consumes: `Deck`, `Evidence`.
- Produces:
  - `grid(deck: Deck, ev: Evidence) -> dict` with keys
    `cols (always 12), margin_in, gutter_in, title_top_in, body_top_in, footer_top_in`
  - `chapter_chip_in(deck: Deck, ev: Evidence) -> float | None`

- [ ] **Step 1: Write the failing test**

```python
from src.services.atb.pptx_import.evidence import Evidence
from src.services.atb.pptx_import.grid import chapter_chip_in, grid
from src.services.atb.pptx_import.read import read_deck


def _gridded_deck():
    """Eight slides laid out like the reference deck: 0.5in margins, title 0.35, body 1.5,
    footnote 7.02, a 0.32in chapter square at 0.5, 0.47, and 4-up cards at a 0.3in gutter."""
    from io import BytesIO
    from pptx import Presentation
    from pptx.util import Inches, Pt
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for _ in range(8):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        def box(x, y, w, h, text, pt=12):
            shp = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
            run = shp.text_frame.paragraphs[0].add_run()
            run.text, run.font.size = text, Pt(pt)
        box(0.5, 0.35, 12.33, 0.6, 'title', 30)
        slide.shapes.add_shape(1, Inches(0.5), Inches(0.47), Inches(0.32), Inches(0.32))
        for i in range(4):
            box(0.5 + i * (2.86 + 0.3), 1.5, 2.86, 2.0, 'card %d' % i)
        box(0.5, 7.02, 11.73, 0.3, 'a footnote', 9)
    out = BytesIO()
    prs.save(out)
    return read_deck(out.getvalue())


def test_measures_the_margin_and_the_bands():
    ev = Evidence()
    g = grid(_gridded_deck(), ev)
    assert g['cols'] == 12
    assert g['margin_in'] == 0.5
    assert g['title_top_in'] == 0.35
    assert g['body_top_in'] == 1.5
    assert g['footer_top_in'] == 7.02


def test_measures_the_gutter_between_cards_in_a_row():
    ev = Evidence()
    assert grid(_gridded_deck(), ev)['gutter_in'] == 0.3


def test_finds_the_chapter_chip():
    ev = Evidence()
    assert chapter_chip_in(_gridded_deck(), ev) == 0.32


def test_a_deck_with_one_slide_still_yields_a_grid():
    from io import BytesIO
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(8.27), Inches(11.69)
    prs.slides.add_slide(prs.slide_layouts[6])
    out = BytesIO()
    prs.save(out)
    ev = Evidence()
    g = grid(read_deck(out.getvalue()), ev)
    assert g['cols'] == 12 and g['margin_in'] > 0
```

- [ ] **Step 2: Run it and watch it fail.**

- [ ] **Step 3: Write `grid.py`**

```python
"""Where things repeatedly sit.

`cols` is always 12 — it is the authoring convenience the hand-written layouts use, and the spec
(§5) records that the reference deck is NOT strictly on twelve columns: 4-up cards are 2.9in,
3-up 3.71in, 5-up 2.45in, and several do not land on a span. That is why an imported region
carries measured inches instead (§7).
"""
from __future__ import annotations

from collections import Counter

from .evidence import Evidence
from .read import Deck

COLS = 12
_DEFAULT_MARGIN = 0.5
_DEFAULT_GUTTER = 0.3
_SAME_ROW_IN = 0.45
_CHIP_MAX_IN = 0.6


def _modal(values: list[float], default: float) -> tuple[float, int]:
    if not values:
        return default, 0
    counts = Counter(round(v, 2) for v in values)
    value, n = counts.most_common(1)[0]
    return value, n


def grid(deck: Deck, ev: Evidence) -> dict:
    lefts, widths = [], []
    top_band, mid_band, bottom_band = [], [], []
    gutters: list[float] = []
    for slide in deck.slides:
        shapes = [s for s in slide if s.box.w > 0.3 and s.box.h > 0.1]
        for shape in shapes:
            lefts.append(shape.box.x)
            widths.append(shape.box.w)
            if shape.box.y < 1.0:
                top_band.append(shape.box.y)
            elif shape.box.y > deck.height_in * 0.85:
                bottom_band.append(shape.box.y)
            else:
                mid_band.append(shape.box.y)
        rows: dict[float, list] = {}
        for shape in shapes:
            key = next((k for k in rows if abs(k - shape.box.y) <= _SAME_ROW_IN), shape.box.y)
            rows.setdefault(key, []).append(shape)
        for row in rows.values():
            row.sort(key=lambda s: s.box.x)
            for a, b in zip(row, row[1:]):
                gap = round(b.box.x - (a.box.x + a.box.w), 2)
                if 0.05 <= gap <= 1.0:
                    gutters.append(gap)

    margin, margin_n = _modal(lefts, _DEFAULT_MARGIN)
    gutter, gutter_n = _modal(gutters, _DEFAULT_GUTTER)
    title_top, title_n = _modal(top_band, 0.35)
    body_top, body_n = _modal(mid_band, 1.5)
    footer_top, footer_n = _modal(bottom_band, round(deck.height_in - 0.48, 2))
    content_w, content_n = _modal(widths, round(deck.width_in - 2 * margin, 2))

    ev.record('grid.margin_in', margin, shapes=margin_n, why='modal left edge')
    ev.record('grid.gutter_in', gutter, gaps=gutter_n, why='modal gap between shapes in a row')
    ev.record('grid.title_top_in', title_top, shapes=title_n, why='modal top edge above 1in')
    ev.record('grid.body_top_in', body_top, shapes=body_n, why='modal top edge in the body band')
    ev.record('grid.footer_top_in', footer_top, shapes=footer_n, why='modal top edge in the bottom band')
    ev.record('grid.content_width_in', content_w, shapes=content_n,
              why=f'modal width; page less two margins is {round(deck.width_in - 2 * margin, 2)}')
    return {
        'cols': COLS,
        'margin_in': margin,
        'gutter_in': gutter,
        'title_top_in': title_top,
        'body_top_in': body_top,
        'footer_top_in': footer_top,
    }


def chapter_chip_in(deck: Deck, ev: Evidence) -> float | None:
    squares: list[float] = []
    for slide in deck.slides:
        for shape in slide:
            w, h = shape.box.w, shape.box.h
            if 0 < w <= _CHIP_MAX_IN and 0 < h <= _CHIP_MAX_IN and abs(w - h) <= 0.05 \
                    and shape.box.y < 1.0:
                squares.append(round(w, 2))
    if not squares:
        return None
    size, n = _modal(squares, 0.32)
    ev.record('chapter_chip_in', size, shapes=n, why='modal square in the title band')
    return size
```

- [ ] **Step 4: Run the tests.** Expected: 4 passed.

- [ ] **Step 5: Commit**

```
feat(atb): the grid from the edges things repeatedly sit on

Margin, gutter, and the title, body and footer bands, each the modal value with its
shape count beside it. cols stays 12 as the authoring convenience — the reference deck
is NOT strictly on twelve columns, which is why an imported region keeps measured
inches instead.

Also the chapter chip: a 0.32in square in the title band on the reference deck, which
the existing title-block component draws as a bar.
```

---

### Task 5: The five values a `.pptx` does not state honestly

**Files:**
- Create: `src/services/atb/pptx_import/derived.py`
- Create: `tests/services/atb/test_derived.py`

**Interfaces:**
- Consumes: `Deck`, `Evidence`, `colours()` output.
- Produces:
  - `series_palette(deck: Deck, ev: Evidence, colours: dict[str, str]) -> list[str]` — never empty
  - `rating_scales(deck: Deck, ev: Evidence) -> dict[str, dict[str, dict[str, str]]]`
  - `writing(deck: Deck, ev: Evidence) -> dict` — `{'locale': 'en-US', 'locale_contested': True, …}`
  - `BRITISH_RE`, `AMERICAN_RE`

- [ ] **Step 1: Write the failing test**

```python
from src.services.atb.pptx_import.derived import rating_scales, series_palette, writing
from src.services.atb.pptx_import.evidence import Evidence
from src.services.atb.pptx_import.read import Deck


def _deck(**kw):
    base = dict(width_in=13.333, height_in=7.5, slides=(), theme={'colours': {}, 'fonts': {}},
                chart_series_colours=(), run_langs={})
    base.update(kw)
    return Deck(**base)


def test_the_series_palette_comes_from_the_charts():
    ev = Evidence()
    deck = _deck(chart_series_colours=('#0F6E78', '#B42D2D', '#172033'))
    assert series_palette(deck, ev, {'ink': '#172033', 'accent': '#2350C8'}) == \
        ['#0F6E78', '#B42D2D', '#172033']


def test_a_deck_with_no_charts_still_gets_a_non_empty_palette():
    # Review Focus 3: series_palette must be non-empty or the pack cannot validate.
    ev = Evidence()
    out = series_palette(_deck(), ev, {'ink': '#172033', 'accent': '#2350C8', 'accent_2': '#0F6E78'})
    assert out, 'a pack with an empty series palette fails validateStyle'
    assert out[0].startswith('#')


def test_a_rating_scale_needs_both_a_vocabulary_and_fills():
    from src.services.atb.pptx_import.read import Box, Shape
    ev = Evidence()
    rows = (('Area', 'Risk'), ('Freight', 'Low'), ('IT', 'High'), ('Tail', 'Low'))
    table = Shape(kind='table', box=Box(0.5, 1.5, 12.33, 4.0), runs=(), fill=None, line=None,
                  table=rows, slide=1, name='Table 1')
    out = rating_scales(_deck(slides=((table,),)), ev)
    # no fills on the cells, so no scale is derived and the reason is recorded
    assert out == {}
    assert any('fill' in i['why'] for i in ev.as_dict()['incidental'])


def test_the_declared_language_is_recorded_and_contested_by_the_spelling():
    ev = Evidence()
    from src.services.atb.pptx_import.read import Box, Run, Shape
    runs = tuple(Run(text=t, size_pt=12, font='Calibri', colour='#172033', bold=False, lang='en-US')
                 for t in ('Virtualisation', 'Mobilise', 'Optimise', 'utilisation'))
    shape = Shape(kind='text', box=Box(0.5, 1.5, 6, 1), runs=runs, fill=None, line=None,
                  table=None, slide=1, name='TextBox 1')
    out = writing(_deck(slides=((shape,),), run_langs={'en-US': 4}), ev)
    assert out['locale'] == 'en-US'
    assert out['locale_contested'] is True
    assert 'en-GB' in out['locale_suggested']


def test_an_uncontested_language_is_not_flagged():
    ev = Evidence()
    from src.services.atb.pptx_import.read import Box, Run, Shape
    runs = (Run(text='organize the color center', size_pt=12, font='Calibri',
                colour='#172033', bold=False, lang='en-US'),)
    shape = Shape(kind='text', box=Box(0.5, 1.5, 6, 1), runs=runs, fill=None, line=None,
                  table=None, slide=1, name='TextBox 1')
    out = writing(_deck(slides=((shape,),), run_langs={'en-US': 1}), ev)
    assert out['locale_contested'] is False
```

- [ ] **Step 2: Run it and watch it fail.**

- [ ] **Step 3: Write `derived.py`**

```python
"""The values validateStyle requires that a .pptx does not simply contain.

Four are derived and one is contested. The deck states a language and spells another: the
reference file declares en-US on all 6,940 runs while writing Virtualisation, Mobilise, Optimise,
utilisation, with 8 -ize uses in 85 slides. So the declared value is recorded, the spelling is
tested, and the contradiction is FLAGGED for a human instead of being resolved by guess.
"""
from __future__ import annotations

import re
from collections import Counter

from .evidence import Evidence
from .read import Deck

BRITISH_RE = re.compile(r'\b\w+(?:isation|ise[sd]?|ising|our|ours)\b', re.I)
AMERICAN_RE = re.compile(r'\b\w+(?:ization|ize[sd]?|izing)\b', re.I)
_BRITISH_FALSE_FRIENDS = re.compile(
    r'\b(?:wise|rise|risen|promise|precise|concise|advertise|exercise|surprise|franchise|'
    r'compromise|merchandise|supervise|televise|revise|devise|arise|otherwise|likewise|'
    r'four|hour|hours|your|yours|tour|tours|pour|flour|our)\b', re.I)


def series_palette(deck: Deck, ev: Evidence, colours: dict[str, str]) -> list[str]:
    if deck.chart_series_colours:
        out = list(deck.chart_series_colours)
        ev.record('series_palette', out, charts=True, why='colours used by the chart series, in first-use order')
        return out
    fallback = [c for c in (colours.get('accent'), colours.get('accent_2'), colours.get('ink'),
                            colours.get('caution'), colours.get('positive'), colours.get('alert_ink'))
                if c]
    out = list(dict.fromkeys(fallback)) or [colours.get('ink', '#000000')]
    ev.record('series_palette', out, charts=False,
              why='the deck holds no charts; the pack\'s accents stand in, because an empty '
                  'series palette fails validateStyle')
    return out


def rating_scales(deck: Deck, ev: Evidence) -> dict:
    """A scale needs a small repeated vocabulary AND a fill per label.

    Without fills there is nothing to colour a chip with, so no scale is derived and the column
    stays text (spec §5a). A human can define one on the review screen.
    """
    scales: dict[str, dict] = {}
    for slide in deck.slides:
        for shape in slide:
            if shape.kind != 'table' or not shape.table or len(shape.table) < 3:
                continue
            header, *body = shape.table
            for col, name in enumerate(header):
                values = [row[col].strip() for row in body if col < len(row) and row[col].strip()]
                if not values:
                    continue
                vocabulary = Counter(values)
                if len(vocabulary) > 5 or len(values) < 3 or max(len(v) for v in values) > 12:
                    continue
                ev.incidental(
                    'rating_scale', f'{name or "column %d" % col} on slide {shape.slide}',
                    'a repeated vocabulary with no cell fill to take a chip colour from; '
                    'the column stays text until a scale is defined by hand')
    return scales


def writing(deck: Deck, ev: Evidence) -> dict:
    declared = Counter(deck.run_langs).most_common(1)
    locale = declared[0][0] if declared else 'en-GB'
    text = ' '.join(run.text for slide in deck.slides for shape in slide for run in shape.runs)
    british = [w for w in BRITISH_RE.findall(text) if not _BRITISH_FALSE_FRIENDS.match(w)]
    american = AMERICAN_RE.findall(text)
    contested = bool(locale.lower().endswith('-us') and len(british) > max(3, len(american)))
    suggested = 'en-GB' if contested else locale
    ev.record('writing.locale', locale,
              runs=declared[0][1] if declared else 0,
              british_spellings=len(british), american_spellings=len(american),
              contested=contested,
              why='declared by the runs' + (
                  '; CONTESTED — the spelling disagrees, a human decides' if contested else ''))
    return {
        'locale': locale,
        'locale_contested': contested,
        'locale_suggested': suggested,
        'title_max_words': 12,
        'title_style': 'assertion',
        'subtitle_style': 'basis',
    }
```

- [ ] **Step 4: Run the tests.** Expected: 5 passed.

- [ ] **Step 5: Prove the non-empty guarantee**

Make `series_palette` return `[]` when there are no charts; confirm
`test_a_deck_with_no_charts_still_gets_a_non_empty_palette` fails. Restore.

- [ ] **Step 6: Commit**

```
feat(atb): the four values a .pptx lacks, and the one it gets wrong

series_palette from the chart parts, with the pack's own accents standing in when a deck
has no charts — an empty one fails validateStyle. A rating scale needs both a repeated
vocabulary and a cell fill to take a chip colour from; without fills the column stays
text and says why.

And the language: the reference deck declares en-US on all 6,940 runs while spelling
Virtualisation, Mobilise, Optimise. The declared value is recorded, the spelling is
tested, and the contradiction is flagged for a human rather than guessed.
```

---

### Task 6: Emit the pack, and hold it to the contract

**Files:**
- Create: `src/services/atb/pptx_import/contract.py`
- Create: `src/services/atb/pptx_import/emit.py`
- Create: `tests/services/atb/test_contract.py`
- Create: `tests/services/atb/test_emit_pack.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces:
  - `validate_pack(pack: dict) -> list[str]` — errors, empty when valid
  - `validate_layout(layout: dict) -> list[str]`
  - `REQUIRED_COLOURS`, `REQUIRED_TYPE_SCALE`, `STYLE_FORMATS`, `COMPONENTS`, `SLOT_TYPES`
  - `build_pack(deck: Deck, ev: Evidence, key: str, name: str) -> dict`
  - `class PackInvalid(Exception)`

- [ ] **Step 1: Write the contract test**

```python
import json
import os
import re

import pytest

from src.services.atb.pptx_import.contract import (
    REQUIRED_COLOURS, REQUIRED_TYPE_SCALE, STYLE_FORMATS, validate_pack)

UI = os.environ.get('BEYOND_PROCWISE_UI', os.path.expanduser('~/PycharmProjects/beyond_procwise_ui'))
VALIDATE_JS = os.path.join(UI, 'src/modules/SpendIQ/atb/validateStyle.js')


def test_rejects_a_pack_missing_a_required_colour():
    errors = validate_pack({'key': 'k', 'format': {'kind': 'deck', 'width_in': 13.333, 'height_in': 7.5},
                            'colours': {'ink': '#000'}, 'type_scale_pt': {}, 'fonts': {},
                            'series_palette': ['#000'], 'rating_scales': {}, 'grid': {'cols': 12},
                            'writing': {'locale': 'en-GB'}})
    assert any('muted' in e for e in errors)


def test_rejects_an_empty_series_palette():
    errors = validate_pack({'key': 'k', 'format': {'kind': 'deck', 'width_in': 1, 'height_in': 1},
                            'colours': {c: '#000000' for c in REQUIRED_COLOURS},
                            'type_scale_pt': {r: 10 for r in REQUIRED_TYPE_SCALE},
                            'fonts': {'body': {'family': 'Calibri', 'fallback': 'sans-serif'}},
                            'series_palette': [], 'rating_scales': {}, 'grid': {'cols': 12},
                            'writing': {'locale': 'en-GB'}})
    assert any('series_palette' in e for e in errors)


@pytest.mark.skipif(not os.path.exists(VALIDATE_JS),
                    reason='set BEYOND_PROCWISE_UI to the UI checkout to run the drift check')
def test_does_not_drift_from_the_javascript_contract():
    """The JS file is the original (it says so in its own header). This asserts the ported
    constants still match it, so the two cannot disagree silently."""
    js = open(VALIDATE_JS, encoding='utf-8').read()
    def arr(name):
        m = re.search(name + r"\s*=\s*\[([^\]]*)\]", js)
        return [v.strip().strip("'\"") for v in m.group(1).split(',') if v.strip()]
    assert arr('REQUIRED_COLOURS') == list(REQUIRED_COLOURS)
    assert arr('REQUIRED_TYPE_SCALE') == list(REQUIRED_TYPE_SCALE)
    assert arr('STYLE_FORMATS') == list(STYLE_FORMATS)
```

- [ ] **Step 2: Write the pack-emit test**

```python
from src.services.atb.pptx_import.contract import validate_pack
from src.services.atb.pptx_import.emit import build_pack
from src.services.atb.pptx_import.evidence import Evidence
from src.services.atb.pptx_import.read import read_deck


def test_a_pack_built_from_a_deck_passes_the_contract(build_deck):
    data = build_deck([[(0.5, 0.35, 12.33, 0.6, 'Title %d' % i, 30, '#172033'),
                        (0.5, 1.5, 6.0, 1.0, 'Body text here', 11.5, '#56627A')]
                       for i in range(12)])
    ev = Evidence()
    pack = build_pack(read_deck(data), ev, key='sample', name='Sample')
    assert validate_pack(pack) == []
    assert pack['format']['kind'] == 'deck'
    assert pack['format']['width_in'] == 13.333


def test_a_portrait_deck_is_a4_portrait_and_states_no_page_size(build_deck):
    # Review Focus 5: kind follows the aspect, and an a4-portrait pack states no width/height.
    data = build_deck([[(0.5, 0.35, 7.0, 0.6, 'Title', 16, '#172033')] for _ in range(12)],
                      width_in=8.27, height_in=11.69)
    ev = Evidence()
    pack = build_pack(read_deck(data), ev, key='p', name='Portrait')
    assert pack['format']['kind'] == 'a4-portrait'
    assert 'width_in' not in pack['format']
    assert validate_pack(pack) == []


def test_the_evidence_travels_with_the_pack(build_deck):
    data = build_deck([[(0.5, 0.35, 12.33, 0.6, 'T', 30, '#172033')] for _ in range(12)])
    ev = Evidence()
    build_pack(read_deck(data), ev, key='k', name='K')
    values = ev.as_dict()['values']
    assert 'colours.ink' in values
    assert values['colours.ink']['value'] == '#172033'
    assert values['colours.ink']['runs'] >= 12
```

- [ ] **Step 3: Run both and watch them fail.**

- [ ] **Step 4: Write `contract.py`**

```python
"""The style-pack and layout contract, ported from the UI's own validators.

SOURCE OF TRUTH: beyond_procwise_ui/src/modules/SpendIQ/atb/validateStyle.js and
validateLayout.js. Their header says "BP_Backend can vendor it", which is what this is. A test
reads the constants back out of the JS and fails if the two drift; it skips when the UI checkout
is not present, and says so rather than passing quietly.
"""
from __future__ import annotations

import re

STYLE_FORMATS = ('deck', 'a4-portrait')
REQUIRED_COLOURS = ('ink', 'muted', 'accent', 'panel')
REQUIRED_TYPE_SCALE = ('title', 'subtitle', 'body', 'table', 'footer')
COMPONENTS = ('title_block', 'kpi_card', 'kpi_card_row', 'text_card', 'text_card_grid',
              'callout', 'paragraph', 'bullet_panel', 'table', 'source_footer')
SLOT_TYPES = ('text', 'list', 'table', 'callout', 'sources', 'chart', 'fact_ref')
FILL_MODES = ('agent', 'bind', 'auto', 'static')

_COLOUR_RE = re.compile(
    r'^(#[0-9a-fA-F]{3}|#[0-9a-fA-F]{6}|#[0-9a-fA-F]{8}|rgba?\([^)]*\)|transparent|inherit|currentColor)$')


class PackInvalid(Exception):
    """The derived pack does not satisfy the contract. Carries the errors."""


def is_colour(value) -> bool:
    return isinstance(value, str) and bool(_COLOUR_RE.match(value.strip()))


def validate_pack(pack: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(pack, dict):
        return ['pack must be an object']
    fmt = pack.get('format')
    if not isinstance(fmt, dict) or fmt.get('kind') not in STYLE_FORMATS:
        errors.append(f'format.kind must be one of {STYLE_FORMATS}')
    elif fmt.get('kind') == 'deck':
        for key in ('width_in', 'height_in'):
            if not isinstance(fmt.get(key), (int, float)) or fmt[key] <= 0:
                errors.append(f'format.{key} is required for a deck')
    colours = pack.get('colours')
    if not isinstance(colours, dict):
        errors.append('colours must be an object')
    else:
        for key in REQUIRED_COLOURS:
            if colours.get(key) is None:
                errors.append(f'colours.{key} is required')
            elif not is_colour(colours[key]):
                errors.append(f'colours.{key} "{colours[key]}" is not a usable CSS colour')
    scale = pack.get('type_scale_pt')
    if not isinstance(scale, dict):
        errors.append('type_scale_pt must be an object')
    else:
        for key in REQUIRED_TYPE_SCALE:
            if not isinstance(scale.get(key), (int, float)) or scale[key] <= 0:
                errors.append(f'type_scale_pt.{key} must be a positive number')
    fonts = pack.get('fonts')
    if not isinstance(fonts, dict) or not fonts:
        errors.append('fonts must be a non-empty object')
    else:
        for role, font in fonts.items():
            if not isinstance(font, dict) or not font.get('family'):
                errors.append(f'fonts.{role}.family must be a string')
            elif not font.get('fallback'):
                errors.append(f'fonts.{role}.fallback is required — the named family may not be installed')
    series = pack.get('series_palette')
    if not isinstance(series, list) or not series:
        errors.append('series_palette must be a non-empty array')
    else:
        errors += [f'series_palette[{i}] "{c}" is not a usable CSS colour'
                   for i, c in enumerate(series) if not is_colour(c)]
    if not isinstance(pack.get('rating_scales'), dict):
        errors.append('rating_scales must be an object')
    else:
        for name, scale_def in pack['rating_scales'].items():
            if not isinstance(scale_def, dict):
                errors.append(f'rating_scales.{name} must be an object')
                continue
            for label, chip in scale_def.items():
                if not isinstance(chip, dict):
                    errors.append(f'rating_scales.{name}.{label} must be an object')
                    continue
                for part in ('bg', 'ink'):
                    if part in chip and not is_colour(chip[part]):
                        errors.append(f'rating_scales.{name}.{label}.{part} is not a usable CSS colour')
    grid = pack.get('grid')
    if not isinstance(grid, dict) or grid.get('cols') != 12:
        errors.append('grid.cols must be 12')
    writing = pack.get('writing')
    if not isinstance(writing, dict) or not writing.get('locale'):
        errors.append('writing.locale is required — it decides the spelling list and number formats')
    return errors


def validate_layout(layout: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(layout, dict):
        return ['layout must be an object']
    if not layout.get('id'):
        errors.append('id is required')
    formats = layout.get('formats')
    if not isinstance(formats, list) or not formats or any(f not in STYLE_FORMATS for f in formats):
        errors.append(f'formats must be a non-empty subset of {STYLE_FORMATS}')
    regions = layout.get('regions')
    if not isinstance(regions, list) or not regions:
        errors.append('regions must be a non-empty array')
    else:
        for region in regions:
            if not isinstance(region, dict):
                errors.append('each region must be an object')
                continue
            if region.get('component') not in COMPONENTS:
                errors.append(f'region {region.get("id")} names an unknown component '
                              f'"{region.get("component")}"')
            has_grid, has_box = 'grid' in region, 'box_in' in region
            if has_grid and has_box:
                errors.append(f'region {region.get("id")} states both grid and box_in — exactly one')
            if has_box:
                box = region['box_in']
                if not isinstance(box, dict) or any(
                        not isinstance(box.get(k), (int, float)) for k in ('x', 'y', 'w', 'h')):
                    errors.append(f'region {region.get("id")} box_in needs numeric x, y, w, h')
    slots = layout.get('slots')
    if not isinstance(slots, dict):
        errors.append('slots must be an object')
    else:
        for name, slot in slots.items():
            if not isinstance(slot, dict):
                errors.append(f'slot {name} must be an object')
                continue
            if slot.get('type') not in SLOT_TYPES:
                errors.append(f'slot {name} has unknown type "{slot.get("type")}"')
            if slot.get('fill') not in FILL_MODES:
                errors.append(f'slot {name} has unknown fill "{slot.get("fill")}"')
            if slot.get('fill') == 'agent' and slot.get('type') in ('text', 'callout') \
                    and slot.get('max_chars') is None and slot.get('max_words') is None:
                errors.append(f'slot {name} is written by the agent and states no length')
    return errors
```

- [ ] **Step 5: Write `emit.py` (pack half)**

```python
"""Assemble the measurements into a pack the existing validators accept."""
from __future__ import annotations

from . import derived, grid as grid_mod, palette, typescale
from .contract import PackInvalid, validate_pack
from .evidence import Evidence
from .read import Deck


def build_pack(deck: Deck, ev: Evidence, key: str, name: str) -> dict:
    colours = palette.colours(deck, ev)
    scale = typescale.type_scale(deck, ev)
    fonts = typescale.fonts(deck, ev)
    grid = grid_mod.grid(deck, ev)
    chip = grid_mod.chapter_chip_in(deck, ev)
    is_deck = deck.width_in > deck.height_in
    fmt: dict = {'kind': 'deck' if is_deck else 'a4-portrait'}
    if is_deck:
        fmt['width_in'] = deck.width_in
        fmt['height_in'] = deck.height_in
    if deck.theme.get('colours'):
        ev.ignored('theme.colours', deck.theme['colours'],
                   'the deck paints its own shapes; a theme colour is not what it looks like')
    pack = {
        'key': key,
        'name': name,
        'format': fmt,
        'colours': colours,
        'type_scale_pt': scale,
        'fonts': fonts,
        'grid': grid,
        'series_palette': derived.series_palette(deck, ev, colours),
        'rating_scales': derived.rating_scales(deck, ev),
        'writing': derived.writing(deck, ev),
    }
    if chip:
        pack['chapter_chip_in'] = chip
    errors = validate_pack(pack)
    if errors:
        raise PackInvalid('; '.join(errors))
    return pack
```

- [ ] **Step 6: Run both test files.** Expected: 3 + 3 passed (the drift test skips unless the UI checkout is present; run it once with `BEYOND_PROCWISE_UI` set and confirm it passes).

- [ ] **Step 7: Commit**

```
feat(atb): emit a pack and refuse to emit an invalid one

The contract is ported from the UI's validateStyle.js and validateLayout.js, whose own
header says BP_Backend can vendor them, with a test that reads the constants back out of
the JS so the two cannot drift silently. It skips when the UI checkout is absent and
says which variable to set.

A pack that fails the contract raises rather than being stored: a half-valid pack renders
a page with a missing colour, and nothing downstream would tell you why.
```

---

### Task 7: Group the slides

**Files:**
- Create: `src/services/atb/pptx_import/cluster.py`
- Create: `tests/services/atb/test_cluster.py`

**Interfaces:**
- Consumes: `Deck`, `Shape`.
- Produces:
  - `signature(shapes: tuple[Shape, ...], height_in: float) -> tuple`
  - `Cluster(signature: tuple, slides: tuple[int, ...], rows: tuple[tuple[Shape, ...], ...])` — frozen dataclass
  - `group(deck: Deck) -> list[Cluster]` — sorted by slide count descending, then first slide
  - `TITLE_BAND_IN = 1.25`, `FOOTER_BAND_IN = 6.9`, `SAME_ROW_IN = 0.45`, `SAME_COL_IN = 0.1`

- [ ] **Step 1: Write the failing test**

```python
from src.services.atb.pptx_import.cluster import group, signature
from src.services.atb.pptx_import.read import Box, Deck, Shape


def _shape(x, y, w=2.0, h=1.0, kind='text', slide=1):
    return Shape(kind=kind, box=Box(x, y, w, h), runs=(), fill=None, line=None,
                 table=None, slide=slide, name='s')


def _deck(slides):
    return Deck(width_in=13.333, height_in=7.5, slides=tuple(slides),
                theme={'colours': {}, 'fonts': {}}, chart_series_colours=(), run_langs={})


def test_the_title_band_and_the_footer_are_not_part_of_the_signature():
    shapes = (_shape(0.5, 0.35, 12.33, 0.6), _shape(0.5, 2.0), _shape(0.5, 7.02, 11.7, 0.3))
    assert signature(shapes, 7.5) == ((1, 'SHAPE'),)


def test_columns_within_a_tenth_of_an_inch_are_one_column():
    shapes = (_shape(0.5, 2.0), _shape(0.55, 2.0), _shape(4.0, 2.0))
    assert signature(shapes, 7.5) == ((2, 'SHAPE'),)


def test_a_table_row_and_a_chart_row_never_merge():
    a = signature((_shape(0.5, 2.0, kind='table'),), 7.5)
    b = signature((_shape(0.5, 2.0, kind='chart'),), 7.5)
    assert a != b


def test_a_trailing_repeated_row_collapses_to_one_marked_repeating():
    rows = tuple(_shape(0.5 + i * 3, y) for y in (2.0, 3.5, 5.0) for i in range(4))
    sig = signature(rows, 7.5)
    assert sig[-1][-1] == 'repeat'
    assert len(sig) == 1


def test_groups_slides_by_signature_and_orders_by_use():
    one = (_shape(0.5, 2.0, kind='table'),)
    two = (_shape(0.5, 2.0), _shape(4.0, 2.0))
    clusters = group(_deck([one, two, one, one]))
    assert [len(c.slides) for c in clusters] == [3, 1]
    assert clusters[0].slides == (1, 3, 4)
```

- [ ] **Step 2: Run it and watch it fail.**

- [ ] **Step 3: Write `cluster.py`**

```python
"""How many layouts a deck really has.

Measured on the reference deck (85 slides): these rules give 34 groups — 8 used by more than one
slide covering 59 of them, and 26 used once. An earlier draft of the spec guessed "the mid-teens";
it was wrong, and generalising the repeat rule from a single row to a repeating SEQUENCE of rows
(tested) changes the count not at all. §6a: the 8 are emitted as layouts and the 26 are listed
for step 2, because a quadrant is not a template.
"""
from __future__ import annotations

from dataclasses import dataclass

from .read import Deck, Shape

TITLE_BAND_IN = 1.25
FOOTER_BAND_IN = 6.9
SAME_ROW_IN = 0.45
SAME_COL_IN = 0.1
_DECORATION_IN = 0.5


@dataclass(frozen=True)
class Cluster:
    signature: tuple
    slides: tuple[int, ...]
    rows: tuple[tuple[Shape, ...], ...]

    @property
    def reused(self) -> bool:
        return len(self.slides) > 1


def _body(shapes: tuple[Shape, ...], height_in: float) -> list[Shape]:
    footer = min(FOOTER_BAND_IN, height_in - 0.6)
    return [s for s in shapes
            if TITLE_BAND_IN <= s.box.y <= footer
            and not (s.box.w < _DECORATION_IN and s.box.h < _DECORATION_IN)]


def rows_of(shapes: tuple[Shape, ...], height_in: float) -> list[list[Shape]]:
    body = sorted(_body(shapes, height_in), key=lambda s: (s.box.y, s.box.x))
    rows: list[list[Shape]] = []
    current: list[Shape] = []
    anchor: float | None = None
    for shape in body:
        if anchor is None or abs(shape.box.y - anchor) <= SAME_ROW_IN:
            current.append(shape)
            anchor = shape.box.y if anchor is None else anchor
        else:
            rows.append(current)
            current = [shape]
            anchor = shape.box.y
    if current:
        rows.append(current)
    return rows


def _row_signature(row: list[Shape]) -> tuple[int, str]:
    columns: list[float] = []
    for shape in sorted(row, key=lambda s: s.box.x):
        if not columns or abs(shape.box.x - columns[-1]) > SAME_COL_IN:
            columns.append(shape.box.x)
    kinds = {s.kind for s in row}
    kind = 'TABLE' if 'table' in kinds else 'CHART' if 'chart' in kinds else 'SHAPE'
    return len(columns), kind


def signature(shapes: tuple[Shape, ...], height_in: float) -> tuple:
    rows = [_row_signature(r) for r in rows_of(shapes, height_in)]
    if not rows:
        return (('empty',),)
    out: list[tuple] = list(rows)
    collapsed = False
    while len(out) >= 2 and out[-1][:2] == out[-2][:2]:
        out.pop()
        collapsed = True
    if collapsed:
        out[-1] = out[-1] + ('repeat',)
    return tuple(out)


def group(deck: Deck) -> list[Cluster]:
    found: dict[tuple, list[int]] = {}
    rows_by_sig: dict[tuple, tuple] = {}
    for index, shapes in enumerate(deck.slides, 1):
        sig = signature(shapes, deck.height_in)
        found.setdefault(sig, []).append(index)
        rows_by_sig.setdefault(sig, tuple(tuple(r) for r in rows_of(shapes, deck.height_in)))
    clusters = [Cluster(signature=sig, slides=tuple(slides), rows=rows_by_sig[sig])
                for sig, slides in found.items()]
    clusters.sort(key=lambda c: (-len(c.slides), c.slides[0]))
    return clusters
```

- [ ] **Step 4: Run the tests.** Expected: 5 passed.

- [ ] **Step 5: Prove rule 3 is load-bearest**

Make `_row_signature` always return `'SHAPE'`; confirm `test_a_table_row_and_a_chart_row_never_merge`
fails. Restore.

- [ ] **Step 6: Commit**

```
feat(atb): group 85 slides into the layouts they actually are

Signature per slide: body rows by column count and content kind, title band and footer
excluded, columns within a tenth of an inch counted as one. Three merges and no more —
near-equal columns, a trailing repeated row, and never a table row with a chart row.

Measured on the reference deck: 34 groups, 8 of them reused across 59 slides. The spec's
earlier "mid-teens" was a guess and this is the correction.
```

---

### Task 8: Regions, slot types, loose fits and unresolved

**Files:**
- Create: `src/services/atb/pptx_import/slots.py`
- Create: `tests/services/atb/test_slots.py`

**Interfaces:**
- Consumes: `Cluster`, `Deck`, `Shape`, the pack's `grid`, `Evidence`.
- Produces:
  - `LOOSE_FIT_IN = 0.15`
  - `regions_and_slots(cluster: Cluster, deck: Deck, pack: dict, ev: Evidence) -> tuple[list[dict], dict, list[dict]]`
    returning `(regions, slots, problems)` where a problem is
    `{'kind': 'loose_fit' | 'unresolved', 'region': str, 'why': str, 'slides': [int]}`

- [ ] **Step 1: Write the failing test**

```python
from src.services.atb.pptx_import.cluster import Cluster, group
from src.services.atb.pptx_import.evidence import Evidence
from src.services.atb.pptx_import.read import Box, Deck, Shape
from src.services.atb.pptx_import.slots import regions_and_slots

PACK = {'grid': {'cols': 12, 'margin_in': 0.5, 'gutter_in': 0.3, 'title_top_in': 0.35,
                 'body_top_in': 1.5, 'footer_top_in': 7.02},
        'type_scale_pt': {'title': 30, 'subtitle': 14, 'body': 11.5, 'table': 10, 'footer': 9},
        'format': {'kind': 'deck', 'width_in': 13.333, 'height_in': 7.5}}


def _shape(x, y, w, h, kind='text', slide=1, table=None):
    return Shape(kind=kind, box=Box(x, y, w, h), runs=(), fill=None, line=None,
                 table=table, slide=slide, name='s')


def _deck(slides):
    return Deck(width_in=13.333, height_in=7.5, slides=tuple(slides),
                theme={'colours': {}, 'fonts': {}}, chart_series_colours=(), run_langs={})


def test_a_table_on_every_member_becomes_a_table_slot():
    rows = (('Area', 'Effect'), ('Freight', 'one rate card'))
    slides = [( _shape(0.5, 1.5, 12.33, 4.2, kind='table', slide=i, table=rows),) for i in (1, 2)]
    deck = _deck(slides)
    cluster = group(deck)[0]
    regions, slots, problems = regions_and_slots(cluster, deck, PACK, Evidence())
    assert any(r['component'] == 'table' for r in regions)
    assert slots['rows']['type'] == 'table'
    assert [c['id'] for c in slots['rows']['columns']] == ['area', 'effect']
    assert problems == []


def test_members_that_disagree_leave_the_region_unresolved():
    a = (_shape(0.5, 1.5, 12.33, 4.2, kind='table', slide=1, table=(('A',), ('b',))),)
    b = (_shape(0.5, 1.5, 12.33, 4.2, kind='chart', slide=2),)
    deck = _deck([a, b])
    # force both into one cluster to exercise the disagreement path
    cluster = Cluster(signature=(('x',),), slides=(1, 2), rows=(tuple(a) + tuple(b),))
    regions, slots, problems = regions_and_slots(cluster, deck, PACK, Evidence())
    assert any(p['kind'] == 'unresolved' for p in problems)


def test_a_member_whose_box_is_off_by_more_than_the_tolerance_is_a_loose_fit():
    a = (_shape(0.5, 1.5, 12.33, 4.2, slide=1),)
    b = (_shape(0.5, 1.5, 11.0, 4.2, slide=2),)
    deck = _deck([a, b])
    cluster = Cluster(signature=(('x',),), slides=(1, 2), rows=(tuple(a) + tuple(b),))
    _, _, problems = regions_and_slots(cluster, deck, PACK, Evidence())
    assert any(p['kind'] == 'loose_fit' and 2 in p['slides'] for p in problems)


def test_an_agent_text_slot_always_carries_a_length():
    slides = [(_shape(0.5, 1.5, 6.0, 1.0, slide=i),) for i in (1, 2)]
    deck = _deck(slides)
    cluster = group(deck)[0]
    _, slots, _ = regions_and_slots(cluster, deck, PACK, Evidence())
    for name, slot in slots.items():
        if slot.get('fill') == 'agent' and slot['type'] in ('text', 'callout'):
            assert slot.get('max_chars') or slot.get('max_words'), name


def test_every_region_carries_a_measured_box_not_a_grid_span():
    slides = [(_shape(0.5, 1.5, 6.0, 1.0, slide=i),) for i in (1, 2)]
    deck = _deck(slides)
    cluster = group(deck)[0]
    regions, _, _ = regions_and_slots(cluster, deck, PACK, Evidence())
    assert all('box_in' in r and 'grid' not in r for r in regions)
```

- [ ] **Step 2: Run it and watch it fail.**

- [ ] **Step 3: Write `slots.py`**

```python
"""What each region is, and what goes in it.

A region's box is the MEDIAN of its members' boxes, in inches, because the reference deck is not
strictly on a twelve-column grid. A member more than LOOSE_FIT_IN off the median is reported with
its slide number rather than averaged away: a cluster that should have been two is then visible.

A region whose members disagree about their content is `unresolved`, carrying both readings. The
document-type resolver took the same ruling when two types tied: an honest unresolved beats a
coin flip.
"""
from __future__ import annotations

from statistics import median

from .cluster import Cluster
from .evidence import Evidence
from .read import Deck, Shape

LOOSE_FIT_IN = 0.15
_CHARS_PER_EM = 0.5
_LINE_HEIGHT = 1.35
_PT_PER_IN = 72


def _slug(text: str, fallback: str) -> str:
    out = ''.join(c.lower() if c.isalnum() else '_' for c in text).strip('_')
    return out or fallback


def _median_box(shapes: list[Shape]) -> dict:
    return {
        'x': round(median(s.box.x for s in shapes), 3),
        'y': round(median(s.box.y for s in shapes), 3),
        'w': round(median(s.box.w for s in shapes), 3),
        'h': round(median(s.box.h for s in shapes), 3),
    }


def _max_chars(box: dict, size_pt: float, lines: int | None = None) -> int:
    per_line = max(1, int((box['w'] * _PT_PER_IN) / (size_pt * _CHARS_PER_EM)))
    if lines is None:
        lines = max(1, int((box['h'] * _PT_PER_IN) / (size_pt * _LINE_HEIGHT)))
    return per_line * lines


def regions_and_slots(cluster: Cluster, deck: Deck, pack: dict, ev: Evidence):
    scale = pack['type_scale_pt']
    regions: list[dict] = []
    slots: dict[str, dict] = {}
    problems: list[dict] = []

    # Every layout has a title block, because every slide in the reference deck does.
    regions.append({'id': 'title', 'component': 'title_block',
                    'box_in': {'x': pack['grid']['margin_in'], 'y': pack['grid']['title_top_in'],
                               'w': round(deck.width_in - 2 * pack['grid']['margin_in'], 3),
                               'h': 0.6}})
    title_box = regions[0]['box_in']
    slots['title'] = {'type': 'text', 'fill': 'agent', 'required': True, 'style': 'assertion',
                      'max_words': max(4, int(_max_chars(title_box, scale['title'], lines=2) / 6.1))}
    slots['subtitle'] = {'type': 'text', 'fill': 'agent', 'required': True, 'style': 'basis',
                         'max_chars': _max_chars(title_box, scale['subtitle'], lines=2)}

    for index, row in enumerate(cluster.rows, 1):
        kinds = {s.kind for s in row}
        box = _median_box(list(row))
        for shape in row:
            off = max(abs(shape.box.x - box['x']), abs(shape.box.y - box['y']),
                      abs(shape.box.w - box['w']), abs(shape.box.h - box['h']))
            if off > LOOSE_FIT_IN:
                problems.append({'kind': 'loose_fit', 'region': f'row{index}',
                                 'why': f'slide {shape.slide} is {round(off, 2)}in off the median box',
                                 'slides': [shape.slide]})
        content = {k for k in kinds if k in ('table', 'chart')}
        if len(content) > 1:
            problems.append({'kind': 'unresolved', 'region': f'row{index}',
                             'why': f'members disagree: {sorted(content)}',
                             'slides': sorted({s.slide for s in row})})
            continue
        if 'table' in kinds:
            table = next((s.table for s in row if s.table), None)
            header = list(table[0]) if table else []
            columns = [{'id': _slug(name, f'col{i}'), 'label': name, 'type': 'text',
                        'max_chars': _max_chars({'w': box['w'] / max(1, len(header)), 'h': 0.3},
                                                scale['table'], lines=2)}
                       for i, name in enumerate(header)]
            regions.append({'id': 'rows', 'component': 'table', 'box_in': box})
            slots['rows'] = {'type': 'table', 'fill': 'agent',
                             'max_rows': max((len(s.table) - 1) for s in row if s.table) or 1,
                             'columns': columns}
            continue
        if 'chart' in kinds:
            regions.append({'id': f'chart{index}', 'component': 'kpi_card_row', 'box_in': box})
            slots[f'chart{index}'] = {'type': 'chart', 'fill': 'bind'}
            continue
        columns = len({round(s.box.x, 1) for s in row})
        if columns > 1:
            regions.append({'id': f'cards{index}', 'component': 'text_card', 'box_in': box})
            slots[f'cards{index}'] = {
                'type': 'list', 'fill': 'agent', 'min': 2, 'max': columns,
                'item': {
                    'heading': {'type': 'text',
                                'max_chars': _max_chars({'w': box['w'] / columns, 'h': 0.4},
                                                        scale.get('card_title', scale['body']), lines=2)},
                    'body': {'type': 'text',
                             'max_chars': _max_chars({'w': box['w'] / columns, 'h': box['h']},
                                                     scale['body'])},
                },
            }
        else:
            regions.append({'id': f'prose{index}', 'component': 'paragraph', 'box_in': box})
            slots[f'prose{index}'] = {'type': 'text', 'fill': 'agent',
                                      'max_chars': _max_chars(box, scale['body'])}

    regions.append({'id': 'footer', 'component': 'source_footer'})
    slots['sources'] = {'type': 'sources', 'fill': 'auto'}
    ev.record(f'layout.{cluster.signature}', {'regions': len(regions), 'problems': len(problems)},
              slides=list(cluster.slides))
    return regions, slots, problems
```

- [ ] **Step 4: Run the tests.** Expected: 5 passed.

- [ ] **Step 5: Commit**

```
feat(atb): regions from the median box, and honest problems

A region's box is the median of its members' in inches, not a grid span, because the
reference deck is not strictly on twelve columns. A member more than 0.15in off is
reported with its slide number instead of averaged away, so a cluster that should have
been two is visible.

Members that disagree about their content leave the region unresolved with both
readings. Every agent text slot gets a length from its own measured box.
```

---

### Task 9: The example fill, and emitting a layout

**Files:**
- Modify: `src/services/atb/pptx_import/emit.py` (add `build_layout`)
- Create: `src/services/atb/pptx_import/example.py`
- Create: `tests/services/atb/test_emit_layout.py`

**Interfaces:**
- Consumes: Tasks 6, 7, 8.
- Produces:
  - `example_fill(cluster: Cluster, deck: Deck, slots: dict) -> tuple[dict, dict]` returning
    `(fill, source)` where `source` is `{'file': str, 'slide': int}` — `file` filled by the caller
  - `build_layout(cluster: Cluster, deck: Deck, pack: dict, ev: Evidence, filename: str) -> dict`
  - `proposed_name(cluster: Cluster) -> str`

- [ ] **Step 1: Write the failing test**

```python
from src.services.atb.pptx_import.cluster import group
from src.services.atb.pptx_import.contract import validate_layout
from src.services.atb.pptx_import.emit import build_layout, proposed_name
from src.services.atb.pptx_import.evidence import Evidence
from src.services.atb.pptx_import.read import Box, Deck, Run, Shape

PACK = {'grid': {'cols': 12, 'margin_in': 0.5, 'gutter_in': 0.3, 'title_top_in': 0.35,
                 'body_top_in': 1.5, 'footer_top_in': 7.02},
        'type_scale_pt': {'title': 30, 'subtitle': 14, 'card_title': 13, 'body': 11.5,
                          'table': 10, 'footer': 9},
        'format': {'kind': 'deck', 'width_in': 13.333, 'height_in': 7.5}}


def _text(x, y, w, h, words, slide=1):
    runs = (Run(text=words, size_pt=12, font='Calibri', colour='#172033', bold=False, lang='en-GB'),)
    return Shape(kind='text', box=Box(x, y, w, h), runs=runs, fill=None, line=None,
                 table=None, slide=slide, name='TextBox')


def _deck(slides):
    return Deck(width_in=13.333, height_in=7.5, slides=tuple(slides),
                theme={'colours': {}, 'fonts': {}}, chart_series_colours=(), run_langs={})


def test_a_layout_passes_the_contract_and_keeps_an_example():
    slides = [(_text(0.5, 0.35, 12.33, 0.6, 'Eight headline recommendations', i),
               _text(0.5, 1.5, 6.0, 1.0, 'Dual-track the hypervisor', i)) for i in (1, 2)]
    deck = _deck(slides)
    cluster = group(deck)[0]
    layout = build_layout(cluster, deck, PACK, Evidence(), 'Pack.pptx')
    assert validate_layout(layout) == []
    assert layout['example_source'] == {'file': 'Pack.pptx', 'slide': 1}
    assert 'Eight headline recommendations' in str(layout['example_fill'])


def test_the_example_is_labelled_as_the_source_document_s_words():
    slides = [(_text(0.5, 0.35, 12.33, 0.6, 'A client sentence', i),) for i in (1, 2)]
    deck = _deck(slides)
    layout = build_layout(group(deck)[0], deck, PACK, Evidence(), 'Pack.pptx')
    assert layout['example_source']['file'] == 'Pack.pptx'
    assert layout['example_source']['slide'] == 1


def test_the_proposed_name_describes_the_structure():
    slides = [(_text(0.5, 1.5, 2.8, 1.0, 'a', i), _text(3.6, 1.5, 2.8, 1.0, 'b', i),
               _text(6.7, 1.5, 2.8, 1.0, 'c', i)) for i in (1, 2)]
    deck = _deck(slides)
    assert '3-up' in proposed_name(group(deck)[0])


def test_the_layout_id_is_stable_across_two_runs_of_the_same_deck():
    slides = [(_text(0.5, 1.5, 6.0, 1.0, 'x', i),) for i in (1, 2)]
    deck = _deck(slides)
    a = build_layout(group(deck)[0], deck, PACK, Evidence(), 'Pack.pptx')
    b = build_layout(group(deck)[0], deck, PACK, Evidence(), 'Pack.pptx')
    assert a['id'] == b['id']
    assert a == b
```

- [ ] **Step 2: Run it and watch it fail.**

- [ ] **Step 3: Write `example.py`**

```python
"""One example fill per layout, taken from its first member slide.

WHY AT ALL: an empty fill renders a blank page, which is indistinguishable from a broken
renderer — that exact bug shipped on 2026-10-02 and was found by review.

WHY LABELLED: these are the source document's words, and they belong to whoever wrote it. The
fill travels with {'file', 'slide'} so every screen can say "as it appeared in <file>, slide N"
and nobody can mistake it for their own figures.
"""
from __future__ import annotations

from .cluster import Cluster
from .read import Deck


def _words(shape) -> str:
    return ' '.join(r.text for r in shape.runs if r.text.strip()).strip()


def example_fill(cluster: Cluster, deck: Deck, slots: dict) -> tuple[dict, dict]:
    slide_no = cluster.slides[0]
    shapes = deck.slides[slide_no - 1]
    fill: dict = {'slots': {}}

    titles = sorted([s for s in shapes if s.box.y < 1.0 and _words(s)], key=lambda s: s.box.y)
    if titles:
        fill['slots']['title'] = {'text': _words(titles[0])}
    if len(titles) > 1:
        fill['slots']['subtitle'] = {'text': _words(titles[1])}

    for name, slot in slots.items():
        if name in ('title', 'subtitle', 'sources'):
            continue
        if slot['type'] == 'list':
            items = [{'heading': _words(s)[:80], 'body': ''} for r in cluster.rows for s in r
                     if _words(s)][:slot.get('max', 4)]
            if items:
                fill['slots'][name] = {'items': items}
        elif slot['type'] == 'table':
            table = next((s.table for r in cluster.rows for s in r if s.table), None)
            if table:
                ids = [c['id'] for c in slot['columns']]
                fill['slots'][name] = {'rows': [dict(zip(ids, row)) for row in table[1:]]}
        elif slot['type'] == 'text':
            prose = [_words(s) for r in cluster.rows for s in r if _words(s)]
            if prose:
                fill['slots'][name] = {'text': prose[0]}
    return fill, {'slide': slide_no}
```

- [ ] **Step 4: Add `build_layout` and `proposed_name` to `emit.py`**

```python
def proposed_name(cluster) -> str:
    """A geometric description, never a name. §9.2: naming is the human's."""
    if cluster.signature == ((('empty',),),) or cluster.signature == (('empty',),):
        return 'title only'
    parts = []
    for row in cluster.signature:
        cols, kind = row[0], row[1]
        repeat = ' repeating' if len(row) > 2 else ''
        word = {'TABLE': 'table', 'CHART': 'chart'}.get(kind, 'cards' if cols > 1 else 'panel')
        parts.append(f'{cols}-up {word}{repeat}' if cols > 1 else f'full-width {word}{repeat}')
    return ' + '.join(parts)


def layout_key(cluster) -> str:
    """Stable across runs: derived from the signature, not from a counter or a uuid."""
    import hashlib
    digest = hashlib.sha256(repr(cluster.signature).encode()).hexdigest()[:10]
    return f'imported_{digest}'


def build_layout(cluster, deck, pack: dict, ev: Evidence, filename: str) -> dict:
    from .contract import validate_layout
    from .example import example_fill
    from .slots import regions_and_slots

    regions, slots, problems = regions_and_slots(cluster, deck, pack, ev)
    fill, source = example_fill(cluster, deck, slots)
    layout = {
        'id': layout_key(cluster),
        'version': 1,
        'proposed_name': proposed_name(cluster),
        'formats': [pack['format']['kind']],
        'regions': regions,
        'slots': slots,
        'slide_refs': list(cluster.slides),
        'example_fill': fill,
        'example_source': {'file': filename, **source},
        'problems': problems,
        'writing_guidance': '',
        'pagination': None,
    }
    errors = validate_layout(layout)
    if errors:
        raise PackInvalid(f'layout {layout["id"]}: ' + '; '.join(errors))
    return layout
```

- [ ] **Step 5: Run the tests.** Expected: 4 passed.

- [ ] **Step 6: Commit**

```
feat(atb): emit a layout, with one labelled example from its own slide

The layout id is a hash of its signature, so importing the same deck twice gives the same
ids — determinism is an acceptance criterion, and a counter or a uuid would break it.

The example fill exists because an empty fill renders a blank page, which looks exactly
like a broken renderer; that bug shipped this morning. It carries the file and slide it
came from so every screen can label it as the source document's words.
```

---

### Task 10: Storage, and the write order AUTOCOMMIT forces

**Files:**
- Create: `deploy/sql/2026-10-02_atb_style_pack.sql`
- Create: `deploy/sql/2026-10-02_atb_style_pack_rollback.sql`
- Create: `src/services/atb/pptx_import/store.py`
- Create: `tests/services/atb/test_store.py`

**Interfaces:**
- Consumes: pack and layout dicts from Tasks 6 and 9.
- Produces:
  - `insert_importing(conn, *, pack_key, version, source_file, source_sha256, slide_count, pack, evidence, user) -> str`
  - `insert_layout(conn, *, pack_id, layout) -> str`
  - `mark_candidate(conn, pack_id) -> None`
  - `packs(conn, *, include_importing=False) -> list[dict]`
  - `pack(conn, pack_id) -> dict | None`
  - `layouts(conn, *, pack_id=None, status=None) -> list[dict]`
  - `rename_layout(conn, layout_id, name, user) -> None`
  - `set_layout_status(conn, layout_id, status, user) -> None`
  - `set_pack_status(conn, pack_id, status, user) -> None`
  - `next_version(conn, pack_key) -> int`
  - `inherited(conn, pack_key) -> dict` — `{'names': {layout_key: name}, 'rating_scales': {...}, 'locale': str | None, 'rejected': [layout_key]}`
  - `define_rating_scale(conn, pack_id, name, chips, user) -> None`

- [ ] **Step 1: Write the migration**

`deploy/sql/2026-10-02_atb_style_pack.sql`:

```sql
-- ATB: a style pack and its page layouts, measured from an uploaded .pptx.
--
-- Two tables, candidates until a human approves them. `importing` is a status no read path
-- serves: get_conn() is AUTOCOMMIT, so an import cannot be one transaction, and a crash must
-- leave something invisible rather than something half-valid.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_style_pack (
    pack_id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    pack_key       text NOT NULL,
    version        integer NOT NULL,
    source_file    text NOT NULL,
    source_sha256  text NOT NULL,
    slide_count    integer NOT NULL,
    format         jsonb NOT NULL,
    tokens         jsonb NOT NULL,
    evidence       jsonb NOT NULL DEFAULT '{}'::jsonb,
    status         text NOT NULL DEFAULT 'importing'
                   CHECK (status IN ('importing', 'candidate', 'approved', 'rejected')),
    notes          text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    created_by     text NOT NULL,
    approved_at    timestamptz,
    approved_by    text,
    CONSTRAINT uq_bp_style_pack_key_version UNIQUE (pack_key, version)
);

CREATE INDEX IF NOT EXISTS ix_bp_style_pack_status_created
    ON proc.bp_style_pack (status, created_at DESC);

CREATE TABLE IF NOT EXISTS proc.bp_page_layout (
    layout_id      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    pack_id        uuid NOT NULL REFERENCES proc.bp_style_pack (pack_id) ON DELETE CASCADE,
    layout_key     text NOT NULL,
    proposed_name  text NOT NULL,
    name           text,
    slide_refs     integer[] NOT NULL DEFAULT '{}',
    regions        jsonb NOT NULL,
    slots          jsonb NOT NULL,
    example_fill   jsonb NOT NULL DEFAULT '{}'::jsonb,
    example_source jsonb NOT NULL DEFAULT '{}'::jsonb,
    problems       jsonb NOT NULL DEFAULT '[]'::jsonb,
    status         text NOT NULL DEFAULT 'candidate'
                   CHECK (status IN ('candidate', 'approved', 'rejected')),
    created_at     timestamptz NOT NULL DEFAULT now(),
    approved_at    timestamptz,
    approved_by    text,
    CONSTRAINT uq_bp_page_layout_pack_key UNIQUE (pack_id, layout_key)
);

CREATE INDEX IF NOT EXISTS ix_bp_page_layout_pack_status
    ON proc.bp_page_layout (pack_id, status);

COMMIT;
```

`deploy/sql/2026-10-02_atb_style_pack_rollback.sql`:

```sql
BEGIN;
DROP TABLE IF EXISTS proc.bp_page_layout;
DROP TABLE IF EXISTS proc.bp_style_pack;
COMMIT;
```

- [ ] **Step 2: Write the failing test**

```python
import json
import os

import pytest

from src.services.atb.pptx_import import store

pytestmark = pytest.mark.skipif(os.environ.get('PROCWISE_TEST_LIVE_DB') != '1',
                                reason='set PROCWISE_TEST_LIVE_DB=1 to run against the database')

PACK = {'key': 'sample', 'format': {'kind': 'deck', 'width_in': 13.333, 'height_in': 7.5},
        'colours': {'ink': '#172033'}, 'writing': {'locale': 'en-GB'}}
LAYOUT = {'id': 'imported_abc1234567', 'proposed_name': '4-up cards', 'regions': [],
          'slots': {}, 'slide_refs': [1, 2], 'example_fill': {}, 'example_source': {},
          'problems': []}


@pytest.fixture
def conn():
    from src.db import get_conn          # the repo's own connection helper
    with get_conn() as c:
        yield c


def test_an_importing_pack_is_never_served(conn):
    pack_id = store.insert_importing(conn, pack_key='t_importing', version=1, source_file='a.pptx',
                                     source_sha256='x', slide_count=2, pack=PACK, evidence={},
                                     user='tester')
    assert all(p['pack_id'] != pack_id for p in store.packs(conn))
    store.mark_candidate(conn, pack_id)
    assert any(p['pack_id'] == pack_id for p in store.packs(conn))


def test_versions_increment_per_key(conn):
    assert store.next_version(conn, 't_version') == 1
    store.insert_importing(conn, pack_key='t_version', version=1, source_file='a.pptx',
                           source_sha256='x', slide_count=1, pack=PACK, evidence={}, user='t')
    assert store.next_version(conn, 't_version') == 2


def test_a_reimport_inherits_names_rating_scales_and_rejections(conn):
    # Review Focus 4: a human renamed everything; re-measuring must not lose that.
    pack_id = store.insert_importing(conn, pack_key='t_inherit', version=1, source_file='a.pptx',
                                     source_sha256='x', slide_count=2, pack=PACK, evidence={}, user='t')
    layout_id = store.insert_layout(conn, pack_id=pack_id, layout=LAYOUT)
    store.mark_candidate(conn, pack_id)
    store.rename_layout(conn, layout_id, 'Eight headline recommendations', 'nick')
    store.define_rating_scale(conn, pack_id, 'hml',
                              {'High': {'bg': '#F9E1E1', 'ink': '#B42D2D'}}, 'nick')
    carried = store.inherited(conn, 't_inherit')
    assert carried['names'][LAYOUT['id']] == 'Eight headline recommendations'
    assert 'hml' in carried['rating_scales']


def test_approval_records_who_and_when(conn):
    pack_id = store.insert_importing(conn, pack_key='t_approve', version=1, source_file='a.pptx',
                                     source_sha256='x', slide_count=1, pack=PACK, evidence={}, user='t')
    store.mark_candidate(conn, pack_id)
    store.set_pack_status(conn, pack_id, 'approved', 'nick')
    row = store.pack(conn, pack_id)
    assert row['status'] == 'approved'
    assert row['approved_by'] == 'nick'
    assert row['approved_at'] is not None
```

- [ ] **Step 3: Apply the migration to bp_testdb, then run the test**

```bash
./venv/bin/python -c "
import os; from src.db import get_conn
sql = open('deploy/sql/2026-10-02_atb_style_pack.sql').read()
with get_conn() as c:
    c.cursor().execute(sql)
print('applied')
"
PROCWISE_TEST_LIVE_DB=1 CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/atb/test_store.py -v
```

Expected: FAIL — `store` has no attribute `insert_importing`.

- [ ] **Step 4: Write `store.py`**

Write each function as plain parameterised SQL against `proc.bp_style_pack` and
`proc.bp_page_layout`, following the repo's existing cursor style. The three rules:

1. `packs()` and `layouts()` add `AND status <> 'importing'` unless `include_importing=True`.
2. `insert_importing` writes `status='importing'`; `mark_candidate` flips it. Nothing else may
   write `'candidate'` on insert.
3. `inherited(conn, pack_key)` reads the highest **non-rejected** version of that key and returns
   `names` (only where `name IS NOT NULL`), `rating_scales` (from `tokens->'rating_scales'` where
   the evidence marks them `defined_by: user`), `locale` (where `evidence` marks it human-set) and
   `rejected` layout keys.

- [ ] **Step 5: Run the test.** Expected: 4 passed.

- [ ] **Step 6: Prove the importing gate**

Remove the `status <> 'importing'` clause from `packs()`; confirm
`test_an_importing_pack_is_never_served` fails. Restore.

- [ ] **Step 7: Apply the migration to bp_sqldb as well, and record it**

```bash
# bp_sqldb is the second database; the governance tables there have been behind before.
PROCWISE_DB_HOST=10.100.10.180 ./venv/bin/python -c "..."   # same apply as Step 3
```

- [ ] **Step 8: Commit**

```
feat(atb): store a pack and its layouts, candidates until approved

Two bp_ tables with the house index naming. The write order is importing → layouts →
candidate, because get_conn() is AUTOCOMMIT and a rollback is a no-op: a crash has to
leave something no read path serves rather than something half-valid.

A re-import inherits what a human decided — layout names, hand-defined rating scales,
the locale, and rejections — because re-measuring everything would throw away the one
part nobody can automate. Applied to both databases.
```

---

### Task 11: The entry point

**Files:**
- Create: `src/services/atb/pptx_import/import_pack.py`
- Create: `tests/services/atb/test_import_pack.py`

**Interfaces:**
- Consumes: Tasks 1–10.
- Produces:
  - `ImportResult(pack_id: str, pack_key: str, version: int, pack: dict, layouts: list[dict], single_use: list[dict], diff: dict, problems: list[dict])`
  - `import_pack(data: bytes, filename: str, user: str, conn=None) -> ImportResult`
  - `class ImportRefused(Exception)`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from src.services.atb.pptx_import.import_pack import ImportRefused, import_pack


def test_emits_the_reused_layouts_and_lists_the_single_use_ones(build_deck, monkeypatch):
    reused = [(0.5, 0.35, 12.33, 0.6, 'Reused title', 30, '#172033'),
              (0.5, 1.5, 6.0, 1.0, 'reused body', 11.5, '#56627A')]
    oneoff = [(0.5, 0.35, 12.33, 0.6, 'One-off', 30, '#172033'),
              (0.5, 1.5, 2.0, 1.0, 'a', 11.5, '#56627A'),
              (3.0, 1.5, 2.0, 1.0, 'b', 11.5, '#56627A'),
              (5.5, 1.5, 2.0, 1.0, 'c', 11.5, '#56627A')]
    data = build_deck([reused] * 11 + [oneoff])
    result = import_pack(data, 'Sample.pptx', 'tester', conn=None)
    assert [l['proposed_name'] for l in result.layouts]
    assert all(len(l['slide_refs']) > 1 for l in result.layouts), 'only reused structures are emitted'
    assert result.single_use, '§6a: the one-offs are listed, not dropped'
    assert result.single_use[0]['slides'] == [12]


def test_refuses_an_unreadable_file():
    with pytest.raises(ImportRefused) as e:
        import_pack(b'nope', 'x.pptx', 'tester', conn=None)
    assert 'could not be opened' in str(e.value)


def test_the_same_file_twice_gives_identical_output(build_deck):
    data = build_deck([[(0.5, 0.35, 12.33, 0.6, 'T', 30, '#172033'),
                        (0.5, 1.5, 6.0, 1.0, 'b', 11.5, '#56627A')]] * 12)
    a = import_pack(data, 'Sample.pptx', 'tester', conn=None)
    b = import_pack(data, 'Sample.pptx', 'tester', conn=None)
    assert a.pack == b.pack
    assert a.layouts == b.layouts
```

- [ ] **Step 2: Run it and watch it fail.**

- [ ] **Step 3: Write `import_pack.py`**

```python
"""One call: bytes in, a stored candidate pack and its layouts out.

§6a: only clusters used by more than one slide become layouts. The single-use structures are
LISTED with their slide numbers — a quadrant is a page someone arranged, not a template, and
emitting 26 single-use layouts would fill the picker with near-identical skeletons. Step 2
imports them as composed pages, and this list is its worklist.
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


def _key_for(filename: str) -> str:
    stem = re.sub(r'\.pptx$', '', filename, flags=re.I)
    return re.sub(r'[^a-z0-9]+', '-', stem.lower()).strip('-') or 'imported-pack'


def import_pack(data: bytes, filename: str, user: str, conn=None) -> ImportResult:
    try:
        deck = read_deck(data)
    except DeckUnreadable as exc:
        raise ImportRefused(str(exc)) from exc

    key = _key_for(filename)
    ev = Evidence()
    try:
        pack = build_pack(deck, ev, key=key, name=filename)
    except PackInvalid as exc:
        raise ImportRefused(f'the derived pack is not valid: {exc}') from exc

    carried = store.inherited(conn, key) if conn is not None else {
        'names': {}, 'rating_scales': {}, 'locale': None, 'rejected': []}
    if carried.get('rating_scales'):
        pack['rating_scales'].update(carried['rating_scales'])
    if carried.get('locale'):
        pack['writing']['locale'] = carried['locale']

    layouts: list[dict] = []
    single_use: list[dict] = []
    problems: list[dict] = []
    for cluster in group(deck):
        if not cluster.reused:
            single_use.append({'slides': list(cluster.slides), 'structure': proposed_name(cluster)})
            continue
        try:
            layout = build_layout(cluster, deck, pack, ev, filename)
        except PackInvalid as exc:
            problems.append({'kind': 'layout_invalid', 'why': str(exc),
                             'slides': list(cluster.slides)})
            continue
        if layout['id'] in carried.get('rejected', []):
            continue
        if layout['id'] in carried.get('names', {}):
            layout['name'] = carried['names'][layout['id']]
        problems.extend(layout['problems'])
        layouts.append(layout)

    result = ImportResult(pack_key=key, version=1, pack=pack, layouts=layouts,
                          single_use=single_use, evidence=ev.as_dict(), problems=problems)
    if conn is None:
        return result

    result.version = store.next_version(conn, key)
    result.pack_id = store.insert_importing(
        conn, pack_key=key, version=result.version, source_file=filename,
        source_sha256=hashlib.sha256(data).hexdigest(), slide_count=len(deck.slides),
        pack=pack, evidence=result.evidence, user=user)
    for layout in layouts:
        store.insert_layout(conn, pack_id=result.pack_id, layout=layout)
    store.mark_candidate(conn, result.pack_id)
    return result
```

- [ ] **Step 4: Run the tests.** Expected: 3 passed.

- [ ] **Step 5: Commit**

```
feat(atb): one call from .pptx bytes to a stored candidate pack

Only structures used by more than one slide become layouts. The single-use ones are
listed with their slide numbers, which is §6a's ruling and step 2's worklist: a quadrant
is a page someone arranged, not a template.

A refusal stores nothing and says why. The same file twice gives identical output, which
is an acceptance criterion rather than a nicety.
```

---

### Task 12: The API

**Files:**
- Create: `src/api/routers/atb.py`
- Modify: `src/api/main.py` (register the router)
- Create: `deploy/sql/2026-10-02_atb_approve_policy.sql`
- Create: `tests/api/test_atb_router.py`

**Interfaces:**
- Consumes: Task 11, Task 10.
- Produces the routes in spec §4, every write taking `require_user`.

- [ ] **Step 1: Write the failing test**

```python
from fastapi.testclient import TestClient


def test_import_refusal_is_a_400_naming_the_reason(atb_client):
    r = atb_client.post('/atb/import', files={'file': ('x.pptx', b'nope', 'application/octet-stream')})
    assert r.status_code == 400
    assert 'could not be opened' in r.json()['detail']


def test_packs_omits_importing(atb_client, importing_pack_id):
    ids = [p['pack_id'] for p in atb_client.get('/atb/packs').json()['packs']]
    assert importing_pack_id not in ids


def test_no_response_field_names_a_route_or_a_table(atb_client, candidate_pack_id):
    body = atb_client.get(f'/atb/packs/{candidate_pack_id}').text
    assert '/atb/' not in body
    assert 'bp_style_pack' not in body
    assert '[withheld]' not in body


def test_approve_requires_a_user(atb_client_no_auth, candidate_pack_id):
    r = atb_client_no_auth.post(f'/atb/packs/{candidate_pack_id}/approve')
    assert r.status_code in (401, 403)


def test_promoting_a_column_to_an_undefined_label_is_refused(atb_client, candidate_pack_id):
    r = atb_client.post(f'/atb/packs/{candidate_pack_id}/rating-scales',
                        json={'name': 'hml', 'chips': {'High': {'bg': '#F9E1E1', 'ink': '#B42D2D'}},
                              'promote': {'layout_key': 'imported_abc1234567', 'column': 'risk',
                                          'labels': ['High', 'Low']}})
    assert r.status_code == 400
    assert 'Low' in r.json()['detail']
```

- [ ] **Step 2: Run it and watch it fail.**

- [ ] **Step 3: Write the router**

Follow `src/api/routers/documents.py` for the upload shape and `require_user` for identity.
Rules: 400 for `ImportRefused` with its message as `detail`; ids only in responses, never route
paths or table names; `POST /atb/packs/{id}/approve` checks the policy row.

- [ ] **Step 4: Write the policy migration**

```sql
-- Approving a pack changes what every future report looks like, so it is governed. A write-class
-- action with no policy row admits any Buyer, which is not the bar for this.
INSERT INTO proc.bp_policy (policy_key, action_class, min_role, note)
VALUES ('atb.pack.approve', 'write', 'Admin',
        'Approving an imported style pack changes the look of every report built from it.')
ON CONFLICT (policy_key) DO NOTHING;
```

(Check the real column names in `deploy/sql/` before running — the insert must match the live
`bp_policy` shape.)

- [ ] **Step 5: Run the tests.** Expected: 5 passed.

- [ ] **Step 6: Commit**

```
feat(atb): import, read and approve over the API

Every write takes require_user, and approving a pack has its own bp_policy row — a
write-class action with no policy row admits any Buyer, and approval changes the look of
every report built from that pack.

Responses carry ids and never a route path or a table name: the output-safety layer
replaces such fields with [withheld], which has broken payloads here before.
```

---

### Task 13: Run it on the real deck

**Files:**
- Create: `scripts/atb_import_report.py`
- Create: `tests/services/atb/test_reference_pack.py`

**Interfaces:**
- Consumes: Task 11.
- Produces: `scripts/atb_import_report.py <file.pptx>` printing the pack, the layout count, the
  single-use list, the problems and the diff against the hand-authored pack.

- [ ] **Step 1: Write the known-answer test**

```python
import os

import pytest

from src.services.atb.pptx_import.import_pack import import_pack

REF = os.environ.get('ATB_REFERENCE_PACK',
                     os.path.expanduser('~/Downloads/Infrastructure-Procurement-Strategy-Pack.pptx'))

pytestmark = pytest.mark.skipif(
    not os.path.exists(REF),
    reason='set ATB_REFERENCE_PACK to the Infrastructure Procurement Strategy Pack to run this')


@pytest.fixture(scope='module')
def result():
    with open(REF, 'rb') as fh:
        return import_pack(fh.read(), os.path.basename(REF), 'test', conn=None)


def test_rediscovers_the_hand_authored_palette(result):
    colours = result.pack['colours']
    assert colours['ink'] == '#172033'
    assert colours['muted'] == '#56627A'
    assert colours['panel'] == '#F3F5F8'
    assert colours['accent'] in ('#2350C8', '#0F6E78')
    assert colours['accent_2'] in ('#2350C8', '#0F6E78')
    assert colours['accent'] != colours['accent_2']


def test_rediscovers_the_type_scale_and_the_fonts(result):
    scale = result.pack['type_scale_pt']
    assert scale['title'] == 30.0
    assert scale['body'] in (11.5, 12.0)
    assert scale['footer'] == 9.0
    assert result.pack['fonts']['heading']['family'] == 'Cambria'
    assert result.pack['fonts']['body']['family'] == 'Calibri'


def test_rediscovers_the_grid_and_corrects_my_body_top(result):
    grid = result.pack['grid']
    assert grid['margin_in'] == 0.5
    assert grid['title_top_in'] == 0.35
    assert grid['footer_top_in'] == 7.02
    # the hand-authored pack says 1.65; the file says 1.5, in 99 shapes
    assert grid['body_top_in'] == 1.5
    assert result.pack['chapter_chip_in'] == 0.32


def test_the_format_is_the_deck_size(result):
    assert result.pack['format'] == {'kind': 'deck', 'width_in': 13.333, 'height_in': 7.5}


def test_the_declared_language_is_contested(result):
    assert result.pack['writing']['locale'] == 'en-US'
    assert result.pack['writing']['locale_contested'] is True
    assert result.pack['writing']['locale_suggested'] == 'en-GB'


def test_emits_eight_reusable_layouts_and_lists_twenty_six(result):
    assert len(result.layouts) == 8, [l['proposed_name'] for l in result.layouts]
    assert sum(len(l['slide_refs']) for l in result.layouts) == 59
    assert len(result.single_use) == 26


def test_the_series_palette_is_a_superset_of_the_hand_authored_one(result):
    hand = {'#172033', '#0F6E78', '#B42D2D', '#9A5B00', '#2350C8', '#1F7A5A'}
    assert hand <= set(result.pack['series_palette'])


def test_every_emitted_layout_passes_the_contract(result):
    from src.services.atb.pptx_import.contract import validate_layout
    for layout in result.layouts:
        assert validate_layout(layout) == [], layout['id']
```

- [ ] **Step 2: Run it**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/atb/test_reference_pack.py -v
```

Expected: every assertion passes, or a **named** difference. A difference is a finding to
explain, not a number to widen: either the measurement is wrong, or the hand-authored pack was.

- [ ] **Step 3: Write the report script**

`scripts/atb_import_report.py` — reads a `.pptx`, runs `import_pack(conn=None)`, prints: the
pack's tokens, the grid, the layout table (proposed name, slides, problems), the single-use list,
the evidence's incidentals and ignored values, and the diff against
`beyond_procwise_ui/src/modules/SpendIQ/atb/styles/consulting-navy-16x9.json` when that file is
reachable.

- [ ] **Step 4: Run the whole suite and the script**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/atb -v
./venv/bin/python scripts/atb_import_report.py ~/Downloads/Infrastructure-Procurement-Strategy-Pack.pptx
```

- [ ] **Step 5: Commit**

```
test(atb): the importer rediscovers the pack I typed by hand

Run on the reference deck it must find ink #172033, muted #56627A, panel #F3F5F8, the
two accents, Cambria on Calibri, 30pt titles, 9pt footnotes, 0.5in margins, the 7.02in
footnote line and the 0.32in chapter chip — and it must correct me on body_top, which I
authored at 1.65in and the file puts at 1.5in in 99 shapes.

Eight reusable layouts covering 59 slides, 26 single-use structures listed. The declared
language comes back en-US with the spelling contradiction flagged.

The deck is a client document and is not in the repo: the test reads ATB_REFERENCE_PACK
and skips with that variable named when it is absent.
```

---

## Self-review

**Spec coverage.** §2 → Task 13 (the known-answer test is the spec's §2 numbers). §4 architecture
→ Tasks 1–12 (module per measurement, two tables, the endpoint table). §5 palette/type/grid →
Tasks 2, 3, 4. §5a the five values → Task 5 (series, scales, locale) and Task 3 (fallbacks) and
Task 9 (`proposed_name` is §5a's "nothing names a layout"). §5b inheritance → Task 10. §6 and §6a
→ Tasks 7 and 11. §7 artefacts and `box_in` → Tasks 6, 8, 9 (`validate_layout` enforces exactly
one of `grid`/`box_in`). §8 review and approval → Task 12 serves it; the screen is the next plan.
§9 refusals 1–10 → Task 2 (1), Task 9 (2), Task 8 (3), Tasks 3 and 6 (4), Task 9 (5), Task 10
(6), Tasks 1, 6, 11 (7), Task 8 (8), Task 5 (9), Task 12 (10). §10 acceptance 1–7 → Task 13 (1,
6), Task 6 (2), Task 11 (4), every task's break step (5), Task 10 (7). **Gap found and closed:**
§10.3 — "every imported layout renders its example fill with no region off the sheet" — is a UI
test (`pageFit.contract.test.js`) and belongs to the screen plan; noted there rather than left
implied here.

**Placeholders.** None: every code step carries real code. Task 10 Step 4 and Task 12 Step 3
describe SQL and router bodies in rules rather than full listings, because both must be written
against this repo's existing cursor and auth style, which the implementer has in front of them —
the rules are exact (three for the store, three for the router) and the tests pin the behaviour.

**Type consistency.** `Deck`, `Shape`, `Box`, `Run` are defined once in Task 1 and used
unchanged. `Evidence.record/incidental/ignored/as_dict` likewise. `colours()`, `type_scale()`,
`fonts()`, `grid()`, `chapter_chip_in()`, `series_palette()`, `rating_scales()`, `writing()`,
`signature()`, `group()`, `Cluster`, `regions_and_slots()`, `example_fill()`, `build_pack()`,
`build_layout()`, `proposed_name()`, `layout_key()`, `import_pack()`, `ImportResult` — each
appears with the same name and signature in its Produces block and at every call site.

**Review Focus.** All five have a test in the task that owns the code: groups (Task 1), theme-only
colours (Task 2), no charts (Task 5), inheritance across a re-import (Task 10), portrait format
(Task 6).
