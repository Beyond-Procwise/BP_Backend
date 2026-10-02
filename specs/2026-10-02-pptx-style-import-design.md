# Design — reading a style pack and a layout registry out of a PowerPoint

**Status:** awaiting review (written 2026-10-02).
**Repo:** BP_Backend (the importer and its tables), with one small schema addition consumed by
`beyond_procwise_ui`.
**Approved in conversation:** approach A ("derive it"), step 1 of three, 2026-10-02.
**Reference file:** `/home/muthu/Downloads/Infrastructure-Procurement-Strategy-Pack.pptx`
(85 slides, 603 KB, placed on this machine 2026-10-02 07:20).

## 1. Why this exists

Nick's ruling of **2026-09-25**: *"A report builder that looks at existing reports, either
previously generated or uploaded to the product from a power point so it can read the style."*
That has never been built. Nothing in any of the three repos reads a `.pptx` for anything but
its text (`src/api/routers/documents.py:165`, which concatenates every shape's words for
retrieval).

What exists instead is two style packs and three page layouts in
`beyond_procwise_ui/src/modules/SpendIQ/atb/`, authored by hand from a build brief's own list of
values. On 2026-10-02 Nick named the consequence in three parts:

1. the output is general and does not look like a board paper;
2. you cannot drag graphs and components into a page;
3. you cannot create packs like his reference deck.

This spec covers **(3)**, which is step 1 of three; (2) is step 2 and gets its own spec; (1) is
the result of the other two rather than a task of its own.

## 2. What the reference pack actually contains

Everything in this section was measured from the file, not read from the brief. The numbers are
here because the acceptance test in §10 is "rediscover them".

**The file.** 85 slides, 13.333 × 7.5 in. One slide master with **one** layout, named
`DEFAULT`, used by all 85 slides. So the deck carries no reusable layouts of its own: every page
is composed of absolutely positioned shapes.

**Shapes.** 1,101 text boxes · 566 auto-shapes · 84 placeholders · 56 tables · **8 charts.**
Shapes per slide run from 3 to 61. The deck is drawn, not laid out.

**The theme is a decoy.** It is stock Office — `dk1 #44546A`, `accent1 #4472C4`, major font
Calibri Light, minor font Calibri. **None of that is what the deck looks like.** An importer that
read the theme would produce a pack with the wrong palette and the wrong heading font.

**The real palette**, by use (fills / text runs / lines):

| colour | fills | text runs | lines | reading |
|---|---|---|---|---|
| `#172033` | 619 | 804 | 563 | ink |
| `#56627A` | 252 | 254 | 252 | muted |
| `#F3F5F8` | 215 | — | 215 | panel |
| `#0F6E78` | 114 | 94 | 114 | accent (teal) |
| `#8A94A8` | 84 | — | — | muted, lighter |
| `#FFFFFF` | 82 | 77 | 140 | paper / reversed ink |
| `#2350C8` | 76 | 45 | 67 | accent (blue) |
| `#E4EBFB` | 40 | — | 40 | panel, blue |
| `#E0F2F3` | 39 | — | — | panel, teal |
| `#9A5B00` | 33 | 25 | — | caution |
| `#FCEFD9` | 24 | — | — | panel, amber |
| `#D5DBE5` | — | — | 61 | rule |
| `#2E62B8` | 17 | — | — | accent, second blue |
| `#5B3F99` | 17 | 9 | — | violet |
| `#A33A4C` / `#B42D2D` | 15 | 11 | — | alert |
| `#1F7A5A` | — | 12 | — | positive |
| `#C9D2E3` | — | 7 | — | faint ink |

**Type scale**, by run count: 30pt (95) · 14pt (154) · 13pt (118) · 12pt (154) · 11.5pt (223) ·
11pt (128) · 10.5pt (114) · 10pt (189) · 9pt (32), with 15/16/18/20/22/34pt in single figures.
**Fonts from the runs:** Calibri 1,218 · Cambria 128.

**The grid**, from repeated edges: left `0.5in` (397 shapes) · content width `12.33in` (133) ·
title band top `0.35in` (83) · subtitle `1.0in` (77) · **body top `1.5in` (99)** · footnote line
`7.02in`, page-number placeholder `7.05in` (84) · a `0.32 × 0.32in` square at `0.5, 0.47`
(66 squares, 61 at that exact origin) which is the chapter marker.

**The grid is not strictly twelve columns.** Band widths in use: 12.33 (full) · 6.53 (2-up) ·
3.71 and 3.6 (3-up) · 2.9 and 2.86 (4-up) · 2.54, 2.5, 2.45 (5-up) · 1.77 · 1.48. Each band
divides the content width its own way; several do not land on a 12-column span.

**Page structures.** Grouping each slide's body shapes into rows and each row by column count
and content kind gives **34 raw signatures**:

- **21 slides** — one full-width table (4, 5, 6, 7, 14, 19, 21, 24, 27, 69, 71, 76, …)
- **10 slides** — `8 | 3 | 4 | 3 | 3 | 3 | 4` (35, 38, 41, 44, 47, 50, 53, 56, 59, 62)
- **10 slides** — `6 | 3×TABLE | 1` (36, 39, 42, 45, 48, 51, 54, 57, 60, 63)
- **9 slides** — `2×TABLE | 8 | 6 | 6` (37, 40, 43, 46, 49, 52, 55, 58, 64)
- **9 slides** across four signatures — chart pages (8, 10, 11, 25, 28, 29, 32, 66)
- **2 slides** — a 4-up card grid repeated down the page (3, 16)
- **26 slides** — one-offs: the spend-against-supply-risk quadrant with a side rail (9), the
  market calendar band (15), the eight-forces grid (12), the price-outlook strip (13), …

The three 10/10/9 groups are slides 35–64 in a repeating cycle of three — one category per
cycle. That is what the brief's `repeat_over` was for.

## 3. Scope

**In:** read one `.pptx`; emit one style pack, N candidate layouts and an evidence record; store
them; serve them; let a human rename, approve and reject them; hold the output to the validators
the hand-authored packs already pass.

**Out, deliberately:** regions as drop targets (step 2); writing `.pptx` (the RGA already does);
applying an imported pack to reports that already exist; the agent writing content into imported
layouts (the fill layer, Phase 2); any scope chain (see §4).

## 4. Architecture

```
POST /atb/import  (a .pptx)
   │
   ├─ src/services/atb/pptx_import/read.py      open, walk shapes (groups flattened), EMU → in
   ├─                              palette.py   colour census → colour tokens + incidentals
   ├─                              typescale.py size census → named type roles
   ├─                              grid.py      edge census → margin, bands, title/body/footer
   ├─                              cluster.py   slide signature → groups → merged layouts
   ├─                              slots.py     region → slot type, columns, limits
   ├─                              evidence.py  every value ← (slide, shape, count)
   ├─                              emit.py      pack JSON + layout JSON (schema §7)
   └─                              store.py     proc.bp_style_pack, proc.bp_page_layout
```

One entry point, `import_pack(file_bytes, filename, user) -> ImportResult`.

**Why a new module and not `src/services/rga/style.py`.** That module is the output of the
settings resolver: platform defaults, honestly unscoped, and its own header forbids growing into
something that *looks* like scope resolution. An imported pack is a different kind of object — a
measurement of a document, with provenance. The new module **feeds** `style.py`'s palette and
font keys; it does not replace them. No scope chain is introduced here either: there is still no
tenant dimension in this platform (`src/api/auth.py`, `x-customer-id` is the constant `"001"`).

**Two tables** (`bp_` prefix, indexes `ix_bp_*`, per house convention), migration
`deploy/sql/2026-10-02_atb_style_pack.sql` with a rollback beside it:

`proc.bp_style_pack`
: `pack_id` (pk) · `pack_key` · `version` · `source_file` · `source_sha256` · `slide_count` ·
  `format` jsonb · `tokens` jsonb · `evidence` jsonb · `status` · `created_at` · `created_by` ·
  `approved_at` · `approved_by` · `notes`. Unique `(pack_key, version)`.

`proc.bp_page_layout`
: `layout_id` (pk) · `pack_id` (fk) · `layout_key` · `proposed_name` · `name` · `slide_refs`
  int[] · `regions` jsonb · `slots` jsonb · `example_fill` jsonb · `example_source` jsonb ·
  `unresolved` jsonb · `status` · `created_at` · `approved_at` · `approved_by`.
  Unique `(pack_id, layout_key)`.

**`get_conn()` is AUTOCOMMIT, so a rollback is a no-op** (that is recorded in this repo's
memory and has bitten before). An import therefore cannot be one transaction. The sequence is:
insert the pack with `status='importing'` → insert every layout → flip the pack to `'candidate'`.
A crash leaves an `importing` pack, which **no read path ever serves**, and a re-import of the
same file supersedes it. Half-imported state is invisible rather than wrong.

**Endpoints** (`src/api/routers/atb.py`, every write taking `require_user`):

| route | does |
|---|---|
| `POST /atb/import` | the import; returns pack id, layout count, token diff against any previous version |
| `GET /atb/packs` | packs, newest first, `importing` excluded |
| `GET /atb/packs/{pack_id}` | one pack with its tokens and layouts |
| `GET /atb/packs/{pack_id}/evidence` | where every value came from |
| `GET /atb/layouts?status=approved` | what the builder's pickers read |
| `POST /atb/layouts/{layout_id}` | rename |
| `POST /atb/layouts/{layout_id}/approve` · `/reject` | the gate |
| `POST /atb/packs/{pack_id}/approve` | the gate |

Responses carry **ids, never route paths or table names** — the output-safety layer replaces any
field that names an internal route or table with `[withheld]`, which has silently broken
payloads here before.

**The UI side** is small, and this spec only states the contract: `atb/bridge.js` keeps the three
bundled layouts as a fallback and prefers what `GET /atb/layouts?status=approved` and
`GET /atb/packs` serve, through the existing `__SPENDIQ_API_AI__` bridge. `renderLayout`,
`validateLayout`, `validateStyle` and the calibrator are unchanged: they do not care whether a
layout came from a file in the repo or from an upload.

## 5. Measuring the pack

**Palette.** Count every colour three ways — solid fill, text run, line — walking into groups.
Resolve a theme reference to its RGB and record that it was a theme reference. Then assign roles
by use, in this order, each role taken by the highest-count colour not already assigned:

- `ink` — most-used text colour;
- `muted` — second most-used text colour;
- `rule` — highest-count colour that appears **only** on lines;
- `paper` — the fill of the slide background, else white;
- `panel`, `panel_*` — fills whose relative luminance is above 0.85, in count order (`panel` is
  the most used; the rest keep a suffix derived from their hue: `panel_blue`, `panel_teal`,
  `panel_amber`, `panel_violet`);
- `accent`, `accent_2` — saturated colours (HSL saturation > 0.3, luminance 0.2–0.7) by count;
- `alert_ink`/`alert_bg`, `caution`, `positive` — saturated colours whose hue falls in the red,
  amber and green bands, when present.

**A colour used fewer than 10 times is incidental**: listed in the evidence, not made a token.
The floor is a stated constant, not a feeling, and the evidence shows what it excluded.

**Type scale.** Histogram the run sizes. The largest size that appears in the title band
(`y < 1.0in`) on more than five slides is `title`. The rest are named by rank against the roles a
pack must define (`subtitle`, `card_title`, `body`, `table`, `small`, `footer`), with sizes
differing by less than 0.6pt merged. Sizes in single figures across 85 slides are reported, not
named.

**Fonts.** The most-used run font is `body`; the most-used font in the title band is `heading`.
The theme's `majorFont`/`minorFont` are recorded as evidence and **not** used — in this file they
are wrong.

**Grid.** Histogram left edges, widths and top edges, quantised to 0.01in.
`margin_in` = modal left edge. Content width = modal full width (sanity check:
`page_w − 2 × margin`, and the measured 12.33 matches 13.333 − 1.0). `title_top_in`,
`body_top_in`, `footer_top_in` = modal top edges inside the top, middle and bottom thirds.
`chapter_chip` = modal size of squares under 0.6in that sit in the title band.
**`cols` stays 12** as the authoring convenience, and `gutter_in` is the modal gap between
adjacent same-row shapes. Any band that does not land on a column span keeps its measured inches
(§7).

### 5a. The five values a `.pptx` does not state honestly

`validateStyle` requires four things the file does not simply contain, and one it states wrongly.
Each gets a measured source and, where the importer has to invent, says so.

**Font fallback stacks.** A `.pptx` names a family and nothing else, and `validateStyle` requires
a fallback because the named family may not be installed. The importer invents the stack from the
family: a known serif (Cambria, Georgia, Times, Garamond, Book Antiqua, Palatino) gets a serif
stack, anything else a sans stack. Marked `invented: true` in the evidence.

**`series_palette`** must be a non-empty array. It comes from the chart parts, in first-use
order. The reference file has 8 charts and yields `#0F6E78 #B42D2D #172033 #2350C8 #9A5B00
#1F7A5A #9AA6BC #2F7FA8 #3A7A5C #6A4FA0` — which contains all six colours of the hand-authored
pack's series palette, in a different order. With no charts in the file, the saturated accents in
count order stand in. It is never empty.

**`rating_scales`.** Derived from table columns whose cells repeat a small vocabulary **and**
carry distinct fills: the vocabulary becomes the labels, the fills and inks become the chips.
On the reference file this is expected to yield little or nothing — the three scales in the
hand-authored pack (`hml`, `influence`, `confidence`) came from the brief, not from the file.
**A rating column the pack cannot back with a scale is emitted as a `text` column**, with the
reason recorded. The alternative is a pack and a layout that are individually valid and illegal
together, which `validateLayoutAgainstStyle` would then reject at render time.

**`writing.locale`.** Taken from the runs' declared language — and then checked. The reference
file declares `en-US` on all 6,940 runs while spelling *Virtualisation*, *Mobilise*, *Optimise*,
*utilisation*, with 8 `-ize/-ization` uses in the whole deck. So the declared language is wrong
and the hand-authored `en-GB` is right. The importer records the declared value, runs a spelling
test (`-ise/-isation/-our` against `-ize/-ization/-or`), and **flags the contradiction for the
human** rather than quietly picking either. This is the theme lesson again: what the file says
about itself is evidence, not the answer.

**Names.** Nothing in the file names a layout. §9.2.

### 5b. A re-import inherits what a human decided

Measured values are re-measured on every import. Values a human supplied or corrected —  layout
names, `writing.locale`, rating scales added on the review screen, rejections — **carry forward
from the previous version of the same `pack_key`**, unless the new file contradicts a measured
value, in which case the measurement wins and the difference is reported in the version diff.
Without this rule every re-import of a revised deck would throw away the naming work, which is
the one part nobody can automate.

## 6. Deciding how many layouts there are

**Signature.** For each slide: drop the title band (`y < 1.25in`) and the footer
(`y > 6.9in`); drop decorations under 0.5 × 0.5in; group the rest into rows by top edge within
0.45in; describe each row as `<column count>×<TABLE|CHART|SHAPE>`. Join with `|`.

**Merges, in order:**

1. widths within 0.1in of each other count as the same column count;
2. two signatures that differ only in how many times the **last** row repeats become one layout
   carrying `repeat_over` on that row;
3. a row holding a table never merges with a row holding a chart, and neither merges with a row
   of plain shapes.

Nothing else merges. The importer **reports the resulting count** — on the reference file the raw
count is 34 and the merged count is whatever these rules give; this spec does not promise 21,
and a number invented in advance would be a target to overfit to.

**Regions.** Each layout's regions are the median box of its members' rows, in inches. A member
whose box differs from the median by more than 0.15in in any dimension is reported as a
**loose fit** on that layout, with its slide number, so a cluster that should have been two is
visible rather than averaged.

**Slot typing**, per region, from what every member puts there:

- a table on every member → `table` slot, columns from the header row, `max_rows` from the
  largest member;
- a chart on every member → `chart` slot, `fill: "bind"`;
- text in the title band → `text`, `style: "assertion"`;
- N equal-width boxes each holding a short run over a longer run → `list`, item
  `{heading, body}`, `min`/`max` from the members;
- a single prose box → `text`;
- a box at the footnote line → `sources`;
- **members disagree → `unresolved`**, carrying both readings and the slides that gave them. The
  layout still imports; the region is marked and the review screen shows it. This is the same
  ruling the document-type resolver took when two types tied: an honest unresolved beats a
  coin-flip.

**Length limits** come from the measured box and the pack's type scale — i.e.
`scripts/atb_calibrate_limits.mjs`, which has had no real input until now. The importer emits
geometric ceilings; the editorial caps stay the human's.

## 7. The artefacts

**The pack** is the existing style-pack shape, so `validateStyle` accepts it unchanged:
`key`, `name`, `format {kind, width_in, height_in}`, `colours`, `type_scale_pt`, `fonts`,
`grid`, `series_palette`, `category_palette`, `chapter_colours`, `rating_scales`, `locale`.
`kind` is `deck` when the page is wider than tall, else `a4-portrait`.
`category_palette` and `chapter_colours` are optional to the validator and are emitted only when
derivable — chapter colours from the fills of the title-band chip where it varies between
sections, the category palette from the fills behind recurring category labels. Omitted
otherwise, because a component that names a missing chapter colour already falls back to the
accent.

**One addition to the layout schema.** A region may carry `grid` (12-column span, as today) **or**
`box_in: {x, y, w, h}` (measured inches). Exactly one of the two. `regionBox()` already returns
inches and already treats a numeric row as an absolute inch offset, so this is a small branch in
one function plus a validator rule, and every hand-authored layout stays valid.

**The evidence record**, stored on the pack:

```json
{ "colours": { "ink": { "value": "#172033", "fills": 619, "runs": 804, "lines": 563,
                        "first_seen": {"slide": 1, "shape": "TextBox 12"} } },
  "type_scale_pt": { "title": { "value": 30, "runs": 95, "slides": [1, 2, 3, "…"] } },
  "grid": { "margin_in": { "value": 0.5, "shapes": 397 } },
  "incidental_colours": [ { "value": "#2E62B8", "count": 17, "why": "below the 10-use floor as an accent, outranked by #2350C8" } ],
  "theme_ignored": { "majorFont": "Calibri Light", "minorFont": "Calibri", "accent1": "#4472C4",
                     "why": "the theme is stock Office; the deck's own shapes disagree with it" } }
```

**The example fill.** Each layout keeps one fill taken from its first member slide, with
`example_source: {file, slide}`. The review screen and the composer both label it as what it is
— *"as it appeared in Infrastructure-Procurement-Strategy-Pack, slide 9"*. It exists because an
empty fill renders a blank page, which is indistinguishable from a broken renderer; it is never
presented as the user's own content, and the words in it belong to whoever wrote the deck.

## 8. Review and approval

A third screen in the report builder, beside the reports index and the composer: **Papers**.
It lists imported packs newest first with source filename, slide count and date. Opening one
shows its tokens as swatches, its evidence behind a disclosure, and each candidate layout drawn
**by `renderLayout` itself** with its example fill — so the thumbnail you approve cannot differ
from the page the report prints. Per layout: rename, approve, reject. Loose fits and unresolved
regions are marked on the card.

It reuses the builder's existing panel, swatch and popover styling. No new visual language.

**The gate.** Only `approved` layouts reach a page's Layout picker; only `approved` packs reach
its Paper picker. An unapproved import is fully visible and completely inert. Approving changes
what every future report looks like, so it gets **its own `bp_policy` row** rather than relying on
the default for an ungoverned `write` class, which admits any Buyer.

**Versions.** Packs are keyed by source file and versioned. Re-importing the same filename makes
a new version instead of overwriting, and `POST /atb/import` returns the token and layout diff
against the previous version, so a revised deck shows what moved. A report records its pack **by
key**, which is already how the report builder saves, so a new version does not rewrite saved
reports.

## 9. What it refuses

1. A colour under the 10-use floor → **incidental**, in the evidence, not a token.
2. Naming a layout → a geometric `proposed_name` only; the name is the human's.
3. A region whose members disagree → `unresolved` with both readings.
4. Reading the theme as the answer → theme values are recorded under `theme_ignored`.
5. Importing content as content → one labelled example fill per layout.
6. Approving anything → everything lands `candidate`.
7. A file it cannot parse, with no slides, or whose derived pack fails `validateStyle` → the
   import **fails with the reason named** and writes no `candidate` pack. There is no
   half-imported pack, because the `importing` status is never served.
8. Guessing a slot type from one member when the layout has many.
9. A rating column it cannot back with a derived scale → emitted as a `text` column, with the
   reason recorded, rather than a pack and layout that are illegal together.

## 10. Acceptance criteria

1. **Known answer.** Run on the reference file, the importer rediscovers: format
   `deck 13.333 × 7.5`; `ink #172033`, `muted #56627A`, `panel #F3F5F8`, `accent #2350C8`,
   `accent_2 #0F6E78`, `panel_blue #E4EBFB`, `panel_teal #E0F2F3`, `panel_amber #FCEFD9`,
   `rule #D5DBE5`; fonts Cambria (heading) and Calibri (body); `title 30pt`, `body 11.5pt`,
   `table 10pt`, `footer 9pt`; `margin_in 0.5`, content width `12.33`, `title_top_in 0.35`,
   `footer_top_in 7.02`, chapter chip `0.32in`. Every difference from the hand-authored
   `consulting-navy-16x9.json` is either absent or **a correction with evidence** — `body_top_in`
   is expected to come back `1.5`, not the authored `1.65`. Three values are expected **not** to
   match, and the test asserts the disagreement rather than the value: `series_palette` comes back
   as a superset in a different order (compared as a set, order reported); `writing.locale` comes
   back `en-US` **with the spelling contradiction flagged**; and `rating_scales` comes back thin or
   empty, inherited from the previous version under §5b rather than invented.
2. **Its output passes the UI's own validators** — `validateStyle` for the pack,
   `validateLayout` for every layout — run in CI beside the existing reference-data checks.
3. **Every imported layout renders its example fill** with no unresolved figure, no region off
   the sheet, into the footer, or overlapping another: `pageFit.contract.test.js` pointed at
   imported layouts.
4. **Deterministic.** The same file imported twice produces byte-identical pack and layout JSON.
   Fixture tests for each merge rule: near-equal widths merge, a repeat-count difference merges,
   a table row and a chart row never merge.
5. **Every guard broken on purpose and watched go red** before it is trusted.
6. **Demonstrated on the real file end to end** — not only on fixtures — reporting the layout
   count, the loose fits, the unresolved regions and the token diff.
7. **An `importing` pack is never served**, proven by a test that leaves one behind.

## 11. To confirm before building

- **The 10-use floor** and the 0.15in loose-fit tolerance are my choices from this one file. They
  are constants in one place and the evidence reports what they excluded.
- **Role assignment for a second pack.** The rules in §5 are written against a deck whose palette
  is unambiguous. A pack with two equally-used accents will assign them in count order, which may
  not be the designer's intent; the review screen is where that gets corrected, and a later
  version can let you drag a swatch between roles.
- **Charts.** Only 8 of 85 slides hold a real chart. A `chart` slot is emitted with
  `fill: "bind"` and no binding — binding a chart to live data is step 2's business.
- **Rating scales are probably not in the file.** If the review screen should let you define a
  scale by hand (pick the labels, pick the swatches), say so and I will scope it; otherwise a
  rating column stays text until a later version.

## 12. What this design does not do

No scope chain. No PowerPoint writing. No automatic application of a pack to existing reports.
No agent-written content. No change to `renderLayout`, `validateStyle` or the calibrator beyond
the one `box_in` branch. The three bundled layouts stay as the offline fallback until approved
imported layouts exist.
