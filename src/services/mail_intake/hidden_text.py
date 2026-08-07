# src/services/mail_intake/hidden_text.py
"""Detect text that is present in a document but not visible to a reader.

Hidden text is how a prompt injection reaches an LLM-backed extractor: the
human reviewing the quote sees a price, and the model additionally sees an
instruction. Detection is therefore a Finding in its own right and quarantines
the artifact — someone attempting it has told you something useful about the
counterparty.

**The false-positive trap, which is the hard part of this module.** Scanned
documents carry an invisible OCR text layer, drawn in PDF text render mode 3,
so the page stays searchable while the reader sees the scan. That is exactly
the signal an injection produces. A detector keyed on render mode alone would
quarantine nearly every scanned invoice, and in this corpus that is most of
them — the review queue would be useless within a day and the feature would be
switched off.

The discriminator is position, not mode: an OCR layer lies **over a raster
image covering the same region**. Injected text floats free of any image. So
the rule implemented here is *invisible text not backed by an underlying image
region*, never *invisible text*.

Scope (spec §8.1): invisible render mode, zero or near-zero opacity,
colour-on-background contrast, off-page geometry, sub-threshold font size, and
hidden optional-content layers. True z-order occlusion analysis is out of
scope — lowest yield, highest false-positive risk on documents that use white
boxes for layout.

Hits carry structural facts only. The hidden string itself is never recorded:
findings are persisted and read by humans, and replaying an injected
instruction into the record would defeat the point of catching it.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

TINY_FONT_PT = 1.5
NEAR_ZERO_OPACITY = 0.05
# Luminance above which text is treated as white-ish for the contrast check.
LIGHT_LUMINANCE = 0.90

# Fraction of a text span that must sit over qualifying scan imagery for it to
# read as an OCR layer rather than free-floating injected text.
OCR_OVERLAP_THRESHOLD = 0.60

# Fraction of the PAGE an image must cover before it counts as scan backing.
#
# This exists because overlap alone is defeatable. An earlier version asked only
# whether some image covered the span; an attacker authoring the PDF — precisely
# the threat model here — could place a 10x10 pixel graphic under the injected
# text and the detection vanished entirely. Demonstrated, not theorised.
# Requiring the backing image to cover a substantial share of the page means
# concealment now costs a large visible graphic the reader would notice, rather
# than an invisible speck.
#
# Measured against the union of a page's images, so tile count does not matter:
# one-, two-, three- and five-tile scans all read as ~100%, while a letterhead
# logo reads as ~2%.
#
# CALIBRATION NOTE, stated because the number reads more authoritative than it
# is: none of the 260 real PDFs in this repo carries an invisible OCR layer, so
# there was no corpus data to tune against. 0.50 passes a full-page scan with
# wide margin and rejects the logo-sized case; a hybrid document whose scanned
# region covers less than half a page would be flagged, which is the safer
# direction to err. Revisit once real inbound mail provides examples.
OCR_PAGE_COVERAGE_MIN = 0.50

_WORD_MIME = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
_EXCEL_MIME = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


@dataclass(frozen=True)
class HiddenTextHit:
    technique: str
    page: int
    bbox: Tuple[float, float, float, float]
    char_count: int


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlap_fraction(span_box, image_box) -> float:
    """How much of span_box lies inside image_box, as a fraction of the span."""
    span_area = _area(span_box)
    if span_area <= 0:
        return 0.0
    inter = (
        max(span_box[0], image_box[0]),
        max(span_box[1], image_box[1]),
        min(span_box[2], image_box[2]),
        min(span_box[3], image_box[3]),
    )
    return _area(inter) / span_area


def _luminance(color: Any) -> float:
    """Perceptual luminance of a texttrace colour tuple (already 0..1 floats)."""
    try:
        r, g, b = (float(c) for c in color[:3])
    except Exception:
        return 0.0
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _image_rects(page) -> List[Any]:
    rects = []
    try:
        for image in page.get_images(full=True):
            rects.extend(page.get_image_rects(image[0]))
    except Exception:
        return []
    return rects


def _union_fraction(rects, box, cols: int = 48, rows: int = 24) -> float:
    """How much of `box` is covered by the UNION of `rects`.

    Sampled on a grid rather than computed as an exact rectangle union, which
    is fiddly and buys no accuracy at these scales. Used for both questions
    below, so they cannot drift apart.
    """
    if not rects or _area(box) <= 0:
        return 0.0
    covered = 0
    for i in range(cols):
        x = box[0] + (box[2] - box[0]) * (i + 0.5) / cols
        for j in range(rows):
            y = box[1] + (box[3] - box[1]) * (j + 0.5) / rows
            if any(r[0] <= x <= r[2] and r[1] <= y <= r[3] for r in rects):
                covered += 1
    return covered / float(cols * rows)


def _scan_backings(image_rects, page_box) -> List[Any]:
    """The page's images, but only if together they plausibly ARE a page scan.

    Qualification is on the UNION, not per image. Judging each image on its own
    breaks on tiled scans, which are ordinary for large or multi-column
    originals: measured here, a two-tile scan puts each tile at exactly 50.000%
    of an A4 page — precisely on the threshold, decided by float comparison —
    and a three-tile scan puts each at 33%, so none qualifies and a wholly
    legitimate scanned document gets flagged. On the union all of those read as
    100% and pass, while a letterhead logo reads as 2% and does not.

    A letterhead logo is not scan backing. This filter is what stops a
    speck-sized graphic from laundering injected text.
    """
    rects = [tuple(rect) for rect in image_rects]
    if not rects:
        return []
    if _union_fraction(rects, page_box) < OCR_PAGE_COVERAGE_MIN:
        return []
    return rects


def _backed_by_scan(span_box, backings) -> bool:
    """Is the span covered by the union of qualifying scan imagery?

    Union again, for the same reason: an OCR span straddling a tile seam
    overlaps neither tile by enough on its own, and testing each rect
    independently flagged exactly that legitimate case.
    """
    if not backings or _area(span_box) <= 0:
        return False
    return _union_fraction(backings, span_box) >= OCR_OVERLAP_THRESHOLD


def _hidden_layers(doc) -> set:
    """Names of optional-content groups that are off by default."""
    hidden = set()
    try:
        for xref, config in (doc.get_ocgs() or {}).items():
            if config.get("on") is False:
                name = config.get("name")
                if name:
                    hidden.add(name)
    except Exception:
        return set()
    return hidden


def _scan_pdf(content: bytes) -> Tuple[HiddenTextHit, ...]:
    try:
        import fitz
    except Exception:  # pragma: no cover - dependency is pinned
        return (HiddenTextHit("unreadable", 0, (0, 0, 0, 0), 0),)

    hits: List[HiddenTextHit] = []
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception:
        return (HiddenTextHit("unreadable", 0, (0, 0, 0, 0), 0),)

    with doc:
        if doc.needs_pass:
            # Encryption is the safety gate's business, not ours.
            return ()
        hidden_layer_names = _hidden_layers(doc)
        for index, page in enumerate(doc):
            try:
                spans = page.get_texttrace()
            except Exception:
                hits.append(HiddenTextHit("unreadable", index, (0, 0, 0, 0), 0))
                continue

            page_box = tuple(page.rect)
            backings = _scan_backings(_image_rects(page), page_box)

            for span in spans:
                box = tuple(span.get("bbox") or (0, 0, 0, 0))
                chars = len(span.get("chars") or ())
                if chars == 0:
                    continue
                technique: Optional[str] = None

                if span.get("type") == 3:
                    # Render mode 3. Benign when it is an OCR layer over a scan.
                    if not _backed_by_scan(box, backings):
                        technique = "invisible_render"
                elif float(span.get("opacity", 1.0) or 0.0) <= NEAR_ZERO_OPACITY:
                    technique = "zero_alpha"
                elif float(span.get("size", 0.0) or 0.0) < TINY_FONT_PT:
                    technique = "tiny_font"
                elif _luminance(span.get("color") or (0, 0, 0)) >= LIGHT_LUMINANCE:
                    # White-ish text is legitimate over a dark fill or an image.
                    # Nothing beneath it means nothing renders.
                    if not _backed_by_scan(box, backings):
                        technique = "low_contrast"

                if technique is None and span.get("layer") in hidden_layer_names:
                    technique = "hidden_layer"

                if technique is None and (
                    box[2] < page_box[0]
                    or box[0] > page_box[2]
                    or box[3] < page_box[1]
                    or box[1] > page_box[3]
                ):
                    technique = "off_page"

                if technique:
                    hits.append(HiddenTextHit(technique, index, box, chars))

    return tuple(hits)


# OOXML is attacker-authored in this threat model, so these must not depend on
# incidental formatting. Substring matching failed on both counts: it flagged
# `<w:vanish w:val="0"/>` — the standard idiom for *disabling* an inherited
# hidden style, i.e. genuinely visible text — and it missed
# `<w:color w:themeColor="text1" w:val="FFFFFF"/>` because `w:val` no longer
# immediately followed the element name. Both reproduced.
_VANISH_ELEMENT = re.compile(r"<w:vanish\b([^>]*)>", re.IGNORECASE)
_WHITE_COLOR_ELEMENT = re.compile(
    r"<w:color\b[^>]*\bw:val\s*=\s*\"(?:FFFFFF|ffffff)\"", re.IGNORECASE
)
_SHEET_HIDDEN = re.compile(
    r"\bstate\s*=\s*\"(?:hidden|veryHidden)\"", re.IGNORECASE
)
# w:val="0"/"false"/"off" all mean the toggle is switched OFF.
_TOGGLE_OFF = re.compile(r"\bw:val\s*=\s*\"(?:0|false|off)\"", re.IGNORECASE)


def _word_hidden_hits(body: str) -> List[HiddenTextHit]:
    """Hidden-run and white-font detection that survives ordinary XML variation."""
    hits: List[HiddenTextHit] = []
    for match in _VANISH_ELEMENT.finditer(body):
        attributes = match.group(1) or ""
        if _TOGGLE_OFF.search(attributes):
            continue  # explicitly switched off: the text is visible
        hits.append(HiddenTextHit("office_hidden", 0, (0, 0, 0, 0), 0))
        break
    if _WHITE_COLOR_ELEMENT.search(body):
        hits.append(HiddenTextHit("office_hidden", 0, (0, 0, 0, 0), 0))
    return hits


def _scan_ooxml(content: bytes, mime: str) -> Tuple[HiddenTextHit, ...]:
    """Structural inspection of the package XML. Nothing is opened or executed."""
    try:
        from defusedxml import ElementTree  # noqa: F401
    except Exception:  # pragma: no cover - dependency is present
        pass

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = zf.namelist()
            hits: List[HiddenTextHit] = []

            if mime == _WORD_MIME:
                if "word/document.xml" not in names:
                    # A valid zip claiming to be a Word document but carrying no
                    # document part is corrupt or a decoy. Falling through
                    # silently would clear it; cannot inspect means cannot clear.
                    return (HiddenTextHit("unreadable", 0, (0, 0, 0, 0), 0),)
                body = zf.read("word/document.xml").decode("utf-8", "replace")
                hits.extend(_word_hidden_hits(body))

            if mime == _EXCEL_MIME:
                if "xl/workbook.xml" not in names:
                    return (HiddenTextHit("unreadable", 0, (0, 0, 0, 0), 0),)
                book = zf.read("xl/workbook.xml").decode("utf-8", "replace")
                if _SHEET_HIDDEN.search(book):
                    hits.append(HiddenTextHit("office_hidden", 0, (0, 0, 0, 0), 0))

            return tuple(hits)
    except Exception:
        return (HiddenTextHit("unreadable", 0, (0, 0, 0, 0), 0),)


def scan(content: bytes, detected_mime: str) -> Tuple[HiddenTextHit, ...]:
    """Return every hidden-text hit. Empty tuple means nothing was found."""
    if detected_mime == "application/pdf":
        return _scan_pdf(content)
    if detected_mime in (_WORD_MIME, _EXCEL_MIME):
        return _scan_ooxml(content, detected_mime)
    return ()
