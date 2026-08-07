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
import zipfile
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

TINY_FONT_PT = 1.5
NEAR_ZERO_OPACITY = 0.05
# Luminance above which text is treated as white-ish for the contrast check.
LIGHT_LUMINANCE = 0.90
# Fraction of a text span's area that must sit over an image for it to read as
# an OCR layer rather than free-floating injected text.
OCR_OVERLAP_THRESHOLD = 0.60

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


def _backed_by_image(span_box, image_rects) -> bool:
    return any(
        _overlap_fraction(span_box, tuple(rect)) >= OCR_OVERLAP_THRESHOLD
        for rect in image_rects
    )


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

            image_rects = _image_rects(page)
            page_box = tuple(page.rect)

            for span in spans:
                box = tuple(span.get("bbox") or (0, 0, 0, 0))
                chars = len(span.get("chars") or ())
                if chars == 0:
                    continue
                technique: Optional[str] = None

                if span.get("type") == 3:
                    # Render mode 3. Benign when it is an OCR layer over a scan.
                    if not _backed_by_image(box, image_rects):
                        technique = "invisible_render"
                elif float(span.get("opacity", 1.0) or 0.0) <= NEAR_ZERO_OPACITY:
                    technique = "zero_alpha"
                elif float(span.get("size", 0.0) or 0.0) < TINY_FONT_PT:
                    technique = "tiny_font"
                elif _luminance(span.get("color") or (0, 0, 0)) >= LIGHT_LUMINANCE:
                    # White-ish text is legitimate over a dark fill or an image.
                    # Nothing beneath it means nothing renders.
                    if not _backed_by_image(box, image_rects):
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

            if mime == _WORD_MIME and "word/document.xml" in names:
                body = zf.read("word/document.xml").decode("utf-8", "replace")
                if "<w:vanish/>" in body or "<w:vanish " in body:
                    hits.append(HiddenTextHit("office_hidden", 0, (0, 0, 0, 0), 0))
                if 'w:color w:val="FFFFFF"' in body:
                    hits.append(HiddenTextHit("office_hidden", 0, (0, 0, 0, 0), 0))

            if mime == _EXCEL_MIME and "xl/workbook.xml" in names:
                book = zf.read("xl/workbook.xml").decode("utf-8", "replace")
                if 'state="hidden"' in book or 'state="veryHidden"' in book:
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
