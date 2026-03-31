"""Adaptive OCR: detects input document quality and selects optimal OCR strategy.

Layer 1 of the extraction pipeline. Wraps the existing OCR pipeline
(ocr_pipeline.py) with quality detection to avoid unnecessary OCR on
digital PDFs and to apply preprocessing on poor scans.

Spec reference: Section 6, Layer 1 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class DocumentQuality(Enum):
    DIGITAL = "digital"          # Generated PDF, no OCR needed
    CLEAN_SCAN = "clean_scan"    # Good quality scan, standard OCR
    POOR_SCAN = "poor_scan"      # Low DPI or noisy, enhanced OCR
    PHOTO = "photo"              # Camera photo, skewed, needs preprocessing


class OCRStrategy(Enum):
    DIRECT_TEXT = "direct_text"      # Extract text directly from PDF
    STANDARD_OCR = "standard_ocr"    # Standard OCR (EasyOCR/Tesseract)
    ENHANCED_OCR = "enhanced_ocr"    # Preprocessing + OCR (deskew, denoise)


class AdaptiveOCR:
    # DPI thresholds for quality detection
    HIGH_DPI_THRESHOLD = 200
    LOW_DPI_THRESHOLD = 100

    def detect_quality(
        self,
        text_content: str = "",
        page_count: int = 1,
        has_extractable_text: bool = False,
        image_ratio: float = 0.0,
        estimated_dpi: Optional[int] = None,
        is_skewed: bool = False,
    ) -> DocumentQuality:
        # Digital PDF: has extractable text and minimal image content
        if has_extractable_text and len(text_content.strip()) > 20:
            return DocumentQuality.DIGITAL

        # Photo: skewed or very low DPI
        if is_skewed or (estimated_dpi is not None and estimated_dpi < self.LOW_DPI_THRESHOLD):
            return DocumentQuality.PHOTO

        # Poor scan: low DPI but not skewed
        if estimated_dpi is not None and estimated_dpi < self.HIGH_DPI_THRESHOLD:
            return DocumentQuality.POOR_SCAN

        # Clean scan: high DPI, no text
        return DocumentQuality.CLEAN_SCAN

    def select_strategy(self, quality: DocumentQuality) -> OCRStrategy:
        return {
            DocumentQuality.DIGITAL: OCRStrategy.DIRECT_TEXT,
            DocumentQuality.CLEAN_SCAN: OCRStrategy.STANDARD_OCR,
            DocumentQuality.POOR_SCAN: OCRStrategy.ENHANCED_OCR,
            DocumentQuality.PHOTO: OCRStrategy.ENHANCED_OCR,
        }[quality]

    def extract_text(
        self,
        quality: DocumentQuality,
        pdf_text: str = "",
        file_path: str = "",
        ocr_func: Optional[Callable] = None,
    ) -> str:
        strategy = self.select_strategy(quality)

        if strategy == OCRStrategy.DIRECT_TEXT:
            logger.info("Digital PDF detected, using direct text extraction")
            return pdf_text

        if ocr_func is None:
            logger.warning("No OCR function provided, falling back to pdf_text")
            return pdf_text

        if strategy == OCRStrategy.ENHANCED_OCR:
            logger.info("Poor quality detected, applying enhanced OCR with preprocessing")
            # Enhanced OCR uses the same function but caller should preprocess first
            return ocr_func(file_path)

        logger.info("Clean scan detected, using standard OCR")
        return ocr_func(file_path)
