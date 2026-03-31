"""Tests for adaptive OCR quality detection and strategy selection."""
import pytest
from unittest.mock import MagicMock, patch


def test_detect_digital_pdf():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality

    ocr = AdaptiveOCR()
    # A digital PDF has extractable text and no images-only pages
    quality = ocr.detect_quality(
        text_content="Invoice #12345\nSupplier: Acme Corp\nTotal: $1,500.00",
        page_count=1,
        has_extractable_text=True,
        image_ratio=0.0,
    )
    assert quality == DocumentQuality.DIGITAL


def test_detect_clean_scan():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality

    ocr = AdaptiveOCR()
    quality = ocr.detect_quality(
        text_content="",
        page_count=1,
        has_extractable_text=False,
        image_ratio=1.0,
        estimated_dpi=300,
    )
    assert quality == DocumentQuality.CLEAN_SCAN


def test_detect_poor_scan():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality

    ocr = AdaptiveOCR()
    quality = ocr.detect_quality(
        text_content="",
        page_count=1,
        has_extractable_text=False,
        image_ratio=1.0,
        estimated_dpi=150,
    )
    assert quality == DocumentQuality.POOR_SCAN


def test_detect_photo():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality

    ocr = AdaptiveOCR()
    quality = ocr.detect_quality(
        text_content="",
        page_count=1,
        has_extractable_text=False,
        image_ratio=1.0,
        estimated_dpi=72,
        is_skewed=True,
    )
    assert quality == DocumentQuality.PHOTO


def test_strategy_for_digital():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality, OCRStrategy

    ocr = AdaptiveOCR()
    strategy = ocr.select_strategy(DocumentQuality.DIGITAL)
    assert strategy == OCRStrategy.DIRECT_TEXT


def test_strategy_for_clean_scan():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality, OCRStrategy

    ocr = AdaptiveOCR()
    strategy = ocr.select_strategy(DocumentQuality.CLEAN_SCAN)
    assert strategy == OCRStrategy.STANDARD_OCR


def test_strategy_for_poor_scan():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality, OCRStrategy

    ocr = AdaptiveOCR()
    strategy = ocr.select_strategy(DocumentQuality.POOR_SCAN)
    assert strategy == OCRStrategy.ENHANCED_OCR


def test_strategy_for_photo():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality, OCRStrategy

    ocr = AdaptiveOCR()
    strategy = ocr.select_strategy(DocumentQuality.PHOTO)
    assert strategy == OCRStrategy.ENHANCED_OCR


def test_extract_text_digital_skips_ocr():
    from services.adaptive_ocr import AdaptiveOCR, DocumentQuality

    ocr = AdaptiveOCR()
    mock_extractor = MagicMock()
    mock_extractor.return_value = "Invoice #12345"

    result = ocr.extract_text(
        quality=DocumentQuality.DIGITAL,
        pdf_text="Invoice #12345",
        file_path="/tmp/test.pdf",
        ocr_func=mock_extractor,
    )
    assert result == "Invoice #12345"
    mock_extractor.assert_not_called()  # Should not call OCR for digital PDFs
