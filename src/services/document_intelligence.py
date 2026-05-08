"""Document intelligence — section detection and smart text building.

Analyzes raw document text to identify structural sections (header,
line items, summary, payment) and builds optimized text for LLM
extraction with priority ordering that never truncates line items.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Section detection markers
_LINE_ITEM_HEADERS = re.compile(
    r"(description|item\b|qty|quantity|unit\s*price|price|amount|total|cost|rate|"
    r"unit\s*cost|line\s*total|ext\.?\s*price|net\s*amount)",
    re.IGNORECASE,
)
_SUMMARY_MARKERS = re.compile(
    r"(sub[\-\s]?total|subtotal|total\s*(before|excl)|net\s*total|"
    r"tax\s|vat\s|gst\s|grand\s*total|total\s*(incl|due|payable|amount)|amount\s*due|balance\s*due)",
    re.IGNORECASE,
)
_PAYMENT_MARKERS = re.compile(
    r"(bank\s*(name|account|code)|sort\s*code|iban|swift|bic|"
    r"payment\s*(info|details|method)|remittance|account\s*(no|number))",
    re.IGNORECASE,
)


@dataclass
class DocumentSection:
    name: str
    start_line: int
    end_line: int
    priority: int  # lower = higher priority for text building


@dataclass
class DocumentStructure:
    sections: List[DocumentSection] = field(default_factory=list)
    total_lines: int = 0

    def get_section(self, name: str) -> Optional[DocumentSection]:
        for s in self.sections:
            if s.name == name:
                return s
        return None


def detect_sections(text: str) -> DocumentStructure:
    """Detect structural sections in document text."""
    lines = text.split("\n")
    total = len(lines)
    structure = DocumentStructure(total_lines=total)

    if total == 0:
        return structure

    item_start = None
    item_end = None
    summary_start = None
    payment_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Detect line item table header (2+ header keywords on same line or pipe-separated)
        if item_start is None:
            matches = _LINE_ITEM_HEADERS.findall(stripped)
            if len(matches) >= 2:
                item_start = i
                continue

        # After line items started, detect summary section
        if item_start is not None and item_end is None:
            if _SUMMARY_MARKERS.search(stripped):
                item_end = i
                summary_start = i
                continue

        # Detect payment section
        if _PAYMENT_MARKERS.search(stripped) and payment_start is None:
            payment_start = i

    # Build sections
    header_end = item_start or summary_start or payment_start or min(20, total)
    structure.sections.append(DocumentSection("header", 0, header_end, priority=2))

    if item_start is not None:
        end = item_end or summary_start or payment_start or total
        structure.sections.append(DocumentSection("line_items", item_start, end, priority=1))

    if summary_start is not None:
        end = payment_start or min(summary_start + 15, total)
        structure.sections.append(DocumentSection("summary", summary_start, end, priority=2))

    if payment_start is not None:
        structure.sections.append(DocumentSection("payment", payment_start, total, priority=4))

    return structure


def build_smart_text(text: str, max_chars: int = 6000) -> str:
    """Build optimized text for LLM extraction.

    Priority ordering ensures line items are NEVER truncated:
    1. All line items (priority 1) — non-negotiable
    2. Header + summary (priority 2) — essential
    3. Remaining content (priority 3+) — if space permits
    4. Boilerplate dropped last
    """
    lines = text.split("\n")
    structure = detect_sections(text)

    if not structure.sections or len(structure.sections) <= 1:
        # No clear structure detected — return as much as fits
        return text[:max_chars]

    # Sort sections by priority
    sorted_sections = sorted(structure.sections, key=lambda s: s.priority)

    parts = []
    total_chars = 0

    for section in sorted_sections:
        section_lines = lines[section.start_line:section.end_line]
        section_text = "\n".join(line for line in section_lines if line.strip())

        if not section_text.strip():
            continue

        tag = section.name.upper().replace("_", " ")

        if total_chars + len(section_text) <= max_chars:
            parts.append(f"[{tag}]\n{section_text}")
            total_chars += len(section_text) + len(tag) + 4
        elif section.priority <= 2:
            # High priority sections — include even if over limit
            parts.append(f"[{tag}]\n{section_text}")
            total_chars += len(section_text)

    result = "\n\n".join(parts)

    # If structured output is too small, fall back to raw text
    if len(result) < 100:
        return text[:max_chars]

    return result


def count_source_line_items(text: str) -> int:
    """Count the number of line item rows in the source text.

    Used for completeness verification after extraction.
    """
    structure = detect_sections(text)
    items_section = structure.get_section("line_items")
    if not items_section:
        return 0

    lines = text.split("\n")
    # Skip the header row itself
    item_lines = lines[items_section.start_line + 1:items_section.end_line]

    # Count non-empty lines that look like data rows (contain numbers)
    count = 0
    number_pattern = re.compile(r"\d")
    for line in item_lines:
        stripped = line.strip()
        if stripped and number_pattern.search(stripped) and len(stripped) > 5:
            count += 1

    return count
