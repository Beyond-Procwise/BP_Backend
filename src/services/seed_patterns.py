"""Seed initial procurement patterns into the knowledge store.

These patterns are derived from verified extraction outcomes and
procurement domain knowledge. They bootstrap AgentNick's intelligence
before the system has accumulated enough data to learn on its own.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# (pattern_type, pattern_text, category, confidence)
INITIAL_PATTERNS: List[Tuple[str, str, str, float]] = [
    # === Extraction Patterns ===
    ("extraction", "UK invoices use DD/MM/YYYY dates and 20% VAT", "general", 0.95),
    ("extraction", "Supplier name appears in letterhead at top of document, not in PAYABLE TO section", "general", 0.95),
    ("extraction", "StructTree fallback needed for Canva-generated PDFs with corrupted content streams", "general", 0.90),
    ("extraction", "PDF octal escape \\243 represents £ symbol — must be decoded before parsing amounts", "general", 0.90),
    ("extraction", "Invoice ID is always prefixed with INV — extract with prefix from filename if LLM misses it", "general", 0.85),
    ("extraction", "PO and Quote IDs can be extracted reliably from filename patterns like 'SUPPLIER PO123456 for QUT789.pdf'", "general", 0.90),
    ("extraction", "Tax amount must always be less than subtotal — if reversed, auto-swap", "general", 0.95),
    ("extraction", "Quantity is always a small count number, unit_price is the cost per item — if qty > price * 10, they are swapped", "general", 0.90),
    ("extraction", "Excel/DOCX files should use external Docwain API for higher accuracy, PDF stays local", "general", 0.85),

    # === Category Intelligence Patterns ===
    ("category", "Office furniture: typical markup 30-40% above wholesale, standard VAT 20%", "office_furniture", 0.80),
    ("category", "IT equipment: prices vary 15-25% across suppliers, delivery lead time is key differentiator", "IT_equipment", 0.80),
    ("category", "Professional services: hourly rates £80-£200, fixed-price projects often have scope creep", "professional_services", 0.75),
    ("category", "Stationery: high-volume orders (100+ units) should trigger volume discount negotiation", "stationery", 0.80),
    ("category", "Office equipment: maintenance contracts add 15-20% to total cost of ownership", "office_equipment", 0.70),
    ("category", "Marketing services: retainer models preferred over per-project for ongoing needs", "marketing", 0.70),

    # === Negotiation Patterns ===
    ("negotiation", "Cooperative strategy effective with suppliers having 5+ successful prior transactions", "general", 0.80),
    ("negotiation", "First counter-offer at 15% below ask yields average 8% discount in office furniture", "office_furniture", 0.75),
    ("negotiation", "Bundling multiple line items into single counter-offer succeeds 60% of the time", "general", 0.75),
    ("negotiation", "Suppliers who respond within 24 hours are more willing to negotiate on price", "general", 0.70),
    ("negotiation", "After 2 rounds with no movement, switching from competitive to collaborative breaks deadlock", "general", 0.75),
    ("negotiation", "Offering faster payment terms (Net 15 vs Net 30) can unlock 2-3% additional discount", "general", 0.70),
    ("negotiation", "Volume commitments across quarters provide stronger leverage than single-order negotiations", "general", 0.70),
    ("negotiation", "New suppliers are most flexible on pricing during first engagement — anchor aggressively", "general", 0.75),

    # === Process Patterns ===
    ("process", "Invoice without PO reference within 30 days of PO issue needs follow-up", "general", 0.85),
    ("process", "Quote validity typically 7-30 days — check expiry proactively and request extension if needed", "general", 0.85),
    ("process", "Invoice amount deviating more than 10% from PO total is an anomaly requiring investigation", "general", 0.90),
    ("process", "Multiple quotes from same category in same quarter indicate consolidation opportunity", "general", 0.80),
    ("process", "PO without delivery confirmation after expected_delivery_date needs supplier follow-up", "general", 0.75),
    ("process", "Duplicate supplier names with slight variations indicate master data quality issue", "general", 0.80),

    # === Supplier Patterns ===
    ("supplier", "Suppliers with Ltd/LLC/PLC suffix are registered companies — verify registration for new ones", "general", 0.80),
    ("supplier", "Bank details in document footer are payment instructions, never supplier identification", "general", 0.95),
    ("supplier", "Supplier address on letterhead is registered office — delivery address may differ", "general", 0.75),
]


def seed_patterns(pattern_service) -> int:
    """Seed initial patterns into the store. Returns count of patterns seeded."""
    count = 0
    for pattern_type, pattern_text, category, confidence in INITIAL_PATTERNS:
        pattern_service.record_pattern(
            pattern_type=pattern_type,
            pattern_text=pattern_text,
            category=category,
            confidence=confidence,
        )
        count += 1
    logger.info("Seeded %d initial procurement patterns", count)
    return count
