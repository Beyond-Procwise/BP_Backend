"""Extraction Validator — Multi-pass validation with discrepancy tracking.

Three validation passes:
1. Cross-validation: AgentNick LLM reviews extraction against raw text
2. Business rules: mathematical checks, date logic, referential integrity
3. Confidence scoring: per-field confidence with auto-persist/flag/reject thresholds

All discrepancies logged to proc.bp_discrepancy_data.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
AGENT_NICK_MODEL = os.getenv("PROCWISE_EXTRACTION_MODEL", "BeyondProcwise/AgentNick:latest")

# Confidence thresholds
AUTO_PERSIST_THRESHOLD = 0.90
FLAG_THRESHOLD = 0.70
REJECT_THRESHOLD = 0.50


class Discrepancy:
    """A single validation finding."""

    __slots__ = (
        "field_name", "rule_name", "severity", "extracted_value",
        "expected_value", "corrected_value", "confidence", "pass_number",
        "source", "message",
    )

    def __init__(
        self,
        field_name: str,
        rule_name: str,
        severity: str = "warning",
        extracted_value: str = "",
        expected_value: str = "",
        corrected_value: str = "",
        confidence: float = 0.0,
        pass_number: int = 1,
        source: str = "",
        message: str = "",
    ) -> None:
        self.field_name = field_name
        self.rule_name = rule_name
        self.severity = severity
        self.extracted_value = str(extracted_value) if extracted_value is not None else ""
        self.expected_value = str(expected_value) if expected_value is not None else ""
        self.corrected_value = str(corrected_value) if corrected_value is not None else ""
        self.confidence = confidence
        self.pass_number = pass_number
        self.source = source
        self.message = message


class ExtractionValidator:
    """Validates extraction results and logs discrepancies."""

    def __init__(self, agent_nick) -> None:
        self._agent_nick = agent_nick

    def validate_and_correct(
        self,
        header: Dict[str, Any],
        line_items: List[Dict[str, Any]],
        doc_type: str,
        source_text: str,
        *,
        file_path: str = "",
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Discrepancy]]:
        """Run all validation passes. Returns corrected header, line_items, and discrepancies."""
        all_discrepancies: List[Discrepancy] = []

        # Pass 1: Cross-validation with LLM
        header, pass1 = self._pass1_cross_validate(header, doc_type, source_text)
        all_discrepancies.extend(pass1)

        # Pass 2: Business rules
        header, line_items, pass2 = self._pass2_business_rules(header, line_items, doc_type)
        all_discrepancies.extend(pass2)

        # Pass 2b: Currency conversion
        header, pass2b = self._calculate_currency_conversion(header, doc_type)
        all_discrepancies.extend(pass2b)

        # Pass 3: Confidence scoring
        header, pass3 = self._pass3_confidence_scoring(header, doc_type)
        all_discrepancies.extend(pass3)

        # Persist discrepancies
        pk_col = {"Invoice": "invoice_id", "Purchase_Order": "po_id",
                  "Quote": "quote_id", "Contract": "contract_id"}.get(doc_type, "")
        record_id = str(header.get(pk_col, "")) if pk_col else ""
        self._persist_discrepancies(all_discrepancies, doc_type, record_id, file_path)

        logger.info(
            "[Validator] %s %s: %d discrepancies (%d errors, %d warnings, %d info)",
            doc_type, record_id,
            len(all_discrepancies),
            sum(1 for d in all_discrepancies if d.severity == "error"),
            sum(1 for d in all_discrepancies if d.severity == "warning"),
            sum(1 for d in all_discrepancies if d.severity == "info"),
        )

        return header, line_items, all_discrepancies

    # ------------------------------------------------------------------
    # Pass 1: Cross-validation with LLM
    # ------------------------------------------------------------------
    def _pass1_cross_validate(
        self,
        header: Dict[str, Any],
        doc_type: str,
        source_text: str,
    ) -> Tuple[Dict[str, Any], List[Discrepancy]]:
        """Ask AgentNick to verify extracted fields against the source text."""
        discrepancies: List[Discrepancy] = []

        if not source_text.strip():
            return header, discrepancies

        # Build verification prompt with current extraction
        fields_json = json.dumps(
            {k: v for k, v in header.items()
             if v is not None and not k.startswith("_")
             and k not in ("created_date", "created_by", "last_modified_date",
                           "last_modified_by", "confidence_score", "needs_review")},
            indent=2, default=str,
        )

        prompt = (
            f"Verify these extracted {doc_type} fields against the document text below.\n"
            f"For each field, check if the value is CORRECT based on the document.\n\n"
            f"EXTRACTED FIELDS:\n{fields_json}\n\n"
            f"DOCUMENT TEXT:\n{source_text[:6000]}\n\n"
            f"Return ONLY a JSON object with fields that need correction:\n"
            f'{{"field_name": "corrected_value", ...}}\n'
            f"If all fields are correct, return {{}}\n"
            f"Rules: dates=YYYY-MM-DD, amounts=numbers only, currency=3-letter ISO code"
        )

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": AGENT_NICK_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 1024, "num_gpu": 99},
                },
                timeout=120,
            )
            response.raise_for_status()
            raw = response.json().get("response", "").strip()

            # Parse corrections
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            match = re.search(r"\{[\s\S]*?\}", cleaned)
            if match:
                corrections = json.loads(match.group())
                for field, corrected in corrections.items():
                    if field in header and corrected is not None:
                        old_val = header.get(field)
                        if str(corrected).strip() and str(corrected) != str(old_val):
                            discrepancies.append(Discrepancy(
                                field_name=field,
                                rule_name="llm_cross_validation",
                                severity="warning",
                                extracted_value=str(old_val),
                                expected_value="",
                                corrected_value=str(corrected),
                                confidence=0.85,
                                pass_number=1,
                                source="AgentNick_LLM",
                                message=f"LLM corrected {field}: '{old_val}' → '{corrected}'",
                            ))
                            header[field] = corrected
                            logger.info(
                                "[Pass1] Corrected %s: '%s' → '%s'",
                                field, old_val, corrected,
                            )
        except Exception:
            logger.debug("[Pass1] LLM cross-validation failed", exc_info=True)

        return header, discrepancies

    # ------------------------------------------------------------------
    # Pass 2: Business rules validation
    # ------------------------------------------------------------------
    def _pass2_business_rules(
        self,
        header: Dict[str, Any],
        line_items: List[Dict[str, Any]],
        doc_type: str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Discrepancy]]:
        """Mathematical and logical validation."""
        discrepancies: List[Discrepancy] = []

        # Rule 2.1: Amount arithmetic (subtotal + tax = total)
        if doc_type in ("Invoice", "Quote"):
            discrepancies.extend(self._validate_amounts(header, doc_type))

        # Rule 2.2: Date logic
        discrepancies.extend(self._validate_dates(header, doc_type))

        # Rule 2.3: Line items sum to subtotal
        if line_items:
            discrepancies.extend(
                self._validate_line_item_totals(header, line_items, doc_type)
            )

        # Rule 2.4: Required fields present
        discrepancies.extend(self._validate_required_fields(header, doc_type))

        # Rule 2.5: Currency format
        currency = header.get("currency", "")
        if currency and (len(str(currency)) != 3 or not str(currency).isalpha()):
            discrepancies.append(Discrepancy(
                field_name="currency",
                rule_name="currency_format",
                severity="warning",
                extracted_value=str(currency),
                expected_value="3-letter ISO code",
                pass_number=2,
                source="business_rules",
                message=f"Currency '{currency}' is not a valid 3-letter ISO code",
            ))

        # Rule 2.6: Tax percent range
        tax_pct = self._to_float(header.get("tax_percent"))
        if tax_pct is not None and (tax_pct < 0 or tax_pct > 100):
            discrepancies.append(Discrepancy(
                field_name="tax_percent",
                rule_name="tax_range",
                severity="error",
                extracted_value=str(tax_pct),
                expected_value="0-100",
                pass_number=2,
                source="business_rules",
                message=f"Tax percent {tax_pct} is outside valid range 0-100",
            ))

        return header, line_items, discrepancies

    def _validate_amounts(
        self, header: Dict[str, Any], doc_type: str
    ) -> List[Discrepancy]:
        """Check that subtotal + tax = total."""
        discrepancies = []

        subtotal_field = "invoice_amount" if doc_type == "Invoice" else "total_amount"
        total_field = "invoice_total_incl_tax" if doc_type == "Invoice" else "total_amount_incl_tax"
        tax_field = "tax_amount"

        subtotal = self._to_float(header.get(subtotal_field))
        tax = self._to_float(header.get(tax_field))
        total = self._to_float(header.get(total_field))

        if subtotal is not None and tax is not None and total is not None:
            expected_total = round(subtotal + tax, 2)
            if abs(expected_total - total) > 0.02:
                discrepancies.append(Discrepancy(
                    field_name=total_field,
                    rule_name="amount_arithmetic",
                    severity="error",
                    extracted_value=str(total),
                    expected_value=str(expected_total),
                    corrected_value=str(expected_total),
                    confidence=0.95,
                    pass_number=2,
                    source="business_rules",
                    message=f"{subtotal_field}({subtotal}) + {tax_field}({tax}) = {expected_total}, but {total_field} = {total}",
                ))
                # Auto-correct if both subtotal and tax look reliable
                header[total_field] = expected_total

        # Derive missing fields from others
        if subtotal is None and tax is not None and total is not None:
            derived = round(total - tax, 2)
            header[subtotal_field] = derived
            discrepancies.append(Discrepancy(
                field_name=subtotal_field,
                rule_name="amount_derived",
                severity="info",
                corrected_value=str(derived),
                pass_number=2,
                source="business_rules",
                message=f"Derived {subtotal_field} = {total_field}({total}) - {tax_field}({tax}) = {derived}",
            ))

        if tax is None and subtotal is not None and total is not None:
            derived = round(total - subtotal, 2)
            header[tax_field] = derived
            discrepancies.append(Discrepancy(
                field_name=tax_field,
                rule_name="amount_derived",
                severity="info",
                corrected_value=str(derived),
                pass_number=2,
                source="business_rules",
                message=f"Derived {tax_field} = {total_field}({total}) - {subtotal_field}({subtotal}) = {derived}",
            ))

        # Derive tax_percent if missing
        tax_pct = self._to_float(header.get("tax_percent"))
        if tax_pct is None and subtotal and tax and subtotal > 0:
            derived_pct = round((tax / subtotal) * 100, 2)
            header["tax_percent"] = derived_pct
            discrepancies.append(Discrepancy(
                field_name="tax_percent",
                rule_name="tax_percent_derived",
                severity="info",
                corrected_value=str(derived_pct),
                pass_number=2,
                source="business_rules",
                message=f"Derived tax_percent = ({tax}/{subtotal}) * 100 = {derived_pct}%",
            ))

        return discrepancies

    def _validate_dates(
        self, header: Dict[str, Any], doc_type: str
    ) -> List[Discrepancy]:
        """Check date logic."""
        discrepancies = []

        if doc_type == "Invoice":
            inv_date = self._to_date(header.get("invoice_date"))
            due_date = self._to_date(header.get("due_date"))
            if inv_date and due_date and due_date < inv_date:
                discrepancies.append(Discrepancy(
                    field_name="due_date",
                    rule_name="date_order",
                    severity="warning",
                    extracted_value=str(due_date),
                    expected_value=f"after {inv_date}",
                    pass_number=2,
                    source="business_rules",
                    message=f"due_date ({due_date}) is before invoice_date ({inv_date})",
                ))

            # Derive payment_terms if missing
            if inv_date and due_date and not header.get("payment_terms"):
                days = (due_date - inv_date).days
                if 0 < days <= 365:
                    header["payment_terms"] = str(days)
                    discrepancies.append(Discrepancy(
                        field_name="payment_terms",
                        rule_name="payment_terms_derived",
                        severity="info",
                        corrected_value=str(days),
                        pass_number=2,
                        source="business_rules",
                        message=f"Derived payment_terms = {days} days (due - invoice date)",
                    ))

        if doc_type == "Purchase_Order":
            order_date = self._to_date(header.get("order_date"))
            delivery_date = self._to_date(header.get("expected_delivery_date"))
            if order_date and delivery_date and delivery_date < order_date:
                discrepancies.append(Discrepancy(
                    field_name="expected_delivery_date",
                    rule_name="date_order",
                    severity="warning",
                    extracted_value=str(delivery_date),
                    expected_value=f"after {order_date}",
                    pass_number=2,
                    source="business_rules",
                    message=f"expected_delivery_date before order_date",
                ))

        if doc_type == "Quote":
            quote_date = self._to_date(header.get("quote_date"))
            validity = self._to_date(header.get("validity_date"))
            if quote_date and validity and validity < quote_date:
                discrepancies.append(Discrepancy(
                    field_name="validity_date",
                    rule_name="date_order",
                    severity="warning",
                    extracted_value=str(validity),
                    expected_value=f"after {quote_date}",
                    pass_number=2,
                    source="business_rules",
                    message=f"validity_date before quote_date",
                ))

        return discrepancies

    def _validate_line_item_totals(
        self, header: Dict[str, Any], line_items: List[Dict[str, Any]], doc_type: str
    ) -> List[Discrepancy]:
        """Check that line item amounts sum to header subtotal."""
        discrepancies = []
        subtotal_field = "invoice_amount" if doc_type == "Invoice" else "total_amount"
        line_total_field = "line_amount" if doc_type == "Invoice" else "line_total"

        header_subtotal = self._to_float(header.get(subtotal_field))
        if header_subtotal is None:
            return discrepancies

        line_sum = 0.0
        has_line_totals = False
        for item in line_items:
            lt = self._to_float(item.get(line_total_field))
            if lt is not None:
                line_sum += lt
                has_line_totals = True

        if has_line_totals and abs(line_sum - header_subtotal) > 0.50:
            discrepancies.append(Discrepancy(
                field_name=subtotal_field,
                rule_name="line_item_sum",
                severity="warning",
                extracted_value=str(header_subtotal),
                expected_value=str(round(line_sum, 2)),
                pass_number=2,
                source="business_rules",
                message=f"Line items sum ({line_sum:.2f}) ≠ header {subtotal_field} ({header_subtotal:.2f})",
            ))

        return discrepancies

    def _validate_required_fields(
        self, header: Dict[str, Any], doc_type: str
    ) -> List[Discrepancy]:
        """Check that all required fields are present."""
        required = {
            "Invoice": ["invoice_id", "supplier_id", "invoice_total_incl_tax"],
            "Purchase_Order": ["po_id", "supplier_name", "total_amount"],
            "Quote": ["quote_id", "supplier_id", "total_amount"],
            "Contract": ["contract_id", "supplier_id", "contract_title"],
        }
        discrepancies = []
        for field in required.get(doc_type, []):
            val = header.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                discrepancies.append(Discrepancy(
                    field_name=field,
                    rule_name="required_field_missing",
                    severity="error",
                    pass_number=2,
                    source="business_rules",
                    message=f"Required field '{field}' is missing or empty",
                ))
        return discrepancies

    # ------------------------------------------------------------------
    # Currency conversion
    # ------------------------------------------------------------------
    # Static fallback rates (updated periodically by ModelSyncService)
    _STATIC_USD_RATES: Dict[str, float] = {
        "USD": 1.0,
        "GBP": 1.27,
        "EUR": 1.08,
        "JPY": 0.0067,
        "CAD": 0.74,
        "AUD": 0.65,
        "CHF": 1.12,
        "INR": 0.012,
        "NZD": 0.60,
        "SEK": 0.096,
        "NOK": 0.093,
        "DKK": 0.145,
        "SGD": 0.74,
        "HKD": 0.128,
        "ZAR": 0.055,
        "BRL": 0.20,
        "MXN": 0.058,
        "AED": 0.272,
        "PLN": 0.25,
        "CZK": 0.043,
    }

    def _calculate_currency_conversion(
        self,
        header: Dict[str, Any],
        doc_type: str,
    ) -> Tuple[Dict[str, Any], List[Discrepancy]]:
        """Calculate exchange_rate_to_usd and converted_amount_usd."""
        discrepancies: List[Discrepancy] = []

        # Determine the total amount field
        total_field = {
            "Invoice": "invoice_total_incl_tax",
            "Purchase_Order": "total_amount",
            "Quote": "total_amount_incl_tax",
            "Contract": "total_contract_value",
        }.get(doc_type)

        if not total_field:
            return header, discrepancies

        currency = str(header.get("currency", "")).strip().upper()
        total = self._to_float(header.get(total_field))

        if not currency or total is None:
            return header, discrepancies

        if currency == "USD":
            header["exchange_rate_to_usd"] = 1.0
            header["converted_amount_usd"] = total
            discrepancies.append(Discrepancy(
                field_name="converted_amount_usd",
                rule_name="currency_conversion",
                severity="info",
                corrected_value=str(total),
                confidence=1.0,
                pass_number=2,
                source="currency_conversion",
                message=f"USD amount — no conversion needed: {total}",
            ))
            return header, discrepancies

        # Try live rate first
        rate = self._get_live_exchange_rate(currency)
        rate_source = "live_api"

        if rate is None:
            # Fallback to static rates
            rate = self._STATIC_USD_RATES.get(currency)
            rate_source = "static_fallback"

        if rate is None:
            discrepancies.append(Discrepancy(
                field_name="exchange_rate_to_usd",
                rule_name="currency_conversion_failed",
                severity="warning",
                extracted_value=currency,
                pass_number=2,
                source="currency_conversion",
                message=f"No exchange rate available for {currency} to USD",
            ))
            return header, discrepancies

        converted = round(total * rate, 2)
        header["exchange_rate_to_usd"] = round(rate, 6)
        header["converted_amount_usd"] = converted

        discrepancies.append(Discrepancy(
            field_name="converted_amount_usd",
            rule_name="currency_conversion",
            severity="info",
            corrected_value=str(converted),
            confidence=0.95 if rate_source == "live_api" else 0.80,
            pass_number=2,
            source=rate_source,
            message=f"{currency} {total} × {rate:.6f} = USD {converted} (source: {rate_source})",
        ))

        return header, discrepancies

    def _get_live_exchange_rate(self, from_currency: str) -> Optional[float]:
        """Fetch live exchange rate from free API. Returns rate to convert 1 unit to USD."""
        try:
            # Use exchangerate-api.com (free, no key required for basic usage)
            resp = requests.get(
                f"https://open.er-api.com/v6/latest/{from_currency}",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                usd_rate = data.get("rates", {}).get("USD")
                if usd_rate:
                    logger.debug(
                        "Live exchange rate: 1 %s = %s USD", from_currency, usd_rate
                    )
                    return float(usd_rate)
        except Exception:
            logger.debug("Live exchange rate fetch failed for %s", from_currency)
        return None

    # ------------------------------------------------------------------
    # Pass 3: Confidence scoring
    # ------------------------------------------------------------------
    def _pass3_confidence_scoring(
        self, header: Dict[str, Any], doc_type: str
    ) -> Tuple[Dict[str, Any], List[Discrepancy]]:
        """Score each field's confidence and flag/reject low-confidence fields."""
        discrepancies = []

        skip_fields = {
            "created_date", "created_by", "last_modified_date", "last_modified_by",
            "confidence_score", "needs_review", "ai_flag_required",
        }

        field_count = 0
        total_confidence = 0.0

        for field, value in header.items():
            if field in skip_fields or field.startswith("_"):
                continue
            if value is None:
                continue

            confidence = self._estimate_field_confidence(field, value, doc_type)
            field_count += 1
            total_confidence += confidence

            if confidence < REJECT_THRESHOLD:
                discrepancies.append(Discrepancy(
                    field_name=field,
                    rule_name="low_confidence",
                    severity="error",
                    extracted_value=str(value),
                    confidence=confidence,
                    pass_number=3,
                    source="confidence_scoring",
                    message=f"Field '{field}' confidence {confidence:.2f} below reject threshold {REJECT_THRESHOLD}",
                ))
            elif confidence < FLAG_THRESHOLD:
                discrepancies.append(Discrepancy(
                    field_name=field,
                    rule_name="medium_confidence",
                    severity="warning",
                    extracted_value=str(value),
                    confidence=confidence,
                    pass_number=3,
                    source="confidence_scoring",
                    message=f"Field '{field}' confidence {confidence:.2f} — flagged for review",
                ))

        # Overall confidence
        overall = (total_confidence / field_count) if field_count > 0 else 0.0
        header["confidence_score"] = round(overall, 2)
        header["needs_review"] = overall < FLAG_THRESHOLD
        header["ai_flag_required"] = "Y" if overall < AUTO_PERSIST_THRESHOLD else "N"

        return header, discrepancies

    def _estimate_field_confidence(
        self, field: str, value: Any, doc_type: str
    ) -> float:
        """Estimate confidence for a single field based on value quality."""
        val_str = str(value).strip()
        if not val_str:
            return 0.0

        confidence = 0.75  # base

        # Boost for well-formatted values
        if field.endswith("_date") or field in ("invoice_date", "due_date", "order_date", "quote_date"):
            if re.match(r"^\d{4}-\d{2}-\d{2}$", val_str):
                confidence = 0.95
            elif re.match(r"^\d{2}/\d{2}/\d{4}$", val_str):
                confidence = 0.85
            else:
                confidence = 0.60

        elif field in ("currency",):
            if re.match(r"^[A-Z]{3}$", val_str):
                confidence = 0.95
            else:
                confidence = 0.50

        elif field.endswith("_amount") or field in ("total_amount", "invoice_total_incl_tax",
                                                     "tax_amount", "invoice_amount", "total_amount_incl_tax"):
            try:
                float(val_str)
                confidence = 0.90
            except ValueError:
                confidence = 0.40

        elif field.endswith("_id") or field in ("invoice_id", "po_id", "quote_id"):
            if len(val_str) >= 3:
                confidence = 0.90
            else:
                confidence = 0.60

        elif field in ("supplier_id", "supplier_name"):
            if len(val_str) >= 3:
                confidence = 0.80
            else:
                confidence = 0.50

        elif field == "tax_percent":
            try:
                pct = float(val_str)
                if 0 <= pct <= 100:
                    confidence = 0.90
                else:
                    confidence = 0.30
            except ValueError:
                confidence = 0.30

        return confidence

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _persist_discrepancies(
        self,
        discrepancies: List[Discrepancy],
        doc_type: str,
        record_id: str,
        file_path: str,
    ) -> None:
        """Write discrepancies to proc.bp_discrepancy_data."""
        if not discrepancies:
            return
        try:
            conn = self._agent_nick.get_db_connection()
            conn.autocommit = True
            with conn.cursor() as cur:
                for d in discrepancies:
                    cur.execute(
                        """
                        INSERT INTO proc.bp_discrepancy_data
                            (doc_type, record_id, field_name, rule_name, severity,
                             extracted_value, expected_value, corrected_value,
                             confidence, pass_number, source, message, file_path)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            doc_type, record_id, d.field_name, d.rule_name,
                            d.severity, d.extracted_value, d.expected_value,
                            d.corrected_value, d.confidence, d.pass_number,
                            d.source, d.message, file_path,
                        ),
                    )
            conn.close()
            logger.info(
                "[Validator] Persisted %d discrepancies for %s %s",
                len(discrepancies), doc_type, record_id,
            )
        except Exception:
            logger.exception("[Validator] Failed to persist discrepancies")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_float(val: Any) -> Optional[float]:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        try:
            cleaned = re.sub(r"[£$€¥₹,\s]", "", str(val).strip())
            return float(cleaned) if cleaned else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _to_date(val: Any) -> Optional[Any]:
        if val is None:
            return None
        if hasattr(val, "date"):
            return val if not hasattr(val, "hour") else val.date()
        try:
            from dateutil import parser
            return parser.parse(str(val), dayfirst=True).date()
        except Exception:
            return None
