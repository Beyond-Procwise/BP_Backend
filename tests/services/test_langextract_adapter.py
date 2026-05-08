"""Tests for the LangExtract adapter — stubs the LangExtract call to keep
these tests offline. Real-Ollama integration is checked separately via
the smoke test in the README."""
import pytest

from services import langextract_adapter as lex


# ---------------------------------------------------------------------------
# Allow-list — adapter must NEVER call the LLM for fields outside this set
# ---------------------------------------------------------------------------

class TestAllowList:
    def test_adapter_returns_empty_when_no_allowed_fields(self):
        # Caller asks for PK / amount / date — none allowed
        result = lex.extract_low_confidence_fields(
            source_text="x" * 200,
            fields_needed=["invoice_id", "invoice_amount", "invoice_date"],
            doc_type="Invoice",
        )
        assert result == {}

    def test_adapter_filters_to_allowed_fields(self, monkeypatch):
        # Caller asks for a mix; only allowed ones go to LangExtract
        captured_fields = []

        def fake_extract(*args, **kwargs):
            prompt = kwargs.get("prompt_description", "")
            captured_fields.append(prompt)
            class _Result:
                extractions = []
            return _Result()

        import langextract as lx
        monkeypatch.setattr(lx, "extract", fake_extract)

        lex.extract_low_confidence_fields(
            source_text="x" * 200,
            fields_needed=["invoice_id", "supplier_name", "invoice_amount",
                           "payment_terms", "buyer_id", "buyer_address",
                           "invoice_date"],
            doc_type="Invoice",
        )
        # Prompt should mention narrative + party + address fields but NOT
        # PK / amount / date which structural extractor owns
        assert len(captured_fields) == 1
        prompt = captured_fields[0]
        assert "supplier_name" in prompt
        assert "payment_terms" in prompt
        assert "buyer_id" in prompt
        assert "buyer_address" in prompt
        assert "invoice_id" not in prompt
        assert "invoice_amount" not in prompt
        assert "invoice_date" not in prompt

    def test_short_source_text_skips_llm(self, monkeypatch):
        called = []
        import langextract as lx
        monkeypatch.setattr(lx, "extract", lambda *a, **k: called.append(1))
        result = lex.extract_low_confidence_fields(
            source_text="too short",
            fields_needed=["supplier_name"],
            doc_type="Invoice",
        )
        assert result == {}
        assert called == []  # never reached the LLM


# ---------------------------------------------------------------------------
# Grounding verification — adapter must drop ungrounded / mis-aligned spans
# ---------------------------------------------------------------------------

class TestGroundingVerification:
    def setup_method(self):
        self.text = (
            "INVOICE\nFrom: PeopleFirst HR Solutions Ltd\n"
            "Payment Terms: Net 30\nTotal: 1000\n"
        )

    def _stub_result(self, monkeypatch, extractions):
        import langextract as lx

        class _Result:
            pass
        result = _Result()
        result.extractions = extractions

        def fake_extract(*args, **kwargs):
            return result

        monkeypatch.setattr(lx, "extract", fake_extract)

    def _make_extraction(self, klass, text, start=None, end=None):
        import langextract as lx
        e = lx.data.Extraction(extraction_class=klass, extraction_text=text)
        if start is not None and end is not None:
            e.char_interval = lx.data.CharInterval(start_pos=start, end_pos=end)
        else:
            e.char_interval = None
        return e

    def test_grounded_extraction_is_kept(self, monkeypatch):
        # "PeopleFirst HR Solutions Ltd" appears at known offset
        start = self.text.find("PeopleFirst HR Solutions Ltd")
        end = start + len("PeopleFirst HR Solutions Ltd")
        e = self._make_extraction("supplier_name",
                                   "PeopleFirst HR Solutions Ltd",
                                   start=start, end=end)
        self._stub_result(monkeypatch, [e])
        result = lex.extract_low_confidence_fields(
            source_text=self.text,
            fields_needed=["supplier_name"],
            doc_type="Invoice",
        )
        assert "supplier_name" in result
        assert result["supplier_name"].value == "PeopleFirst HR Solutions Ltd"
        assert result["supplier_name"].source == "langextract"

    def test_ungrounded_extraction_is_dropped(self, monkeypatch):
        # No char_interval — must be rejected as a hallucination
        e = self._make_extraction("supplier_name", "Some Made-Up Company",
                                   start=None, end=None)
        self._stub_result(monkeypatch, [e])
        result = lex.extract_low_confidence_fields(
            source_text=self.text,
            fields_needed=["supplier_name"],
            doc_type="Invoice",
        )
        assert result == {}

    def test_misaligned_grounding_is_dropped(self, monkeypatch):
        # char_interval points at "Net 30" but extraction_text says
        # "Acme Corp" — adapter must catch the mismatch
        net30_start = self.text.find("Net 30")
        e = self._make_extraction("supplier_name", "Acme Corp",
                                   start=net30_start, end=net30_start + 6)
        self._stub_result(monkeypatch, [e])
        result = lex.extract_low_confidence_fields(
            source_text=self.text,
            fields_needed=["supplier_name"],
            doc_type="Invoice",
        )
        assert result == {}

    def test_disallowed_class_dropped_even_if_grounded(self, monkeypatch):
        # Even if LangExtract somehow returns invoice_id with grounding,
        # the adapter's allow-list must drop it
        start = self.text.find("INVOICE")
        e = self._make_extraction("invoice_id", "INVOICE", start=start, end=start+7)
        self._stub_result(monkeypatch, [e])
        result = lex.extract_low_confidence_fields(
            source_text=self.text,
            fields_needed=["supplier_name", "invoice_id"],
            doc_type="Invoice",
        )
        assert "invoice_id" not in result


# ---------------------------------------------------------------------------
# Failure modes never break the pipeline
# ---------------------------------------------------------------------------

class TestNeverRaises:
    def test_langextract_exception_returns_empty(self, monkeypatch):
        import langextract as lx

        def fake_extract(*args, **kwargs):
            raise RuntimeError("simulated network failure")

        monkeypatch.setattr(lx, "extract", fake_extract)
        result = lex.extract_low_confidence_fields(
            source_text="some content " * 50,
            fields_needed=["supplier_name"],
            doc_type="Invoice",
        )
        assert result == {}

    def test_empty_source_text_returns_empty(self):
        result = lex.extract_low_confidence_fields(
            source_text="",
            fields_needed=["supplier_name"],
            doc_type="Invoice",
        )
        assert result == {}
