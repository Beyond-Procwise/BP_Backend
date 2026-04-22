from src.services.structural_extractor.llm_fallback import (
    extract_fields_with_llm,
    verify_grounded,
)


def test_verify_grounded_accepts_substring():
    ok = verify_grounded(value="INV600254", doc_text="Invoice INV600254 dated...")
    assert ok is True


def test_verify_grounded_rejects_hallucination():
    ok = verify_grounded(value="FakeID", doc_text="Invoice INV600254 dated...")
    assert ok is False


def test_verify_grounded_ignores_whitespace():
    # The normalizer strips whitespace from both sides
    assert verify_grounded("Acme Ltd", "...from Acme  Ltd invoiced us") is True


def test_verify_grounded_empty_inputs():
    assert verify_grounded("", "something") is False
    assert verify_grounded("foo", "") is False


def test_extract_fields_drops_ungrounded(monkeypatch):
    import src.services.structural_extractor.llm_fallback as lf

    def _mock_llm(prompt):
        return (
            '{"invoice_id": {"value": "INV600254", "anchor": "Invoice INV600254"},'
            ' "supplier": {"value": "Fake Corp", "anchor": "no match"}}'
        )

    monkeypatch.setattr(lf, "_call_llm", _mock_llm)
    out = extract_fields_with_llm(
        doc_text="Invoice INV600254 dated 2020-04-01",
        fields_needed=["invoice_id", "supplier"],
        prior_attempts=[],
        attempt_no=5,
    )
    assert out["invoice_id"] == "INV600254"
    assert "supplier" not in out  # dropped: "Fake Corp" not in doc


def test_extract_fields_ignores_unknown_field(monkeypatch):
    import src.services.structural_extractor.llm_fallback as lf

    monkeypatch.setattr(
        lf,
        "_call_llm",
        lambda p: '{"invoice_id": {"value": "X1", "anchor": "X1"}, '
        '"extra": {"value": "X1", "anchor": "X1"}}',
    )
    out = extract_fields_with_llm(
        doc_text="ref X1 here",
        fields_needed=["invoice_id"],
        prior_attempts=[],
        attempt_no=5,
    )
    assert out == {"invoice_id": "X1"}


def test_extract_fields_handles_malformed_json(monkeypatch):
    import src.services.structural_extractor.llm_fallback as lf

    monkeypatch.setattr(lf, "_call_llm", lambda p: "garbage no json here")
    out = extract_fields_with_llm(
        doc_text="anything",
        fields_needed=["invoice_id"],
        prior_attempts=[],
        attempt_no=5,
    )
    assert out == {}


def test_extract_fields_handles_broken_json(monkeypatch):
    import src.services.structural_extractor.llm_fallback as lf

    monkeypatch.setattr(lf, "_call_llm", lambda p: '{"invoice_id": {broken')
    out = extract_fields_with_llm(
        doc_text="anything",
        fields_needed=["invoice_id"],
        prior_attempts=[],
        attempt_no=5,
    )
    assert out == {}


def test_extract_fields_empty_prompt_response(monkeypatch):
    import src.services.structural_extractor.llm_fallback as lf

    monkeypatch.setattr(lf, "_call_llm", lambda p: "")
    assert (
        extract_fields_with_llm(
            doc_text="doc",
            fields_needed=["invoice_id"],
            prior_attempts=[],
            attempt_no=5,
        )
        == {}
    )


def test_extract_fields_includes_prior_context(monkeypatch):
    """When prior_attempts is provided, the prompt should contain those field values."""
    import src.services.structural_extractor.llm_fallback as lf

    captured = {"prompt": ""}

    def _spy_llm(prompt):
        captured["prompt"] = prompt
        return '{"invoice_id": {"value": "INV1", "anchor": "x"}}'

    monkeypatch.setattr(lf, "_call_llm", _spy_llm)
    extract_fields_with_llm(
        doc_text="INV1 exists",
        fields_needed=["invoice_id"],
        prior_attempts=[{"supplier": "Acme"}],
        attempt_no=6,
    )
    assert "supplier" in captured["prompt"]
    assert "Acme" in captured["prompt"]
    # attempt_no >= 6 triggers extra anti-hallucination warning
    assert "Do NOT invent" in captured["prompt"]
