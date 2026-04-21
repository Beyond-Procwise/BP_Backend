def test_ner_run_returns_spans(monkeypatch):
    from src.services.structural_extractor.nlu import ner

    def _mock_pipeline(text):
        return [
            {"entity_group": "ORG", "word": "Acme Ltd", "start": 0, "end": 8, "score": 0.99}
        ]

    monkeypatch.setattr(ner, "_get_pipeline", lambda: _mock_pipeline)
    spans = ner.run("Acme Ltd invoiced us.")
    assert spans[0]["entity_group"] == "ORG"
    assert spans[0]["word"] == "Acme Ltd"


def test_ner_run_empty_text_returns_empty():
    from src.services.structural_extractor.nlu import ner

    assert ner.run("") == []
    assert ner.run("   ") == []
