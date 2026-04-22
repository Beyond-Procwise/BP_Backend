def test_detect_tables_stub(monkeypatch):
    from src.services.structural_extractor.nlu import table_transformer as tt

    monkeypatch.setattr(
        tt,
        "_get_detector",
        lambda: (lambda img: [{"bbox": [0, 0, 100, 100], "score": 0.9}]),
    )
    regions = tt.detect_tables(page_image_bytes=b"fake-image", page_num=1)
    assert len(regions) == 1
    assert regions[0]["bbox"] == [0, 0, 100, 100]
