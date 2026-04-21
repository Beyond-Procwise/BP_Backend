def test_layout_regions_stub(monkeypatch):
    from src.services.structural_extractor.nlu import layout

    monkeypatch.setattr(
        layout,
        "_get_detector",
        lambda: (lambda img: [{"bbox": [0, 0, 200, 100], "label": "title"}]),
    )
    regions = layout.detect_regions(b"fake", page_num=1)
    assert regions[0]["label"] == "title"


def test_layout_default_loader_raises():
    from src.services.structural_extractor.nlu import layout

    # Reset cache so _get_detector takes the uncached path
    layout._DETECTOR_CACHE["d"] = None
    try:
        layout._get_detector()
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
