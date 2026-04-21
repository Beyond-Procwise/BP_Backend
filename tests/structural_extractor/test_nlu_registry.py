import threading

from src.services.structural_extractor.nlu._registry import ModelRegistry


def test_model_registry_load_once():
    ModelRegistry._instances.clear()
    call_count = {"n": 0}

    def _loader(name):
        call_count["n"] += 1
        return f"loaded-{name}"

    ModelRegistry._loader = staticmethod(_loader)
    threads = [threading.Thread(target=lambda: ModelRegistry.get("foo")) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert call_count["n"] == 1
    assert ModelRegistry.get("foo") == "loaded-foo"


def test_model_registry_warm_loads_all():
    ModelRegistry._instances.clear()
    ModelRegistry._loader = staticmethod(lambda name: f"w-{name}")
    ModelRegistry.warm(["a", "b"])
    assert ModelRegistry._instances["a"] == "w-a"
    assert ModelRegistry._instances["b"] == "w-b"


def test_model_registry_default_loader_raises():
    ModelRegistry._instances.clear()
    ModelRegistry._loader = None
    try:
        ModelRegistry.get("no_loader")
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
