"""The filler: what the user is looking at is translated before everything else."""
from __future__ import annotations

from src.services.i18n.filler import BACKGROUND, SCREEN, Filler
from src.services.i18n.service import TranslateResult


class FakeService:
    batch_size = 2

    def __init__(self):
        self.calls = []

    def translate(self, lang, texts, *, lang_name=None):
        self.calls.append((lang, sorted(texts.values())))
        return TranslateResult(translations={k: v.upper() for k, v in texts.items()})


def test_screen_before_background_and_batched_per_language():
    svc = FakeService()
    f = Filler(svc, start_thread=False)
    f.enqueue("es", {"a": "bg1", "b": "bg2"}, BACKGROUND)
    f.enqueue("es", {"c": "scr"}, SCREEN)
    assert f.pending("es") == 3
    f.run_once()
    # batch_size 2: the screen string plus the oldest background one
    assert svc.calls[0] == ("es", ["bg1", "scr"])
    f.run_once()
    assert f.pending("es") == 0 and f.run_once() == 0


def test_duplicates_are_queued_once_and_priority_upgrades():
    svc = FakeService()
    f = Filler(svc, start_thread=False)
    f.enqueue("de", {"c": "y"}, BACKGROUND)
    assert f.enqueue("fr", {"a": "x"}, BACKGROUND) == 1
    assert f.enqueue("fr", {"b": "x"}, SCREEN) == 0
    f.run_once()
    assert svc.calls[0] == ("fr", ["x"])


def test_worker_survives_a_failing_batch():
    class Boom(FakeService):
        def translate(self, lang, texts, *, lang_name=None):
            raise RuntimeError("gpu fell over")

    f = Filler(Boom(), start_thread=False)
    f.enqueue("es", {"a": "x"}, SCREEN)
    assert f.run_once() == 1 and f.pending("es") == 0


def test_thread_drains_the_queue():
    import time

    svc = FakeService()
    f = Filler(svc)
    f.enqueue("es", {"a": "x", "b": "y", "c": "z"}, SCREEN)
    deadline = time.monotonic() + 5
    while sum(len(c[1]) for c in svc.calls) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert f.pending("es") == 0 and sum(len(c[1]) for c in svc.calls) == 3
