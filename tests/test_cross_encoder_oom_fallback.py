"""A full GPU must cost us the reranker's speed, never the whole API.

`load_cross_encoder` is documented as initialising "on the desired device with
GPU fallback", but the only fallback it had was `except NotImplementedError`
whose message contains "meta tensor". `torch.OutOfMemoryError` -- the ordinary
way a GPU load fails -- propagated instead, out of `RAGPipeline(agent_nick)`,
into lifespan's outer `except Exception`, which nulls `agent_nick`, the
orchestrator, the agent registry and eleven other pieces of app state and then
serves requests anyway. One optional reranker took the whole system down to a
degraded boot, and the only symptom was a CRITICAL line in the journal.

Observed on this host: Ollama's model runner holds 19,520 MiB of the 23,028 MiB
card, leaving ~1.1 GiB against the reranker's 2.07 GiB request.

The subtle part, and the reason this file tests the ambient device rather than
just the outcome: `configure_gpu()` calls `torch.set_default_device("cuda")`
process-wide, so a fallback that merely passes `device="cpu"` **still OOMs on
CUDA** -- transformers resolves a CUDA device map from the global default and
warms its allocator there. Verified by running it. The fallback must neutralise
the default device for the duration of the CPU construction, which is what
`test_the_cpu_retry_neutralises_the_global_default_device` pins.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from utils import gpu as gpu_utils


@pytest.fixture(autouse=True)
def _clear_cache():
    gpu_utils._CROSS_ENCODER_CACHE.clear()
    yield
    gpu_utils._CROSS_ENCODER_CACHE.clear()


@pytest.fixture(autouse=True)
def _restore_default_device():
    """`set_default_device` is global; never leak it into another test."""
    yield
    torch.set_default_device(None)


class FakeEncoder:
    """Stands in for `sentence_transformers.CrossEncoder`.

    Refuses any device but CPU, the way a real load does when the card is full,
    and records the *ambient* default device it was constructed under -- which
    is what actually decides where transformers puts the weights.
    """

    instances: list["FakeEncoder"] = []

    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device
        self.ambient_device = torch.empty(0).device.type
        FakeEncoder.instances.append(self)
        if device is not None and str(device) != "cpu":
            raise torch.OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 2.07 GiB. GPU 0 has a "
                "total capacity of 22.06 GiB of which 1.10 GiB is free."
            )

    def to(self, device):  # a real CrossEncoder exposes this
        self.device = device
        return self


@pytest.fixture
def fake():
    FakeEncoder.instances = []
    return FakeEncoder


class TestAFullGpuFallsBackToCpu:
    def test_it_returns_a_usable_encoder_instead_of_raising(self, fake):
        encoder = gpu_utils.load_cross_encoder("bge-reranker-large", fake, "cuda")
        assert encoder is not None

    def test_the_encoder_is_on_the_cpu(self, fake):
        gpu_utils.load_cross_encoder("bge-reranker-large", fake, "cuda")
        assert fake.instances[-1].device == "cpu"

    def test_the_cpu_retry_neutralises_the_global_default_device(self, fake):
        """The whole reason a naive `device="cpu"` fallback does not work.

        With `set_default_device("cuda")` left standing, transformers resolves a
        CUDA device map and OOMs anyway -- confirmed against the real model
        before this fix was written.
        """
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

        gpu_utils.load_cross_encoder("bge-reranker-large", fake, "cuda")

        assert fake.instances[-1].ambient_device == "cpu", (
            "the CPU retry ran with a CUDA default device still installed; "
            "transformers will place the weights on the GPU and OOM again"
        )

    def test_it_is_not_moved_back_onto_the_full_gpu(self, fake):
        """The meta-tensor path moves the model to the GPU afterwards.

        That is right when the GPU merely refused one init path, and wrong here:
        the card had no room, so moving it back reintroduces the OOM we caught.
        """
        gpu_utils.load_cross_encoder("bge-reranker-large", fake, "cuda")
        assert fake.instances[-1].device == "cpu"

    def test_the_fallback_is_cached_so_the_oom_is_paid_once(self, fake):
        gpu_utils.load_cross_encoder("bge-reranker-large", fake, "cuda")
        first = len(fake.instances)

        gpu_utils.load_cross_encoder("bge-reranker-large", fake, "cuda")

        assert len(fake.instances) == first, "second call re-attempted the load"


class TestWhatMustStillHappen:
    def test_a_healthy_gpu_still_loads_on_the_gpu(self):
        """The fallback must not quietly demote every deployment to CPU."""

        class Healthy:
            def __init__(self, model_name, device=None):
                self.device = device

        encoder = gpu_utils.load_cross_encoder("m", Healthy, "cuda")
        assert encoder.device == "cuda"

    def test_the_meta_tensor_fallback_still_works(self):
        """Pre-existing behaviour; this fix must not displace it."""
        seen = []

        class MetaThenOk:
            def __init__(self, model_name, device=None):
                seen.append(device)
                self.device = device
                if device is not None and device != "cpu":
                    raise NotImplementedError("Cannot copy out of meta tensor")

            def to(self, device):
                self.device = device
                return self

        encoder = gpu_utils.load_cross_encoder("m", MetaThenOk, "cuda")
        assert seen == ["cuda", "cpu"]
        assert encoder.device == "cuda", "meta-tensor path still promotes to GPU"

    def test_an_unrelated_error_still_propagates(self):
        """Catching OOM must not turn into catching everything."""

        class Broken:
            def __init__(self, model_name, device=None):
                raise ValueError("model name is not a model")

        with pytest.raises(ValueError):
            gpu_utils.load_cross_encoder("m", Broken, "cuda")
