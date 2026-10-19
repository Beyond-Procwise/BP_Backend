"""Shared test doubles for the embedding pipeline."""
import hashlib
import math
import re
from typing import List, Sequence


class HashEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar unit vectors. No model, no GPU."""

    def __init__(self, dim: int = 1024, fail_times: int = 0):
        self.dim, self.fail_times, self.calls = dim, fail_times, 0

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("embedding service unavailable")
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))
