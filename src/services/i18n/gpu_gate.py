"""Whether background translation should wait: the GPU is wanted by foreground work.

Two signals. The first is exact: the shared Ollama client counts foreground generations in
this process (extraction runs here, in the API process). The second covers other processes
on the box (the extraction observer and telemetry services): the card's utilisation, read
from nvidia-smi. An unreadable card never blocks on its own -- the in-process count is the
guarantee; the utilisation check is a courtesy to the neighbours.
"""
from __future__ import annotations

import logging
import subprocess
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _nvidia_utilisation() -> Optional[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        return max(int(x) for x in out.split() if x.strip().isdigit())
    except Exception:
        return None


def _foreground_busy() -> bool:
    from src.services.ollama_client import foreground_busy
    return foreground_busy()


class GpuGate:
    def __init__(self, *, foreground: Callable[[], bool] = _foreground_busy,
                 utilisation: Callable[[], Optional[int]] = _nvidia_utilisation,
                 util_threshold: int = 30):
        self._foreground, self._utilisation, self._threshold = foreground, utilisation, util_threshold

    def busy(self) -> bool:
        if self._foreground():
            return True
        if self._threshold <= 0:
            return False
        util = self._utilisation()
        return util is not None and util >= self._threshold
