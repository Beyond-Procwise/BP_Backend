"""Per-failure diagnostic capture for LLM extraction calls.

When a structured-extraction call fails to parse the LLM's JSON response
(or the LLM call itself errors), we want to capture enough context to
diagnose the underlying issue offline without spamming the journal:

  - prompt size, prompt head, prompt tail
  - raw model response (full body)
  - parse error / HTTP error
  - call-site (model, doc_type, file_path)

Each failure writes one JSON file to:

    artifacts/llm_failures/<YYYYMMDDTHHMMSS>_<sanitised-doc>_<site>.json

Subsequent reads from disk are independent of journald rotation.

The capture is intentionally low-cost: a single file write, lazy
directory creation, and a 1MB cap on response bytes to prevent runaway
disk use on a degenerate model.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


__all__ = ["capture_llm_failure"]


_DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts" / "llm_failures"
_lock = threading.Lock()
_RAW_CAP = 1_000_000  # 1 MB
_HEAD_TAIL = 800       # head/tail snippet length


def _sanitise(name: str, max_len: int = 60) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return safe[:max_len] or "anon"


def capture_llm_failure(
    *,
    site: str,
    prompt: Optional[str] = None,
    raw_response: Optional[str] = None,
    error: Optional[BaseException] = None,
    doc_type: Optional[str] = None,
    file_path: Optional[str] = None,
    model: Optional[str] = None,
    extra: Optional[dict] = None,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Write a single JSON capture file describing one LLM failure.

    Returns the path written, or None if the capture was skipped (which
    must never propagate — diagnostic capture is best-effort).
    """
    try:
        out_dir = Path(output_dir) if output_dir else _DEFAULT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        token = _sanitise(os.path.basename(file_path or "anon"))
        site_token = _sanitise(site, max_len=24)
        path = out_dir / f"{ts}_{token}_{site_token}.json"

        prompt_str = prompt or ""
        prompt_len = len(prompt_str)
        prompt_head = prompt_str[:_HEAD_TAIL]
        prompt_tail = prompt_str[-_HEAD_TAIL:] if prompt_len > _HEAD_TAIL else ""

        raw_str = raw_response or ""
        raw_len = len(raw_str)
        raw_capped = raw_str[:_RAW_CAP]

        record: dict[str, Any] = {
            "ts": ts,
            "site": site,
            "model": model,
            "doc_type": doc_type,
            "file_path": file_path,
            "prompt_chars": prompt_len,
            "prompt_head": prompt_head,
            "prompt_tail": prompt_tail,
            "response_chars": raw_len,
            "response": raw_capped,
            "response_truncated": raw_len > _RAW_CAP,
            "error_type": type(error).__name__ if error is not None else None,
            "error_message": str(error)[:2000] if error is not None else None,
            "extra": extra or {},
        }
        with _lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, default=str)
        logger.warning(
            "[llm-diag] captured failure → %s "
            "(site=%s prompt=%d response=%d error=%s)",
            path, site, prompt_len, raw_len,
            type(error).__name__ if error is not None else "",
        )
        return path
    except Exception:
        # Diagnostic capture is best-effort — never let it fail the caller.
        logger.debug("[llm-diag] failed to write capture", exc_info=True)
        return None
