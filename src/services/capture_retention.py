"""Age out the on-disk captures of document and agent content.

Two directories accumulate content lifted out of customer documents, outside the
database and outside every retention rule the product has:

  ``artifacts/llm_failures/``  prompt head, prompt tail and the model's raw
                               response, written by
                               ``llm_diagnostics.capture_llm_failure`` on every
                               failed LLM call.
  ``data/conversations/``      agent context and response, written by
                               ``models.context_trainer``.

Neither was ever deleted. A right-to-erasure request could not reach them, and
the honest answer to "how long do you keep this" was "forever".

DELETING BY AGE IS THE DANGEROUS PART, so the shape here is deliberately narrow:

  * Only the two directories named in ``CAPTURE_DIRS`` may be purged. Passing
    anything else raises rather than proceeding — an age-based delete pointed at
    a source tree is a data-loss incident, not a bug to find later.
  * Files only. Directories are left alone.
  * Symlinks are removed as links, by their own presence rather than their
    target's age. The target being safe is not this module's doing —
    ``Path.unlink()`` removes a link and can never reach what it points at. What
    the branch buys is that a link to a RECENT file elsewhere still ages out,
    instead of ``stat()`` following it, reading a fresh timestamp and keeping a
    stale pointer into somebody else's tree forever.
  * A retention of zero or less raises. It would mean "delete everything", and
    nobody types that on purpose.

This does not solve retention generally — the database, S3 and the external
vector index all still keep what they keep. It closes the two places where
document text was accumulating on local disk with no policy at all.
"""
from __future__ import annotations

import logging
from src.services.governed_limits import limit as _governed_limit
import os
import time
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent

# The only directories this module will delete from. Membership is checked by
# resolved path, so a caller cannot reach a third directory via `..`.
CAPTURE_DIRS: tuple[Path, ...] = (
    _ROOT / "artifacts" / "llm_failures",
    _ROOT / "data" / "conversations",
)

DEFAULT_RETENTION_DAYS = 30


def retention_days() -> int:
    """Days to keep a capture. ``CAPTURE_RETENTION_DAYS`` overrides the default.

    An unparseable or non-positive value falls back to the default rather than
    to zero: a typo in an environment variable must not become "delete
    everything on the next scheduler tick".
    """
    # AutonomousOperationPolicy (P9): how long captured data is kept is a
    # data-protection commitment, not a tuning knob.
    raw = os.getenv("CAPTURE_RETENTION_DAYS", "").strip()
    if not raw:
        return _governed_limit("autonomous_operation", "capture_retention_days",
                               cast=int)
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "CAPTURE_RETENTION_DAYS=%r is not a number; using %d days",
            raw, DEFAULT_RETENTION_DAYS,
        )
        return DEFAULT_RETENTION_DAYS
    if value <= 0:
        logger.warning(
            "CAPTURE_RETENTION_DAYS=%d is not positive; using %d days",
            value, DEFAULT_RETENTION_DAYS,
        )
        return DEFAULT_RETENTION_DAYS
    return value


def _is_known_capture_dir(directory: Path) -> bool:
    try:
        resolved = directory.resolve()
    except OSError:
        return False
    for known in CAPTURE_DIRS:
        try:
            if resolved == known.resolve():
                return True
        except OSError:
            continue
    return False


def purge(directory: Path | str, *, max_age_days: int | None = None) -> int:
    """Delete captures in ``directory`` older than ``max_age_days``. Returns the count.

    ``max_age_days`` is keyword-only so a positional second argument cannot be
    read as something else at a call site.
    """
    directory = Path(directory)
    days = retention_days() if max_age_days is None else int(max_age_days)

    if days <= 0:
        raise ValueError(
            f"max_age_days must be positive, got {days}: a non-positive "
            f"retention would delete every capture"
        )
    if not _is_known_capture_dir(directory):
        raise ValueError(
            f"{directory} is not a known capture directory; purge only removes "
            f"files from {[str(d) for d in CAPTURE_DIRS]}"
        )
    if not directory.is_dir():
        return 0

    cutoff = time.time() - days * 86400
    removed = 0
    for entry in directory.iterdir():
        # is_file() follows symlinks, so a link to a live file elsewhere would
        # look like an ordinary capture. Check the link itself.
        if entry.is_symlink():
            try:
                entry.unlink()   # removes the link, never the target
                removed += 1
            except OSError:
                logger.debug("could not remove symlink %s", entry, exc_info=True)
            continue
        if not entry.is_file():
            continue
        try:
            if entry.stat().st_mtime >= cutoff:
                continue
            entry.unlink()
            removed += 1
        except OSError:
            # A capture that cannot be read or removed is not worth failing the
            # sweep for; the next run will try again.
            logger.debug("could not remove capture %s", entry, exc_info=True)

    if removed:
        logger.info("capture retention: removed %d file(s) from %s older than "
                    "%d days", removed, directory, days)
    return removed


def purge_all(dirs: Iterable[Path] | None = None) -> dict[str, int]:
    """Purge every capture directory. Returns {directory: files removed}."""
    out: dict[str, int] = {}
    for directory in (dirs if dirs is not None else CAPTURE_DIRS):
        try:
            out[str(directory)] = purge(directory)
        except Exception:
            logger.exception("capture retention failed for %s", directory)
            out[str(directory)] = 0
    return out
