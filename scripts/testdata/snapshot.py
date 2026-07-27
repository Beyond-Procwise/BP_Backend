"""Snapshot and restore the test databases.

A full build takes over an hour, most of it deal assignment. Resetting to a
pristine base between demos should take a minute, not an afternoon, so the
built state is dumped once and restored on demand.

    python -m scripts.testdata.snapshot save    --label demo-base
    python -m scripts.testdata.snapshot restore --label demo-base
    python -m scripts.testdata.snapshot list

Restoring drops and recreates the target, so it runs through the same refusal
that guards every other destructive operation here: a live database name cannot
be a restore target, with or without a flag.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from config.settings import Settings
from scripts.testdata.db import create_database
from scripts.testdata.guards import assert_safe_target

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = ROOT / "backups" / "testdata"

# Which databases a snapshot covers. Both are captured together: restoring one
# without the other leaves the crosswalk pointing at suppliers that are not
# there, which is worse than no snapshot at all.
PAIRS: tuple[tuple[str, str], ...] = (
    ("bp_testdb", "bp"),
    ("uicanvas_test", "uicanvas"),
)

_LABEL = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class SnapshotError(RuntimeError):
    """Raised when a snapshot cannot be taken or restored."""


@dataclass(frozen=True)
class Snapshot:
    label: str
    path: Path
    database: str
    size_bytes: int
    taken_at: datetime


def _check_label(label: str) -> str:
    """Labels become filenames, so they may not wander out of the directory."""
    if not _LABEL.match(label or ""):
        raise SnapshotError(
            f"invalid snapshot label {label!r}: use lower-case letters, digits, "
            f"dot, dash or underscore, up to 64 characters"
        )
    return label


def _pg_env(settings: Settings) -> dict[str, str]:
    env = dict(os.environ)
    env["PGPASSWORD"] = settings.db_password
    return env


def path_for(label: str, database: str) -> Path:
    return SNAPSHOT_DIR / f"{_check_label(label)}.{database}.dump"


def save(label: str, pairs: Sequence[tuple[str, str]] = PAIRS) -> list[Snapshot]:
    """Dump each database to its own custom-format file."""
    _check_label(label)
    settings = Settings()
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    taken: list[Snapshot] = []
    for database, _alias in pairs:
        target = path_for(label, database)
        result = subprocess.run(
            [
                "pg_dump", "--format=custom", "--no-owner", "--no-privileges",
                "-h", settings.db_host, "-p", str(settings.db_port),
                "-U", settings.db_user, "-d", database,
                "-f", str(target),
            ],
            env=_pg_env(settings), capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise SnapshotError(f"pg_dump of {database} failed:\n{result.stderr}")
        stat = target.stat()
        taken.append(
            Snapshot(
                label=label, path=target, database=database,
                size_bytes=stat.st_size,
                taken_at=datetime.fromtimestamp(stat.st_mtime),
            )
        )
    return taken


def restore(label: str, pairs: Sequence[tuple[str, str]] = PAIRS) -> list[str]:
    """Drop, recreate and reload each database from its snapshot."""
    _check_label(label)
    settings = Settings()

    missing = [
        str(path_for(label, database))
        for database, _ in pairs
        if not path_for(label, database).exists()
    ]
    if missing:
        raise SnapshotError("no such snapshot:\n  " + "\n  ".join(missing))

    restored: list[str] = []
    for database, _alias in pairs:
        # Same refusal as everywhere else: a live name is not a restore target.
        assert_safe_target(database)
        create_database(database, drop_first=True)

        result = subprocess.run(
            [
                "pg_restore", "--no-owner", "--no-privileges",
                "-h", settings.db_host, "-p", str(settings.db_port),
                "-U", settings.db_user, "-d", database,
                str(path_for(label, database)),
            ],
            env=_pg_env(settings), capture_output=True, text=True,
        )
        # pg_restore reports 1 for warnings it recovered from, which a dump of a
        # cloned schema reliably produces (extensions, comments it cannot own).
        if result.returncode not in (0, 1):
            raise SnapshotError(f"pg_restore of {database} failed:\n{result.stderr}")
        restored.append(database)
    return restored


def available() -> list[Snapshot]:
    if not SNAPSHOT_DIR.exists():
        return []
    found: list[Snapshot] = []
    for path in sorted(SNAPSHOT_DIR.glob("*.dump")):
        label, _, rest = path.name.partition(".")
        database = rest.rsplit(".dump", 1)[0]
        stat = path.stat()
        found.append(
            Snapshot(
                label=label, path=path, database=database,
                size_bytes=stat.st_size,
                taken_at=datetime.fromtimestamp(stat.st_mtime),
            )
        )
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.testdata.snapshot")
    parser.add_argument("action", choices=("save", "restore", "list"))
    parser.add_argument("--label", default="demo-base")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    try:
        if args.action == "save":
            for snapshot in save(args.label):
                size = snapshot.size_bytes / 1_048_576
                print(f"saved {snapshot.database} -> {snapshot.path} ({size:.1f} MB)")
        elif args.action == "restore":
            for database in restore(args.label):
                print(f"restored {database} from snapshot {args.label!r}")
        else:
            snapshots = available()
            if not snapshots:
                print(f"no snapshots in {SNAPSHOT_DIR}")
            for snapshot in snapshots:
                size = snapshot.size_bytes / 1_048_576
                print(
                    f"{snapshot.label:24} {snapshot.database:16} "
                    f"{size:8.1f} MB  {snapshot.taken_at:%Y-%m-%d %H:%M}"
                )
    except SnapshotError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
