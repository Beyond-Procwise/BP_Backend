"""CI gate: every formula must reproduce every one of its golden vectors.

Importing the definitions package is itself the check --- a formula whose
maths, contract or expectations have drifted refuses to register. This script
exists so CI fails with a LIST of what broke rather than an ImportError naming
whichever module happened to load first.

    ./.venv/bin/python -m scripts.formulas.check_goldens

Exit status 0 = every vector reproduces. 1 = something drifted.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


def main() -> int:
    try:
        from src.services.formulas import REGISTRY, verify_all
        from src.services.formulas import definitions  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        # The traceback is the useful half: an import failure here usually names
        # the exact formula whose vector stopped reproducing.
        import traceback

        traceback.print_exc()
        print(f"FAIL: the formula definitions did not import: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    failures = verify_all()
    if failures:
        print(f"FAIL: {len(failures)} formula(s) no longer reproduce their golden "
              f"vectors:\n", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        print(
            "\nIf the change to the maths is intended, update the vectors AND bump "
            "the formula's `version`. A vector edited without a version bump makes "
            "every stored evaluation record a lie about what produced it.",
            file=sys.stderr,
        )
        return 1

    vectors = sum(len(spec.golden) for spec in REGISTRY.values())
    print(f"OK: {len(REGISTRY)} formulas, {vectors} golden vectors, all reproducing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
