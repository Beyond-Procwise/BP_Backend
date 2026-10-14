"""Run a formula command with the governed limits served from the test seed.

CI has no database. Several formulas read a governed limit while their golden
vectors run, and a missing limit RAISES by design (services/governed_limits.py),
so importing the definitions fails before a single vector is checked. Locally,
against a database carrying the limit rows, the same command passes.

This serves the limits from ``tests.conftest.GOVERNED_LIMIT_SEED`` instead. That
seed is not a new copy: it is the one the test suite already uses, and
``tests/governance/test_governed_limits.py`` fails if it disagrees with the live
rows. Reusing it keeps CI checking the values the product runs on.

The engine is swapped before the target is imported, not in a fixture, because
the formula tests import the definitions at collection time -- before any
fixture has run.

    python -m scripts.formulas.seeded_limits scripts.formulas.check_goldens
    python -m scripts.formulas.seeded_limits src.services.formulas.inventory
    python -m scripts.formulas.seeded_limits pytest tests/services/formulas -q
"""
from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


def _seed_governed_limits() -> None:
    from tests.conftest import _SeededPolicyEngine

    # The module is importable under two names (src/ is on the path too), and
    # each name is a separate module object with its own _engine. Patch both, or
    # a formula importing the other spelling still finds no limits.
    for name in ("src.services.governed_limits", "services.governed_limits"):
        module = importlib.import_module(name)
        module.reset_cache()
        module._engine = lambda: _SeededPolicyEngine()


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    _seed_governed_limits()
    target = sys.argv[1]
    sys.argv = [target, *sys.argv[2:]]
    runpy.run_module(target, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
