"""Label the harvested corpus and print the baseline.

Read-only against the document store; writes only the labelled set.
"""
import sys

from src.services.truth.baseline import format_report, score
from src.services.truth.build_set import build

CORPUS = "src/data/training/auto_collected_examples.jsonl"
OUT = "src/data/training/verified_examples.jsonl"


def main() -> int:
    summary = build(CORPUS, OUT)
    print(f"examples {summary['examples']}  from corpus {summary['with_source']}"
          f"  recovered {summary['recovered']}  unrecoverable {summary['unrecoverable']}"
          f"  malformed {summary['malformed']}")
    print()
    print(format_report(score(OUT)))
    return 0


if __name__ == "__main__":
    # An operator tool acts for the platform across every company; with row-level security on, a tool
    # that names no company would otherwise see no rows (D-47).
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from src.services.tenant_context import enter_system as _enter_system
    _enter_system()
    sys.exit(main())
