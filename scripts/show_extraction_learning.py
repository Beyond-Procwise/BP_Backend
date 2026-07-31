"""What the pipeline has learned, and what it is still guessing at.

    set -a; . ./.env; set +a
    PYTHONPATH=.:src ./venv/bin/python scripts/show_extraction_learning.py
"""
from __future__ import annotations

from src.services.extraction.pattern_registry import PatternRegistry
from src.services.extraction_feedback.accuracy import MIN_SAMPLE, load_accuracy
from src.services.db import get_conn


def main() -> int:
    accuracy = load_accuracy()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT doc_type, field_name, COALESCE(pattern_name, source),
                              COUNT(*),
                              COUNT(*) FILTER (WHERE verdict IN ('confirmed','rejected'))
                         FROM proc.bp_extraction_verdict
                        GROUP BY 1,2,3 ORDER BY 4 DESC""")
        counts = cur.fetchall()

    if not counts:
        print("No verdicts recorded yet — every reader is still on its hand-set prior.")
        print(f"A reader needs {MIN_SAMPLE} judgements before its measured rate is used.")
        return 0

    print(f"{'doc_type':<10} {'field':<20} {'reader':<26} {'n':>4} {'measured':>9} "
          f"{'prior':>6}  in force")
    for doc_type, field, reader, n, agreed in counts:
        rate = accuracy.get((doc_type, field, reader))
        prior = None
        try:
            reg = PatternRegistry(doc_type)
            prior = next((p.prior_confidence for p in reg._by_field.get(field, [])
                          if p.name == reader), None)
        except Exception:
            pass
        in_force = "yes" if (rate is not None and prior is not None and rate < prior) else "no"
        print(f"{doc_type:<10} {field:<20} {reader:<26} {n:>4} "
              f"{(f'{rate:.2f}' if rate is not None else '—'):>9} "
              f"{(f'{prior:.2f}' if prior is not None else '—'):>6}  {in_force}"
              f"{'' if n >= MIN_SAMPLE else f'  (needs {MIN_SAMPLE - n} more)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
