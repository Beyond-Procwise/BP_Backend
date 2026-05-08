"""Dump the parsed-text view of a production PDF.

Usage: python scripts/dump_parsed_text.py <file_path_in_process_monitor>

Prints the structural extractor's view of the document — the same
representation that gets fed to the LLM salvage prompt. If line items
are NOT visible in this output, the parser is the bottleneck. If line
items ARE visible but the LLM still drops them, the prompt or the
column hints are the problem.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from agents.base_agent import AgentNick  # noqa: E402
from services.direct_extraction_service import DirectExtractionService  # noqa: E402
from src.services.extraction_v2.fingerprint import compute_fingerprint  # noqa: E402
from src.services.structural_extractor.parsing import parse  # noqa: E402


def main(file_path: str) -> int:
    nick = AgentNick()
    svc = DirectExtractionService(nick)
    text, file_bytes = svc._get_document_text(file_path)
    if not file_bytes:
        print(f"No bytes for {file_path}", file=sys.stderr)
        return 1

    filename = os.path.basename(file_path)
    doc = parse(file_bytes, filename)

    print(f"=== {filename} ===")
    print(f"  format:       {doc.source_format}")
    print(f"  pages:        {doc.pages_or_sheets}")
    print(f"  tables:       {len(doc.tables or [])}")
    print(f"  fingerprint:  {compute_fingerprint(doc)[:8]}")
    print(f"  full_text len: {len(doc.full_text or '')}")

    print("\n=== full_text (first 4000 chars) ===")
    print((doc.full_text or "")[:4000])
    if len(doc.full_text or "") > 4000:
        print(f"\n[... truncated, {len(doc.full_text) - 4000} more chars ...]")

    print("\n=== tables ===")
    for i, t in enumerate(doc.tables or []):
        rows = t.rows or []
        print(f"\n  -- table[{i}] rows={len(rows)} cols={len(rows[0]) if rows else 0} --")
        for j, row in enumerate(rows[:30]):
            cells = [str(c)[:60] for c in row]
            print(f"    row[{j}]: {cells}")
        if len(rows) > 30:
            print(f"    [... {len(rows) - 30} more rows ...]")

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/dump_parsed_text.py <file_path>", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
