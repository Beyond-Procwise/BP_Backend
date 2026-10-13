"""Every prompt render in a run, from its run id.

    python scripts/prompt_renders_for_run.py <run_id>           # what was rendered
    python scripts/prompt_renders_for_run.py <run_id> --text    # and the text itself

--text prints prompt wording with customer values filled in. Use it only where
that may be shown. A render older than the retention window has no text left;
its hash, prompt and version are still listed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from services.prompt_engine import render  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id")
    parser.add_argument("--text", action="store_true", help="print the rendered text too")
    args = parser.parse_args()

    uses = render.renders_for_run(args.run_id)
    if not uses:
        print(f"no renders recorded for run {args.run_id}")
        return 1
    for use in uses:
        print(f"{use['used_at']}  step {use['step_id']} attempt {use['attempt']}  "
              f"{use['prompt_key']} v{use['version']}  {use['render_hash']}  "
              f"decision {use['decision_id']}")
        try:
            stored = render.get_render(use["render_hash"])
        except render.RenderNotFound:
            print("    (text aged out)")
            continue
        m = stored["manifest"]
        print(f"    fragments {m['fragments']}  obligations {m['obligations_applied']}  "
              f"{m['token_count']} tokens")
        if args.text:
            print("    ---")
            for line in stored["rendered"]["instructions"].splitlines():
                print(f"    {line}")
            print("    ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
