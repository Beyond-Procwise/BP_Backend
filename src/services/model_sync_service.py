"""Model sync service for BeyondProcwise/AgentNick.

Periodically updates the Modelfile with learned vendor patterns from
bp_vendor_extraction_profiles and rebuilds the model LOCALLY.

It does NOT push to any registry. The learned block bakes live customer supplier
names into the model's system prompt, and `ollama push` would send that off-box to
a shared registry — a data-governance exposure with no upside here. The sanctioned
channel for feeding learned vendor knowledge to the live model is the governed,
human-approved per-request hint path (bp_prompt 'extraction_vendor_hint' -> HINT_STORE
-> context_layer), which is additive, auditable and never leaves the box.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MODEL_NAME = "BeyondProcwise/AgentNick:latest"
MODELFILE_PATH = Path(__file__).resolve().parents[2] / "Modelfile"
SYNC_INTERVAL_HOURS = 6

# A learned "vendor" that is really an invoice id, a phone number, a fragment, or the token
# UNKNOWN is bad data from bp_vendor_extraction_profiles. Baking it into the system prompt
# teaches the model that "INV600784" or "800-531-8575" is a supplier — the opposite of
# accuracy. These are skipped at injection time (the upstream data-quality issue is separate).
_BAD_VENDOR_NAME = re.compile(
    r"^\s*$"                    # empty
    r"|^\d"                     # starts with a digit (phone/invoice numbers: 800-..., 600...)
    r"|^INV[\s\-]?\d"           # invoice id used as a name (INV600784)
    r"|INV\s*No"               # 'Sarah Thompson INV No: INV-2025-058'
    r"|UNKNOWN"                 # '... QUOTE TRADING LTD. UNKNOWN'
    r"|^TRADING\s+LTD\.?\s*$",  # bare fragment
    re.I,
)
# Values that appear in the currency column but are not currencies.
_NOT_A_CURRENCY = {"VAT", "TAX"}


def _clean_currency(currency: Optional[str]) -> Optional[str]:
    c = (currency or "").strip()
    if len(c) == 3 and c.isalpha() and c.upper() not in _NOT_A_CURRENCY:
        return c
    return None


class ModelSyncService:
    """Syncs learned vendor patterns into the AgentNick model periodically."""

    def __init__(self, agent_nick) -> None:
        self._agent_nick = agent_nick
        self._lock = threading.Lock()
        self._last_sync: Optional[datetime] = None

    def sync_model(self) -> bool:
        """Rebuild and push the model with latest vendor patterns.

        Returns True if sync succeeded.
        """
        with self._lock:
            try:
                # 1. Load learned vendor patterns from DB
                vendor_knowledge = self._load_vendor_knowledge()

                # 2. Read current Modelfile
                if not MODELFILE_PATH.exists():
                    logger.warning("Modelfile not found at %s", MODELFILE_PATH)
                    return False

                modelfile_content = MODELFILE_PATH.read_text(encoding="utf-8")

                # 3. Inject vendor knowledge into Modelfile
                updated_content = self._inject_vendor_knowledge(
                    modelfile_content, vendor_knowledge
                )

                # 4. Write updated Modelfile
                MODELFILE_PATH.write_text(updated_content, encoding="utf-8")

                # 5. Rebuild model
                result = subprocess.run(
                    ["ollama", "create", MODEL_NAME, "-f", str(MODELFILE_PATH)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    logger.error(
                        "ollama create failed: %s", result.stderr
                    )
                    return False
                logger.info("Model rebuilt locally: %s", MODEL_NAME)

                # Deliberately NO `ollama push`. The rebuilt model carries live customer
                # supplier names in its system prompt; pushing it would send that off-box.
                # The model stays local; governed per-request hints are the sanctioned path
                # for learned vendor knowledge (see module docstring).

                self._last_sync = datetime.now(timezone.utc)
                return True

            except Exception:
                logger.exception("Model sync failed")
                return False

    def _load_vendor_knowledge(self) -> str:
        """Load learned vendor patterns from bp_vendor_extraction_profiles."""
        try:
            conn = self._agent_nick.get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT supplier_name, doc_type, date_format_hint,
                               currency_hint, extraction_count
                        FROM proc.bp_vendor_extraction_profiles
                        WHERE extraction_count >= 2
                        ORDER BY extraction_count DESC
                        LIMIT 50
                        """
                    )
                    rows = cur.fetchall()
            finally:
                conn.close()

            if not rows:
                return ""

            lines = ["LEARNED VENDOR PATTERNS (from successful extractions):"]
            skipped = 0
            for name, doc_type, date_fmt, currency, count in rows:
                if not name or _BAD_VENDOR_NAME.search(str(name)):
                    skipped += 1
                    continue
                parts = [f"- {name} ({doc_type})"]
                if date_fmt:
                    parts.append(f"dates={date_fmt}")
                cur_ccy = _clean_currency(currency)
                if cur_ccy:
                    parts.append(f"currency={cur_ccy}")
                parts.append(f"extractions={count}")
                lines.append(", ".join(parts))
            if skipped:
                logger.info("model_sync: skipped %d invalid vendor rows (not real names)", skipped)

            # Also load common field patterns from extraction history
            try:
                conn = self._agent_nick.get_db_connection()
                try:
                    with conn.cursor() as cur:
                        # Get most common supplier names
                        cur.execute(
                            """
                            SELECT DISTINCT supplier_name
                            FROM proc.bp_supplier
                            WHERE supplier_name IS NOT NULL
                            ORDER BY supplier_name
                            LIMIT 100
                            """
                        )
                        suppliers = [r[0] for r in cur.fetchall() if r[0]]
                        if suppliers:
                            lines.append("")
                            lines.append(
                                "KNOWN SUPPLIERS (match against these): "
                                + ", ".join(suppliers[:50])
                            )
                finally:
                    conn.close()
            except Exception:
                pass

            return "\n".join(lines)

        except Exception:
            logger.debug("Failed to load vendor knowledge", exc_info=True)
            return ""

    # Matches EVERY learned block, both the current marker and the legacy
    # "=== LEARNED VENDOR PATTERNS (auto-updated) ===" one, from the header up to the next
    # section (=== ) or the end of the SYSTEM prompt ("""). The old regex only knew the current
    # marker, so it left the legacy block untouched and appended a new one beside it — that is
    # why the live Modelfile grew two learned sections with divergent, partly-garbage vendor
    # lists. Catching both variants makes injection genuinely idempotent.
    _LEARNED_BLOCK = re.compile(
        r"\n*=== LEARNED(?: VENDOR)? PATTERNS \(auto-updated\) ===.*?(?=\n=== |\s*\"\"\")",
        re.DOTALL,
    )

    def _inject_vendor_knowledge(
        self, modelfile: str, vendor_knowledge: str
    ) -> str:
        """Inject vendor knowledge into the Modelfile system prompt.

        Removes ALL existing learned sections (current + legacy markers) and writes exactly
        one, so running it twice yields byte-identical output — no duplication, no bloat.
        """
        # Strip every learned block first, regardless of vendor_knowledge, so a stale/corrupt
        # file is cleaned even on an empty load. Then collapse the blank lines the removal
        # leaves behind — without this the injection seam gains one blank line per run and the
        # output is not byte-stable (inject twice != inject once).
        modelfile = self._LEARNED_BLOCK.sub("\n", modelfile)
        modelfile = re.sub(r"\n{3,}", "\n\n", modelfile)

        if not vendor_knowledge:
            return modelfile

        marker = "=== QUALITY STANDARDS ==="
        # No leading newline: the marker is already preceded by one blank line, so the seam is
        # a fixed "\n\n=== LEARNED ...", the same on every run.
        injection = f"=== LEARNED PATTERNS (auto-updated) ===\n\n{vendor_knowledge}\n\n"

        if marker in modelfile:
            return modelfile.replace(marker, injection + marker)
        return modelfile.rstrip().rstrip('"').rstrip() + f"\n\n{injection}" + '"""\n'

    @property
    def last_sync(self) -> Optional[datetime]:
        return self._last_sync
