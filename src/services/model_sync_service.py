"""Model sync service for BeyondProcwise/AgentNick.

Periodically updates the Modelfile with learned vendor patterns from
bp_vendor_extraction_profiles, rebuilds the model, and pushes to the
Ollama registry.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MODEL_NAME = "BeyondProcwise/AgentNick:latest"
MODELFILE_PATH = Path(__file__).resolve().parents[2] / "Modelfile"
SYNC_INTERVAL_HOURS = 6


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
                logger.info("Model rebuilt successfully: %s", MODEL_NAME)

                # 6. Push to registry
                push_result = subprocess.run(
                    ["ollama", "push", MODEL_NAME],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if push_result.returncode != 0:
                    logger.warning(
                        "ollama push failed (non-critical): %s",
                        push_result.stderr,
                    )
                else:
                    logger.info("Model pushed to registry: %s", MODEL_NAME)

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
            for name, doc_type, date_fmt, currency, count in rows:
                parts = [f"- {name} ({doc_type})"]
                if date_fmt:
                    parts.append(f"dates={date_fmt}")
                if currency:
                    parts.append(f"currency={currency}")
                parts.append(f"extractions={count}")
                lines.append(", ".join(parts))

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

    def _inject_vendor_knowledge(
        self, modelfile: str, vendor_knowledge: str
    ) -> str:
        """Inject vendor knowledge into the Modelfile system prompt."""
        if not vendor_knowledge:
            return modelfile

        marker = "=== QUALITY STANDARDS ==="
        injection = f"\n=== LEARNED PATTERNS (auto-updated) ===\n\n{vendor_knowledge}\n\n"

        if marker in modelfile:
            return modelfile.replace(marker, injection + marker)
        else:
            # Append before the closing triple-quote
            return modelfile.rstrip().rstrip('"') + injection + '"""\n'

    @property
    def last_sync(self) -> Optional[datetime]:
        return self._last_sync
