"""The output-safety contract: the agent may KNOW the internals, it may never SAY them.

ProcWise holds a model of itself — how a document flows, which agent decides what, which
table a screen reads. That knowledge is the point: it is what lets the support agent answer
"why did my upload fail" instead of filing a ticket. It is also, verbatim, a description of
the backend. On 2026-07-14 the live agent was asked five hostile questions and gave up an
internal route, `proc.process_monitor`, an env var, a source directory, a class name and the
model name. It refused the SQL question — which is worse than failing all five, because it
looks safe until it isn't.

A prompt rule is a request. This is a guarantee.

Two checks, because there are two ways to leak:

  * IDENTIFIER — a table, a path, a route, an env var, a class, a stack trace, a credential.
    Grounded in the *running system* (the live `information_schema`, the live route table,
    the real `.env` keys) rather than a word list someone has to remember to update.

  * REGISTER — the answer names nothing at all and is still a description of the backend:
    "extraction is triggered before the upload finishes, so the watcher timed out". No
    identifier, total leak. The user's rule is that answers are explained in *product* terms
    — the screen, the button, the status — never in backend process-flow terms.

The second is the one that makes this more than a regex. It is also the one that can wreck a
good answer, because `invoice` and `supplier` are real tables in `proc` AND ordinary English
words a buyer says all day. So identifiers match on **qualified** forms only — `proc.invoice`,
`bp_invoice_trgt` — and a supplier legitimately called "Invoice Ltd" walks straight through.
That is not a nicety; a gate that mangles real answers is a gate that gets switched off.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


# What the user gets instead. Never a redacted fragment: `SELECT * FROM ███` still tells the
# reader there is a SQL layer and a table, which is exactly what we are hiding.
SAFE_REPLY = "I couldn't retrieve that. I've raised it with the team."

# Handed back to the model when its draft was mechanism-level. It already has the facts — it
# framed them wrong — so this is a re-frame, not a refusal.
RETRY_INSTRUCTION = (
    "Your previous answer described how the system works internally. Never do that. "
    "Answer again in terms of what the USER sees and does: the screen they are on, the "
    "button they press, the status they will see, what it means for their document or "
    "their deal. Do not mention databases, tables, files, code, endpoints, environment "
    "variables, pipelines, stages, triggers, queues, workers or models. Same facts, "
    "explained as the product — not as the architecture."
)


@dataclass(frozen=True)
class Violation:
    kind: str
    match: str  # For the LOG only. Never returned to a user — that is the whole point.

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Violation {self.kind}: {self.match!r}>"


# --------------------------------------------------------------------------------------
# Grounded vocabulary. Read from the real system, cached, and degraded safely: if the DB is
# unreachable we fall back to the structural patterns rather than opening the gate.
# --------------------------------------------------------------------------------------

_db_tables: Optional[Set[str]] = None
_route_paths: Set[str] = set()
_env_keys: Optional[Set[str]] = None


def register_routes(paths: Iterable[str]) -> None:
    """Called once at startup with the live FastAPI route table."""
    global _route_paths
    _route_paths = {p for p in paths if p and p != "/"}
    log.info("output_safety: %d internal routes registered", len(_route_paths))


def _tables() -> Set[str]:
    """Distinctive table names from the live `proc` schema.

    Only names that are *distinctive* — they contain an underscore, so `process_monitor` and
    `cat_product_mapping` are blocked bare, while `invoice` and `supplier` are not. Those two
    are real tables AND real English words; blocking them bare would gut every honest answer
    about an invoice. They are still caught in qualified form (`proc.invoice`) by pattern.
    """
    global _db_tables
    if _db_tables is not None:
        return _db_tables

    names: Set[str] = set()
    try:
        import psycopg2

        # The app's own Settings, not raw os.getenv. Reading the environment directly meant
        # that starting uvicorn without the .env exported produced a gate with an empty table
        # vocabulary — which fails *open* on exactly the class of leak (`process_monitor`,
        # unqualified) it exists to stop, and says nothing louder than a WARNING while it
        # does. Sharing the app's config makes that impossible: if the app can reach the DB,
        # so can the gate.
        from config.settings import settings

        conn = psycopg2.connect(
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            connect_timeout=5,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'proc'"
                )
                names = {r[0].lower() for r in cur.fetchall() if "_" in r[0]}
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        # No DB is not a reason to start leaking. The qualified patterns below still hold.
        log.warning("output_safety: table vocabulary unavailable (%s); patterns only", exc)

    _db_tables = names
    log.info("output_safety: %d distinctive table names loaded", len(names))
    return names


def _env_var_keys() -> Set[str]:
    """Config key names, from `.env` — not `os.environ`, which is full of PATH and HOME."""
    global _env_keys
    if _env_keys is not None:
        return _env_keys

    keys: Set[str] = set()
    try:
        for line in (_REPO_ROOT / ".env").read_text().splitlines():
            m = re.match(r"^([A-Z][A-Z0-9_]{3,})\s*=", line.strip())
            if m:
                keys.add(m.group(1))
    except Exception:  # noqa: BLE001
        pass
    _env_keys = keys
    return keys


# --------------------------------------------------------------------------------------
# IDENTIFIER patterns. Qualified forms only.
# --------------------------------------------------------------------------------------

_PATTERNS: Sequence[tuple[str, re.Pattern[str]]] = [
    # proc.<anything>, and the bp_ / _stg / _trgt naming conventions, wherever they appear.
    ("db_table", re.compile(r"\bproc\.[a-z_][a-z0-9_]*", re.I)),
    ("db_table", re.compile(r"\bbp_[a-z0-9_]+", re.I)),
    ("db_table", re.compile(r"\b[a-z0-9_]+_(?:stg|trgt)\b", re.I)),
    ("db_table", re.compile(r"\binformation_schema\b", re.I)),
    # SQL as prose. Needs the pairing — a bare "select" is an ordinary English verb.
    ("sql", re.compile(r"\bSELECT\b[\s\S]{0,200}?\bFROM\b", re.I)),
    ("sql", re.compile(r"\b(?:INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|"
                       r"CREATE\s+TABLE|ALTER\s+TABLE|DROP\s+TABLE)\b", re.I)),
    ("sql", re.compile(r"\b(?:INNER|LEFT|RIGHT|OUTER)\s+JOIN\b", re.I)),
    # Source paths and modules.
    ("file_path", re.compile(r"\b(?:src|tests|scripts|services|agents|api)/[\w./-]+")),
    ("file_path", re.compile(r"\b[\w./-]+\.(?:py|yaml|yml|json|sql|env|cfg|ini)\b")),
    ("file_path", re.compile(r"(?:/home/|/var/|/etc/|/opt/|/usr/|[A-Z]:\\)[\w./\\-]+")),
    # Env / config keys. Two or more underscores is the tell — prose does not write
    # UPLOAD_FILE_WAIT_TIMEOUT_S. Single-underscore keys we actually ship (DB_HOST) are
    # matched separately against the real `.env`, below.
    ("env_var", re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+){2,}\b")),
    # Internal HTTP routes.
    ("route", re.compile(r"(?:\b(?:GET|POST|PUT|PATCH|DELETE)\s+)?/[a-z0-9][a-z0-9-]*"
                         r"(?:/[a-z0-9{}_-]+)+", re.I)),
    # Code identifiers: CamelCase things that are plainly classes, and function calls.
    ("code_identifier", re.compile(
        r"\b[A-Z][a-zA-Z0-9]*(?:Agent|Service|Engine|Pipeline|Manager|Handler|Router|"
        r"Repository|Adapter|Worker|Extractor|Resolver|Orchestrator)\b")),
    ("code_identifier", re.compile(r"\b[a-z_][a-z0-9_]*\(\s*\)")),
    # Source lines pasted verbatim. Anchored to the start of a line: an unanchored
    # `from\s+\w` matches the English "from the Documents screen", which is exactly the
    # sentence we most want to keep.
    ("code_identifier", re.compile(r"^\s*(?:def|class|import|from)\s+\w+", re.M)),
    # Models and infrastructure.
    ("model_name", re.compile(
        r"\b(?:AgentNick|Ollama|ollama|qwen[\w.:-]*|llama[\w.:-]*|GGUF|"
        r"num_predict|keep_alive|temperature\s*=)\b")),
    ("infra", re.compile(
        r"\b(?:psycopg2|Neo4j|neo4j|Qdrant|qdrant|Cypher|uvicorn|FastAPI FastAPI|"
        r"boto3|SQLAlchemy|pydantic|Postgres|PostgreSQL|S3 bucket|presigned)\b")),
    # Stack traces.
    ("stack_trace", re.compile(r"Traceback \(most recent call last\)")),
    ("stack_trace", re.compile(r'File "[^"]+", line \d+')),
    ("stack_trace", re.compile(r"\b\w+(?:Error|Exception)\b\s*:", re.I)),
    ("stack_trace", re.compile(r"\b\w+\.errors\.\w+")),
    # Credentials and connection strings.
    ("credential", re.compile(r"\b(?:postgres(?:ql)?|mysql|redis|mongodb|amqp)://\S+", re.I)),
    ("credential", re.compile(
        r"\b(?:password|passwd|api[_-]?key|secret|token|credential)\s*[=:]\s*\S+", re.I)),
    ("credential", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}:\d{2,5}\b")),
]


# --------------------------------------------------------------------------------------
# REGISTER — backend process-flow vocabulary. Prose only.
#
# These describe the *mechanism*. The product vocabulary a user actually sees — the screens,
# the statuses (Running, Extracted), the fields — is deliberately absent, so a real answer
# ("you'll see the status change from Running to Extracted") passes.
# --------------------------------------------------------------------------------------

_MECHANISM = re.compile(
    r"\b("
    r"pipeline|pipelines"
    r"|extraction\s+(?:stage|pass|layer|engine|logic|process)"
    r"|(?:database|db)\s+trigger|triggered\s+by\s+the|fires?\s+the\s+trigger"
    r"|watcher|poller|listener\s+process"
    r"|presign\w*|pre-?signed"
    r"|orchestrat\w+|blackboard|workflow\s+engine|DAG|graph\s+node"
    r"|promotion\s+gate|promoted?\s+to\s+(?:the\s+)?(?:final|target)\s+table|staging\s+table"
    r"|regex\s+(?:pass|layer|rule)|engineered\s+(?:pass|layer)|AI\s+judge"
    r"|embedding|vector\s+(?:store|db|database)|chunk\w*\s+(?:the|into)"
    r"|system\s+prompt|prompt\s+template|tool[\s-]call\w*|inference|LLM|token\s+limit"
    r"|middleware|cron\b|scheduler|background\s+worker|worker\s+process|job\s+queue|enqueue\w*"
    r"|message\s+queue|celery"
    r"|schema\s+(?:migration|change)|foreign\s+key|primary\s+key|index\s+on"
    r"|endpoint|API\s+call|REST\b|payload|request\s+body|response\s+body"
    r"|microservice|backend\s+service|the\s+backend\b|server[\s-]side"
    r"|codebase|source\s+code|repository|deployment|container|docker"
    r")\b",
    re.I,
)


def _looks_like_route(fragment: str) -> bool:
    """Distinguish an internal API route from ordinary prose that happens to hold a slash.

    "and/or", "24/7", "km/h" are not routes. A route is registered, or it is a path with at
    least two segments that is not just two words joined by a slash.
    """
    path = re.sub(r"^(?:GET|POST|PUT|PATCH|DELETE)\s+", "", fragment, flags=re.I).strip()
    if path in _route_paths:
        return True
    if any(path.startswith(known.split("{")[0].rstrip("/")) for known in _route_paths if known):
        return True
    # Not in the live table — it may be the gateway's route, or one the model invented (it
    # once volunteered "there is NO /spendiq/invoices endpoint", which is a leak that also
    # happens to be a map). The leading slash plus two segments is already the tell: prose
    # writes "and/or" and "24/7", never "/spendiq/invoices".
    segments = [s for s in path.split("/") if s]
    return path.startswith("/") and len(segments) >= 2


def _identifier_violations(text: str) -> List[Violation]:
    out: List[Violation] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            frag = m.group(0)
            if kind == "route" and not _looks_like_route(frag):
                continue
            out.append(Violation(kind, frag))

    # Config keys we actually ship that the pattern above misses because they only have one
    # underscore (DB_HOST, S3_BUCKET_NAME). Read from the real `.env`, so this stays true.
    for key in _env_var_keys():
        if re.search(rf"\b{re.escape(key)}\b", text):
            out.append(Violation("env_var", key))

    lowered = text.lower()
    for name in _tables():
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            out.append(Violation("db_table", name))
    return out


def inspect(text: str, *, prose: bool = True) -> List[Violation]:
    """What, if anything, would this text leak?

    ``prose=True`` also applies the register check — use it for anything the user reads as a
    sentence. ``prose=False`` is for data values (a supplier name, a line description), where
    only identifiers matter and register words are just words.
    """
    if not text or not isinstance(text, str):
        return []

    found = _identifier_violations(text)
    if prose:
        for m in _MECHANISM.finditer(text):
            found.append(Violation("mechanism", m.group(0)))
    return found


def is_safe(text: str, *, prose: bool = True) -> bool:
    return not inspect(text, prose=prose)


def enforce(text: str, *, prose: bool = True, where: str = "") -> str:
    """Return the text, or — if it would leak anything — the safe reply instead.

    The whole answer goes, not the offending clause. A half-answer with a hole in it still
    tells the reader what shape the hole is.
    """
    violations = inspect(text, prose=prose)
    if not violations:
        return text

    log.warning(
        "output_safety: BLOCKED %s | kinds=%s | matched=%s",
        where or "response",
        sorted({v.kind for v in violations}),
        [v.match for v in violations][:8],
    )
    return SAFE_REPLY


# --------------------------------------------------------------------------------------
# Payload scrubbing — the boundary backstop.
# --------------------------------------------------------------------------------------

# Fields a user reads as a sentence. These get both checks.
PROSE_FIELDS = {
    "reply", "answer", "summary", "summary_text", "rationale", "justification", "message",
    "detail", "error", "body", "reason", "decision_reason", "note", "next_question",
    "recommendation", "narrative", "processing_issue", "delta", "text", "content",
    "explanation", "description_text", "prompt", "guidance", "advice",
}


def scrub_payload(obj: Any, *, where: str = "") -> Any:
    """Walk a response body and make it safe.

    Prose fields get the full contract. Every other string still gets the identifier check —
    a table name in some field nobody thought about is still a leak — but not the register
    check, so a line item reading "Pipeline fittings, 40mm" is left exactly as it is.
    """
    if isinstance(obj, dict):
        return {k: _scrub_value(k, v, where) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_payload(v, where=where) for v in obj]
    if isinstance(obj, str):
        return enforce(obj, prose=False, where=where)
    return obj


def _scrub_value(key: str, value: Any, where: str) -> Any:
    if isinstance(value, (dict, list)):
        return scrub_payload(value, where=where)
    if isinstance(value, str):
        is_prose = key.lower() in PROSE_FIELDS
        return enforce(value, prose=is_prose, where=f"{where}.{key}" if where else key)
    return value
