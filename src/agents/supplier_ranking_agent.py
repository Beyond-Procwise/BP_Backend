from __future__ import annotations

import json
import logging
import math
import multiprocessing
import re
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

from utils.gpu import configure_gpu
from utils.instructions import parse_instruction_sources
from utils.db import read_sql_compat
from utils.reference_loader import load_reference_dataset
from services.supplier_relationship_service import SupplierRelationshipService
from .base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus

logger = logging.getLogger(__name__)

# Configure GPU and suppress pandas warnings
configure_gpu()
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pandas only supports SQLAlchemy connectable",
        category=UserWarning,
    )


def _json_safe(v: Any) -> Any:
    """Coerce a value to a JSON-native type (str, int, float, bool, None, list, dict).

    Handles the common cases that arise when DataFrame rows are turned into
    plain Python dicts:

    * pandas NA / NaT / pd.NA  → None
    * numpy NaN / numpy float scalars → None (if NaN) or native float
    * numpy integer scalars     → native int
    * numpy bool_               → native bool
    * pandas Timestamp / any object with .isoformat() → ISO string
    * native float NaN          → None
    """
    if v is None:
        return None
    # numpy bool_ must be checked before float/int because np.bool_ inherits
    # from Python int on some numpy builds.
    if isinstance(v, np.bool_):
        return bool(v)
    # numpy/pandas floating (np.float64 inherits from Python float, so we
    # must coerce to native float rather than returning the subclass).
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else f
    # numpy integer
    if isinstance(v, np.integer):
        return int(v)
    # pandas NA / NaT (must come after numpy scalar checks so the cheap
    # isinstance paths above catch the common numpy scalars first).
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    # Timestamps and date-like objects with isoformat
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _parse_payment_terms_days(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if not s:
        return None

    if any(keyword in s for keyword in ("immediate", "due on receipt", "upon receipt")):
        return 0.0

    match = re.search(r"(\d+)\s*(?:day|days|d)?", s)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    return None


def _normalize_days_to_score(
    days: Optional[float], min_days: float = 0.0, max_days: float = 90.0
) -> Optional[float]:
    if days is None or (isinstance(days, float) and np.isnan(days)):
        return None
    try:
        numeric = float(days)
    except (TypeError, ValueError):
        return None
    clamped = max(min_days, min(max_days, numeric))
    if max_days == min_days:
        return 100.0
    score = (1.0 - (clamped - min_days) / (max_days - min_days)) * 100
    return float(round(score, 2))


def ensure_payment_terms_score(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    if "payment_terms_days" not in df.columns or df["payment_terms_days"].isna().all():
        source_cols = [
            column
            for column in df.columns
            if column.lower() in {"payment_terms", "terms", "pay_terms"}
        ]
        if source_cols:
            src = source_cols[0]
            df["payment_terms_days"] = df[src].apply(_parse_payment_terms_days)
        else:
            df["payment_terms_days"] = np.nan

    for column in df.columns:
        if column.lower() in {
            "payment_terms_days",
            "payment_terms_in_days",
            "pt_days",
        }:
            df["payment_terms_days"] = pd.to_numeric(df[column], errors="coerce")
            break

    needs_score = "payment_terms_score" not in df.columns or df[
        "payment_terms_score"
    ].isna().all()
    if needs_score:
        df["payment_terms_score"] = df["payment_terms_days"].apply(
            _normalize_days_to_score
        )

    # A supplier whose payment terms we never read does not get a score. Imputing a
    # neutral 50 here made every unmeasured supplier look measured, and defeated the
    # weight renormalisation downstream (an all-NaN metric is meant to drop out of the
    # weighted sum; a column of 50s never does). NaN means "not measured" and is the
    # honest answer.
    unknown = int(df["payment_terms_score"].isna().sum())
    if unknown:
        logger.info(
            "payment_terms_score left NULL for %d supplier(s) with no terms on file",
            unknown,
        )

    return df


class SupplierRankingAgent(BaseAgent):
    """Rank suppliers using procurement data, policies and contextual scores."""

    # Which way is "good" for each metric, when no NormalizationDirectionPolicy says
    # otherwise. Without these, a criterion could hold real numbers yet never be scored:
    # the weight map accepts a criterion on its raw column, but scoring needs a _score
    # column, and only the normaliser creates one. The metric would then drop out
    # silently -- and a supplier with usable data would be ranked as though they had none.
    # Governance policy still overrides these.
    DEFAULT_SCORE_DIRECTIONS = {
        "price": "lower_is_better",
        "risk": "lower_is_better",
        "delivery": "higher_is_better",
        "payment_terms": "higher_is_better",
    }

    AGENTIC_PLAN_STEPS = (
        "Aggregate supplier performance, spend, and policy context relevant to the query.",
        "Normalise metrics, apply weightings, and compute composite supplier scores.",
        "Return ranked suppliers with rationale and hand-offs for downstream decisions.",
    )

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.prompt_library: Dict[str, Any] = {}
        self.justification_template: Dict[str, Any] = {}
        self.policy_engine = agent_nick.policy_engine
        self.query_engine = agent_nick.query_engine
        self._device = configure_gpu()
        self._supplier_alias_map: Dict[str, str] = {}
        self._supplier_lookup: Dict[str, Optional[str]] = {}
        self._scoring_reference = load_reference_dataset("supplier_scoring_reference")
        try:
            self._relationship_service = SupplierRelationshipService(agent_nick)
        except Exception:
            logger.debug(
                "SupplierRankingAgent failed to initialise SupplierRelationshipService",
                exc_info=True,
            )
            self._relationship_service = None
        self._cache = {}
        self._max_workers = min(32, (multiprocessing.cpu_count() or 1) * 4)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize required database schema."""
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE SCHEMA IF NOT EXISTS proc;
                        
                        CREATE TABLE IF NOT EXISTS proc.procurement_flow (
                            supplier_id VARCHAR(255),
                            profile JSONB,
                            vector_embedding BYTEA,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (supplier_id)
                        );
                        
                        CREATE INDEX IF NOT EXISTS idx_proc_flow_supplier 
                        ON proc.procurement_flow(supplier_id);
                    """)
                conn.commit()
        except Exception:
            logger.exception("Failed to initialize schema")

    @lru_cache(maxsize=1000)
    def _fetch_supplier_profile(self, supplier_id: str) -> Dict[str, Any]:
        """Cache supplier profiles with vector handling."""
        cache_key = f"profile:{supplier_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT profile, vector_embedding 
                        FROM proc.procurement_flow 
                        WHERE supplier_id = %s 
                        LIMIT 1
                    """, (supplier_id,))
                    row = cur.fetchone()
                    if not row:
                        return {}
                    
                    profile = dict(row[0]) if row[0] else {}
                    if row[1] is not None:  # Handle vector embedding if present
                        try:
                            profile['vector_embedding'] = np.frombuffer(row[1], dtype=np.float32)
                        except Exception:
                            logger.warning(f"Failed to parse vector embedding for supplier {supplier_id}")
                    
                    self._cache[cache_key] = profile
                    return profile
        except Exception:
            logger.exception(f"Failed to fetch profile for supplier {supplier_id}")
            return {}

    def _batch_fetch_supplier_profiles(self, supplier_ids: Set[str], batch_size: int = 50) -> Dict[str, Dict[str, Any]]:
        """Fetch supplier profiles in batches with better error handling."""
        profiles = {}
        supplier_batches = [list(supplier_ids)[i:i + batch_size] for i in range(0, len(supplier_ids), batch_size)]
        
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Check if table exists
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'proc'
                            AND table_name = 'procurement_flow'
                        );
                    """)
                    table_exists = cur.fetchone()[0]
                    
                    if not table_exists:
                        logger.warning("proc.procurement_flow table does not exist - initializing schema")
                        self._init_schema()
                        return profiles

                    for batch in supplier_batches:
                        try:
                            cur.execute("""
                                SELECT supplier_id, profile, vector_embedding
                                FROM proc.procurement_flow 
                                WHERE supplier_id = ANY(%s)
                            """, (batch,))
                            
                            for row in cur.fetchall():
                                sid, profile_data, vector_data = row
                                if not profile_data:
                                    continue
                                    
                                profile = dict(profile_data)
                                if vector_data is not None:
                                    try:
                                        profile['vector_embedding'] = np.frombuffer(vector_data, dtype=np.float32)
                                    except Exception:
                                        logger.warning(f"Failed to parse vector embedding for supplier {sid}")
                                
                                cache_key = f"profile:{sid}"
                                self._cache[cache_key] = profile
                                profiles[sid] = profile
                                
                        except Exception:
                            logger.exception(f"Failed to fetch profiles for batch of {len(batch)} suppliers")
                            continue
                    
        except Exception:
            logger.exception("Database connection failed while fetching supplier profiles")
            
        return profiles

    def _instruction_sources_from_prompt(self, prompt: Dict[str, Any]) -> List[Any]:
        sources: List[Any] = []
        if not isinstance(prompt, dict):
            return sources
        for field in ("prompt_config", "metadata", "prompts_desc", "template"):
            value = prompt.get(field)
            if value:
                sources.append(value)
        return sources

    def _instruction_sources_from_policy(self, policy: Dict[str, Any]) -> List[Any]:
        sources: List[Any] = []
        if not isinstance(policy, dict):
            return sources
        for field in ("policy_details", "details", "policy_desc", "description"):
            value = policy.get(field)
            if value:
                sources.append(value)
        return sources

    def _collect_instruction_bundle(self, context: AgentContext) -> Dict[str, Any]:
        sources: List[Any] = []
        for policy in context.input_data.get("policies") or []:
            sources.extend(self._instruction_sources_from_policy(policy))
        for prompt in context.input_data.get("prompts") or []:
            sources.extend(self._instruction_sources_from_prompt(prompt))
        return parse_instruction_sources(sources)

    def _ingest_prompt_payload(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("templates") and not self.prompt_library:
            self.prompt_library = dict(payload)
        if payload.get("prompt_template") and not self.justification_template:
            self.justification_template = dict(payload)

    def _load_prompt_from_db(self, prompt_id: int) -> None:
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return
        try:
            with get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT prompts_desc FROM proc.bp_prompt WHERE prompt_id = %s",
                        (prompt_id,),
                    )
                    row = cursor.fetchone()
            if not row:
                return
            raw = row[0]
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode(errors="ignore")
            if isinstance(raw, str):
                text = raw.strip()
                if not text:
                    return
                try:
                    payload = json.loads(text)
                except Exception:
                    return
            elif isinstance(raw, dict):
                payload = raw
            else:
                return
            self._ingest_prompt_payload(payload)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load prompt %s from database", prompt_id)

    def _ensure_prompt_assets(self, context: AgentContext) -> None:
        if self.prompt_library and self.justification_template:
            return

        seen_ids: set[int] = set()
        for prompt in context.input_data.get("prompts") or []:
            if not isinstance(prompt, dict):
                continue
            self._ingest_prompt_payload(prompt)
            pid = prompt.get("promptId")
            try:
                if pid is not None:
                    seen_ids.add(int(pid))
            except (TypeError, ValueError):
                continue

        if (not self.prompt_library or not self.justification_template) and seen_ids:
            for pid in seen_ids:
                if self.prompt_library and self.justification_template:
                    break
                self._load_prompt_from_db(pid)

        if not self.prompt_library:
            self.prompt_library = {"templates": []}
        if not self.justification_template:
            self.justification_template = {
                "prompt_template": (
                    "Supplier {supplier_name} achieved a final score of {final_score:.2f}."
                    "\n{score_breakdown}"
                )
            }

    def _coerce_numeric_map(self, payload: Any) -> Dict[str, float]:
        result: Dict[str, float] = {}
        if isinstance(payload, dict):
            for key, value in payload.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if numeric >= 0:
                    result[str(key).strip()] = numeric
        elif isinstance(payload, (list, tuple)):
            for item in payload:
                if isinstance(item, dict):
                    result.update(self._coerce_numeric_map(item))
        elif isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                return self._coerce_numeric_map(parsed)
        return result

    def _ensure_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                pass
            return [token.strip() for token in re.split(r"[,;]", text) if token.strip()]
        return [str(value).strip()]

    def _payment_terms_to_days(self, series: Any) -> pd.Series:
        if series is None:
            return pd.Series(dtype="float64")
        if isinstance(series, pd.Series):
            values = series
        else:
            try:
                values = pd.Series(series)
            except Exception:
                return pd.Series(dtype="float64")
        reference_mapping = {}
        if isinstance(self._scoring_reference, dict):
            reference_mapping = self._scoring_reference.get("payment_terms_days", {}) or {}
        mapping = {
            str(key).lower(): float(value)
            for key, value in reference_mapping.items()
            if value is not None
        }
        days: List[Optional[float]] = []
        for value in values:
            if pd.isna(value):
                days.append(None)
                continue
            if isinstance(value, (int, float)):
                days.append(float(value))
                continue
            text = str(value).strip()
            if not text:
                days.append(None)
                continue
            lower = text.lower()
            normalized = re.sub(r"[^a-z0-9]", "", lower)
            mapped = mapping.get(normalized)
            if mapped is None:
                match = re.search(r"(\d+)", lower)
                if match:
                    mapped = float(match.group(1))
            days.append(mapped)
        return pd.Series(days, index=values.index, dtype="float64")

    def _coerce_policy(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        entry = dict(policy)
        details = entry.get("details")
        if isinstance(details, str):
            try:
                entry["details"] = json.loads(details)
            except Exception:  # pragma: no cover - defensive
                entry["details"] = {}
        elif isinstance(details, dict):
            entry["details"] = dict(details)
        else:
            entry["details"] = {}
        if not entry["details"]:
            raw_details = entry.get("policy_details")
            if isinstance(raw_details, str):
                try:
                    entry["details"] = json.loads(raw_details)
                except Exception:  # pragma: no cover - defensive
                    entry["details"] = {}
            elif isinstance(raw_details, dict):
                entry["details"] = dict(raw_details)
        rules = entry["details"].get("rules") if isinstance(entry["details"], dict) else {}
        if isinstance(rules, str):
            try:
                entry["details"]["rules"] = json.loads(rules)
            except Exception:  # pragma: no cover - defensive
                entry["details"]["rules"] = {}
        elif isinstance(rules, dict):
            entry["details"]["rules"] = dict(rules)
        else:
            entry["details"]["rules"] = {}
        return entry

    def _resolve_policy_bundle(self, context: AgentContext) -> List[Dict[str, Any]]:
        raw = context.input_data.get("policies")
        bundle: List[Dict[str, Any]] = []
        if isinstance(raw, list):
            for policy in raw:
                if isinstance(policy, dict):
                    bundle.append(self._coerce_policy(policy))
        if bundle:
            return bundle
        return [self._coerce_policy(policy) for policy in self.policy_engine.supplier_policies]

    def _find_policy(
        self, policies: List[Dict[str, Any]], name: str
    ) -> Optional[Dict[str, Any]]:
        target = str(name).strip().lower()
        for policy in policies:
            policy_name = (
                policy.get("policyName")
                or policy.get("policy_name")
                or policy.get("name")
                or policy.get("description")
            )
            if policy_name and str(policy_name).strip().lower() == target:
                return policy
        return None

    def _extract_policy_rules(self, policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not policy:
            return {}
        details = policy.get("details")
        if not isinstance(details, dict):
            return {}
        rules = details.get("rules")
        return dict(rules) if isinstance(rules, dict) else {}

    def _governed_default_weights(self, context: "AgentContext") -> Dict[str, float]:
        """Default weights from the orchestrator-injected governance envelope
        (context.input_data['governed']), or {} if none. Never raises."""
        try:
            gov = (getattr(context, "input_data", None) or {}).get("governed") or {}
            for policy in gov.get("policies") or []:
                details = policy.get("details")
                if isinstance(details, str):
                    import ast
                    import json as _json
                    try:
                        details = _json.loads(details)
                    except Exception:
                        try:
                            details = ast.literal_eval(details)
                        except Exception:
                            details = {}
                if isinstance(details, dict):
                    dw = (details.get("rules") or {}).get("default_weights")
                    if isinstance(dw, dict) and dw:
                        return {k: float(v) for k, v in dw.items()
                                if isinstance(v, (int, float)) or str(v).replace(".", "", 1).isdigit()}
        except Exception:  # noqa: BLE001
            logger.debug("supplier_ranking: governed weights read failed", exc_info=True)
        return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, context: AgentContext) -> AgentOutput:
        logger.info("SupplierRankingAgent: Starting ranking...")

        supplier_data = context.input_data.get("supplier_data")
        if supplier_data is None:
            try:
                supplier_data = self.query_engine.fetch_supplier_data(
                    context.input_data
                )
            except Exception:
                logger.exception("Failed to fetch supplier data")
                return self._with_plan(
                    context,
                    AgentOutput(
                        status=AgentStatus.FAILED,
                        data={},
                        error="Failed to fetch supplier data",
                    ),
                )
        try:
            df = (
                supplier_data.copy()
                if isinstance(supplier_data, pd.DataFrame)
                else pd.DataFrame(supplier_data)
            )
        except Exception:
            logger.exception("Invalid supplier_data format")
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.FAILED,
                    data={},
                    error="Failed to parse supplier_data into DataFrame",
                ),
            )

        if "supplier_id" in df.columns:
            df["supplier_id"] = df["supplier_id"].astype(str).str.strip()


        if df.empty:
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.FAILED,
                    data={},
                    error="supplier_data is empty",
                ),
            )

        if "supplier_id" not in df.columns:
            if "supplier_name" in df.columns:
                df["supplier_id"] = df["supplier_name"].astype(str).str.strip()
                logger.warning(
                    "supplier_data missing 'supplier_id'; derived IDs from supplier_name"
                )
            else:
                logger.error("supplier_data missing 'supplier_id' column")
                return self._with_plan(
                    context,
                    AgentOutput(
                        status=AgentStatus.FAILED,
                        data={},
                        error="supplier_data missing 'supplier_id' column",
                    ),
                )

        self._ensure_prompt_assets(context)

        instructions = self._collect_instruction_bundle(context)

        policy_bundle = self._resolve_policy_bundle(context)

        raw_directory = context.input_data.get("supplier_directory")
        directory_entries: List[Dict[str, Any]] = []
        if isinstance(raw_directory, list):
            directory_entries = [entry for entry in raw_directory if isinstance(entry, dict)]
        if not directory_entries:
            directory_entries = self._build_directory_from_dataframe(df)
        directory_lookup: Dict[str, Dict[str, Any]] = {}
        directory_map: Dict[str, Optional[str]] = {}
        for entry in directory_entries:
            if not isinstance(entry, dict):
                continue
            supplier_id = entry.get("supplier_id")
            if supplier_id is None:
                continue
            sid = str(supplier_id).strip()
            if not sid:
                continue
            directory_lookup[sid] = entry
            supplier_name = entry.get("supplier_name")
            directory_map[sid] = (
                str(supplier_name).strip() if isinstance(supplier_name, str) and supplier_name.strip() else None
            )


        if directory_map and "supplier_id" in df.columns:
            mapped_names = df["supplier_id"].map(directory_map)
            if "supplier_name" in df.columns:
                df["supplier_name"] = mapped_names.combine_first(df["supplier_name"])
            else:
                df["supplier_name"] = mapped_names

        candidate_ids = context.input_data.get("supplier_candidates")
        candidate_set = self._normalise_id_set(candidate_ids)
        directory_ids = {sid for sid in directory_map.keys() if sid}
        if directory_ids:
            if candidate_set:
                candidate_set = {sid for sid in candidate_set if sid in directory_ids}
                if not candidate_set:
                    candidate_set = directory_ids
            else:
                candidate_set = directory_ids
        if candidate_set:
            df = df[df["supplier_id"].astype(str).str.strip().isin(candidate_set)].copy()
            df = self._ensure_candidate_rows(df, candidate_set, directory_lookup)
            if directory_map and "supplier_id" in df.columns:
                mapped_names = df["supplier_id"].map(directory_map)
                if "supplier_name" in df.columns:
                    df["supplier_name"] = mapped_names.combine_first(df["supplier_name"])
                else:
                    df["supplier_name"] = mapped_names

            if df.empty:
                return self._with_plan(
                    context,
                    AgentOutput(
                        status=AgentStatus.FAILED,
                        data={},
                        error="No matching suppliers found for candidates",
                    ),
                )

        if "supplier_name" in df.columns:
            df["supplier_name"] = df["supplier_name"].apply(
                lambda val: val.strip() if isinstance(val, str) else val
            )

        self._prime_supplier_aliases(df, directory_entries)

        if self._supplier_lookup and "supplier_name" in df.columns:
            canonical_names = df["supplier_id"].map(self._supplier_lookup.get)
            df["supplier_name"] = canonical_names.combine_first(df["supplier_name"])

        supplier_scope = set()
        if candidate_set:
            supplier_scope = {sid for sid in candidate_set if sid}
        if not supplier_scope and "supplier_id" in df.columns:
            supplier_scope = {
                sid
                for sid in (
                    self._coerce_supplier_id(value) for value in df["supplier_id"]
                )
                if sid
            }
        tables = self._load_procurement_tables(supplier_scope, df)
        df = self._merge_supplier_metrics(df, tables)
        profiles = self._build_supplier_profiles(tables, df["supplier_id"].astype(str))

        # Competitive quotes are the one place real, comparable supplier signal lives.
        # The supplier master (proc.bp_supplier) is a name registry: risk_score,
        # delivery_lead_time_days, incoterms and credit_limit_amount are empty for all
        # 87 rows, which is why this agent could only ever score 0.00. Rank suppliers on
        # what they actually bid for THIS deal instead.
        self._deal_priced = False
        deal_id = context.input_data.get("deal_id")
        deal_id = str(deal_id).strip() if deal_id else None
        if not deal_id and "supplier_id" in df.columns:
            deal_id = self._infer_deal_id(df["supplier_id"].dropna().astype(str))

        if deal_id:
            deal_quotes = self._load_deal_quotes(deal_id)
            if not deal_quotes.empty:
                df = df.merge(deal_quotes, on="supplier_id", how="left")
                self._deal_priced = True
                self._deal_id = deal_id
                logger.info(
                    "Deal %s: loaded standing offers from %d supplier(s) for price scoring",
                    deal_id,
                    len(deal_quotes),
                )
            else:
                logger.info("Deal %s: no quotes found; no price signal to rank on", deal_id)
        else:
            logger.info(
                "No deal to rank on: these suppliers did not compete against each other. "
                "There is no global fallback -- the supplier master holds no risk, "
                "delivery or reliability data to score."
            )

        external_profiles = context.input_data.get("supplier_category_profiles")
        if isinstance(external_profiles, str):
            try:
                external_profiles = json.loads(external_profiles)
            except Exception:  # pragma: no cover - defensive
                external_profiles = {}
        if isinstance(external_profiles, dict):
            for supplier_id, extra in external_profiles.items():
                if not isinstance(extra, dict):
                    continue
                sid = str(supplier_id).strip()
                if not sid:
                    continue
                profile = profiles.setdefault(
                    sid,
                    {
                        "supplier_id": sid,
                        "po_ids": [],
                        "invoice_ids": [],
                        "items": [],
                        "categories": [],
                    },
                )
                primary_category = extra.get("primary_category")
                if primary_category and not profile.get("primary_category"):
                    profile["primary_category"] = primary_category
                categories_payload = extra.get("category_breakdown") or extra.get("categories")
                if isinstance(categories_payload, list):
                    profile["categories"] = categories_payload
                elif isinstance(categories_payload, dict):
                    profile["categories"] = [
                        {"category": key, "occurrences": value}
                        for key, value in categories_payload.items()
                    ]
                products = extra.get("products")
                if isinstance(products, list) and products:
                    profile["products"] = products
                sources = extra.get("sources")
                if isinstance(sources, list) and sources:
                    profile["sources"] = sources

        if profiles and "supplier_id" in df.columns:
            primary_category_map = {
                sid: profile.get("primary_category")
                for sid, profile in profiles.items()
                if isinstance(profile, dict)
            }
            df["primary_category"] = df["supplier_id"].map(primary_category_map)

        contexts_by_id, contexts_by_name = self._fetch_relationship_context(df)
        if contexts_by_id or contexts_by_name:
            df = self._merge_relationship_context(df, contexts_by_id, contexts_by_name)

        coverage_series = self._derive_metric_coverage(df)
        if not coverage_series.empty:
            df["flow_coverage"] = coverage_series
        else:
            if "flow_coverage" not in df.columns:
                df["flow_coverage"] = 0.0

        flow_payload = context.input_data.get("data_flow_snapshot")
        if isinstance(flow_payload, dict):
            flow_index, flow_name_index = self._build_flow_index(flow_payload)
        else:
            flow_index, flow_name_index = self._build_flow_index(context.input_data)
        alias_tokens_map = self._alias_tokens_by_supplier()
        if flow_index or flow_name_index:
            df = self._annotate_flow_coverage(
                df, flow_index, flow_name_index, alias_tokens_map
            )

        intent = context.input_data.get("intent", {})
        requested = intent.get("parameters", {}).get("criteria", [])
        criteria_override = (
            instructions.get("criteria")
            or instructions.get("metrics")
            or instructions.get("focus_metrics")
        )
        override_criteria = self._ensure_list(criteria_override)
        if override_criteria:
            requested = override_criteria
            intent.setdefault("parameters", {})["criteria"] = override_criteria

        weight_policy = self._find_policy(policy_bundle, "WeightAllocationPolicy")
        weight_rules = self._extract_policy_rules(weight_policy)
        default_weights = weight_rules.get("default_weights", {})
        # Prefer the governance the orchestrator injected (single governed source),
        # falling back to the code-resolved bundle when absent.
        governed_weights = self._governed_default_weights(context)
        if governed_weights:
            default_weights = governed_weights
            logger.info("supplier_ranking: using governed weights from envelope: %s", governed_weights)
        override_weights_map: Dict[str, float] = {}
        for key in ("metric_weights", "weights", "weightings", "default_weights"):
            override_weights_map = self._coerce_numeric_map(instructions.get(key))
            if override_weights_map:
                break
        if override_weights_map:
            total_override = sum(override_weights_map.values())
            if total_override > 0:
                default_weights = {
                    metric: value / total_override
                    for metric, value in override_weights_map.items()
                    if value >= 0
                }
            else:
                default_weights = override_weights_map

        criteria = requested if requested else list(default_weights.keys())
        weights = {
            crit: default_weights.get(crit, 0.0)
            for crit in criteria
            if default_weights.get(crit, 0.0) > 0
        }

        if not weights:
            # avg_unit_price counts as price signal: _prepare_scoring_columns promotes it
            # to `price` below when there is no competitive quote to score instead.
            fallback_sources = {"price": ("price", "avg_unit_price")}
            fallback_metrics = [
                metric
                for metric in ("price", "delivery", "risk", "payment_terms")
                if any(
                    col in df.columns
                    for col in fallback_sources.get(metric, (metric,))
                )
                or f"{metric}_score" in df.columns
            ]
            if fallback_metrics:
                equal_weight = 1.0 / len(fallback_metrics)
                weights = {metric: equal_weight for metric in fallback_metrics}

        df = self._prepare_scoring_columns(df, weights)
        df = ensure_payment_terms_score(df)
        scored_df = self._score_categorical_criteria(df, weights.keys(), policy_bundle)
        norm_policy = self._find_policy(policy_bundle, "NormalizationDirectionPolicy")
        # Built-in directions first, governed policy on top -- so a criterion always has a
        # way to be scored, but the DB can still override how.
        direction_map = {
            crit: direction
            for crit, direction in self.DEFAULT_SCORE_DIRECTIONS.items()
            if crit in weights
        }
        direction_map.update(self._extract_policy_rules(norm_policy) or {})
        scored_df = self._normalize_numeric_scores(scored_df, direction_map)

        # Authoritative price scoring: deal-scoped, and NULL unless suppliers actually
        # competed. Runs after the generic normaliser so it overrides any global
        # price pass, which would rank a supplier against unrelated deals.
        if getattr(self, "_deal_priced", False):
            scored_df = self._score_deal_price(scored_df)

        normalised_weights = self._normalise_weight_map(scored_df, weights)
        if normalised_weights:
            weights = normalised_weights

        # Score each supplier only on the metrics we actually hold for them, renormalising
        # that supplier's weights over those metrics.
        #
        # The previous fillna(0) charged a supplier the full weight of every metric while
        # scoring them 0 on any they were missing -- so a supplier whose payment terms we
        # simply never read was ranked as though they had offered the worst terms on the
        # table. That punishes gaps in OUR data as if they were faults in THEIR bid.
        #
        # A supplier with no measurable metric at all scores NaN, not 0.0: we have no
        # opinion on them, and saying "0" would be inventing one.
        criteria_cols = {
            crit: f"{crit}_score"
            for crit in weights
            if f"{crit}_score" in scored_df.columns
        }
        for crit in weights:
            if crit not in criteria_cols:
                logger.warning("Criterion column missing: %s_score", crit)

        if criteria_cols:
            score_frame = scored_df[list(criteria_cols.values())].apply(
                pd.to_numeric, errors="coerce"
            )
            weight_row = pd.Series(
                {col: float(weights[crit]) for crit, col in criteria_cols.items()}
            )
            present = score_frame.notna()
            weighted_sum = (score_frame.fillna(0) * weight_row).sum(axis=1)
            weight_present = present.mul(weight_row, axis=1).sum(axis=1)
            scored_df["final_score"] = np.where(
                weight_present > 0, weighted_sum / weight_present, np.nan
            )
            scored_df["scored_on"] = present.apply(
                lambda r: [c.removesuffix("_score") for c in present.columns[r.values]],
                axis=1,
            )
        else:
            scored_df["final_score"] = np.nan
            scored_df["scored_on"] = [[] for _ in range(len(scored_df))]

        scored_df = self._apply_flow_bonus(
            scored_df, flow_index, flow_name_index, alias_tokens_map
        )

        # Refuse to publish a ranking we cannot stand behind. Emitting a confident-looking
        # table of 0.00s (which is what this agent did until now) is worse than failing:
        # a buyer cannot tell "these suppliers are bad" from "we knew nothing about them".
        if scored_df["final_score"].isna().all():
            reason = (
                "insufficient data to rank: no supplier had a measurable metric. "
                "Competitive price scoring needs a deal_id with 2+ supplier quotes; "
                "the supplier master holds no risk or delivery data to fall back on."
            )
            logger.warning("SupplierRankingAgent: %s", reason)
            return self._with_plan(
                context,
                AgentOutput(status=AgentStatus.FAILED, data={}, error=reason),
            )

        ranked_df = scored_df.sort_values(
            by="final_score", ascending=False
        ).reset_index(drop=True)

        top_n = intent.get("parameters", {}).get("top_n")
        if not top_n:
            query_text = context.input_data.get("query", "")
            match = re.search(r"top[-\s]*(\d+)", query_text, re.IGNORECASE)
            top_n = int(match.group(1)) if match else 3
        override_top_n = (
            instructions.get("top_n")
            or instructions.get("max_suppliers")
            or instructions.get("supplier_limit")
        )
        if override_top_n is not None:
            try:
                top_n = int(float(override_top_n))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass
        top_n = max(1, min(int(top_n), len(ranked_df)))

        top_df = ranked_df.head(top_n).copy()
        top_df["justification"] = top_df.apply(
            lambda row: self._generate_justification(row, weights.keys()), axis=1
        )

        ranking = [
            self._prepare_ranking_entry(row, profiles.get(str(row.get("supplier_id"))), weights)
            for _, row in top_df.iterrows()
        ]

        if candidate_set:
            ranking = [
                entry
                for entry in ranking
                if str(entry.get("supplier_id", "")).strip() in candidate_set
            ]

        total_rankings = len(ranking)
        for index, entry in enumerate(ranking, start=1):
            entry["rank_position"] = index
            entry["rank_count"] = total_rankings

        logger.info(
            "SupplierRankingAgent: Ranking complete with %d entries", len(ranking)
        )

        self._persist_ranking(ranking, getattr(context, "workflow_id", None))

        output_data = {
            "ranking": ranking,
            "supplier_profiles": profiles,
            "rank_count": total_rankings,
        }
        pass_fields = {
            "ranking": ranking,
            "supplier_profiles": profiles,
        }
        if total_rankings:
            pass_fields["rank_count"] = total_rankings
        return self._with_plan(
            context,
            AgentOutput(
                status=AgentStatus.SUCCESS,
                data=output_data,
                pass_fields=pass_fields,
                next_agents=["EmailDraftingAgent"],
            ),
        )

    def _persist_ranking(self, ranking: List[Dict], workflow_id: Optional[str]) -> None:
        """Write the computed ranking to proc.bp_supplier_ranking.

        This agent computed genuine scores (final/price/delivery/risk) and then threw them
        away: the result lived only in the workflow blackboard and the HTTP response, and
        vanished when the run ended. Nothing persisted it, so the Suppliers view could only
        ever show master data with no ranking. (proc.procurement_flow — the table an earlier
        design intended — has 0 rows.)

        Best-effort: a persistence failure must not fail the ranking.
        """
        if not ranking:
            return
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    for e in ranking:
                        sid = e.get("supplier_id")
                        if not sid:
                            continue
                        cur.execute(
                            """
                            INSERT INTO proc.bp_supplier_ranking
                                (workflow_id, supplier_id, supplier_name, rank_position,
                                 rank_count, final_score, price_score, delivery_score,
                                 risk_score, payment_terms_score, avg_unit_price, total_spend,
                                 po_count, invoice_count, lead_time_days, justification)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            """,
                            (
                                workflow_id, str(sid), e.get("supplier_name"),
                                e.get("rank_position"), e.get("rank_count"),
                                e.get("final_score"), e.get("price_score"),
                                e.get("delivery_score"), e.get("risk_score"),
                                e.get("payment_terms_score"), e.get("avg_unit_price"),
                                e.get("total_spend"), e.get("po_count"),
                                e.get("invoice_count"), e.get("lead_time_days"),
                                e.get("justification"),
                            ),
                        )
            logger.info("SupplierRankingAgent: persisted %d ranking row(s)", len(ranking))
        except Exception as exc:  # pragma: no cover - persistence is best-effort
            logger.error("Failed to persist supplier ranking: %s", exc)

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------
    def _normalise_id_set(self, ids: Optional[Iterable]) -> set[str]:
        if not ids:
            return set()
        try:
            return {str(val).strip() for val in ids if str(val).strip()}
        except Exception:
            return {str(ids).strip()}

    def _normalise_supplier_token(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
        else:
            text = str(value).strip()
        if not text:
            return None
        return re.sub(r"\s+", " ", text.lower())

    def _coerce_supplier_id(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
        else:
            text = str(value).strip()
        return text or None

    def _build_directory_from_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        if "supplier_id" not in df.columns:
            return []

        entries: Dict[str, Dict[str, Any]] = {}
        names: pd.Series
        if "supplier_name" in df.columns:
            names = df["supplier_name"]
        else:
            names = pd.Series([None] * len(df), index=df.index, dtype="object")

        for supplier_id, supplier_name in zip(df["supplier_id"], names):
            sid = self._coerce_supplier_id(supplier_id)
            if not sid:
                continue
            entry = entries.setdefault(sid, {"supplier_id": sid})
            if isinstance(supplier_name, str) and supplier_name.strip():
                entry.setdefault("supplier_name", supplier_name.strip())
        return list(entries.values())

    def _prime_supplier_aliases(
        self, supplier_df: pd.DataFrame, directory_entries: Iterable[Dict[str, Any]]
    ) -> None:
        alias_map: Dict[str, str] = {}
        lookup: Dict[str, Optional[str]] = {}

        if isinstance(supplier_df, pd.DataFrame) and not supplier_df.empty:
            alias_fields = [
                col for col in ("supplier_name", "trading_name") if col in supplier_df.columns
            ]
            for row in supplier_df.itertuples(index=False):
                sid = self._coerce_supplier_id(getattr(row, "supplier_id", None))
                if not sid:
                    continue
                canonical_name = getattr(row, "supplier_name", None)
                if not isinstance(canonical_name, str) or not canonical_name.strip():
                    for field in alias_fields:
                        value = getattr(row, field, None)
                        if isinstance(value, str) and value.strip():
                            canonical_name = value
                            break
                if isinstance(canonical_name, str):
                    lookup.setdefault(sid, canonical_name.strip())
                else:
                    lookup.setdefault(sid, None)
                norm_id = self._normalise_supplier_token(sid)
                if norm_id:
                    alias_map.setdefault(norm_id, sid)
                for field in alias_fields:
                    value = getattr(row, field, None)
                    norm = self._normalise_supplier_token(value)
                    if norm and norm not in alias_map:
                        alias_map[norm] = sid

        for entry in directory_entries or []:
            if not isinstance(entry, dict):
                continue
            sid = self._coerce_supplier_id(entry.get("supplier_id"))
            if not sid:
                continue
            name = entry.get("supplier_name")
            if isinstance(name, str) and name.strip():
                lookup.setdefault(sid, name.strip())
            norm_id = self._normalise_supplier_token(sid)
            if norm_id:
                alias_map.setdefault(norm_id, sid)
            for candidate in (entry.get("supplier_name"), entry.get("trading_name")):
                norm = self._normalise_supplier_token(candidate)
                if norm and norm not in alias_map:
                    alias_map[norm] = sid

        self._supplier_alias_map = {key: val for key, val in alias_map.items() if key and val}
        self._supplier_lookup = {sid: lookup.get(sid) for sid in {val for val in alias_map.values()}}

    def _resolve_supplier_identifier(self, value: Any) -> Optional[str]:
        sid = self._coerce_supplier_id(value)
        norm = self._normalise_supplier_token(value)
        if norm and norm in self._supplier_alias_map:
            return self._supplier_alias_map[norm]
        if sid and (sid in self._supplier_lookup or sid in self._supplier_alias_map.values()):
            return sid
        return sid

    def _map_supplier_ids(self, df: pd.DataFrame, name_fields: Iterable[str]) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()
        if df.empty:
            return df

        out = df.copy()

        if "supplier_id" in out.columns:
            out["supplier_id"] = out["supplier_id"].apply(self._coerce_supplier_id)
        else:
            out["supplier_id"] = pd.Series([None] * len(out), index=out.index, dtype="object")

        missing_mask = out["supplier_id"].isna()
        for field in name_fields:
            if field not in out.columns:
                continue
            resolved = out.loc[missing_mask, field].apply(self._resolve_supplier_identifier)
            out.loc[missing_mask, "supplier_id"] = resolved
            missing_mask = out["supplier_id"].isna()
            if not missing_mask.any():
                break

        if "supplier_name" in out.columns:
            out["supplier_name"] = out["supplier_name"].apply(
                lambda val: val.strip() if isinstance(val, str) else val
            )
            if self._supplier_lookup:
                canonical = out["supplier_id"].map(self._supplier_lookup.get)
                out["supplier_name"] = canonical.combine_first(out["supplier_name"])

        return out

    def _ensure_candidate_rows(
        self,
        df: pd.DataFrame,
        candidate_ids: set[str],
        directory_lookup: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        if not candidate_ids:
            return df

        present_ids: set[str] = set()
        if "supplier_id" in df.columns and not df.empty:
            present_ids = set(df["supplier_id"].astype(str).str.strip())

        missing = [cid for cid in candidate_ids if cid not in present_ids]

        if not missing and not df.empty:
            return df

        fallback_rows: List[Dict[str, Any]] = []
        for cid in missing:
            entry = directory_lookup.get(cid, {})
            row: Dict[str, Any] = dict(entry) if isinstance(entry, dict) else {}
            row["supplier_id"] = cid
            supplier_name = row.get("supplier_name")
            if isinstance(supplier_name, str):
                row["supplier_name"] = supplier_name.strip()
            elif supplier_name is None:
                row["supplier_name"] = pd.NA
            fallback_rows.append(row)

        if not fallback_rows and df.empty:
            fallback_rows = [{"supplier_id": cid, "supplier_name": pd.NA} for cid in candidate_ids]

        if not fallback_rows:
            return df

        fallback_df = pd.DataFrame(fallback_rows)
        if "supplier_id" in fallback_df.columns:
            fallback_df["supplier_id"] = fallback_df["supplier_id"].astype(str).str.strip()

        if df.empty:
            return fallback_df

        combined = pd.concat([df, fallback_df], ignore_index=True, sort=False)
        if "supplier_id" in combined.columns:
            combined["supplier_id"] = combined["supplier_id"].astype(str).str.strip()
        return combined

    def _load_procurement_tables(
        self, supplier_ids: Iterable[str], supplier_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        suppliers = {str(s).strip() for s in supplier_ids if str(s).strip()}
        supplier_names: set[str] = set()
        if isinstance(supplier_df, pd.DataFrame) and not supplier_df.empty:
            if "supplier_name" in supplier_df.columns:
                supplier_names = {
                    str(name).strip().lower()
                    for name in supplier_df["supplier_name"].dropna()
                    if str(name).strip()
                }

        tables: Dict[str, pd.DataFrame] = {}

        try:
            po_df = self.query_engine.fetch_purchase_order_data(
                supplier_ids=suppliers or None,
                supplier_names=supplier_names or None,
            )
        except Exception:
            logger.exception("Failed to load purchase orders")
            po_df = pd.DataFrame()
        tables["purchase_orders"] = self._map_supplier_ids(po_df, ("supplier_name",))

        try:
            invoice_df = self.query_engine.fetch_invoice_data(
                supplier_ids=suppliers or None,
                supplier_names=supplier_names or None,
            )
        except Exception:
            logger.exception("Failed to load invoices")
            invoice_df = pd.DataFrame()
        tables["invoices"] = self._map_supplier_ids(invoice_df, ("supplier_name",))

        po_ids: List[str] = []
        if not po_df.empty and "po_id" in po_df.columns:
            po_ids = [
                str(value).strip()
                for value in po_df["po_id"].dropna()
                if str(value).strip()
            ]
        invoice_ids: List[str] = []
        if not invoice_df.empty and "invoice_id" in invoice_df.columns:
            invoice_ids = [
                str(value).strip()
                for value in invoice_df["invoice_id"].dropna()
                if str(value).strip()
            ]

        tables["po_lines"] = self._read_table(
            "proc.bp_po_line_items_trgt",
            "po_id = ANY(%s)" if po_ids else None,
            [po_ids] if po_ids else None,
            columns=(
                "po_id",
                "po_line_id",
                "line_number",
                "item_id",
                "item_description",
                "quantity",
                "unit_price",
                "line_total",
                "total_amount",
            ),
        )
        tables["invoice_lines"] = self._read_table(
            "proc.bp_invoice_line_items_trgt",
            "invoice_id = ANY(%s)" if invoice_ids else None,
            [invoice_ids] if invoice_ids else None,
            columns=(
                "invoice_id",
                "invoice_line_id",
                "po_id",
                "item_description",
                "total_amount_incl_tax",
            ),
        )
        try:
            flow = self.query_engine.fetch_procurement_flow(
                embed=False,
                supplier_ids=suppliers or None,
                supplier_names=supplier_names or None,
            )
        except Exception:
            logger.exception("Failed to load procurement flow")
            flow = pd.DataFrame()
        flow = self._map_supplier_ids(flow, ("supplier_name",))
        if not flow.empty and suppliers and "supplier_id" in flow.columns:
            supplier_series = flow["supplier_id"].apply(self._coerce_supplier_id)
            mask = supplier_series.isin(suppliers)
            flow = flow[mask].copy()
            flow["supplier_id"] = supplier_series.loc[flow.index]
        tables["procurement_flow"] = flow
        return tables

    # ------------------------------------------------------------------
    # Deal-scoped competitive quotes
    # ------------------------------------------------------------------
    @staticmethod
    def _quote_version(quote_id: Any) -> Optional[int]:
        """Pull the bidding round out of a quote_id like 'STC-RFQ-2204 (V3 (BAFO))'."""
        if not isinstance(quote_id, str):
            return None
        match = re.search(r"\(\s*V(\d+)", quote_id, re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    def _infer_deal_id(self, supplier_ids: Iterable[str]) -> Optional[str]:
        """Find the deal these suppliers actually competed on, when nobody named one.

        Only a deal where 2+ of the candidates bid is useful -- one bidder is not a
        contest. Where several qualify we take the one the most candidates bid on, and
        refuse to choose on a tie rather than silently ranking against an arbitrary deal.
        """
        suppliers = [str(s).strip() for s in supplier_ids if str(s).strip()]
        if len(suppliers) < 2:
            return None

        df = self._read_table(
            "proc.bp_quote_trgt",
            "supplier_id = ANY(%s) AND deal_id IS NOT NULL",
            ([suppliers],),
            columns=("deal_id", "supplier_id"),
        )
        if df.empty:
            return None

        counts = (
            df.drop_duplicates(["deal_id", "supplier_id"])
            .groupby("deal_id")["supplier_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        contested = counts[counts >= 2]
        if contested.empty:
            return None
        if len(contested) > 1 and contested.iloc[0] == contested.iloc[1]:
            logger.warning(
                "Candidates bid on %d deals with equal participation; cannot tell which "
                "one to rank. Pass deal_id explicitly.",
                int((contested == contested.iloc[0]).sum()),
            )
            return None

        deal_id = str(contested.index[0])
        logger.info(
            "Inferred deal %s for ranking (%d of the candidate suppliers bid on it)",
            deal_id,
            int(contested.iloc[0]),
        )
        return deal_id

    def _load_deal_quotes(self, deal_id: str) -> pd.DataFrame:
        """Each supplier's live offer on one deal, plus how they got there.

        Suppliers bid in rounds (V1 -> V2 -> V3/BAFO). Only the latest round is a real,
        standing offer; the superseded ones are history. Treating all of them as separate
        bids would both invent competitors that do not exist and triple-count a single
        supplier's money.
        """
        quotes = self._read_table(
            "proc.bp_quote_trgt",
            "deal_id = %s AND supplier_id IS NOT NULL",
            (deal_id,),
            columns=(
                "quote_id",
                "deal_id",
                "supplier_id",
                "quote_date",
                "total_amount",
                "currency",
            ),
        )
        if quotes.empty:
            return pd.DataFrame()

        quotes = quotes.copy()
        quotes["total_amount"] = pd.to_numeric(quotes["total_amount"], errors="coerce")
        quotes["_version"] = quotes["quote_id"].apply(self._quote_version)
        quotes["_qdate"] = pd.to_datetime(quotes["quote_date"], errors="coerce")

        rows: List[Dict[str, Any]] = []
        for supplier_id, grp in quotes.groupby("supplier_id", dropna=True):
            priced = grp.dropna(subset=["total_amount"])
            if priced.empty:
                continue
            # Latest round wins: explicit version if the supplier used one, else date.
            # If neither can order the bids, we cannot tell which offer stands, so we
            # decline to guess -- we take the single row only when there is just one.
            ordered = priced.sort_values(
                by=["_version", "_qdate"], ascending=True, na_position="first"
            )
            if ordered["_version"].notna().any() or ordered["_qdate"].notna().any():
                latest = ordered.iloc[-1]
                opening = ordered.iloc[0]
            elif len(ordered) == 1:
                latest = opening = ordered.iloc[0]
            else:
                logger.warning(
                    "Deal %s supplier %s: %d quotes with no version or date to order "
                    "them by; cannot identify the standing offer, skipping",
                    deal_id,
                    supplier_id,
                    len(ordered),
                )
                continue

            final_amount = float(latest["total_amount"])
            open_amount = float(opening["total_amount"])
            concession_pct = None
            if open_amount > 0 and len(ordered) > 1:
                concession_pct = round((open_amount - final_amount) / open_amount * 100, 2)

            rows.append(
                {
                    "supplier_id": str(supplier_id).strip(),
                    "price": final_amount,  # lower is better; scored within the deal
                    "final_quote_amount": final_amount,
                    "opening_quote_amount": open_amount,
                    "quote_currency": latest.get("currency"),
                    "quote_rounds": int(len(ordered)),
                    "concession_pct": concession_pct,
                    "winning_quote_id": latest.get("quote_id"),
                }
            )

        return pd.DataFrame(rows)

    def _score_deal_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score the standing offers against each other. 100 = cheapest on this deal.

        Price is only meaningful against a rival. A lone bidder gets NULL, not 100 -- an
        uncontested quote is no evidence of a good price, and scoring it top would let a
        single-supplier deal masquerade as a competitive win.
        """
        out = df.copy()
        if "price" not in out.columns:
            return out

        vals = pd.to_numeric(out["price"], errors="coerce")
        bidders = int(vals.notna().sum())
        if bidders < 2:
            if bidders == 1:
                logger.info(
                    "Only one supplier bid on this deal; price_score left NULL "
                    "(nothing to compare against)"
                )
            out["price_score"] = np.nan
            return out

        # Score each bid against the CHEAPEST bid, not across the min-max spread.
        #
        # Min-max stretches whatever gap exists across the full 0-100 scale, so on the
        # live TEST005 deal it scored Northgate 0.00 against SteelCore's 100.00 -- when
        # Northgate was 1.1% more expensive. All three bids sat within GBP 3,050 of each
        # other. The ordering was right but the numbers slandered the runners-up, and a
        # buyer reading "0 out of 100" would draw a conclusion the data does not support.
        #
        # Ratio-to-best keeps the same ordering and makes the score mean something
        # absolute: 98.9 means "1.1% off the best price".
        cheapest = float(vals.min())
        if cheapest <= 0:
            logger.warning(
                "Cheapest bid on this deal is %s; cannot score price as a ratio", cheapest
            )
            out["price_score"] = np.nan
            return out

        out["price_score"] = (100.0 * cheapest / vals).round(2)
        return out

    def _read_table(
        self,
        table: str,
        where: Optional[str] = None,
        params: Optional[Any] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        column_sql = ", ".join(columns) if columns else "*"
        sql = f"SELECT {column_sql} FROM {table}"
        if where:
            sql += f" WHERE {where}"
        pandas_conn = getattr(self.agent_nick, "pandas_connection", None)
        try:
            if callable(pandas_conn):
                with pandas_conn() as conn:
                    return read_sql_compat(sql, conn, params=params)
            with self.agent_nick.get_db_connection() as conn:
                return read_sql_compat(sql, conn, params=params)
        except Exception:
            logger.exception("Failed to read table %s", table)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Metric enrichment
    # ------------------------------------------------------------------
    def _merge_supplier_metrics(
        self, df: pd.DataFrame, tables: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        if df.empty:
            return df

        po_summary = self._summarise_purchase_orders(tables.get("purchase_orders", pd.DataFrame()))
        po_line_summary = self._summarise_po_lines(
            tables.get("po_lines", pd.DataFrame()),
            tables.get("purchase_orders", pd.DataFrame()),
        )
        invoice_summary = self._summarise_invoices(tables.get("invoices", pd.DataFrame()))
        invoice_line_summary = self._summarise_invoice_lines(
            tables.get("invoice_lines", pd.DataFrame()),
            tables.get("invoices", pd.DataFrame()),
        )

        result = df.copy()
        for summary in [po_summary, po_line_summary, invoice_summary, invoice_line_summary]:
            if summary.empty:
                continue
            result = result.merge(summary, on="supplier_id", how="left")

        for column in [
            "po_total_value",
            "po_line_spend",
            "invoice_total_value",
            "avg_unit_price",
            "total_volume",
            "avg_lead_time_days",
            "total_spend",
            "avg_payment_term_days",
            "avg_paid_days",
        ]:
            if column in result.columns:
                result[column] = pd.to_numeric(result[column], errors="coerce")

        spend_components = [
            result.get("po_total_value"),
            result.get("invoice_total_value"),
            result.get("po_line_spend"),
        ]
        numeric_components: List[pd.Series] = []
        for comp in spend_components:
            if isinstance(comp, pd.Series):
                numeric_components.append(pd.to_numeric(comp, errors="coerce").fillna(0.0))

        if "total_spend" in result.columns:
            total_spend_series = pd.to_numeric(
                result["total_spend"], errors="coerce"
            ).fillna(0.0)
        else:
            total_spend_series = pd.Series(0.0, index=result.index, dtype="float64")

        for comp_series in numeric_components:
            total_spend_series = total_spend_series.add(comp_series, fill_value=0.0)

        result["total_spend"] = total_spend_series

        if "avg_unit_price" not in result.columns:
            result["avg_unit_price"] = pd.NA
        if "avg_unit_price" in result.columns:
            missing_price = result["avg_unit_price"].isna()
            if "po_line_spend" in result.columns and "total_volume" in result.columns:
                # Fix deprecated use_inf_as_na
                calculated = (
                    result["po_line_spend"].fillna(0) / 
                    result["total_volume"].replace([np.inf, -np.inf], np.nan)
                ).fillna(0)
                # Ensure compatible dtype
                result.loc[missing_price, "avg_unit_price"] = calculated[missing_price].astype(float)

        if "payment_terms" not in result.columns and "po_payment_terms" in result.columns:
            result["payment_terms"] = result["po_payment_terms"]

        return result

    def _derive_metric_coverage(self, df: pd.DataFrame) -> pd.Series:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.Series(dtype="float64")

        def _signal(columns: Tuple[str, ...]) -> Optional[pd.Series]:
            series_list: List[pd.Series] = []
            for column in columns:
                if column not in df.columns:
                    continue
                series = pd.to_numeric(df[column], errors="coerce")
                if series.isna().all():
                    continue
                series_list.append(series.fillna(0.0))
            if not series_list:
                return None
            combined = sum(series_list)
            if isinstance(combined, pd.Series):
                return (combined.fillna(0.0) > 0).astype(float)
            return None

        signals: List[pd.Series] = []
        for cols in (
            ("po_total_value", "po_line_spend"),
            ("invoice_total_value", "invoice_count"),
            ("invoice_item_count", "invoice_count"),
            ("total_spend",),
        ):
            signal = _signal(cols)
            if signal is not None:
                signals.append(signal)

        if not signals:
            return pd.Series(0.0, index=df.index, dtype="float64")

        combined = sum(signals)
        if not isinstance(combined, pd.Series):
            return pd.Series(0.0, index=df.index, dtype="float64")

        return (combined / len(signals)).clip(lower=0.0, upper=1.0)

    def _summarise_purchase_orders(self, po_df: pd.DataFrame) -> pd.DataFrame:
        if po_df.empty:
            return pd.DataFrame()
        df = self._map_supplier_ids(po_df, ("supplier_name",))
        if "supplier_id" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["supplier_id"]).copy()
        df["supplier_id"] = df["supplier_id"].astype(str)
        value_col = "total_amount_gbp" if "total_amount_gbp" in df.columns else "total_amount"
        if value_col in df.columns:
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
        else:
            df[value_col] = 0.0

        if "order_date" in df.columns:
            df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        if "expected_delivery_date" in df.columns:
            df["expected_delivery_date"] = pd.to_datetime(
                df["expected_delivery_date"], errors="coerce"
            )
            df["lead_time_days"] = (
                df["expected_delivery_date"] - df.get("order_date")
            ).dt.days

        agg = df.groupby("supplier_id").agg(
            po_total_value=(value_col, "sum"),
            po_count=("po_id", "nunique"),
            po_payment_terms=("payment_terms", lambda s: self._mode_value(s)),
            last_order_date=("order_date", "max"),
            avg_lead_time_days=("lead_time_days", "mean"),
        )
        return agg.reset_index()

    def _summarise_po_lines(
        self, po_lines: pd.DataFrame, po_df: pd.DataFrame
    ) -> pd.DataFrame:
        if po_lines.empty or "po_id" not in po_lines.columns:
            return pd.DataFrame()
        if po_df.empty or "po_id" not in po_df.columns:
            return pd.DataFrame()
        po_lookup = self._map_supplier_ids(po_df, ("supplier_name",))
        join_cols = ["po_id"]
        extra_cols = [col for col in ("supplier_id", "supplier_name") if col in po_lookup.columns]
        if not extra_cols:
            return pd.DataFrame()
        join_df = po_lookup[join_cols + extra_cols].drop_duplicates("po_id")
        df = po_lines.merge(join_df, on="po_id", how="left")
        df = self._map_supplier_ids(df, ("supplier_name",))
        if "supplier_id" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["supplier_id"]).copy()
        df["supplier_id"] = df["supplier_id"].astype(str)

        for col in ["unit_price", "line_total", "quantity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "line_total" not in df.columns:
            df["line_total"] = df.get("unit_price", 0.0) * df.get("quantity", 0.0)

        agg = df.groupby("supplier_id").agg(
            po_line_spend=("line_total", "sum"),
            avg_unit_price=("unit_price", "mean"),
            total_volume=("quantity", "sum"),
            catalog_items=("item_description", lambda s: self._top_values(s, limit=5)),
        )
        return agg.reset_index()

    def _summarise_invoices(self, invoice_df: pd.DataFrame) -> pd.DataFrame:
        if invoice_df.empty:
            return pd.DataFrame()
        df = self._map_supplier_ids(invoice_df, ("supplier_name",))
        if "supplier_id" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["supplier_id"]).copy()
        df["supplier_id"] = df["supplier_id"].astype(str)
        total_col = "invoice_total_incl_tax" if "invoice_total_incl_tax" in df.columns else "invoice_amount"
        if total_col in df.columns:
            df[total_col] = pd.to_numeric(df[total_col], errors="coerce").fillna(0.0)
        else:
            df[total_col] = 0.0
        if "invoice_date" in df.columns:
            df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        if "due_date" in df.columns:
            df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")
            df["term_days"] = (df["due_date"] - df.get("invoice_date")).dt.days
        if "invoice_paid_date" in df.columns:
            df["invoice_paid_date"] = pd.to_datetime(
                df["invoice_paid_date"], errors="coerce"
            )
            df["paid_days"] = (df["invoice_paid_date"] - df.get("invoice_date")).dt.days
        agg = df.groupby("supplier_id").agg(
            invoice_total_value=(total_col, "sum"),
            invoice_count=("invoice_id", "nunique"),
            last_invoice_date=("invoice_date", "max"),
            avg_payment_term_days=("term_days", "mean"),
            avg_paid_days=("paid_days", "mean"),
        )
        return agg.reset_index()

    def _summarise_invoice_lines(
        self, invoice_lines: pd.DataFrame, invoice_df: pd.DataFrame
    ) -> pd.DataFrame:
        if invoice_lines.empty or "invoice_id" not in invoice_lines.columns:
            return pd.DataFrame()
        if invoice_df.empty or "invoice_id" not in invoice_df.columns:
            return pd.DataFrame()
        invoice_lookup = self._map_supplier_ids(invoice_df, ("supplier_name",))
        join_cols = ["invoice_id"]
        extra_cols = [
            col for col in ("supplier_id", "supplier_name") if col in invoice_lookup.columns
        ]
        if not extra_cols:
            return pd.DataFrame()
        join_df = invoice_lookup[join_cols + extra_cols].drop_duplicates("invoice_id")
        df = invoice_lines.merge(join_df, on="invoice_id", how="left")
        df = self._map_supplier_ids(df, ("supplier_name",))
        if "supplier_id" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["supplier_id"]).copy()
        df["supplier_id"] = df["supplier_id"].astype(str)
        if "total_amount_incl_tax" in df.columns:
            df["total_amount_incl_tax"] = pd.to_numeric(
                df["total_amount_incl_tax"], errors="coerce"
            )
        agg = df.groupby("supplier_id").agg(
            invoice_item_count=("invoice_line_id", "nunique"),
        )
        return agg.reset_index()

    # ------------------------------------------------------------------
    # Profile construction
    # ------------------------------------------------------------------
    def _build_supplier_profiles(
        self, tables: Dict[str, pd.DataFrame], supplier_ids: Iterable[str]
    ) -> Dict[str, Dict]:
        """Build supplier profiles with optimized batch processing."""
        suppliers = {str(s).strip() for s in supplier_ids if str(s).strip()}
        if not suppliers:
            return {}

        # First try batch fetch from cache/database
        profiles = self._batch_fetch_supplier_profiles(suppliers)
        
        # Process any missing suppliers
        flow = tables.get("procurement_flow", pd.DataFrame())
        if not flow.empty:
            missing_suppliers = suppliers - set(profiles.keys())
            if missing_suppliers:
                for supplier_id in missing_suppliers:
                    supplier_flow = flow[flow["supplier_id"] == supplier_id]
                    if not supplier_flow.empty:
                        profile = self._build_single_profile(supplier_flow)
                        if profile:
                            cache_key = f"profile:{supplier_id}"
                            self._cache[cache_key] = profile
                            profiles[supplier_id] = profile

        return profiles

    def _build_single_profile(self, supplier_flow: pd.DataFrame) -> Dict[str, Any]:
        """Build profile for a single supplier."""
        if supplier_flow.empty:
            return {}
            
        descriptions = [
            str(val).strip()
            for val in supplier_flow.get("item_description", pd.Series(dtype="object")).dropna()
            if str(val).strip()
        ]
        description_counter = Counter(descriptions)
        
        return {
            "supplier_id": supplier_flow["supplier_id"].iloc[0],
            "po_ids": sorted({str(val).strip() for val in supplier_flow.get("po_id", pd.Series(dtype="object")).dropna()}),
            "invoice_ids": sorted({str(val).strip() for val in supplier_flow.get("invoice_id", pd.Series(dtype="object")).dropna()}),
            "items": [val for val, _ in description_counter.most_common(5)],
            "primary_item": description_counter.most_common(1)[0][0] if description_counter else None,
            "categories": self._extract_categories(supplier_flow),
            "products": sorted({
                str(val).strip()
                for val in supplier_flow.get("product", pd.Series(dtype="object")).dropna()
                if str(val).strip()
            })
        }

    def _extract_categories(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Extract category information efficiently."""
        categories = {}
        for level in range(1, 6):
            col = f"category_level_{level}"
            if col in df.columns:
                categories[col] = sorted({
                    str(val).strip()
                    for val in df[col].dropna()
                    if str(val).strip()
                })
        return categories

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    def _prepare_scoring_columns(self, df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """Prepare scoring columns with vector similarity if available."""
        result = df.copy()

        # Give the "price" criterion something to score. Nothing ever populated a `price`
        # column outside a deal, so ranking on price silently scored every supplier 0.00
        # even when their historical unit prices were sitting right there in
        # avg_unit_price (derived from PO and invoice lines).
        #
        # A live quote on the deal is the better signal and already sets `price`, so this
        # only fills the gap when there is no competitive bid to use.
        if "price" not in result.columns and "avg_unit_price" in result.columns:
            historical = pd.to_numeric(result["avg_unit_price"], errors="coerce")
            if historical.notna().any():
                result["price"] = historical
                logger.info(
                    "No competitive quote for this deal; scoring price on historical "
                    "avg_unit_price for %d supplier(s)",
                    int(historical.notna().sum()),
                )

        # Add vector similarity scores if available
        if "vector_embedding" in result.columns:
            try:
                embeddings = np.stack(result["vector_embedding"].dropna())
                if len(embeddings) > 1:
                    similarities = np.dot(embeddings, embeddings.T)
                    np.fill_diagonal(similarities, 0)  # Exclude self-similarity
                    result["similarity_score"] = similarities.mean(axis=1)
                    result["final_score"] = result["final_score"] * (1 + result["similarity_score"] * 0.1)
            except Exception:
                logger.warning("Failed to compute vector similarities, continuing without them")

        return result

    def _score_categorical_criteria(
        self,
        df: pd.DataFrame,
        criteria: Iterable[str],
        policies: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        out = df.copy()
        policy = self._find_policy(policies, "CategoricalScoringPolicy")
        if not policy:
            return out
        rules = self._extract_policy_rules(policy)
        for crit in criteria:
            raw_col = crit
            score_col = f"{crit}_score"
            if raw_col in out.columns and crit in rules:
                mapping = rules[crit]
                out[score_col] = out[raw_col].map(mapping).fillna(mapping.get("default", 0))
        return out

    def _normalize_numeric_scores(self, df: pd.DataFrame, dirs: dict) -> pd.DataFrame:
        out = df.copy()
        for crit, direction in dirs.items():
            raw_col = crit
            score_col = f"{crit}_score"
            if raw_col not in df.columns:
                continue
            vals = pd.to_numeric(df[raw_col], errors="coerce")
            if vals.isna().all():
                # Unmeasured, not zero. A 0.0 here asserts "we measured this and it was
                # the worst possible"; NaN says "no data", which is what we actually know.
                # It also lets _normalise_weight_map drop the criterion instead of
                # dragging every supplier's composite score down by its weight.
                out[score_col] = np.nan
                continue
            min_v, max_v = vals.min(), vals.max()
            # 0-100, matching payment_terms_score and price_score. This used to emit a
            # 0-10 scale while payment terms emitted 0-100, so payment terms silently
            # carried ~10x its configured weight in the composite. Harmless while every
            # score was 0.0; a real distortion now that scores carry data.
            if max_v - min_v == 0:
                out[score_col] = 100.0
            else:
                range_diff = float(max_v - min_v)
                if direction == "lower_is_better":
                    out[score_col] = 100 * (max_v - vals) / range_diff
                else:
                    out[score_col] = 100 * (vals - min_v) / range_diff
        return out

    def _normalise_weight_map(
        self, df: pd.DataFrame, weights: Dict[str, float]
    ) -> Dict[str, float]:
        available: Dict[str, float] = {}
        for crit, weight in weights.items():
            if weight <= 0:
                continue
            # Only a criterion with a real _score column can contribute. Accepting one on
            # the strength of its raw column let weights and scoring disagree: the weight
            # map kept the criterion, the scoring loop skipped it for want of a _score
            # column, and its weight silently vanished from the composite.
            score_col = f"{crit}_score"
            if score_col not in df.columns:
                continue
            series = pd.to_numeric(df[score_col], errors="coerce")
            if series.isna().all():
                continue
            available[crit] = float(weight)
        if not available:
            return {}
        total = sum(available.values())
        if total <= 0:
            return {}
        return {crit: value / total for crit, value in available.items()}

    def _build_flow_index(
        self, payload: Any
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        flow_index: Dict[str, Dict[str, Any]] = {}
        name_index: Dict[str, Dict[str, Any]] = {}
        if not isinstance(payload, dict):
            return flow_index, name_index

        candidates: List[Any] = []
        graph = payload.get("graph")
        if isinstance(graph, dict):
            graph_flows = graph.get("supplier_flows") or []
            if isinstance(graph_flows, list):
                candidates.extend(graph_flows)
        direct_flows = payload.get("supplier_flows")
        if isinstance(direct_flows, list):
            candidates.extend(direct_flows)

        for entry in candidates:
            if not isinstance(entry, dict):
                continue
            supplier_id = entry.get("supplier_id")
            if supplier_id is None:
                continue
            sid = str(supplier_id).strip()
            if not sid:
                continue
            flow_index[sid] = entry
            supplier_name = entry.get("supplier_name")
            token = self._normalise_supplier_token(supplier_name)
            if token:
                name_index.setdefault(token, entry)
        return flow_index, name_index

    def _alias_tokens_by_supplier(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        for token, supplier_id in self._supplier_alias_map.items():
            if not supplier_id:
                continue
            mapping.setdefault(supplier_id, []).append(token)
        return mapping

    def _lookup_flow_entry(
        self,
        supplier_id: Optional[Any],
        supplier_name: Optional[Any],
        flow_index: Dict[str, Dict[str, Any]],
        flow_name_index: Dict[str, Dict[str, Any]],
        alias_tokens: Dict[str, List[str]],
    ) -> Optional[Dict[str, Any]]:
        sid_token = str(supplier_id).strip() if supplier_id is not None else ""
        if sid_token and sid_token in flow_index:
            return flow_index.get(sid_token)

        tokens: List[str] = []
        if supplier_name:
            token = self._normalise_supplier_token(supplier_name)
            if token:
                tokens.append(token)
        if sid_token and sid_token in self._supplier_lookup:
            lookup_name = self._supplier_lookup.get(sid_token)
            token = self._normalise_supplier_token(lookup_name)
            if token:
                tokens.append(token)
        for token in alias_tokens.get(sid_token, []):
            if token:
                tokens.append(token)

        for token in dict.fromkeys(tokens):
            flow = flow_name_index.get(token)
            if flow:
                return flow
        return None

    @staticmethod
    def _coverage_from_flow_entry(entry: Optional[Dict[str, Any]]) -> float:
        if not isinstance(entry, dict):
            return 0.0
        coverage = entry.get("coverage_ratio")
        if isinstance(coverage, (int, float)):
            value = float(coverage)
            if value < 0:
                return 0.0
            if value > 1.0:
                return 1.0
            return value
        total = 0
        hits = 0
        for key in ("contracts", "purchase_orders", "invoices", "quotes"):
            details = entry.get(key)
            if isinstance(details, dict):
                total += 1
                if details.get("count"):
                    hits += 1
        return hits / total if total else 0.0

    def _annotate_flow_coverage(
        self,
        df: pd.DataFrame,
        flow_index: Dict[str, Dict[str, Any]],
        flow_name_index: Dict[str, Dict[str, Any]],
        alias_tokens: Dict[str, List[str]],
    ) -> pd.DataFrame:
        if df.empty or "supplier_id" not in df.columns:
            if "flow_coverage" not in df.columns:
                df["flow_coverage"] = 0.0
            return df
        annotated = df.copy()
        coverage_values: List[float] = []
        for _, row in annotated.iterrows():
            entry = self._lookup_flow_entry(
                row.get("supplier_id"),
                row.get("supplier_name"),
                flow_index,
                flow_name_index,
                alias_tokens,
            )
            coverage_values.append(self._coverage_from_flow_entry(entry))
        if "flow_coverage" in annotated.columns:
            baseline = (
                pd.to_numeric(annotated["flow_coverage"], errors="coerce")
                .fillna(0.0)
                .tolist()
            )
        else:
            baseline = [0.0] * len(annotated)
        combined = [max(base, cov) for base, cov in zip(baseline, coverage_values)]
        annotated["flow_coverage"] = combined
        return annotated

    def _apply_flow_bonus(
        self,
        df: pd.DataFrame,
        flow_index: Dict[str, Dict[str, Any]],
        flow_name_index: Dict[str, Dict[str, Any]],
        alias_tokens: Dict[str, List[str]],
    ) -> pd.DataFrame:
        if df.empty or "supplier_id" not in df.columns:
            if "flow_coverage" not in df.columns:
                df["flow_coverage"] = 0.0
            return df
        augmented = df.copy()
        coverage_values: List[float] = []
        for _, row in augmented.iterrows():
            entry = self._lookup_flow_entry(
                row.get("supplier_id"),
                row.get("supplier_name"),
                flow_index,
                flow_name_index,
                alias_tokens,
            )
            coverage = self._coverage_from_flow_entry(entry)
            coverage_values.append(coverage)
        augmented["flow_coverage"] = coverage_values
        # Coverage is reported, not scored. This used to multiply final_score by up to
        # 1.1x, which was inert while every score was 0.0 but would silently reorder
        # suppliers now that scores are real -- a supplier could out-rank a cheaper rival
        # on nothing but having more documents on file. A ranking a buyer cannot trace
        # back to the quoted prices is not one they can defend, so the bonus stays off
        # until it is a deliberate, tested choice.
        return augmented

    @staticmethod
    def _negotiation_note(row: pd.Series) -> str:
        """The deal's price story in plain figures. Narrated, never scored.

        Rewarding a big concession would reward opening high, so movement across rounds
        stays out of the score -- but a buyer still needs to see it to read the room.
        """
        final_amount = row.get("final_quote_amount")
        if not isinstance(final_amount, (int, float)) or pd.isna(final_amount):
            return ""

        currency = row.get("quote_currency") or ""
        symbol = {"GBP": "£", "USD": "$", "EUR": "€"}.get(str(currency).upper(), "")
        unit = symbol if symbol else (f"{currency} " if currency else "")

        note = f"Standing offer: {unit}{final_amount:,.2f}"
        quote_id = row.get("winning_quote_id")
        if isinstance(quote_id, str) and quote_id.strip():
            note += f" ({quote_id.strip()})"

        rounds = row.get("quote_rounds")
        opening = row.get("opening_quote_amount")
        concession = row.get("concession_pct")
        if (
            isinstance(rounds, (int, float))
            and rounds > 1
            and isinstance(opening, (int, float))
            and isinstance(concession, (int, float))
            and not pd.isna(concession)
        ):
            direction = "down" if concession > 0 else "up"
            note += (
                f", {direction} {abs(concession):.1f}% from their {unit}{opening:,.2f} "
                f"opener across {int(rounds)} rounds"
            )
        note += "."
        return note

    def _generate_justification(self, row: pd.Series, criteria: Iterable[str]) -> str:
        # Lead with facts we can point at in the source documents. The LLM's job is to
        # phrase them, not to supply them -- when it was handed nothing but a score it
        # produced "achieved a final score of 0.00. Risk: N/A. No further details."
        scored_on = list(row.get("scored_on") or [])
        breakdown = []
        for crit in criteria:
            score_col = f"{crit}_score"
            if score_col in row:
                score_value = row.get(score_col)
                if isinstance(score_value, (int, float)) and not pd.isna(score_value):
                    breakdown.append(f"- {crit.replace('_', ' ').title()}: {score_value:.2f}/100")
                else:
                    breakdown.append(
                        f"- {crit.replace('_', ' ').title()}: not measured (no data on file)"
                    )

        negotiation = self._negotiation_note(row)
        final_score = row.get("final_score")
        facts = []
        if negotiation:
            facts.append(negotiation)
        if scored_on:
            facts.append(
                "Scored on: " + ", ".join(c.replace("_", " ") for c in scored_on) + "."
            )

        # A deterministic, fully-grounded justification. Used verbatim if the LLM is
        # unavailable, so a ranking is never left unexplained.
        deterministic = " ".join(facts) if facts else "No measurable data for this supplier."

        if not self.justification_template:
            return deterministic

        score_text = (
            f"{final_score:.2f}/100"
            if isinstance(final_score, (int, float)) and not pd.isna(final_score)
            else "not scored (no measurable data)"
        )
        try:
            prompt = self.justification_template["prompt_template"].format(
                supplier_name=row.get("supplier_name", "Unknown"),
                final_score=score_text,
                score_breakdown="\n".join(breakdown + ([negotiation] if negotiation else [])),
            )
        except KeyError:
            logger.warning("Justification template missing an expected field; using facts only")
            return deterministic
        try:
            fallback_model = getattr(self.settings, "extraction_model", None)
            resolver = getattr(self.agent_nick, "get_agent_model", None)
            model_name = fallback_model
            if callable(resolver):
                try:
                    candidate = resolver(
                        self.__class__.__name__, fallback=fallback_model
                    )
                except Exception:  # pragma: no cover - defensive logging
                    logger.debug(
                        "SupplierRankingAgent model resolution failed", exc_info=True
                    )
                else:
                    if isinstance(candidate, str) and candidate.strip():
                        model_name = candidate.strip()
            resp = self.call_ollama(prompt, model=model_name)
            text = (resp.get("response") or "").strip()
            # Fall back to the grounded facts rather than shipping an empty or failed
            # justification: an unexplained ranking is not actionable for a buyer.
            return text if text else deterministic
        except Exception:
            logger.exception("Justification generation failed; falling back to source facts")
            return deterministic

    def _prepare_ranking_entry(
        self, row: pd.Series, profile: Optional[Dict], weights: Dict[str, float]
    ) -> Dict:
        # Coerce score/metric fields to JSON-native types so that pandas NA,
        # numpy scalars, and other non-serialisable objects never leak into the
        # API response.
        # NULL, never 0.0, for an unmeasured score. 0.0 asserts "we measured this supplier
        # and they scored bottom"; NULL says "we have no data on them". Collapsing the
        # second into the first is how this agent came to publish a table of confident
        # 0.00s about suppliers it knew nothing about.
        final_score_safe = _json_safe(row.get("final_score"))
        entry = {
            "supplier_id": _json_safe(row.get("supplier_id")),
            "supplier_name": _json_safe(row.get("supplier_name")),
            "final_score": final_score_safe if isinstance(final_score_safe, (int, float)) else None,
            "scored_on": list(row.get("scored_on") or []),
            "price_score": _json_safe(row.get("price_score")),
            "final_quote_amount": _json_safe(row.get("final_quote_amount")),
            "opening_quote_amount": _json_safe(row.get("opening_quote_amount")),
            "quote_currency": _json_safe(row.get("quote_currency")),
            "quote_rounds": _json_safe(row.get("quote_rounds")),
            "concession_pct": _json_safe(row.get("concession_pct")),
            "winning_quote_id": _json_safe(row.get("winning_quote_id")),
            "delivery_score": _json_safe(row.get("delivery_score")),
            "risk_score": _json_safe(row.get("risk_score")),
            "payment_terms_score": _json_safe(row.get("payment_terms_score")),
            "payment_terms": _json_safe(row.get("payment_terms")),
            "avg_unit_price": _json_safe(row.get("avg_unit_price")),
            "total_spend": _json_safe(row.get("total_spend")),
            "po_count": _json_safe(row.get("po_count")),
            "invoice_count": _json_safe(row.get("invoice_count")),
            "lead_time_days": _json_safe(row.get("avg_lead_time_days")),
            "justification": _json_safe(row.get("justification")),
            "contact_name": _json_safe(row.get("contact_name_1")),
            "contact_email": _json_safe(row.get("contact_email_1")),
            "weights": dict(weights),
        }
        coverage = _json_safe(row.get("flow_coverage"))
        if isinstance(coverage, (int, float)):
            entry["flow_coverage"] = float(coverage)
        relationship_cov = row.get("relationship_coverage")
        if isinstance(relationship_cov, (int, float)):
            entry["relationship_coverage"] = float(relationship_cov)
        relationship_summary = row.get("relationship_summary")
        if isinstance(relationship_summary, str) and relationship_summary.strip():
            entry["relationship_summary"] = relationship_summary.strip()
        relationship_statements = row.get("relationship_statements")
        if isinstance(relationship_statements, list) and relationship_statements:
            cleaned_statements = [
                statement.strip()
                for statement in relationship_statements
                if isinstance(statement, str) and statement.strip()
            ]
            if cleaned_statements:
                entry["relationship_statements"] = cleaned_statements
        if profile:
            entry.update(
                {
                    "po_ids": profile.get("po_ids", []),
                    "invoice_ids": profile.get("invoice_ids", []),
                    "primary_item": profile.get("primary_item"),
                    "items": profile.get("items", []),
                    "categories": profile.get("categories", {}),
                    "products": profile.get("products", []),
                }
            )
        return entry

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _mode_value(self, series: pd.Series) -> Optional[str]:
        try:
            cleaned = series.dropna().astype(str)
            if cleaned.empty:
                return None
            return cleaned.mode().iloc[0]
        except Exception:
            return None

    def _top_values(self, series: pd.Series, limit: int = 5) -> List[str]:
        cleaned = [
            str(val).strip()
            for val in series.dropna()
            if isinstance(val, str) and str(val).strip()
        ]
        counter = Counter(cleaned)
        return [val for val, _ in counter.most_common(limit)]

    def _fetch_relationship_context(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        service = getattr(self, "_relationship_service", None)
        if service is None or df.empty:
            return {}, {}

        contexts_by_id: Dict[str, Dict[str, Any]] = {}
        contexts_by_name: Dict[str, Dict[str, Any]] = {}
        seen: set[Tuple[str, str]] = set()

        for row in df.itertuples(index=False):
            sid = str(getattr(row, "supplier_id", "") or "").strip()
            name = getattr(row, "supplier_name", None)
            key = (sid, str(name or "").strip())
            if key in seen:
                continue
            seen.add(key)
            try:
                payloads = service.fetch_relationship(
                    supplier_id=sid or None,
                    supplier_name=name,
                    limit=1,
                )
            except Exception:
                logger.exception("Failed to load relationship context for supplier %s", sid or name)
                continue
            if not payloads:
                continue
            payload = payloads[0]
            if sid:
                contexts_by_id[sid] = payload
            token = self._normalise_supplier_token(name)
            if token:
                contexts_by_name[token] = payload

        return contexts_by_id, contexts_by_name

    def _merge_relationship_context(
        self,
        df: pd.DataFrame,
        contexts_by_id: Dict[str, Dict[str, Any]],
        contexts_by_name: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        if df.empty:
            return df

        augmented = df.copy()
        summaries: List[Optional[str]] = []
        statements_list: List[List[str]] = []
        coverage_values: List[Optional[float]] = []

        for _, row in augmented.iterrows():
            sid = str(row.get("supplier_id") or "").strip()
            name = row.get("supplier_name")
            payload = contexts_by_id.get(sid)
            if payload is None:
                token = self._normalise_supplier_token(name)
                if token:
                    payload = contexts_by_name.get(token)

            if isinstance(payload, dict):
                summary = payload.get("summary") or payload.get("content")
                summaries.append(summary.strip() if isinstance(summary, str) else None)
                rel_statements_raw = payload.get("relationship_statements")
                if isinstance(rel_statements_raw, list):
                    cleaned = [
                        statement.strip()
                        for statement in rel_statements_raw
                        if isinstance(statement, str) and statement.strip()
                    ]
                else:
                    cleaned = []
                statements_list.append(cleaned)
                coverage_raw = payload.get("coverage_ratio")
                try:
                    coverage_values.append(float(coverage_raw))
                except (TypeError, ValueError):
                    coverage_values.append(None)
            else:
                summaries.append(None)
                statements_list.append([])
                coverage_values.append(None)

        augmented["relationship_summary"] = summaries
        augmented["relationship_statements"] = statements_list

        if "flow_coverage" in augmented.columns:
            merged_coverage: List[Optional[float]] = []
            for base, extra in zip(augmented["flow_coverage"], coverage_values):
                if isinstance(extra, (int, float)):
                    base_val = float(base) if isinstance(base, (int, float)) else 0.0
                    merged_coverage.append(max(base_val, float(extra)))
                else:
                    merged_coverage.append(
                        float(base) if isinstance(base, (int, float)) else 0.0
                    )
            augmented["flow_coverage"] = merged_coverage
        else:
            augmented["flow_coverage"] = [
                float(value) if isinstance(value, (int, float)) else 0.0
                for value in coverage_values
            ]
        augmented["relationship_coverage"] = [
            float(value) if isinstance(value, (int, float)) else None
            for value in coverage_values
        ]

        return augmented

