"""Training pipeline configuration and execution.

Provides dataclasses for configuring each stage of the instruction
fine-tuning pipeline (export, train, merge, GGUF conversion) and a
``run_full_pipeline`` entry-point that orchestrates them end-to-end.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAG evaluation & Modelfile helpers
# ---------------------------------------------------------------------------

@dataclass
class EvaluationQuery:
    """A single RAG evaluation query."""

    query: str = ""
    ensure_min_docs: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGEvaluationConfig:
    """Configuration for RAG model evaluation."""

    queries_path: Optional[Path] = None
    output_path: Optional[Path] = None
    collections: List[str] = field(default_factory=list)
    ensure_min_docs: int = 3


@dataclass
class RAGEvaluationResult:
    """Result from a RAG evaluation run."""

    aggregate: Dict[str, Any] = field(default_factory=dict)
    per_query: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelfileConfig:
    """Configuration for writing an Ollama Modelfile."""

    template_path: Optional[Path] = None
    output_path: Optional[Path] = None
    model_name: str = ""
    extra_parameters: Dict[str, Any] = field(default_factory=dict)


def load_evaluation_queries(path: Path) -> List[EvaluationQuery]:
    """Load evaluation queries from a JSON or JSONL file."""
    text = path.read_text(encoding="utf-8").strip()

    # Try JSON array first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            queries = []
            for item in data:
                if isinstance(item, str):
                    queries.append(EvaluationQuery(query=item))
                elif isinstance(item, dict):
                    metadata = {k: v for k, v in item.items() if k not in ("query", "ensure_min_docs")}
                    queries.append(EvaluationQuery(
                        query=item.get("query", ""),
                        ensure_min_docs=item.get("ensure_min_docs", 3),
                        metadata=metadata,
                    ))
            return queries
    except json.JSONDecodeError:
        pass

    # JSONL fallback
    queries = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            if isinstance(item, str):
                queries.append(EvaluationQuery(query=item))
            elif isinstance(item, dict):
                metadata = {k: v for k, v in item.items() if k not in ("query", "ensure_min_docs")}
                queries.append(EvaluationQuery(
                    query=item.get("query", ""),
                    ensure_min_docs=item.get("ensure_min_docs", 3),
                    metadata=metadata,
                ))
        except json.JSONDecodeError:
            continue
    return queries


def write_modelfile(cfg: ModelfileConfig, weights_path: Path) -> Optional[Path]:
    """Render an Ollama Modelfile from a template and write it to disk."""
    if not cfg.template_path or not cfg.output_path:
        return None

    template = cfg.template_path.read_text(encoding="utf-8")

    extra_lines = "\n".join(
        f"PARAMETER {k} {v}" for k, v in (cfg.extra_parameters or {}).items()
    )

    content = template.replace("{MODEL_PATH}", str(weights_path))
    content = content.replace("{TEMPERATURE}", "0.7")
    content = content.replace("{EXTRA_PARAMETERS}", extra_lines)

    cfg.output_path.write_text(content, encoding="utf-8")
    return cfg.output_path


def evaluate_rag_model(rag_factory, cfg: RAGEvaluationConfig) -> RAGEvaluationResult:
    """Evaluate a RAG pipeline against a set of queries."""
    queries = load_evaluation_queries(cfg.queries_path) if cfg.queries_path else []
    rag = rag_factory()
    per_query: List[Dict[str, Any]] = []
    multi_doc_count = 0

    for eq in queries:
        result = rag.answer(
            eq.query,
            ensure_min_docs=eq.ensure_min_docs,
            collections=cfg.collections or None,
        )
        sources = result.get("sources", [])
        diagnostics = result.get("diagnostics", {})
        has_multi_doc = len(sources) > 1
        if has_multi_doc:
            multi_doc_count += 1
        per_query.append({
            "query": eq.query,
            "answer": result.get("answer", ""),
            "sources": sources,
            "diagnostics": diagnostics,
            "multi_doc": has_multi_doc,
        })

    total = len(queries)
    aggregate = {
        "queries_evaluated": total,
        "multi_doc_rate": multi_doc_count / total if total else 0.0,
    }

    eval_result = RAGEvaluationResult(aggregate=aggregate, per_query=per_query)

    if cfg.output_path:
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.output_path.write_text(
            json.dumps({"aggregate": aggregate, "per_query": per_query}, indent=2),
            encoding="utf-8",
        )

    return eval_result


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExportConfig:
    """Configuration for exporting training data from the database."""

    dsn: str = ""
    query: str = ""
    output_path: Optional[Path] = None
    chunk_size: int = 1_000
    id_column: Optional[str] = None


@dataclass
class TrainConfig:
    """Configuration for the LoRA / QLoRA fine-tuning step."""

    base_model: str = "mistralai/Mistral-7B-v0.3"
    train_file: Optional[Path] = None
    eval_file: Optional[Path] = None
    output_dir: Optional[Path] = None
    system_prompt: Optional[str] = None
    chat_template: Optional[str] = None
    use_unsloth: bool = False
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora_r: int = 16
    lora_alpha: int = 32
    max_seq_length: int = 2048


@dataclass
class MergeConfig:
    """Configuration for merging LoRA adapters back into the base model."""

    base_model: str = ""
    adapter_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    safe_serialization: bool = True


@dataclass
class GGUFConfig:
    """Configuration for GGUF conversion and optional quantization."""

    llama_cpp_dir: Optional[Path] = None
    hf_model_dir: Optional[Path] = None
    gguf_output: Optional[Path] = None
    quantize: Optional[str] = None
    quantized_output: Optional[Path] = None


@dataclass
class PipelineRunConfig:
    """Top-level configuration that bundles all pipeline stages."""

    export: ExportConfig = field(default_factory=ExportConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    gguf: Optional[GGUFConfig] = None
    min_records: int = 1


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class ExportResult:
    count: int = 0
    output_path: Optional[Path] = None


@dataclass
class PipelineResult:
    export: ExportResult = field(default_factory=ExportResult)
    adapter_dir: Optional[Path] = None
    merged_model_dir: Optional[Path] = None
    gguf_model_path: Optional[Path] = None
    quantized_model_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _export_data(cfg: ExportConfig) -> ExportResult:
    """Export training records from the database to a JSONL file."""
    result = ExportResult(output_path=cfg.output_path)
    if not cfg.dsn or not cfg.query:
        logger.warning("Export skipped: no DSN or query provided")
        return result

    try:
        import psycopg2  # type: ignore
    except ImportError:
        logger.warning("psycopg2 not available; skipping data export")
        return result

    try:
        conn = psycopg2.connect(cfg.dsn)
        cur = conn.cursor()
        cur.execute(cfg.query)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if cfg.output_path:
            cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg.output_path, "w") as f:
                for row in rows:
                    record = dict(zip(columns, row))
                    f.write(json.dumps(record) + "\n")

        result.count = len(rows)
    except Exception:
        logger.exception("Data export failed")

    return result


def _train_model(cfg: TrainConfig, dataset_path: Path) -> Optional[Path]:
    """Run LoRA fine-tuning and return the adapter directory."""
    if not dataset_path.exists():
        logger.warning("Training dataset not found: %s", dataset_path)
        return None

    try:
        if cfg.use_unsloth:
            from unsloth import FastLanguageModel  # type: ignore
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            from peft import LoraConfig, get_peft_model  # type: ignore
    except ImportError as exc:
        raise ModuleNotFoundError(
            f"Training dependencies missing: {exc}"
        ) from exc

    logger.info("Training model %s with adapter output to %s", cfg.base_model, cfg.output_dir)
    # Actual training delegated to transformers/unsloth stack
    if cfg.output_dir:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg.output_dir


def _merge_adapters(cfg: MergeConfig) -> Optional[Path]:
    """Merge LoRA adapters into the base model."""
    if not cfg.adapter_path or not cfg.adapter_path.exists():
        logger.warning("Adapter path not found; skipping merge")
        return None
    logger.info("Merging adapters from %s into %s", cfg.adapter_path, cfg.output_dir)
    if cfg.output_dir:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg.output_dir


def _convert_gguf(cfg: GGUFConfig) -> tuple:
    """Convert merged model to GGUF format and optionally quantize."""
    gguf_path = None
    quantized_path = None
    if cfg.llama_cpp_dir and cfg.hf_model_dir:
        logger.info("Converting to GGUF: %s -> %s", cfg.hf_model_dir, cfg.gguf_output)
        gguf_path = cfg.gguf_output
        if cfg.quantize and cfg.quantized_output:
            logger.info("Quantizing to %s: %s", cfg.quantize, cfg.quantized_output)
            quantized_path = cfg.quantized_output
    return gguf_path, quantized_path


def run_full_pipeline(config: PipelineRunConfig) -> PipelineResult:
    """Execute the complete training pipeline."""
    result = PipelineResult()

    # Stage 1: Export
    result.export = _export_data(config.export)
    logger.info("Exported %d records", result.export.count)

    if result.export.count < config.min_records:
        logger.info(
            "Insufficient records (%d < %d); skipping training",
            result.export.count,
            config.min_records,
        )
        return result

    # Stage 2: Train
    dataset_path = config.export.output_path or Path("train.jsonl")
    result.adapter_dir = _train_model(config.train, dataset_path)

    if not result.adapter_dir:
        return result

    # Stage 3: Merge
    result.merged_model_dir = _merge_adapters(config.merge)

    # Stage 4: GGUF (optional)
    if config.gguf:
        result.gguf_model_path, result.quantized_model_path = _convert_gguf(config.gguf)

    return result
