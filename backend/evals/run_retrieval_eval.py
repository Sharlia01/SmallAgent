#!/usr/bin/env python3
"""Run the production retriever against a versioned JSONL evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
BACKEND_DIRECTORY = SCRIPT_DIRECTORY.parent
APP_DIRECTORY = BACKEND_DIRECTORY / "app"
DEFAULT_DATASET = SCRIPT_DIRECTORY / "data" / "guodian_power_eval_v1.jsonl"
DEFAULT_OUTPUT_DIRECTORY = SCRIPT_DIRECTORY / "results"

if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))

from service.core.retrieval_evaluation import (  # noqa: E402
    build_error_case,
    build_summary,
    evaluate_retrieval_case,
)
from service.core.evidence_sufficiency import (  # noqa: E402
    DEFAULT_EVIDENCE_SUFFICIENCY_MODEL,
)
from service.core.rag.nlp.model import RERANKER_MODEL  # noqa: E402


REQUIRED_SAMPLE_FIELDS = {
    "schema_version",
    "dataset_version",
    "id",
    "question",
    "reference_answer",
    "answerable",
    "question_type",
    "relevant_evidence",
    "metadata",
}


def environment_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().casefold() not in {"0", "false", "no", "off"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the production Elasticsearch retriever without calling "
            "the chat model."
        )
    )
    parser.add_argument(
        "--index-name",
        action="append",
        help=(
            "Elasticsearch index/tenant name. For the current app this is the "
            "authenticated user's numeric ID. Repeat or use commas for more "
            "than one index. Required unless --validate-only is used."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Evaluation JSONL path (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"Result directory (default: {DEFAULT_OUTPUT_DIRECTORY})",
    )
    parser.add_argument(
        "--run-name",
        default="retrieval_baseline",
        help="Output filename prefix. Only letters, digits, dot, dash and underscore.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--similarity-threshold", type=float, default=0.1)
    parser.add_argument(
        "--vector-weight",
        type=float,
        default=0.6,
        help="Vector branch weight in weighted RRF (default: 0.6).",
    )
    parser.add_argument(
        "--candidate-size",
        type=int,
        default=100,
        help="Candidate count retrieved independently by each branch.",
    )
    parser.add_argument(
        "--rerank-candidate-size",
        type=int,
        default=20,
        help="Top RRF candidates sent to semantic reranking.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF rank smoothing constant (default: 60).",
    )
    parser.add_argument(
        "--final-reranker-weight",
        type=float,
        default=0.7,
        help=(
            "Semantic reranker weight in second-stage rank fusion "
            "(default: 0.7)."
        ),
    )
    parser.add_argument(
        "--final-rrf-k",
        type=int,
        default=10,
        help="Second-stage RRF rank smoothing constant (default: 10).",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.8,
        help="Minimum normalized evidence coverage required for a gold match.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        help="Run only selected sample IDs. May be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N selected samples (useful for a smoke test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing result files with the same run name.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first retrieval error instead of recording it.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and summarize the dataset without calling retrieval.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0")
    if not 0 <= args.similarity_threshold <= 1:
        raise ValueError("--similarity-threshold must be between 0 and 1")
    if not 0 <= args.vector_weight <= 1:
        raise ValueError("--vector-weight must be between 0 and 1")
    if args.candidate_size <= 0:
        raise ValueError("--candidate-size must be greater than 0")
    if args.rerank_candidate_size <= 0:
        raise ValueError("--rerank-candidate-size must be greater than 0")
    if args.rrf_k <= 0:
        raise ValueError("--rrf-k must be greater than 0")
    if not 0 <= args.final_reranker_weight <= 1:
        raise ValueError("--final-reranker-weight must be between 0 and 1")
    if args.final_rrf_k <= 0:
        raise ValueError("--final-rrf-k must be greater than 0")
    if not 0 <= args.match_threshold <= 1:
        raise ValueError("--match-threshold must be between 0 and 1")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_name):
        raise ValueError("--run-name contains unsupported characters")


def expand_index_names(values: list[str]) -> list[str]:
    names = []
    for value in values:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise ValueError("At least one non-empty --index-name is required")
    return names


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples = []
    identifiers = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON on dataset line {line_number}: {error}"
            ) from error
        if not isinstance(sample, dict):
            raise ValueError(f"Dataset line {line_number} is not a JSON object")

        missing_fields = REQUIRED_SAMPLE_FIELDS - set(sample)
        if missing_fields:
            raise ValueError(
                f"Dataset line {line_number} is missing fields: "
                f"{sorted(missing_fields)}"
            )
        if not isinstance(sample["answerable"], bool):
            raise ValueError(
                f"Dataset line {line_number} has a non-boolean answerable field"
            )
        if not isinstance(sample["relevant_evidence"], list):
            raise ValueError(
                f"Dataset line {line_number} has invalid relevant_evidence"
            )
        for evidence_index, evidence in enumerate(
            sample["relevant_evidence"],
            start=1,
        ):
            if not isinstance(evidence, dict):
                raise ValueError(
                    f"Sample {sample['id']!r} evidence {evidence_index} is not "
                    "a JSON object"
                )
            missing_evidence_fields = {
                "document_name",
                "page",
                "text",
            } - set(evidence)
            if missing_evidence_fields:
                raise ValueError(
                    f"Sample {sample['id']!r} evidence {evidence_index} is "
                    f"missing fields: {sorted(missing_evidence_fields)}"
                )
            if not str(evidence["document_name"]).strip():
                raise ValueError(
                    f"Sample {sample['id']!r} evidence {evidence_index} has "
                    "an empty document_name"
                )
            if not isinstance(evidence["page"], int) or evidence["page"] <= 0:
                raise ValueError(
                    f"Sample {sample['id']!r} evidence {evidence_index} has "
                    "an invalid page"
                )
            if not str(evidence["text"]).strip():
                raise ValueError(
                    f"Sample {sample['id']!r} evidence {evidence_index} has "
                    "empty text"
                )
        if sample["answerable"] and not sample["relevant_evidence"]:
            raise ValueError(
                f"Answerable sample {sample['id']!r} has no gold evidence"
            )
        if not sample["answerable"] and sample["relevant_evidence"]:
            raise ValueError(
                f"Unanswerable sample {sample['id']!r} has gold evidence"
            )
        if sample["id"] in identifiers:
            raise ValueError(f"Duplicate sample ID: {sample['id']!r}")
        identifiers.add(sample["id"])
        samples.append(sample)

    if not samples:
        raise ValueError("Dataset is empty")
    return samples


def select_samples(
    samples: list[dict[str, Any]],
    sample_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = samples
    if sample_ids:
        requested = set(sample_ids)
        available = {str(sample["id"]) for sample in samples}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Unknown --sample-id values: {missing}")
        selected = [sample for sample in samples if sample["id"] in requested]
    if limit is not None:
        selected = selected[:limit]
    return selected


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def write_results(
    result_path: Path,
    summary_path: Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    jsonl_content = "".join(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
        for result in results
    )
    atomic_write_text(result_path, jsonl_content)
    atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )


def load_retrieval_function():
    """Load service dependencies only for an actual retrieval run."""
    from dotenv import load_dotenv

    load_dotenv(BACKEND_DIRECTORY / ".env")
    from service.core.retrieval import retrieve_raw_results

    return retrieve_raw_results


def main() -> int:
    args = parse_args()
    validate_args(args)
    dataset_path = args.dataset.resolve()
    #加载数据集并筛选样本
    samples = load_dataset(dataset_path)
    samples = select_samples(samples, args.sample_id, args.limit)

    if args.validate_only:
        print(
            json.dumps(
                {
                    "dataset": str(dataset_path),
                    "selected_samples": len(samples),
                    "dataset_versions": sorted(
                        {
                            str(sample.get("dataset_version", "unknown"))
                            for sample in samples
                        }
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    index_names = expand_index_names(args.index_name or [])
    retrieve_raw_results = load_retrieval_function()

    output_directory = args.output_dir.resolve()
    result_path = output_directory / f"{args.run_name}.jsonl"
    summary_path = output_directory / f"{args.run_name}_summary.json"
    existing_outputs = [path for path in (result_path, summary_path) if path.exists()]
    #避免误刷历史结果
    if existing_outputs and not args.overwrite:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            f"Result files already exist: {paths}. Use --overwrite or another "
            "--run-name."
        )

    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    results = []
    retrieval_index_argument: str | list[str] = (
        index_names[0] if len(index_names) == 1 else index_names
    )

    for position, sample in enumerate(samples, start=1):
        query_started = time.perf_counter()
        try:
            raw_result = retrieve_raw_results(
                retrieval_index_argument,
                str(sample["question"]),
                page_size=args.top_k,
                similarity_threshold=args.similarity_threshold,
                vector_similarity_weight=args.vector_weight,
                candidate_size=args.candidate_size,
                rerank_candidate_size=args.rerank_candidate_size,
                rrf_k=args.rrf_k,
                final_reranker_weight=args.final_reranker_weight,
                final_rrf_k=args.final_rrf_k,
            )
            latency_ms = (time.perf_counter() - query_started) * 1000
            result = evaluate_retrieval_case(
                sample,
                raw_result,
                latency_ms=latency_ms,
                top_k=args.top_k,
                match_threshold=args.match_threshold,
            )
            hit_value = result["metrics"][f"hit_at_{args.top_k}"]
            hit_label = "N/A" if hit_value is None else str(hit_value)
            print(
                f"[{position}/{len(samples)}] {sample['id']} "
                f"retrieved={result['retrieval']['retrieved_count']} "
                f"hit={hit_label} latency_ms={result['latency_ms']:.3f}",
                flush=True,
            )
        except Exception as error:
            latency_ms = (time.perf_counter() - query_started) * 1000
            if args.fail_fast:
                raise
            result = build_error_case(
                sample,
                error=error,
                latency_ms=latency_ms,
                top_k=args.top_k,
            )
            print(
                f"[{position}/{len(samples)}] {sample['id']} ERROR: "
                f"{result['error']}",
                file=sys.stderr,
                flush=True,
            )
        results.append(result)

    completed_at = datetime.now(timezone.utc)
    metric_summary = build_summary(results, top_k=args.top_k)
    summary = {
        "run_name": args.run_name,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": round(time.perf_counter() - started_clock, 3),
        "dataset": {
            "path": str(dataset_path),
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "selected_sample_count": len(samples),
            "versions": sorted(
                {
                    str(sample.get("dataset_version", "unknown"))
                    for sample in samples
                }
            ),
        },
        "retrieval_config": {
            "index_names": index_names,
            "top_k": args.top_k,
            "similarity_threshold": args.similarity_threshold,
            "vector_rrf_weight": args.vector_weight,
            "candidate_size_per_branch": args.candidate_size,
            "rerank_candidate_size": args.rerank_candidate_size,
            "reranker_model": RERANKER_MODEL,
            "rrf_k": args.rrf_k,
            "final_reranker_weight": args.final_reranker_weight,
            "final_retrieval_rrf_weight": (
                1.0 - args.final_reranker_weight
            ),
            "final_rrf_k": args.final_rrf_k,
            "evidence_match_threshold": args.match_threshold,
            "evidence_sufficiency_enabled": environment_flag(
                "RAG_EVIDENCE_SUFFICIENCY_ENABLED",
                True,
            ),
            "evidence_sufficiency_model": os.getenv(
                "RAG_EVIDENCE_SUFFICIENCY_MODEL",
                DEFAULT_EVIDENCE_SUFFICIENCY_MODEL,
            ),
            "evidence_sufficiency_fail_open": environment_flag(
                "RAG_EVIDENCE_SUFFICIENCY_FAIL_OPEN",
                True,
            ),
        },
        "result_file": str(result_path),
        "metrics": metric_summary,
    }
    write_results(result_path, summary_path, results, summary)

    print(f"Results: {result_path}")
    print(f"Summary: {summary_path}")
    print(
        json.dumps(metric_summary["overall"], ensure_ascii=False, indent=2),
        flush=True,
    )
    return 1 if metric_summary["overall"]["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
