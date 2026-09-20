#!/usr/bin/env python3
"""Run retrieval, answer, faithfulness, citation and refusal evaluation."""

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
DEFAULT_DATASET = SCRIPT_DIRECTORY / "data" / "power_reports_eval_v2.jsonl"
DEFAULT_OUTPUT_DIRECTORY = SCRIPT_DIRECTORY / "results"

if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from openai import OpenAI  # noqa: E402

from run_retrieval_eval import (  # noqa: E402
    atomic_write_text,
    expand_index_names,
    load_dataset,
    select_samples,
)
from service.core.citation_evaluation import evaluate_citations  # noqa: E402
from service.core.end_to_end_evaluation import (  # noqa: E402
    E2E_RESULT_SCHEMA_VERSION,
    build_e2e_summary,
    evaluate_answer_quality,
    evaluate_refusal,
    evaluate_sufficiency,
)
from service.core.faithfulness_evaluation import (  # noqa: E402
    evaluate_faithfulness,
)
from service.core.grounded_answer import (  # noqa: E402
    DEFAULT_ANSWER_MODEL,
    execute_grounded_turn,
)
from service.core.retrieval_evaluation import (  # noqa: E402
    build_error_case,
    evaluate_retrieval_case,
    resolve_metric_ks,
)


DEFAULT_JUDGE_MODEL = "qwen3.7-flash-2026-07-15"


def environment_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().casefold() not in {"0", "false", "no", "off"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the production KB retriever, answer prompt, "
            "faithfulness, citations and deterministic refusal policy "
            "without DB/Redis side effects."
        )
    )
    parser.add_argument("--index-name", action="append", required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--run-name", default="e2e_baseline")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--similarity-threshold", type=float, default=0.1)
    parser.add_argument("--vector-weight", type=float, default=0.6)
    parser.add_argument("--candidate-size", type=int, default=100)
    parser.add_argument("--rerank-candidate-size", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--final-reranker-weight", type=float, default=0.4)
    parser.add_argument("--final-rrf-k", type=int, default=10)
    parser.add_argument("--match-threshold", type=float, default=0.8)
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-timeout", type=float, default=20.0)
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help=(
            "Disable answer, faithfulness and citation-support judges; "
            "keep deterministic citation/refusal metrics."
        ),
    )
    parser.add_argument("--sample-id", action="append")
    parser.add_argument(
        "--split",
        action="append",
        help=(
            "Run only samples whose metadata.split matches this value. "
            "Repeat or use commas for more than one split."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0")
    if not 0 <= args.similarity_threshold <= 1:
        raise ValueError("--similarity-threshold must be between 0 and 1")
    if not 0 <= args.vector_weight <= 1:
        raise ValueError("--vector-weight must be between 0 and 1")
    if args.candidate_size <= 0 or args.rerank_candidate_size <= 0:
        raise ValueError("candidate sizes must be greater than 0")
    if args.rrf_k <= 0 or args.final_rrf_k <= 0:
        raise ValueError("RRF constants must be greater than 0")
    if not 0 <= args.final_reranker_weight <= 1:
        raise ValueError("--final-reranker-weight must be between 0 and 1")
    if not 0 <= args.match_threshold <= 1:
        raise ValueError("--match-threshold must be between 0 and 1")
    if args.judge_timeout <= 0:
        raise ValueError("--judge-timeout must be greater than 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_name):
        raise ValueError("--run-name contains unsupported characters")


def validate_e2e_samples(samples: list[dict[str, Any]]) -> None:
    for sample in samples:
        expected_behavior = sample.get("expected_behavior")
        if expected_behavior is not None and expected_behavior not in {
            "answer",
            "refuse",
        }:
            raise ValueError(
                f"Sample {sample.get('id')!r} has invalid expected_behavior"
            )
        expected_sufficient = sample.get("expected_evidence_sufficient")
        if expected_sufficient is not None and not isinstance(
            expected_sufficient,
            bool,
        ):
            raise ValueError(
                f"Sample {sample.get('id')!r} has invalid "
                "expected_evidence_sufficient"
            )


def build_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise RuntimeError(
            "DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL are required"
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def error_result(
    sample: dict[str, Any],
    *,
    error: Exception,
    latency_ms: float,
    top_k: int,
) -> dict[str, Any]:
    retrieval_error = build_error_case(
        sample,
        error=error,
        latency_ms=latency_ms,
        top_k=top_k,
    )
    return {
        "result_schema_version": E2E_RESULT_SCHEMA_VERSION,
        "id": sample.get("id"),
        "question": sample.get("question"),
        "question_type": sample.get("question_type"),
        "split": str(
            (sample.get("metadata") or {}).get("split") or "unknown"
        ),
        "answerable": bool(sample.get("answerable")),
        "error": retrieval_error["error"],
        "latency_ms": {
            "retrieval": None,
            "answer": None,
            "total": round(float(latency_ms), 3),
        },
        "retrieval": retrieval_error["retrieval"],
        "retrieval_metrics": retrieval_error["retrieval_metrics"],
        "post_gate_metrics": retrieval_error["post_gate_metrics"],
        "gate_metrics": retrieval_error["gate_metrics"],
        "answer": "",
        "response_mode": "error",
        "documents": [],
        "answer_metrics": {},
        "faithfulness_metrics": {},
        "citation_metrics": {},
        "refusal_metrics": {},
        "sufficiency_metrics": {},
    }


def write_outputs(
    result_path: Path,
    summary_path: Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    atomic_write_text(
        result_path,
        "".join(
            json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
            for result in results
        ),
    )
    atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )


def main() -> int:
    args = parse_args()
    validate_args(args)

    from dotenv import load_dotenv

    load_dotenv(BACKEND_DIRECTORY / ".env")
    samples = select_samples(
        load_dataset(args.dataset.resolve()),
        args.sample_id,
        args.limit,
        splits=args.split,
    )
    validate_e2e_samples(samples)
    index_names = expand_index_names(args.index_name)
    index_argument: str | list[str] = (
        index_names[0] if len(index_names) == 1 else index_names
    )
    answer_model = args.answer_model or os.getenv(
        "CHAT_MODEL",
        DEFAULT_ANSWER_MODEL,
    )
    judge_model = None if args.no_judge else (
        args.judge_model
        or os.getenv("E2E_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    )
    client = build_client()

    output_directory = args.output_dir.resolve()
    result_path = output_directory / f"{args.run_name}.jsonl"
    summary_path = output_directory / f"{args.run_name}_summary.json"
    existing = [path for path in (result_path, summary_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Result files already exist: "
            + ", ".join(str(path) for path in existing)
        )

    retrieval_options = {
        "page_size": args.top_k,
        "similarity_threshold": args.similarity_threshold,
        "vector_similarity_weight": args.vector_weight,
        "candidate_size": args.candidate_size,
        "rerank_candidate_size": args.rerank_candidate_size,
        "rrf_k": args.rrf_k,
        "final_reranker_weight": args.final_reranker_weight,
        "final_rrf_k": args.final_rrf_k,
    }
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    results = []

    for position, sample in enumerate(samples, start=1):
        case_started = time.perf_counter()
        try:
            turn = execute_grounded_turn(
                index_names=index_argument,
                question=str(sample["question"]),
                answer_client=client,
                answer_model=answer_model,
                retrieval_options=retrieval_options,
            )
            retrieval_case = evaluate_retrieval_case(
                sample,
                turn.raw_retrieval,
                latency_ms=turn.latency_ms["retrieval"],
                top_k=args.top_k,
                match_threshold=args.match_threshold,
            )
            citation_metrics = evaluate_citations(
                turn.answer,
                turn.documents,
                judge_client=(
                    client
                    if judge_model and turn.response_mode == "answer"
                    else None
                ),
                judge_model=judge_model,
                judge_timeout=args.judge_timeout,
            )
            faithfulness_metrics = evaluate_faithfulness(
                question=str(sample["question"]),
                answer=turn.answer,
                documents=turn.documents,
                response_mode=turn.response_mode,
                judge_client=client if judge_model else None,
                judge_model=judge_model,
                judge_timeout=args.judge_timeout,
            )
            if sample.get("answerable") and turn.response_mode == "answer":
                answer_metrics = evaluate_answer_quality(
                    question=str(sample["question"]),
                    answer=turn.answer,
                    reference_answer=str(sample.get("reference_answer") or ""),
                    judge_client=client if judge_model else None,
                    judge_model=judge_model,
                    judge_timeout=args.judge_timeout,
                )
            else:
                answer_metrics = {
                    "correctness_score": None,
                    "completeness_score": None,
                    "judge_source": "skipped",
                    "judge_reason": None,
                    "judge_error": None,
                }
            refusal_metrics = evaluate_refusal(sample, turn.response_mode)
            sufficiency_metrics = evaluate_sufficiency(sample, retrieval_case)
            result = {
                "result_schema_version": E2E_RESULT_SCHEMA_VERSION,
                "id": sample.get("id"),
                "question": sample.get("question"),
                "question_type": sample.get("question_type"),
                "split": str(
                    (sample.get("metadata") or {}).get("split") or "unknown"
                ),
                "answerable": bool(sample.get("answerable")),
                "reference_answer": sample.get("reference_answer"),
                "error": None,
                "latency_ms": turn.latency_ms,
                "retrieval": retrieval_case["retrieval"],
                "retrieval_metrics": retrieval_case["retrieval_metrics"],
                "post_gate_metrics": retrieval_case["post_gate_metrics"],
                "gate_metrics": retrieval_case["gate_metrics"],
                "evidence_matches_before_sufficiency": retrieval_case[
                    "evidence_matches_before_sufficiency"
                ],
                "answer": turn.answer,
                "thinking": turn.thinking,
                "response_mode": turn.response_mode,
                "documents": turn.documents,
                "answer_metrics": answer_metrics,
                "faithfulness_metrics": faithfulness_metrics,
                "citation_metrics": citation_metrics,
                "refusal_metrics": refusal_metrics,
                "sufficiency_metrics": sufficiency_metrics,
            }
            print(
                f"[{position}/{len(samples)}] {sample['id']} "
                f"mode={turn.response_mode} "
                f"retrieval_hit={retrieval_case['retrieval_metrics'][f'hit_at_{args.top_k}']} "
                f"citations={citation_metrics['citation_count']} "
                f"faithfulness={faithfulness_metrics['faithfulness_score']} "
                f"behavior_correct={refusal_metrics['correct']} "
                f"latency_ms={turn.latency_ms['total']:.3f}",
                flush=True,
            )
        except Exception as error:
            latency_ms = (time.perf_counter() - case_started) * 1000
            if args.fail_fast:
                raise
            result = error_result(
                sample,
                error=error,
                latency_ms=latency_ms,
                top_k=args.top_k,
            )
            print(
                f"[{position}/{len(samples)}] {sample['id']} ERROR: {result['error']}",
                file=sys.stderr,
                flush=True,
            )
        results.append(result)

    completed_at = datetime.now(timezone.utc)
    summary = {
        "result_schema_version": E2E_RESULT_SCHEMA_VERSION,
        "run_name": args.run_name,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": round(time.perf_counter() - started_clock, 3),
        "dataset": {
            "path": str(args.dataset.resolve()),
            "sha256": hashlib.sha256(args.dataset.resolve().read_bytes()).hexdigest(),
            "selected_sample_count": len(samples),
            "versions": sorted(
                {
                    str(sample.get("dataset_version", "unknown"))
                    for sample in samples
                }
            ),
            "splits": sorted(
                {
                    str(
                        (sample.get("metadata") or {}).get("split")
                        or "unknown"
                    )
                    for sample in samples
                }
            ),
        },
        "config": {
            "pipeline": "knowledge_base_only",
            "index_names": index_names,
            "answer_model": answer_model,
            "answer_temperature": 0,
            "judge_model": judge_model,
            "judge_enabled": judge_model is not None,
            "faithfulness_enabled": judge_model is not None,
            "sequential_retrieval_enabled": environment_flag(
                "RAG_SEQUENTIAL_RETRIEVAL_ENABLED",
                True,
            ),
            "bridge_extraction_model": os.getenv(
                "RAG_BRIDGE_EXTRACTION_MODEL",
                "qwen3.7-flash-2026-07-15",
            ),
            "evidence_sufficiency_enabled": environment_flag(
                "RAG_EVIDENCE_SUFFICIENCY_ENABLED",
                True,
            ),
            "evidence_sufficiency_model": os.getenv(
                "RAG_EVIDENCE_SUFFICIENCY_MODEL",
                "qwen3.7-flash-2026-07-15",
            ),
            "evidence_sufficiency_fail_open": environment_flag(
                "RAG_EVIDENCE_SUFFICIENCY_FAIL_OPEN",
                False,
            ),
            "top_k": args.top_k,
            "metric_ks": list(resolve_metric_ks(args.top_k)),
            "match_threshold": args.match_threshold,
            **retrieval_options,
        },
        "result_file": str(result_path),
        "metrics": build_e2e_summary(results),
    }
    write_outputs(result_path, summary_path, results, summary)
    print(f"Results: {result_path}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary["metrics"]["overall"], ensure_ascii=False, indent=2))
    return 1 if summary["metrics"]["overall"]["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
