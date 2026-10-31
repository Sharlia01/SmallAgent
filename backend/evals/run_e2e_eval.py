#!/usr/bin/env python3
"""执行检索、回答、忠实度、引用和拒答评测。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


if __name__ == "__main__":
    from evaluation_logging import is_logging_child, run_logged_script

    if not is_logging_child(__file__):
        raise SystemExit(run_logged_script(__file__))


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
BACKEND_DIRECTORY = SCRIPT_DIRECTORY.parent
APP_DIRECTORY = BACKEND_DIRECTORY / "app"
if not (APP_DIRECTORY / "service").is_dir():
    # 容器编排将应用包直接挂载在 /app 目录下。
    APP_DIRECTORY = BACKEND_DIRECTORY
DEFAULT_DATASET = SCRIPT_DIRECTORY / "data" / "power_reports_eval_v2_dev.jsonl"
DEFAULT_OUTPUT_DIRECTORY = SCRIPT_DIRECTORY / "results"

if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from openai import OpenAI  # noqa: E402
from e2e_artifact import write_outputs  # noqa: E402

from run_retrieval_eval import (  # noqa: E402
    expand_index_names,
    load_dataset,
    select_samples,
)
from evaluation.citations import evaluate_citations  # noqa: E402
from evaluation.end_to_end import (  # noqa: E402
    E2E_RESULT_SCHEMA_VERSION,
    build_e2e_summary,
    evaluate_answer_quality,
    evaluate_refusal,
    evaluate_sufficiency,
)
from evaluation.faithfulness import (  # noqa: E402
    evaluate_faithfulness,
)
from service.core.answering.turn import DEFAULT_ANSWER_MODEL  # noqa: E402
from service.core.runtime.clients import agent_model, agent_fallback_model  # noqa: E402
from evaluation.agent import execute_agent_turn  # noqa: E402
from evaluation.retrieval import (  # noqa: E402
    build_error_case,
    evaluate_retrieval_case,
    resolve_metric_ks,
)


DEFAULT_JUDGE_MODEL = "qwen3.7-plus"
logger = logging.getLogger(__name__)


def run_stage(sample_id: str, stage: str, function: Callable[..., Any], *args, **kwargs) -> Any:
    """给执行过程加上统一的阶段日志和耗时记录，不改变被包装函数的计算逻辑"""
    started = time.perf_counter()
    logger.info("EVAL sample_id=%s stage=%s status=start", sample_id, stage)
    try:
        result = function(*args, **kwargs)
    except Exception as error:
        logger.exception("EVAL sample_id=%s stage=%s status=error error_type=%s duration_ms=%.1f",
                         sample_id, stage, type(error).__name__, (time.perf_counter() - started) * 1000)
        raise
    logger.info("EVAL sample_id=%s stage=%s status=completed duration_ms=%.1f",
                sample_id, stage, (time.perf_counter() - started) * 1000)
    return result


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
    parser.add_argument("--index-name", action="append")
    parser.add_argument("--validate-only", action="store_true", help="Validate samples without model or retrieval calls.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--run-name", default="e2e_baseline")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--similarity-threshold", type=float, default=0.1)
    parser.add_argument("--vector-weight", type=float, default=0.6)
    parser.add_argument("--candidate-size", type=int, default=100)
    parser.add_argument("--rerank-candidate-size", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--final-reranker-weight", type=float, default=0.7)
    parser.add_argument("--final-rrf-k", type=int, default=10)
    parser.add_argument("--match-threshold", type=float, default=0.8)
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--planner-model", default=None)
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
    if not args.validate_only and not args.index_name:
        raise ValueError("--index-name is required unless --validate-only is set")
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
        history = sample.get("conversation_history", [])
        if not isinstance(history, list) or not all(
            isinstance(turn, dict) and turn.get("role") in {"user", "assistant"}
            and isinstance(turn.get("content"), str) for turn in history
        ):
            raise ValueError(f"Sample {sample.get('id')!r} has invalid conversation_history")
        expected_behavior = sample.get("expected_behavior")
        if expected_behavior is not None and expected_behavior not in {
            "answer",
            "refuse",
            "clarify",
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
        "outcome": "execution_error",
        "latency_ms": {
            "retrieval": None,
            "answer": None,
            "total": round(float(latency_ms), 3),
        },
        "retrieval": retrieval_error["retrieval"],
        "retrieval_metrics": retrieval_error["retrieval_metrics"],
        "post_gate_metrics": retrieval_error["post_gate_metrics"],
        "verification_evidence_metrics": None,
        "answer_evidence_metrics": None,
        "gate_metrics": retrieval_error["gate_metrics"],
        "evidence_sufficiency_reference": retrieval_error["evidence_sufficiency_reference"],
        "answer": "",
        "response_mode": "error",
        "documents": [],
        "answer_metrics": {},
        "faithfulness_metrics": {},
        "citation_metrics": {},
        "refusal_metrics": {},
        "sufficiency_metrics": {},
    }


def score_result(result: dict[str, Any], *, client: Any, judge_model: str | None,
                 judge_timeout: float) -> None:
    """只对已保存的答案和证据评分，不重新执行问答或检索。"""
    faithfulness_metrics = run_stage(result["id"], "faithfulness", evaluate_faithfulness,
        question=str(result["question"]),
        answer=result["answer"],
        documents=result["documents"],
        response_mode=result["response_mode"],
        judge_client=client if judge_model else None,
        judge_model=judge_model,
        judge_timeout=judge_timeout,
    )
    citation_metrics = run_stage(result["id"], "citation_support", evaluate_citations,
        result["answer"],
        result["documents"],
        judge_client=(
            client
            if judge_model and result["response_mode"] == "answer"
            else None
        ),
        judge_model=judge_model,
        judge_timeout=judge_timeout,
        classified_claims=faithfulness_metrics["claims"],
    )
    if result.get("answerable") and result["response_mode"] == "answer":
        answer_metrics = run_stage(result["id"], "answer_quality", evaluate_answer_quality,
            question=str(result["question"]),
            answer=result["answer"],
            reference_answer=str(result.get("reference_answer") or ""),
            judge_client=client if judge_model else None,
            judge_model=judge_model,
            judge_timeout=judge_timeout,
        )
    else:
        answer_metrics = {
            "correctness_score": None,
            "completeness_score": None,
            "judge_source": "skipped",
            "judge_reason": None,
            "judge_error": None,
        }
    result.update(answer_metrics=answer_metrics, faithfulness_metrics=faithfulness_metrics,
                  citation_metrics=citation_metrics)


@dataclass
class EvaluationRunState:
    """一次评测的结果快照与落盘状态。"""

    results: list[dict[str, Any]]
    summary: dict[str, Any]
    result_path: Path
    started_clock: float


def load_selected_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = select_samples(
        load_dataset(args.dataset.resolve()),
        args.sample_id,
        args.limit,
        splits=args.split,
    )
    validate_e2e_samples(samples)
    return samples


def build_retrieval_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "page_size": args.top_k,
        "similarity_threshold": args.similarity_threshold,
        "vector_similarity_weight": args.vector_weight,
        "candidate_size": args.candidate_size,
        "rerank_candidate_size": args.rerank_candidate_size,
        "rrf_k": args.rrf_k,
        "final_reranker_weight": args.final_reranker_weight,
        "final_rrf_k": args.final_rrf_k,
    }


def resolve_run_config(args: argparse.Namespace) -> dict[str, Any]:
    index_names = expand_index_names(args.index_name)
    answer_model = args.answer_model or os.getenv(
        "CHAT_MODEL",
        DEFAULT_ANSWER_MODEL,
    )
    judge_model = None if args.no_judge else (
        args.judge_model
        or os.getenv("E2E_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    )
    return {
        "pipeline": "knowledge_base_only",
        "index_names": index_names,
        "answer_model": answer_model,
        "execution_mode": "agent",
        "planner_model": args.planner_model or agent_model(),
        "planner_fallback_model": agent_fallback_model(),
        "answer_temperature": 0,
        "judge_model": judge_model,
        "judge_enabled": judge_model is not None,
        "faithfulness_enabled": judge_model is not None,
        "execution_scope": "agent_knowledge_base",
        "bridge_extraction_model": os.getenv(
            "RAG_BRIDGE_EXTRACTION_MODEL",
            "qwen3.7-flash",
        ),
        "evidence_sufficiency_enabled": environment_flag(
            "RAG_EVIDENCE_SUFFICIENCY_ENABLED",
            True,
        ),
        "evidence_sufficiency_model": os.getenv(
            "RAG_EVIDENCE_SUFFICIENCY_MODEL",
            "qwen3.6-plus",
        ),
        "evidence_sufficiency_fail_open": environment_flag(
            "RAG_EVIDENCE_SUFFICIENCY_FAIL_OPEN",
            False,
        ),
        "reranker_backend": os.getenv("RERANKER_BACKEND", "local"),
        "reranker_score_mode": os.getenv(
            "RERANKER_SCORE_MODE",
            "probability",
        ),
        "reranker_remote_score_mode": os.getenv(
            "RERANKER_REMOTE_SCORE_MODE",
            "raw_logits",
        ),
        "top_k": args.top_k,
        "metric_ks": list(resolve_metric_ks(args.top_k)),
        "match_threshold": args.match_threshold,
        **build_retrieval_options(args), # 检索参数，比如相似度权重等
    }


def prepare_output_paths(args: argparse.Namespace) -> Path:
    output_directory = args.output_dir.resolve()
    result_path = output_directory / f"{args.run_name}.json"
    html_path = result_path.with_suffix(".html")
    existing = [path for path in (result_path, html_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Result files already exist: "
            + ", ".join(str(path) for path in existing)
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    return result_path


def build_initial_summary(
    args: argparse.Namespace, samples: list[dict[str, Any]],
    config: dict[str, Any], result_path: Path,
) -> dict[str, Any]:
    return {
        "result_schema_version": E2E_RESULT_SCHEMA_VERSION,
        "run_name": args.run_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "status": "answering",
        "duration_seconds": 0,
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
        "config": config,
        "result_file": str(result_path),
        "metrics": build_e2e_summary([], top_k=args.top_k),
    }


def checkpoint(state: EvaluationRunState, status: str, *, render_html: bool = True) -> None:
    summary, results = state.summary, state.results
    summary["status"] = status
    summary["updated_at"] = datetime.now(timezone.utc).isoformat()
    summary["duration_seconds"] = round(time.perf_counter() - state.started_clock, 3)
    summary["answered_sample_count"] = len(results)
    summary["scored_sample_count"] = sum(r.get("scoring_status") == "completed" for r in results)
    summary["scoring_status_counts"] = dict(Counter(r.get("scoring_status", "pending") for r in results))
    summary["metrics"] = build_e2e_summary(results, top_k=summary["config"]["top_k"])
    if status == "completed":
        summary["completed_at"] = summary["updated_at"]
    write_outputs(state.result_path, results, summary, render_html=render_html)


def evaluate_sample(
    sample: dict[str, Any], *, args: argparse.Namespace,
    config: dict[str, Any], client: OpenAI,
) -> dict[str, Any]:
    """执行单题问答并生成确定性指标；异常交由阶段循环处理。"""
    turn = run_stage(sample["id"], "agent_answer", execute_agent_turn,
        index_names=(config["index_names"][0] if len(config["index_names"]) == 1 else config["index_names"]),
        question=str(sample["question"]),
        answer_client=client,
        answer_model=config["answer_model"],
        planner_client=client,
        planner_model=args.planner_model,
        retrieval_options=build_retrieval_options(args),
        conversation_history=sample.get("conversation_history"),
    )
    retrieval_case = run_stage(sample["id"], "retrieval_metrics", evaluate_retrieval_case,
        sample,
        turn.raw_retrieval,
        latency_ms=turn.latency_ms["retrieval"],
        top_k=args.top_k,
        match_threshold=args.match_threshold,
    )
    outcome = (turn.raw_retrieval.get("agent_execution") or {}).get("outcome")
    refusal_metrics = evaluate_refusal(sample, turn.response_mode, outcome=outcome)
    sufficiency_metrics = evaluate_sufficiency(sample, retrieval_case)
    return {
        "result_schema_version": E2E_RESULT_SCHEMA_VERSION,
        "id": sample.get("id"),
        "question": sample.get("question"),
        "question_type": sample.get("question_type"),
        "split": str(
            (sample.get("metadata") or {}).get("split") or "unknown"
        ),
        "answerable": bool(sample.get("answerable")),
        "reference_answer": sample.get("reference_answer"),
        "error": outcome if outcome in {"invalid_plan", "calculation_error", "execution_error"} else None,
        "outcome": outcome,
        "latency_ms": turn.latency_ms,
        "retrieval": retrieval_case["retrieval"],
        "retrieval_metrics": retrieval_case["retrieval_metrics"],
        "post_gate_metrics": retrieval_case["post_gate_metrics"],
        "verification_evidence_metrics": retrieval_case.get("verification_evidence_metrics"),
        "answer_evidence_metrics": retrieval_case.get("answer_evidence_metrics"),
        "gate_metrics": retrieval_case["gate_metrics"],
        "evidence_sufficiency_reference": retrieval_case["evidence_sufficiency_reference"],
        "evidence_matches_before_sufficiency": retrieval_case[
            "evidence_matches_before_sufficiency"
        ],
        "evidence_matches": retrieval_case["evidence_matches"],
        "answer": turn.answer,
        "thinking": turn.thinking,
        "response_mode": turn.response_mode,
        "documents": turn.documents,
        "answer_metrics": {},
        "faithfulness_metrics": {},
        "citation_metrics": {},
        "refusal_metrics": refusal_metrics,
        "sufficiency_metrics": sufficiency_metrics,
    }


def run_answering_phase(
    samples: list[dict[str, Any]], *, args: argparse.Namespace,
    state: EvaluationRunState, client: OpenAI,
) -> None:
    results = state.results
    checkpoint(state, "answering")
    for position, sample in enumerate(samples, start=1): # position是当前样本的序号，表示第几道题
        case_started = time.perf_counter()
        try:
            result = evaluate_sample(sample, args=args, config=state.summary["config"], client=client)
            print(
                f"[{position}/{len(samples)}] {sample['id']} "
                f"mode={result['response_mode']} outcome={result['outcome']} "
                f"verification_recall={(result.get('verification_evidence_metrics') or {}).get('recall')} "
                f"behavior_correct={result['refusal_metrics']['correct']} "
                f"latency_ms={result['latency_ms']['total']:.3f}",
                flush=True,
            )
        except Exception as error:
            latency_ms = (time.perf_counter() - case_started) * 1000
            if args.fail_fast:
                results.append(error_result(sample, error=error, latency_ms=latency_ms, top_k=args.top_k))
                results[-1]["scoring_status"] = "skipped"
                checkpoint(state, "failed", render_html=True)
                raise
            traceback.print_exc()
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
        result["scoring_status"] = "pending" if result["response_mode"] != "error" else "skipped"
        results.append(result)
        checkpoint(state, "answering")


def run_scoring_phase(
    *, args: argparse.Namespace, state: EvaluationRunState, client: OpenAI,
) -> None:
    results = state.results
    # 第二阶段只读取第一阶段结果中的快照；评分模型与业务核验模型独立配置。
    checkpoint(state, "scoring")
    print("所有题目问答执行完毕，开始评分。", flush=True)
    for position, result in enumerate(results, start=1):
        if result["scoring_status"] == "skipped":
            continue
        try:
            score_result(result, client=client, judge_model=state.summary["config"]["judge_model"],
                         judge_timeout=args.judge_timeout)
            # 评分器可能自行捕获接口异常并返回 judge_error，也应标记评分失败。
            scoring_failed = any(result[key].get("judge_error") for key in (
                "answer_metrics", "faithfulness_metrics", "citation_metrics"))
            result["scoring_status"] = "failed" if scoring_failed else "completed"
        except Exception as error:
            # 评分故障不能覆盖成功的问答结果或伪装成业务执行失败。
            result["scoring_status"] = "failed"
            result["scoring_error"] = f"{type(error).__name__}: {error}"
            if args.fail_fast:
                checkpoint(state, "failed", render_html=True)
                raise
            traceback.print_exc()
        checkpoint(state, "scoring")
        print(f"[评分 {position}/{len(results)}] {result['id']} {result['scoring_status']}", flush=True)


def print_run_summary(state: EvaluationRunState) -> None:
    print(f"Results: {state.result_path}")
    print(f"HTML report: {state.result_path.with_suffix('.html')}")
    print(json.dumps({key: value for key, value in state.summary["metrics"]["overall"].items()
                      if key not in {"diagnostics", "sample_issues"}}, ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()
    validate_args(args)

    from dotenv import load_dotenv

    load_dotenv(BACKEND_DIRECTORY / ".env")
    samples = load_selected_samples(args)
    if args.validate_only:
        print(json.dumps({"dataset": str(args.dataset.resolve()), "selected_samples": len(samples),
                          "execution_mode": "agent", "valid": True}, ensure_ascii=False, indent=2))
        return 0

    config = resolve_run_config(args)
    client = build_client()
    result_path = prepare_output_paths(args)
    started_clock = time.perf_counter()
    state = EvaluationRunState(
        results=[], summary=build_initial_summary(args, samples, config, result_path),
        result_path=result_path, started_clock=started_clock,
    )
    run_answering_phase(samples, args=args, state=state, client=client)
    run_scoring_phase(args=args, state=state, client=client)
    checkpoint(state, "completed", render_html=True)
    print_run_summary(state)
    return 1 if any(result.get("error") for result in state.results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
