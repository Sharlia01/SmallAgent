"""纯检索评测直接接收类型化结果，并在输出边界生成 JSON。"""

from tests.evidence_helpers import bind_assessment

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from domain.assessment import EvidenceAssessment
from service.core.evidence.adapter import collection_from_records
from service.core.query.contract import QueryGoal
from service.core.retrieval.result import RetrievalResult
from tests.evidence_helpers import make_query


pytestmark = pytest.mark.unit


def test_cli_writes_typed_retrieval_result_with_full_evaluation_metadata(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[3] / "evals" / "run_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("retrieval_eval_entry", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    sample = {
        "schema_version": "1.0", "dataset_version": "test", "id": "approval", "question": "差旅如何审批？",
        "reference_answer": "差旅申请需要审批。", "answerable": True, "question_type": "fact",
        "relevant_evidence": [{"document_name": "制度.pdf", "page": 1, "text": "差旅申请需要审批。"}],
        "metadata": {},
    }
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(sample, ensure_ascii=False) + "\n")
    output = tmp_path / "results"
    calls = []

    def retrieve_evidence(index, question, **options):
        calls.append((index, question, options))
        query = make_query(question, (QueryGoal("approval", "差旅审批要求", source_spans=("审批",)),))
        sources = collection_from_records([
            {"chunk_id": "approval", "docnm_kwd": "制度.pdf", "content_with_weight": "差旅申请需要审批。",
             "rerank_score": 0.8, "retrieval_ranks": {"keyword_original": 2}},
        ])
        assessment = EvidenceAssessment(True, "审批要求已支持", supporting_chunk_indices=[1], source="rule",
                                        evaluated_chunk_count=1, evaluated_chunk_ids=["approval"])
        return RetrievalResult(evidence=sources, diagnostics={"retrieval_fusion": {"method": "weighted_rrf"}}, total_candidates=37)

    monkeypatch.setattr(cli, "load_retrieval_function", lambda: retrieve_evidence)
    monkeypatch.setattr(sys, "argv", [str(script), "--dataset", str(dataset), "--output-dir", str(output),
                                     "--run-name", "typed-entry", "--index-name", "test-index", "--fail-fast"])
    assert cli.main() == 0
    report = json.loads((output / "typed-entry.jsonl").read_text())
    summary = json.loads((output / "typed-entry_summary.json").read_text())
    assert len(calls) == 1 and calls[0][:2] == ("test-index", sample["question"])
    assert report["error"] is None
    assert report["retrieval"]["total_candidates"] == 37
    assert report["retrieval"]["retrieved_count"] == 1
    assert report["retrieval"]["chunks_before_sufficiency"][0]["chunk_id"] == "approval"
    assert report["retrieval"]["chunks_before_sufficiency"][0]["rerank_score"] == 0.8
    assert report["retrieval"]["query_contract"] == {}
    assert report["retrieval"]["query_understanding"] == {}
    assert report["retrieval"]["retrieval_fusion"] == {"method": "weighted_rrf"}
    assert summary["dataset"]["selected_sample_count"] == 1
