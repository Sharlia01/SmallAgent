"""评测入口应把完整集合及每次调用指标写入结果，而不是只在内存中保留。"""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("planner_model", [None, "test-planner"])
def test_no_judge_cli_writes_full_evidence_and_call_metrics(monkeypatch, tmp_path, planner_model):
    # 本地目录与测试容器分别将 evals 放在 backend/evals 和 /evals。
    script = Path(__file__).resolve().parents[3] / "evals" / "run_e2e_eval.py"
    spec = importlib.util.spec_from_file_location("e2e_eval_entry", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    sample = {
        "schema_version": "1.0", "dataset_version": "test", "id": "sixth",
        "question": "培训频率？", "reference_answer": "每年一次", "answerable": True,
        "question_type": "fact", "metadata": {},
        "relevant_evidence": [{"document_name": "制度.pdf", "page": 1, "text": "培训每年一次"}],
    }
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(sample, ensure_ascii=False) + "\n")
    gold = {"chunk_id": "gold", "document_name": "制度.pdf", "content_with_weight": "培训每年一次"}
    noise = [{**gold, "chunk_id": str(i), "content_with_weight": f"无关内容{i}"} for i in range(5)]
    turn = SimpleNamespace(
        answer="培训每年一次##1$$。", thinking="", response_mode="answer", documents=[gold],
        latency_ms={"retrieval": 2, "answer": 1, "total": 3},
        raw_retrieval={
            "evaluation_scope": "agent_evidence_sets", "chunks_before_sufficiency": noise + [gold],
            "chunks": [gold], "evidence_sufficiency": {"source": "rule", "sufficient": True},
            "retrieval_calls": [
                {"call_id": 1, "query": "培训频率", "top_k": 5, "latency_ms": 1, "result": {"chunks": noise}},
                {"call_id": 2, "query": "培训频率 补查", "top_k": 5, "latency_ms": 1, "result": {"chunks": [gold]}},
            ],
        },
    )
    client = object()
    calls = []
    monkeypatch.setattr(cli, "build_client", lambda: client)
    monkeypatch.setattr(cli, "execute_agent_turn", lambda **kwargs: calls.append(kwargs) or turn)
    monkeypatch.setenv("AGENT_MODEL", "default-planner")
    monkeypatch.setattr(sys, "argv", [str(script), "--dataset", str(dataset), "--output-dir", str(tmp_path),
                                     "--run-name", "agent-evidence", "--index-name", "test", "--no-judge", "--fail-fast"]
                        + (["--planner-model", planner_model] if planner_model else []))
    assert cli.main() == 0
    assert len(calls) == 1
    assert calls[0]["planner_client"] is client
    assert calls[0]["planner_model"] == planner_model
    report = json.loads((tmp_path / "agent-evidence.jsonl").read_text())
    summary_report = json.loads((tmp_path / "agent-evidence_summary.json").read_text())
    assert summary_report["config"]["execution_mode"] == "agent"
    assert summary_report["config"]["planner_model"] == (planner_model or "default-planner")
    summary = summary_report["metrics"]
    assert report["result_schema_version"] == "3.1"
    assert report["retrieval_metrics"] is None and report["post_gate_metrics"] is None
    assert len(report["retrieval"]["chunks_before_sufficiency"]) == 6
    assert report["sufficiency_metrics"]["unsafe_accept"] is False
    assert report["evidence_sufficiency_reference"]["expected_sufficient"] is True
    assert report["sufficiency_metrics"]["reference_source"] == "gold_evidence"
    assert report["evidence_matches"][0]["matched"] is True
    for metrics in (summary["overall"], summary["by_question_type"]["fact"]):
        assert metrics["verification_evidence_metrics"]["recall"] == 1
        assert metrics["answer_evidence_metrics"]["recall"] == 1
        assert metrics["retrieval_call_metrics"]["by_top_k"]["5"]["hit_at_5"] == 0.5
        assert metrics["gate_metrics"]["evidence_sufficient_answerable_decision_count"] == 1
    assert (tmp_path / "agent-evidence.html").is_file()
