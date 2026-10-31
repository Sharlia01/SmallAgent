"""评测入口应把完整集合及每次调用指标写入结果，而不是只在内存中保留。"""
from tests.evidence_helpers import requirements_for_atoms
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("planner_model", [None, "test-planner"])
@pytest.mark.parametrize("sample_count", [1, 2])
def test_no_judge_cli_writes_full_evidence_and_call_metrics(monkeypatch, tmp_path, planner_model, caplog, sample_count):
    # 本地目录与测试容器分别将 evals 放在 backend/evals 和 /evals。
    script = Path(__file__).resolve().parents[3] / "evals" / "run_e2e_eval.py"
    spec = importlib.util.spec_from_file_location("e2e_eval_entry", script)
    cli = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, cli)
    spec.loader.exec_module(cli)
    caplog.set_level("INFO", logger=cli.__name__)
    sample = {
        "schema_version": "1.0", "dataset_version": "test", "id": "sixth",
        "question": "培训频率？", "reference_answer": "每年一次", "answerable": True,
        "question_type": "fact", "metadata": {},
        "evidence_requirements": requirements_for_atoms([{"document_name": "制度.pdf", "page": 1, "text": "培训每年一次"}]),
    }
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("".join(json.dumps({**sample, "id": "sixth" if i == 0 else f"case-{i}"}, ensure_ascii=False) + "\n"
                               for i in range(sample_count)))
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
    original_score = cli.score_result
    scored = []

    def check_score_order(result, **kwargs):
        # 开始任何评分前，所有问答必须完成且逐题快照已经写入磁盘。
        assert len(calls) == sample_count
        snapshots = json.loads((tmp_path / "agent-evidence.json").read_text())["results"]
        assert len(snapshots) == sample_count
        assert all(row["scoring_status"] == ("completed" if row["id"] in scored else "pending")
                   for row in snapshots)
        assert all(row["documents"] == [gold] and row["answer"] == turn.answer for row in snapshots)
        scored.append(result["id"])
        return original_score(result, **kwargs)

    monkeypatch.setattr(cli, "score_result", check_score_order)
    monkeypatch.setenv("AGENT_MODEL", "default-planner")
    monkeypatch.setattr(sys, "argv", [str(script), "--dataset", str(dataset), "--output-dir", str(tmp_path),
                                     "--run-name", "agent-evidence", "--index-name", "test", "--no-judge", "--fail-fast"]
                        + (["--planner-model", planner_model] if planner_model else []))
    assert cli.main() == 0
    assert len(calls) == sample_count
    assert len(scored) == sample_count
    assert calls[0]["planner_client"] is client
    assert calls[0]["planner_model"] == planner_model
    summary_report = json.loads((tmp_path / "agent-evidence.json").read_text())
    report = summary_report["results"][0]
    assert summary_report["config"]["execution_mode"] == "agent"
    assert summary_report["config"]["planner_model"] == (planner_model or "default-planner")
    summary = summary_report["summary"]
    assert report["result_schema_version"] == "3.1"
    assert report["retrieval_metrics"] is None and report["post_gate_metrics"] is None
    assert len(report["retrieval"]["chunks_before_sufficiency"]) == 6
    assert report["sufficiency_metrics"]["unsafe_accept"] is False
    assert report["evidence_sufficiency_reference"]["expected_sufficient"] is True
    assert report["sufficiency_metrics"]["reference_source"] == "gold_evidence"
    assert report["evidence_matches"][0]["matched"] is True
    for metrics in (summary["overall"], summary["by_question_type"]["fact"]):
        diagnostics = metrics["diagnostics"]
        assert diagnostics["verification_evidence_metrics"]["recall"] == 1
        assert diagnostics["answer_evidence_metrics"]["recall"] == 1
        assert diagnostics["retrieval_call_metrics"]["by_top_k"]["5"]["hit_at_5"] == 0.5
        assert diagnostics["verification"]["gate_metrics"]["evidence_sufficient_answerable_decision_count"] == sample_count
    assert (tmp_path / "agent-evidence.html").is_file()
    for stage in ("agent_answer", "retrieval_metrics", "faithfulness", "citation_support", "answer_quality"):
        assert f"sample_id=sixth stage={stage} status=start" in caplog.text
        assert f"sample_id=sixth stage={stage} status=completed" in caplog.text


def test_stage_failure_logs_sample_and_stage_before_propagating(caplog, monkeypatch):
    script = Path(__file__).resolve().parents[3] / "evals" / "run_e2e_eval.py"
    spec = importlib.util.spec_from_file_location("e2e_stage_entry", script)
    cli = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, cli)
    spec.loader.exec_module(cli)
    caplog.set_level("INFO", logger=cli.__name__)
    original = TimeoutError("评分请求超时")

    def fail():
        raise original

    with pytest.raises(TimeoutError) as caught:
        cli.run_stage("dev-gd-002", "citation_support", fail)
    assert caught.value is original
    assert "sample_id=dev-gd-002 stage=citation_support status=start" in caplog.text
    assert "sample_id=dev-gd-002 stage=citation_support status=error error_type=TimeoutError" in caplog.text


@pytest.fixture
def cli(monkeypatch):
    script = Path(__file__).resolve().parents[3] / "evals" / "run_e2e_eval.py"
    spec = importlib.util.spec_from_file_location("e2e_failure_entry", script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fail_fast", [False, True])
def test_answering_failure_checkpoints_before_stopping(cli, monkeypatch, tmp_path, fail_fast):
    from copy import deepcopy

    state = cli.EvaluationRunState([], {"config": {}}, tmp_path / "run.json", 0)
    snapshots = []
    monkeypatch.setattr(cli, "checkpoint", lambda state, status, **kw:
                        snapshots.append((status, deepcopy(state.results))))
    original = TimeoutError("问答超时")
    calls = []

    def evaluate(sample, **kwargs):
        calls.append(sample["id"])
        raise original

    monkeypatch.setattr(cli, "evaluate_sample", evaluate)
    samples = [{"id": "first", "question": "问题"}, {"id": "second", "question": "问题"}]
    args = SimpleNamespace(fail_fast=fail_fast, top_k=5)
    if fail_fast:
        with pytest.raises(TimeoutError) as caught:
            cli.run_answering_phase(samples, args=args, state=state, client=object())
        assert caught.value is original
        assert calls == ["first"]
        assert snapshots[-1][0] == "failed"
    else:
        cli.run_answering_phase(samples, args=args, state=state, client=object())
        assert calls == ["first", "second"]
        assert snapshots[-1][0] == "answering"
    assert len(snapshots[-1][1]) == len(calls)
    assert all(row["scoring_status"] == "skipped" and row["error"] for row in snapshots[-1][1])


@pytest.mark.parametrize("fail_fast", [False, True])
@pytest.mark.parametrize("failure", ["exception", "judge_error"])
def test_scoring_failure_preserves_answer_snapshot(cli, monkeypatch, tmp_path, fail_fast, failure):
    from copy import deepcopy

    original = TimeoutError("评分超时")
    result = {
        "id": "first", "answer": "原始答案", "documents": [{"chunk_id": "gold"}],
        "error": None, "scoring_status": "pending", "answer_metrics": {},
        "faithfulness_metrics": {}, "citation_metrics": {},
    }
    state = cli.EvaluationRunState(
        [result, {"id": "skipped", "scoring_status": "skipped"}],
        {"config": {"judge_model": "test-judge"}}, tmp_path / "run.json", 0,
    )
    snapshots, calls = [], []
    monkeypatch.setattr(cli, "checkpoint", lambda state, status, **kw:
                        snapshots.append((status, deepcopy(state.results))))

    def score(row, **kwargs):
        calls.append(row["id"])
        if failure == "exception":
            raise original
        row["citation_metrics"]["judge_error"] = str(original)

    monkeypatch.setattr(cli, "score_result", score)
    args = SimpleNamespace(fail_fast=fail_fast, judge_timeout=10)
    if fail_fast and failure == "exception":
        with pytest.raises(TimeoutError) as caught:
            cli.run_scoring_phase(args=args, state=state, client=object())
        assert caught.value is original
        assert snapshots[-1][0] == "failed"
    else:
        cli.run_scoring_phase(args=args, state=state, client=object())
        assert snapshots[-1][0] == "scoring"
    assert calls == ["first"]
    saved = snapshots[-1][1][0]
    assert saved["scoring_status"] == "failed"
    assert saved["answer"] == "原始答案"
    assert saved["documents"] == [{"chunk_id": "gold"}]
    assert saved["error"] is None
