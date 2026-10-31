"""统一证据标注的结构、数据集完整性和 AND/OR 指标语义。"""
import importlib.util
from pathlib import Path

import pytest

from evaluation.retrieval import evaluate_retrieval_case, build_error_case, iter_evidence_atoms
from tests.evidence_helpers import requirements_for_atoms

pytestmark = pytest.mark.unit
EVALS = Path(__file__).resolve().parents[3] / "evals"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, EVALS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atom(text, identifier):
    return {"id": identifier, "type": "text", "document_name": "报告.pdf", "page": 1, "text": text}


def score(requirements, texts):
    return evaluate_retrieval_case(
        {"id": "case", "answerable": True, "evidence_requirements": requirements},
        {"chunks": [{"chunk_id": str(i), "document_name": "报告.pdf", "content_with_weight": text}
                    for i, text in enumerate(texts)]},
        latency_ms=1, top_k=5, match_threshold=1,
    )


def test_independent_requirements_need_all_information():
    requirements = requirements_for_atoms([atom("营收下降", "a"), atom("煤价下降", "b")])
    partial = score(requirements, ["营收下降"])
    assert partial["post_gate_metrics"]["hit_at_5"] is False
    assert partial["post_gate_metrics"]["recall_at_5"] == 0.5
    assert partial["post_gate_metrics"]["mrr_at_5"] == 0
    complete = score(requirements, ["营收下降", "无关内容", "煤价下降"])
    assert complete["post_gate_metrics"]["hit_at_5"] is True
    assert complete["post_gate_metrics"]["mrr_at_5"] == pytest.approx(1 / 3)


def test_alternative_requires_whole_combination_and_counts_as_one_requirement():
    requirements = [{"id": "cause", "alternatives": [
        [atom("完整原因", "a")], [atom("原因上半段", "b"), atom("原因下半段", "c")],
    ]}]
    assert score(requirements, ["原因上半段"])["post_gate_metrics"]["recall_at_5"] == 0
    for texts in (["完整原因"], ["原因上半段", "原因下半段"]):
        case = score(requirements, texts)
        assert case["gold_evidence_count"] == 1
        assert case["post_gate_metrics"]["recall_at_5"] == 1
    error = build_error_case({"evidence_requirements": requirements}, error=ValueError("failed"), latency_ms=1, top_k=5)
    assert error["gold_evidence_count"] == 1


@pytest.mark.parametrize("filename", [
    "guodian_power_eval_v1.jsonl", "guodian_power_eval_v2.jsonl",
    "power_reports_eval_v2_dev.jsonl", "power_reports_eval_v2_test.jsonl",
])
def test_all_repository_datasets_use_requirements_and_preserve_claim_links(filename):
    cli = load_script("run_retrieval_eval")
    samples = cli.load_dataset(EVALS / "data" / filename)
    assert len(samples) == (28 if filename.endswith("v1.jsonl") else 50)
    for sample in samples:
        assert "relevant_evidence" not in sample
        assert sample["dataset_version"].endswith("-requirements-v1")
        atoms = list(iter_evidence_atoms(sample))
        assert bool(atoms) == sample["answerable"]
        assert all(a.get("id") and a.get("page") and a.get("text") for a in atoms)
        ids = {a["id"] for a in atoms}
        assert all(set(c["required_evidence_ids"]) <= ids for c in sample.get("reference_claims", []))


@pytest.mark.parametrize("requirements", [None, {}, [], [{"id": "r", "alternatives": []}],
    [{"id": "r", "alternatives": [[{"type": "table_row", "document_name": "报告.pdf"}]]}]])
def test_invalid_answerable_requirements_rejected(requirements):
    cli = load_script("run_retrieval_eval")
    with pytest.raises(ValueError):
        cli.validate_evidence_requirements({"id": "bad", "answerable": True, "evidence_requirements": requirements})


def test_old_field_is_rejected_instead_of_silently_ignored():
    cli = load_script("run_retrieval_eval")
    with pytest.raises(ValueError, match="migrate relevant_evidence"):
        cli.validate_evidence_requirements({"id": "old", "answerable": False,
                                           "evidence_requirements": [], "relevant_evidence": []})


def test_builder_emits_only_valid_requirements():
    builder = load_script("build_guodian_power_eval_v2")
    cli = load_script("run_retrieval_eval")
    samples = [builder.migrate_v1_sample(sample) for sample in builder.load_jsonl(builder.V1_PATH)] + builder.new_samples()
    assert len(samples) == 50
    assert samples == builder.load_jsonl(builder.OUTPUT_PATH)
    for i, sample in enumerate(samples, 1):
        assert "relevant_evidence" not in sample
        cli.validate_evidence_requirements(sample)
        cli.validate_v2_annotations(sample, i)


def test_reused_atom_ids_are_allowed_but_conflicting_contents_are_rejected():
    from copy import deepcopy

    cli = load_script("run_retrieval_eval")
    sample = deepcopy(cli.load_dataset(EVALS / "data" / "power_reports_eval_v2_dev.jsonl")[0])
    repeated = deepcopy(sample["evidence_requirements"][0])
    repeated["id"] = "same_source_another_requirement"
    sample["evidence_requirements"].append(repeated)
    cli.validate_v2_annotations(sample, 1)
    repeated["alternatives"][0][0]["text"] = "另一个事实"
    with pytest.raises(ValueError, match="conflicting evidence id"):
        cli.validate_v2_annotations(sample, 1)


def test_text_only_table_annotation_retains_modality_and_usability():
    cli = load_script("run_retrieval_eval")
    sample = {"id": "table", "answerable": True, "evidence_requirements": requirements_for_atoms([
        {"type": "table_row", "document_name": "报告.pdf", "text": "收入 100"},
    ])}
    cli.validate_evidence_requirements(sample)
    case = evaluate_retrieval_case(sample, {
        "chunks_before_sufficiency": [{"chunk_id": "row", "document_name": "报告.pdf",
            "content_with_weight": "<table><tr><td>收入</td><td>100</td></tr></table>",
            "table_binding_kwd": "bound"}],
        "chunks": [],
    }, latency_ms=1, top_k=5, match_threshold=1)
    assert case["source_modality"] == "table"
    assert case["evidence_sufficiency_reference"]["expected_sufficient"] is True
