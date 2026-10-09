from domain.facts import FactEvidenceBundle
from service.core.evidence_adapter import collection_from_records
from types import SimpleNamespace

import pytest

from service.core import evidence_sufficiency


class FakeCompletions:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


def fake_client(completions):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )


def chunks():
    return [
        {
            "chunk_id": "chunk-1",
            "docnm_kwd": "report.pdf",
            "content_with_weight": "2025年全年营业收入为100亿元。",
        },
        {
            "chunk_id": "chunk-2",
            "docnm_kwd": "report.pdf",
            "content_with_weight": "2025年全年归母净利润为10亿元。",
        },
    ]


@pytest.mark.unit
def test_model_accepts_evidence_that_covers_every_requirement(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    completions = FakeCompletions(
        '{"sufficient":true,"reason":"两项指标均有全年实际值",'
        '"missing_requirements":[],"supporting_chunk_indices":[1,2]}'
    )

    decision = evidence_sufficiency.assess_evidence_sufficiency("2025年全年营业收入和归母净利润是多少？",
        collection_from_records(chunks()),
        client=fake_client(completions),
        facts=FactEvidenceBundle())

    assert decision.sufficient is True
    assert decision.source == "model"
    assert decision.missing_requirements == []
    assert decision.supporting_chunk_indices == [1, 2]
    assert decision.evaluated_chunk_ids == ["chunk-1", "chunk-2"]
    call = completions.calls[0]
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}
    assert call["extra_body"] == {"enable_thinking": False}
    assert "candidate_evidence" in call["messages"][1]["content"]


@pytest.mark.unit
def test_system_prompt_preserves_company_internal_entity_scope(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    completions = FakeCompletions(
        '{"sufficient":true,"reason":"公司内部板块比较与后续事实均有证据",'
        '"missing_requirements":[],"supporting_chunk_indices":[1,2]}'
    )
    company_segments = [
        {
            "chunk_id": "segment-profit",
            "docnm_kwd": "report.pdf",
            "content_with_weight": (
                "2025年上半年，公司煤电板块归母净利润19.67亿元，"
                "气电板块0.0018亿元，水电板块8.83亿元。"
            ),
        },
        {
            "chunk_id": "thermal-generation",
            "docnm_kwd": "report.pdf",
            "content_with_weight": (
                "2025年上半年火电发电量1614.67亿千瓦时，"
                "同比下降7.40%。"
            ),
        },
    ]

    decision = evidence_sufficiency.assess_evidence_sufficiency("先找出归母净利润最高的电源板块，再说明该类电源的发电量。",
        collection_from_records(company_segments),
        client=fake_client(completions),
        facts=FactEvidenceBundle())

    assert decision.sufficient is True
    system_prompt = completions.calls[0]["messages"][0]["content"]
    assert "不得擅自扩大" in system_prompt
    assert "同一公司" in system_prompt
    assert "不得要求全行业各公司数据" in system_prompt
    assert "该板块" in system_prompt


@pytest.mark.unit
def test_model_rejects_partial_evidence_and_names_missing_requirement(
    monkeypatch,
):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    completions = FakeCompletions(
        '{"sufficient":false,"reason":"只有上半年数据",'
        '"missing_requirements":["2025年全年实际数据"],'
        '"supporting_chunk_indices":[1]}'
    )

    decision = evidence_sufficiency.assess_evidence_sufficiency("2025年全年实际营业收入是多少？",
        collection_from_records([
            {
                "chunk_id": "half-year",
                "content_with_weight": "2025年上半年营业收入为50亿元。",
            }
        ]),
        client=fake_client(completions),
        facts=FactEvidenceBundle())

    assert decision.sufficient is False
    assert decision.missing_requirements == ["2025年全年实际数据"]
    assert decision.evaluated_chunk_count == 1


@pytest.mark.unit
def test_empty_evidence_is_rejected_without_calling_model(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    completions = FakeCompletions(error=AssertionError("must not be called"))

    decision = evidence_sufficiency.assess_evidence_sufficiency("这份报告的收入是多少？",
        collection_from_records([]),
        client=fake_client(completions),
        facts=FactEvidenceBundle())

    assert decision.sufficient is False
    assert decision.source == "rule"
    assert completions.calls == []


@pytest.mark.unit
def test_preparation_keeps_every_returned_top_chunk_within_total_budget():
    source_chunks = [
        {
            "chunk_id": f"chunk-{index}",
            "content_with_weight": str(index) * 4000,
        }
        for index in range(1, 21)
    ]

    prepared = evidence_sufficiency._prepare_evidence(collection_from_records(source_chunks))

    assert len(prepared) == 20
    assert [item.evidence.id for item in prepared] == [
        f"chunk-{index}" for index in range(1, 21)
    ]
    assert sum(len(item.text) for item in prepared) <= (
        evidence_sufficiency.MAX_EVIDENCE_TOTAL_CHARS
    )


@pytest.mark.unit
def test_provider_failure_is_fail_closed_by_default(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    monkeypatch.delenv("RAG_EVIDENCE_SUFFICIENCY_FAIL_OPEN", raising=False)
    completions = FakeCompletions(error=TimeoutError("provider timeout"))

    decision = evidence_sufficiency.assess_evidence_sufficiency("2025年全年营业收入是多少？",
        collection_from_records(chunks()),
        client=fake_client(completions),
        facts=FactEvidenceBundle())

    assert decision.sufficient is False
    assert decision.source == "fallback"
    assert decision.missing_requirements == ["无法确认证据是否充分"]


@pytest.mark.unit
def test_provider_failure_can_fail_open(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_FAIL_OPEN", "true")
    completions = FakeCompletions(error=TimeoutError("provider timeout"))

    decision = evidence_sufficiency.assess_evidence_sufficiency("2025年全年营业收入是多少？",
        collection_from_records(chunks()),
        client=fake_client(completions),
        facts=FactEvidenceBundle())

    assert decision.sufficient is True
    assert decision.source == "fallback"
    assert decision.missing_requirements == []


@pytest.mark.unit
def test_disabled_check_preserves_retrieved_evidence(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "false")
    completions = FakeCompletions(error=AssertionError("must not be called"))

    decision = evidence_sufficiency.assess_evidence_sufficiency("2025年全年营业收入是多少？",
        collection_from_records(chunks()),
        client=fake_client(completions),
        facts=FactEvidenceBundle())

    assert decision.sufficient is True
    assert decision.source == "disabled"
    assert completions.calls == []
