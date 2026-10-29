from service.core.evidence.adapter import collection_from_records
from types import SimpleNamespace
import json

import pytest

from service.core.evidence import sufficiency as evidence_sufficiency


@pytest.mark.unit
@pytest.mark.parametrize("status", ["supported", "missing", "conflict"])
@pytest.mark.parametrize("legacy_top_level", [False, True])
def test_goal_indices_are_aggregated_without_top_level_authority(status, legacy_top_level):
    from service.core.query.contract import QueryGoal

    sufficient = status == "supported"
    payload = {
        "sufficient": sufficient, "reason": "覆盖检查",
        "missing_requirements": [] if sufficient else ["净利润同期口径依据"],
        "goal_coverage": [
            {"goal_id": "revenue", "status": "supported", "supporting_chunk_indices": [2, 1, 2]},
            {"goal_id": "profit", "status": status, "supporting_chunk_indices": [3, 2]},
        ],
    }
    if legacy_top_level:
        payload["supporting_chunk_indices"] = [4]
    report = evidence_sufficiency._parse_assessment(
        payload, chunk_count=4, chunk_ids=["context", "shared", "profit", "unrelated"],
        goals=(QueryGoal("revenue", "收入"), QueryGoal("profit", "净利润")),
    )
    assert report.supporting_chunk_indices == [1, 2, 3]
    assert report.goal_coverage[0]["supporting_chunk_ids"] == ["shared", "context"]
    assert report.goal_coverage[1]["supporting_chunk_ids"] == ["profit", "shared"]


@pytest.mark.unit
def test_goal_without_support_cannot_be_accepted_via_legacy_top_level():
    from service.core.query.contract import QueryGoal

    with pytest.raises(ValueError, match="supported goal needs evidence"):
        evidence_sufficiency._parse_assessment(
            {"sufficient": True, "reason": "充分", "missing_requirements": [],
             "supporting_chunk_indices": [1],
             "goal_coverage": [{"goal_id": "g1", "status": "supported", "supporting_chunk_indices": []}]},
            chunk_count=1, chunk_ids=["c1"], goals=(QueryGoal("g1", "收入"),),
        )


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
@pytest.mark.parametrize("sufficient", [True, False])
def test_long_reason_preserves_model_decision(monkeypatch, sufficient):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    reason = "解释" * 200
    missing = [] if sufficient else ["缺少全年数据"]
    indices = [1] if sufficient else []
    completions = FakeCompletions(json.dumps({
        "sufficient": sufficient, "reason": reason,
        "missing_requirements": missing, "supporting_chunk_indices": indices,
    }))
    decision = evidence_sufficiency.assess_evidence_sufficiency(
        "2025年全年营业收入是多少？", collection_from_records(chunks()),
        client=fake_client(completions))
    assert decision.source == "model"
    assert decision.sufficient is sufficient
    assert decision.reason == reason[:299] + "…"
    assert decision.missing_requirements == missing
    assert decision.supporting_chunk_indices == indices
    assert len(completions.calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize("reason,cause", [(None, "invalid_type"), ("  ", "empty")])
def test_reason_truncation_does_not_accept_invalid_text(reason, cause):
    with pytest.raises(ValueError, match=f"field=reason cause={cause}"):
        evidence_sufficiency._parse_assessment({
            "sufficient": True, "reason": reason,
            "missing_requirements": [], "supporting_chunk_indices": [1],
        }, chunk_count=1, chunk_ids=["chunk-1"])


@pytest.mark.unit
def test_long_missing_requirement_is_preserved_without_retry(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    missing = "需要取得相关原始资料。" * 20 + "甲公司2025年第二季度收入，不得使用上半年累计值。"
    completions = FakeCompletions(json.dumps({
        "sufficient": False, "reason": "资料不足",
        "missing_requirements": [missing], "supporting_chunk_indices": [],
    }))
    decision = evidence_sufficiency.assess_evidence_sufficiency(
        "甲公司2025年第二季度收入？", collection_from_records(chunks()),
        client=fake_client(completions))
    assert len(missing) > 160
    assert decision.source == "model"
    assert decision.sufficient is False
    assert decision.missing_requirements == [missing]
    assert len(completions.calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize("missing,cause", [(None, "invalid_type"), ("  ", "empty")])
def test_missing_requirement_still_requires_nonempty_text(missing, cause):
    with pytest.raises(ValueError, match=rf"field=missing_requirements\[0\] cause={cause}"):
        evidence_sufficiency._parse_assessment({
            "sufficient": False, "reason": "资料不足",
            "missing_requirements": [missing], "supporting_chunk_indices": [],
        }, chunk_count=1, chunk_ids=["chunk-1"])


@pytest.mark.unit
def test_model_accepts_evidence_that_covers_every_requirement(monkeypatch):
    monkeypatch.delenv("RAG_EVIDENCE_SUFFICIENCY_MODEL", raising=False)
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    completions = FakeCompletions(
        '{"sufficient":true,"reason":"两项指标均有全年实际值",'
        '"missing_requirements":[],"supporting_chunk_indices":[1,2]}'
    )

    decision = evidence_sufficiency.assess_evidence_sufficiency("2025年全年营业收入和归母净利润是多少？",
        collection_from_records(chunks()),
        client=fake_client(completions))

    assert decision.sufficient is True
    assert decision.source == "model"
    assert decision.missing_requirements == []
    assert decision.supporting_chunk_indices == [1, 2]
    assert decision.evaluated_chunk_ids == ["chunk-1", "chunk-2"]
    call = completions.calls[0]
    assert call["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
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
        client=fake_client(completions))

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
        client=fake_client(completions))

    assert decision.sufficient is False
    assert decision.missing_requirements == ["2025年全年实际数据"]
    assert decision.evaluated_chunk_count == 1


@pytest.mark.unit
def test_empty_evidence_is_rejected_without_calling_model(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    completions = FakeCompletions(error=AssertionError("must not be called"))

    decision = evidence_sufficiency.assess_evidence_sufficiency("这份报告的收入是多少？",
        collection_from_records([]),
        client=fake_client(completions))

    assert decision.sufficient is False
    assert decision.source == "rule"
    assert completions.calls == []


@pytest.mark.unit
def test_preparation_keeps_full_text_and_more_than_twenty_chunks_when_they_fit():
    records = [{"chunk_id": f"c{i}", "content_with_weight": "相关资料。"} for i in range(25)]
    records[0]["content_with_weight"] = "原文正文" * 800 + "最后一句是答案。"
    sources = collection_from_records(records)
    complete = tuple(evidence_sufficiency.VerificationItem(item, item.text) for item in sources)
    prepared = evidence_sufficiency._prepare_evidence(complete, sources)
    assert prepared is complete
    assert len(prepared) == 25
    assert prepared[0].text == sources.get("c0").text
    assert prepared[0].text.endswith("最后一句是答案。")


@pytest.mark.unit
def test_over_budget_selection_covers_goals_with_complete_chunks(monkeypatch):
    from service.core.query.contract import QueryGoal
    goals = (QueryGoal("revenue", "营业收入"), QueryGoal("profit", "净利润"))
    records = [{"chunk_id": f"revenue{i}", "content_with_weight": "营业收入为80亿元。" * 80}
               for i in range(4)]
    records.append({"chunk_id": "profit", "content_with_weight": "净利润为10亿元。" * 80})
    sources = collection_from_records(records)
    expected = [evidence_sufficiency.VerificationItem(sources.get(key), sources.get(key).text)
                for key in ("revenue0", "profit")]
    budget = evidence_sufficiency._request_token_estimate("比较收入与利润", expected, goals)
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_CONTEXT_TOKENS", str(budget))
    complete = tuple(evidence_sufficiency.VerificationItem(item, item.text) for item in sources)
    selected = evidence_sufficiency._prepare_evidence(complete, sources, question="比较收入与利润", goals=goals)
    assert [item.evidence.id for item in selected] == ["revenue0", "profit"]
    assert all(item.text == item.evidence.text for item in selected)
    assert evidence_sufficiency._request_token_estimate("比较收入与利润", selected, goals) <= budget


@pytest.mark.unit
@pytest.mark.parametrize("supported", [True, False])
def test_omitted_material_is_recorded_without_claiming_all_evidence_is_missing(monkeypatch, supported):
    import json
    sources = collection_from_records([
        {"chunk_id": "huge", "content_with_weight": "没有被送审的完整长材料。" * 5000},
        {"chunk_id": "answer", "content_with_weight": "营业收入为80亿元。"},
    ])
    completions = FakeCompletions(json.dumps({
        "sufficient": supported, "reason": "本批次核验结果",
        "missing_requirements": [] if supported else ["净利润"],
        "supporting_chunk_indices": [1],
    }))
    result = evidence_sufficiency.assess_evidence_sufficiency(
        "营业收入是多少？", sources, client=fake_client(completions))
    assert result.evaluated_chunk_ids == ["answer"]
    assert result.to_dict()["omitted_chunk_ids"] == ["huge"]
    assert result.to_dict()["input_limited"] is True
    assert result.verification_status == ("completed" if supported else "unavailable")
    if not supported:
        assert result.diagnostics["failure_kind"] == "input_limit"
        assert result.diagnostics["evaluated_missing_requirements"] == ["净利润"]
        assert result.missing_requirements == ["完成未送审材料的证据核验"]
    sent = json.loads(completions.calls[0]["messages"][1]["content"])["candidate_evidence"]
    assert sent[0]["chunk_id"] == "answer" and sent[0]["index"] == 1


@pytest.mark.unit
def test_oversized_single_chunk_does_not_call_model_or_claim_no_sources():
    sources = collection_from_records([
        {"chunk_id": "huge", "content_with_weight": "超长表格及脚注" * 10000}])
    completions = FakeCompletions(error=AssertionError("must not be called"))
    result = evidence_sufficiency.assess_evidence_sufficiency(
        "收入是多少？", sources, client=fake_client(completions))
    assert completions.calls == []
    assert result.verification_status == "unavailable"
    assert result.diagnostics["failure_kind"] == "input_limit"
    assert result.diagnostics["omitted_chunk_ids"] == ["huge"]
    assert result.evaluated_chunk_count == 0


@pytest.mark.unit
def test_provider_failure_is_fail_closed_by_default(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    monkeypatch.delenv("RAG_EVIDENCE_SUFFICIENCY_FAIL_OPEN", raising=False)
    completions = FakeCompletions(error=TimeoutError("provider timeout"))

    decision = evidence_sufficiency.assess_evidence_sufficiency("2025年全年营业收入是多少？",
        collection_from_records(chunks()),
        client=fake_client(completions))

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
        client=fake_client(completions))

    assert decision.sufficient is True
    assert decision.source == "fallback"
    assert decision.missing_requirements == []


@pytest.mark.unit
def test_disabled_check_preserves_retrieved_evidence(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "false")
    completions = FakeCompletions(error=AssertionError("must not be called"))

    decision = evidence_sufficiency.assess_evidence_sufficiency("2025年全年营业收入是多少？",
        collection_from_records(chunks()),
        client=fake_client(completions))

    assert decision.sufficient is True
    assert decision.source == "disabled"
    assert completions.calls == []


@pytest.mark.unit
def test_table_metadata_counts_towards_budget_and_survives_selection(monkeypatch):
    from dataclasses import replace
    from domain.evidence import EvidenceCollection, EvidenceStructure, EvidenceTable
    base = collection_from_records([{"chunk_id": "table", "content_with_weight": "营业收入80亿元"}]).get("table")
    table = replace(base, structure=EvidenceStructure(kind="table", table=EvidenceTable(
        caption="年度报告", metadata={"footnotes": "单位及适用范围。" * 4000})))
    item = evidence_sufficiency.VerificationItem(table, table.text)
    budget = evidence_sufficiency._request_token_estimate("收入？", (item,), ())
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_CONTEXT_TOKENS", str(budget - 1))
    assert evidence_sufficiency._prepare_evidence((item,), EvidenceCollection([table]), question="收入？") == ()
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_CONTEXT_TOKENS", str(budget))
    selected = evidence_sufficiency._prepare_evidence((item,), EvidenceCollection([table]), question="收入？")
    assert selected[0].model_record(1) == item.model_record(1)
