from types import SimpleNamespace
import json

import pytest

from evaluation.citations import _judge_claims, evaluate_citations, parse_answer_claims


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "2025年上半年收入下降", "- 2027E：3%", "-2.5%", "0.40元",
    "**2448 + 10049 = 12497百万元**", "### 经营情况", "1. 收入为10亿元",
])
def test_claim_parsing_preserves_numeric_and_markdown_prefixes(text):
    claims = parse_answer_claims(f"  {text} ##1$$  ")
    assert len(claims) == 1
    assert claims[0]["text"] == text
    assert claims[0]["citation_ids"] == [1]


class FakeCompletions:
    def create(self, **_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"assessments":['
                            '{"claim_index":1,"requires_citation":true,'
                            '"supported":true,"effective_citation_ids":[1],"reason":"直接支持"},'
                            '{"claim_index":2,"requires_citation":true,'
                            '"supported":false,"effective_citation_ids":[],"reason":"没有引用"}]}'
                        )
                    )
                )
            ]
        )


@pytest.mark.unit
def test_judge_shares_evidence_without_changing_each_claim_citations():
    # 重复引用只传一份正文；相同正文的不同来源仍保留各自编号。
    documents = [
        {"document_name": "甲.pdf", "content_with_weight": "收入10亿元"},
        {"document_name": "乙.pdf", "content_with_weight": "收入10亿元"},
        {"document_name": "未引用.pdf", "content_with_weight": "不应发送"},
    ]
    claims = [
        {"claim_index": 1, "text": "甲收入10亿元", "citation_ids": [1, 1]},
        {"claim_index": 2, "text": "比较两份报告", "citation_ids": [2, 1]},
        {"claim_index": 3, "text": "无效引用", "citation_ids": [0, 9]},
        {"claim_index": 4, "text": "没有引用", "citation_ids": []},
    ]

    def create(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        assert [item["citation_ids"] for item in payload["items"]] == [[1], [2, 1], [], []]
        assert payload["evidence"] == [
            {"citation_id": index, "document_name": document["document_name"],
             "content": document["content_with_weight"]}
            for index, document in enumerate(documents[:2], start=1)
        ]
        assert all("cited_evidence" not in item for item in payload["items"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "assessments": [
                {"claim_index": claim["claim_index"], "requires_citation": True,
                 "supported": claim["claim_index"] <= 2, "reason": "测试支持关系",
                 "effective_citation_ids": [i for i in claim["citation_ids"] if i in (1, 2)]}
                for claim in claims
            ],
        })))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assessments = _judge_claims(claims, documents, client=client, model="judge", timeout=20)
    assert [item["supported"] for item in assessments] == [True, True, False, False]
    assert claims[0]["citation_ids"] == [1, 1]  # 输入不被修改，原有统计口径保持不变。


@pytest.mark.unit
def test_citation_evaluation_checks_ids_and_claim_support():
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    result = evaluate_citations(
        "收入为10亿元。##1$$利润为2亿元。##3$$",
        [
            {
                "document_name": "报告.pdf",
                "content_with_weight": "收入为10亿元。",
            }
        ],
        judge_client=client,
        judge_model="judge",
    )

    assert result["citation_ids"] == [1, 3]
    assert result["invalid_citation_ids"] == [3]
    assert result["citation_validity"] == 0.5
    assert result["citation_precision"] == 1.0
    assert result["citation_recall"] == 0.5
    assert result["judge_source"] == "model"


@pytest.mark.unit
def test_shared_classification_is_hint_and_can_correct_intro_classification():
    answer = "财务数据如下：\n以下列出利润数据：\n利润为2亿元##1$$。"
    classified = parse_answer_claims(answer)
    for claim in classified:
        claim["requires_evidence"] = claim["claim_index"] != 1
    requests = []

    def create(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        requests.append(payload)
        # 引用模型结合完整答案可纠正事实分类提示。
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "assessments": [
                {"claim_index": 1, "requires_citation": False, "supported": False, "reason": "纯引导语", "effective_citation_ids": []},
                {"claim_index": 2, "requires_citation": False, "supported": False, "reason": "该片段不需要独立证据", "effective_citation_ids": []},
                {"claim_index": 3, "requires_citation": True, "supported": True, "reason": "直接支持", "effective_citation_ids": [1]},
            ],
        })))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = evaluate_citations(answer, [{"content_with_weight": "收入10亿元，利润2亿元。"}],
                                judge_client=client, judge_model="judge", classified_claims=classified)
    assert [item["claim_index"] for item in requests[0]["items"]] == [1, 2, 3]
    assert [item["requires_citation"] for item in requests[0]["items"]] == [False, True, True]
    assert result["claims"][0]["requires_citation"] is False
    assert result["factual_claim_count"] == 1
    assert result["citation_recall"] == 1.0
    assert result["classification_source"] == "citation_judge_with_faithfulness_hint"


@pytest.mark.unit
@pytest.mark.parametrize("hint", [None, False])
def test_missing_or_negative_classification_does_not_skip_numeric_heading(hint):
    answer = "### 收入10亿元"
    classified = parse_answer_claims(answer)
    classified[0]["requires_evidence"] = hint
    requests = []
    result = evaluate_citations(answer, [],
        judge_client=semantic_client([judgement(1, [], supported=False)], requests.append),
        judge_model="judge", classified_claims=classified)
    assert len(requests) == 1
    assert len(requests[0]["items"]) == 1
    if hint is None:
        assert "requires_citation" not in requests[0]["items"][0]
        assert result["classification_source"] == "citation_judge"
    else:
        assert requests[0]["items"][0]["requires_citation"] is False
    assert result["claims"][0]["requires_citation"] is True
    assert result["factual_claim_count"] == 1
    assert result["citation_recall"] == 0
    assert result["judge_error"] is None


@pytest.mark.unit
@pytest.mark.parametrize("classified", [[], [{"claim_index": 9, "text": "其他答案", "requires_evidence": False}]])
def test_mismatched_classification_is_ignored(classified):
    requests = []
    result = evaluate_citations("收入10亿元", [],
        judge_client=semantic_client([judgement(1, [], supported=False)], requests.append),
        judge_model="judge", classified_claims=classified)
    assert "requires_citation" not in requests[0]["items"][0]
    assert result["citation_recall"] == 0
    assert result["judge_error"] is None


@pytest.mark.unit
def test_all_non_factual_claims_are_still_reviewed_by_citation_model():
    classified = parse_answer_claims("数据如下：")
    classified[0]["requires_evidence"] = False
    requests = []
    assessment = {**judgement(1, [], supported=False), "requires_citation": False}
    result = evaluate_citations("数据如下：", [],
        judge_client=semantic_client([assessment], requests.append),
        judge_model="judge", classified_claims=classified)
    assert len(requests) == 1
    assert requests[0]["items"][0]["requires_citation"] is False
    assert result["factual_claim_count"] == 0
    assert result["citation_recall"] is None
    assert result["judge_error"] is None


def semantic_client(assessments, inspect=lambda payload: None):
    def create(**kwargs):
        inspect(json.loads(kwargs['messages'][1]['content']))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({'assessments': assessments})))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def judgement(index, ids, *, supported=True):
    return dict(claim_index=index, requires_citation=True, supported=supported,
                effective_citation_ids=ids, reason='已核对来源关联和计算')


@pytest.mark.unit
@pytest.mark.parametrize('derived', ['差额为3', '前者是后者的2倍', '换算为0.06亿元', '综上，前者为6'])
def test_derived_and_repeated_claims_inherit_explicit_sources(derived):
    answer = '甲为6##1$$。\n乙为3##2$$。\n' + derived
    def inspect(payload):
        assert payload['answer'] == answer
        assert len(payload['answer_claims']) == 3
        assert payload['items'][2]['citation_ids'] == []
        assert len(payload['evidence']) == 2  # 未引用资料不能用于填补来源关联
    assessments = [judgement(1, [1]), judgement(2, [2]),
                   judgement(3, [1, 2])]
    result = evaluate_citations(answer, [{'content_with_weight': '甲6'}, {'content_with_weight': '乙3'},
                                        {'content_with_weight': '未引用资料'}],
                                judge_client=semantic_client(assessments, inspect), judge_model='judge')
    assert result['citation_recall'] == 1
    assert result['claims'][2]['valid_citation_ids'] == []
    assert result['claims'][2]['effective_citation_ids'] == [1, 2]


@pytest.mark.unit
@pytest.mark.parametrize('supported', [True, False])
def test_numeric_heading_can_use_following_calculation_but_wrong_math_fails(supported):
    answer = '总额9\n甲6##1$$\n乙3##1$$'
    result = evaluate_citations(answer, [{'content_with_weight': '甲6，乙3'}],
        judge_client=semantic_client([
            judgement(1, [1], supported=supported),
            judgement(2, [1]), judgement(3, [1])]), judge_model='judge')
    assert result['citation_recall'] == (1 if supported else 0.666667)


@pytest.mark.unit
def test_evidence_support_without_answer_attribution_is_still_missing():
    result = evaluate_citations('甲6##1$$\n乙3', [{'content_with_weight': '甲6，乙3'}],
        judge_client=semantic_client([judgement(1, [1]),
            judgement(2, [], supported=False)]), judge_model='judge')
    assert result['citation_recall'] == 0.5


@pytest.mark.unit
@pytest.mark.parametrize('ids', [[2], [0], [9], [True], ["1"], None])
def test_judge_cannot_use_unreferenced_or_invalid_source(ids):
    result = evaluate_citations('甲6##1$$\n重复甲6', [{'content_with_weight': '甲6'}, {'content_with_weight': '甲6'}],
        judge_client=semantic_client([judgement(1, [1]), judgement(2, ids)]), judge_model='judge')
    assert result['judge_source'] == 'fallback'
    assert result['citation_recall'] is None


@pytest.mark.unit
def test_long_answer_and_source_are_preserved_in_full():
    long_claim = '收入说明' * 210
    answer = '\n'.join([f'第{i}项##1$$' for i in range(41)] + [long_claim + '##1$$'])
    source = '表格背景' * 650 + '末尾关键证据'
    parsed = parse_answer_claims(answer)
    assert len(parsed) == 42
    assert parsed[-1]['text'] == long_claim
    requests = []
    result = evaluate_citations(answer, [{'content_with_weight': source}],
        judge_client=semantic_client([judgement(i, [1]) for i in range(1, 43)], requests.append),
        judge_model='judge', judge_context_tokens=100000)
    assert len(requests[0]['items']) == 42
    assert requests[0]['items'][-1]['claim'] == long_claim
    assert requests[0]['evidence'][0]['content'] == source
    assert result['citation_recall'] == 1
    assert result['evaluation_status'] == 'completed'
    assert result['diagnostics']['evidence_truncated'] is False


@pytest.mark.unit
def test_over_budget_is_unscored_without_calling_model_or_truncating():
    def unexpected(payload):
        pytest.fail('超预算不应发送残缺材料')
    result = evaluate_citations('收入10亿元##1$$', [{'content_with_weight': '很长的材料' * 500}],
        judge_client=semantic_client([], unexpected), judge_model='judge', judge_context_tokens=100)
    assert result['evaluation_status'] == 'incomplete'
    assert result['diagnostics']['input_limited'] is True
    assert result['citation_recall'] is None
    assert result['citation_precision'] is None
    assert result['citation_completeness'] is None
    assert result['citation_validity'] == 1  # 编号检查仍可完成
    assert 'token budget' in result['judge_error']


@pytest.mark.unit
def test_truncated_model_output_is_not_a_partial_score():
    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='length',
            message=SimpleNamespace(content=json.dumps({'assessments': [judgement(1, [1])]})))])
    result = evaluate_citations('收入10亿元##1$$', [{'content_with_weight': '收入10亿元'}],
        judge_client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        judge_model='judge')
    assert result['evaluation_status'] == 'incomplete'
    assert result['citation_recall'] is None
    assert 'output was truncated' in result['judge_error']


@pytest.mark.unit
@pytest.mark.parametrize('fields', [
    {},
    dict(supported=None, effective_citation_ids=None),
    dict(supported=True, effective_citation_ids=[]),
    dict(supported=True, effective_citation_ids=[1]),
])
def test_non_factual_support_fields_cannot_fail_whole_answer(fields):
    intro = dict(claim_index=1, requires_citation=False, reason='纯引导语', **fields)
    result = evaluate_citations('数据如下：\n存货4976##1$$', [{'content_with_weight': '存货4976'}],
        judge_client=semantic_client([intro, judgement(2, [1])]), judge_model='judge')
    assert result['judge_error'] is None
    assert result['factual_claim_count'] == 1
    assert result['citation_recall'] == 1
    assert result['claims'][0]['supported'] is False
    assert result['claims'][0]['effective_citation_ids'] == []


@pytest.mark.unit
def test_unit_intro_inherits_following_citations():
    result = evaluate_citations('根据预测表（单位：百万元）：\n2023年存货4976##1$$\n2027E存货5705##1$$',
        [{'content_with_weight': '预测表（百万元），2023存货4976，2027E存货5705'}],
        judge_client=semantic_client([
            judgement(1, [1]),
            judgement(2, [1]), judgement(3, [1])]), judge_model='judge')
    assert result['citation_recall'] == 1
    assert result['factual_claim_count'] == 3


@pytest.mark.unit
def test_factual_support_without_source_still_fails_with_details():
    result = evaluate_citations('存货4976', [],
        judge_client=semantic_client([judgement(1, [], supported=True)]), judge_model='judge')
    assert result['citation_recall'] is None
    assert result['evaluation_status'] == 'incomplete'
    detail = result['diagnostics']['invalid_assessment']
    assert detail['claim_index'] == 1
    assert detail['requires_citation'] is True
    assert detail['supported'] is True
    assert detail['effective_citation_ids'] == []
    assert 'citation support requires an attributable source' in result['judge_error']


@pytest.mark.unit
@pytest.mark.parametrize('reason_fields,expected', [
    ({}, None), ({'reason': None}, None), ({'reason': '   '}, None),
    ({'reason': {'unexpected': 'object'}}, None),
    ({'reason': '  引用充分\n支持事实  '}, '引用充分 支持事实'),
])
def test_reason_is_optional_debug_information(reason_fields, expected):
    assessment = dict(claim_index=1, requires_citation=True, effective_citation_ids=[1], supported=True)
    result = evaluate_citations('收入100##1$$', [{'content_with_weight': '收入100'}],
        judge_client=semantic_client([{**assessment, **reason_fields}]), judge_model='judge')
    assert result['evaluation_status'] == 'completed'
    assert result['citation_recall'] == 1
    assert result['claims'][0]['judge_reason'] == expected


@pytest.mark.unit
def test_multistep_inheritance_uses_model_verdict_without_chain_fields():
    answer = '去年收入100##1$$\n今年收入120##2$$\n收入增长20%\n收入增速超过15%'
    assessments = [judgement(1, [1]), judgement(2, [2]), judgement(3, [1, 2]), judgement(4, [1, 2])]
    # 模型可以乱序返回；通过 claim_index 恢复对应关系。
    result = evaluate_citations(answer, [{'content_with_weight': '去年收入100'}, {'content_with_weight': '今年收入120'}],
        judge_client=semantic_client(list(reversed(assessments))), judge_model='judge')
    assert result['citation_recall'] == 1
    assert [c['effective_citation_ids'] for c in result['claims']] == [[1], [2], [1, 2], [1, 2]]
    assert result['evaluation_policy_version'] == 'semantic_citation_v5'
    assert set(result['claims'][-1]) == {
        'claim_index', 'text', 'citation_ids', 'valid_citation_ids', 'invalid_citation_ids',
        'requires_citation', 'effective_citation_ids', 'supported', 'judge_reason',
    }


@pytest.mark.unit
@pytest.mark.parametrize('indices', [[1], [1, 1], [1, 3], [True, 2]])
def test_missing_duplicate_or_unknown_claims_do_not_shrink_scoring_denominator(indices):
    result = evaluate_citations('收入100##1$$\n利润20##1$$', [{'content_with_weight': '收入100，利润20'}],
        judge_client=semantic_client([judgement(i, [1]) for i in indices]), judge_model='judge')
    assert result['evaluation_status'] == 'incomplete'
    assert result['citation_recall'] is None


@pytest.mark.unit
@pytest.mark.parametrize('field,value', [('requires_citation', 'true'), ('supported', None), ('supported', 'false')])
def test_core_judgement_fields_require_boolean_values(field, value):
    assessment = {**judgement(1, [1]), field: value}
    result = evaluate_citations('收入100##1$$', [{'content_with_weight': '收入100'}],
        judge_client=semantic_client([assessment]), judge_model='judge')
    assert result['evaluation_status'] == 'incomplete'
    assert result['citation_recall'] is None
