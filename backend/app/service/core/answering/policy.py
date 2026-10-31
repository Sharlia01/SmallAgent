"""生产和评测共用的确定性回答与拒答策略。"""

from __future__ import annotations

from typing import Any, Iterable


DEFAULT_REFUSAL_MESSAGE = "当前资料不足以可靠回答这个问题。"


def render_refusal(missing_requirements: Any = None) -> str:
    """仅输出纯拒答；兼容旧参数，但不展示可能含事实的模型缺口解释。

    详细缺口和原因保留在核验报告中，供补查、日志和评测诊断使用。
    """
    return DEFAULT_REFUSAL_MESSAGE


def render_evidence_refusal(decision: dict[str, Any]) -> str:
    if decision.get("failure_kind") == "input_limit":
        return "本次输入超过证据核验容量，尚未完成全部材料核验。请缩小问题范围后重试。"
    messages = {
        "invalid_plan": "本次查询计划未能通过结构或关键约束校验，请调整问题后重试。",
        "unsupported_capability": "已尝试检索，但当前无法可靠执行所需的计算或口径核验，暂不能提供确定性结果。",
        "calculation_error": "计算执行或校验未通过，暂不能提供可靠的数值结果。",
        "execution_error": "本次模型服务或工具执行未能完成，请稍后重试。",
    }
    if decision.get("outcome") in messages:
        return messages[decision["outcome"]]
    if decision.get("verification_status") == "unavailable":
        return "本次未能完成证据核验，请稍后重试。"
    return render_refusal(decision.get("missing_requirements"))


def refusal_from_tool_results(
    tool_results: Iterable[Any],
) -> tuple[str, dict[str, Any]] | None:
    """仅当知识库搜索成功但没有来源时返回拒答。

    网页结果、会话文档或后续成功的知识库重试仍可用于回答。
    工具运行故障交由既有错误或回退路径处理，不能误报为可靠的“文档中没有答案”。
    """
    results = list(tool_results)
    if any(getattr(result, "evidence", None) for result in results):
        return None

    empty_kb_results = [
        result
        for result in results
        if getattr(result, "tool_name", None) == "search_knowledge_base"
        and getattr(result, "success", False)
        and not getattr(result, "evidence", None)
    ]
    if not empty_kb_results:
        return None

    latest = empty_kb_results[-1]
    assessment = getattr(latest, "assessment", None)
    decision = assessment.to_dict() if assessment else {}
    return render_evidence_refusal(decision), dict(decision)
