"""用来源记录构造测试业务对象，避免测试依赖生产序列化的内部实现。"""

from dataclasses import replace
from tests.financial_helpers import financial_spec
from domain.evidence import EvidenceCollection
from agent.schemas import ToolResult
from service.core.evidence.adapter import collection_from_records
from service.core.answering.answer_context import (
    build_answer_context, build_answer_messages as format_messages,
)
from service.core.evidence.verification import verify_evidence
from service.core.evidence.adapter import assessment_from_record
from service.core.evidence.assessment import finalize_assessment
from service.core.finance.facts import assess_financial_facts, financial_rejection
from service.core.query.context import QueryContext
from service.core.query.contract import QueryRequest, QueryGoal, QueryUnderstanding


def make_query(question, goals=()):
    return QueryContext(QueryRequest.from_user(question), QueryUnderstanding(tuple(goals), requires_evidence=False, source="rule", financial=financial_spec(question, goals)))


def bind_assessment(query, evidence, assessment, *, require_semantic=None):
    if not assessment.query_version and not assessment.facts.applicable:
        fact_assessment = assess_financial_facts(query.financial_requirements, evidence)
        assessment = (financial_rejection(evidence, fact_assessment, query.goals)
                      if fact_assessment.applicable and not fact_assessment.sufficient and assessment.sufficient
                      else replace(assessment, facts=fact_assessment))
    if require_semantic is None:
        require_semantic = assessment.require_semantic if assessment.require_semantic is not None else True
    return finalize_assessment(query, evidence, assessment, require_semantic=require_semantic)


def make_evidence_record(*, source_type, source_id, title, content, score=None, url=None, metadata=None):
    return {"chunk_id": source_id, "document_id": source_id, "document_name": title,
            "content_with_weight": content, "source_type": source_type, "similarity": score,
            **({"url": url} if url else {}), **(metadata or {})}


def make_tool_result(*, tool_name, query, content, sources=(), metadata=None, **kwargs):
    metadata = metadata or {}
    if sources and sources[0].get("source_type") == "web":
        links = [{"source_type": "web", "source_id": item["chunk_id"], "title": item["document_name"],
                  "content": item["content_with_weight"], "url": item.get("url"),
                  "metadata": {key: item[key] for key in ("site_name", "index", "icon") if key in item}}
                 for item in sources]
        identifier = metadata.get("document_id") or f"web-search-{sources[0]['chunk_id']}"
        sources = [{"chunk_id": identifier, "document_id": identifier, "document_name": f"实时网络搜索：{query}",
                    "content_with_weight": content, "source_type": "web", "url": sources[0].get("url"), "web_sources": links}]
    evidence = collection_from_records(sources, query=query)
    candidates = collection_from_records(metadata.get("candidate_chunks") or [], query=query)
    return ToolResult(tool_name, query, content, evidence=evidence, candidates=candidates,
                      diagnostics={key: value for key, value in metadata.items() if key != "candidate_chunks"}, **kwargs)


def messages_for_records(question, records, history, *, goals=()):
    query = make_query(question, goals)
    evidence = collection_from_records(records)
    assessment = verify_evidence(query, evidence, require_semantic=False)
    if not assessment.sufficient:
        raise ValueError("；".join(assessment.facts.missing_requirements))
    return format_messages(build_answer_context(query, evidence, assessment), history)
