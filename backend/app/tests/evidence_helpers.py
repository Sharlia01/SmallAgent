"""用来源记录构造测试业务对象，避免测试依赖生产序列化的内部实现。"""

from dataclasses import replace
from domain.evidence import EvidenceCollection
from agent.schemas import ToolResult
from service.core.evidence_adapter import collection_from_records
from service.core.evidence_pipeline import (
    build_answer_context, build_answer_messages as format_messages,
)
from service.core.evidence_verification import verify_evidence
from service.core.evidence_adapter import assessment_from_record
from service.core.evidence_assessment import finalize_assessment
from service.core.financial_facts import assess_financial_facts, financial_rejection
from service.core.query_context import QueryContext
from service.core.query_contract import QueryRequest, QueryGoal
from service.core.query_intent import QueryAnalysis
from service.core.query_planning import PreparedRetrievalRequest, RetrievalPlan, RetrievalRoute


def make_query(question, goals=()):
    query = QueryContext(QueryRequest.from_user(question))
    if goals:
        analysis = QueryAnalysis("lookup_query", False, len(goals), True, False, True, "测试目标", "rule", tuple(goals))
        query.prepared = PreparedRetrievalRequest(analysis, RetrievalRoute("single", "测试", "rule"), RetrievalPlan("single"))
    return query


def bind_assessment(query, evidence, assessment, *, require_semantic=None):
    if not assessment.query_version and not assessment.facts.applicable:
        facts = assess_financial_facts(query.question, evidence, query.goals)
        assessment = (financial_rejection(evidence, facts, query.goals)
                      if facts.applicable and not facts.sufficient and assessment.sufficient
                      else replace(assessment, facts=facts))
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
    assessment = None
    if metadata.get("evidence_sufficiency"):
        question = (metadata.get("query_contract") or {}).get("resolved_question") or query
        goals = tuple(QueryGoal(**{key: tuple(value) if key != "text" and key != "id" else value
                                   for key, value in item.items()})
                      for item in (metadata.get("query_intent") or {}).get("goals") or [])
        context = make_query(question, goals)
        assessment = bind_assessment(context, candidates or evidence, assessment_from_record(metadata["evidence_sufficiency"]))
    return ToolResult(tool_name, query, content, evidence=evidence, candidates=candidates, assessment=assessment,
                      clarification=metadata.get("clarification"), diagnostics={key: value for key, value in metadata.items()
                      if key not in {"query_contract", "query_intent", "evidence_sufficiency", "candidate_chunks", "clarification", "response_mode"}}, **kwargs)


def messages_for_records(question, records, history, *, goals=()):
    query = make_query(question, goals)
    evidence = collection_from_records(records)
    assessment = verify_evidence(query, evidence, require_semantic=False)
    if not assessment.sufficient:
        raise ValueError("；".join(assessment.facts.missing_requirements))
    return format_messages(build_answer_context(query, evidence, assessment), history)
