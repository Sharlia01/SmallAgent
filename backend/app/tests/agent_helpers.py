"""Agent 测试夹具：固定模型响应和工具来源，不访问网络。"""
import json
from types import SimpleNamespace
from agent.schemas import ToolResult
from service.core.evidence.adapter import collection_from_records
from domain.assessment import EvidenceAssessment
from service.core.evidence.assessment import finalize_assessment


class ModelClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict):
            response = json.dumps(response, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=response), finish_reason="stop")])


def goal(text="员工培训要求", id="g1", constraints=(), depends_on=()):
    return dict(id=id, text=text, constraints=list(constraints), source_spans=[text], depends_on=list(depends_on))


def task(id="s1", query="员工培训要求", goal_ids=("g1",), tool="search_knowledge_base", **kwargs):
    return dict(id=id, query=query, tool=tool, goal_ids=list(goal_ids), **kwargs)


def initial(goals=None, tasks=None, *, requires_evidence=True, clarification=None, entities=()):
    return {"understanding": {"goals": [goal()] if goals is None else goals,
                             "entities": list(entities), "requires_evidence": requires_evidence, "clarification": clarification},
            "plan": {"tasks": [task()] if tasks is None else tasks}}


def records(identifier="c1", text="员工培训要求：每年培训一次", document="doc1"):
    return [{"chunk_id": identifier, "doc_id": document, "docnm_kwd": "制度", "content_with_weight": text}]


class SearchTool:
    name = "search_knowledge_base"
    description = "知识库搜索"
    parameters = {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}}

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def run_for_task(self, context, *, task, **kwargs):
        self.calls.append(task)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        evidence = collection_from_records(response, query=task.query)
        return ToolResult(self.name, task.query, "片段", evidence=evidence, task=task)


def coverage_verifier(required_count=1, calls=None):
    def verify(context, evidence, *, previous=None, require_semantic=True):
        if calls is not None:
            calls.append((context, evidence.copy()))
        sufficient = len(evidence) >= required_count or not require_semantic
        assessment = EvidenceAssessment(sufficient, "覆盖检查", evaluated_chunk_ids=[item.id for item in evidence],
                                        supporting_chunk_indices=list(range(1, len(evidence) + 1)),
                                        missing_requirements=[] if sufficient else ["需要更多证据"])
        return finalize_assessment(context, evidence, assessment, require_semantic=require_semantic)
    return verify
