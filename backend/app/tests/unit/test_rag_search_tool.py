from tests.evidence_helpers import bind_assessment
from service.core.evidence.adapter import collection_from_records
import pytest

from agent.schemas import ToolResult
from agent.tools import rag_search
from agent.tools.base import AgentTool
from service.core.retrieval.result import RetrievalResult
from service.core.evidence.adapter import assessment_from_record


@pytest.mark.unit
def test_rag_tool_returns_standard_result_and_preserves_sources(monkeypatch):
    calls = []

    def fake_retrieve_content(index_name, query, *, page_size):
        calls.append((index_name, query, page_size))
        evidence = collection_from_records([
            {
                "rank": 1,
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "document_name": "内部制度.pdf",
                "content_with_weight": "差旅申请需要提前审批。",
                "similarity": 0.91,
                "vector_similarity": 0.88,
                "term_similarity": 0.44,
                "positions": [[5, 10]],
                "kb_id": "42",
                "image_id": "image-1",
            }
        ])
        return RetrievalResult(evidence=evidence)

    monkeypatch.setattr(
        rag_search,
        "retrieve_evidence",
        fake_retrieve_content,
    )
    tool = rag_search.RagSearchTool(user_id=42)

    result = tool.run(query="  差旅如何申请？  ", top_k=8)

    assert isinstance(tool, AgentTool)
    assert isinstance(result, ToolResult)
    assert calls == [("42", "差旅如何申请？", 8)]
    assert result.success is True
    assert result.error is None
    assert result.diagnostics == {"result_count": 1, "top_k": 8}
    assert not hasattr(result, "assessment")
    assert result.content == "[1] 内部制度.pdf\n差旅申请需要提前审批。"
    evidence = next(iter(result.evidence))
    assert evidence.source.kind == "knowledge_base"
    assert evidence.id == "chunk-1"
    assert result.evidence.traces(evidence.id)[0].scores["similarity"] == 0.91
    assert evidence.source.document_id == "doc-1"
    assert result.to_dict()["evidence"][0]["document_name"] == "内部制度.pdf"


@pytest.mark.unit
def test_rag_tool_treats_no_matches_as_a_success(monkeypatch):
    def retrieve(*args, **kwargs):
        evidence = collection_from_records([])
        return RetrievalResult(evidence=evidence)

    monkeypatch.setattr(rag_search, "retrieve_evidence", retrieve)

    result = rag_search.search_knowledge_base(42, "不存在的资料")

    assert result.success is True
    assert len(result.evidence) == 0
    assert result.content == rag_search.EMPTY_RESULT_MESSAGE
    assert result.diagnostics["result_count"] == 0
    assert not hasattr(result, "assessment")


@pytest.mark.unit
def test_rag_tool_converts_retrieval_exception_into_failure(monkeypatch):
    def fail_retrieval(*_args, **_kwargs):
        raise ConnectionError("Elasticsearch is unavailable")

    monkeypatch.setattr(
        rag_search,
        "retrieve_evidence",
        fail_retrieval,
    )

    result = rag_search.search_knowledge_base(42, "测试问题")

    assert result.success is False
    assert len(result.evidence) == 0
    assert result.content == ""
    assert result.error == rag_search.FAILED_RESULT_MESSAGE
    assert result.diagnostics["error_type"] == "ConnectionError"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("query", "top_k"),
    [
        ("", 5),
        ("   ", 5),
        ("测试", 0),
        ("测试", 21),
        ("测试", True),
    ],
)
def test_rag_tool_rejects_invalid_model_arguments(query, top_k):
    result = rag_search.search_knowledge_base(42, query, top_k)

    assert result.success is False
    assert result.diagnostics["error_type"] == "validation_error"
