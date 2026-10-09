import datetime
import hashlib
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import xxhash

from service.core.rag.app.naive import chunk
from service.core.rag.nlp.model import EMBEDDING_VECTOR_FIELD, generate_embedding
from service.core.rag.utils.es_conn import ESConnection
from service.core.api.utils.file_utils import get_project_base_directory
from service.core.evidence_metadata import evidence_metadata
from service.core.table_models import TABLE_PARSER_VERSION
from service.core.entity_attribution import (
    annotate_chunk, annotate_document, annotate_table_object, get_entity_registry,
)

PARSER_VERSION = TABLE_PARSER_VERSION


def _tenant_storage_key(index_name: str) -> str:
    return hashlib.sha256(index_name.encode()).hexdigest()[:24]

def _save_evidence_image(image, index_name: str, doc_id: str) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    # 对租户和文档编号计算摘要，避免将外部输入直接作为路径片段。
    tenant = _tenant_storage_key(index_name)
    relative = Path("storage/evidence") / tenant / doc_id / (hashlib.sha256(data).hexdigest() + ".png")
    path = Path(get_project_base_directory()) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return relative.as_posix()


def _save_table_object(
    table_object: dict[str, Any],
    index_name: str,
    doc_id: str,
    file_name: str,
    image_ref: str | None,
    entity_annotation: dict[str, Any] | None = None,
) -> str:
    """将按内容寻址的父表对象保存到片段索引之外。"""
    table_id = str(table_object.get("table_id") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", table_id):
        raise ValueError("Table object requires a SHA-256 table_id")

    payload = json.loads(json.dumps(table_object, ensure_ascii=False))
    if entity_annotation is not None:
        # 将实体标注信息添加到payload中
        payload["entity_attribution"] = entity_annotation
    provenance = payload.setdefault("provenance", {})
    provenance.update({
        "doc_id": doc_id,
        "kb_id": index_name,
        "document_name": file_name,
        "parser_version": PARSER_VERSION,
    })
    if image_ref:
        provenance["image_ref_kwd"] = image_ref

    relative = (
        Path("storage/table_objects")
        / _tenant_storage_key(index_name)
        / doc_id
        / f"{table_id}.json"
    )
    path = Path(get_project_base_directory()) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if not path.exists() or path.read_text(encoding="utf-8") != serialized:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(serialized, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return relative.as_posix()


def load_table_object(reference: str) -> dict[str, Any]:
    """读取已保存的父表对象，并拒绝访问存储目录以外的路径。"""
    base = Path(get_project_base_directory()).resolve()
    path = (base / reference).resolve()
    table_root = (base / "storage/table_objects").resolve()
    if table_root not in path.parents or path.suffix != ".json":
        raise ValueError("Invalid table object reference")
    return json.loads(path.read_text(encoding="utf-8"))


def dummy(prog=None, msg=""):
    pass


def parse(file_path):
    # 使用自定义的 PDF 解析器
    result = chunk(file_path, callback=dummy)
    return result


def batch_generate_embeddings(
    texts: List[str],
    batch_size: int | None = None,
) -> List[List[float]]:
    """批量生成文本向量。

    参数：
        texts：文本列表。
        batch_size：向量推理批量大小，未传入时读取环境变量。
    返回：
        向量列表。
    """
    embeddings = generate_embedding(texts, batch_size=batch_size)
    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Embedding provider returned a different number of vectors than inputs"
        )
    return embeddings


def process_items(
    items: List[Dict[str, Any]],
    file_name: str,
    index_name: str,
) -> List[Dict[str, Any]]:
    """批量处理文档数据项。

    参数：
        items：数据项列表。
        file_name：文件名。
        index_name：Elasticsearch 索引名称。
    返回：
        处理后的数据项列表。
    """

    # 获取实体注册表并对文档进行注释
    registry = get_entity_registry()
    document_annotation = annotate_document(items, file_name, registry)
    # 准备批量处理的数据
    texts = [
        item.get("table_semantic_text_with_weight")
        or item["content_with_weight"]
        for item in items
    ]
    # 使用当前配置的本地或远程 provider 批量生成向量
    embeddings = batch_generate_embeddings(texts)

    # 处理每个数据项
    results = []
    doc_id = xxhash.xxh64(file_name.encode("utf-8")).hexdigest()
    image_refs = {}
    table_object_refs = {}
    for ordinal, (item, embedding) in enumerate(zip(items, embeddings)):
        # 生成 chunk_id
        identity = [PARSER_VERSION, index_name, doc_id, ordinal,
                    item.get("position_int", []), item.get("table_id_kwd", ""),
                    item.get("table_row_start_int"), item.get("table_row_end_int"),
                    item.get("table_semantic_text_with_weight")
                    or item["content_with_weight"]]
        chunck_id = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()

        # annotate_chunk表示给每个chunk进行实体标注
        d = {
            **evidence_metadata(item),
            **annotate_chunk(item, document_annotation, registry),
            "parser_version_kwd": PARSER_VERSION,
            "id": chunck_id,
            "content_ltks": item["content_ltks"],
            "content_with_weight": item["content_with_weight"],
            "content_sm_ltks": item["content_sm_ltks"],
            "important_kwd": [],
            "important_tks": [],
            "question_kwd": [],
            "question_tks": [],
            "create_time": str(datetime.datetime.now()).replace("T", " ")[:19],
            "create_timestamp_flt": datetime.datetime.now().timestamp(),
        }

        d["kb_id"] = index_name
        d["docnm_kwd"] = file_name
        d["title_tks"] = item["title_tks"]
        d["doc_id"] = doc_id
        d["docnm"] = file_name
        image_ref = None
        if item.get("image") is not None:
            image_key = id(item["image"])
            if image_key not in image_refs:
                image_refs[image_key] = _save_evidence_image(item["image"], index_name, doc_id)
            image_ref = image_refs[image_key]
            d["image_ref_kwd"] = image_ref

        table_object = item.get("_table_object")
        if isinstance(table_object, dict):
            table_id = str(table_object.get("table_id") or "")
            if table_id != str(item.get("table_id_kwd") or ""):
                raise ValueError("Table chunk and table object IDs do not match")
            if table_id not in table_object_refs:
                table_object_refs[table_id] = _save_table_object(
                    table_object,
                    index_name,
                    doc_id,
                    file_name,
                    image_ref,
                    annotate_table_object(table_object, document_annotation, registry),
                )
            d["table_object_ref_kwd"] = table_object_refs[table_id]

        d[EMBEDDING_VECTOR_FIELD] = embedding

        results.append(d)

    return results


def execute_insert_process(file_path: str, file_name: str, index_name: str):
    """处理文档并写入 Elasticsearch。

    参数：
        file_path：文件路径。
        file_name：文件名。
        index_name：Elasticsearch 索引名称。
    """
    # 解析文档
    documents = parse(file_path)
    if not documents:
        raise RuntimeError(f"No documents found in {file_path}; existing chunks retained")

    # 批量生成嵌入向量并存储到processed_documents中
    processed_documents = process_items(documents, file_name, index_name)
    if not processed_documents:
        raise RuntimeError(f"Failed to process documents from {file_path}")

    es_connection = ESConnection()
    return es_connection.replace_document(
        processed_documents, index_name, processed_documents[0]["doc_id"]
    )

# 测试代码
if __name__ == "__main__":
    file_path = "/mnt/d/wsl/project/gsk-poc/storage/file/【兴证电子】世运电路2023中报点评.pdf"
    session_id = "40e2743ccffa4207"
    output_file = "/mnt/d/wsl/project/gsk-poc/storage/output/result.json"

    # 如果本地文件不存在，则解析文件并保存结果
    if not os.path.exists(output_file):
        documents = parse(file_path)
        
        # 批量处理文档
        result = process_items(documents, file_path, session_id)

        # 将结果保存到本地文件
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"结果已保存到本地文件: {output_file}")
    else:
        # 如果本地文件存在，则从文件中读取结果
        with open(output_file, "r", encoding="utf-8") as f:
            result = json.load(f)
        print(f"从本地文件加载结果: {output_file}")

    # 创建 ESConnection 的实例并插入数据
    es_connection = ESConnection()
    es_connection.insert(documents=result, indexName="世运电路2023中报点评")
