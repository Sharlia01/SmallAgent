"""父表对象和图片证据的持久化；入库与检索共用，不加载解析器或嵌入模型。"""

import hashlib
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from domain.tables import TABLE_PARSER_VERSION
from service.core.evidence.metadata import without_entity_annotations
from service.core.resources.paths import get_project_base_directory

PARSER_VERSION = TABLE_PARSER_VERSION


def _tenant_storage_key(index_name: str) -> str:
    return hashlib.sha256(index_name.encode()).hexdigest()[:24]


def save_evidence_image(image, index_name: str, doc_id: str) -> str:
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


def save_table_object(
    table_object: dict[str, Any],
    index_name: str,
    doc_id: str,
    file_name: str,
    image_ref: str | None,
) -> str:
    """将按内容寻址的父表对象保存到片段索引之外。"""
    table_id = str(table_object.get("table_id") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", table_id):
        raise ValueError("Table object requires a SHA-256 table_id")

    # 父表保留原始表头、单元格和出处，不再保存规则推断的公司归属。
    payload = without_entity_annotations(json.loads(json.dumps(table_object, ensure_ascii=False)))
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
