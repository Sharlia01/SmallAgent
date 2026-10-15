"""目录迁移后，配置和已有证据仍能独立于业务模块的位置读取。"""

import json
import subprocess
import sys
from pathlib import Path

from service.core.resources import paths


def test_builtin_configs_do_not_follow_deployment_storage_root(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PROJECT_BASE", str(tmp_path))

    assert paths.get_project_base_directory("storage") == str(tmp_path / "storage")
    for name in ("entity_registry.json", "query_domain.json"):
        assert json.loads(paths.config_file(name).read_text(encoding="utf-8"))


def test_default_storage_root_and_legacy_tokenizer_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PROJECT_BASE", None)
    assert Path(paths.get_project_base_directory()) == paths.config_file("query_domain.json").parent.parent
    cache_name = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
    assert (Path(paths.tokenizer_cache_directory()) / cache_name).is_file()

    monkeypatch.setattr(paths, "PROJECT_BASE", str(tmp_path))
    (tmp_path / cache_name).write_text("legacy cache", encoding="utf-8")
    assert paths.tokenizer_cache_directory() == str(tmp_path)


def test_existing_parent_table_load_does_not_import_ingestion(tmp_path):
    # 在干净进程读取旧格式引用，避免其他测试已导入的解析器掩盖依赖问题。
    reference = "storage/table_objects/tenant/doc/table.json"
    table_file = tmp_path / reference
    table_file.parent.mkdir(parents=True)
    table_file.write_text('{"table_id": "existing-table", "rows": []}', encoding="utf-8")
    script = """
import sys
from service.core.resources import paths
from service.core.persistence.evidence_store import load_table_object
paths.PROJECT_BASE = sys.argv[1]
assert load_table_object(sys.argv[2])["table_id"] == "existing-table"
assert not any(name.startswith(("service.core.ingestion", "service.core.deepdoc", "torch"))
               for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path), reference], check=True)
