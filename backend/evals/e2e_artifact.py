"""完整端到端评测文件的原子保存；不依赖模型客户端或数据库。"""
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from html_report import render_html_report


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as output:
            temporary = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_outputs(
    result_path: Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    *, render_html: bool = True,
) -> None:
    artifact = {
        "artifact_schema_version": "1.0",
        "run": {key: value for key, value in summary.items() if key not in {"config", "metrics"}},
        "config": summary["config"],
        "summary": summary["metrics"],
        "results": results,
    }
    atomic_write_text(result_path, json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    if render_html:
        atomic_write_text(result_path.with_suffix(".html"), render_html_report(results, summary))

