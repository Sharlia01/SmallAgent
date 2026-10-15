"""原始表格行到领域证据行的轻量转换，供入库归属标注和来源适配共用。"""

from collections.abc import Mapping
from typing import Any

from domain.evidence import EvidenceTableRow


def table_row_from_record(record: dict[str, Any]) -> EvidenceTableRow:
    def sequence(value):
        return value if isinstance(value, (list, tuple)) else ()

    headers, cells = sequence(record.get("headers")), sequence(record.get("cells"))
    original = record.get("source_row", record)
    original = original if isinstance(original, Mapping) else {}
    columns = sequence(record["column_indices"]) if "column_indices" in record else range(len(cells))
    return EvidenceTableRow(
        tuple(str(value) for value in headers), tuple(str(value) for value in cells), record.get("row_index"),
        tuple(columns),
        tuple(str(value) for value in sequence(original.get("headers"))),
        tuple(str(value) for value in sequence(original.get("cells"))),
        str(record.get("context") or ""), str(record.get("preamble") or ""),
    )
