"""模型输入共用的表格原始说明整理，不修改来源快照。"""
from domain.evidence import EvidenceTable


def table_source_notes(table: EvidenceTable) -> list[str]:
    """与来源内容指纹共用说明提取规则，避免模型输入与冲突判定不一致。"""
    return table.source_notes()
