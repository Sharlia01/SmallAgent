"""PDF 结构模型到表格对象的无损语义传递协议。"""

import json
from html import escape


STRUCTURE_VERSION = "1"
COLUMN_HEADER = "table column header"
ROW_GROUP = "table projected row header"
SPANNING_CELL = "table spanning cell"


def cell_structure(boxes):
    """合并单元格内保留所有检测区域，不用文字内容推断标题类型。"""
    regions = []
    for box in boxes:
        for region in box.get("table_structure", []):
            if region not in regions:
                regions.append(region)
    labels = {region["label"] for region in regions}
    if COLUMN_HEADER in labels and ROW_GROUP in labels:
        role = "ambiguous"
    elif COLUMN_HEADER in labels:
        role = "column_header"
    elif ROW_GROUP in labels:
        role = "row_group"
    else:
        role = "data"
    return role, regions


def structure_attributes(boxes):
    role, regions = cell_structure(boxes)
    payload = escape(json.dumps(regions, ensure_ascii=False, separators=(",", ":")), quote=True)
    return role, f' data-table-role="{role}" data-table-regions="{payload}"'
