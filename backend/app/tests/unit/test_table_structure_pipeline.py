"""结构检测输出到行绑定的回归测试；不加载或调用模型。"""

from copy import deepcopy

import pytest
from PIL import Image

from domain.table_structure import COLUMN_HEADER, ROW_GROUP, SPANNING_CELL
from domain.tables import build_table_chunks
from service.core.deepdoc.parser.pdf_parser import RAGFlowPdfParser
from service.core.deepdoc.vision.table_structure_recognizer import TableStructureRecognizer
from service.core.ingestion.tables import bind_table_rows, parse_table_object


pytestmark = pytest.mark.unit


def detected_table(monkeypatch):
    """1008 表的结构缩影：列头、成长能力、盈利能力、估值倍数。"""
    values = [
        ["会计年度", "2024", "2025E", "2026E", "2027E"],
        ["成长能力", "", "", "", ""],
        ["营收增长率", "27.61%", "24.81%", "20.96%", "15.68%"],
        ["盈利能力", "", "", "", ""],
        ["ROA", "13.52%", "13.56%", "15.25%", "16.90%"],
        ["估值倍数", "", "", "", ""],
        ["P/S", "9.45", "7.57", "6.26", "5.41"],
    ]
    boxes, regions = [], []
    for r, row in enumerate(values):
        top = 20 + r * 40
        regions.append(dict(label="table row", score=0.98, x0=10, x1=510, top=top, bottom=top + 25))
        if r in {0, 1, 3, 5}:
            regions.append(dict(label=COLUMN_HEADER if r == 0 else ROW_GROUP,
                                score=0.96, x0=10, x1=510, top=top, bottom=top + 25))
        for c, text in enumerate(row):
            if text:
                boxes.append(dict(text=text, x0=20 + c * 100, x1=95 + c * 100,
                                  top=top + 2, bottom=top + 20,
                                  layout_type="table", page_number=1, layoutno="0"))
    for c in range(5):
        regions.append(dict(label="table column", score=0.99, x0=10 + c * 100,
                            x1=110 + c * 100, top=20, bottom=285))
    parser = RAGFlowPdfParser.__new__(RAGFlowPdfParser)
    parser.page_images = [Image.new("RGB", (600, 400))]
    parser.page_layout = [[dict(type="table", x0=10, x1=510, top=10, bottom=300)]]
    parser.page_cum_height = [0, 400]
    parser.page_from = 0
    parser.boxes = boxes
    parser.tbl_det = lambda images: [deepcopy(regions)]
    parser._table_transformer_job(1)
    # 不依赖分词器模型，保留与本例相同的数字/非数字分类。
    monkeypatch.setattr(TableStructureRecognizer, "blockType", staticmethod(
        lambda box: "Nu" if box["text"].rstrip("%").replace(".", "").isdigit() else "Ot"
    ))
    return parser.boxes


def test_detector_roles_survive_html_and_parent_object(monkeypatch):
    boxes = detected_table(monkeypatch)
    group = next(box for box in boxes if box["text"] == "盈利能力")
    assert group["table_structure"] == [{
        "label": ROW_GROUP, "score": 0.96, "page_number": 1,
        "coordinate_space": "page", "bbox": [10.0, 140.0, 510.0, 165.0],
    }]
    html = TableStructureRecognizer.construct_table(deepcopy(boxes), html=True)
    table = parse_table_object(None, html, provenance={"document_name": "润本股份.pdf"})
    assert table.cells_json["structure_version"] == "1"
    assert table.cells_json["binding_status"] == "bound"
    rows = table.cells_json["data_rows"]
    assert [row["context"] for row in rows] == ["成长能力", "盈利能力", "估值倍数"]
    assert all(row["headers"] == ["会计年度", "2024", "2025E", "2026E", "2027E"] for row in rows)
    assert rows[1]["cells"] == ["ROA", "13.52%", "13.56%", "15.25%", "16.90%"]
    cell = next(cell for cell in table.cells_json["cells"] if cell["text"] == "盈利能力")
    assert cell["role"] == "row_group"
    assert cell["structure_regions"] == group["table_structure"]
    chunks = build_table_chunks(table)
    assert "2025E：7.57" in chunks[2].semantic_text
    assert "盈利能力" not in chunks[2].semantic_text


def test_missing_model_column_header_is_not_replaced_by_text_heuristics(monkeypatch):
    boxes = detected_table(monkeypatch)
    for box in boxes:
        box["table_structure"] = [region for region in box["table_structure"] if region["label"] != COLUMN_HEADER]
    html = TableStructureRecognizer.construct_table(boxes, html=True)
    assert bind_table_rows(html)[0]["table_binding_kwd"] == "unbound"


def test_conflicting_model_roles_are_not_silently_bound(monkeypatch):
    boxes = detected_table(monkeypatch)
    group = next(box for box in boxes if box["text"] == "盈利能力")
    group["table_structure"].append({**group["table_structure"][0], "label": COLUMN_HEADER})
    html = TableStructureRecognizer.construct_table(boxes, html=True)
    assert bind_table_rows(html)[0]["table_binding_kwd"] == "unbound"


def test_spanning_header_keeps_geometry_and_multilevel_binding():
    def box(text, role, bbox, **spans):
        return dict(text=text, x0=bbox[0], top=bbox[1], x1=bbox[2], bottom=bbox[3],
                    table_structure=[dict(label=role, score=0.9, page_number=1,
                                          coordinate_space="page", bbox=bbox)], **spans)
    metric = box("指标", COLUMN_HEADER, [0, 0, 50, 40], rowspan=2)
    income = box("收入", COLUMN_HEADER, [50, 0, 150, 20], colspan=2)
    income["table_structure"].append({**income["table_structure"][0], "label": SPANNING_CELL})
    grid = [
        [[metric], [income], None],
        [None, [box("2025E", COLUMN_HEADER, [50, 20, 100, 40])],
         [box("2026E", COLUMN_HEADER, [100, 20, 150, 40])]],
        [[dict(text=text, x0=0, top=40, bottom=60, table_structure=[])] for text in ["产品", "100", "120"]],
    ]
    html = TableStructureRecognizer._TableStructureRecognizer__html_table("", set(), grid, structured=True)
    table = parse_table_object(None, html, provenance={})
    assert table.cells_json["data_rows"][0]["headers"] == ["指标", "收入 / 2025E", "收入 / 2026E"]
    cell = next(cell for cell in table.cells_json["cells"] if cell["text"] == "收入")
    assert cell["colspan"] == 2
    assert {region["label"] for region in cell["structure_regions"]} == {COLUMN_HEADER, SPANNING_CELL}
    assert table.cells_json["cell_refs"][0][1] == table.cells_json["cell_refs"][0][2]


def test_span_geometry_does_not_overwrite_header_geometry():
    # 合并范围只有前两列；列标题检测范围则包含三列。
    cells = [dict(text="收入", x0=0, x1=200, top=0, bottom=20,
                  rn=0, cn=0, SP=0, SP_left=0, SP_right=200, SP_top=0, SP_bott=20,
                  H_left=0, H_right=300, H_top=0, H_bott=40),
             dict(text="其他", x0=200, x1=300, top=0, bottom=20, rn=0, cn=2)]
    columns = [[dict(x0=i * 100, x1=(i + 1) * 100)] for i in range(3)]
    grid = [[[cells[0]], [], [cells[1]]]]
    result = TableStructureRecognizer._TableStructureRecognizer__cal_spans(
        cells, [cells], columns, grid, True)
    assert result[0][0][0]["colspan"] == 2
    assert result[0][1] is None
    assert result[0][2][0]["text"] == "其他"


def test_actual_new_column_header_starts_new_section(monkeypatch):
    boxes = detected_table(monkeypatch)
    # 将第二个分组换为模型明确识别出的完整新列头。
    template = next(box for box in boxes if box["text"] == "盈利能力")
    boxes.remove(template)
    for c, text in enumerate(["指标", "2028E", "2029E", "2030E", "2031E"]):
        new = deepcopy(template)
        original_column = next(box for box in boxes if box["top"] == 22 and box["x0"] == 20 + c * 100)
        new.update(text=text, x0=20 + c * 100, x1=95 + c * 100,
                   **{key: original_column[key] for key in ("C", "C_left", "C_right") if key in original_column})
        new["table_structure"][0]["label"] = COLUMN_HEADER
        boxes.append(new)
    rows = bind_table_rows(TableStructureRecognizer.construct_table(boxes, html=True))
    assert rows[0]["table_binding_kwd"] == "bound", rows[0]["content"]
    assert rows[0]["table_cells_slice_obj"]["rows"][0]["headers"][1] == "2024"
    assert rows[1]["table_cells_slice_obj"]["rows"][0]["headers"][1] == "2028E"
    assert rows[1]["table_context_kwd"] == ""
    assert rows[2]["table_context_kwd"] == "估值倍数"


def test_missing_group_detection_does_not_leak_previous_group(monkeypatch):
    boxes = detected_table(monkeypatch)
    next(box for box in boxes if box["text"] == "盈利能力")["table_structure"] = []
    html = TableStructureRecognizer.construct_table(boxes, html=True)
    assert bind_table_rows(html)[0]["table_binding_kwd"] == "unbound"
