"""有本地模型和 1008 截图时，验证真实结构推理；不访问 ES 或模型 API。"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from service.core.resources.paths import get_project_base_directory


pytestmark = pytest.mark.integration


def test_1008_table_from_local_structure_model(monkeypatch):
    base = Path(get_project_base_directory())
    image_path = base / "storage/evidence/6b86b273ff34fce19d6b804e/c043942e936dc442/2f3a021e78b5e45a7abe68e33f5a2c2ea8c6d8354ca519495df2bf9ea6d5feb5.png"
    if not image_path.exists() or not all((base / "rag/res/deepdoc" / name).exists()
                                          for name in ("tsr.onnx", "det.onnx", "rec.onnx")):
        pytest.skip("Requires local DeepDoc models and the 1008 table screenshot")

    from service.core.deepdoc.parser.pdf_parser import RAGFlowPdfParser
    from service.core.deepdoc.vision import OCR, TableStructureRecognizer
    from service.core.ingestion.tables import parse_table_object

    monkeypatch.setenv("FIGURE_VISION_ENABLED", "false")

    image = Image.open(image_path).convert("RGB")
    parser = RAGFlowPdfParser.__new__(RAGFlowPdfParser)
    parser.page_images = [image]
    parser.page_layout = [[dict(type="table", x0=10, top=10,
                               x1=image.width - 10, bottom=image.height - 10)]]
    parser.page_cum_height = [0, image.height]
    parser.page_from = 0
    parser.boxes = [dict(text=text[0], x0=float(bbox[0][0]), x1=float(bbox[1][0]),
                        top=float(bbox[0][1]), bottom=float(bbox[-1][1]),
                        layout_type="table", page_number=1, layoutno="0")
                    for bbox, text in OCR()(np.array(image))]
    parser.tbl_det = TableStructureRecognizer()
    parser._table_transformer_job(1)
    html = parser.tbl_det.construct_table(parser.boxes, html=True)
    table = parse_table_object(None, html, provenance={"document_name": "润本股份.pdf"})
    print("\n1008 detected structure:", [(r["label"], round(r["score"], 3))
                                           for r in parser.tb_cpns if "header" in r["label"]])
    print("1008 reconstructed table:\n", table.markdown)
    # 当前本地 tsr 模型漏检两个分组：必须保留原始结构并请求复核，
    # 不能把后续行错误绑定到“成长能力”。此测试固定记录该真实模型行为。
    cells = {cell["text"]: cell for cell in table.cells_json["cells"]}
    assert cells["成长能力"]["role"] == "row_group"
    assert cells["盈利能力"]["role"] == "data"
    assert cells["估值倍数"]["role"] == "data"
    assert table.cells_json["binding_status"] == "unbound"
    assert table.cells_json["data_rows"] == []
    assert table.cells_json["raw_grid"][0][1:] == ["2024", "2025E", "2026E", "2027E"]
