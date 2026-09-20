#!/usr/bin/env python3
"""Build the enriched 50-question Guodian Power evaluation dataset.

The v1 file is intentionally kept immutable as a historical baseline. This
builder migrates its 28 samples to the v2 schema and appends 22 confusion-heavy
samples whose facts are all grounded in the same manually reviewed report.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
V1_PATH = SCRIPT_DIRECTORY / "data" / "guodian_power_eval_v1.jsonl"
OUTPUT_PATH = SCRIPT_DIRECTORY / "data" / "guodian_power_eval_v2.jsonl"
DATASET_VERSION = "guodian-power-v2"
DOCUMENT_NAME = "国电电力.pdf"
SOURCE_DATE = "2025-08-20"


TABLE_ROWS = {
    "guodian-021": ["固定资产", "360117", "383684", "427026", "467602", "453840"],
    "guodian-022": ["财务费用", "6711", "6551", "8389", "10049", "10216"],
    "guodian-023": ["每股红利", "0.82", "0.82", "0.24", "0.27", "0.29"],
    "guodian-024": ["经营活动现金流", "3071", "23940", "31078", "26415", "29468"],
    "guodian-025": ["企业自由现金流", "0", "(7749)", "(23181)", "(20808)", "34791"],
    "guodian-026": [
        "001289.SZ",
        "龙源电力",
        "16.65",
        "1,392",
        "0.76",
        "0.85",
        "0.89",
        "0.94",
        "21.9",
        "19.6",
        "18.7",
        "17.7",
        "8.7%",
        "优于大市",
    ],
}


# Each tuple contains claim text and one-based indexes into relevant_evidence.
CLAIM_OVERRIDES: dict[str, list[tuple[str, list[int]]]] = {
    "guodian-001": [
        ("2025年上半年营业收入为776.55亿元，同比下降9.52%。", [1]),
        ("2025年上半年归母净利润为36.87亿元，同比下降27.39%。", [1]),
        ("2025年上半年扣非归母净利润为34.10亿元，同比增长56.12%。", [1]),
    ],
    "guodian-002": [
        ("2025年第二季度收入为378.42亿元，同比下降6.04%。", [1]),
        ("2025年第二季度归母净利润为18.76亿元，同比下降61.96%。", [1]),
        ("2025年第二季度扣非归母净利润为18.03亿元，同比增长302.47%。", [1]),
    ],
    "guodian-003": [
        ("煤电板块归母净利润为19.67亿元，同比下降1.4%。", [1]),
        ("气电板块归母净利润为0.0018亿元，同比下降18.2%。", [1]),
        ("水电板块归母净利润为8.83亿元，同比扭亏为盈。", [1]),
        ("风电板块归母净利润为5.29亿元，同比下降31.1%。", [1]),
        ("光伏板块归母净利润为5.99亿元，同比增长39.0%。", [1]),
    ],
    "guodian-004": [
        ("火电发电量为1614.67亿千瓦时，同比下降7.40%。", [1]),
        ("火电上网电量为1518.40亿千瓦时，同比下降7.51%。", [1]),
    ],
    "guodian-005": [
        ("光伏发电量为103.35亿千瓦时，同比增长122.55%。", [1]),
        ("光伏上网电量为102.02亿千瓦时，同比增长122.85%。", [1]),
    ],
    "guodian-006": [
        ("2025年上半年毛利率为16.27%，同比增加1.65个百分点。", [1]),
        ("毛利率提升主要受煤价下降影响，入炉综合标煤单价同比下降9.52%。", [1]),
    ],
    "guodian-007": [
        ("2025年上半年财务费用率为3.71%，同比减少0.04个百分点。", [1]),
        ("2025年上半年管理费用率为1.18%，同比减少0.15个百分点。", [1]),
    ],
    "guodian-008": [
        ("2025年上半年ROE为6.36%，较上年同期减少6.50个百分点。", [1]),
        ("经营性净现金流为259.77亿元，同比增长18.87%。", [1]),
    ],
    "guodian-009": [
        ("截至2025年6月累计控股装机容量为12015.56万千瓦。", [1]),
        ("煤电、气电和水电装机容量分别为7460.90、202.40和1495.06万千瓦。", [1]),
        ("风电和光伏装机容量分别为1016.91和1840.29万千瓦。", [1]),
    ],
    "guodian-010": [
        ("2025年上半年新增装机容量为846万千瓦。", [1]),
        ("新增煤电、风电和光伏装机分别为200、33和612万千瓦。", [1]),
    ],
    "guodian-011": [
        ("营业收入下降主要因为平均上网电价同比下降6.7%。", [1]),
        ("毛利率提升主要因为入炉综合标煤单价同比下降9.52%。", [2]),
    ],
    "guodian-012": [
        ("光伏发电量和上网电量分别同比增长122.55%和122.85%。", [2]),
        ("光伏板块归母净利润为5.99亿元，同比增长39.0%。", [1]),
    ],
    "guodian-013": [
        ("前期及基建支出为211.91亿元，并披露了各类项目投入。", [1]),
        ("在建风电、光伏和大渡河水电投产计划支持未来装机增长。", [2]),
    ],
    "guodian-014": [
        ("2025至2027年每年现金分配利润原则上不低于当年归母净利润的60%，且每股现金红利不低于0.22元。", [1]),
        ("2025年上半年拟每10股派1.00元，预计分红17.84亿元，占归母净利润48.38%。", [2]),
    ],
    "guodian-015": [
        ("2025年上半年平均上网电价为409.70元/兆瓦时，同比下降29.51元/兆瓦时。", [1]),
    ],
    "guodian-016": [
        ("报告预计2025至2027年归母净利润分别为70.50、78.95和87.17亿元。", [1]),
        ("对应EPS为0.40、0.44和0.49元，PE为11.4、10.2和9.2倍。", [1]),
    ],
    "guodian-017": [
        ("报告维持国电电力“优于大市”评级。", [1]),
        ("优于大市指股价表现优于市场代表性指数10%以上。", [2]),
    ],
}


PARALLEL_LEGACY_IDS = {
    "guodian-011",
    "guodian-012",
    "guodian-013",
    "guodian-014",
    "guodian-017",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evidence_atom(evidence: dict[str, Any]) -> dict[str, Any]:
    evidence_type = evidence.get("evidence_type", "text")
    if evidence_type == "table":
        return {
            "type": "table_row",
            "document_name": evidence["document_name"],
            "caption": evidence.get("locator", ""),
            "row_cells": evidence["row_cells"],
        }
    if evidence_type == "figure":
        return {
            "type": "text",
            "document_name": evidence["document_name"],
            "text": evidence.get("locator") or evidence["text"],
        }
    return {
        "type": "text",
        "document_name": evidence["document_name"],
        "text": evidence["text"],
    }


def requirements_for_evidence(
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "id": f"requirement_{index}",
            "alternatives": [[evidence_atom(item)]],
        }
        for index, item in enumerate(evidence, start=1)
    ]


def migrate_v1_sample(sample: dict[str, Any]) -> dict[str, Any]:
    migrated = copy.deepcopy(sample)
    sample_id = migrated["id"]
    migrated["schema_version"] = "2.0"
    migrated["dataset_version"] = DATASET_VERSION
    migrated["expected_behavior"] = (
        "answer" if migrated["answerable"] else "refuse"
    )

    if migrated["question_type"] == "multi_hop":
        migrated["question_type"] = "parallel_multi_evidence"

    evidence_ids = []
    for index, evidence in enumerate(migrated["relevant_evidence"], start=1):
        evidence["id"] = f"{sample_id}-e{index}"
        evidence.setdefault("evidence_type", "text")
        if sample_id in TABLE_ROWS:
            evidence["row_cells"] = TABLE_ROWS[sample_id]
        evidence_ids.append(evidence["id"])

    if migrated["answerable"]:
        claim_specs = CLAIM_OVERRIDES.get(
            sample_id,
            [(migrated["reference_answer"], list(range(1, len(evidence_ids) + 1)))],
        )
        migrated["reference_claims"] = [
            {
                "id": f"c{index}",
                "text": text,
                "required_evidence_ids": [
                    evidence_ids[evidence_index - 1]
                    for evidence_index in required_indexes
                ],
            }
            for index, (text, required_indexes) in enumerate(
                claim_specs,
                start=1,
            )
        ]
    else:
        migrated["reference_claims"] = []
        migrated["expected_evidence_sufficient"] = False

    if sample_id in PARALLEL_LEGACY_IDS:
        migrated["evidence_requirements"] = requirements_for_evidence(
            migrated["relevant_evidence"]
        )
        migrated["expected_retrieval"] = {
            "mode": "parallel",
            "hop_count": 1,
            "subquery_count": len(migrated["relevant_evidence"]),
        }
    elif sample_id == "guodian-016":
        migrated["expected_retrieval"] = {
            "mode": "parallel",
            "hop_count": 1,
            "subquery_count": 3,
        }
    elif sample_id in {"guodian-019", "guodian-020"}:
        migrated["expected_retrieval"] = {
            "mode": "parallel",
            "hop_count": 1,
            "subquery_count": 2,
        }
    else:
        migrated["expected_retrieval"] = {
            "mode": "single",
            "hop_count": 1,
        }

    metadata = migrated["metadata"]
    metadata["split"] = "dev"
    metadata["tags"] = list(dict.fromkeys([*metadata["tags"], "legacy_v1"]))
    if migrated["answerable"]:
        modalities = sorted(
            {
                evidence.get("evidence_type", "text")
                for evidence in migrated["relevant_evidence"]
            }
        )
        metadata["source_modality"] = "+".join(modalities)
    else:
        metadata["source_modality"] = "none"

    return migrated


def text_evidence(
    evidence_id: str,
    page: int,
    text: str,
) -> dict[str, Any]:
    return {
        "id": evidence_id,
        "document_name": DOCUMENT_NAME,
        "page": page,
        "text": text,
        "evidence_type": "text",
    }


def table_evidence(
    evidence_id: str,
    page: int,
    locator: str,
    text: str,
    row_cells: list[str],
) -> dict[str, Any]:
    return {
        "id": evidence_id,
        "document_name": DOCUMENT_NAME,
        "page": page,
        "text": text,
        "evidence_type": "table",
        "locator": locator,
        "row_cells": row_cells,
    }


def figure_evidence(
    evidence_id: str,
    page: int,
    locator: str,
    text: str,
) -> dict[str, Any]:
    return {
        "id": evidence_id,
        "document_name": DOCUMENT_NAME,
        "page": page,
        "text": text,
        "evidence_type": "figure",
        "locator": locator,
    }


EVIDENCE = {
    "h1_results": text_evidence(
        "gd-h1-results",
        1,
        "2025 年上半年，公司实现营业收入 776.55 亿元，同比下降 9.52%；归母净利润 36.87 亿元，同比下降 27.39%；扣非归母净利润 34.10 亿元，同比增长 56.12%。",
    ),
    "q2_results": text_evidence(
        "gd-q2-results",
        2,
        "2025 年第二季度，公司实现收入 378.42 亿元，同比下降 6.04%；归母净利润 18.76 亿元，同比下降 61.96%；扣非归母净利润 18.03 亿元，同比增长 302.47%。",
    ),
    "segment_profit": text_evidence(
        "gd-segment-profit",
        2,
        "分板块归母净利润情况：2025 年上半年，公司煤电板块归母净利润 19.67 亿元（-1.4%），气电板块归母净利润 0.0018 亿元（-18.2%），水电板块归母净利润 8.83 亿元（扭亏为盈），风电板块归母净利润 5.29 亿元（-31.1%），光伏板块归母净利润 5.99 亿元（+39.0%）。",
    ),
    "thermal_volume": text_evidence(
        "gd-thermal-volume",
        3,
        "火电发电量 1614.67 亿千瓦时（-7.40%），上网电量 1518.40 亿千瓦时（-7.51%）；",
    ),
    "pv_volume": text_evidence(
        "gd-pv-volume",
        3,
        "光伏发电量 103.35 亿千瓦时（+122.55%），上网电量 102.02 亿千瓦时（+122.85%）。",
    ),
    "margin_coal": text_evidence(
        "gd-margin-coal",
        3,
        "2025 年上半年，公司毛利率为 16.27%，同比增加 1.65pct，毛利率水平有所增加，主要系煤价下降影响，2025 年上半年公司入炉综合标煤单价 831.48 元/吨，同比下降 87.46 元/吨，降幅 9.52%。",
    ),
    "expense_rates": text_evidence(
        "gd-expense-rates",
        3,
        "2025 年上半年，公司财务费用率、管理费用率分别为 3.71%、1.18%，财务费用率同比减少 0.04pct，管理费用率同比减少 0.15pct。",
    ),
    "actual_cash": text_evidence(
        "gd-actual-cash",
        3,
        "2025 年上半年，由于净利率下降，公司 ROE 下行，较 2024 年同期减少 6.50pct 至 6.36%。现金流方面，2025 年上半年，公司经营性净现金流为 259.77 亿元，同比增长 18.87%，主要受收入下降和燃料成本下降的综合影响；",
    ),
    "total_capacity": text_evidence(
        "gd-total-capacity",
        4,
        "截至 2025 年 6 月，公司累计控股装机容量 12015.56 万千瓦，其中煤电 7460.90 万千瓦，气电 202.40 万千瓦，水电 1495.06 万千瓦，风电 1016.91 万千瓦，光伏 1840.29 万千瓦。",
    ),
    "added_capacity": text_evidence(
        "gd-added-capacity",
        1,
        "2025 年上半年，公司新增装机容量 846 万千瓦，其中煤电 200 万千瓦、风电 33 万千瓦、光伏 612 万千瓦。",
    ),
    "projects": text_evidence(
        "gd-projects",
        4,
        "截至 2025 年 6 月，公司在建风电项目 211.83 万千瓦，主要分布在内蒙、甘肃等区域，在建光伏发电项目 343.34 万千瓦，主要分布在天津、新疆、内蒙等区域。大渡河流域水电站 2025、2026 年计划投产 136.50、215.50 万千瓦。",
    ),
    "forecast": text_evidence(
        "gd-profit-forecast",
        1,
        "预计 2025-2027 年公司归母净利润分别为 70.50/78.95/87.17 亿元（2025-2027 年原预测值为 74.97/83.22/90.73 亿元），分别同比增长-28.3%/12.0%/10.4%；EPS 分别为 0.40/0.44/0.49 元，当前股价对应 PE 为 11.4/10.2/9.2x。",
    ),
    "profit_row": table_evidence(
        "gd-profit-row",
        5,
        "财务预测与估值｜利润表（百万元）｜归属于母公司净利润行",
        "归属于母公司净利润 5609 9831 7050 7895 8717",
        ["归属于母公司净利润", "5609", "9831", "7050", "7895", "8717"],
    ),
    "eps_row": table_evidence(
        "gd-eps-row",
        5,
        "财务预测与估值｜关键财务与估值指标｜每股收益行",
        "每股收益 0.31 0.55 0.40 0.44 0.49",
        ["每股收益", "0.31", "0.55", "0.40", "0.44", "0.49"],
    ),
    "pe_row": table_evidence(
        "gd-pe-row",
        5,
        "财务预测与估值｜关键财务与估值指标｜P/E行",
        "P/E 14.3 8.2 11.4 10.2 9.2",
        ["P/E", "14.3", "8.2", "11.4", "10.2", "9.2"],
    ),
    "fixed_assets": table_evidence(
        "gd-fixed-assets-row",
        5,
        "财务预测与估值｜资产负债表（百万元）｜固定资产行",
        "固定资产 360117 383684 427026 467602 453840",
        ["固定资产", "360117", "383684", "427026", "467602", "453840"],
    ),
    "finance_expense": table_evidence(
        "gd-finance-expense-row",
        5,
        "财务预测与估值｜利润表（百万元）｜财务费用行",
        "财务费用 6711 6551 8389 10049 10216",
        ["财务费用", "6711", "6551", "8389", "10049", "10216"],
    ),
    "dividend_row": table_evidence(
        "gd-dividend-row",
        5,
        "财务预测与估值｜关键财务与估值指标｜每股红利行",
        "每股红利 0.82 0.82 0.24 0.27 0.29",
        ["每股红利", "0.82", "0.82", "0.24", "0.27", "0.29"],
    ),
    "operating_cf": table_evidence(
        "gd-operating-cf-row",
        5,
        "财务预测与估值｜现金流量表（百万元）｜经营活动现金流行",
        "经营活动现金流 3071 23940 31078 26415 29468",
        ["经营活动现金流", "3071", "23940", "31078", "26415", "29468"],
    ),
    "free_cf": table_evidence(
        "gd-free-cf-row",
        5,
        "财务预测与估值｜现金流量表（百万元）｜企业自由现金流行",
        "企业自由现金流 0 (7749) (23181) (20808) 34791",
        ["企业自由现金流", "0", "(7749)", "(23181)", "(20808)", "34791"],
    ),
    "longyuan": table_evidence(
        "gd-longyuan-row",
        4,
        "表1：可比公司估值表｜龙源电力行｜PE 25E列",
        "001289.SZ 龙源电力 16.65 1,392 0.76 0.85 0.89 0.94 21.9 19.6 18.7 17.7 8.7% 优于大市",
        [
            "001289.SZ",
            "龙源电力",
            "16.65",
            "1,392",
            "0.76",
            "0.85",
            "0.89",
            "0.94",
            "21.9",
            "19.6",
            "18.7",
            "17.7",
            "8.7%",
            "优于大市",
        ],
    ),
    "figure_profit": figure_evidence(
        "gd-figure-profit",
        2,
        "图3：国电电力归母净利润及增速（单位：亿元）",
        "图3中，2021年的归母净利润柱位于零轴下方；2020、2022、2023和2024年的柱均位于零轴上方。",
    ),
    "figure_quarterly_revenue": figure_evidence(
        "gd-figure-quarterly-revenue",
        2,
        "图2：国电电力单季营业收入（单位：亿元）",
        "图2中，2025年仅在Q1和Q2显示了营业收入柱，Q3和Q4没有2025年柱。",
    ),
}


def clone_evidence(*keys: str) -> list[dict[str, Any]]:
    return [copy.deepcopy(EVIDENCE[key]) for key in keys]


def requirement(
    requirement_id: str,
    *alternatives: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": requirement_id,
        "alternatives": [copy.deepcopy(item) for item in alternatives],
    }


def claims(*items: tuple[str, str | list[str]]) -> list[dict[str, Any]]:
    result = []
    for index, (text, evidence_ids) in enumerate(items, start=1):
        if isinstance(evidence_ids, str):
            evidence_ids = [evidence_ids]
        result.append(
            {
                "id": f"c{index}",
                "text": text,
                "required_evidence_ids": evidence_ids,
            }
        )
    return result


def metadata(
    split: str,
    difficulty: str,
    tags: list[str],
    source_modality: str,
    *,
    unanswerable_reason: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "difficulty": difficulty,
        "tags": tags,
        "split": split,
        "source_date": SOURCE_DATE,
        "source_modality": source_modality,
    }
    if unanswerable_reason:
        result["unanswerable_reason"] = unanswerable_reason
    return result


def answerable_sample(
    *,
    sample_id: str,
    question: str,
    reference_answer: str,
    question_type: str,
    evidence: list[dict[str, Any]],
    reference_claims: list[dict[str, Any]],
    expected_retrieval: dict[str, Any],
    sample_metadata: dict[str, Any],
    evidence_requirements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sample = {
        "schema_version": "2.0",
        "dataset_version": DATASET_VERSION,
        "id": sample_id,
        "question": question,
        "reference_answer": reference_answer,
        "answerable": True,
        "expected_behavior": "answer",
        "question_type": question_type,
        "relevant_evidence": evidence,
        "reference_claims": reference_claims,
        "expected_retrieval": expected_retrieval,
        "metadata": sample_metadata,
    }
    if evidence_requirements:
        sample["evidence_requirements"] = evidence_requirements
    return sample


def unanswerable_sample(
    *,
    sample_id: str,
    question: str,
    reference_answer: str,
    expected_retrieval: dict[str, Any],
    sample_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "dataset_version": DATASET_VERSION,
        "id": sample_id,
        "question": question,
        "reference_answer": reference_answer,
        "answerable": False,
        "expected_behavior": "refuse",
        "expected_evidence_sufficient": False,
        "question_type": "unanswerable",
        "relevant_evidence": [],
        "reference_claims": [],
        "expected_retrieval": expected_retrieval,
        "metadata": sample_metadata,
    }


def new_samples() -> list[dict[str, Any]]:
    text = lambda key: evidence_atom(EVIDENCE[key])

    return [
        answerable_sample(
            sample_id="guodian-029",
            question="不要把实际值和预测值混在一起：国电电力2025年上半年实际归母净利润是多少，报告预测2025年全年归母净利润是多少？",
            reference_answer="2025年上半年实际归母净利润为36.87亿元；报告预测2025年全年归母净利润为70.50亿元。前者是半年度实际值，后者是全年预测值。",
            question_type="parallel_multi_evidence",
            evidence=clone_evidence("h1_results", "forecast", "profit_row"),
            reference_claims=claims(
                ("2025年上半年实际归母净利润为36.87亿元。", "gd-h1-results"),
                ("2025年全年预测归母净利润为70.50亿元。", ["gd-profit-forecast", "gd-profit-row"]),
            ),
            evidence_requirements=[
                requirement("actual_h1_profit", [text("h1_results")]),
                requirement(
                    "forecast_full_year_profit",
                    [text("forecast")],
                    [text("profit_row")],
                ),
            ],
            expected_retrieval={"mode": "parallel", "hop_count": 1, "subquery_count": 2},
            sample_metadata=metadata(
                "test", "hard", ["时间口径", "实际值", "预测值", "归母净利润"], "text+table"
            ),
        ),
        answerable_sample(
            sample_id="guodian-030",
            question="国电电力2025年上半年和第二季度的营业收入分别是多少？请区分累计口径与单季度口径。",
            reference_answer="2025年上半年累计营业收入为776.55亿元；第二季度单季度收入为378.42亿元。",
            question_type="parallel_multi_evidence",
            evidence=clone_evidence("h1_results", "q2_results"),
            reference_claims=claims(
                ("2025年上半年累计营业收入为776.55亿元。", "gd-h1-results"),
                ("2025年第二季度单季度收入为378.42亿元。", "gd-q2-results"),
            ),
            evidence_requirements=[
                requirement("half_year_revenue", [text("h1_results")]),
                requirement("second_quarter_revenue", [text("q2_results")]),
            ],
            expected_retrieval={"mode": "parallel", "hop_count": 1, "subquery_count": 2},
            sample_metadata=metadata(
                "test", "hard", ["时间口径", "累计值", "单季度", "营业收入"], "text"
            ),
        ),
        answerable_sample(
            sample_id="guodian-031",
            question="毛利率提高1.65个百分点和煤价下降9.52%是不是同一个口径？请分别给出毛利率、变动幅度和煤价数据。",
            reference_answer="不是同一口径。毛利率为16.27%，同比提高1.65个百分点；入炉综合标煤单价为831.48元/吨，同比下降87.46元/吨，降幅9.52%。",
            question_type="fact",
            evidence=clone_evidence("margin_coal"),
            reference_claims=claims(
                ("毛利率为16.27%，同比提高1.65个百分点。", "gd-margin-coal"),
                ("入炉综合标煤单价为831.48元/吨，同比下降87.46元/吨，降幅9.52%。", "gd-margin-coal"),
            ),
            expected_retrieval={"mode": "single", "hop_count": 1},
            sample_metadata=metadata(
                "test", "hard", ["百分比", "百分点", "毛利率", "煤价"], "text"
            ),
        ),
        answerable_sample(
            sample_id="guodian-032",
            question="请区分费用率和费用金额：国电电力2025年上半年财务费用率是多少，2026E财务费用金额是多少？",
            reference_answer="2025年上半年财务费用率为3.71%；2026E财务费用为10049百万元，即100.49亿元。",
            question_type="parallel_multi_evidence",
            evidence=clone_evidence("expense_rates", "finance_expense"),
            reference_claims=claims(
                ("2025年上半年财务费用率为3.71%。", "gd-expense-rates"),
                ("2026E财务费用为10049百万元，即100.49亿元。", "gd-finance-expense-row"),
            ),
            evidence_requirements=[
                requirement("actual_expense_rate", [text("expense_rates")]),
                requirement("forecast_expense_amount", [text("finance_expense")]),
            ],
            expected_retrieval={"mode": "parallel", "hop_count": 1, "subquery_count": 2},
            sample_metadata=metadata(
                "test", "hard", ["费用率", "费用金额", "实际值", "预测值", "表格"], "text+table"
            ),
        ),
        answerable_sample(
            sample_id="guodian-033",
            question="国电电力2025年上半年实际经营性净现金流和2027E经营活动现金流预测分别是多少？",
            reference_answer="2025年上半年实际经营性净现金流为259.77亿元；2027E经营活动现金流预测为29468百万元，即294.68亿元。",
            question_type="parallel_multi_evidence",
            evidence=clone_evidence("actual_cash", "operating_cf"),
            reference_claims=claims(
                ("2025年上半年实际经营性净现金流为259.77亿元。", "gd-actual-cash"),
                ("2027E经营活动现金流预测为29468百万元，即294.68亿元。", "gd-operating-cf-row"),
            ),
            evidence_requirements=[
                requirement("actual_operating_cash", [text("actual_cash")]),
                requirement("forecast_operating_cash", [text("operating_cf")]),
            ],
            expected_retrieval={"mode": "parallel", "hop_count": 1, "subquery_count": 2},
            sample_metadata=metadata(
                "test", "hard", ["现金流", "实际值", "预测值", "时间口径"], "text+table"
            ),
        ),
        answerable_sample(
            sample_id="guodian-034",
            question="不要混淆累计装机和新增装机：截至2025年6月公司累计控股装机、上半年新增装机分别是多少？其中光伏分别是多少？",
            reference_answer="累计控股装机为12015.56万千瓦，其中光伏1840.29万千瓦；2025年上半年新增装机为846万千瓦，其中光伏612万千瓦。",
            question_type="parallel_multi_evidence",
            evidence=clone_evidence("total_capacity", "added_capacity"),
            reference_claims=claims(
                ("累计控股装机为12015.56万千瓦，其中光伏1840.29万千瓦。", "gd-total-capacity"),
                ("上半年新增装机为846万千瓦，其中光伏612万千瓦。", "gd-added-capacity"),
            ),
            evidence_requirements=[
                requirement("cumulative_capacity", [text("total_capacity")]),
                requirement("new_capacity", [text("added_capacity")]),
            ],
            expected_retrieval={"mode": "parallel", "hop_count": 1, "subquery_count": 2},
            sample_metadata=metadata(
                "test", "hard", ["累计装机", "新增装机", "光伏", "口径混淆"], "text"
            ),
        ),
        answerable_sample(
            sample_id="guodian-035",
            question="2025年上半年光伏发电量和上网电量哪个更高，相差多少亿千瓦时？",
            reference_answer="光伏发电量为103.35亿千瓦时，上网电量为102.02亿千瓦时；发电量更高，相差1.33亿千瓦时。",
            question_type="fact",
            evidence=clone_evidence("pv_volume"),
            reference_claims=claims(
                ("光伏发电量为103.35亿千瓦时，上网电量为102.02亿千瓦时。", "gd-pv-volume"),
                ("光伏发电量比上网电量高1.33亿千瓦时。", "gd-pv-volume"),
            ),
            expected_retrieval={"mode": "single", "hop_count": 1},
            sample_metadata=metadata(
                "test", "medium", ["发电量", "上网电量", "光伏", "计算"], "text"
            ),
        ),
        answerable_sample(
            sample_id="guodian-036",
            question="2025年上半年火电和光伏发电量分别是多少，火电比光伏多多少亿千瓦时？",
            reference_answer="火电发电量为1614.67亿千瓦时，光伏发电量为103.35亿千瓦时；火电比光伏多1511.32亿千瓦时。",
            question_type="parallel_multi_evidence",
            evidence=clone_evidence("thermal_volume", "pv_volume"),
            reference_claims=claims(
                ("火电发电量为1614.67亿千瓦时。", "gd-thermal-volume"),
                ("光伏发电量为103.35亿千瓦时。", "gd-pv-volume"),
                ("火电发电量比光伏多1511.32亿千瓦时。", ["gd-thermal-volume", "gd-pv-volume"]),
            ),
            evidence_requirements=[
                requirement("thermal_generation", [text("thermal_volume")]),
                requirement("pv_generation", [text("pv_volume")]),
            ],
            expected_retrieval={"mode": "parallel", "hop_count": 1, "subquery_count": 2},
            sample_metadata=metadata(
                "test", "hard", ["火电", "光伏", "发电量", "跨片段计算"], "text"
            ),
        ),
        answerable_sample(
            sample_id="guodian-037",
            question="先找出2025年上半年归母净利润最高的电源板块，再说明该类电源同期发电量及同比变化。",
            reference_answer="归母净利润最高的是煤电板块，为19.67亿元；对应的火电发电量为1614.67亿千瓦时，同比下降7.40%。",
            question_type="sequential_multi_hop",
            evidence=clone_evidence("segment_profit", "thermal_volume"),
            reference_claims=claims(
                ("归母净利润最高的是煤电板块，为19.67亿元。", "gd-segment-profit"),
                ("对应火电发电量为1614.67亿千瓦时，同比下降7.40%。", "gd-thermal-volume"),
            ),
            evidence_requirements=[
                requirement("identify_highest_profit_segment", [text("segment_profit")]),
                requirement("segment_generation", [text("thermal_volume")]),
            ],
            expected_retrieval={"mode": "sequential", "hop_count": 2, "bridge_value": "煤电"},
            sample_metadata=metadata(
                "test", "hard", ["递进检索", "煤电", "板块利润", "发电量"], "text"
            ),
        ),
        answerable_sample(
            sample_id="guodian-038",
            question="先找出2025年上半年归母净利润最低的电源板块，再查询该类电源截至2025年6月的控股装机容量。",
            reference_answer="归母净利润最低的是气电板块，为0.0018亿元；截至2025年6月气电控股装机容量为202.40万千瓦。",
            question_type="sequential_multi_hop",
            evidence=clone_evidence("segment_profit", "total_capacity"),
            reference_claims=claims(
                ("归母净利润最低的是气电板块，为0.0018亿元。", "gd-segment-profit"),
                ("气电控股装机容量为202.40万千瓦。", "gd-total-capacity"),
            ),
            evidence_requirements=[
                requirement("identify_lowest_profit_segment", [text("segment_profit")]),
                requirement("segment_capacity", [text("total_capacity")]),
            ],
            expected_retrieval={"mode": "sequential", "hop_count": 2, "bridge_value": "气电"},
            sample_metadata=metadata(
                "test", "hard", ["递进检索", "气电", "板块利润", "装机容量"], "text"
            ),
        ),
        answerable_sample(
            sample_id="guodian-039",
            question="2025年上半年新增装机最多的是哪类电源？继续给出该类电源的累计装机和同期归母净利润表现。",
            reference_answer="新增装机最多的是光伏，新增612万千瓦；截至2025年6月光伏累计装机1840.29万千瓦；光伏板块归母净利润5.99亿元，同比增长39.0%。",
            question_type="sequential_multi_hop",
            evidence=clone_evidence("added_capacity", "total_capacity", "segment_profit"),
            reference_claims=claims(
                ("新增装机最多的是光伏，新增612万千瓦。", "gd-added-capacity"),
                ("光伏累计装机为1840.29万千瓦。", "gd-total-capacity"),
                ("光伏板块归母净利润为5.99亿元，同比增长39.0%。", "gd-segment-profit"),
            ),
            evidence_requirements=[
                requirement("identify_largest_new_capacity", [text("added_capacity")]),
                requirement("cumulative_capacity_for_source", [text("total_capacity")]),
                requirement("profit_for_source", [text("segment_profit")]),
            ],
            expected_retrieval={"mode": "sequential", "hop_count": 2, "bridge_value": "光伏"},
            sample_metadata=metadata(
                "test", "hard", ["递进检索", "光伏", "新增装机", "累计装机", "利润"], "text"
            ),
        ),
        answerable_sample(
            sample_id="guodian-040",
            question="截至2025年6月在建风电和光伏项目合计多少万千瓦？哪一类规模更大、相差多少？",
            reference_answer="在建风电211.83万千瓦、在建光伏343.34万千瓦，合计555.17万千瓦；光伏规模更大，比风电多131.51万千瓦。",
            question_type="fact",
            evidence=clone_evidence("projects"),
            reference_claims=claims(
                ("在建风电和光伏规模分别为211.83和343.34万千瓦。", "gd-projects"),
                ("两者合计555.17万千瓦，光伏比风电多131.51万千瓦。", "gd-projects"),
            ),
            expected_retrieval={"mode": "single", "hop_count": 1},
            sample_metadata=metadata(
                "test", "hard", ["在建项目", "风电", "光伏", "计算"], "text"
            ),
        ),
        answerable_sample(
            sample_id="guodian-041",
            question="报告把2025E至2027E归母净利润预测分别下调了多少亿元？请区分当前预测与原预测。",
            reference_answer="当前预测为70.50、78.95和87.17亿元，原预测为74.97、83.22和90.73亿元，因此分别下调4.47、4.27和3.56亿元。",
            question_type="fact",
            evidence=clone_evidence("forecast"),
            reference_claims=claims(
                ("当前预测为70.50、78.95和87.17亿元，原预测为74.97、83.22和90.73亿元。", "gd-profit-forecast"),
                ("2025E至2027E分别下调4.47、4.27和3.56亿元。", "gd-profit-forecast"),
            ),
            expected_retrieval={"mode": "single", "hop_count": 1},
            sample_metadata=metadata(
                "challenge", "hard", ["当前预测", "原预测", "归母净利润", "计算"], "text"
            ),
        ),
        answerable_sample(
            sample_id="guodian-042",
            question="国电电力2025E每股收益和每股红利分别是多少？按这两个预测值计算的每股派息率是多少？",
            reference_answer="2025E每股收益为0.40元，每股红利为0.24元；按两者计算的每股派息率为60%。",
            question_type="parallel_multi_evidence",
            evidence=clone_evidence("forecast", "eps_row", "dividend_row"),
            reference_claims=claims(
                ("2025E每股收益为0.40元。", ["gd-profit-forecast", "gd-eps-row"]),
                ("2025E每股红利为0.24元。", "gd-dividend-row"),
                ("按两个预测值计算的每股派息率为60%。", ["gd-eps-row", "gd-dividend-row"]),
            ),
            evidence_requirements=[
                requirement("eps_2025", [text("forecast")], [text("eps_row")]),
                requirement("dividend_2025", [text("dividend_row")]),
            ],
            expected_retrieval={"mode": "parallel", "hop_count": 1, "subquery_count": 2},
            sample_metadata=metadata(
                "challenge", "hard", ["EPS", "每股红利", "派息率", "表格计算"], "text+table"
            ),
        ),
        answerable_sample(
            sample_id="guodian-043",
            question="报告给出的国电电力2025E和2027E市盈率分别是多少，哪一年更低、低多少倍？",
            reference_answer="2025E市盈率为11.4倍，2027E为9.2倍；2027E更低，低2.2倍。",
            question_type="fact",
            evidence=clone_evidence("forecast", "pe_row"),
            reference_claims=claims(
                ("2025E和2027E市盈率分别为11.4倍和9.2倍。", ["gd-profit-forecast", "gd-pe-row"]),
                ("2027E市盈率低2.2倍。", ["gd-profit-forecast", "gd-pe-row"]),
            ),
            evidence_requirements=[
                requirement("pe_years", [text("forecast")], [text("pe_row")]),
            ],
            expected_retrieval={"mode": "single", "hop_count": 1},
            sample_metadata=metadata(
                "challenge", "medium", ["PE", "年份", "估值", "计算"], "text+table"
            ),
        ),
        answerable_sample(
            sample_id="guodian-044",
            question="财务预测表中2026E和2027E固定资产分别是多少？后一年比前一年变化多少亿元？",
            reference_answer="2026E固定资产为467602百万元，2027E为453840百万元；后一年减少13762百万元，即137.62亿元。",
            question_type="fact",
            evidence=clone_evidence("fixed_assets"),
            reference_claims=claims(
                ("2026E和2027E固定资产分别为467602和453840百万元。", "gd-fixed-assets-row"),
                ("2027E比2026E减少13762百万元，即137.62亿元。", "gd-fixed-assets-row"),
            ),
            expected_retrieval={"mode": "single", "hop_count": 1},
            sample_metadata=metadata(
                "challenge", "hard", ["固定资产", "年份错位", "单位换算", "表格"], "table"
            ),
        ),
        answerable_sample(
            sample_id="guodian-045",
            question="利润预测表中的2026E和2027E财务费用分别是多少？2027E增加了多少亿元？",
            reference_answer="2026E财务费用为10049百万元，2027E为10216百万元；增加167百万元，即1.67亿元。",
            question_type="fact",
            evidence=clone_evidence("finance_expense"),
            reference_claims=claims(
                ("2026E和2027E财务费用分别为10049和10216百万元。", "gd-finance-expense-row"),
                ("2027E比2026E增加167百万元，即1.67亿元。", "gd-finance-expense-row"),
            ),
            expected_retrieval={"mode": "single", "hop_count": 1},
            sample_metadata=metadata(
                "challenge", "hard", ["财务费用", "年份错位", "单位换算", "表格"], "table"
            ),
        ),
        answerable_sample(
            sample_id="guodian-046",
            question="2027E经营活动现金流和企业自由现金流分别是多少？哪一个更高、相差多少亿元？",
            reference_answer="2027E经营活动现金流为29468百万元，企业自由现金流为34791百万元；企业自由现金流更高，相差5323百万元，即53.23亿元。",
            question_type="parallel_multi_evidence",
            evidence=clone_evidence("operating_cf", "free_cf"),
            reference_claims=claims(
                ("2027E经营活动现金流为29468百万元。", "gd-operating-cf-row"),
                ("2027E企业自由现金流为34791百万元。", "gd-free-cf-row"),
                ("企业自由现金流高5323百万元，即53.23亿元。", ["gd-operating-cf-row", "gd-free-cf-row"]),
            ),
            evidence_requirements=[
                requirement("operating_cash_flow", [text("operating_cf")]),
                requirement("free_cash_flow", [text("free_cf")]),
            ],
            expected_retrieval={"mode": "parallel", "hop_count": 1, "subquery_count": 2},
            sample_metadata=metadata(
                "challenge", "hard", ["经营活动现金流", "企业自由现金流", "表格计算"], "table"
            ),
        ),
        answerable_sample(
            sample_id="guodian-047",
            question="可比公司表中的龙源电力与报告预测的国电电力2025E市盈率分别是多少？哪一个更高、相差多少倍？",
            reference_answer="龙源电力2025E市盈率为19.6倍，国电电力为11.4倍；龙源电力更高，相差8.2倍。",
            question_type="parallel_multi_evidence",
            evidence=clone_evidence("longyuan", "forecast", "pe_row"),
            reference_claims=claims(
                ("龙源电力2025E市盈率为19.6倍。", "gd-longyuan-row"),
                ("国电电力2025E市盈率为11.4倍。", ["gd-profit-forecast", "gd-pe-row"]),
                ("龙源电力比国电电力高8.2倍。", ["gd-longyuan-row", "gd-profit-forecast"]),
            ),
            evidence_requirements=[
                requirement("longyuan_pe", [text("longyuan")]),
                requirement("guodian_pe", [text("forecast")], [text("pe_row")]),
            ],
            expected_retrieval={"mode": "parallel", "hop_count": 1, "subquery_count": 2},
            sample_metadata=metadata(
                "challenge", "hard", ["公司混淆", "可比公司", "PE", "跨表比较"], "text+table"
            ),
        ),
        answerable_sample(
            sample_id="guodian-048",
            question="不要混淆图2和图3：哪张图显示2021年归母净利润为负，哪张图显示2025年只有第一、第二季度营业收入柱？",
            reference_answer="图3显示2021年归母净利润为负；图2显示2025年只有第一季度和第二季度的单季营业收入柱。",
            question_type="parallel_multi_evidence",
            evidence=clone_evidence("figure_profit", "figure_quarterly_revenue"),
            reference_claims=claims(
                ("图3显示2021年归母净利润为负。", "gd-figure-profit"),
                ("图2显示2025年只有第一季度和第二季度的单季营业收入柱。", "gd-figure-quarterly-revenue"),
            ),
            evidence_requirements=[
                requirement("profit_figure", [text("figure_profit")]),
                requirement("quarterly_revenue_figure", [text("figure_quarterly_revenue")]),
            ],
            expected_retrieval={"mode": "parallel", "hop_count": 1, "subquery_count": 2},
            sample_metadata=metadata(
                "challenge", "hard", ["图表定位", "图2", "图3", "近邻混淆"], "figure"
            ),
        ),
        unanswerable_sample(
            sample_id="guodian-049",
            question="国电电力2025年上半年财务费用金额是多少亿元，同比增减了多少亿元？",
            reference_answer="报告只披露了2025年上半年财务费用率及其同比变动，没有披露同期财务费用金额及金额同比变化，无法据此回答。",
            expected_retrieval={"mode": "single", "hop_count": 1},
            sample_metadata=metadata(
                "challenge",
                "hard",
                ["近邻负例", "财务费用", "财务费用率", "金额缺失"],
                "none",
                unanswerable_reason="报告披露财务费用率，不披露2025年上半年财务费用金额及金额同比变化。",
            ),
        ),
        unanswerable_sample(
            sample_id="guodian-050",
            question="国电电力2025年第二季度单季度经营性净现金流是多少，同比增长多少？",
            reference_answer="报告披露的是2025年上半年累计经营性净现金流，没有给出第二季度单季度经营性净现金流及同比增速，无法据此回答。",
            expected_retrieval={"mode": "single", "hop_count": 1},
            sample_metadata=metadata(
                "challenge",
                "hard",
                ["时间粒度", "近邻负例", "单季度", "现金流"],
                "none",
                unanswerable_reason="报告只有上半年累计经营性净现金流，未披露第二季度单季度口径。",
            ),
        ),
    ]


def main() -> None:
    migrated = [migrate_v1_sample(sample) for sample in load_jsonl(V1_PATH)]
    samples = [*migrated, *new_samples()]
    if len(samples) != 50:
        raise RuntimeError(f"Expected 50 samples, got {len(samples)}")
    if len({sample["id"] for sample in samples}) != len(samples):
        raise RuntimeError("Duplicate sample IDs")

    content = "\n".join(
        json.dumps(sample, ensure_ascii=False, separators=(",", ":"))
        for sample in samples
    )
    OUTPUT_PATH.write_text(content + "\n", encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
