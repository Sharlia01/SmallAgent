"""Readable, self-contained E2E reports; also render saved runs without model calls."""

from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path
from typing import Any


SCORES = (
    ("正确性", "answer_metrics", "correctness_score"),
    ("完整性", "answer_metrics", "completeness_score"),
    ("忠实度", "faithfulness_metrics", "faithfulness_score"),
    ("引用有效性", "citation_metrics", "citation_validity"),
    ("引用精确率", "citation_metrics", "citation_precision"),
    ("引用召回率", "citation_metrics", "citation_recall"),
)


def _text(value: Any) -> str:
    return escape(str(value) if value is not None else "", quote=True)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _percent(value: Any) -> str:
    return f"{value:.1%}" if _number(value) else "未评分"


def _issues(result: dict[str, Any]) -> list[str]:
    issues = []
    if result.get("error"):
        issues.append(f"执行失败：{result['error']}")
    for label, section in (("答案", "answer_metrics"), ("忠实度", "faithfulness_metrics"), ("引用", "citation_metrics")):
        metrics = result.get(section) or {}
        if metrics.get("judge_error"):
            issues.append(f"{label}评分失败：{metrics['judge_error']}")
    answer = result.get("answer_metrics") or {}
    if any(_number(answer.get(key)) and answer[key] < 1 for key in ("correctness_score", "completeness_score")):
        issues.append("答案未获满分：" + str(answer.get("judge_reason") or "未记录评分理由"))
    faithfulness = result.get("faithfulness_metrics") or {}
    for claim in faithfulness.get("unsupported_claims") or []:
        issues.append(f"主张缺少证据支持：{claim.get('claim', '')}；{claim.get('reason', '')}")
    if _number(faithfulness.get("faithfulness_score")) and faithfulness["faithfulness_score"] < 1 and not faithfulness.get("unsupported_claims"):
        issues.append("忠实度未获满分，未记录具体主张理由。")
    citation = result.get("citation_metrics") or {}
    if citation.get("invalid_citation_ids"):
        issues.append(f"引用编号无效：{citation['invalid_citation_ids']}")
    for claim in citation.get("claims") or []:
        if claim.get("requires_citation") and (not claim.get("valid_citation_ids") or claim.get("supported") is False):
            issues.append(f"引用缺失或不支持主张：{claim.get('text', '')}；{claim.get('judge_reason') or '未记录理由'}")
    if any(_number(citation.get(key)) and citation[key] < 1 for key in ("citation_validity", "citation_precision", "citation_recall")) and not any(item.startswith("引用编号") or item.startswith("引用缺失") for item in issues):
        issues.append("引用评分未获满分，未记录具体理由。")
    refusal = result.get("refusal_metrics") or {}
    if refusal.get("correct") is False:
        issues.append(f"回答行为不符合预期：应为 {refusal.get('expected_behavior')}，实际为 {refusal.get('actual_behavior')}。")
    if (result.get("sufficiency_metrics") or {}).get("correct") is False:
        gate = (result.get("retrieval") or {}).get("evidence_sufficiency") or {}
        issues.append("证据充分性判断与预期不符：" + str(gate.get("reason") or "未记录原因"))
    return list(dict.fromkeys(issues))


def render_html_report(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Escape all model/dataset text and omit raw retrieval records and thinking."""
    cards = []
    attention_count = 0
    for position, result in enumerate(results, 1):
        issues = _issues(result)
        attention_count += bool(issues)
        scores = []
        for label, section, key in SCORES:
            value = (result.get(section) or {}).get(key)
            tone = "missing" if not _number(value) else "good" if value == 1 else "bad"
            scores.append(f'<span class="score {tone}">{label} <b>{_percent(value)}</b></span>')
        behavior = (result.get("refusal_metrics") or {}).get("correct")
        if isinstance(behavior, bool):
            scores.append(f'<span class="score {"good" if behavior else "bad"}">回答行为 <b>{"符合预期" if behavior else "不符合预期"}</b></span>')
        missing = any(not _number((result.get(section) or {}).get(key)) for _, section, key in SCORES)
        status = "需关注" if issues else "部分指标未评分" if missing else "已评分，未发现异常"
        reason = '<ul>' + ''.join(f'<li>{_text(item)}</li>' for item in issues) + '</ul>' if issues else '<p>未记录失败原因。未评分指标不代表通过。</p>'
        latency = (result.get("latency_ms") or {}).get("total")
        timing = f" · {latency / 1000:.2f} 秒" if _number(latency) else ""
        cards.append(f'''<article class="case" data-attention="{str(bool(issues)).lower()}">
<div class="case-meta"><span>题目 {position} · {_text(result.get('id'))} · {_text(result.get('question_type'))}{timing}</span><span class="status {'bad' if issues else 'missing' if missing else 'good'}">{status}</span></div>
<h2>{_text(result.get('question'))}</h2><div class="scores">{''.join(scores)}</div>
<div class="answers"><section><h3>参考答案</h3><div class="answer">{_text(result.get('reference_answer')) or '未提供'}</div></section><section><h3>实际答案</h3><div class="answer">{_text(result.get('answer')) or '未生成答案'}</div></section></div>
<section class="reasons"><h3>失败原因 / 需关注项</h3>{reason}</section></article>''')
    overall = (summary.get("metrics") or {}).get("overall") or {}
    overview = [('题目数', str(len(results))), ('需关注', str(attention_count)), ('答案正确性', _percent(overall.get('answer_correctness'))), ('答案完整性', _percent(overall.get('answer_completeness'))), ('忠实度', _percent(overall.get('faithfulness'))), ('忠实度评分覆盖率', _percent(overall.get('faithfulness_coverage')))]
    tiles = ''.join(f'<div class="tile"><span>{label}</span><strong>{value}</strong></div>' for label, value in overview)
    name = _text(summary.get('run_name') or '端到端评测')
    return '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>''' + name + ''' · 评测报告</title><style>
:root{color-scheme:light;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;color:#203047;background:#f3f5f9}*{box-sizing:border-box}body{margin:0}main{max-width:1180px;margin:auto;padding:40px 24px}header{margin-bottom:28px}h1{font-size:30px;margin:10px 0;overflow-wrap:anywhere}.eyebrow{color:#4263a1;font-weight:650;font-size:13px;letter-spacing:.12em}.subtitle,.note{color:#657388;line-height:1.7;font-size:14px}.overview{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:24px 0}.tile{background:white;border:1px solid #e1e6ef;border-radius:12px;padding:18px}.tile span{display:block;color:#657388;font-size:12px}.tile strong{display:block;font-size:25px;margin-top:10px}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:24px 0}input,select{font:inherit;border:1px solid #ccd5e2;border-radius:8px;padding:11px;background:white}input{flex:1;min-width:220px}#count{font-size:13px;color:#657388}.case{background:white;border:1px solid #e1e6ef;border-radius:14px;padding:24px;margin-bottom:20px;box-shadow:0 3px 12px #20304704}.case-meta{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;font-size:12px;color:#657388}h2{font-size:19px;line-height:1.6;margin:16px 0}h3{font-size:13px;color:#657388;margin:0 0 12px}.scores{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}.score,.status{border-radius:6px;padding:6px 9px;font-size:12px}.score b{margin-left:5px}.good{color:#14674b;background:#eaf7f0}.bad{color:#a33c34;background:#fff0eb}.missing{color:#677080;background:#eef1f5}.answers{display:grid;grid-template-columns:1fr 1fr;gap:16px}.answers section{background:#f7f9fc;border-radius:10px;padding:18px}.answer{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;line-height:1.85}.reasons{border-top:1px solid #e7ebf2;margin-top:20px;padding-top:18px;font-size:13px;line-height:1.8;overflow-wrap:anywhere}.reasons p{color:#657388;margin:0}.reasons ul{margin:0;padding-left:20px;color:#a33c34}[hidden]{display:none!important}#empty{padding:30px;text-align:center;color:#657388}@media(max-width:850px){.overview{grid-template-columns:repeat(3,1fr)}}@media(max-width:600px){main{padding:24px 14px}.overview{grid-template-columns:repeat(2,1fr)}.answers{grid-template-columns:1fr}.case{padding:18px}h1{font-size:24px}}@media print{.toolbar{display:none}.case{break-inside:avoid;box-shadow:none}body{background:white}main{padding:0}}
</style></head><body><main><header><div class="eyebrow">AGENTIC RAG · 评测报告</div><h1>''' + name + '''</h1><p class="subtitle">问题、答案与主要评分。详细检索记录保存在同名 JSONL 原始文件中。</p></header>
<div class="overview">''' + tiles + '''</div><p class="note">分数按百分比展示；低于满分列为需关注。未评分表示未执行、不适用或评分失败，不计作零分。答案中的引用标记保留原样。</p>
<div class="toolbar"><input id="search" type="search" aria-label="搜索题目、答案或失败原因" placeholder="搜索题目、答案或失败原因"><select id="filter" aria-label="筛选题目"><option value="all">全部题目</option><option value="attention">只看需关注</option></select><span id="count" aria-live="polite"></span></div>
<div id="cases">''' + ''.join(cards) + '''</div><p id="empty" hidden>没有匹配的题目。</p>
</main><script>
const search=document.getElementById('search'),filter=document.getElementById('filter'),cases=[...document.querySelectorAll('.case')];
function update(){const query=search.value.trim().toLocaleLowerCase();let visible=0;for(const item of cases){item.hidden=!(item.textContent.toLocaleLowerCase().includes(query)&&(filter.value==='all'||item.dataset.attention==='true'));if(!item.hidden)visible++;}document.getElementById('count').textContent=`显示 ${visible} / ${cases.length} 题`;document.getElementById('empty').hidden=visible!==0;}
search.addEventListener('input',update);filter.addEventListener('change',update);update();
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Saved E2E JSONL file")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary_path = args.summary or args.results.with_name(args.results.stem + "_summary.json")
    output_path = args.output or args.results.with_suffix(".html")
    if output_path.resolve() in {args.results.resolve(), summary_path.resolve()}:
        parser.error("HTML output must not overwrite the source results or summary")
    if output_path.exists() and not args.overwrite:
        parser.error("HTML report already exists; use --overwrite to replace it")
    results = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_path.write_text(render_html_report(results, summary), encoding="utf-8")
    print(f"HTML report: {output_path.resolve()}")


if __name__ == "__main__":
    main()
