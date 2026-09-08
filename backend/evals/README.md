# RAG 评估

本目录保存 RAG 系统的离线评估数据和后续评估脚本。JSONL 是评测集的主格式，每行是一个完整、独立的 JSON 对象。

## 当前数据集

`data/guodian_power_eval_v1.jsonl` 基于 `backend/国电电力.pdf` 制作，共 28 条：

- 18 条 `fact`：单个证据片段可直接回答，其中包含 6 条表格题和 2 条图表题。
- 4 条 `multi_hop`：需要联合多个证据片段回答。
- 3 条 `paraphrase`：使用口语或不同表达方式提问。
- 3 条 `unanswerable`：主题相关，但文档没有提供答案。

全部样本当前属于 `test`，不应拿来调整提示词、检索参数或模型。后续调参应另建 `dev` 数据集，避免测试集泄漏。

## 数据结构

- `schema_version`：JSONL 结构版本。
- `dataset_version`：评测集版本，结果文件应记录该值。
- `id`：样本的稳定唯一标识。
- `question`：模拟真实用户的提问。
- `reference_answer`：只根据文档内容编写的参考答案。
- `answerable`：文档能否回答问题。
- `question_type`：`fact`、`multi_hop`、`paraphrase` 或 `unanswerable`。
- `relevant_evidence`：标准证据数组；每项包含文件名、PDF 页码和证据内容。若同一段原文在摘要和正文中重复出现，`page` 记录它首次出现的物理页码，页码从 PDF 首页按 1 开始计算。表格和图表证据还包含 `evidence_type` 与 `locator`，分别记录证据模态和具体表格/图号位置。
- `metadata`：难度、标签、数据划分、源文档日期及不可回答原因等信息。

检索层现已通过 `retrieve_raw_results()` 保留完整结果，包括 `chunk_id` 和三类排名得分。当前评测集尚未把标准证据映射到 ES 中的稳定 `chunk_id`，因此评测脚本先校验文档名，再对返回片段和 `relevant_evidence[].text` 做规范化文本匹配。完成第一轮基线并确认切块稳定后，可以把标准 chunk ID 补入证据对象，进一步减少模糊匹配。

对于 `evidence_type=figure` 的样本，`text` 是人工核对图片后写下的视觉事实，不要求它作为连续文字出现在 PDF 文本层中。这类样本用于检查当前文档解析和检索链路是否真正保留了图表信息，应与纯文本、表格样本分开统计。

## 数据质量要求

1. 所有可回答题目的参考答案必须被 `relevant_evidence` 完整支持。
2. 不使用仅靠常识即可回答的问题。
3. 不把当前 RAG 系统生成的回答当成参考答案。
4. 不可回答题必须保持 `relevant_evidence` 为空，系统应明确表示文档未提供答案。
5. 修改样本内容时同步更新 `dataset_version`，并重新运行 JSONL 结构校验。

## 运行检索评估

评测脚本直接调用生产环境使用的检索器，不调用聊天模型。它会保存 Top K chunk、`chunk_id`、综合得分、向量得分、关键词得分、位置和耗时，但不会把体积较大的 embedding 向量写入结果文件。
当前 Embedding 为本地 `bge-small-zh-v1.5`（512 维）。旧的 1024 维
`text-embedding-v3` 文档必须重新上传生成 `q_512_vec` 后才能评测。

Docker Compose 会把 `backend/evals` 挂载到 API 容器的 `/app/evals`。首次加入该挂载后，重新创建 API 容器：

```bash
cd backend
docker compose up -d --force-recreate swxy_api
```

先只校验数据集，不调用 Elasticsearch、本地 BGE 或 DashScope：

```bash
docker compose exec swxy_api \
  python /app/evals/run_retrieval_eval.py \
  --validate-only
```

当前系统把登录用户的数字 ID 作为知识库 Elasticsearch 索引名。正式运行前，可从 PostgreSQL 查询用户 ID：

```bash
docker compose exec gsk_pg \
  sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT id, username FROM users ORDER BY id;"'
```

建议先跑一条样本，确认 Elasticsearch、本地 BGE 和 DashScope
`qwen3-rerank` 配置正常：

```bash
docker compose exec swxy_api \
  python /app/evals/run_retrieval_eval.py \
  --index-name <用户ID> \
  --limit 1 \
  --run-name retrieval_smoke
```

再运行完整的 Top 5 基线：

```bash
docker compose exec swxy_api \
  python /app/evals/run_retrieval_eval.py \
  --index-name <用户ID> \
  --run-name retrieval_baseline
```

结果写入：

- `results/retrieval_baseline.jsonl`：逐题检索结果、证据匹配和指标。
- `results/retrieval_baseline_summary.json`：整体、题型和证据模态切片指标。

同名结果默认不会被覆盖。需要明确覆盖时添加 `--overwrite`，或通过 `--run-name` 使用一个新名称。

当前自动计算：

- `Hit@K`：Top K 是否至少命中一条标准证据。
- `Recall@K`：每道题的标准证据召回比例，再做宏平均。
- `MRR@K`：第一条正确证据排名的倒数，再做宏平均。
- `empty_rate`：没有返回任何 chunk 的查询比例。
- `average_latency_ms`：成功查询的平均检索耗时。

不可回答题没有正向标准证据，因此不计入 Hit、Recall 和 MRR；摘要中单独给出它们的 `unanswerable_empty_rate`。图表题的证据是人工视觉转录，只有检索 chunk 真正包含对应视觉事实时才算命中。
