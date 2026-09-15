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

检索层现已通过 `retrieve_raw_results()` 保留完整结果，包括 `chunk_id`、各召回分支的原始排名与分数、RRF 分数和最终语义重排分数。当前评测集尚未把标准证据映射到 ES 中的稳定 `chunk_id`，因此评测脚本先校验文档名，再对返回片段和 `relevant_evidence[].text` 做规范化文本匹配。完成第一轮基线并确认切块稳定后，可以把标准 chunk ID 补入证据对象，进一步减少模糊匹配。

检索入口默认先执行 RAG 查询意图识别，再只对确有需要的口语、简称或上下文查询进行规范化和关键词扩展。结果中的 `retrieval.query_intent` 记录意图、是否需要改写、是否建议分解、原因及判定来源；`retrieval.query_rewrite` 记录原问题和规范化结果；`retrieval.query_transform` 记录扩展查询、最终生效查询及两级保留校验。检索器分别执行原问题关键词、扩展查询关键词和原问题向量召回，使用加权 RRF 合并后截取候选，最后只使用原问题进行语义重排；没有有效扩展时只执行原问题关键词和原问题向量两路召回。重排后的最终 Top 片段还会执行证据充分性检查，不能完整覆盖问题时清空 `chunks`，并在 `retrieval.evidence_sufficiency` 中记录缺失条件和判定来源。若要恢复原来的全量改写策略，可设置 `RAG_QUERY_INTENT_ENABLED=false`；若要跑完全不含改写与扩展的对照基线，可设置 `QUERY_REWRITE_ENABLED=false`；若只关闭扩展，可设置 `QUERY_EXPANSION_ENABLED=false`；若要关闭充分性检查，可设置 `RAG_EVIDENCE_SUFFICIENCY_ENABLED=false`。

语义重排后不会直接按模型分数输出，而是进行第二阶段加权 RRF：语义重排名次权重为 `0.7`，第一阶段检索 RRF 名次权重为 `0.3`，默认平滑常数为 `10`。这种做法只融合两边的名次，不直接混合量纲不同的原始分数。最终再执行确定性的限定条件兼容性检查，当前覆盖时间范围、实际值/预测值/目标、表图定位符和明确实体；候选未出现可比限定时记为 `unknown`，只有出现明确互斥限定时才记为 `conflict`。明确冲突先降级，其余候选按第二阶段融合分排序。每个结果的 `constraint_compatibility` 和 `final_ranking` 字段分别保存限定诊断、两路名次及融合贡献。

对于 `evidence_type=figure` 的样本，`text` 是人工核对图片后写下的视觉事实，不要求它作为连续文字出现在 PDF 文本层中。这类样本用于检查当前文档解析和检索链路是否真正保留了图表信息，应与纯文本、表格样本分开统计。

## 数据质量要求

1. 所有可回答题目的参考答案必须被 `relevant_evidence` 完整支持。
2. 不使用仅靠常识即可回答的问题。
3. 不把当前 RAG 系统生成的回答当成参考答案。
4. 不可回答题必须保持 `relevant_evidence` 为空，系统应明确表示文档未提供答案。
5. 修改样本内容时同步更新 `dataset_version`，并重新运行 JSONL 结构校验。

## 运行检索评估

评测脚本直接调用生产环境使用的检索器，不调用最终回答模型，但会调用查询处理、重排和证据充分性校验所配置的模型。它会保存 Top K chunk、`chunk_id`、各分支排名和原始分数、RRF 分数、语义重排分数、充分性诊断、位置和耗时，但不会把体积较大的 embedding 向量写入结果文件。
当前 Embedding 为本地 `bge-small-zh-v1.5`（512 维）。旧的 1024 维
`text-embedding-v3` 文档必须重新上传生成 `q_512_vec` 后才能评测。
重排使用 DashScope `qwen3.7-text-rerank`；切换重排服务不需要重新生成向量。

Docker Compose 会把 `backend/evals` 挂载到 API 容器的 `/app/evals`。首次加入该挂载后，重新创建 API 容器：

```bash
cd backend
docker compose up -d --force-recreate LS_api
```

先只校验数据集，不调用 Elasticsearch、本地 BGE 或 DashScope：

```bash
docker compose exec LS_api \
  python /app/evals/run_retrieval_eval.py \
  --validate-only
```

当前系统把登录用户的数字 ID 作为知识库 Elasticsearch 索引名。正式运行前，可从 PostgreSQL 查询用户 ID：

```bash
docker compose exec LS_pg \
  sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT id, username FROM users ORDER BY id;"'
```

建议先跑一条样本，确认 Elasticsearch、本地 BGE embedding、
DashScope `qwen3.7-text-rerank` 和查询变换配置正常：

```bash
docker compose exec LS_api \
  python /app/evals/run_retrieval_eval.py \
  --index-name <用户ID> \
  --limit 1 \
  --run-name retrieval_smoke
```

再运行完整的 Top 5 基线：

```bash
docker compose exec LS_api \
  python /app/evals/run_retrieval_eval.py \
  --index-name <用户ID> \
  --candidate-size 100 \
  --rerank-candidate-size 20 \
  --rrf-k 60 \
  --vector-weight 0.6 \
  --final-reranker-weight 0.7 \
  --final-rrf-k 10 \
  --run-name retrieval_baseline
```

`--vector-weight` 现在表示原问题向量分支在加权 RRF 中的权重；剩余权重分配给关键词分支。存在扩展查询时，两路关键词平分剩余权重。
`--final-reranker-weight` 表示第二阶段语义重排名次的权重，其余权重自动分配给第一阶段检索 RRF 名次。
`--rerank-candidate-size` 会直接影响在线重排的输入数量、请求耗时和费用。证据充分性校验只读取最终 Top K 片段，每道题额外调用一次轻量文本模型。

结果写入：

- `results/retrieval_baseline.jsonl`：逐题检索结果、证据匹配和指标。
- `results/retrieval_baseline_summary.json`：整体、题型和证据模态切片指标。

同名结果默认不会被覆盖。需要明确覆盖时添加 `--overwrite`，或通过 `--run-name` 使用一个新名称。

当前自动计算：

- `Hit@K`：Top K 是否至少命中一条标准证据。
- `Recall@K`：每道题的标准证据召回比例，再做宏平均。
- `MRR@K`：第一条正确证据排名的倒数，再做宏平均。
- `empty_rate`：没有返回任何 chunk 的查询比例。
- `answerable_empty_rate`：可回答题被证据门清空的比例，越低越好。
- `unanswerable_rejection_rate`：不可回答题被证据门正确拒绝的比例，越高越好。
- `evidence_sufficiency_fallback_rate`：充分性模型调用失败而进入 fallback 的比例。
- `average_latency_ms`：成功查询的平均检索耗时。

不可回答题没有正向标准证据，因此不计入 Hit、Recall 和 MRR；摘要中单独给出它们的 `unanswerable_empty_rate`。图表题的证据是人工视觉转录，只有检索 chunk 真正包含对应视觉事实时才算命中。
