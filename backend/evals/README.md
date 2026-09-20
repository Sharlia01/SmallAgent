# RAG 评估

本目录保存 RAG 系统的离线评估数据和后续评估脚本。JSONL 是评测集的主格式，每行是一个完整、独立的 JSON 对象。

## 当前数据集

默认使用 `data/power_reports_eval_v2.jsonl`。它基于 `backend/国电电力.pdf` 和
`backend/润本股份.pdf` 逐页核对制作，共 36 条，并按用途隔离：

- `dev` 8 条：允许用于提示词、检索参数和充分性阈值调试。
- `test` 16 条：以未在 `dev` 中出现的润本股份问题为主，用于常规回归。
- `challenge` 12 条：6 条跨文档比较题和 6 条高混淆不可回答题，用于压力测试。

题型覆盖 9 条单事实题、11 条并行多证据题、3 条递进式多跳题、2 条口语改写题和
11 条不可回答题；来源覆盖正文与表格。不可回答题包含时间错位、近邻指标替换、
缺失拆分、实时数据、反事实敏感性和跨文档缺失信息，不只是与主题无关的问题。

`data/guodian_power_eval_v1.jsonl` 是兼容保留的首版单文档基线，共 28 条：

- 18 条 `fact`：单个证据片段可直接回答，其中包含 6 条表格题和 2 条图表题。
- 4 条 `multi_hop`：需要联合多个证据片段回答。
- 3 条 `paraphrase`：使用口语或不同表达方式提问。
- 3 条 `unanswerable`：主题相关，但文档没有提供答案。

v1 全部样本都标为 `test`，历史上若已用它调过参数，就不应再把其分数当作无泄漏的
泛化结果。新实验优先使用 v2，并只在 `dev` 上调参。

`data/guodian_power_eval_v2.jsonl` 是国电电力单文档专项集，共 50 条，用于在保留
v1 历史基线的同时加强召回回归：

- `dev` 28 条：由 v1 完整迁移并标记 `legacy_v1`，不得当作未见测试集。
- `test` 12 条：重点覆盖实际值/预测值、累计值/单季度、费用率/费用金额、累计装机/
  新增装机，以及真正依赖桥接值的递进检索。
- `challenge` 10 条：覆盖年份错位、单位换算、跨表计算、近邻图表、相似指标缺失和
  时间粒度缺失。

该专项集包括 45 条可回答题和 5 条不可回答题，题型分布为 25 条单事实题、14 条
并行多证据题、3 条递进式多跳题、3 条口语改写题和 5 条不可回答题。标准证据覆盖
正文、表格和图表；多证据题使用 `evidence_requirements` 表示必须覆盖的独立信息，
同一事实存在正文与表格两种合法来源时使用 `alternatives` 表示任一来源均可命中。

该文件由 `build_guodian_power_eval_v2.py` 从冻结的 v1 基线和人工编写的新增题确定性
生成。修改构建定义后重新生成并校验：

```bash
cd backend
python evals/build_guodian_power_eval_v2.py
python evals/run_retrieval_eval.py \
  --dataset evals/data/guodian_power_eval_v2.jsonl \
  --validate-only
```

## 数据结构

- `schema_version`：JSONL 结构版本。
- `dataset_version`：评测集版本，结果文件应记录该值。
- `id`：样本的稳定唯一标识。
- `question`：模拟真实用户的提问。
- `reference_answer`：只根据文档内容编写的参考答案。
- `answerable`：文档能否回答问题。
- `question_type`：v2 使用 `fact`、`parallel_multi_evidence`、
  `sequential_multi_hop`、`paraphrase` 或 `unanswerable`。
- `expected_behavior`：端到端流程应当 `answer` 还是 `refuse`。
- `reference_claims`：把参考答案拆成可独立核验的主张，并通过
  `required_evidence_ids` 绑定标准证据。
- `expected_retrieval`：期望的 `single`、`parallel` 或 `sequential` 检索模式及跳数；
  当前用于诊断和后续路由评测，不参与检索相关性分数。
- `relevant_evidence`：标准证据数组；每项包含文件名、PDF 页码和证据内容。若同一段原文在摘要和正文中重复出现，`page` 记录它首次出现的物理页码，页码从 PDF 首页按 1 开始计算。表格和图表证据还包含 `evidence_type` 与 `locator`，分别记录证据模态和具体表格/图号位置。
- `evidence_requirements`：多证据题的强约束；只有所有要求都命中时，Hit 才算成功。
- `metadata`：难度、标签、`dev/test/challenge` 数据划分、源文档日期及不可回答原因等信息。

v2 校验还会检查证据 ID 唯一性、claim 到 evidence 的引用关系、期望行为、检索模式，
并要求不可回答题显式标注 `expected_evidence_sufficient=false`。

检索层现已通过 `retrieve_raw_results()` 保留完整结果，包括 `chunk_id`、各召回分支的原始排名与分数、RRF 分数和最终语义重排分数。当前评测集尚未把标准证据映射到 ES 中的稳定 `chunk_id`，因此评测脚本先校验文档名，再对返回片段和 `relevant_evidence[].text` 做规范化文本匹配。完成第一轮基线并确认切块稳定后，可以把标准 chunk ID 补入证据对象，进一步减少模糊匹配。

检索入口默认先执行 RAG 查询意图识别，再只对确有需要的口语、简称或上下文查询进行规范化和关键词扩展。结果中的 `retrieval.query_intent` 记录意图、`single/parallel/sequential` 检索模式、原因及可执行子查询计划；`retrieval.query_rewrite` 记录原问题和规范化结果；`retrieval.query_transform` 记录扩展查询、最终生效查询及两级保留校验。只有 `sequential` 会执行递进式二跳：第一跳证据中经过原文校验的桥接值用于生成第二跳查询，同文档问题还会继承文档范围。结果中的 `retrieval.sequential_retrieval` 保存桥接值、来源 chunk 和每跳 query，chunk 的 `sequential_subqueries` 保存它覆盖的步骤。普通检索仍分别执行原问题关键词、扩展查询关键词和原问题向量召回，使用加权 RRF 合并后截取候选；没有有效扩展时只执行原问题关键词和原问题向量两路召回。所有检索跳完成后统一执行证据充分性检查，不能完整覆盖问题时清空 `chunks`，并在 `retrieval.evidence_sufficiency` 中记录缺失条件和判定来源。若要恢复原来的全量改写策略，可设置 `RAG_QUERY_INTENT_ENABLED=false`；若只关闭递进式检索并回退单跳，可设置 `RAG_SEQUENTIAL_RETRIEVAL_ENABLED=false`；若要跑完全不含改写与扩展的对照基线，可设置 `QUERY_REWRITE_ENABLED=false`；若只关闭扩展，可设置 `QUERY_EXPANSION_ENABLED=false`；若要关闭充分性检查，可设置 `RAG_EVIDENCE_SUFFICIENCY_ENABLED=false`。

每一跳的语义重排后都会进行第二阶段加权 RRF：语义重排名次权重为 `0.4`，第一阶段检索 RRF 名次权重为 `0.6`，默认平滑常数为 `10`。这种做法只融合两边的名次，不直接混合量纲不同的原始分数。再执行确定性的限定条件兼容性检查，当前覆盖时间范围、实际值/预测值/目标、表图定位符和明确实体；候选未出现可比限定时记为 `unknown`，只有出现明确互斥限定时才记为 `conflict`。明确冲突先降级，其余候选按第二阶段融合分排序。

递进式检索还会把两跳结果去重，以用户原始问题对合并候选统一重排一次，
并按原问题重新检查限定条件。不同跳的原始分数不直接比较。首页容量允许时，
保留第一跳已校验的桥接来源片段以及第二跳按本跳排名选出的独立证据，
其余位置按统一排名补齐；最后按统一排名展示选中的证据，不按检索跳次固定置顶。
统一重排不再施加相似度阈值，以免丢失对原问题分数较低、但回答所必需的辅助证据。
`top_k=1` 时无法保证两跳覆盖，按原问题排名返回一条结果。

`retrieval.retrieval_fusion.final_rerank` 记录原始问题、候选数、判定来源、
保留的证据 ID 和失败原因。候选数最多为两跳返回片段数之和，去重后只请求一次模型。
每个 chunk 的 `final_ranking` 记录统一重排名次与 `coverage_reserved`；
本跳的语义分数、RRF 分数、融合名次和限定诊断保存在 `sequential_subqueries` 中。
如果额外重排调用失败，保留已取得的两跳证据，按各跳名次做等权 RRF 回退
（使用 `rrf_k`）；`final_rerank.source=fallback`，统一语义分数为 `null`，
`final_ranking.fallback_rrf_score` 记录回退分数。

对于 `evidence_type=figure` 的样本，`text` 是人工核对图片后写下的视觉事实。当前检索评分优先匹配 `locator`（图标题），因此命中表示找到了目标图的文字定位信息，不代表片段已保留视觉事实或足以回答问题。图表样本应与纯文本、表格样本分开统计。

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
重排使用本地 `BAAI/bge-reranker-v2-m3`；切换重排模型不需要重新生成向量。

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

只校验某个数据划分：

```bash
docker compose exec LS_api \
  python /app/evals/run_retrieval_eval.py \
  --validate-only \
  --split test
```

运行 v2 前，目标用户知识库必须同时包含 `国电电力.pdf` 和 `润本股份.pdf`，且文件名
保持不变。`--split` 可重复，也可写成逗号分隔，例如 `--split test,challenge`。

当前系统把登录用户的数字 ID 作为知识库 Elasticsearch 索引名。正式运行前，可从 PostgreSQL 查询用户 ID：

```bash
docker compose exec LS_pg \
  sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT id, username FROM users ORDER BY id;"'
```

建议先跑一条样本，确认 Elasticsearch、本地 BGE embedding、
本地 `bge-reranker-v2-m3` 和查询变换配置正常：

```bash
docker compose exec LS_api \
  python /app/evals/run_retrieval_eval.py \
  --index-name <用户ID> \
  --limit 1 \
  --run-name retrieval_smoke
```

再运行完整的 Top 10 基线，同时得到 `@1/@3/@5/@10` 指标：

```bash
docker compose exec LS_api \
  python /app/evals/run_retrieval_eval.py \
  --index-name <用户ID> \
  --top-k 10 \
  --candidate-size 100 \
  --rerank-candidate-size 20 \
  --rrf-k 60 \
  --vector-weight 0.6 \
  --final-reranker-weight 0.4 \
  --final-rrf-k 10 \
  --split test \
  --run-name retrieval_baseline
```

`--vector-weight` 现在表示原问题向量分支在加权 RRF 中的权重；剩余权重分配给关键词分支。存在扩展查询时，两路关键词平分剩余权重。
`--final-reranker-weight` 表示第二阶段语义重排名次的权重，其余权重自动分配给第一阶段检索 RRF 名次。
`--rerank-candidate-size` 会直接影响在线重排的输入数量、请求耗时和费用。证据充分性校验只读取最终 Top K 片段，每道题额外调用一次轻量文本模型。

结果写入：

- `results/retrieval_baseline.jsonl`：逐题检索结果、证据匹配和指标。
- `results/retrieval_baseline_summary.json`：整体、题型和证据模态切片指标。

同名结果默认不会被覆盖。需要明确覆盖时添加 `--overwrite`，或通过 `--run-name` 使用一个新名称。

新结果使用 `result_schema_version: "2.1"`，将指标分为三组：

| 分组 | 数据来源 | 用途 |
| --- | --- | --- |
| `retrieval_metrics` | 充分性检查前的 Top K chunks | 衡量检索是否找到了标准证据 |
| `gate_metrics` | 充分性判定及样本的 `answerable` 标签 | 衡量拒答、放行及检查失败情况 |
| `post_gate_metrics` | 充分性检查后的 Top K chunks | 衡量最终交给回答阶段的证据 |

逐题 JSONL 中，这三组位于根节点。汇总文件中的整体指标路径分别为
`metrics.overall.retrieval_metrics`、`metrics.overall.gate_metrics`、
`metrics.overall.post_gate_metrics`；`by_question_type` 和 `by_source_modality`
中的每个切片也包含同样三组。v2 结果还提供 `by_split`，避免把调参集、常规测试集
和挑战集混成一个总分。

检查前后的检索评分使用相同的证据匹配规则：

- `Hit@K`：Top K 是否至少命中一条标准证据。
- `Recall@K`：每道题的标准证据召回比例，再做宏平均。
- `Precision@K`：Top K 中命中任一标准证据或 requirement atom 的 chunk 数除以 K；
  因此返回不足 K 条时，缺少的位置也会降低精确率。该指标依赖较完整的标准证据标注，
  否则未标注但有效的证据会被当作无关结果。
- `MRR@K`：第一条正确证据排名的倒数，再做宏平均。
- `AllRequirementsHit@K`：仅用于带 `evidence_requirements` 的样本，表示是否完整满足
  全部强制证据要求。
- `duplicate_rate@K`：返回结果中，同文档且规范化文本完全相同的重复 chunk 比例；空结果
  记为 `null`，避免把没有返回任何内容误算成零重复。
- 若样本使用 `evidence_requirements`，Hit 要求全部要求满足，MRR 使用完成全部要求时的排名。
- 指标默认在不超过本次 `top_k` 的 `1/3/5/10` 上计算，并始终包含实际 `top_k`；例如
  `--top-k 5` 产生 `@1/@3/@5`，`--top-k 10` 产生 `@1/@3/@5/@10`。
- `empty_rate`、`answerable_empty_rate`、`unanswerable_empty_rate`：对应阶段的空结果比例，不能直接视为拒答比例。

汇总结果还包含 `error_rate`、`average_latency_ms`、`latency_p50_ms` 和
`latency_p95_ms`。延迟统计只使用成功完成的查询；失败查询通过 `error_count/error_rate`
单独体现。建议始终结合 `by_question_type`、`by_source_modality` 和 `by_split` 查看切片，
不要只比较整体均值。

拒答统计以显式的 `sufficient=false` 为依据，不根据 chunks 是否为空推断：

- `answerable_rejection_rate`：被拒绝的可回答题数 / 有有效判定的可回答题数。
- `unanswerable_rejection_rate`：被拒绝的不可回答题数 / 有有效判定的不可回答题数。
- `rejection_rate`：全部有效判定中的拒绝比例。
- `fallback_rate`：有效判定中因模型调用或响应校验失败而进入 fallback 的比例。
  fail-open 和 fail-closed 的实际放行或拒绝结果仍计入对应拒答率。
- 同时记录各项计数及分母 `decision_query_count`、`answerable_decision_count`、
  `unanswerable_decision_count`。检查关闭、缺少判定或检索报错的样本分别计入
  `disabled_query_count`、`unavailable_query_count`、`error_count`，不进入拒答率分母。

可回答题拒答率不等同于误拒率：原文有答案，不代表召回片段足以回答。
若要统计误拒率，还需要标注检查前的证据是否充分。

逐题结果保存 `retrieval.chunks_before_sufficiency`、
`retrieval.retrieved_count_before_sufficiency`、`retrieval.empty_before_sufficiency`，
以及 `evidence_matches_before_sufficiency`，以便复查被拒绝的证据及其原始排名。
`retrieval.total_before_evidence_sufficiency` 保存检查前的候选总数，
与 Top K 片段数不同。检查前后的片段均通过同一序列化逻辑去除 embedding 向量。

兼容性说明：原有逐题 `metrics`、`evidence_matches`、`retrieval.chunks` 及汇总的
平铺指标继续表示检查后的结果。旧汇总字段 `unanswerable_rejection_rate` 保留原先
按空结果计算的口径；新的拒答分析应读取 `gate_metrics.unanswerable_rejection_rate`。
旧 `evidence_sufficiency_fallback_rate` 也保留原口径，新分析读取 `gate_metrics.fallback_rate`。
`average_latency_ms` 仍是完整检索流程的平均耗时，包含充分性检查，并非某一阶段的耗时。

历史结果没有检查前快照时，检查前指标及片段标为 `null`，计入 `unavailable_query_count`，
不补成零分，也不从最终输出推测。已记录的空列表则是实际空检索，会计入评分。
检索评分的分母由 `evaluated_query_count`、`answerable_query_count` 等字段给出；
不可回答题不计入 Hit、Recall 和 MRR，没有可用分母时指标为 `null`。
历史文件不会被自动修改；重新评测时使用新的 `--run-name` 即可生成三组指标。

## 端到端 RAG 评估

`run_e2e_eval.py` 在不读写聊天数据库、Redis、会话标题和推荐问题的情况下，执行与
线上相同的知识库检索、证据充分性检查、回答提示词和流式模型响应。该 runner 固定为
`knowledge_base_only`，不会让 Web 搜索补答知识库不可回答题；Agent 工具路由应使用
独立数据集评估。

```bash
docker compose exec LS_api \
  python /app/evals/run_e2e_eval.py \
  --index-name <用户ID> \
  --split test \
  --run-name e2e_test_v2
```

调参时只跑 `--split dev`；参数冻结后再跑 `--split test`，最后单独跑
`--split challenge`。不要根据 test/challenge 失败案例继续调参后仍沿用同一版数据集，
否则需要提升 `dataset_version` 并重新留出未见测试题。

默认使用 `CHAT_MODEL` 生成答案，使用 `E2E_JUDGE_MODEL` 判断答案正确性、
Faithfulness，以及逐条 claim 与引用证据的蕴含关系。可通过 `--answer-model`、
`--judge-model` 临时覆盖；使用 `--no-judge` 时只运行确定性的引用编号、充分性和
拒答评估，不产生答案正确性、Faithfulness 或引用支持度分数。Faithfulness 是一次
独立 judge 调用，因此每个实际回答的样本会比旧版多一次模型请求。

每条结果包含：

- `retrieval_metrics`、`gate_metrics`、`post_gate_metrics`：与检索 runner 相同的三阶段指标。
- `answer_metrics`：答案正确性和要求覆盖度；仅对实际回答的可回答题评分。
- `faithfulness_metrics`：把答案拆成事实主张，并使用全部最终检索证据判断是否直接
  支持；不依赖回答有没有引用标记。包含 `faithfulness_score`、事实主张数、支持及
  不支持主张数、逐条判定和 `unsupported_claims`。拒答、没有事实主张、judge
  关闭或失败时分数为 `null`，不会被补成满分。默认 Top 5 会完整送入 judge；为限制
  极端 `top-k` 的上下文长度，最多评估前 20 个最终证据片段，并用
  `evidence_truncated` 标记片段列表是否被截断。
- `citation_metrics`：编号合法率，以及 judge 启用时的引用完整率、精确率和召回率。
- `refusal_metrics`：期望与实际的 `answer/refuse` 行为、危险作答和过度拒答。
- `sufficiency_metrics`：以本次闸门前 Top K 是否覆盖全部标准证据作为期望标签，统计
  危险放行和错误拒绝。未提取视觉事实的图表和结构未绑定的表格默认不算充分；如有
  人工复核标签，可用样本字段 `expected_evidence_sufficient` 显式覆盖。

Faithfulness 的逐题公式为：

```text
faithfulness_score = 被全部最终证据支持的事实主张数 / 事实主张总数
```

汇总中的 `faithfulness` 是逐题分数的宏平均，`faithfulness_micro` 是把所有已评估
事实主张合并后的微平均；同时记录参与评分的样本数、评分覆盖率、judge 失败数和
支持/不支持主张总数。该指标衡量回答是否忠于检索证据，不判断检索证据本身是否
真实，也不替代答案正确性。
端到端结果结构版本已更新为 `1.1`。

证据不充分时，runner 与线上知识库工具共用确定性拒答策略，不调用回答模型。
数据集可选字段 `expected_behavior` 可显式取 `answer` 或 `refuse`；省略时从
`answerable` 推导。v2 对该字段进行强校验；现有 `guodian_power_eval_v1.jsonl`
仍可以通过 `--dataset` 显式指定运行，无需迁移。

结果写入：

- `results/<run-name>.jsonl`：逐题回答、引用解析、检索 trace 和各项指标。
- `results/<run-name>_summary.json`：整体与题型切片汇总。
