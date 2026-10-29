# RAG 评估

本目录保存 Agent RAG 系统的离线评估数据和评估脚本。JSONL 是评测集的主格式，每行是一个完整、独立的 JSON 对象。

当前边界：`run_retrieval_eval.py` 只执行给定查询的召回与排序；`run_e2e_eval.py` 默认运行完整 Agent，包括理解、初始规划、工具调用、证据核验、缺口补查和回答。纯检索不执行意图识别或充分性判定。检索结果结构版本为 `3.2`，端到端结果结构版本为 `3.1`；不应与旧版内含规划的检索成绩直接比较。端到端从 3.0 起将每次搜索排名、完整核验证据、实际回答上下文分别评分，不再把累计顺序当作 Top K 排名；3.1 起充分性评分及拒答归因共用独立的评测侧标签。

## 当前数据集

多报告评测集现拆为两个独立 JSONL 文件，基于 `backend/pdf/国电电力.pdf` 和
`backend/pdf/润本股份.pdf` 制作，`dataset_version` 为 `power-reports-v2.1-split-50`，
结构版本仍为 `2.0`。检索和端到端评测入口均默认使用 dev 文件：

| 文件 | 用途 | 题数 | 可回答 / 不可回答 |
| --- | --- | --- | --- |
| `data/power_reports_eval_v2_dev.jsonl` | 国电电力问题，用于提示词、检索参数和充分性阈值调试 | 50 | 42 / 8 |
| `data/power_reports_eval_v2_test.jsonl` | 润本股份及跨报告比较问题，用于参数冻结后的回归 | 50 | 39 / 11 |

| 题型 | dev | test |
| --- | --- | --- |
| `fact` | 22 | 16 |
| `parallel_multi_evidence` | 10 | 14 |
| `sequential_multi_hop` | 4 | 4 |
| `paraphrase` | 6 | 5 |
| `unanswerable` | 8 | 11 |

原混合集 `power_reports_eval_v2.jsonl` 已由上述两个文件替代。原 dev 8 题保留在
dev，原 test 16 题和 challenge 12 题保留在 test；另新增 dev 42 题、test 22 题。
原 challenge 样本 ID 增加 `test-` 前缀，并保留 `metadata.original_id`、
`metadata.original_split=challenge` 和 `challenge` 标签，便于追溯历史压力测试题。
两个原本依赖上下文的 dev 问题补齐了公司或年份，其余原题保留。

每个文件只包含同名的 `metadata.split`，100 个问题与样本 ID 均不重复。
这是问题级划分：test 的跨报告题与 dev 共用部分国电电力事实，并非文档或事实完全
隔离的未见测试集；与历史国电电力专项集也存在主题交集。不要将已用于调参的问题
当作未见测试题。题目覆盖正文、表格、单位换算、跨片段计算及桥接检索；不可回答题
覆盖时间错位、近邻指标替换、缺失拆分、实时数据和反事实敏感性。

文件可直接编辑；改题时应同步维护答案、证据、claim、划分及版本号。分别校验：

```bash
cd backend
python evals/run_retrieval_eval.py --dataset evals/data/power_reports_eval_v2_dev.jsonl --validate-only
python evals/run_retrieval_eval.py --dataset evals/data/power_reports_eval_v2_test.jsonl --validate-only
```

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
- `evidence_requirements`：多证据题的强约束；只有所有要求都命中时，Hit 才算成功。每个 `alternatives` 分支内的 atom 必须全部满足，分支之间任选其一；atom 类型支持 `text`、`table_row` 和 `figure`，其中图表须提供 `locator` 或 `text`。
- `metadata`：难度、标签、`dev/test/challenge` 数据划分、源文档日期及不可回答原因等信息。

v2 校验还会检查证据 ID 唯一性、claim 到 evidence 的引用关系、期望行为、检索模式，
并要求不可回答题显式标注 `expected_evidence_sufficient=false`。

评测脚本通过统一入口 `retrieve_evidence()` 获取 `RetrievalResult`，在评测输出处使用 `retrieval_record()` 生成 JSON。结果保留 `chunk_id`、各召回分支的原始排名与分数、RRF 分数、最终语义重排分数、候选总数及查询诊断。当前评测集尚未把标准证据映射到 ES 中的稳定 `chunk_id`，因此评测脚本先校验文档名，再对返回片段和 `relevant_evidence[].text` 做规范化文本匹配。完成第一轮基线并确认切块稳定后，可以把标准 chunk ID 补入证据对象，进一步减少模糊匹配。

当前检索入口只执行给定查询，包括混合召回、重排、过滤、固定别名扩展及表格上下文补全。它不理解需求、不生成子查询、不补查，也不判定答案是否充分。设置 `RETRIEVAL_ALIAS_EXPANSION_ENABLED=false` 可关闭固定别名扩展。原模型改写和扩展开关已移除。

端到端问答由 Agent 一次生成需求和初始计划，统一执行并行、多跳和跨工具任务，根据累计证据缺口补查。知识库限定评测只注册知识库工具，统一调用 `execute_agent_turn`，无需选择执行模式。模块边界见 [核心架构说明](../app/service/core/README.md#agent-统一理解与规划)。

新纯检索结果与旧版包含规划、补查、充分性过滤的结果不可直接比较。检索结果表示命中而非允许回答，端到端评测才统计最终拒答和回答质量。

对于 `evidence_type=figure` 的样本，`text` 是人工核对图片后写下的视觉事实。当前检索评分优先匹配 `locator`（图标题），因此命中表示找到了目标图的文字定位信息，不代表片段已保留视觉事实或足以回答问题。图表样本应与纯文本、表格样本分开统计。

## 数据质量要求

1. 所有可回答题目的参考答案必须被 `relevant_evidence` 完整支持。
2. 不使用仅靠常识即可回答的问题。
3. 不把当前 RAG 系统生成的回答当成参考答案。
4. 不可回答题必须保持 `relevant_evidence` 为空，系统应明确表示文档未提供答案。
5. 修改样本内容时同步更新 `dataset_version`，并重新运行 JSONL 结构校验。

## 运行检索评估

纯检索评测调用生产搜索入口，仅使用 Embedding 和重排模型，不调用生成式查询处理或充分性核验模型。它会保存 Top K chunk、`chunk_id`、各分支排名和原始分数、RRF 分数、语义重排分数、充分性诊断、位置和耗时，但不会把体积较大的 embedding 向量写入结果文件。
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

校验独立的 test 评测集：

```bash
docker compose exec LS_api \
  python /app/evals/run_retrieval_eval.py \
  --dataset /app/evals/data/power_reports_eval_v2_test.jsonl \
  --validate-only \
  --split test
```

运行 v2 前，目标用户知识库必须同时包含 `国电电力.pdf` 和 `润本股份.pdf`，且文件名
保持不变。使用 `--dataset` 切换 dev/test 文件；`--split` 只在所选文件内部筛选，
不会自动切换文件。对仍包含多个划分的国电专项集，`--split` 可重复或用逗号分隔。

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

再运行完整的 Top 10 基线，输出 `@10` 指标：

```bash
docker compose exec LS_api \
  python /app/evals/run_retrieval_eval.py \
  --index-name <用户ID> \
  --top-k 10 \
  --candidate-size 100 \
  --rerank-candidate-size 20 \
  --rrf-k 60 \
  --vector-weight 0.6 \
  --final-reranker-weight 0.7 \
  --final-rrf-k 10 \
  --dataset /app/evals/data/power_reports_eval_v2_test.jsonl \
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

历史 2.x 结果将指标分为三组；当前纯检索 3.2 沿用这些字段：

以下三组字段兼容历史记录。当前纯检索仅以 `retrieval_metrics` 衡量效果，`gate_metrics` 没有实际判定，`post_gate_metrics` 不代表回答放行。端到端 3.0 的累计集合改用下文的独立指标。

| 分组 | 数据来源 | 用途 |
| --- | --- | --- |
| `retrieval_metrics` | 充分性检查前的 Top K chunks | 衡量检索是否找到了标准证据 |
| `gate_metrics` | 充分性判定及样本的 `answerable` 标签 | 衡量拒答、放行及检查失败情况 |
| `post_gate_metrics` | 充分性检查后的 Top K chunks | 衡量最终交给回答阶段的证据 |

逐题 JSONL 中，这三组位于根节点。汇总文件中的整体指标路径分别为
`metrics.overall.retrieval_metrics`、`metrics.overall.gate_metrics`、
`metrics.overall.post_gate_metrics`；`by_question_type` 和 `by_source_modality`
中的每个切片也包含同样三组。需要分别比较调参集、常规测试集和挑战集时，
请分别运行评测。

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
- 指标默认只在实际 `top_k` 上计算；例如 `--top-k 5` 只产生 `@5`，避免 summary
  重复展示多个 cutoff。需要排查排序问题时，逐题结果中的 `first_relevant_rank` 会保留
  首次命中的具体位置，内部调用也仍可显式请求多个 cutoff。
- `empty_rate`、`answerable_empty_rate`、`unanswerable_empty_rate`：对应阶段的空结果比例，不能直接视为拒答比例。

汇总结果还包含 `error_rate`、`average_latency_ms`、`latency_p50_ms` 和
`latency_p95_ms`、`latency_p99_ms`。延迟统计只使用成功完成的查询；失败查询通过 `error_count/error_rate`
单独体现。建议结合 `by_question_type` 和 `by_source_modality` 查看切片，
不要只比较整体均值。

逐题 `retrieval.stage_latency_ms` 记录阶段耗时。当前纯检索主要记录召回、重排与总耗时；下列其余字段用于读取旧结果，不能把缺省值解释为执行了对应阶段。历史并发节点的工作时间会重叠，不能将各字段相加作为总延迟。

- `query_transform`：意图识别、查询改写和扩展。
- `initial_retrieval`：关键词/向量召回及 RRF 融合；多查询或多跳时为各次召回之和。
- `rerank`：语义重排、约束排序和最终融合；多查询或多跳时包括统一重排。
- `multi_hop`：仅递进式检索有值，记录桥接值抽取、第二跳构造和多跳编排开销，
  不重复计算各跳召回与重排。
- `gate`：首次证据充分性判定。
- `repair`：缺失目标补查及再次充分性检查。
- `other`：扣除已记录工作时间后的非负剩余估计；并发时不能当成独立墙钟阶段。
- `total`：完整检索流程耗时。

汇总中的 `stage_latency_ms.<stage>` 分别提供 `sample_count`、`average_ms`、
`p50_ms` 和 `p95_ms`。未执行的多跳阶段记为 `null`，因此只在实际多跳样本上聚合。

每个整体或切片汇总还包含 `non_perfect_sample_ids`。它按原指标的层级和名称列出未达到
理想值的样本，例如
`non_perfect_sample_ids.retrieval_metrics.hit_at_5`。Hit、Recall、Precision、MRR、
AllRequirementsHit、拒答召回率等指标以 `1` 为理想值；`error_rate`、`duplicate_rate`、
`fallback_rate`、可回答题拒绝率等指标以 `0` 为理想值。指标为 `null` 的样本不作为零分
列入；原有数值字段保持不变，已有汇总读取逻辑无需迁移。

拒答统计以显式的 `sufficient=false` 为依据，不根据 chunks 是否为空推断：

- `answerable_rejection_rate`：被拒绝的可回答题数 / 有有效判定的可回答题数。
- `unanswerable_rejection_rate`：被拒绝的不可回答题数 / 有有效判定的不可回答题数。
- `rejection_rate`：全部有效判定中的拒绝比例。
- `correct_rejection_after_insufficient_evidence_rate`：门控前标准证据不完整时，
  被正确拒绝的比例；计数由 `correct_rejection_after_insufficient_evidence_count`
  给出。这部分损失归因于上游检索，而不是门控误杀。
- `false_rejection_with_sufficient_evidence_rate`：门控前标准证据已经完整时，
  仍被错误拒绝的比例；计数由 `false_rejection_with_sufficient_evidence_count`
  给出。这才是门控自身的误拒率。
- `fallback_rate`：有效判定中因模型调用或响应校验失败而进入 fallback 的比例。
  fail-open 和 fail-closed 的实际放行或拒绝结果仍计入对应拒答率。
- 同时记录各项计数及分母 `decision_query_count`、`answerable_decision_count`、
  `unanswerable_decision_count`。归因分母分别由
  `evidence_sufficient_answerable_decision_count` 和
  `evidence_insufficient_answerable_decision_count` 给出；没有门控前快照的历史样本
  计入 `evidence_attribution_unavailable_answerable_decision_count`。被拒可回答题中，
  可归因和无法归因的数量分别记录在
  `answerable_rejected_evidence_attribution_count` 和
  `answerable_rejected_evidence_attribution_unavailable_count`。检查关闭、缺少判定或
  检索报错的样本分别计入
  `disabled_query_count`、`unavailable_query_count`、`error_count`，不进入拒答率分母。

可回答题拒答率不等同于误拒率：原文有答案，不代表召回片段足以回答。历史版本使用
门控前 `Recall@K=1` 做近似归因；当前使用统一的 `evidence_sufficiency_reference`，
不再由 recall 重新推断。该标签独立于系统自己的放行决定。
逐题 `gate_metrics` 记录
`pre_gate_evidence_sufficient` 和 `rejection_attribution`；后者可能为
`correct_rejection_after_insufficient_evidence`、
`false_rejection_with_sufficient_evidence` 或 `correct_unanswerable_rejection`。

`evidence_sufficiency_reference` 保存 `expected_sufficient`（true/false/null）、
`status`（sufficient/insufficient/unknown）、`source`、`reason` 及规则版本。
逐题 `gate_metrics`、`sufficiency_metrics` 和拒答归因汇总读取同一份已保存标签。
规则如下：

- 显式 `expected_evidence_sufficient` 优先，标记来源为 `annotation`；这是评测方的覆盖标签，
  必须确保适用于本次证据，不能把可回答题的静态 answerable 标签当作它使用。
- 没有核验前快照时为 unknown；有快照的不可回答题为 insufficient。
- 对可回答题，检查每个标准证据或 requirement 的全部匹配候选。文本匹配之外，
  表格要求 `table_binding_kwd=bound`，图表要求 `visual_status_kwd=extracted`。
  明确 unbound 或未提取的来源不可用；状态缺失或无法识别时为 unknown。
- requirement 的一个分支必须全部 atom 可用；只要有一个完整可用分支即可。
  最早的坏片段不会遮住后面的好片段，也不能将不同分支的残缺证据拼成一个完整方案。
- 所有要求均可用才充分；存在明确无法满足的要求则不充分；其余为 unknown。
  `candidate_matches` 记录全部命中来源及可用性原因，`alternative_matches` 记录全部分支，
  原有最早命中排名继续用于检索指标，不用于替代可用性判断。

unknown 不进入充分性正确率、危险放行率和误拒率分母，也不获得正确/错误拒答归因。
端到端汇总的 `sufficiency_metrics.expected_unknown_count` 和 `actual_unavailable_count`
分别统计期望标签未知与缺少实际布尔判定，二者可能重叠；`decision_count` 是两者都
可用的评分分母。门控仍保留实际拒答行为计数，无法归因的可回答题单独计数。
旧记录缺少统一标签时，归因汇总记为未知，不从旧 recall 猜测。历史已存分数保留其
原始口径，重新评测才会生成新标签。这一规则仍依赖现有文本匹配，不代表数值语义
核验；模糊匹配对关键数字的容错问题另行处理。

逐题结果保存 `retrieval.chunks_before_sufficiency`、
`retrieval.retrieved_count_before_sufficiency`、`retrieval.empty_before_sufficiency`，
以及 `evidence_matches_before_sufficiency`，以便复查被拒绝的证据及其位置。
当前类型化检索结果只有一份 `evidence`；纯检索出口从它生成
`chunks` 和 `chunks_before_sufficiency`。端到端问答的快照来自 Agent
实际累计并交给 `verify_evidence` 入口的完整资料，按累计顺序去重，不再截成前 K 条；
这是核验流程的来源集合，不是内部语义模型经过投影和长度限制后的请求正文。
最终回答上下文按实际引用顺序单独写入 `chunks`，其 `content_with_weight` 是回答模型
实际看到的 block 正文，原始来源保存在 `source_content_with_weight` 和 `documents`。
拒答时保留核验快照，回答集合为空；没有执行核验时快照为 `null`。
此快照不包含检索器内部已被重排或 Top-K 筛除的全部候选。
纯检索或历史结果中的 `retrieval.total_before_evidence_sufficiency` 保存检查前的候选总数，
与 Top K 片段数不同。Agent 的每次调用候选数在 `retrieval.calls` 中，不能用最后一次
调用代表全部搜索。所有评分片段均通过同一序列化逻辑去除 embedding 向量。

2.2 不再写出与 `post_gate_metrics` 完全重复的逐题 `metrics`，也移除了汇总中重复平铺的
Hit、Recall、MRR、空结果率、旧拒答率和旧 fallback 率。新结果统一读取上述三组指标；
`evidence_matches`、`retrieval.chunks` 等诊断字段不变。汇总最外层的 `metrics` 仍作为
`overall`、`by_question_type` 和 `by_source_modality` 的容器保留。
聚合器仍可读取带旧 `metrics` 字段的 2.1 历史逐题结果。
2.3 在保持上述字段兼容的基础上增加门控拒绝归因指标。
2.4 增加逐题阶段耗时及其整体和切片聚合。
`average_latency_ms` 是当前执行边界的平均耗时。3.0 的纯检索不包含规划和充分性检查，历史 2.x 结果可能包含。

历史结果没有检查前快照时，检查前指标及片段标为 `null`，计入 `unavailable_query_count`，
不补成零分，也不从最终输出推测。已记录的空列表则是实际空检索，会计入评分。
检索评分的分母由 `evaluated_query_count`、`answerable_query_count` 等字段给出；
不可回答题不计入 Hit、Recall 和 MRR，没有可用分母时指标为 `null`。
历史文件不会被自动修改；重新评测时使用新的 `--run-name` 即可生成三组指标。

## 端到端 RAG 评估

评测分为两个阶段：先完成所有题目的 Agent 问答，再统一执行忠实性、引用支持度和
答案质量评分。问答中的证据充分性核验仍正常执行；评分使用 `--judge-model` 配置。
每完成一道题，答案、证据及执行结果即写入 `<run-name>_answers.jsonl`，其中评分状态
为 `pending`（执行异常的题为 `skipped`）。该文件保留未评分快照，不被后续评分覆盖。
全部评分结束后生成原有 JSONL、汇总和 HTML 报告；最终记录的 `scoring_status` 区分
`completed`、`failed` 和 `skipped`，评分异常不会覆盖原问答结果。`--no-judge` 仍不调用
评分模型。`--fail-fast` 遇到抛出的异常仍提前停止，此前完成的问答快照保留。

`run_e2e_eval.py` 在不读写聊天数据库、Redis、会话标题和推荐问题的情况下，执行与
线上相同的知识库检索、证据充分性检查、回答提示词和流式模型响应。该 runner 固定为
`knowledge_base_only`，不会让 Web 搜索补答知识库不可回答题。统一通过真实 Agent 和 KB 工具执行，支持 `--planner-model`。已移除 `--execution-mode` 参数，旧命令需删去该选项及其值；报告中的 `execution_mode` 固定记录为 `agent`。Web/KB 混合工具路由仍需独立数据集评估。

```bash
docker compose exec LS_api \
  python /app/evals/run_e2e_eval.py \
  --index-name <用户ID> \
  --dataset /app/evals/data/power_reports_eval_v2_test.jsonl \
  --split test \
  --run-name e2e_test_v2
```

调参时使用默认 dev 文件；参数冻结后通过 `--dataset` 显式选择 test 文件。
多报告集不再有独立的 `challenge` 划分，原压力题已包含在 test 的 50 题中。
不要根据 test 失败案例继续调参后仍将其视为未见评测；需要提升 `dataset_version`
并重新留出未见测试题。

默认使用 `CHAT_MODEL` 生成答案，使用 `E2E_JUDGE_MODEL` 判断答案正确性、
Faithfulness，以及逐条 claim 与引用证据的蕴含关系。可通过 `--answer-model`、
`--judge-model` 临时覆盖；使用 `--no-judge` 时只运行确定性的引用编号、充分性和
拒答评估，不产生答案正确性、Faithfulness 或引用支持度分数。Faithfulness 是一次
独立 judge 调用，因此每个实际回答的样本会比旧版多一次模型请求。

端到端流程先运行忠实度评判，引用评判沿用其逐条 `requires_evidence` 分类：非事实陈述跳过引用评判，事实陈述只检查其引用片段是否支持内容。分类不可用时引用语义评分也记为不可用，不重新独立分类；引用编号有效性仍可计算。原始结果中的 `citation_metrics.classification_source=faithfulness` 标记这一口径，历史分数需注意分类口径差异。

每条结果包含：

- `retrieval.calls`：每次真实检索的查询、K、耗时、错误、独立排序片段及 `retrieval_metrics`。
- `verification_evidence_metrics`：完整核验来源集合的覆盖指标，无 Top-K 截断。
- `answer_evidence_metrics`：实际回答上下文的覆盖指标，只评估采用的片段和实际正文。
- `evidence_sufficiency_reference`：根据完整核验集合匹配结果与模态状态生成的独立评测标签。
- `gate_metrics`：拒答及归因；归因读取统一评测标签，不从 recall 推断。
- 根节点 `retrieval_metrics`、`post_gate_metrics` 在新 Agent 记录中为 `null`，不再伪造累计排名；旧文件仍能按原口径读取。
- `answer_metrics`：答案正确性和要求覆盖度；仅对实际回答的可回答题评分。
- `faithfulness_metrics`：把答案拆成事实主张，并使用全部最终检索证据判断是否直接
  支持；不依赖回答有没有引用标记。包含 `faithfulness_score`、事实主张数、支持及
  不支持主张数、逐条判定和 `unsupported_claims`。拒答、没有事实主张、judge
  关闭或失败时分数为 `null`，不会被补成满分。默认 Top 5 会完整送入 judge；为限制
  极端 `top-k` 的上下文长度，最多评估前 20 个最终证据片段，并用
  `evidence_truncated` 标记片段列表是否被截断。
- `citation_metrics`：编号合法率，以及 judge 启用时的引用完整率、精确率和召回率。
- `refusal_metrics`：期望与实际的 `answer/refuse/clarify` 行为、危险作答和过度拒答。
- `sufficiency_metrics`：将实际判定与 `evidence_sufficiency_reference` 比较，统计
  危险放行和错误拒绝；标签未知时 `correct=null`，不参与正确率。同步保留标签来源与原因。

Faithfulness 的逐题公式为：

```text
faithfulness_score = 被全部最终证据支持的事实主张数 / 事实主张总数
```

汇总中的 `faithfulness` 是逐题分数的宏平均，`faithfulness_micro` 是把所有已评估
事实主张合并后的微平均；同时记录参与评分的样本数、评分覆盖率、judge 失败数和
支持/不支持主张总数。该指标衡量回答是否忠于检索证据，不判断检索证据本身是否
真实，也不替代答案正确性。
汇总中的 `non_perfect_sample_ids` 同样按指标层级列出未达到理想值的样本；例如
`non_perfect_sample_ids.faithfulness`、
`non_perfect_sample_ids.refusal_metrics.refusal_recall` 和
`non_perfect_sample_ids.sufficiency_metrics.accuracy`。危险作答、过度拒答、危险放行和
错误拒绝等反向指标以 `0` 为理想值。
端到端结果结构版本为 `3.1`。旧结果中的累计 Top K 已经截断，或缺少完整匹配候选及模态可用性信息；需要重新运行，不能与新分数直接横比。

端到端汇总按题型切片，以下路径也存在于 `metrics.by_question_type.<题型>`：

- `metrics.overall.verification_evidence_metrics.recall`：完整核验来源集合的宏平均覆盖率。
- `metrics.overall.answer_evidence_metrics.recall`：实际回答上下文的宏平均覆盖率。
- 两个集合指标组还包含 `hit`、`precision`、`all_requirements_hit`、`duplicate_rate`，
  不带 `@K`，也不提供 MRR。precision 的分母为该集合实际片段数，空集合时为 `null`。
  已知空集合的可回答题 recall 为 0；缺少快照时为未知。各指标同时报告评分分母，
  并在组内 `non_perfect_sample_ids` 列出未达标样本。
- `metrics.overall.retrieval_call_metrics.by_top_k.5`：所有 K=5 的真实检索调用的排名统计。
  这里的 `query_count/evaluated_query_count/error_count` 按调用计数，不按用户问题计数；
  不同 K 分开汇总。每次调用都以整道题的 gold 作诊断（`reference_scope=whole_question`），
  子查询本来只覆盖部分要求，不能把这个分数当作子查询成功率或整体任务完成率。
- `metrics.overall.gate_metrics`：放行、拒绝、检查失败及拒绝原因归因统计。

集合质量只对成功且数据可用的可回答题做宏平均；不可回答题、执行异常和缺失数据
不补成零分。`retrieval_call_metrics` 区分没有调用（已记录的空列表）与没有调用快照
（未知）。每次检索仍沿用原有 Hit/MRR 规则。纯检索 runner 的 @K 口径不变。
这些统计不依赖 judge，使用 `--no-judge` 时仍然输出；历史结果文件不会被自动改写。

证据不充分时，runner 与线上知识库工具共用确定性拒答策略，不调用回答模型。
数据集可选字段 `expected_behavior` 可显式取 `answer`、`refuse` 或 `clarify`；`conversation_history` 可提供 user/assistant 历史消息。省略行为时从
`answerable` 推导。v2 对该字段进行强校验；现有 `guodian_power_eval_v1.jsonl`
仍可以通过 `--dataset` 显式指定运行，无需迁移。

结果写入：

- `results/<run-name>.jsonl`：逐题回答、引用解析、检索 trace 和各项指标。
- `results/<run-name>_summary.json`：整体与题型切片汇总。
- `results/<run-name>.html`：可离线打开的阅读报告，包含问题、参考答案、实际答案、主要评分和失败原因，支持搜索和筛选需关注题目。评分失败显示为未评分，不计作零分；详细检索记录仍保存在 JSONL 中。

已有端到端结果无需重新调用模型，可直接补生成 HTML：

```bash
cd backend
python evals/html_report.py evals/results/agent_test_smoke_20261008.jsonl
```

默认读取同名 `_summary.json`，输出同名 `.html`。已有 HTML 需显式添加 `--overwrite` 才会替换。


### 评测终端日志与轮转

通过命令行启动 `run_e2e_eval.py` 或 `run_retrieval_eval.py` 时，自动将完整的
stdout/stderr 双写到终端和固定文件 `backend/logs/evaluation.log`。无需新增参数，
不受 `--run-name`、`--output-dir` 或 `--overwrite` 影响；目录自动创建，后续运行追加。
Docker 中路径为 `/app/logs/evaluation.log`，Compose 已配置 `./logs:/app/logs` 挂载。
已有容器需要按更新后的 Compose 配置重新创建，挂载才会生效。

使用标准库 `RotatingFileHandler` 的备份管理机制，按实际字节数限制文件大小：

- 每个文件最大 `50 * 1024 * 1024` 字节（50 MiB）；文件已满后，下一次写入先轮转。
- 保留 `evaluation.log.1`、`.2`、`.3`，其中 `.1` 最新；下一次轮转自动删除最旧备份。
- 超长单次输出会分段写入，四个日志文件总计最多约 200 MiB。
- `evaluation.log.lock` 是用于跨进程协调的小型锁文件，不是日志备份；运行时不要删除。

每次运行有唯一 Run ID，文件中的 START/END 分隔行记录入口、命令参数、UTC 时间、
耗时和退出状态。`SUCCESS` 表示脚本退出码为 0，`FAILED` 表示非零退出或启动异常；
这不是评测分数，Agent 内部转为拒答的故障仍按原有评测逻辑处理。评测异常堆栈会
随 stderr 保存；逐题异常即使没有开启 `--fail-fast` 也会打印完整堆栈。
轮转可能把一次运行分散到不同备份，排查时应搜索当前文件及所有备份中的 Run ID。

端到端评测还按样本记录各阶段的 `status=start/completed/error` 和耗时，格式为
`EVAL sample_id=dev-gd-002 stage=citation_support status=start`。
阶段包括 `agent_answer`（含回答流读取）、`retrieval_metrics`、`faithfulness`、
`citation_support` 和 `answer_quality`。`completed` 表示阶段函数已返回；评分是否可用仍以
结果中的 `judge_source`、`judge_error` 为准，不能把单条 HTTP 200 日志当作整题完成。

回答流遇到结束事件、读取异常或前端中途断开时会显式关闭响应，归还连接；只关闭响应流，
不关闭后续评分仍需复用的客户端。避免把未读完的响应推迟到垃圾回收，在下一次请求持有
连接池锁时清理旧流而发生重入死锁。已经卡死的旧评测进程不会因文件更新自动恢复，需要
停止旧进程后重新启动评测，无需重新上传文件或重建索引。

实现采用一个轻量父进程监督原评测脚本，在应用依赖导入之前建立捕获。父进程通过
管道实时转发两个输出流，不修改现有 `print()`、应用 logger 或全局输出流，不注册
额外的应用 logging handler。因此独立 handler、模块导入日志、`os.write`、二进制
stdout 和继承这些输出流的子进程也会被记录。子进程使用 Python 无缓冲输出。
文件写入和轮转使用线程锁及 POSIX 文件锁；多个独立评测不会竞争重命名备份。
同时运行的评测内容仍可能交错，若需仅凭开始/结束边界区分完整内容，应串行运行。

覆盖范围与限制：

- 支持当前 Linux/Docker、macOS 环境；不支持原生 Windows（使用 `fcntl`）。
- 命令行运行自动捕获；仅导入模块或直接调用其 `main()` 不会重定向宿主程序输出。
- stdout/stderr 各自的字节顺序保留，跨两个流无法保证严格的时间先后顺序。
- 直接写其他文件、独立终端、浏览器控制台或远端服务的内容不在捕获范围内。
- Ctrl+C/SIGTERM 会转发给评测进程，已输出内容会排空并记录 FAILED；SIGKILL、断电
  或存储故障无法保证 END 标记及尚在管道中的数据保存。
- 原始字节原样保存，轮转边界可能切开一条长日志或 UTF-8 字符；需要时按旧到新拼接
  备份再读取。进程内部被吞掉且没有打印的异常细节，不会由输出捕获自动补全。

日志测试不需要数据库或模型调用：

```bash
python3 -m unittest discover -s backend/evals -p 'test_evaluation_logging.py' -v
```

### 查询规划与 Agent 回归

先用 `python /app/evals/run_e2e_eval.py --validate-only` 检查端到端样本结构，不需要索引或真实模型调用。

```bash
docker compose exec LS_api python /app/evals/run_e2e_eval.py \
  --dataset /app/evals/data/power_reports_eval_v2_dev.jsonl \
  --index-name <用户ID> --split dev --run-name agent_dev_v3
```

逐题 `agent_execution` 记录任务、工具查询、依赖绑定和停止原因。`query_metrics` 保留历史规划字段供旧记录读取；新 Agent 不再生成旧 RAG 的路由与补查报告，缺失字段不能当成成功率。补查统计优先读取 `agent_execution`：`steps > 1` 表示实际执行过补查（初始执行不计入），只生成计划但未执行不算；补查后 `stop_reason == evidence_ready` 计为补齐证据成功，与最终回答或 judge 是否成功无关。`repair_count` 按样本计数而非轮数，`repair_success_rate` 的分母为执行过补查的样本数；`repair_unavailable_count` 及分组内的 `unavailable_count` 标记缺失执行数据的样本。没有 Agent 执行记录时兼容旧 `query_repair`。请求 runtime 记录规划、证据核验、桥接及工具搜索预算；最终回答流和 judge 成本另行统计。样本数、失败率和各切片应与成本、P95/P99 一起审阅。

### 统一证据核验回归

规划只输出通用目标、限定条件、实体和工具任务，不再生成 `financial`。每轮检索后对累计原文进行一次模型核验；
补查新证据时重新核验，同版本报告直接复用，网络故障仍受有限重试与请求预算约束。

数值问题检查输入、公司、期间、单位、口径和计算关系是否明确，最终回答模型直接计算，不要求资料列出最终结果。
测试覆盖原文及带数字原因保留、未知指标、表格、计算输入不足后的补查、核验故障停止、引用和缓存隔离。
`test_unified_evidence.py` 使用固定模型响应，不联网；这些测试不代表实际模型语义和计算准确率。

新记录不再输出财务需求快照、`financial_fact_bundle`、逐事实/关系状态或服务端计算结果。
历史字段可继续读取为旧数据，但不再汇总旧财务专项指标，也不能把已停用的检查统计成通过。
评测重点比较充分性误放行与误拒、最终答案正确性与完整性、计算正确性、引用、成本和延迟，并区分服务故障与资料缺失。
