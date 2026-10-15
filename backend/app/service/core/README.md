# 核心模块目录

目录按职责划分。文件名描述模块实际做什么，避免仅按 `query_`、`evidence_` 等旧前缀归类。

| 目录 | 职责 | 主要文件 |
| --- | --- | --- |
| `query/` | 共享请求契约、指代消解、语义规则与保真校验 | `contract.py`、`context.py`、`preservation.py`、`validation.py` |
| `retrieval/` | 执行指定查询、召回、排序、父表上下文扩展 | `pipeline.py`、`fusion.py`、`table_context.py` |
| `entities/` | 实体注册表、文档与片段的实体归属标注 | `attribution.py` |
| `evidence/` | 证据适配、元数据、核验、充分性评估 | `adapter.py`、`verification.py`、`sufficiency.py`、`bindings.py` |
| `finance/` | 财务指标语义、事实提取与核验、确定性计算 | `metrics.py`、`facts.py`、`rules.py`、`model_extraction.py`、`calculations.py` |
| `ingestion/` | 解析入库流程、表格和图片证据构建 | `pipeline.py`、`tables.py`、`figures.py` |
| `answering/` | 回答策略、回答上下文、生成与流式输出 | `answer_context.py`、`turn.py`、`chat.py`、`streaming.py` |
| `runtime/` | 查询运行上下文、预算、调用封装与诊断 | `context.py`、`execution.py`、`model_calls.py`、`diagnostics.py`、`clients.py` |
| `persistence/` | 父表对象与图片证据的文件读写 | `evidence_store.py` |
| `integrations/` | RAGFlow 助手、会话等外部系统接口 | `ragflow_assistant.py`、`ragflow_session.py` |
| `resources/` | 内置配置、数据根目录和词表缓存的路径定位 | `paths.py` |
| `conf/` | 实体注册表、查询与指标配置 | `entity_registry.json`、`query_domain.json` |
| `rag/`、`deepdoc/` | 现有检索基础组件和文档解析组件 | 保留各自内部结构及资源目录 |
| `storage/` | 运行时生成的证据图片与父表数据 | 保留已有数据位置 |

纯领域数据结构位于 `app/domain/`，例如 `facts.py`、`evidence.py`、`tables.py`。离线评测位于 `app/evaluation/`，执行入口仍在 `backend/evals/`，不和线上业务模块混放。

## 主要入口和依赖约定

- 文档入库从 `ingestion.pipeline` 进入；切片构建与存储读写分别交给对应模块。
- Agent 问答从 `agent.orchestrator` 进入；直接检索从 `retrieval.pipeline` 进入，不触发理解、规划或核验。
- `evidence` 组织通用证据核验，财务规则位于 `finance`；回答上下文由 `answering.answer_context` 构建。
- `finance.rules` 提供确定性提取规则，`finance.model_extraction` 处理模型候选，`finance.facts` 组织事实核验。规则模块不反向依赖核验流程。
- `runtime` 不依赖具体查询、财务或证据业务。它包含预算、超时和执行控制，所以不能整体归入 `dbg`；诊断相关代码集中在 `runtime.diagnostics`。
- 查询理解、模型校验、检索和证据核验失败时写普通日志。运行上下文只保留预算与缓存，不再积累逐步事件或返回 `query_runtime.events`。计划节点的最终结果仍用于汇总执行结果，不能当作过程事件历史。
- 检索读取父表使用 `persistence.evidence_store`，不通过入库入口加载解析器与嵌入模型。
- 使用完整模块路径导入；包的 `__init__.py` 保持轻量，不通过集中重导出隐式加载全部模块。旧顶层模块路径已统一替换，不保留同名兼容副本。

## Agent 统一理解与规划

主流程：指代消解 → 理解需求并生成初始计划 → 执行工具 → 合并证据并核验 → 按缺口重规划或生成回答。

| 模块 | 职责 |
| --- | --- |
| `agent/planner.py` | 一次模型调用生成 `QueryUnderstanding` 和 `AgentPlan`；后续只更新计划 |
| `agent/understanding.py` | 解码并校验目标、约束和财务需求，不单独调用模型 |
| `agent/financial_requirements.py` | 财务需求协议解码、实体/期间规范化、口径和计算引用校验 |
| `agent/financial_coverage.py` | 理解输出的财务防遗漏检查，只报错，不追加另一套目标 |
| `agent/financial_bindings.py` | 用执行证据绑定未知主体或期间，生成独立核验快照 |
| `agent/plans.py` | 任务协议、目标覆盖、约束保真、工具名称和依赖图校验 |
| `agent/executor.py` | 有界并发执行工具，填充有证据来源的变量，继承已确认文档范围 |
| `agent/loop.py` | 累积证据、统一核验、缺口反馈、预算和停止条件 |
| `agent/orchestrator.py` | 准备会话请求，并使用循环内核验报告构建回答和引用 |
| `retrieval/pipeline.py` | 一次搜索请求只执行一次混合检索，保留部分证据，不自主补查 |
| `evidence/verification.py` | 共享核验服务，保留财务事实、来源冲突、视觉证据和报告版本检查 |
| `evidence/bindings.py` | 从证据中抽取依赖变量并绑定原文来源，不猜测桥接值 |

`QueryRequest` 保留原文、历史指代绑定和消解后的问题。`QueryUnderstanding` 的目标描述可以规范化口语，但不能擅自指定会计口径；`source_spans` 保留输入原文依据。

`AgentTask` 是计划任务，包含工具、目标编号、查询或模板、依赖和输出变量。`RetrievalTask` 是可执行搜索请求，包含具体查询、目标编号及服务端绑定的文档范围。用户身份不由模型提供。

目标与任务不是一一对应关系。比较两个年份可以拆成两次查询，重规划可以换表达或工具，但不能删除目标或扩大明确来源范围。跨工具依赖也由同一执行器管理；未知公司必须先从证据确认，才能代入网页查询。同一报告范围只能从知识库证据继承。

`RetrievalResult` 和 `ToolResult` 只返回来源、排名、执行状态和诊断，不包含意图或充分性报告。部分证据始终返回 Agent。循环按完整目标核验累计来源，核验通过才生成回答；同一来源和需求版本的报告可以复用。规划、核验或预算失败不再暗中回退到另一套 RAG 分析。

检索与工具结果统一使用 `evidence` 保存交给 Agent 的完整资料，不再另设 `candidates` 副本；`evidence` 不表示已经核验通过。外部旧格式的 `candidate_chunks` 在适配入口合并，保留原有候选优先顺序、来源、检索轨迹和版本冲突。核验结论由 `EvidenceAssessment` 表达，最终回答内容与引用由 `AnswerContext` 选择。评测继续记录 `chunks_before_sufficiency`：纯检索导出本次返回资料，端到端问答导出 Agent 累计资料，即使拒答也保留。该快照不是重排或 Top-K 筛选前的全部召回片段。

配置：`AGENT_MODEL` 控制理解与规划，`AGENT_MAX_STEPS` 限制执行轮数，`AGENT_CONTEXT_MODEL` 可单独指定历史指代消解模型。`RETRIEVAL_ALIAS_EXPANSION_ENABLED` 仅控制固定等价词表。原 `RAG_QUERY_INTENT_*`、`QUERY_REWRITE_*`、`QUERY_EXPANSION_*` 以及内部并行、递进、图规划和补查开关已移除。现有 `RAG_QUERY_*` 预算与并发配置继续用于整个请求，所有工具共享额度。

独立问答和端到端评测通过 `answering.turn` 使用相同 Agent，并仅注册知识库工具。`run_retrieval_eval.py` 衡量给定查询的召回和排序，不执行需求理解或证据充分性判定；历史带语义规划的检索成绩不能直接与其比较。

## 财务需求的单一来源

`QueryUnderstanding.financial` 是财务需求的权威来源。首次 Agent 模型响应同时生成目标、财务事实要求、计算要求和工具计划；校验失败时沿用一次修正机会，不在核验阶段再解析问句。

- `financial=null` 表示没有财务数值要求；空对象无效。不支持的指标、计算或未解决歧义必须写入 `unresolved`，不能被当成无需财务核验。
- `facts` 每项关联 `goal_id`，声明公司名称、指标、期间、实际/预测、报表口径、业务范围、币种、度量和原文依据。公司 ID 和规范期间由服务端生成，模型不能提供核验状态。
- `calculations` 使用固定操作和 `operand_ids` 引用事实要求。`change_percent` 的输入顺序是当前值、基期值；`decline_percent` 是基期值、当前值。计算模块不再根据年份出现顺序或公司名称猜测方向。
- `entity_ref`、`period_ref` 声明待取证条件。任务 `output_slot` 可以直接提供同名值；补查使用新的输出变量时，通过 `binding_key` 指向固定引用，保持理解结果不变。
- `document_subject` 是保留主体引用，只用于用户明确指向报告或附件的需求。绑定依赖可靠的文档主体标注，不能仅凭文件名、正文出现某家公司就确定查询主体。

`agent.financial_bindings` 根据实际证据生成 `QueryContext.resolved_financial`，同时保留绑定出处。原始理解不变，核验版本包含固定需求、已绑定需求、来源证明、目标范围和证据版本；绑定变化后不能复用旧报告。

`assess_financial_facts(requirements, evidence)` 只抽取证据事实、逐要求匹配、检查冲突并调用确定性计算。计算输入由已经通过核验的 `requirement_id` 选择，保留原始数值和来源；缺少操作数、口径不兼容或除零时返回缺口。通用语义核验继续负责原因解释、视觉证据等内容。

`FinancialFactAssessment.applicable` 直接由事实要求或未解决需求是否存在推导，不另存 `verification_required` 标记。合法计算必须引用已声明的事实要求；仅有歧义或不支持的需求时，也必须保留财务核验缺口。历史报告恢复时忽略旧标记，不根据是否提取到事实或计算出结果判断适用性。

原 `build_fact_requirements`、`_part_requirements`、从检索结果隐式推断用户主体的入口，以及充分性模块中的问句计算回退均已移除。`finance/metrics.py` 保留证据抽取也使用的指标词表和度量规则；需求判别与防遗漏规则位于 Agent。

评测输出同时记录 `query_understanding.financial`、当前 `financial_requirements` 和 `financial_bindings`。事实协议版本为 `financial-facts-v5`，请求策略版本为 `query-v7`，旧核验报告不能作为新请求的已通过报告复用。

## 财务关系的混合核验

规则未匹配不等于证据不存在。混合核验按固定 `FinancialRequirements` 先抽取候选，再核验公司—指标—期间—数值—口径的关系，不重新识别用户意图。

| 模块 | 职责 |
| --- | --- |
| `finance/facts.py` | 核验编排、结果汇总、只投影已确认事实 |
| `finance/rules.py` | 清晰关系快速确认，保留主体或期间未确认的数值候选 |
| `finance/matching.py` | 要求匹配、可比性、歧义和冲突判定 |
| `finance/model_sources.py` | 构建正文及可靠原表来源，校验逐字引文、字符区间和完整数值 |
| `finance/model_extraction.py` | 按固定要求提出候选，处理口径依据、两阶段缓存和执行故障 |
| `finance/relationship_verification.py` | 独立核验字段关系，保留模型判断与来源证明 |
| `finance/model_policy.py` | 模式、模型、超时及来源/候选数量上限 |
| `finance/calculations.py` | 只用已核验操作数计算，保留计算缺口类型 |

候选的 `field_evidence` 为主体、指标、期间、数值、单位和已声明口径分别提供来源 ID 与字符区间。一个候选可以使用同一证据片段中的多段依据，包括表头、原始单元格、题注和已有的表格上下文；不支持跨片段拼接事实。数值必须对应完整原始数字，模型不能将 `180` 截成 `80`。模型重建但没有独立来源依据的表格不能自行证明可靠，需要 Agent 补查原表。已有可信文档主体证明可作为关系上下文，但主体名称与数值共现本身不构成证明。

清晰正文配对和可靠行列映射可直接使用规则；未知关系、范围歧义、冲突及依赖文档主体与章节的表格归属按需进入模型路径，不再依赖 `COMPLEX` 关键词。模型输出必须再次通过来源检查；对同一数值位置的明确正文配对不能任意改写。语义补全可以替代同位置的未确认候选，旧记录仍保留审计状态。近似数保留“约”，不能作为精确计算操作数。

`FinancialFactAssessment.requirement_results` 区分 `supported`、`missing`、`ambiguous`、`conflict`、`unsupported`；`relationship_results` 保存逐候选核验记录。`verification_status=unavailable` 单独表示服务、格式或预算故障，不作为“资料缺失”。Agent 收到缺失后补查，遇到歧义补上下文或澄清，遇到冲突查版本与口径；遇到核验故障沿用停止机制，不重复检索。未确认事实不进入回答或计算，上述状态不会自动开启部分回答。

默认模式为 `hybrid`；`RAG_FINANCIAL_VERIFICATION_MODE=rules` 可明确关闭模型补充。旧 `RAG_FINANCIAL_FACT_MODEL_ENABLED` 在未配置新模式时仍生效，因此旧部署显式关闭模型的设置需要迁移。配置样例已采用 hybrid，每批最多 3 个来源、12 个候选，模型调用仍计入请求总预算。达到上下文上限会报告歧义，不冒充全面检索无结果；补查新增来源优先于已审阅来源。

候选缓存与关系核验缓存分别绑定固定要求、原文/结构版本、实体目录和模型策略。失败的关系结论不缓存；缓存某个指标的事实不意味着该片段其他指标已抽取完毕。配置变化会使旧核验报告失效。

## 路径兼容

移动业务文件不会改变既有索引、数据格式或默认存储位置，不需要因此重新解析文档。

- 内置配置统一通过 `resources.paths.config_file()` 定位到 `core/conf/`，不再相对调用方的 `__file__` 计算。
- 数据根目录继续支持 `RAG_PROJECT_BASE`、`RAG_DEPLOY_BASE`；未配置时仍为 `service/core/`。已保存的 `storage/table_objects/...` 和 `storage/evidence/...` 相对引用保持有效。
- 内置 tiktoken 缓存归入 `rag/res/tiktoken/`，路径定位兼容旧部署根目录中的缓存文件。
- `rag` 和 `deepdoc` 的其他资源位置保持不变，仅同步公共路径工具的导入。
