# 核心模块目录

目录按职责划分。文件名描述模块实际做什么，避免仅按 `query_`、`evidence_` 等旧前缀归类。

| 目录 | 职责 | 主要文件 |
| --- | --- | --- |
| `query/` | 共享请求契约、指代消解、语义规则与保真校验 | `contract.py`、`context.py`、`preservation.py`、`validation.py` |
| `retrieval/` | 执行指定查询、召回、排序、父表上下文扩展 | `pipeline.py`、`fusion.py`、`table_context.py` |
| `entities/` | 查询侧实体名称、别名和证券代码的规范化 | `registry.py` |
| `evidence/` | 证据适配、元数据、核验、充分性评估 | `adapter.py`、`verification.py`、`sufficiency.py`、`bindings.py` |
| `ingestion/` | 解析入库流程、表格和图片证据构建 | `pipeline.py`、`tables.py`、`figures.py` |
| `answering/` | 回答策略、回答上下文、生成与流式输出 | `answer_context.py`、`turn.py`、`chat.py`、`streaming.py` |
| `runtime/` | 查询运行上下文、预算、调用封装与诊断 | `context.py`、`execution.py`、`model_calls.py`、`diagnostics.py`、`clients.py` |
| `persistence/` | 父表对象与图片证据的文件读写 | `evidence_store.py` |
| `integrations/` | RAGFlow 助手、会话等外部系统接口 | `ragflow_assistant.py`、`ragflow_session.py` |
| `resources/` | 内置配置、数据根目录和词表缓存的路径定位 | `paths.py` |
| `conf/` | 实体注册表、查询与指标配置 | `entity_registry.json`、`query_domain.json` |
| `rag/`、`deepdoc/` | 现有检索基础组件和文档解析组件 | 保留各自内部结构及资源目录 |
| `storage/` | 运行时生成的证据图片与父表数据 | 保留已有数据位置 |

纯领域数据结构位于 `app/domain/`，例如 `assessment.py`、`evidence.py`、`tables.py`。离线评测位于 `app/evaluation/`，执行入口仍在 `backend/evals/`，不和线上业务模块混放。

## 主要入口和依赖约定

- 文档入库从 `ingestion.pipeline` 进入；切片构建与存储读写分别交给对应模块。
- Agent 问答从 `agent.orchestrator` 进入；直接检索从 `retrieval.pipeline` 进入，不触发理解、规划或核验。
- `evidence` 组织统一证据核验；回答上下文由 `answering.answer_context` 构建。
- 回答上下文保留核验器已有的逐目标支持关系：通过稳定证据 ID 映射到筛选后的本轮引用编号，携带原目标文本和约束。该关系仅作资料导航，不能替代原问题、完整证据或片段内主体与期间的核对；没有逐目标映射时沿用原回答路径。
- `runtime` 不依赖具体查询、财务或证据业务。它包含预算、超时和执行控制，所以不能整体归入 `dbg`；诊断相关代码集中在 `runtime.diagnostics`。
- 查询理解、模型校验、检索和证据核验失败时写普通日志。运行上下文只保留预算与缓存，不再积累逐步事件或返回 `query_runtime.events`。计划节点的最终结果仍用于汇总执行结果，不能当作过程事件历史。
- 检索读取父表使用 `persistence.evidence_store`，不通过入库入口加载解析器与嵌入模型。
- 入库保存原文、文档名和表格结构，不推断文档、片段或表格行的公司归属。查询侧 `CompanyHints` 只用于排序和诊断；召回仍遵守租户、知识库、文档和可用状态范围，不附加公司标签过滤。主体关系由统一证据核验模型结合原文判断。
- 旧索引及父表中的实体归属标签在证据边界被忽略，不需要为此重建索引。原始公司列、证券代码、表头、单元格和行列位置继续保留。
- 使用完整模块路径导入；包的 `__init__.py` 保持轻量，不通过集中重导出隐式加载全部模块。旧顶层模块路径已统一替换，不保留同名兼容副本。

## Agent 统一理解与规划

主流程：指代消解 → 理解需求并生成初始计划 → 执行工具 → 合并证据并核验 → 按缺口重规划或生成回答。

| 模块 | 职责 |
| --- | --- |
| `agent/planner.py` | 一次模型调用生成 `QueryUnderstanding` 和 `AgentPlan`；后续只更新计划 |
| `agent/understanding.py` | 解码并校验目标、约束与主体来源，不单独调用模型 |
| `agent/plans.py` | 任务协议、目标覆盖、约束保真、工具名称和依赖图校验 |
| `agent/executor.py` | 有界并发执行工具，填充有证据来源的变量，继承已确认文档范围 |
| `agent/loop.py` | 累积证据、统一核验、缺口反馈、预算和停止条件 |
| `agent/orchestrator.py` | 准备会话请求，并使用循环内核验报告构建回答和引用 |
| `retrieval/pipeline.py` | 一次搜索请求只执行一次混合检索，保留部分证据，不自主补查 |
| `evidence/verification.py` | 统一核验服务，保留来源冲突、视觉证据和报告版本检查 |
| `evidence/bindings.py` | 从证据中抽取依赖变量并绑定原文来源，不猜测桥接值 |

`QueryRequest` 保留原文、历史指代绑定和消解后的问题。`QueryUnderstanding` 的目标描述可以规范化口语，但不能擅自指定会计口径；`source_spans` 保留输入原文依据。

`AgentTask` 是计划任务，包含工具、目标编号、查询或模板、依赖和输出变量。`RetrievalTask` 是可执行搜索请求，包含具体查询、目标编号及服务端绑定的文档范围。用户身份不由模型提供。

目标与任务不是一一对应关系。比较两个年份可以拆成两次查询，重规划可以换表达或工具，但不能删除目标或扩大明确来源范围。跨工具依赖也由同一执行器管理；未知公司必须先从证据确认，才能代入网页查询。同一报告范围只能从知识库证据继承。

`RetrievalResult` 和 `ToolResult` 只返回来源、排名、执行状态和诊断，不包含意图或充分性报告。部分证据始终返回 Agent。循环按完整目标核验累计来源，核验通过才生成回答；同一来源和需求版本的报告可以复用。规划、核验或预算失败不再暗中回退到另一套 RAG 分析。

补查核验支持按缺口缩小目标范围：首轮仍使用原有完整问题、全部目标和响应协议。
只有旧模型报告已完成且查询、旧证据、策略及模式均匹配时，才继承 supported 目标，
对 missing/conflict 目标使用独立的缺口提示，并保留已支持目标的引用原文。
模型必须返回 recheck_goal_ids；发现旧结论受新材料影响或出现冲突时恢复完整核验。
关联原文因输入预算未能送审时也恢复完整核验；服务故障不额外重复请求。
程序按证据 ID 合并新旧目标判定并重建支持索引，报告仍绑定完整查询及累计证据。
诊断记录本轮实际送审 ID、复用目标及其证据版本，区别于跨轮累计评估范围。

检索与工具结果统一使用 `evidence` 保存交给 Agent 的完整资料，不再另设 `candidates` 副本；`evidence` 不表示已经核验通过。外部旧格式的 `candidate_chunks` 在适配入口合并，保留原有候选优先顺序、来源、检索轨迹和版本冲突。核验结论由 `EvidenceAssessment` 表达，最终回答内容与引用由 `AnswerContext` 选择。评测继续记录 `chunks_before_sufficiency`：纯检索导出本次返回资料，端到端问答导出 Agent 累计资料，即使拒答也保留。该快照不是重排或 Top-K 筛选前的全部召回片段。

配置：`AGENT_MODEL` 控制理解与规划首轮模型（默认 `qwen3.7-flash`），失败后第二次使用 `AGENT_FALLBACK_MODEL`（默认 `qwen3.8-max`）。`AGENT_MAX_STEPS` 限制执行轮数，`AGENT_CONTEXT_MODEL` 与 `AGENT_CONTEXT_FALLBACK_MODEL` 可单独指定历史指代消解的首轮和兜底模型。每个阶段最多两次调用，成功或正常澄清不升级。`RETRIEVAL_ALIAS_EXPANSION_ENABLED` 仅控制固定等价词表。原 `RAG_QUERY_INTENT_*`、`QUERY_REWRITE_*`、`QUERY_EXPANSION_*` 以及内部并行、递进、图规划和补查开关已移除。现有 `RAG_QUERY_*` 预算与并发配置继续用于整个请求，所有工具共享额度。

独立问答和端到端评测通过 `answering.turn` 使用相同 Agent，并仅注册知识库工具。`run_retrieval_eval.py` 衡量给定查询的召回和排序，不执行需求理解或证据充分性判定；历史带语义规划的检索成绩不能直接与其比较。

补查由 replan 根据历史任务、核验缺口、缓存命中、检索与变量绑定状态、本轮新增证据和剩余轮数决定；没有具体可行的新取证动作时返回空任务。简短原因记录在 `agent_execution.replan_decisions`，不替代核验结论。执行前仅拦截整份均命中成功缓存的无依赖、无输出变量直接搜索（工具、实际查询、主体范围、文档范围和 top_k 一致）；不同范围、失败重试、依赖与表格恢复任务继续交给执行器处理。

## 统一证据核验与回答计算

所有问题共用 `QueryUnderstanding.goals`，期间、口径、单位、比较方向等保存在目标文本和 `constraints` 中。
规划不再生成 `financial`、事实要求、计算操作或操作数引用。财务专用领域对象、抽取、匹配和计算模块已移除。

每轮工具执行后，`verify_evidence` 对累计原始证据调用一次充分性模型；相同查询、来源、规则和模式的报告可以复用。
送审预算由 `RAG_EVIDENCE_SUFFICIENCY_CONTEXT_TOKENS` 配置（默认 16384），使用本地 cl100k_base 估算完整请求并预留输出及安全余量，不是 Qwen 精确 token 计数。预算内完整保留片段及表格结构，超限按目标文本、定位与检索线索轮流选取完整片段，不再平均截断正文。报告记录 `input_limited` 和 `omitted_chunk_ids`；送审子集不足且仍有遗漏材料时标记核验未完成，停止无效补查并提示缩小范围，不宣称全部资料缺失。当前不自动分批核验超大材料。
基础来源冲突、图表视觉可用性、响应格式、目标编号和引用范围检查保留。
网络故障仍可按既有配置重试一次；补查新来源后重新核验，不保证整个请求只调用一次模型。

核验器接收原问题、目标及有界原文，检查主体、指标、期间、单位、口径、冲突和全部目标覆盖。
计算问题只需确认输入与关系明确，不要求文档直接披露计算结果，也不提前执行计算。
核验成功后，回答模型直接使用所引用的完整原文选择公式、执行计算和解释结果，引用原始输入来源。
回答参考资料以 `[编号]` 加 JSON 组织，包含文档 ID、文档名、原文和可用的页码、章节或表格位置；表格另保留标题、表头、公司列、单位上下文及原始行，图表保留视觉提取状态。网页来源保留链接。
引用编号与前端来源使用同一份 `Citation` 映射；`AnswerBlock.text` 仍是原文，不把文档名或模型推断的主体写进原文。向量、排序分数、检索轨迹和旧归属标签不进入回答提示词。
全局引用和逐目标引用共同选择已经送审的支持片段。核验器须把确认主体、期间、单位和桥接关系依赖的上下文也列为支持来源，避免回答阶段只剩孤立数字。此流程不自动加载未召回的普通文本邻段或整份文档。
不再生成中间财务事实，也不按数字或财务词筛掉原因说明。

`EvidenceAssessment` 只保存通用充分性、缺口、目标覆盖、支持来源、故障及版本。
历史财务字段可在报告适配入口忽略，不恢复为当前核验权限；新报告不生成 `financial_fact_bundle` 或服务端计算结果。
缓存策略版本使用 `unified-evidence-v4`，不会复用旧财务流程或缺少上下文保留要求的报告。

未知主体的多步检索保留 `subject_slot` 和 `BridgeExtraction`。补查任务可通过 `binding_key` 关联既有主体目标，
后续模板消费该任务新的 `output_slot`；来源和文档范围仍由执行器确认，不依赖财务契约。

配置只需 `RAG_EVIDENCE_SUFFICIENCY_*`；旧 `RAG_FINANCIAL_*` 不再使用。规划、桥接和回答模型配置保持各自职责。
旧财务逐事实指标已停用，评测继续统计目标覆盖、充分性误放行/误拒、回答正确性、引用和运行故障。
单元测试使用固定模型响应验证流程，不代表真实模型对语义或计算的准确率。
