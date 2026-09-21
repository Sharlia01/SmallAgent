# 智能文档问答系统 (GSK POC)

一个基于RAG（检索增强生成）技术的智能文档问答系统，支持多种文档格式的解析和问答。

## 🚀 快速启动

### 环境要求

- Docker 和 Docker Compose
- 至少 4GB 可用内存
- 10GB 可用磁盘空间
- 本地 `bge-small-zh-v1.5` 模型目录

### 启动步骤

1. **克隆项目并进入目录**
```bash
cd LS-p1
```

2. **.env 配置文件**
```bash
DASHSCOPE_API_KEY="your-api-key"
EMBEDDING_BACKEND=local
BGE_MODEL_HOST_PATH=../../models/bge-small-zh-v1.5
RERANKER_BACKEND=local
RERANKER_MODEL_HOST_PATH=../../models/bge-reranker-v2-m3
RAG_QUERY_INTENT_ENABLED=true
QUERY_REWRITE_ENABLED=true
QUERY_EXPANSION_ENABLED=true
RAG_EVIDENCE_SUFFICIENCY_ENABLED=true
```

Embedding 和重排默认分别使用本地 `bge-small-zh-v1.5` 与
`bge-reranker-v2-m3`，也可以各自切换到远程推理服务；DashScope Key 用于 RAG
查询意图识别、查询规范化与扩展及聊天模型。本地模型宿主机路径都相对于本目录的
`docker-compose.yml`。

远程 GPU 模式配置示例：

```bash
EMBEDDING_BACKEND=remote
EMBEDDING_REMOTE_BASE_URL=https://gpu.example.com
EMBEDDING_REMOTE_API_KEY=replace-with-shared-key
RERANKER_BACKEND=remote
RERANKER_REMOTE_BASE_URL=https://gpu.example.com
RERANKER_REMOTE_API_KEY=replace-with-shared-key
# AgentRAG 的相似度阈值使用 0～1 概率；GPU 服务仍返回 raw logits。
RERANKER_SCORE_MODE=probability
RERANKER_REMOTE_SCORE_MODE=raw_logits
```

远程模型机只需复制仓库根目录的 `gpu_inference/`，不需要复制 Agent、数据库、
文档解析或前端代码：

```bash
cd gpu_inference
cp .env.example .env
# 修改模型路径与 INFERENCE_SERVER_API_KEY
docker compose up -d --build
```

`EMBEDDING_BACKEND` 与 `RERANKER_BACKEND` 可以独立设置。本地与远程
Embedding 必须使用相同模型版本、归一化方式和 512 维输出，否则必须重新入库。
完整 GPU 主机部署说明见 `gpu_inference/README.md`。

RAG 检索入口会先识别查询类型和检索模式。明确的表号、图号、章节或页码查询直接检索；口语、简称和上下文指代查询先规范化，再生成一条包含专业字段和同义词的关键词扩展查询。扩展结果必须通过实体、时间、数字、单位、否定词和定位信息保留校验，否则自动回退到规范化查询或原查询。普通问题使用 `single` 模式；只有后一个子问题依赖第一步证据中的术语或实体时才使用 `sequential` 模式。递进式检索最多执行两跳，桥接值必须逐字存在于第一跳来源 chunk；“这份研报/该报告”场景会把第二跳限定在同一文档。检索器内部仍分别执行原问题关键词、扩展查询关键词和原问题向量召回，在 Python 层通过第一阶段加权 RRF 合并候选，再只使用当前跳的问题进行语义重排；每跳以语义重排名次 `0.4`、第一阶段检索 RRF 名次 `0.6` 做第二阶段加权 RRF。

两跳结果去重合并后，额外使用用户原始问题统一重排一次，并重新检查原问题的限定条件。首页在容量允许时保留经过校验的桥接来源片段和第二跳的独立证据，其余位置按统一排名补齐；保留证据也按统一排名展示，不固定置顶。最终重排只决定顺序，不再次用原问题分数过滤辅助证据。统一重排失败时保留两跳结果，回退到各跳名次的等权 RRF，并记录失败来源。设置 `RAG_SEQUENTIAL_RETRIEVAL_ENABLED=false` 可关闭整个递进式检索并回退单跳。

重排后的最终 Top 片段会再经过证据充分性检查：只有当所有时间范围、指标、限定条件和子问题都有直接证据时才保留结果，否则返回空结果并记录 `evidence_sufficiency`诊断。设置 `RAG_EVIDENCE_SUFFICIENCY_ENABLED=false` 可关闭该检查。调用异常默认保守拒绝返回；只有在可用性优先于证据保证时才设置 `RAG_EVIDENCE_SUFFICIENCY_FAIL_OPEN=true`。

设置 `RAG_QUERY_INTENT_ENABLED=false` 可关闭意图判断并恢复全量改写策略，设置 `QUERY_REWRITE_ENABLED=false` 可完全关闭改写和扩展，设置 `QUERY_EXPANSION_ENABLED=false` 可只保留规范化改写。


3. **启动所有服务**
```bash
# 启动所有服务（首次启动会自动构建镜像）
docker compose up -d --build

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f LS_api
```

4. **等待服务完全启动**
```bash
# 检查服务健康状态
curl http://localhost:8000/docs
```

### 终端 CLI（无需启动前端）

CLI 通过 HTTP 复用 FastAPI 的现有接口，不会另写一套 RAG 逻辑。因此终端与网页
使用相同的用户隔离、会话历史、检索与重排、证据充分性检查、引用和拒答行为。

```bash
# 首次使用：注册账号，密码会隐藏输入
docker compose exec LS_api python rag_cli.py --username <用户名> --register

# 以后直接登录并进入交互问答
docker compose exec LS_api python rag_cli.py --username <用户名>
```

进入后可以直接输入问题，也可以使用：

- `/new`：创建新会话
- `/sessions`、`/use <session_id>`：列出和切换会话
- `/history`：查看当前会话历史
- `/upload <文件...>`：上传个人知识库文件；带空格的路径需要加引号
- `/quick <文件>`：上传只供当前会话使用的临时文档
- `/files`：查看知识库文件
- `/sources`：再次查看上一轮引用来源
- `/thinking on|off`：控制是否显示模型思考内容
- `/help`：查看完整帮助，`/exit` 退出

后端依赖已安装在本机时，也可在 `backend` 目录运行：

```bash
python app/rag_cli.py --username <用户名>
```

默认连接 `http://localhost:8000`。远程 API 可通过 `--base-url` 或环境变量
`RAG_API_URL` 指定。使用 `--question "问题"` 可只提一个问题并在回答后退出。
CLI 请求默认关闭回答后的推荐问题生成；会话标题直接取首次问题的前 30 个字符，
不再为标题额外调用模型。网页端仍默认生成推荐问题。

### 表格、图表与重新入库

表格解析不需要新增大模型：识别出的 HTML 按行生成证据，每行携带多级表头、
行标签、表标题及表内单位说明。`rowspan/colspan` 会先展开，空白值不会变成 0；
无法可靠绑定的表格保留原始 HTML，并标为 `table_binding_kwd=unbound`。
上游 OCR 或行列识别错误仍需核对原文，行绑定不会自动纠正识别错误。

PDF 图表默认只保存题注/OCR，并明确标注“未提取视觉事实”。仅有这些片段时，
视觉类问题不能把图标题命中当成充分证据。若需要在入库时调用支持图片的模型，
设置 `FIGURE_VISION_ENABLED=true` 和 `FIGURE_VISION_MODEL`；可单独配置
`FIGURE_VISION_API_KEY`、`FIGURE_VISION_BASE_URL`，留空时沿用 DashScope 配置。
模型和接口必须支持图片输入及 JSON 输出；不应填入只支持文本的聊天模型。
这不会修改 `CHAT_MODEL`。未配置、超时、空结果或响应格式错误时保留文字定位信息，
不生成猜测事实；成功结果标记为模型提取，未经人工核验。
该可选视觉模块当前处理 DeepDOC 识别出的 PDF 图表区域。

入库保留页码、坐标、证据类型、解析版本和裁剪图片路径。图片保存在
`app/service/core/storage/evidence`，供来源核查，不作为公开图片接口。
新解析只会影响后续入库；历史片段需要从原始文档重新解析：

```bash
# 默认只预检文件与所属用户索引，不写入。
docker compose exec LS_api python reindex_documents.py \
  --index-name 1 \
  --file /app/app/service/core/storage/file/1/报告.pdf

# 确认目标后执行；可重复传入 --file 批量重建已有文档。
docker compose exec LS_api python reindex_documents.py \
  --index-name 1 \
  --file /app/app/service/core/storage/file/1/报告.pdf --apply
```

新片段与本地 BGE 向量全部写入成功后，才按旧片段 ID 清理该文档的旧版本，
不会删除其他文件或修改 PostgreSQL 上传记录。解析/向量化失败保留旧片段；
ES 批量写入失败也不会清理旧片段，可重新执行恢复。替换不是原子切换，
执行期间或失败后可能暂时同时检索到新旧片段；同一文档应串行重建。
原始文件缺失时不能仅靠旧片段完成重新解析。

Compose 的 API 工作目录为 `/app/app`，与代码挂载一致。首次应用此配置后运行
`docker compose up -d --no-deps --force-recreate LS_api`，使服务加载新代码。

### 本地或远程重排模型

检索器通过 `BAAI/bge-reranker-v2-m3` 对“原问题、候选片段”逐对打分。
本地模式默认从 `/models/bge-reranker-v2-m3` 加载，并通过
`RERANKER_MODEL_HOST_PATH` 只读挂载；远程模式调用 `/v1/rerank`。切换重排部署位置
无需重新生成现有 embedding 或重新上传文档。远程服务如果返回 0～1 概率，需要将
`RERANKER_REMOTE_SCORE_MODE` 设置为 `probability`。`RERANKER_SCORE_MODE` 默认是
`probability`，确保检索器的 `similarity_threshold` 始终作用在 0～1 的统一量纲上；
当远程服务返回 raw logits 时，客户端会在过滤候选前自动转换。

### 服务列表

项目包含以下服务：
- **LS_api**: 主应用服务 (端口: 8000)
- **LS_pg**: PostgreSQL数据库
- **LS-es-01**: Elasticsearch搜索引擎
- **LS_redis**: Redis缓存

### 停止服务

```bash
# 停止所有服务
docker compose down

# 停止并删除数据卷（注意：这会删除所有数据）
docker compose down -v
```

## 🔧 开发调试

### 查看日志
```bash
# 查看所有服务日志
docker compose logs

# 查看特定服务日志
docker compose logs LS_api
docker compose logs LS_pg
docker compose logs LS-es-01
docker compose logs LS_redis

# 实时跟踪日志
docker compose logs -f LS_api
```

### 进入容器调试
```bash
# 进入主应用容器
docker compose exec LS_api bash

# 进入数据库容器
docker compose exec LS_pg psql -U postgres -d gsk
```

### 重新构建服务
```bash
# 重新构建并启动
docker compose up --build -d

# 仅重新构建特定服务
docker compose build LS_api
docker compose up -d LS_api
```

## 📋 常见问题

1. **端口被占用**: 确保8000端口未被其他程序占用
2. **内存不足**: Elasticsearch需要至少1GB内存，建议系统有4GB+可用内存
3. **首次启动慢**: 首次启动需要下载镜像和初始化数据，请耐心等待
4. **服务连接失败**: 等待所有服务完全启动后再测试API

## 🎯 访问地址

- API文档: http://localhost:8000/docs
