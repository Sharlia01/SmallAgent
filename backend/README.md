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
cd swxy-p1
```

2. **.env 配置文件**
```bash
DASHSCOPE_API_KEY="your-api-key"
BGE_MODEL_HOST_PATH=../../models/bge-small-zh-v1.5
RAG_QUERY_INTENT_ENABLED=true
QUERY_REWRITE_ENABLED=true
QUERY_EXPANSION_ENABLED=true
RAG_EVIDENCE_SUFFICIENCY_ENABLED=true
```

Embedding 使用本地 `bge-small-zh-v1.5`，重排使用 DashScope `qwen3.7-text-rerank`；DashScope Key 同时用于 RAG 查询意图识别、查询规范化与扩展及聊天模型。
`BGE_MODEL_HOST_PATH` 相对于本目录的 `docker-compose.yml`。

RAG 检索入口会先识别查询类型。明确的表号、图号、章节或页码查询直接检索；口语、简称和上下文指代查询先规范化，再生成一条包含专业字段和同义词的关键词扩展查询。扩展结果必须通过实体、时间、数字、单位、否定词和定位信息保留校验，否则自动回退到规范化查询或原查询。检索器分别执行原问题关键词、扩展查询关键词和原问题向量召回，在 Python 层通过第一阶段加权 RRF 合并候选，再只使用原问题进行语义重排；最终以语义重排名次 `0.7`、第一阶段检索 RRF 名次 `0.3` 做第二阶段加权 RRF。未产生有效扩展时自动去掉重复关键词分支。

重排后的最终 Top 片段会再经过证据充分性检查：只有当所有时间范围、指标、限定条件和子问题都有直接证据时才保留结果，否则返回空结果并记录 `evidence_sufficiency`诊断。设置 `RAG_EVIDENCE_SUFFICIENCY_ENABLED=false` 可关闭该检查。调用异常默认保留原结果；如需异常时也拒绝返回，设置 `RAG_EVIDENCE_SUFFICIENCY_FAIL_OPEN=false`。

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

### 在线重排模型

检索器通过 DashScope `qwen3.7-text-rerank` 对“原问题、候选片段”逐对打分。
只需配置 `DASHSCOPE_API_KEY`，无需下载额外重排权重，也无需重新生成现有
embedding 或重新上传文档。

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
