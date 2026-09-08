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
```

Embedding 使用本地 BGE；DashScope Key 仍用于聊天模型和 `qwen3-rerank`。
`BGE_MODEL_HOST_PATH` 相对于本目录的 `docker-compose.yml`。


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

### 服务说明

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
