# AgentRAG GPU 推理服务

这个目录是完全独立的部署单元，只包含 Embedding 和 Reranker 推理服务。复制
`gpu_inference/` 到 GPU 主机即可，不需要上传 AgentRAG 后端、前端、数据库或
Elasticsearch 代码。

## 接口

- `GET /health`
- `POST /v1/embeddings`
- `POST /v1/rerank`

默认模型：

- `BAAI/bge-small-zh-v1.5`，归一化后的 512 维向量
- `BAAI/bge-reranker-v2-m3`，默认返回原始 logits

## Docker 部署（推荐）

GPU 主机需要安装 NVIDIA 驱动、Docker、Docker Compose 和 NVIDIA Container
Toolkit。先准备两个完整的 Hugging Face 模型目录，然后执行：

```bash
cd gpu_inference
cp .env.example .env
```

修改 `.env` 中的模型绝对路径和 API Key：

```env
EMBEDDING_MODEL_HOST_PATH=/data/models/bge-small-zh-v1.5
RERANKER_MODEL_HOST_PATH=/data/models/bge-reranker-v2-m3
INFERENCE_SERVER_API_KEY=replace_with_a_long_random_key
```

启动服务：

```bash
docker compose up -d --build
docker compose logs -f model-server
```

验证服务：

```bash
curl -H "Authorization: Bearer replace_with_a_long_random_key" \
  http://127.0.0.1:9000/health
```

`PRELOAD_MODELS=true` 会在容器启动时加载两个模型。模型目录错误、CUDA 不可用或
显存不足时，容器会直接启动失败，而不是等到第一条业务请求才报错。

## 不使用 Docker

先根据 GPU 驱动和 CUDA 版本安装匹配的 PyTorch，再安装其余依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --host 0.0.0.0 --port 9000 --workers 1
```

不要增加 Uvicorn worker 数量；每个 worker 都会各自加载一份模型并占用一份显存。
单进程内部默认最多同时执行一个推理请求，可以通过
`INFERENCE_MAX_CONCURRENCY` 调整，但增加并发前应进行显存压力测试。

## AgentRAG 主机配置

在 AgentRAG 的 `backend/.env` 中配置：

```env
EMBEDDING_BACKEND=remote
EMBEDDING_REMOTE_BASE_URL=http://GPU主机地址:9000
EMBEDDING_REMOTE_API_KEY=replace_with_a_long_random_key

RERANKER_BACKEND=remote
RERANKER_REMOTE_BASE_URL=http://GPU主机地址:9000
RERANKER_REMOTE_API_KEY=replace_with_a_long_random_key
RERANKER_REMOTE_SCORE_MODE=raw_logits
```

然后重新创建 API 容器：

```bash
cd backend
docker compose up -d --no-deps --force-recreate LS_api
```

Embedding 和 Reranker 可以独立设置为 `local` 或 `remote`。远程 Embedding 必须
与已有索引使用相同模型版本、Pooling、归一化方式和 512 维输出；否则需要重新入库。

生产环境应通过内网、VPN 或 TLS 反向代理访问该服务，不建议将 9000 端口直接暴露
到公网。API Key 为空时鉴权会被关闭，因此生产环境必须设置非空 Key。

## 测试

```bash
python -m pip install -r requirements-test.txt
pytest
```

