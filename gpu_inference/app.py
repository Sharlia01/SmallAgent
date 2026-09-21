"""Standalone HTTP service for AgentRAG embedding and reranker models."""

import os
import secrets
from contextlib import asynccontextmanager
from functools import lru_cache
from threading import BoundedSemaphore

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from providers import (
    EMBEDDING_DIMENSION,
    SUPPORTED_SCORE_MODES,
    EmbeddingEngine,
    RerankerEngine,
)


load_dotenv()

DEFAULT_EMBEDDING_MODEL_ID = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"


def positive_int_environment(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def boolean_environment(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true or false")


def score_mode_environment() -> str:
    score_mode = os.getenv(
        "INFERENCE_SERVER_RERANKER_SCORE_MODE",
        "raw_logits",
    ).strip().lower()
    if score_mode not in SUPPORTED_SCORE_MODES:
        supported = ", ".join(sorted(SUPPORTED_SCORE_MODES))
        raise RuntimeError(
            "INFERENCE_SERVER_RERANKER_SCORE_MODE must be one of: "
            f"{supported}"
        )
    return score_mode


def embedding_model_id() -> str:
    return os.getenv("EMBEDDING_MODEL_ID", DEFAULT_EMBEDDING_MODEL_ID)


def reranker_model_id() -> str:
    return os.getenv("RERANKER_MODEL_ID", DEFAULT_RERANKER_MODEL_ID)


@lru_cache(maxsize=1)
def get_embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine(
        model_path=os.getenv(
            "EMBEDDING_MODEL_PATH",
            "/models/bge-small-zh-v1.5",
        ),
        device=os.getenv("EMBEDDING_DEVICE", "cuda"),
        dimension=EMBEDDING_DIMENSION,
    )


@lru_cache(maxsize=1)
def get_reranker_engine() -> RerankerEngine:
    return RerankerEngine(
        model_path=os.getenv(
            "RERANKER_MODEL_PATH",
            "/models/bge-reranker-v2-m3",
        ),
        device=os.getenv("RERANKER_DEVICE", "cuda"),
        score_mode=score_mode_environment(),
    )


inference_slots = BoundedSemaphore(
    positive_int_environment("INFERENCE_MAX_CONCURRENCY", 1)
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if boolean_environment("PRELOAD_MODELS", False):
        get_embedding_engine()
        get_reranker_engine()
    yield


app = FastAPI(
    title="AgentRAG GPU inference service",
    version="1.0.0",
    lifespan=lifespan,
)


class EmbeddingRequest(BaseModel):
    model: str
    input: list[str] = Field(min_length=1)
    normalize: bool = True


class RerankerRequest(BaseModel):
    model: str
    query: str
    documents: list[str] = Field(min_length=1)
    top_n: int | None = Field(default=None, gt=0)
    return_documents: bool = False
    max_length: int | None = Field(default=None, gt=0)


def authorize(authorization: str | None = Header(default=None)) -> None:
    expected_key = os.getenv("INFERENCE_SERVER_API_KEY", "")
    if not expected_key:
        return
    supplied_key = ""
    if authorization and authorization.startswith("Bearer "):
        supplied_key = authorization.removeprefix("Bearer ")
    if not secrets.compare_digest(supplied_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid inference API key")


def validate_request_size(items: list[str]) -> None:
    maximum = positive_int_environment("INFERENCE_MAX_REQUEST_ITEMS", 128)
    if len(items) > maximum:
        raise HTTPException(
            status_code=413,
            detail=f"Request contains more than {maximum} items",
        )


@app.get("/health")
def health(_: None = Depends(authorize)):
    return {
        "status": "ok",
        "embedding": {
            "model": embedding_model_id(),
            "dimension": EMBEDDING_DIMENSION,
            "device": os.getenv("EMBEDDING_DEVICE", "cuda"),
            "loaded": get_embedding_engine.cache_info().currsize > 0,
        },
        "reranker": {
            "model": reranker_model_id(),
            "score_mode": score_mode_environment(),
            "device": os.getenv("RERANKER_DEVICE", "cuda"),
            "loaded": get_reranker_engine.cache_info().currsize > 0,
        },
    }


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest, _: None = Depends(authorize)):
    expected_model = embedding_model_id()
    if request.model != expected_model:
        raise HTTPException(
            status_code=400,
            detail=f"Requested embedding model {request.model!r} is not loaded",
        )
    if not request.normalize:
        raise HTTPException(
            status_code=400,
            detail="AgentRAG requires normalized embeddings",
        )
    validate_request_size(request.input)

    with inference_slots:
        vectors = get_embedding_engine().embed(
            request.input,
            batch_size=positive_int_environment("EMBEDDING_BATCH_SIZE", 32),
        )
    return {
        "model": expected_model,
        "dimension": EMBEDDING_DIMENSION,
        "data": [
            {"index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
    }

# 在FastAPI应用中定义了一个POST请求的路由`/v1/rerank`，用于处理重排序（reranking）请求。
# 该路由接收一个`RerankerRequest`对象作为请求体，并依赖于`authorize`函数进行身份验证。
@app.post("/v1/rerank")
def rerank(request: RerankerRequest, _: None = Depends(authorize)):
    expected_model = reranker_model_id()
    if request.model != expected_model:
        raise HTTPException(
            status_code=400,
            detail=f"Requested reranker model {request.model!r} is not loaded",
        )
    validate_request_size(request.documents)

    with inference_slots:
        scores = get_reranker_engine().rerank(
            request.query,
            request.documents,
            batch_size=positive_int_environment("RERANKER_BATCH_SIZE", 8),
            max_length=request.max_length
            or positive_int_environment("RERANKER_MAX_LENGTH", 1024),
        )
    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: (-scores[index], index),
    )
    if request.top_n is not None:
        ranked_indexes = ranked_indexes[:request.top_n]

    results = []
    for index in ranked_indexes:
        item = {
            "index": index,
            "relevance_score": scores[index],
        }
        if request.return_documents:
            item["document"] = request.documents[index]
        results.append(item)
    return {
        "model": expected_model,
        "score_mode": score_mode_environment(),
        "results": results,
    }

