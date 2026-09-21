import math
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Protocol

import httpx
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


DEFAULT_EMBEDDING_MODEL_PATH = "/models/bge-small-zh-v1.5"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DIMENSION = 512
EMBEDDING_VECTOR_FIELD = f"q_{EMBEDDING_DIMENSION}_vec"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_MODEL_PATH = "/models/bge-reranker-v2-m3"
DEFAULT_RERANKER_DEVICE = "cpu"
DEFAULT_RERANKER_BATCH_SIZE = 8
DEFAULT_RERANKER_MAX_LENGTH = 1024
DEFAULT_REMOTE_TIMEOUT_SECONDS = 30.0
DEFAULT_REMOTE_MAX_RETRIES = 2
DEFAULT_EMBEDDING_REMOTE_PATH = "/v1/embeddings"
DEFAULT_RERANKER_REMOTE_PATH = "/v1/rerank"
SUPPORTED_INFERENCE_BACKENDS = {"local", "remote"}
SUPPORTED_RERANKER_SCORE_MODES = {"raw_logits", "probability"}

def _positive_int_environment(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _non_negative_int_environment(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} must be a non-negative integer") from error
    if value < 0:
        raise RuntimeError(f"{name} must be a non-negative integer")
    return value


def _positive_float_environment(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default))
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} must be a positive number") from error
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{name} must be a positive number")
    return value


def _environment_choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        supported = ", ".join(sorted(choices))
        raise RuntimeError(f"{name} must be one of: {supported}")
    return value


def _required_environment(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be configured for remote inference")
    return value


def _remote_endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _remote_client(api_key: str, timeout_seconds: float) -> httpx.Client:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return httpx.Client(headers=headers, timeout=timeout_seconds)


def _post_remote_json(
    client: httpx.Client,
    endpoint: str,
    payload: dict,
    *,
    operation: str,
    max_retries: int,
    headers: dict[str, str] | None = None,
) -> dict:
    """POST one idempotent inference request with bounded transient retries."""
    retryable_statuses = {408, 429, 500, 502, 503, 504}
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.post(endpoint, json=payload, headers=headers)
            if response.status_code in retryable_statuses and attempt < max_retries:
                time.sleep(0.2 * (2 ** attempt))
                continue
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise RuntimeError("response JSON must be an object")
            return body
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError, RuntimeError) as error:
            last_error = error
            retryable = (
                isinstance(error, httpx.RequestError)
                or (
                    isinstance(error, httpx.HTTPStatusError)
                    and error.response.status_code in retryable_statuses
                )
            )
            if not retryable or attempt >= max_retries:
                break
            time.sleep(0.2 * (2 ** attempt))

    raise RuntimeError(
        f"Remote {operation} request failed: {type(last_error).__name__}: {last_error}"
    ) from last_error


def _convert_reranker_scores(
    scores: np.ndarray,
    *,
    source_mode: str,
    target_mode: str,
) -> np.ndarray:
    if source_mode == target_mode:
        return scores
    if source_mode == "raw_logits" and target_mode == "probability":
        # This stable form avoids overflow for large negative logits.
        positive = scores >= 0
        converted = np.empty_like(scores, dtype=float)
        converted[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
        exp_scores = np.exp(scores[~positive])
        converted[~positive] = exp_scores / (1.0 + exp_scores)
        return converted
    if source_mode == "probability" and target_mode == "raw_logits":
        if np.any((scores < 0) | (scores > 1)):
            raise RuntimeError(
                "Remote reranker declared probability scores outside [0, 1]"
            )
        epsilon = np.finfo(float).eps
        clipped = np.clip(scores, epsilon, 1.0 - epsilon)
        return np.log(clipped / (1.0 - clipped))
    raise RuntimeError(
        f"Unsupported reranker score conversion: {source_mode} -> {target_mode}"
    )


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the local Sentence Transformers model once per API process."""
    model_path = Path(
        os.getenv("EMBEDDING_MODEL_PATH", DEFAULT_EMBEDDING_MODEL_PATH)
    ).expanduser()
    if not model_path.is_dir():
        raise RuntimeError(
            "Local embedding model directory does not exist: "
            f"{model_path}. Check BGE_MODEL_HOST_PATH and the Docker volume."
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is required for local BGE embeddings"
        ) from error

    device = os.getenv("EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE)
    try:
        return SentenceTransformer(
            str(model_path),
            device=device,
            local_files_only=True,
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to load local embedding model from {model_path}: {error}"
        ) from error


@lru_cache(maxsize=1)
def get_reranker_model():
    """Load the local cross-encoder reranker once per API process."""
    model_path = Path(
        os.getenv("RERANKER_MODEL_PATH", DEFAULT_RERANKER_MODEL_PATH)
    ).expanduser()
    if not model_path.is_dir():
        raise RuntimeError(
            "Local reranker model directory does not exist: "
            f"{model_path}. Check RERANKER_MODEL_HOST_PATH and the Docker volume."
        )

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as error:
        raise RuntimeError(
            "transformers is required for local BGE reranking"
        ) from error

    device = os.getenv("RERANKER_DEVICE", DEFAULT_RERANKER_DEVICE)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        model.to(device)
        model.eval()
    except Exception as error:
        raise RuntimeError(
            f"Failed to load local reranker model from {model_path}: {error}"
        ) from error
    return tokenizer, model, device


class EmbeddingProvider(Protocol):
    backend: str

    def embed(
        self,
        texts: list[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        ...


class RerankerProvider(Protocol):
    backend: str

    def rerank(
        self,
        query: str,
        texts: list[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> list[float]:
        ...


class LocalEmbeddingProvider:
    backend = "local"

    def embed(
        self,
        texts: list[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        model = get_embedding_model()
        try:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as error:
            raise RuntimeError(
                f"Local BGE embedding inference failed: {error}"
            ) from error
        return np.asarray(embeddings, dtype=np.float32).tolist()


class RemoteEmbeddingProvider:
    backend = "remote"

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        api_key: str = "",
        endpoint_path: str = DEFAULT_EMBEDDING_REMOTE_PATH,
        timeout_seconds: float = DEFAULT_REMOTE_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_REMOTE_MAX_RETRIES,
        client: httpx.Client | None = None,
    ):
        self.endpoint = _remote_endpoint(base_url, endpoint_path)
        self.model_id = model_id
        self.max_retries = max_retries
        self.request_headers = (
            {"Authorization": f"Bearer {api_key}"}
            if api_key
            else None
        )
        self.client = client or _remote_client(api_key, timeout_seconds)

    @staticmethod
    def _parse_batch(body: dict, expected_count: int) -> list[list[float]]:
        declared_dimension = body.get("dimension")
        if declared_dimension is not None and declared_dimension != EMBEDDING_DIMENSION:
            raise RuntimeError(
                "Remote embedding returned dimension "
                f"{declared_dimension}, expected {EMBEDDING_DIMENSION}"
            )

        data = body.get("data")
        if data is None and isinstance(body.get("embeddings"), list):
            data = [
                {"index": index, "embedding": embedding}
                for index, embedding in enumerate(body["embeddings"])
            ]
        if not isinstance(data, list):
            raise RuntimeError("Remote embedding response must contain a data list")

        ordered: list[list[float] | None] = [None] * expected_count
        for fallback_index, item in enumerate(data):
            if not isinstance(item, dict):
                raise RuntimeError("Remote embedding data items must be objects")
            index = item.get("index", fallback_index)
            if not isinstance(index, int) or not 0 <= index < expected_count:
                raise RuntimeError("Remote embedding returned an invalid input index")
            if ordered[index] is not None:
                raise RuntimeError("Remote embedding returned a duplicate input index")
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise RuntimeError("Remote embedding item is missing its vector")
            ordered[index] = vector

        if any(vector is None for vector in ordered):
            raise RuntimeError(
                f"Remote embedding returned {len(data)} vectors for "
                f"{expected_count} inputs"
            )
        return [vector for vector in ordered if vector is not None]

    def embed(
        self,
        texts: list[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        embeddings = []
        for start in range(0, len(texts), batch_size):
            text_batch = texts[start:start + batch_size]
            body = _post_remote_json(
                self.client,
                self.endpoint,
                {
                    "model": self.model_id,
                    "input": text_batch,
                    "normalize": True,
                },
                operation="embedding",
                max_retries=self.max_retries,
                headers=self.request_headers,
            )
            returned_model = body.get("model")
            if returned_model is not None and returned_model != self.model_id:
                raise RuntimeError(
                    "Remote embedding served model "
                    f"{returned_model!r}, expected {self.model_id!r}"
                )
            embeddings.extend(self._parse_batch(body, len(text_batch)))
        return embeddings


class LocalRerankerProvider:
    backend = "local"

    def __init__(self, *, output_score_mode: str = "raw_logits"):
        self.output_score_mode = output_score_mode

    def rerank(
        self,
        query: str,
        texts: list[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> list[float]:
        tokenizer, model, device = get_reranker_model()
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("torch is required for local BGE reranking") from error

        score_batches = []
        try:
            for start in range(0, len(texts), batch_size):
                text_batch = texts[start:start + batch_size]
                pairs = [[query, text] for text in text_batch]
                inputs = tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length,
                )
                inputs = {
                    name: tensor.to(device)
                    for name, tensor in inputs.items()
                }
                with torch.no_grad():
                    logits = model(
                        **inputs,
                        return_dict=True,
                    ).logits.reshape(-1)
                score_batches.append(
                    logits.detach().float().cpu().numpy()
                )
        except Exception as error:
            raise RuntimeError(
                f"Local {RERANKER_MODEL} inference failed: {error}"
            ) from error

        scores = np.concatenate(score_batches).astype(float, copy=False)
        scores = _convert_reranker_scores(
            scores,
            source_mode="raw_logits",
            target_mode=self.output_score_mode,
        )
        return scores.tolist()


class RemoteRerankerProvider:
    backend = "remote"

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        api_key: str = "",
        endpoint_path: str = DEFAULT_RERANKER_REMOTE_PATH,
        timeout_seconds: float = DEFAULT_REMOTE_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_REMOTE_MAX_RETRIES,
        response_score_mode: str = "raw_logits",
        output_score_mode: str = "raw_logits",
        client: httpx.Client | None = None,
    ):
        self.endpoint = _remote_endpoint(base_url, endpoint_path)
        self.model_id = model_id
        self.max_retries = max_retries
        self.response_score_mode = response_score_mode
        self.output_score_mode = output_score_mode
        self.request_headers = (
            {"Authorization": f"Bearer {api_key}"}
            if api_key
            else None
        )
        self.client = client or _remote_client(api_key, timeout_seconds)

    @staticmethod
    def _parse_batch(body: dict, expected_count: int) -> np.ndarray:
        results = body.get("results")
        if results is None and isinstance(body.get("scores"), list):
            results = [
                {"index": index, "relevance_score": score}
                for index, score in enumerate(body["scores"])
            ]
        if not isinstance(results, list):
            raise RuntimeError("Remote reranker response must contain a results list")

        ordered: list[float | None] = [None] * expected_count
        for fallback_index, item in enumerate(results):
            if not isinstance(item, dict):
                raise RuntimeError("Remote reranker results must be objects")
            index = item.get("index", fallback_index)
            if not isinstance(index, int) or not 0 <= index < expected_count:
                raise RuntimeError("Remote reranker returned an invalid input index")
            if ordered[index] is not None:
                raise RuntimeError("Remote reranker returned a duplicate input index")
            score = item.get("relevance_score", item.get("score"))
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RuntimeError("Remote reranker item is missing its numeric score")
            ordered[index] = float(score)

        if any(score is None for score in ordered):
            raise RuntimeError(
                f"Remote reranker returned {len(results)} scores for "
                f"{expected_count} candidates"
            )
        return np.asarray(
            [score for score in ordered if score is not None],
            dtype=float,
        )

    def rerank(
        self,
        query: str,
        texts: list[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> list[float]:
        score_batches = []
        for start in range(0, len(texts), batch_size):
            text_batch = texts[start:start + batch_size]
            body = _post_remote_json(
                self.client,
                self.endpoint,
                {
                    "model": self.model_id,
                    "query": query,
                    "documents": text_batch,
                    "top_n": len(text_batch),
                    "return_documents": False,
                    "max_length": max_length,
                },
                operation="reranker",
                max_retries=self.max_retries,
                headers=self.request_headers,
            )
            returned_model = body.get("model")
            if returned_model is not None and returned_model != self.model_id:
                raise RuntimeError(
                    "Remote reranker served model "
                    f"{returned_model!r}, expected {self.model_id!r}"
                )
            returned_score_mode = body.get("score_mode")
            if (
                returned_score_mode is not None
                and returned_score_mode != self.response_score_mode
            ):
                raise RuntimeError(
                    "Remote reranker returned score mode "
                    f"{returned_score_mode!r}, configured as "
                    f"{self.response_score_mode!r}"
                )
            score_batches.append(self._parse_batch(body, len(text_batch)))

        scores = np.concatenate(score_batches).astype(float, copy=False)
        scores = _convert_reranker_scores(
            scores,
            source_mode=self.response_score_mode,
            target_mode=self.output_score_mode,
        )
        return scores.tolist()


@lru_cache(maxsize=2)
def _build_embedding_provider(backend: str) -> EmbeddingProvider:
    if backend == "local":
        return LocalEmbeddingProvider()
    return RemoteEmbeddingProvider(
        base_url=_required_environment("EMBEDDING_REMOTE_BASE_URL"),
        model_id=os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-small-zh-v1.5"),
        api_key=os.getenv("EMBEDDING_REMOTE_API_KEY", ""),
        endpoint_path=os.getenv(
            "EMBEDDING_REMOTE_PATH",
            DEFAULT_EMBEDDING_REMOTE_PATH,
        ),
        timeout_seconds=_positive_float_environment(
            "EMBEDDING_REMOTE_TIMEOUT_SECONDS",
            DEFAULT_REMOTE_TIMEOUT_SECONDS,
        ),
        max_retries=_non_negative_int_environment(
            "EMBEDDING_REMOTE_MAX_RETRIES",
            DEFAULT_REMOTE_MAX_RETRIES,
        ),
    )


def get_embedding_provider() -> EmbeddingProvider:
    backend = _environment_choice(
        "EMBEDDING_BACKEND",
        "local",
        SUPPORTED_INFERENCE_BACKENDS,
    )
    return _build_embedding_provider(backend)


@lru_cache(maxsize=2)
def _build_reranker_provider(backend: str) -> RerankerProvider:
    output_score_mode = _environment_choice(
        "RERANKER_SCORE_MODE",
        "probability",
        SUPPORTED_RERANKER_SCORE_MODES,
    )
    if backend == "local":
        return LocalRerankerProvider(output_score_mode=output_score_mode)
    return RemoteRerankerProvider(
        base_url=_required_environment("RERANKER_REMOTE_BASE_URL"),
        model_id=os.getenv("RERANKER_MODEL_ID", RERANKER_MODEL),
        api_key=os.getenv("RERANKER_REMOTE_API_KEY", ""),
        endpoint_path=os.getenv(
            "RERANKER_REMOTE_PATH",
            DEFAULT_RERANKER_REMOTE_PATH,
        ),
        timeout_seconds=_positive_float_environment(
            "RERANKER_REMOTE_TIMEOUT_SECONDS",
            DEFAULT_REMOTE_TIMEOUT_SECONDS,
        ),
        max_retries=_non_negative_int_environment(
            "RERANKER_REMOTE_MAX_RETRIES",
            DEFAULT_REMOTE_MAX_RETRIES,
        ),
        response_score_mode=_environment_choice(
            "RERANKER_REMOTE_SCORE_MODE",
            "raw_logits",
            SUPPORTED_RERANKER_SCORE_MODES,
        ),
        output_score_mode=output_score_mode,
    )


def get_reranker_provider() -> RerankerProvider:
    backend = _environment_choice(
        "RERANKER_BACKEND",
        "local",
        SUPPORTED_INFERENCE_BACKENDS,
    )
    return _build_reranker_provider(backend)


def clear_inference_caches() -> None:
    """Clear provider/model caches after changing inference configuration."""
    _build_embedding_provider.cache_clear()
    _build_reranker_provider.cache_clear()
    get_embedding_model.cache_clear()
    get_reranker_model.cache_clear()


def get_chat_completion_block(session_id, question, references):
    """
    结合知识库内容生成回答，并在回答中标注引用来源。

    :param question: 用户问题
    :param references: 知识库内容，格式为 [{"id": 1, "content": "..."}, ...]
    :return: 模型的回答
    """
    try:
        
        # 初始化 OpenAI 客户端
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL")
        )
        # 格式化参考内容
        formatted_references = "\n".join([f"[{ref['id']}] {ref['content']}" for ref in references])
    
        # 构造提示词
        
    
        # 调用模型生成回答
        completion = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
    
        return completion.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

def rerank_similarity(query, texts, *, batch_size: int | None = None):
    """Score query-document pairs with the configured reranker provider."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if isinstance(texts, str):
        raise TypeError("texts must be an iterable of strings, not a string")
    texts = list(texts)
    if not all(isinstance(text, str) for text in texts):
        raise TypeError("texts must contain only strings")
    if not texts:
        return np.array([], dtype=float), None

    resolved_batch_size = (
        _positive_int_environment(
            "RERANKER_BATCH_SIZE",
            DEFAULT_RERANKER_BATCH_SIZE,
        )
        if batch_size is None
        else batch_size
    )
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    max_length = _positive_int_environment(
        "RERANKER_MAX_LENGTH",
        DEFAULT_RERANKER_MAX_LENGTH,
    )

    provider = get_reranker_provider()
    scores = np.asarray(
        provider.rerank(
            query,
            texts,
            batch_size=resolved_batch_size,
            max_length=max_length,
        ),
        dtype=float,
    )
    if scores.shape != (len(texts),):
        raise RuntimeError(
            f"{provider.backend.capitalize()} reranker returned "
            f"{scores.shape[0] if scores.ndim else 1} scores for "
            f"{len(texts)} candidates"
        )
    if not np.isfinite(scores).all():
        raise RuntimeError(
            f"{provider.backend.capitalize()} reranker returned non-finite scores"
        )
    return scores, None


def generate_embedding(
    text: str | List[str],
    *,
    batch_size: int | None = None,
) -> list[float] | list[list[float]]:
    """Generate normalized embeddings with the configured provider.

    A single string returns one vector. A list returns vectors in the same
    order. Raising on model or inference failures prevents uploads from being
    recorded with missing embeddings.
    """
    if isinstance(text, str):
        texts = [text]
        single_input = True
    elif isinstance(text, list) and all(isinstance(item, str) for item in text):
        if not text:
            return []
        texts = text
        single_input = False
    else:
        raise TypeError("text must be a string or a list of strings")

    resolved_batch_size = (
        _positive_int_environment(
            "EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE
        )
        if batch_size is None
        else batch_size
    )
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    provider = get_embedding_provider()
    embeddings = provider.embed(texts, batch_size=resolved_batch_size)
    embedding_array = np.asarray(embeddings, dtype=np.float32)
    expected_shape = (len(texts), EMBEDDING_DIMENSION)
    if embedding_array.shape != expected_shape:
        raise RuntimeError(
            f"{provider.backend.capitalize()} embedding returned an unexpected "
            "embedding shape: "
            f"{embedding_array.shape}, expected {expected_shape}"
        )
    if not np.isfinite(embedding_array).all():
        raise RuntimeError(
            f"{provider.backend.capitalize()} embedding returned non-finite values"
        )

    serialized_embeddings = embedding_array.tolist()
    return serialized_embeddings[0] if single_input else serialized_embeddings


# 示例调用
if __name__ == "__main__":
    # 示例调用
    question = "法国的首都是哪里？"
    references = [
        {"id": 1, "content": "法国的首都是巴黎。"},
        {"id": 2, "content": "巴黎是欧洲的文化中心之一。"},
    ]
    session_id = "sd"
    
    response = get_chat_completion_block(session_id, question, references)
    print(response)
