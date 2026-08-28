import os
from functools import lru_cache
from pathlib import Path
from typing import List

import dashscope
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


DEFAULT_EMBEDDING_MODEL_PATH = "/models/bge-small-zh-v1.5"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DIMENSION = 512
EMBEDDING_VECTOR_FIELD = f"q_{EMBEDDING_DIMENSION}_vec"


def _positive_int_environment(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


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

def rerank_similarity(query, texts):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    texts = list(texts)

    # === 重排前：输入顺序（按 sres.ids 排序） ===
    print(f"\n[rerank_similarity] query={query[:60]!r} n_texts={len(texts)}")
    print(f"[rerank_similarity] 重排前（输入顺序）:")
    for i, t in enumerate(texts):
        print(f"  texts[{i}]: {t[:60]!r}")

    # 直接调 DashScope rerank,用 result.index 回填到原序数组
    resp = dashscope.TextReRank.call(
        model="gte-rerank-v2",
        top_n=len(texts),
        query=query,
        documents=texts,
        api_key=api_key,
    )

    scores = np.zeros(len(texts), dtype=float)
    for r in resp.output.results:
        scores[r.index] = r.relevance_score

    # === 重排后：按 rerank 分数降序的新排名 ===
    print(f"[rerank_similarity] 重排后（按 rerank 分数降序）:")
    for rank, r in enumerate(resp.output.results):
        snippet = texts[r.index][:60]
        print(f"  rank {rank}: orig_idx={r.index}  score={r.relevance_score:.4f}  text={snippet!r}")
    print(f"[rerank_similarity] 对齐后 scores（与 texts 同序）={np.round(scores, 4).tolist()}\n")

    return scores, None




def generate_embedding(
    text: str | List[str],
    *,
    batch_size: int | None = None,
) -> list[float] | list[list[float]]:
    """Generate normalized embeddings with the local BGE model.

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

    model = get_embedding_model()
    try:
        embeddings = model.encode(
            texts,
            batch_size=resolved_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as error:
        raise RuntimeError(f"Local BGE embedding inference failed: {error}") from error

    embedding_array = np.asarray(embeddings, dtype=np.float32)
    expected_shape = (len(texts), EMBEDDING_DIMENSION)
    if embedding_array.shape != expected_shape:
        raise RuntimeError(
            "Local BGE returned an unexpected embedding shape: "
            f"{embedding_array.shape}, expected {expected_shape}"
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
