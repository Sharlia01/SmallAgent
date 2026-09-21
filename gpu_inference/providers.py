"""Local model engines used by the standalone GPU inference service."""

from pathlib import Path
from threading import Lock

import numpy as np


EMBEDDING_DIMENSION = 512
SUPPORTED_SCORE_MODES = {"raw_logits", "probability"}

# 一个数值转换函数，负责在raw_logits和probability之间进行转换
def convert_scores(
    scores: np.ndarray,
    *,
    source_mode: str,
    target_mode: str,
) -> np.ndarray:
    if source_mode not in SUPPORTED_SCORE_MODES:
        raise ValueError(f"Unsupported source score mode: {source_mode}")
    if target_mode not in SUPPORTED_SCORE_MODES:
        raise ValueError(f"Unsupported target score mode: {target_mode}")
    if source_mode == target_mode:
        return scores
    if source_mode == "raw_logits":
        positive = scores >= 0
        converted = np.empty_like(scores, dtype=float)
        converted[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
        exp_scores = np.exp(scores[~positive])
        converted[~positive] = exp_scores / (1.0 + exp_scores)
        return converted

    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Probability scores must be between zero and one")
    epsilon = np.finfo(float).eps
    clipped = np.clip(scores, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


# 封装 sentence-transformers 的 SentenceTransformer 模型
class EmbeddingEngine:
    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        dimension: int = EMBEDDING_DIMENSION,
    ):
        resolved_path = Path(model_path).expanduser()
        if not resolved_path.is_dir():
            raise RuntimeError(
                f"Embedding model directory does not exist: {resolved_path}"
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required for embedding inference"
            ) from error

        try:
            # 从本地加载模型，不联网
            self.model = SentenceTransformer(
                str(resolved_path),
                device=device,
                local_files_only=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load embedding model from {resolved_path}: {error}"
            ) from error
        self.dimension = dimension
        self.device = device
        self._lock = Lock()

    def embed(self, texts: list[str], *, batch_size: int) -> list[list[float]]:
        if not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError("texts must be a non-empty list of strings")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        try:
            with self._lock:
                # 将文本转换为嵌入向量，返回numpy数组
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
        except Exception as error:
            raise RuntimeError(f"Embedding inference failed: {error}") from error

        embedding_array = np.asarray(embeddings, dtype=np.float32)
        expected_shape = (len(texts), self.dimension)
        # 检查输出shape和有限性
        if embedding_array.shape != expected_shape:
            raise RuntimeError(
                "Embedding model returned shape "
                f"{embedding_array.shape}, expected {expected_shape}"
            )
        if not np.isfinite(embedding_array).all():
            raise RuntimeError("Embedding model returned non-finite values")
        return embedding_array.tolist()


class RerankerEngine:
    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        score_mode: str,
    ):
        if score_mode not in SUPPORTED_SCORE_MODES:
            supported = ", ".join(sorted(SUPPORTED_SCORE_MODES))
            raise RuntimeError(f"score_mode must be one of: {supported}")

        resolved_path = Path(model_path).expanduser()
        if not resolved_path.is_dir():
            raise RuntimeError(
                f"Reranker model directory does not exist: {resolved_path}"
            )
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "torch and transformers are required for reranker inference"
            ) from error

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(resolved_path),
                local_files_only=True,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(resolved_path),
                local_files_only=True,
            )
            self.model.to(device)
            self.model.eval()
        except Exception as error:
            raise RuntimeError(
                f"Failed to load reranker model from {resolved_path}: {error}"
            ) from error

        self.torch = torch
        self.device = device
        self.score_mode = score_mode
        self._lock = Lock()

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> list[float]:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not documents or not all(isinstance(item, str) for item in documents):
            raise ValueError("documents must be a non-empty list of strings")
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")

        score_batches = []
        try:
            with self._lock:
                for start in range(0, len(documents), batch_size):
                    document_batch = documents[start:start + batch_size]
                    pairs = [[query, document] for document in document_batch]
                    inputs = self.tokenizer(
                        pairs,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=max_length,
                    )
                    inputs = {
                        name: tensor.to(self.device)
                        for name, tensor in inputs.items()
                    }
                    with self.torch.inference_mode():
                        logits = self.model(
                            **inputs,
                            return_dict=True,
                        ).logits.reshape(-1)
                    score_batches.append(
                        logits.detach().float().cpu().numpy()
                    )
        except Exception as error:
            raise RuntimeError(f"Reranker inference failed: {error}") from error

        scores = np.concatenate(score_batches).astype(float, copy=False)
        expected_shape = (len(documents),)
        if scores.shape != expected_shape:
            raise RuntimeError(
                f"Reranker returned shape {scores.shape}, expected {expected_shape}"
            )
        if not np.isfinite(scores).all():
            raise RuntimeError("Reranker returned non-finite scores")
        return convert_scores(
            scores,
            source_mode="raw_logits",
            target_mode=self.score_mode,
        ).tolist()

