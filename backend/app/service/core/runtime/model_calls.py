"""执行受请求预算约束的模型调用；请求失败时记录日志并向上抛出。"""

from __future__ import annotations

import logging
from typing import Any

from service.core.runtime.context import current_runtime

logger = logging.getLogger(__name__)

# 调用模型
def model_completion(client: Any, stage: str, **kwargs: Any) -> Any:
    runtime = current_runtime()
    try:
        if runtime is not None:
            runtime.reserve("model")
            kwargs["timeout"] = min(float(kwargs.get("timeout", 20)), runtime.remaining())
        # 禁用 SDK 的隐式重试，由业务层在同一请求预算内决定是否重试。
        if runtime is not None and callable(getattr(client, "with_options", None)):
            client = client.with_options(max_retries=0)
        return client.chat.completions.create(**kwargs)
    except Exception as error:
        logger.warning("%s model request failed: %s: %s", stage, type(error).__name__, str(error)[:500])
        raise
