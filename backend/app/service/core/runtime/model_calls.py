"""执行受请求预算约束的模型调用；请求失败时记录日志并向上抛出。"""

from __future__ import annotations

import logging
from typing import Any

from service.core.runtime.context import current_runtime

logger = logging.getLogger(__name__)


def _json_object_messages(messages):
    """JSON 模式必须有明确的系统指令，不能依赖用户或证据中偶然出现 JSON。"""
    messages = list(messages)
    for message in messages:
        if message.get("role") not in {"system", "developer"}:
            continue
        content = message.get("content")
        texts = [content] if isinstance(content, str) else [
            part.get("text", "") for part in (content or [])
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        if any(isinstance(text, str) and "json" in text.lower() for text in texts):
            return messages

    instruction = "请仅返回符合上述结构的有效 JSON 对象，不要输出 Markdown 或解释文字。"
    # 复制被修改的消息及内容，避免纠错、重规划或并发请求污染共享提示词。
    for index, message in enumerate(messages):
        if message.get("role") not in {"system", "developer"}:
            continue
        content = message.get("content")
        if isinstance(content, str):
            messages[index] = {**message, "content": content + "\n" + instruction}
            return messages
        if isinstance(content, list):
            messages[index] = {**message, "content": [*content, {"type": "text", "text": instruction}]}
            return messages
    return [{"role": "system", "content": instruction}, *messages]


# 调用模型
def model_completion(client: Any, stage: str, **kwargs: Any) -> Any:
    runtime = current_runtime()
    try:
        if (kwargs.get("response_format") or {}).get("type") == "json_object":
            kwargs["messages"] = _json_object_messages(kwargs.get("messages", []))
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
