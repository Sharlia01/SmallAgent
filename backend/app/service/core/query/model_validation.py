"""解析并校验模型输出；校验失败时允许模型修正一次。"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from service.core.query.validation import QueryValidationError
from service.core.runtime.diagnostics import failure_details
from service.core.runtime.model_calls import model_completion
from service.core.runtime.context import QueryBudgetExceeded

logger = logging.getLogger(__name__)

# 先从模型回答中解析并填充结构化数据，再校验内容。
def parse_and_validate(content: str, *, parse: Callable[[str], Any],
                       validate: Callable[[Any], Any], finish_reason: str | None = None) -> Any:
    """先解析结构，再校验内容；被长度限制截断的响应不能通过校验。"""
    value = parse(content)
    if finish_reason == "length":
        raise QueryValidationError("response_truncated", "model output reached the token limit", field="response.finish_reason")
    return validate(value)


def validated_completion(client: Any, stage: str, *, parse: Callable[[str], Any],
                         validate: Callable[[Any], Any], fallback_model: str | None = None,
                         fallback_timeout: float | None = None, **kwargs: Any) -> Any:
    """调用模型并检查结果，第一次校验失败后携带错误要求模型修正。

    配置 fallback_model 时，第二次改用兜底模型，且请求失败也可触发兜底。
    未配置时仅重试输出校验失败；预算耗尽及第二次失败均向上抛出。
    两次调用共用请求预算，失败原因写入普通日志。
    """
    messages = list(kwargs.pop("messages"))
    if fallback_model and callable(getattr(client, "with_options", None)):
        client = client.with_options(max_retries=0)
    # 调用两次，第一次如果失败了就把错误交给模型修正；第二次失败直接抛出异常。
    for attempt in (1, 2):
        if attempt == 2 and fallback_model:
            logger.info("%s model fallback: from=%s to=%s", stage, kwargs.get("model"), fallback_model)
            kwargs["model"] = fallback_model
            if fallback_timeout is not None:
                kwargs["timeout"] = fallback_timeout
        try:
            completion = model_completion(client, stage, messages=messages, **kwargs)
        except QueryBudgetExceeded:
            raise
        except Exception:
            if attempt == 2 or not fallback_model:
                raise
            continue
        choices = getattr(completion, "choices", None) or []
        choice = choices[0] if choices else None
        finish_reason = getattr(choice, "finish_reason", None)
        raw_content = getattr(getattr(choice, "message", None), "content", None)
        content = raw_content.strip() if isinstance(raw_content, str) else ""
        # content是模型返回的JSON文本；parse_and_validate会先解析结构，再校验内容。
        try:
            if not choices:
                raise QueryValidationError("missing_choices", "model returned no choices", field="response.choices")
            if not content:
                raise QueryValidationError("empty_content", "model returned empty content", field="response.content")
            return parse_and_validate(content, parse=parse, validate=validate, finish_reason=finish_reason)
        except (ValueError, TypeError, KeyError) as error:
            details = failure_details(error)
            logger.warning("%s validation failed (attempt %s/2): %s",
                           stage, attempt, json.dumps(details, ensure_ascii=False))
            if attempt == 2:
                raise
            if finish_reason == "length":
                # 理解输出包含结构化事实要求，修正时不能反而降低原有额度。
                original_limit = int(kwargs.get("max_tokens", 1800))
                kwargs["max_tokens"] = max(original_limit, min(8192, original_limit * 2))
            # 把错误交给模型修正；保留原始问题，不能用错误结果替换用户需求。
            messages = [*messages,
                        {"role": "assistant", "content": content[:6000] or "{}"},
                        {"role": "user", "content":
                         "上一份输出未通过校验：" + json.dumps(details, ensure_ascii=False) +
                         "。请根据原始问题修正并返回完整JSON。保留未知关系，不要猜测答案；"
                         "分别绑定各目标的主体、时间和指标，允许等价专业术语。"}]
