"""证据与财务核验共用的模型客户端配置，不依赖核验业务实现。"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@lru_cache(maxsize=1)
def get_verification_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise RuntimeError(
            "DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL are required for "
            "evidence sufficiency checks"
        )
    return OpenAI(api_key=api_key, base_url=base_url)


@lru_cache(maxsize=1)
def get_context_client() -> OpenAI:
    """指代消解使用相同提供方，职责不再依赖旧的 RAG 意图识别模块。"""
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise RuntimeError("指代消解模型未配置")
    return OpenAI(api_key=api_key, base_url=base_url)


def context_model() -> str:
    return os.getenv("AGENT_CONTEXT_MODEL", os.getenv("AGENT_MODEL", "deepseek-v4-pro"))


def response_content(completion) -> str:
    choices = getattr(completion, "choices", None) or []
    content = getattr(getattr(choices[0], "message", None), "content", None) if choices else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("模型未返回有效文本")
    return content.strip()
