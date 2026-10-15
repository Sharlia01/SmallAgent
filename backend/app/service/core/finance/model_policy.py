"""财务语义核验的配置、预算和调用边界。"""

import json
import os

from service.core.runtime.context import positive_setting
from service.core.runtime.model_calls import model_completion


def verification_mode():
    mode = os.getenv("RAG_FINANCIAL_VERIFICATION_MODE")
    if mode is None:
        legacy = os.getenv("RAG_FINANCIAL_FACT_MODEL_ENABLED")
        mode = "rules" if legacy is not None and legacy.strip().lower() not in {"true", "1", "yes", "on"} else "hybrid"
    if mode not in {"rules", "hybrid"}:
        raise ValueError("invalid_financial_verification_mode")
    return mode


def settings():
    return dict(mode=verification_mode(),
                model=os.getenv("RAG_FINANCIAL_FACT_MODEL") or os.getenv("RAG_EVIDENCE_SUFFICIENCY_MODEL", "qwen3.7-flash-2026-07-15"),
                timeout=positive_setting("RAG_FINANCIAL_FACT_TIMEOUT_SECONDS", 8, maximum=30),
                max_chunks=int(positive_setting("RAG_FINANCIAL_FACT_MAX_CHUNKS", 3, maximum=8)),
                max_candidates=int(positive_setting("RAG_FINANCIAL_FACT_MAX_CANDIDATES", 12, maximum=32)))


def completion(client, stage, prompt, payload):
    config = settings()
    result = model_completion(client, stage, model=config["model"],
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        response_format={"type": "json_object"}, temperature=0, max_tokens=6000,
        timeout=config["timeout"], stream=False, extra_body={"enable_thinking": False})
    if not result.choices or getattr(result.choices[0], "finish_reason", None) == "length":
        raise ValueError("incomplete_financial_model_response")
    payload = json.loads(result.choices[0].message.content)
    if not isinstance(payload, dict):
        raise ValueError("invalid_financial_model_response")
    return payload
