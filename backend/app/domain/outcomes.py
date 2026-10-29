"""运行结果与回答模式分开记录，防止把技术故障当成正确拒答。"""
NON_EVIDENCE_OUTCOMES = frozenset({"invalid_plan", "unsupported_capability", "calculation_error", "execution_error"})


def result_outcome(record):
    retrieval = record.get("retrieval") or {}
    return (record.get("outcome") or (retrieval.get("agent_execution") or {}).get("outcome")
            or (retrieval.get("evidence_sufficiency") or {}).get("outcome"))
