"""用已声明且有证据来源的变量生成核验快照，保持首次理解不变。"""

from dataclasses import replace

from domain.facts import FinancialBinding
from service.core.entities.attribution import get_entity_registry
from service.core.finance.rules import company_mentions, document_subject, period_value


def bind_financial_requirements(context, evidence, executions):
    financial = context.understanding.financial
    if financial is None:
        context.resolved_financial = None
        context.financial_bindings = ()
        return
    registry = get_entity_registry()
    required = {ref for item in financial.facts for ref in (item.entity_ref, item.period_ref) if ref}
    bindings = {}
    conflicts = set()
    for execution in executions.values():
        binding = execution.binding
        key = execution.task.binding_key or (binding.slot if binding else None)
        if binding is None or key not in required or binding.slot != execution.task.output_slot:
            continue
        source = evidence.get(binding.source_chunk_id)
        # 不能只相信任务状态；绑定必须仍能在当前证据快照中找到出处。
        if source is None or binding.document_id != source.source.document_id:
            continue
        quote = binding.source_quote or binding.value
        if quote not in source.text or binding.value not in quote:
            continue
        value = FinancialBinding(key, binding.value, source.id, binding.document_id, quote)
        if key in bindings and bindings[key].value != value.value:
            conflicts.add(key)
        bindings[key] = value
    if "document_subject" in required:
        candidates = [(item, document_subject(item, registry)[0]) for item in evidence]
        subjects = {subject for _, subject in candidates if subject}
        if len(subjects) == 1:
            identifier = next(iter(subjects))
            source = next(item for item, subject in candidates if subject == identifier)
            # document_subject 已按可靠文档归属校验；记录元数据证明，不伪造正文引文。
            bindings["document_subject"] = FinancialBinding(
                "document_subject", registry.entities[identifier].canonical_name,
                source.id, source.source.document_id, "",
            )
    for slot in conflicts:
        bindings.pop(slot, None)
    facts = []
    used = set()
    for item in financial.facts:
        entity_id, entity_name, period = item.entity_id, item.entity_name, item.period
        missing = False
        for ref, field in ((item.entity_ref, "entity"), (item.period_ref, "period")):
            if not ref:
                continue
            binding = bindings.get(ref)
            if binding is None:
                missing = True
                continue
            if field == "entity":
                mentions = company_mentions(binding.value, registry)
                ids = registry.resolved_ids(mentions, "company")
                if len(ids) != 1:
                    missing = True
                    continue
                entity_id, entity_name = ids[0], binding.value
            else:
                period = period_value(binding.value)
                if period is None:
                    missing = True
                    continue
            used.add(ref)
        status = "deferred" if missing else "resolved" if entity_id else "unknown"
        facts.append(replace(item, entity_id=entity_id, entity_name=entity_name,
                             period=period, binding_status=status))
    context.resolved_financial = replace(financial, facts=tuple(facts))
    context.financial_bindings = tuple(bindings[ref] for ref in sorted(used))
