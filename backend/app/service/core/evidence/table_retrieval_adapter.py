"""查询期表格对象到现有证据/API 字段的输出适配，内部不维护可变镜像。"""
from domain.evidence import plain, fingerprint
from domain.tables import build_table_markdown
from domain.table_retrieval import TableCandidate, TableContext


def candidate_record(candidate: TableCandidate) -> dict:
    record = plain(candidate.source)
    rows = plain(candidate.matched_rows)
    primary = rows[0]
    text = build_table_markdown({'caption': candidate.caption, 'binding_status': 'bound', 'data_rows': rows})
    # 只有在兼容输出边界才生成旧字段，原始索引快照保持不变。
    record.pop('table_context_obj', None)
    record.update(
        chunk_id=candidate.candidate_id,
        content_with_weight=text,
        table_semantic_text_with_weight=candidate.ranking_text,
        table_markdown_excerpt_with_weight=text,
        table_chunk_kind_kwd='matched_rows',
        table_row_int=primary['row_index'], table_row_start_int=primary['row_index'],
        table_row_end_int=max(candidate.matched_row_indices),
        table_cells_slice_obj={'caption': candidate.caption, 'binding_status': 'bound', 'rows': rows},
        table_candidate_obj={
            'candidate_id': candidate.candidate_id, 'member_chunk_ids': list(candidate.member_chunk_ids),
            'rows': rows, 'matched_row_indices': list(candidate.matched_row_indices),
            'omitted_chunk_ids': list(candidate.omitted_chunk_ids),
            'scored_row_indices': list(candidate.scored_row_indices),
            'row_ranking_source': candidate.ranking_source,
            'rerank_text_truncated': candidate.ranking_text_truncated,
        },
    )
    for field in ('parallel_subqueries', 'sequential_subqueries'):
        if any(field in row.source for row in candidate.source_rows):
            record[field] = [trace for row in candidate.source_rows for trace in plain(row.source.get(field, ()))]
    return record


def context_record(context: TableContext, retrieval_record: dict | None = None) -> dict:
    base = (candidate_record(context.candidate) if isinstance(context.candidate, TableCandidate)
            else plain(context.candidate.source))
    record = {**base, **(retrieval_record or {})}
    record['content_with_weight'] = context.text
    record['table_context_obj'] = plain(context.diagnostics)
    if context.rows:
        record.pop('table_markdown_excerpt_with_weight', None)
        record['table_context_obj']['rows'] = plain(context.rows)
        # 未展示的命中行只能从私有恢复快照取得，不能从候选元数据绕过裁剪。
        record['table_cells_slice_obj'] = {
            'caption': str(context.diagnostics.get('caption') or ''),
            'binding_status': 'bound' if context.diagnostics.get('reliable') else 'unbound',
            'rows': plain(context.rows),
        }
        if 'table_candidate_obj' in record:
            record['table_candidate_obj'] = {**record['table_candidate_obj'], 'rows': plain(context.matched_rows)}
        # 回答与核验只能看到本次实际展示的行；不能借旧重排文本恢复被裁掉的行。
        lines = [str(context.diagnostics.get('caption') or '')]
        for row in context.rows:
            lines.extend((str(row.get('context') or ''), str(row.get('preamble') or '')))
            lines.append('；'.join(
                f"{header}：{value or '（空白）'}"
                for header, value in zip(row.get('headers', ()), row.get('cells', ()), strict=True)
            ))
        record['table_semantic_text_with_weight'] = '\n'.join(line for line in lines if line)
    recovery = []
    for snapshot in context.recovery_rows:
        row = plain(snapshot)
        # 每行独立交付，避免再次进入原窗口裁剪；原始列完整保留。
        restored = {key: value for key, value in base.items()
                    if key not in {'table_candidate_obj', 'table_context_obj',
                                   'table_semantic_text_with_weight', 'table_markdown_excerpt_with_weight',
                                   '_table_recovery_records'}}
        caption = str(context.diagnostics.get('caption') or '')
        restored.update(
            table_cells_slice_obj={'caption': caption, 'binding_status': 'bound', 'rows': [row]},
            table_row_int=row['row_index'], table_row_start_int=row['row_index'], table_row_end_int=row['row_index'],
            table_chunk_kind_kwd='recovered_row',
            content_with_weight=build_table_markdown({'caption': caption, 'binding_status': 'bound', 'data_rows': [row]}),
        )
        restored['chunk_id'] = 'recovery-' + fingerprint(restored)
        recovery.append(restored)
    # 私有边界字段：在来源适配时单独保存，不进入 Evidence 元数据或模型资料。
    if recovery:
        record['_table_recovery_records'] = recovery
    return record
