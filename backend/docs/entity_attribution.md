# 入库时的公司归属标注

上传和重新入库都会自动标注，入口是 `service/core/file_parse.py` 的
`process_items`。这一步使用规则，不增加模型调用，也不改变原文、向量、片段编号
或现有检索筛选方式。标签通过 `EVIDENCE_FIELDS` 传到检索结果、Agent 工具来源和
引用对象；完整表格也会保存对应的行列归属记录。
证据充分性模型只接收简短标签，详细匹配引文和坐标不重复加入每次模型提示。

## 名称对应表

默认读取 `app/service/core/conf/entity_registry.json`，也可用
`RAG_ENTITY_REGISTRY_PATH` 指向另一份 JSON。修改后，下一次入库会重新读取；
已经保存的资料不会自动更新。配置错误会直接阻止入库，避免静默丢失标签。

每个对象需要固定编号、标准名称、类型、经过确认的别名和带交易所后缀的股票代码。
例如：

```json
{
  "version": "2",
  "entities": [
    {
      "entity_id": "company:example",
      "canonical_name": "示例股份",
      "kind": "company",
      "aliases": ["已确认的公司简称", "已确认的公司全称"],
      "stock_codes": ["123456.SH"],
      "source": "确认名称对应关系的原始资料"
    }
  ]
}
```

`kind` 支持 `company` 和 `publisher`。发布方是一个来源角色，公司也可以作为
自己报告的发布方。母子公司使用不同编号；不能只因名称相近就合并。同一个别名
指向多家公司时保留歧义，不按配置顺序选一家。裸数字不能登记为股票代码，以免
把财务数值误当成公司。当前默认表登记知识库原文中确认过的国电电力、润本股份、
可比公司表中的华能国际、龙源电力、华电国际，以及国信证券、华源证券；新增公司
和别名需补入对应表。

## 保存的字段

| 字段 | 含义 |
| --- | --- |
| `document_subject_entity_ids` | 资料主要研究的公司，由文件名或开头的明确主体标签确定 |
| `document_subject_status_kwd` | `resolved`、`multiple`、`ambiguous` 或 `unknown` |
| `publisher_entity_ids` | 发布方；明确署名与研究所提供数据的线索分别标注 |
| `publisher_status_kwd` | 明确署名为 `resolved`，仅研究所署名线索为 `candidate`，否则保留歧义或未知 |
| `mentioned_entity_ids` | 当前片段明确提到的已登记对象，包括公司和发布方 |
| `subject_entity_ids` | 通过明确公司列或公司表头确认归属的表格行/列对象 |
| `candidate_subject_entity_ids` | 文字提及或报告背景提供的可能归属，尚不能证明数字属于它 |
| `unresolved_entity_names_kwd` | 明确公司标签中尚未登记或无法确认的名称 |
| `entity_attribution_status_kwd` | 当前片段的归属状态 |
| `entity_attribution_obj` | 原文引文、匹配位置、表格行号和对应列，供后续检查使用 |
| `entity_annotation_version_kwd` / `entity_registry_version_kwd` | 标注规则版本和名称表版本（含内容摘要） |

片段状态的含义：

- `explicit_table`：表格有明确公司列（含公司简称）或公司表头，记录每条绑定的行号和列号。
- `mentions_only`：文字提到了公司，但没有证明每个数字的归属。
- `document_context`：只有整份报告的主体信息，保存为可能归属。
- `ambiguous`：名称有歧义、公司列与公司表头冲突，或不能唯一确定归属。
- `unbound_table`：表格行列本身尚未可靠绑定。
- `unknown`：没有足够信息。

后两类以及有歧义的表格不会自动继承文件名中的公司编号。普通文字也不会仅因
包含某公司名称而写入 `subject_entity_ids`。公司列与已登记股票代码冲突时，也会
保留歧义；代码列不计入财务数值列。所有记录的
`fact_verification_status` 都是 `not_verified`：当前标注解决身份和明确表格归属，
尚未验证指标、时间、数值、单位、实际/预测及报表口径。

## 已有资料和边界

旧资料仍可正常读取，缺少字段时保持原有行为。历史资料需通过现有
`app/reindex_documents.py` 重新入库才会获得新标签，例如在后端依赖环境中：

```bash
cd backend/app
python reindex_documents.py --index-name 1 \
  --file service/core/storage/file/1/国电电力.pdf \
  --file service/core/storage/file/1/润本股份.pdf
# 上面的命令仅检查目标。加 --apply 后才重新解析、生成向量并更新资料。
```

这次改动没有自动重建历史资料，也没有启用按公司过滤检索或答案发出前校验。
后续使用标签时应区分明确表格归属与候选背景，不能把候选编号直接当成已核实事实。
检索后扩展出的邻近表格行也可能涉及其他公司，应读取完整表格中每一行自己的绑定。

测试包含公司别名、股票代码、多公司同表、母子公司、发布方与研究对象不同、
未登记名称、有歧义名称、未绑定表格，以及入库到检索再到引用对象的字段保留。

2026-10-09 验证：新增 19 项归属测试通过。后端全量单元测试有 388 项通过、
10 项失败；在修改前的代码上复现了相同 10 项失败，分别来自原有公司名称提取
和查询规划验收测试。两份本地报告的只读预览也已检查：国电电力可比公司表中的
华能国际、龙源电力、华电国际三行分别归到对应公司，其余只有报告背景的财务行
继续标为候选归属；国信证券和华源证券的研究所署名保留为发布方线索。
