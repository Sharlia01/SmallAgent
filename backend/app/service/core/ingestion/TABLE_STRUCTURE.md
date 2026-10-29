# PDF 表格结构与绑定

`table-object-v6` 从结构检测到父表对象保留三类检测区域：

- `table column header`：列标题。
- `table projected row header`：行分组标题。
- `table spanning cell`：合并单元格区域，与标题角色独立。

每个区域保留类别、置信度、页码和 `[left, top, right, bottom]` 坐标。
PDF 路径的坐标为未缩放的页内坐标（`coordinate_space=page`）；
独立图片调试入口使用图片坐标（`coordinate_space=image`）。
合并区域使用独立的 `SP_*` 几何字段，不覆盖标题的 `H_*` 字段。

HTML 是内部传递载体：表标记 `data-table-structure="1"`，单元格携带
`data-table-role` 和 JSON 编码的 `data-table-regions`。父表的
`cells_json.cells` 保留 `role`、`structure_regions`、`rowspan`、`colspan`；
`cell_refs` 表达合并单元格在网格中的覆盖关系。

带结构协议的表格不使用“非数字多数就是表头”或 `<th>` 推断规则：
列标题单独维护，可组成多级路径；分组只更新行上下文，不替换列标题。
真正的新列头区段重置旧列头和分组。标题角色冲突、数据列无标题、
或无法区分漏检分组与空数据行时，返回未绑定结果，由现有视觉复核
配置决定是否继续解析；不启用视觉复核时保留原始结构，不猜测绑定。

未携带协议的 DOCX/HTML 输入保留既有解析路径。没有为历史 PDF HTML
增加表头修补规则。现存 v5 父表及 ES 文档需从原 PDF 重新解析入库，
才能获得新结构信息；修改解析器不会自动改写历史数据。

验证：`test_table_structure_pipeline.py` 覆盖检测结果传递、分组、列头切换、
多级合并与失败降级。`test_table_structure_local.py` 在本地模型和原截图
存在时验证 1008 表：当前模型漏检“盈利能力”和“估值倍数”，应标为
未绑定，避免把这两个分组的数据误归入“成长能力”。此测试不访问 ES
或外部视觉模型服务。
