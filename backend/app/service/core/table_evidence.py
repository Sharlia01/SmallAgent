"""Turn recognized HTML tables into self-contained rows without an LLM."""

from dataclasses import dataclass
from html import escape
from html.parser import HTMLParser
import re


@dataclass
class Cell:
    text: str
    header: bool
    rowspan: int = 1
    colspan: int = 1
    scope: str = ""


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.caption = ""
        self.row = None
        self.cell = None
        self.in_caption = False
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self.depth += 1
            if self.depth > 1:
                raise ValueError("Nested tables cannot be bound safely")
        elif tag == "caption":
            self.in_caption = True
        elif tag == "tr":
            self.row = []
        elif tag in {"th", "td"} and self.row is not None:
            spans = [int(attrs.get(name, "1")) for name in ("rowspan", "colspan")]
            if any(value < 1 or value > 256 for value in spans):
                raise ValueError("Unsupported table span")
            self.cell = Cell("", tag == "th", *spans, attrs.get("scope", ""))
        elif tag == "br" and self.cell is not None:
            self.cell.text += " "

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.text += data
        elif self.in_caption:
            self.caption += data

    def handle_endtag(self, tag):
        if tag in {"th", "td"} and self.cell is not None:
            self.cell.text = " ".join(self.cell.text.split())
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None
        elif tag == "caption":
            self.in_caption = False
        elif tag == "table":
            self.depth -= 1


def _grid(rows):
    grid = {}
    for r, cells in enumerate(rows):
        c = 0
        for cell in cells:
            while (r, c) in grid:
                c += 1
            if c + cell.colspan > 256 or r + cell.rowspan > len(rows):
                raise ValueError("Invalid table dimensions")
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    key = (r + dr, c + dc)
                    if key in grid:
                        raise ValueError("Overlapping table spans")
                    grid[key] = cell
            c += cell.colspan
    width = max((c for _, c in grid), default=-1) + 1
    return [[grid.get((r, c)) for c in range(width)] for r in range(len(rows))]


def _numeric_cell(text):
    return bool(re.fullmatch(r"[+\-−(（]?\d[\d,，.％%()）\s-]*", text))


def _year_label(text):
    return bool(re.fullmatch(r"(?:19|20)\d{2}(?:[AEae]|年)?|\d{2}[AEae]", text))


def bind_table_rows(html: str) -> list[dict]:
    """Keep ambiguous tables intact; never invent headers for numeric rows."""
    parser = TableParser()
    try:
        parser.feed(html)
        grid = _grid(parser.rows)
    except (ValueError, TypeError):
        grid = []
    caption = " ".join(parser.caption.split())
    fallback = [{"content": "【表格结构未可靠绑定；以下为原始识别结果，不得猜测错位或缺失的表头对应关系。】\n" + html, "locator_kwd": caption,
                 "evidence_type_kwd": "table", "table_binding_kwd": "unbound"}]
    if not grid:
        return fallback

    # OCR commonly labels the first key/value pair as <th>. For narrow tables,
    # require a recognizable column heading instead of turning values into labels.
    first = next((row for row in parser.rows if len(row) > 1), [])
    if len(grid[0]) == 2 and first:
        heading = re.sub(r"\s|[：:]", "", first[0].text).casefold()
        if heading not in {
            "指标", "项目", "类别", "名称", "年度", "年份", "时间", "季度",
            "产品", "业务", "地区", "序号", "公司", "企业", "股东", "参数",
            "item", "metric", "name", "year", "date", "category", "parameter",
        } and first[0].scope != "col":
            return fallback

    headers = []
    context = []
    result = []
    previous_header = False
    section = ""
    previous_label = ""
    for r, row in enumerate(grid):
        raw = parser.rows[r]
        # Full-width units/section labels apply to subsequent rows, not columns.
        if len(raw) == 1 and raw[0].colspan == len(row) and len(row) > 1:
            if raw[0].text:
                if headers and not re.search(r"单位\s*[:：]", raw[0].text):
                    section = raw[0].text
                else:
                    context.append(raw[0].text)
            continue
        has_values = any(_numeric_cell(cell.text) and not _year_label(cell.text) for cell in raw)
        # OCR tags negative/accounting values as <th> surprisingly often.
        explicit_header = all(cell.header and cell.scope != "row" for cell in raw) and not has_values
        inferred_header = (
            not headers and not result and len(raw) > 1
            and not has_values
        )
        if explicit_header or inferred_header:
            # A repeated header inside the body starts a new header block.
            if result and not previous_header:
                headers = []
            headers.append(row)
            previous_header = True
            continue
        previous_header = False
        if not headers:
            return fallback
        if len(headers) > 1 and any(
            (not upper[c] or not upper[c].text) and lower[c] and lower[c].text
            for upper, lower in zip(headers, headers[1:]) for c in range(1, len(row))
        ):
            # Missing colspan information cannot be reconstructed from whitespace.
            return fallback

        # 按列索引把多层表头拼成每列的标签
        labels = []
        for c in range(len(row)):
            path = []
            for header in headers:
                value = header[c].text if header[c] else ""
                if value and (not path or path[-1] != value):
                    path.append(value)
            labels.append(" / ".join(path) or f"第{c + 1}列（无表头）")
        values = [cell.text if cell else "" for cell in row]
        if not any(values):
            continue
        title = "；".join([part for part in [caption, *context, section] if part])
        bound = "；".join(f"{label}：{value or '（空白）'}" for label, value in zip(labels, values))
        if previous_label and re.fullmatch(r"[（(]?\+/?[-−]%[）)]?|同比|同比增速|增速|增长率", values[0]):
            bound = f"上一数据行指标：{previous_label}\n{bound}"
        else:
            previous_label = values[0]
        # Keep a small normalized HTML row for audit and existing table evaluation.
        snippet = "<table>" + (f"<caption>{escape(title)}</caption>" if title else "")
        snippet += "<tr>" + "".join(f"<th>{escape(label)}</th>" for label in labels) + "</tr>"
        snippet += "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in values) + "</tr></table>"
        result.append({
            "content": f"{title}\n{bound}\n{snippet}".strip(),
            "locator_kwd": caption,
            "evidence_type_kwd": "table",
            "table_binding_kwd": "bound",
            "table_row_int": r + 1,
        })
    return result or fallback
