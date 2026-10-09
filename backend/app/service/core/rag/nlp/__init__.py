from . import rag_tokenizer
import re
import copy
from collections import Counter
import logging
import chardet
from service.core.rag.utils import num_tokens_from_string
from PIL import Image
from service.core.table_evidence import parse_table_object
from service.core.table_models import build_table_chunks
from service.core.figure_evidence import figure_evidence

def is_english(texts):
    eng = 0
    if not texts:
        return False
    for t in texts:
        if re.match(r"[ `a-zA-Z.,':;/\"?<>!\(\)-]", t.strip()):
            eng += 1
    if eng / len(texts) > 0.8:
        return True
    return False

def tokenize_chunks_docx(chunks, doc, eng, images):
    res = []
    # 封装为 Elasticsearch 文档。
    for ck, image in zip(chunks, images):
        if len(ck.strip()) == 0:
            continue
        logging.debug("-- {}".format(ck))
        d = copy.deepcopy(doc)
        d["image"] = image
        tokenize(d, ck, eng)
        res.append(d)
    return res
    
def concat_img(img1, img2):
    if img1 and not img2:
        return img1
    if not img1 and img2:
        return img2
    if not img1 and not img2:
        return None
    width1, height1 = img1.size
    width2, height2 = img2.size

    new_width = max(width1, width2)
    new_height = height1 + height2
    new_image = Image.new('RGB', (new_width, new_height))

    new_image.paste(img1, (0, 0))
    new_image.paste(img2, (0, height1))

    return new_image


def naive_merge_docx(sections, chunk_token_num=128, delimiter="\n。；！？"):
    if not sections:
        return [], []

    cks = [""]
    images = [None]
    tk_nums = [0]

    def add_chunk(text, image, pos=""):
        nonlocal cks, tk_nums, delimiter
        tnum = num_tokens_from_string(text)
        if tnum < 8:
            pos = ""
        if tk_nums[-1] > chunk_token_num:
            if text.find(pos) < 0:
                text += pos
            cks.append(text)
            images.append(image)
            tk_nums.append(tnum)
        else:
            if cks[-1].find(pos) < 0:
                text += pos
            cks[-1] += text
            images[-1] = concat_img(images[-1], image)
            tk_nums[-1] += tnum

    for sec, image in sections:
        add_chunk(sec, image, '')

    # cks是包含多段正文的列表，images是对应的图片列表
    return cks, images


def naive_merge(sections, chunk_token_num=128, delimiter="\n。；！？"):
    if not sections:
        return []
    if isinstance(sections[0], type("")):
        sections = [(s, "") for s in sections]
    cks = [""]
    tk_nums = [0]

    def add_chunk(text, pos):
        nonlocal cks, tk_nums, delimiter
        tnum = num_tokens_from_string(text)
        if not pos:
            pos = ""
        if tnum < 8:
            pos = ""
        # 确保合并片段的词元数不超过 chunk_token_num。
        if tk_nums[-1] > chunk_token_num:

            if text.find(pos) < 0:
                text += pos
            cks.append(text)
            tk_nums.append(tnum)
        else:
            if cks[-1].find(pos) < 0:
                text += pos
            cks[-1] += text
            tk_nums[-1] += tnum

    for sec, pos in sections:
        add_chunk(sec, pos)

    # cks是包含多段正文的列表
    return cks


all_codecs = [
    'utf-8', 'gb2312', 'gbk', 'utf_16', 'ascii', 'big5', 'big5hkscs',
    'cp037', 'cp273', 'cp424', 'cp437',
    'cp500', 'cp720', 'cp737', 'cp775', 'cp850', 'cp852', 'cp855', 'cp856', 'cp857',
    'cp858', 'cp860', 'cp861', 'cp862', 'cp863', 'cp864', 'cp865', 'cp866', 'cp869',
    'cp874', 'cp875', 'cp932', 'cp949', 'cp950', 'cp1006', 'cp1026', 'cp1125',
    'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254', 'cp1255', 'cp1256',
    'cp1257', 'cp1258', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr',
    'gb2312', 'gb18030', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2',
    'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1',
    'iso8859_2', 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7',
    'iso8859_8', 'iso8859_9', 'iso8859_10', 'iso8859_11', 'iso8859_13',
    'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab', 'koi8_r', 'koi8_t', 'koi8_u',
    'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2', 'mac_roman',
    'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213',
    'utf_32', 'utf_32_be', 'utf_32_le', 'utf_16_be', 'utf_16_le', 'utf_7', 'windows-1250', 'windows-1251',
    'windows-1252', 'windows-1253', 'windows-1254', 'windows-1255', 'windows-1256',
    'windows-1257', 'windows-1258', 'latin-2'
]

BULLET_PATTERN = [[
    r"第[零一二三四五六七八九十百0-9]+(分?编|部分)",
    r"第[零一二三四五六七八九十百0-9]+章",
    r"第[零一二三四五六七八九十百0-9]+节",
    r"第[零一二三四五六七八九十百0-9]+条",
    r"[\(（][零一二三四五六七八九十百]+[\)）]",
], [
    r"第[0-9]+章",
    r"第[0-9]+节",
    r"[0-9]{,2}[\. 、]",
    r"[0-9]{,2}\.[0-9]{,2}[^a-zA-Z/%~-]",
    r"[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}",
    r"[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}",
], [
    r"第[零一二三四五六七八九十百0-9]+章",
    r"第[零一二三四五六七八九十百0-9]+节",
    r"[零一二三四五六七八九十百]+[ 、]",
    r"[\(（][零一二三四五六七八九十百]+[\)）]",
    r"[\(（][0-9]{,2}[\)）]",
], [
    r"PART (ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)",
    r"Chapter (I+V?|VI*|XI|IX|X)",
    r"Section [0-9]+",
    r"Article [0-9]+"
]
]


def find_codec(blob):
    detected = chardet.detect(blob[:1024])
    if detected['confidence'] > 0.5:
        return detected['encoding']

    for c in all_codecs:
        try:
            blob[:1024].decode(c)
            return c
        except Exception:
            pass
        try:
            blob.decode(c)
            return c
        except Exception:
            pass

    return "utf-8"


def tokenize(d, t, eng):
    d["content_with_weight"] = t
    t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", t)
    d["content_ltks"] = rag_tokenizer.tokenize(t)
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])

def add_positions(d, poss):
    if not poss:
        return
    page_num_int = []
    position_int = []
    top_int = []
    for pn, left, right, top, bottom in poss:
        page_num_int.append(int(pn + 1))
        top_int.append(int(top))
        position_int.append((int(pn + 1), int(left), int(right), int(top), int(bottom)))
    d["page_num_int"] = page_num_int
    d["position_int"] = position_int
    d["top_int"] = top_int

def tokenize_table(tbls, doc, eng, batch_size=10):
    res = []
    # 追加表格片段。
    """
    img   # 表格区域截图
    rows  # 表格识别出来的文本
    poss  # 表格所在页码和坐标
    """
    for table_ordinal, ((img, rows), poss) in enumerate(tbls, start=1):
        if not rows:
            continue
        if isinstance(rows, dict) and rows.get("kind") == "figure":
            records = [figure_evidence(img, rows.get("text", ""))]
        # 如果识别的表格是HTML字符串，则解析出表格的行和列
        elif isinstance(rows, str) and re.search(r"<table\b", rows, re.I):
            location = {}
            add_positions(location, poss)
            table_object = parse_table_object(
                img,
                rows,
                provenance={
                    "document_name": str(doc.get("docnm_kwd") or ""),
                    "table_ordinal": table_ordinal,
                    "page_num_int": location.get("page_num_int") or [],
                    "position_int": location.get("position_int") or [],
                },
            )
            table_object_payload = table_object.to_dict()
            # 将表格对象拆分为多个块，每个块包含一定数量的行，以便后续处理
            for table_chunk in build_table_chunks(table_object):
                d = copy.deepcopy(doc)
                # 只有 semantic_text 参与关键词检索和向量生成；
                # 传给语言模型的内容使用确定性生成的 Markdown 表格。
                tokenize(d, table_chunk.semantic_text, eng)
                d["content_with_weight"] = table_chunk.markdown_excerpt
                d.update(table_chunk.to_record())
                d.update({
                    "evidence_type_kwd": "table",
                    "locator_kwd": table_object.cells_json.get("caption") or "",
                    "table_binding_kwd": table_object.cells_json.get("binding_status"),
                    "table_binding_source_kwd": table_object.cells_json.get("binding_source"),
                    "table_model_kwd": table_object.cells_json.get("binding_model"),
                })
                slice_rows = table_chunk.cells_slice.get("rows") or []
                if len(slice_rows) == 1:
                    row = slice_rows[0]
                    d["table_headers_kwd"] = row.get("headers") or []
                    d["table_cells_kwd"] = row.get("cells") or []
                    d["table_row_int"] = row.get("row_index")
                # 同一张表的子片段共享一个内存中的父表对象；
                # 持久化层按 table_id 只写入一次。
                d["_table_object"] = table_object_payload
                if img is not None:
                    d["image"] = img
                add_positions(d, poss)
                res.append(d)
            continue
        else:
            records = []
        if records:
            for record in records:
                d = copy.deepcopy(doc)
                tokenize(d, record["content"], eng)
                # 把record里的其他字段（除了content，例如标题、表头）更新到d中
                d.update({key: value for key, value in record.items() if key != "content"})
                if img is not None:
                    d["image"] = img
                add_positions(d, poss)
                res.append(d)
            continue
        if isinstance(rows, str):
            d = copy.deepcopy(doc)
            tokenize(d, rows, eng)
            d["content_with_weight"] = rows
            if img:
                #原始表格截图
                d["image"] = img
            if poss:
                add_positions(d, poss)
            res.append(d)
            continue
        de = "; " if eng else "； "
        #解析器返回的表格不是HTML字符串时就按照batch_size来分批处理
        for i in range(0, len(rows), batch_size):
            d = copy.deepcopy(doc)
            #每10行连接成一个chunk
            r = de.join(rows[i:i + batch_size])
            tokenize(d, r, eng)
            d["image"] = img
            add_positions(d, poss)
            res.append(d)
    return res

def not_bullet(line):
    patt = [
        r"0", r"[0-9]+ +[0-9~个只-]", r"[0-9]+\.{2,}"
    ]
    return any([re.match(r, line) for r in patt])

def bullets_category(sections):
    global BULLET_PATTERN
    hits = [0] * len(BULLET_PATTERN)
    for i, pro in enumerate(BULLET_PATTERN):
        for sec in sections:
            for p in pro:
                if re.match(p, sec) and not not_bullet(sec):
                    hits[i] += 1
                    break
    maxium = 0
    res = -1
    for i, h in enumerate(hits):
        if h <= maxium:
            continue
        res = i
        maxium = h
    return res


def title_frequency(bull, sections):
    bullets_size = len(BULLET_PATTERN[bull])
    levels = [bullets_size+1 for _ in range(len(sections))]
    if not sections or bull < 0:
        return bullets_size+1, levels

    for i, (txt, layout) in enumerate(sections):
        for j, p in enumerate(BULLET_PATTERN[bull]):
            if re.match(p, txt.strip()) and not not_bullet(txt):
                levels[i] = j
                break
        else:
            if re.search(r"(title|head)", layout) and not not_title(txt.split("@")[0]):
                levels[i] = bullets_size
    most_level = bullets_size+1
    for level, c in sorted(Counter(levels).items(), key=lambda x:x[1]*-1):
        if level <= bullets_size:
            most_level = level
            break
    return most_level, levels

def tokenize_chunks(chunks, doc, eng, pdf_parser=None):
    res = []
    # 封装为 Elasticsearch 文档。
    for chunk in chunks:
        if len(chunk.strip()) == 0:
            continue
        logging.debug("-- {}".format(chunk))
        # 复制公共元数据
        d = copy.deepcopy(doc)
        if pdf_parser:
            try:
                # 根据chunk中的坐标标签裁剪页面
                d["image"], poss = pdf_parser.crop(chunk, need_position=True)
                add_positions(d, poss)
                # 删除内部坐标标签
                chunk = pdf_parser.remove_tag(chunk)
            except NotImplementedError:
                pass
        #对正文进行分词，content_ltks、content_sm_ltks、content_with_weight
        tokenize(d, chunk, eng)

        #d中含有分词信息，文本的位置坐标信息
        res.append(d)
    return res


def docx_question_level(p, bull = -1):
    txt = re.sub(r"\u3000", " ", p.text).strip()
    if p.style.name.startswith('Heading'):
        return int(p.style.name.split(' ')[-1]), txt
    else:
        if bull < 0:
            return 0, txt
        for j, title in enumerate(BULLET_PATTERN[bull]):
            if re.match(title, txt):
                return j+1, txt
    return len(BULLET_PATTERN[bull]), txt
