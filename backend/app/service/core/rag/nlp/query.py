#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import json
import re
from service.core.rag.utils.doc_store_conn import MatchTextExpr

from service.core.rag.nlp import rag_tokenizer, term_weight, synonym


class FulltextQueryer:
    def __init__(self):
        self.tw = term_weight.Dealer()
        self.syn = synonym.Dealer()
        self.query_fields = [
            "title_tks^10",
            "title_sm_tks^5",
            "important_kwd^30",
            "important_tks^20",
            "question_tks^20",
            "content_ltks^2",
            "content_sm_ltks",
        ]

    @staticmethod
    def subSpecialChar(line):
        return re.sub(r"([:\{\}/\[\]\-\*\"\(\)\|\+~\^])", r"\\\1", line).strip()

    @staticmethod
    def isChinese(line):
        arr = re.split(r"[ \t]+", line)
        if len(arr) <= 3:
            return True
        e = 0
        for t in arr:
            if not re.match(r"[a-zA-Z]+$", t):
                e += 1
        return e * 1.0 / len(arr) >= 0.7

    @staticmethod
    def rmWWW(txt):
        patts = [
            (
                r"是*(什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀|谁|哪位|哪个)是*",
                "",
            ),
            (r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "),
            (
                r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down|of|to|or|and|if) ",
                " ")
        ]
        for r, p in patts:
            txt = re.sub(r, p, txt, flags=re.IGNORECASE)
        return txt

    @staticmethod
    def _clean_question_text(txt):
        """统一大小写、繁简体和标点，再删除搜索价值较低的疑问词。"""
        normalized_text = re.sub(
            r"[ :|\r\n\t,，。？?/`!！&^%%()\[\]{}<>]+",
            " ",
            rag_tokenizer.tradi2simp(rag_tokenizer.strQ2B(txt.lower())),
        ).strip()
        return FulltextQueryer.rmWWW(normalized_text)

    def _clean_english_weighted_tokens(self, tokens):
        """计算英文词权重，并删除不适合放入 ES 查询语句的字符。"""
        weighted_tokens = self.tw.weights(tokens, preprocess=False)
        weighted_tokens = [
            (re.sub(r"[ \\\"'^]", "", token), weight)
            for token, weight in weighted_tokens
        ]
        weighted_tokens = [
            (re.sub(r"^[a-z0-9]$", "", token), weight)
            for token, weight in weighted_tokens
            if token
        ]
        weighted_tokens = [
            (re.sub(r"^[\+-]", "", token), weight)
            for token, weight in weighted_tokens
            if token
        ]
        return [
            (token.strip(), weight)
            for token, weight in weighted_tokens
            if token.strip()
        ]

    def _build_english_query(self, txt):
        """为英文或其他非中文问题生成关键词查询表达式。"""
        txt = FulltextQueryer.rmWWW(txt)
        tokens = rag_tokenizer.tokenize(txt).split()
        keywords = [token for token in tokens if token]
        weighted_tokens = self._clean_english_weighted_tokens(tokens)

        # 同义词参与搜索，但权重只有原词的四分之一。
        synonym_queries = []
        for token, weight in weighted_tokens:
            token_synonyms = self.syn.lookup(token)
            token_synonyms = rag_tokenizer.tokenize(
                " ".join(token_synonyms)
            ).split()
            keywords.extend(token_synonyms)
            weighted_synonyms = [
                '"{}"^{:.4f}'.format(synonym, weight / 4.)
                for synonym in token_synonyms
                if synonym.strip()
            ]
            synonym_queries.append(" ".join(weighted_synonyms))

        # 每个原词建立查询后，再补充相邻双词短语查询。
        query_parts = [
            "({}^{:.4f}".format(token, weight)
            + " {})".format(synonym_query)
            for (token, weight), synonym_query in zip(
                weighted_tokens, synonym_queries
            )
            if token and not re.match(r"[.^+\(\)-]", token)
        ]
        for index in range(1, len(weighted_tokens)):
            previous_token, previous_weight = weighted_tokens[index - 1]
            current_token, current_weight = weighted_tokens[index]
            if not previous_token.strip() or not current_token.strip():
                continue
            query_parts.append(
                '"%s %s"^%.4f'
                % (
                    previous_token,
                    current_token,
                    max(previous_weight, current_weight) * 2,
                )
            )

        if not query_parts:
            query_parts.append(txt)
        query = " ".join(query_parts)
        return MatchTextExpr(self.query_fields, query, 100), keywords

    @staticmethod
    def _needs_fine_grained_tokenization(token):
        """判断中文词是否需要继续拆成更细的词。"""
        if len(token) < 3:
            return False
        if re.match(r"[0-9a-z\.\+#_\*-]+$", token):
            return False
        return True

    def _get_fine_grained_tokens(self, token):
        """把较长中文词细分，并清理、转义各个子词。"""
        if not self._needs_fine_grained_tokenization(token):
            return []

        fine_tokens = rag_tokenizer.fine_grained_tokenize(token).split()
        fine_tokens = [
            re.sub(
                r"[ ,\./;'\[\]\\`~!@#$%\^&\*\(\)=\+_<>\?:\"\{\}\|，。；‘’【】、！￥……（）——《》？：“”-]+",
                "",
                fine_token,
            )
            for fine_token in fine_tokens
        ]
        fine_tokens = [
            FulltextQueryer.subSpecialChar(fine_token)
            for fine_token in fine_tokens
            if len(fine_token) > 1
        ]
        return [
            fine_token for fine_token in fine_tokens if len(fine_token) > 1
        ]

    def _build_chinese_token_query(self, token, weight, keywords):
        """处理一个中文词，返回查询片段以及是否达到关键词上限。"""
        fine_tokens = self._get_fine_grained_tokens(token)

        if len(keywords) < 32:
            keywords.append(re.sub(r"[ \"']+", "", token))
            keywords.extend(fine_tokens)

        token_synonyms = self.syn.lookup(token)
        token_synonyms = [
            FulltextQueryer.subSpecialChar(synonym)
            for synonym in token_synonyms
        ]
        if len(keywords) < 32:
            keywords.extend([synonym for synonym in token_synonyms if synonym])

        synonym_queries = [
            rag_tokenizer.fine_grained_tokenize(synonym)
            for synonym in token_synonyms
            if synonym
        ]
        synonym_queries = [
            f'"{synonym}"' if synonym.find(" ") > 0 else synonym
            for synonym in synonym_queries
        ]

        # 保留原规则：达到约 32 个关键词后停止处理当前词组。
        if len(keywords) >= 32:
            return None, True

        token_query = FulltextQueryer.subSpecialChar(token)
        if token_query.find(" ") > 0:
            token_query = '"%s"' % token_query
        if synonym_queries:
            token_query = "(%s OR (%s)^0.2)" % (
                token_query,
                " ".join(synonym_queries),
            )
        if fine_tokens:
            token_query = '%s OR "%s" OR ("%s"~2)^0.5' % (
                token_query,
                " ".join(fine_tokens),
                " ".join(fine_tokens),
            )
        if not token_query.strip():
            return None, False
        return (token_query, weight), False

    def _build_chinese_group_query(self, text_group, keywords):
        """把一个中文词组转换成带权重的查询片段。"""
        weighted_tokens = self.tw.weights([text_group])
        group_synonyms = self.syn.lookup(text_group)
        if group_synonyms and len(keywords) < 32:
            keywords.extend(group_synonyms)
        logging.debug(json.dumps(weighted_tokens, ensure_ascii=False))

        token_queries = []
        for token, weight in sorted(
            weighted_tokens, key=lambda item: item[1] * -1
        ):
            token_query, reached_keyword_limit = self._build_chinese_token_query(
                token, weight, keywords
            )
            if reached_keyword_limit:
                break
            if token_query:
                token_queries.append(token_query)

        group_query = " ".join(
            [f"({query})^{weight}" for query, weight in token_queries]
        )

        # 多词词组额外添加一个允许相隔两个位置的短语查询。
        if len(weighted_tokens) > 1:
            group_query += ' ("%s"~2)^1.5' % rag_tokenizer.tokenize(text_group)

        synonym_query = " OR ".join(
            [
                '"%s"'
                % rag_tokenizer.tokenize(
                    FulltextQueryer.subSpecialChar(synonym)
                )
                for synonym in group_synonyms
            ]
        )
        if synonym_query and group_query:
            group_query = f"({group_query})^5 OR ({synonym_query})^0.7"
        return group_query

    def _build_chinese_query(self, txt, min_match):
        """为中文问题生成全文检索表达式。"""
        txt = FulltextQueryer.rmWWW(txt)
        group_queries = []
        keywords = []

        # 最多处理前 256 个词组，避免查询语句无限膨胀。
        for text_group in self.tw.split(txt)[:256]:
            if not text_group:
                continue
            keywords.append(text_group)
            group_queries.append(
                self._build_chinese_group_query(text_group, keywords)
            )

        if group_queries:
            query = " OR ".join(
                [
                    f"({group_query})"
                    for group_query in group_queries
                    if group_query
                ]
            )
            return MatchTextExpr(
                self.query_fields,
                query,
                100,
                {"minimum_should_match": min_match},
            ), keywords
        return None, keywords

    def question(self, txt, tbl="qa", min_match: float = 0.6):
        """清洗用户问题，再按语言交给对应的查询构造函数。"""
        # tbl 是为兼容旧调用保留的参数，目前没有参与处理。
        cleaned_text = self._clean_question_text(txt)
        if not self.isChinese(cleaned_text):
            return self._build_english_query(cleaned_text)
        return self._build_chinese_query(cleaned_text, min_match)

    def hybrid_similarity(self, avec, bvecs, atks, btkss, tkweight=0.3, vtweight=0.7):
        from sklearn.metrics.pairwise import cosine_similarity as CosineSimilarity
        import numpy as np

        sims = CosineSimilarity([avec], bvecs)
        tksim = self.token_similarity(atks, btkss)
        return np.array(sims[0]) * vtweight + np.array(tksim) * tkweight, tksim, sims[0]

    def token_similarity(self, atks, btkss):
        def toDict(tks):
            d = {}
            if isinstance(tks, str):
                tks = tks.split()
            for t, c in self.tw.weights(tks, preprocess=False):
                if t not in d:
                    d[t] = 0
                d[t] += c
            return d

        atks = toDict(atks)
        btkss = [toDict(tks) for tks in btkss]
        return [self.similarity(atks, btks) for btks in btkss]

    def similarity(self, qtwt, dtwt):
        if isinstance(dtwt, type("")):
            dtwt = {t: w for t, w in self.tw.weights(self.tw.split(dtwt), preprocess=False)}
        if isinstance(qtwt, type("")):
            qtwt = {t: w for t, w in self.tw.weights(self.tw.split(qtwt), preprocess=False)}
        s = 1e-9
        for k, v in qtwt.items():
            if k in dtwt:
                s += v  # * dtwt[k]
        q = 1e-9
        for k, v in qtwt.items():
            q += v
        return s / q

    def paragraph(self, content_tks: str, keywords: list = [], keywords_topn=30):
        if isinstance(content_tks, str):
            content_tks = [c.strip() for c in content_tks.strip() if c.strip()]
        tks_w = self.tw.weights(content_tks, preprocess=False)

        keywords = [f'"{k.strip()}"' for k in keywords]
        for tk, w in sorted(tks_w, key=lambda x: x[1] * -1)[:keywords_topn]:
            tk_syns = self.syn.lookup(tk)
            tk_syns = [FulltextQueryer.subSpecialChar(s) for s in tk_syns]
            tk_syns = [rag_tokenizer.fine_grained_tokenize(s) for s in tk_syns if s]
            tk_syns = [f"\"{s}\"" if s.find(" ") > 0 else s for s in tk_syns]
            tk = FulltextQueryer.subSpecialChar(tk)
            if tk.find(" ") > 0:
                tk = '"%s"' % tk
            if tk_syns:
                tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)
            if tk:
                keywords.append(f"{tk}^{w}")

        return MatchTextExpr(self.query_fields, " ".join(keywords), 100,
                             {"minimum_should_match": min(3, len(keywords) / 10)})
