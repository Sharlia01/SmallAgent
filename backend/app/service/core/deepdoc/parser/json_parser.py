# -*- coding: utf-8 -*-
#
# 第三方源码声明：以下保留原始版权与 Apache 2.0 许可条款。
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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

# 主要参考以下实现，并根据本项目需要进行了适配：
# https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/json.py

import json
from typing import Any

from service.core.rag.nlp import find_codec
class RAGFlowJsonParser:
    def __init__(
        self, max_chunk_size: int = 2000, min_chunk_size: int | None = None
    ):
        super().__init__()
        self.max_chunk_size = max_chunk_size * 2
        self.min_chunk_size = (
            min_chunk_size
            if min_chunk_size is not None
            else max(max_chunk_size - 200, 50)
        )

    def __call__(self, binary):
        encoding = find_codec(binary)
        txt = binary.decode(encoding, errors="ignore")
        json_data = json.loads(txt)
        chunks = self.split_json(json_data, True)   
        sections = [json.dumps(line, ensure_ascii=False) for line in chunks if line]
        return sections

    @staticmethod
    def _json_size(data: dict) -> int:
        """计算 JSON 对象序列化后的大小。"""
        return len(json.dumps(data, ensure_ascii=False))

    @staticmethod
    def _set_nested_dict(d: dict, path: list[str], value: Any) -> None:
        """按给定路径设置嵌套字典中的值。"""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def _list_to_dict_preprocessing(self, data: Any) -> Any:
        if isinstance(data, dict):
            # 逐个处理字典中的键值对。
            return {k: self._list_to_dict_preprocessing(v) for k, v in data.items()}
        elif isinstance(data, list):
            # 将列表转换为以元素下标为键的字典。
            return {
                str(i): self._list_to_dict_preprocessing(item)
                for i, item in enumerate(data)
            }
        else:
            # 递归结束条件：既不是字典也不是列表的元素直接返回。
            return data
        
    def _json_split(
        self,
        data,
        current_path: list[str] | None,
        chunks: list[dict] | None,
    ) -> list[dict]:
        """在保留原有结构的前提下，将 JSON 拆分为大小受限的字典片段。"""
        current_path = current_path or []
        chunks = chunks or [{}]
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = current_path + [key]
                chunk_size = self._json_size(chunks[-1])
                size = self._json_size({key: value})
                remaining = self.max_chunk_size - chunk_size

                if size < remaining:
                    # 将当前元素加入正在构建的片段。
                    self._set_nested_dict(chunks[-1], new_path, value)
                else:
                    if chunk_size >= self.min_chunk_size:
                        # 当前片段达到最小大小后，开始构建新片段。
                        chunks.append({})

                    # 遍历剩余元素。
                    self._json_split(value, new_path, chunks)
        else:
            # 处理单个元素。
            self._set_nested_dict(chunks[-1], current_path, data)
        return chunks

    def split_json(
        self,
        json_data,
        convert_lists: bool = False,
    ) -> list[dict]:
        """将 JSON 拆分为多个 JSON 片段。"""

        if convert_lists:
            preprocessed_data = self._list_to_dict_preprocessing(json_data)
            chunks = self._json_split(preprocessed_data, None, None)
        else:
            chunks = self._json_split(json_data, None, None)

        # 最后一个片段为空时将其删除。
        if not chunks[-1]:
            chunks.pop()
        return chunks

    def split_text(
        self,
        json_data: dict[str, Any],
        convert_lists: bool = False,
        ensure_ascii: bool = True,
    ) -> list[str]:
        """将 JSON 拆分为多个 JSON 格式字符串。"""

        chunks = self.split_json(json_data=json_data, convert_lists=convert_lists)

        # 将片段转换为字符串。
        return [json.dumps(chunk, ensure_ascii=ensure_ascii) for chunk in chunks]
