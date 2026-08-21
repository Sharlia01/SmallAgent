from openai import OpenAI
import dashscope
import numpy as np
from typing import List

import os
from dotenv import load_dotenv
load_dotenv()

def get_chat_completion_block(session_id, question, references):
    """
    结合知识库内容生成回答，并在回答中标注引用来源。

    :param question: 用户问题
    :param references: 知识库内容，格式为 [{"id": 1, "content": "..."}, ...]
    :return: 模型的回答
    """
    try:
        
        # 初始化 OpenAI 客户端
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL")
        )
        # 格式化参考内容
        formatted_references = "\n".join([f"[{ref['id']}] {ref['content']}" for ref in references])
    
        # 构造提示词
        
    
        # 调用模型生成回答
        completion = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
    
        return completion.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

def rerank_similarity(query, texts):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    texts = list(texts)

    # === 重排前：输入顺序（按 sres.ids 排序） ===
    print(f"\n[rerank_similarity] query={query[:60]!r} n_texts={len(texts)}")
    print(f"[rerank_similarity] 重排前（输入顺序）:")
    for i, t in enumerate(texts):
        print(f"  texts[{i}]: {t[:60]!r}")

    # 直接调 DashScope rerank,用 result.index 回填到原序数组
    resp = dashscope.TextReRank.call(
        model="gte-rerank",
        top_n=len(texts),
        query=query,
        documents=texts,
        api_key=api_key,
    )

    scores = np.zeros(len(texts), dtype=float)
    for r in resp.output.results:
        scores[r.index] = r.relevance_score

    # === 重排后：按 rerank 分数降序的新排名 ===
    print(f"[rerank_similarity] 重排后（按 rerank 分数降序）:")
    for rank, r in enumerate(resp.output.results):
        snippet = texts[r.index][:60]
        print(f"  rank {rank}: orig_idx={r.index}  score={r.relevance_score:.4f}  text={snippet!r}")
    print(f"[rerank_similarity] 对齐后 scores（与 texts 同序）={np.round(scores, 4).tolist()}\n")

    return scores, None




def generate_embedding(text: str | List[str], api_key: str = None, base_url: str = None, model_name: str = "text-embedding-v3", dimensions: int = 1024, encoding_format: str = "float", max_batch_size: int = 10):
    """
    生成文本的向量嵌入
    
    Args:
        text: 单个文本或文本列表
        api_key: API密钥
        base_url: API基础URL
        model_name: 模型名称
        dimensions: 向量维度
        encoding_format: 编码格式
        max_batch_size: 最大批量大小，默认为10（阿里云DashScope限制）
    
    Returns:
        单个文本时返回向量，文本列表时返回向量列表
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL")    

    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # 如果是单个文本，直接处理
    if isinstance(text, str):
        try:
            completion = client.embeddings.create(
                model=model_name,
                input=text,
                dimensions=dimensions,
                encoding_format=encoding_format
            )
            return completion.data[0].embedding
        except Exception as e:
            print(f"OpenAI API 请求失败: {e}")
            return None
    
    # 如果是文本列表，需要分批处理
    if isinstance(text, list):
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(text), max_batch_size):
            batch = text[i:i + max_batch_size]
            
            try:
                completion = client.embeddings.create(
                    model=model_name,
                    input=batch,
                    dimensions=dimensions,
                    encoding_format=encoding_format
                )
                
                # 收集这一批的向量
                batch_embeddings = [item.embedding for item in completion.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"OpenAI API 批量请求失败 (batch {i//max_batch_size + 1}): {e}")
                # 如果批量失败，为这一批添加空向量
                all_embeddings.extend([None] * len(batch))
        
        return all_embeddings


# 示例调用
if __name__ == "__main__":
    # 示例调用
    question = "法国的首都是哪里？"
    references = [
        {"id": 1, "content": "法国的首都是巴黎。"},
        {"id": 2, "content": "巴黎是欧洲的文化中心之一。"},
    ]
    session_id = "sd"
    
    response = get_chat_completion_block(session_id, question, references)
    print(response)