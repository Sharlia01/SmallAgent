import os

import jieba
from elasticsearch import Elasticsearch
from openai import OpenAI


def generate_embedding(
    text: str,
    api_key: str = None,
    base_url: str = None,
    model_name: str = "text-embedding-v3",
    dimensions: int = 1024,
    encoding_format: str = "float",
):
    """调用 DashScope 生成向量，密钥只从参数或环境变量读取。"""
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    base_url = base_url or os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is not configured")

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        completion = client.embeddings.create(
            model=model_name,
            input=text,
            dimensions=dimensions,
            encoding_format=encoding_format,
        )
        return completion.data[0].embedding
    except Exception as error:
        print(f"OpenAI API 请求失败: {error}")
        return None


def retrieve_content(session_id: str, question: str):
    print("连接数据库")
    es_password = os.getenv("ELASTIC_PASSWORD")
    if not es_password:
        raise ValueError("ELASTIC_PASSWORD is not configured")

    es = Elasticsearch(
        [os.getenv("ES_HOST", "http://es01:9200")],
        basic_auth=(os.getenv("ELASTIC_USERNAME", "elastic"), es_password),
        verify_certs=False,
        timeout=600,
    )

    tokens = jieba.lcut(question)
    question_vector = generate_embedding(question)
    search_body = {
        "size": 3,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "should": [
                            {"terms": {"content_ltks": tokens}},
                            {"terms": {"content_with_weight": tokens}},
                        ]
                    }
                },
                "script": {
                    "source": """
                        cosineSimilarity(params.query_vector, 'q_1024_vec') + 1.0 + _score
                    """,
                    "params": {"query_vector": question_vector},
                },
            }
        },
        "highlight": {
            "fields": {
                "content_ltks": {},
                "content_with_weight": {},
            }
        },
        "track_total_hits": True,
    }

    print("session_id:" + session_id)
    response = es.search(index=session_id, body=search_body)
    threshold = 3.0
    filtered_results = [
        hit for hit in response["hits"]["hits"] if hit["_score"] >= threshold
    ]

    return [
        {
            "doc_id": hit["_source"].get("doc_id", "N/A"),
            "docnm": hit["_source"].get("docnm", "N/A"),
            "content_with_weight": hit["_source"].get(
                "content_with_weight", "N/A"
            ),
        }
        for hit in filtered_results
    ]


if __name__ == "__main__":
    example_session_id = "162264171a2b4d7f"
    example_question = "世运电路2023业绩增长原因分析"
    print(retrieve_content(example_session_id, example_question))
