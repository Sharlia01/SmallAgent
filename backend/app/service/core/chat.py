from openai import OpenAI
import os
import json
import redis
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from service.session_access import DEFAULT_SESSION_NAME
from utils.database import get_db
from fastapi import HTTPException
from utils import logger
from dotenv import load_dotenv

load_dotenv()

# 聊天回答使用的模型，以及快速解析文档的长度限制。
# 把这些“可调整的数字”集中放在这里，后面修改时不必进入业务代码里寻找。
CHAT_MODEL = "deepseek-v4-pro"
MAX_QUICK_PARSE_PROMPT_LENGTH = 4000
MAX_QUICK_PARSE_DOCUMENT_LENGTH = 2000

# Redis 客户端初始化
def get_redis_client():
    """获取 Redis 客户端"""
    redis_host = os.getenv('REDIS_HOST', 'redis')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    redis_db = int(os.getenv('REDIS_DB', 0))
    return redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)

def get_quick_parse_content(session_id: str) -> str:
    """从 Redis 获取快速解析的文档内容"""
    try:
        redis_client = get_redis_client()
        content = redis_client.get(session_id)
        if content:
            logger.info(f"从 Redis 获取到快速解析内容，session_id: {session_id}, 长度: {len(content)}")
            return content
        else:
            logger.info(f"Redis 中未找到快速解析内容，session_id: {session_id}")
            return None
    except Exception as e:
        logger.error(f"从 Redis 获取快速解析内容失败: {str(e)}")
        return None

def generate_recommended_questions(user_question, retrieved_content=None, session_id=None):
    """
    根据用户提问生成相关推荐问题。

    :param user_question: 用户提问
    :param retrieved_content: 检索到的内容（可选，用于判断是否有相关文档）
    :param session_id: 会话ID（可选）
    :return: 推荐问题列表
    """
    # 判断是否有文档上下文
    has_documents = bool(retrieved_content and len(retrieved_content) > 0)
    
    # 获取文档主题信息（简化版）
    document_topics = []
    if has_documents:
        # 只获取文档名称作为主题参考，避免内容过长
        document_names = list(set([ref.get('document_name', '') for ref in retrieved_content if ref.get('document_name')]))
        document_topics = document_names[:3]  # 最多3个文档名称

   # 构造优化后的提示词
    context_info = ""
    if has_documents and document_topics:
        context_info = f"当前对话基于这些文档：{', '.join(document_topics)}"
    
    prompt = f"""
你是一个智能助手，请基于用户的问题生成3个相关的推荐问题，帮助用户更深入地探索这个话题。

用户问题：{user_question}
{context_info}

要求：
1. 生成的问题应该与用户问题相关，但从不同角度深入
2. 问题要具体、有价值，能够引导用户获得更多有用信息
3. 如果有文档上下文，可以围绕文档主题生成相关问题
4. 返回JSON格式，包含recommended_questions数组

输出格式：
{{
  "recommended_questions": [
    "具体问题1",
    "具体问题2", 
    "具体问题3"
  ]
}}

请直接返回JSON，不要包含其他文字。
    """
    
    try:
        # 调用大模型生成推荐问题
        client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url=os.getenv("DASHSCOPE_BASE_URL")
            )
        completion = client.chat.completions.create(
            model="qwen2.5-7b-instruct",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            stream=False,
            timeout=30,  # 添加超时设置
        )

        # 提取生成的推荐问题
        if completion.choices:
            response = completion.choices[0].message.content
            logger.info(f"大模型返回的推荐问题原始响应: {response}")
            
            try:
                # 清理响应内容，去掉可能的markdown代码块标识符
                import re
                cleaned_response = response.strip()
                
                # 使用正则表达式去掉```json开头和```结尾
                json_pattern = r'^```(?:json)?\s*\n?(.*?)\n?```$'
                match = re.search(json_pattern, cleaned_response, re.DOTALL | re.IGNORECASE)
                
                if match:
                    cleaned_response = match.group(1).strip()
                    logger.info(f"检测到markdown代码块，已清理")
                
                logger.info(f"清理后的响应内容: {cleaned_response}")
                
                # 解析 JSON 响应
                response_json = json.loads(cleaned_response)
                recommended_questions = response_json.get("recommended_questions", [])
                logger.info(f"解析后的推荐问题: {recommended_questions}")
                
                # 验证推荐问题格式
                if isinstance(recommended_questions, list) and len(recommended_questions) > 0:
                    return recommended_questions
                else:
                    logger.warning("推荐问题格式不正确或为空")
                    return []
                    
            except json.JSONDecodeError as e:
                logger.error(f"解析推荐问题JSON失败: {str(e)}")
                logger.error(f"原始响应内容: {response}")
                logger.error(f"清理后内容: {cleaned_response if 'cleaned_response' in locals() else '未处理'}")
                return []
        else:
            logger.warning("大模型没有返回任何选择")
            return []
            
    except Exception as e:
        logger.error(f"调用大模型生成推荐问题时发生错误: {str(e)}")
        return []

def generate_session_name(user_question):
    prompt = f"""
    请根据以下用户提问，生成一个简洁且具有代表性的会话名称：
    用户提问：{user_question}

    要求：
    1. 会话名称应简洁明了，能够概括用户提问的主题。
    2. 返回一个 JSON 对象，包含一个字段 "session_name"，值为生成的会话名称。

    输出格式示例：
    {{
      "session_name": "会话名称内容"
    }}

    请严格按照上述格式返回 JSON 对象。
    """
    
    # 调用大模型生成会话名称
    try:
        client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url=os.getenv("DASHSCOPE_BASE_URL")
            )
        completion = client.chat.completions.create(
            model="qwen2.5-72b-instruct",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            stream=False,
        )

        # 提取生成的会话名称
        if completion.choices:
            response = completion.choices[0].message.content
            try:
                # 解析 JSON 响应
                response_json = json.loads(response)
                session_name = response_json.get("session_name")
                print("生成的会话名称：\n")
                print(session_name)
                return session_name
            except json.JSONDecodeError:
                print("Failed to parse JSON response.")
                return user_question
    except Exception as e:
        print(f"An error occurred: {e}")
        return user_question


def write_chat_to_db(session_id: str, user_question: str, model_answer: str, retrieval_content, recommended_questions, think ):
    """
    将对话数据写入数据库。

    :param session_id: 会话 ID
    :param user_question: 用户问题
    :param model_answer: 大模型的回答
    :param retrieval_content: 检索内容
    """
    db = next(get_db())  # 获取数据库会话
    try:
        documents_json = json.dumps(retrieval_content, ensure_ascii=False)

        db.execute(
            text(
                """
                INSERT INTO messages (session_id, user_question, model_answer, documents, recommended_questions, think )
                VALUES (:session_id, :user_question, :model_answer, :documents, :recommended_questions, :think)
                """
            ),
            {
                "session_id": session_id,
                "user_question": user_question,
                "model_answer": model_answer,
                "documents": documents_json,
                "recommended_questions": recommended_questions,
                "think": think,
            }
        )
        db.commit()
        logger.info("对话数据插入成功。。。")
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write to database: {str(e)}"
        )
    finally:
        db.close()

def update_session_name(session_id: str, question: str, user_id: int):
    """
    根据 session_id 查数据库的表 sessions，有的话直接跳过，没有的话先生成 session_name，再插入。

    :param session_id: 会话 ID
    :param user_id: 用户 ID
    """
    db = next(get_db())  # 获取数据库会话
    try:
        # 查询 sessions 表中是否存在该 session_id
        query_result = db.execute(
            text(
                """
                SELECT session_name
                FROM sessions
                WHERE session_id = :session_id AND user_id = :user_id
                """
            ),
            {"session_id": session_id, "user_id": user_id}
        ).fetchone()

        if query_result:
            if query_result.session_name == DEFAULT_SESSION_NAME and question:
                session_name = generate_session_name(question)
                db.execute(
                    text(
                        """
                        UPDATE sessions
                        SET session_name = :session_name, updated_at = CURRENT_TIMESTAMP
                        WHERE session_id = :session_id AND user_id = :user_id
                        """
                    ),
                    {
                        "session_id": session_id,
                        "user_id": user_id,
                        "session_name": session_name,
                    },
                )
                db.commit()
                logger.info(f"Session {session_id} name updated.")
            else:
                logger.info(f"Session {session_id} already has a name, skipping.")
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Database operation failed: {str(e)}"
        )
    finally:
        db.close()

def _format_reference_content(retrieved_content, quick_parse_content):
    """把两种来源的资料整理成一段带编号的文本，供大模型阅读。"""
    reference_sections = []
    next_reference_id = 1

    # 第一种来源：RAG 从知识库中搜索出来的 chunk。
    knowledge_base_references = []
    for reference in retrieved_content or []:
        content = reference.get("content_with_weight")
        if not content:
            continue

        knowledge_base_references.append(
            f"[{next_reference_id}] {content}"
        )
        next_reference_id += 1

    if knowledge_base_references:
        reference_sections.append(
            "**知识库内容：**\n" + "\n".join(knowledge_base_references)
        )

    # 第二种来源：用户在当前会话中上传后，被“快速解析”的文档。
    # 这里只放前 4000 个字符进提示词，避免一次请求携带太多文字。
    if quick_parse_content and quick_parse_content.strip():
        truncated_content = quick_parse_content[:MAX_QUICK_PARSE_PROMPT_LENGTH]
        if len(quick_parse_content) > MAX_QUICK_PARSE_PROMPT_LENGTH:
            truncated_content += "...(内容已截断)"

        reference_sections.append(
            "**当前会话文档内容：**\n"
            f"[{next_reference_id}] {truncated_content}"
        )

    if not reference_sections:
        return "暂无相关参考内容"

    return "\n\n".join(reference_sections)


def _build_answer_prompt(question, formatted_references):
    """把参考资料和用户问题装进最终发送给大模型的提示词。"""
    return f"""
你是一个专业的智能助手，擅长基于提供的参考资料回答用户问题。请遵循以下原则：

**回答要求：**
1. 优先基于参考内容回答，确保答案准确可靠
2. 在回答中，每一块内容都必须标注引用的来源，格式为：##引用编号$$。例如：##1$$ 表示引用自第1条参考内容。
3. 如果参考内容不足以完全回答问题，可以结合常识补充，但需明确区分
4. 回答要条理清晰、语言自然流畅
5. 如果没有相关参考内容，请诚实说明并提供一般性建议

**参考内容：**
{formatted_references}

**用户问题：**
{question}

请基于以上信息提供专业、准确的回答。
    """.strip()


def _split_quick_parse_content(content):
    """按段落把快速解析文本拆成较小的块，方便前端展示引用资料。"""
    if len(content) <= MAX_QUICK_PARSE_DOCUMENT_LENGTH:
        return [content]

    paragraphs = [paragraph.strip() for paragraph in content.split("\n") if paragraph.strip()]
    content_chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # candidate 表示“如果把当前段落也放进来”，新内容块会是什么样。
        candidate = f"{current_chunk}\n{paragraph}" if current_chunk else paragraph
        if len(candidate) <= MAX_QUICK_PARSE_DOCUMENT_LENGTH:
            current_chunk = candidate
            continue

        if current_chunk:
            content_chunks.append(current_chunk)
        current_chunk = paragraph

    if current_chunk:
        content_chunks.append(current_chunk)

    return content_chunks


def _build_response_documents(session_id, retrieved_content, quick_parse_content):
    """构造先发给前端的 documents 列表，供前端展示引用来源。"""
    # list(...) 会创建一个新列表，避免 append 时修改调用者原来的列表。
    all_documents = list(retrieved_content or [])

    if not quick_parse_content:
        return all_documents

    content_chunks = _split_quick_parse_content(quick_parse_content)
    for index, content_chunk in enumerate(content_chunks):
        document_id = f"quick_parse_{session_id}_{index}"
        document_name = (
            f"当前会话文档-第{index + 1}部分"
            if len(content_chunks) > 1
            else "当前会话文档"
        )
        all_documents.append(
            {
                "document_id": document_id,
                "document_name": document_name,
                "content_with_weight": content_chunk,
                "id": document_id,
                "positions": [],
            }
        )

    logger.info(f"快速解析内容已添加到文档列表，共{len(content_chunks)}个部分")
    return all_documents


def _create_streaming_completion(prompt):
    """向大模型发起请求，并返回可以逐块读取的流式响应。"""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
    )
    return client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )


def _make_sse_event(event_name, data):
    """把 Python 数据转换成浏览器能识别的 SSE 文本格式。"""
    # [DONE] 本身就是字符串；普通消息则需要先转成 JSON 字符串。
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event_name}\ndata: {payload}\n\n"


def _read_stream_chunk(chunk):
    """从模型返回的一小块数据中，取出结束标志、正文和思考内容。"""
    if not chunk.choices:
        return None, "", ""

    choice = chunk.choices[0]
    delta = choice.delta

    # getattr(对象, 属性名, 默认值) 是 Python 的安全取属性方式。
    # 某些模型没有 reasoning_content 属性，使用 getattr 就不会因此报错。
    answer_text = getattr(delta, "content", None) or ""
    thinking_text = getattr(delta, "reasoning_content", None) or ""
    return choice.finish_reason, answer_text, thinking_text


def _generate_recommended_questions_safely(question, retrieved_content, session_id):
    """生成推荐问题；失败时返回空列表，不影响主要回答。"""
    try:
        logger.info("开始生成推荐问题...")
        questions = generate_recommended_questions(
            question,
            retrieved_content,
            session_id,
        )
        logger.info(f"推荐问题生成结果: {questions}")
        return questions or []
    except Exception as error:
        logger.error(f"生成推荐问题失败: {str(error)}")
        return []


def get_chat_completion(session_id, question, retrieved_content, user_id: int):
    """
    流式生成聊天回答，并把结果包装成 SSE 事件交给前端。

    这个函数中使用了 yield，所以它不是一次性返回最终答案的普通函数，
    而是一个“生成器”。每执行一次 yield，StreamingResponse 就能立即把
    当前这小段内容发给前端，随后再继续执行函数的剩余部分。

    :param session_id: 当前会话 ID
    :param question: 用户提出的问题
    :param retrieved_content: RAG 从知识库检索到的文档 chunk 列表
    :param user_id: 当前用户 ID
    :return: 逐个产生 SSE 格式字符串的生成器
    """
    try:
        # 第一步：准备大模型需要阅读的资料和问题。
        quick_parse_content = get_quick_parse_content(session_id)
        formatted_references = _format_reference_content(
            retrieved_content,
            quick_parse_content,
        )
        prompt = _build_answer_prompt(question, formatted_references)

        # 第二步：真正调用大模型。stream=True 使回答可以一小块一小块返回。
        completion = _create_streaming_completion(prompt)

        # 第三步：先把引用资料发给前端，前端可以据此展示“答案来源”。
        all_documents = _build_response_documents(
            session_id,
            retrieved_content,
            quick_parse_content,
        )
        yield _make_sse_event("message", {"documents": all_documents})

        # 用列表暂存每一小段文本，最后再 join 成完整字符串。
        # 这比每次使用 answer = answer + 新文字更适合流式累加。
        answer_parts = []
        thinking_parts = []

        # 第四步：不断读取模型返回的小块，并立即转发给前端。
        for chunk in completion:
            finish_reason, answer_text, thinking_text = _read_stream_chunk(chunk)

            # finish_reason 不是 None，表示模型已经停止生成。
            if finish_reason is not None:
                break

            if answer_text:
                answer_parts.append(answer_text)
                yield _make_sse_event(
                    "message",
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "thinking": False,
                    },
                )
            elif thinking_text:
                thinking_parts.append(thinking_text)
                yield _make_sse_event(
                    "message",
                    {
                        "role": "assistant",
                        "content": thinking_text,
                        "thinking": True,
                    },
                )

        # 第五步：流式回答结束后，拼出完整正文和完整思考过程。
        model_answer = "".join(answer_parts)
        think = "".join(thinking_parts)

        recommended_questions = _generate_recommended_questions_safely(
            question,
            retrieved_content,
            session_id,
        )
        if recommended_questions:
            yield _make_sse_event(
                "message",
                {"recommended_questions": recommended_questions},
            )
            logger.info("推荐问题已发送给前端")
        else:
            logger.warning("推荐问题生成为空")

        # 告诉前端：本次流式回答已经全部发送完毕。
        yield _make_sse_event("end", "[DONE]")

        # 最后保存完整聊天记录，并在首次聊天时生成会话名称。
        logger.info(f"模型回答生成完成，回答长度: {len(model_answer)}")
        write_chat_to_db(
            session_id,
            question,
            model_answer,
            all_documents,
            recommended_questions,
            think,
        )
        update_session_name(session_id, question, user_id)

    except Exception as error:
        # 即使后端出错，也按 SSE 格式告诉前端，而不是直接中断连接。
        logger.error(f"流式聊天处理失败: {str(error)}")
        yield _make_sse_event(
            "error",
            {
                "role": "error",
                "content": str(error),
            },
        )
