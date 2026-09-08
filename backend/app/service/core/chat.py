from openai import OpenAI
import os
import json
import redis
import time
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from service.session_access import DEFAULT_SESSION_NAME
from utils.database import get_db
from fastapi import HTTPException
from utils import logger
from dotenv import load_dotenv
from agent.orchestrator import AgentOrchestrator
from agent.tools.rag_search import RagSearchTool
from agent.tools.web_search import WebSearchTool

load_dotenv()

# 聊天回答使用的模型，以及快速解析文档的长度限制。
# 把这些“可调整的数字”集中放在这里，后面修改时不必进入业务代码里寻找。
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-v4-pro")
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen3.7-flash-2026-07-15")
AGENT_MAX_STEPS = 3


def _positive_int_env(name: str, default: int, maximum: int) -> int:
    """Read a bounded positive integer without breaking startup on bad config."""
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return min(value, maximum) if value > 0 else default


CHAT_HISTORY_MAX_TURNS = _positive_int_env("CHAT_HISTORY_MAX_TURNS", 8, 20)
CHAT_HISTORY_MAX_CHARS = _positive_int_env(
    "CHAT_HISTORY_MAX_CHARS",
    12000,
    50000,
)

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
            return content
        return None
    except Exception as e:
        logger.error(f"从 Redis 获取快速解析内容失败: {str(e)}")
        return None


def _truncate_history_text(content: str, limit: int) -> str:
    """Keep both ends of an oversized message within an exact char limit."""
    if len(content) <= limit:
        return content
    marker = "\n...(历史消息已截断)...\n"
    if limit <= len(marker):
        return content[:limit]

    available = limit - len(marker)
    head_length = (available * 2) // 3
    tail_length = available - head_length
    return f"{content[:head_length]}{marker}{content[-tail_length:]}"


def _fit_history_turn(
    question: str,
    answer: str,
    budget: int,
) -> tuple[str, str]:
    """Fit one user/assistant pair while preserving space for both sides."""
    if not question:
        return "", _truncate_history_text(answer, budget)
    if not answer:
        return _truncate_history_text(question, budget), ""

    question_budget = min(len(question), budget // 2)
    answer_budget = min(len(answer), budget - question_budget)
    remaining = budget - question_budget - answer_budget

    if remaining:
        extra_question = min(len(question) - question_budget, remaining)
        question_budget += extra_question
        remaining -= extra_question
    if remaining:
        answer_budget += min(len(answer) - answer_budget, remaining)

    return (
        _truncate_history_text(question, question_budget),
        _truncate_history_text(answer, answer_budget),
    )


def load_conversation_history(
    session_id: str,
    user_id: int,
) -> list[dict[str, str]]:
    """Load recent turns owned by the user and apply a total char budget."""
    db = None
    try:
        db = next(get_db())
        rows = db.execute(
            text(
                """
                SELECT m.user_question, m.model_answer
                FROM messages AS m
                INNER JOIN sessions AS s ON s.session_id = m.session_id
                WHERE m.session_id = :session_id AND s.user_id = :user_id
                ORDER BY m.created_at DESC, m.message_id DESC
                LIMIT :max_turns
                """
            ),
            {
                "session_id": session_id,
                "user_id": user_id,
                "max_turns": CHAT_HISTORY_MAX_TURNS,
            },
        ).fetchall()

        selected_turns = []
        remaining_chars = CHAT_HISTORY_MAX_CHARS
        for row in rows:
            user_question = str(row.user_question or "").strip()
            model_answer = str(row.model_answer or "").strip()
            if not user_question and not model_answer:
                continue

            turn_length = len(user_question) + len(model_answer)
            if turn_length <= remaining_chars:
                selected_turns.append((user_question, model_answer))
                remaining_chars -= turn_length
                continue

            # The newest turn is still useful even when it alone exceeds the
            # budget. Older turns are dropped once the budget is exhausted.
            if not selected_turns and remaining_chars > 0:
                selected_turns.append(
                    _fit_history_turn(
                        user_question,
                        model_answer,
                        remaining_chars,
                    )
                )
            break

        history = []
        for user_question, model_answer in reversed(selected_turns):
            if user_question:
                history.append({"role": "user", "content": user_question})
            if model_answer:
                history.append({"role": "assistant", "content": model_answer})
        return history
    except Exception as error:
        # History improves continuity but must not make a fresh question fail.
        logger.exception("读取多轮对话历史失败，继续按单轮处理: %s", error)
        return []
    finally:
        if db is not None:
            db.close()

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
            model="qwen3.7-flash-2026-07-15",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            stream=False,
            timeout=30,  # 添加超时设置
        )

        # 提取生成的推荐问题
        if completion.choices:
            response = completion.choices[0].message.content
            
            try:
                # 清理响应内容，去掉可能的markdown代码块标识符
                import re
                cleaned_response = response.strip()
                
                # 使用正则表达式去掉```json开头和```结尾
                json_pattern = r'^```(?:json)?\s*\n?(.*?)\n?```$'
                match = re.search(json_pattern, cleaned_response, re.DOTALL | re.IGNORECASE)
                
                if match:
                    cleaned_response = match.group(1).strip()
                
                # 解析 JSON 响应
                response_json = json.loads(cleaned_response)
                recommended_questions = response_json.get("recommended_questions", [])
                
                # 验证推荐问题格式
                if isinstance(recommended_questions, list) and len(recommended_questions) > 0:
                    return recommended_questions
                else:
                    return []
                    
            except json.JSONDecodeError as e:
                logger.error(f"解析推荐问题JSON失败: {str(e)}")
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
            model="qwen3.7-flash-2026-07-15",
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

def _create_agent_orchestrator(user_id: int) -> AgentOrchestrator:
    """Create one request-scoped two-model Agent orchestration pipeline."""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
    )
    rag_tool = RagSearchTool(user_id=user_id)
    return AgentOrchestrator(
        planner_client=client,
        planner_model=AGENT_MODEL,
        answer_client=client,
        answer_model=CHAT_MODEL,
        tools=[rag_tool, WebSearchTool()],
        fallback_tool=rag_tool,
        max_steps=AGENT_MAX_STEPS,
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
        questions = generate_recommended_questions(
            question,
            retrieved_content,
            session_id,
        )
        return questions or []
    except Exception as error:
        logger.error(f"生成推荐问题失败: {str(error)}")
        return []


def get_chat_completion(
    session_id,
    question,
    retrieved_content=None,
    user_id: int | None = None,
):
    """
    流式生成聊天回答，并把结果包装成 SSE 事件交给前端。

    这个函数中使用了 yield，所以它不是一次性返回最终答案的普通函数，
    而是一个“生成器”。每执行一次 yield，StreamingResponse 就能立即把
    当前这小段内容发给前端，随后再继续执行函数的剩余部分。

    :param session_id: 当前会话 ID
    :param question: 用户提出的问题
    :param retrieved_content: 兼容旧调用的预检索文档；None 时由 Agent 决定是否检索
    :param user_id: 当前用户 ID
    :return: 逐个产生 SSE 格式字符串的生成器
    """
    request_started_at = time.perf_counter()
    try:
        if user_id is None:
            raise ValueError("运行 Agent 必须提供 user_id。")

        # 第一步：读取历史与会话文档，让 Agent 结合上下文决定工具调用。
        conversation_history = load_conversation_history(session_id, user_id)
        quick_parse_content = get_quick_parse_content(session_id)

        # AgentOrchestrator 统一完成小模型规划、工具执行、证据归一化，
        # 并启动大模型的最终回答流。
        orchestrator = _create_agent_orchestrator(user_id)
        execution = orchestrator.run(
            session_id=session_id,
            question=question,
            session_context=quick_parse_content,
            conversation_history=conversation_history,
            retrieved_content=retrieved_content,
        )
        retrieved_content = execution.retrieved_content
        all_documents = execution.response_documents
        completion = execution.answer_stream

        # 第二步：先把引用资料发给前端，前端可以据此展示“答案来源”。
        yield _make_sse_event("message", {"documents": all_documents})

        # 用列表暂存每一小段文本，最后再 join 成完整字符串。
        # 这比每次使用 answer = answer + 新文字更适合流式累加。
        answer_parts = []
        thinking_parts = []
        answer_stream_started_at = time.perf_counter()
        first_output_logged = False
        stream_chunk_count = 0

        # 第三步：不断读取回答模型的小块，并立即转发给前端。
        for chunk in completion:
            stream_chunk_count += 1
            finish_reason, answer_text, thinking_text = _read_stream_chunk(chunk)

            # finish_reason 不是 None，表示模型已经停止生成。
            if finish_reason is not None:
                break

            if not first_output_logged and (answer_text or thinking_text):
                first_output_logged = True
                logger.info(
                    "PERF session_id=%s stage=answer_first_output "
                    "answer_wait_ms=%.1f request_elapsed_ms=%.1f",
                    session_id,
                    (time.perf_counter() - answer_stream_started_at) * 1000,
                    (time.perf_counter() - request_started_at) * 1000,
                )

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

        # 第四步：流式回答结束后，拼出完整正文和完整思考过程。
        model_answer = "".join(answer_parts)
        think = "".join(thinking_parts)
        logger.info(
            "PERF session_id=%s stage=answer_stream status=%s chunks=%s "
            "answer_chars=%s thinking_chars=%s duration_ms=%.1f",
            session_id,
            "ok" if first_output_logged else "empty",
            stream_chunk_count,
            len(model_answer),
            len(think),
            (time.perf_counter() - answer_stream_started_at) * 1000,
        )

        recommended_started_at = time.perf_counter()
        recommended_questions = _generate_recommended_questions_safely(
            question,
            retrieved_content,
            session_id,
        )
        logger.info(
            "PERF session_id=%s stage=recommended_questions count=%s "
            "duration_ms=%.1f",
            session_id,
            len(recommended_questions),
            (time.perf_counter() - recommended_started_at) * 1000,
        )
        if recommended_questions:
            yield _make_sse_event(
                "message",
                {"recommended_questions": recommended_questions},
            )

        # 告诉前端：本次流式回答已经全部发送完毕。
        yield _make_sse_event("end", "[DONE]")

        # 最后保存完整聊天记录，并在首次聊天时生成会话名称。
        write_chat_to_db(
            session_id,
            question,
            model_answer,
            all_documents,
            recommended_questions,
            think,
        )
        update_session_name(session_id, question, user_id)
        logger.info(
            "PERF session_id=%s stage=request_total status=ok duration_ms=%.1f",
            session_id,
            (time.perf_counter() - request_started_at) * 1000,
        )

    except Exception as error:
        # 即使后端出错，也按 SSE 格式告诉前端，而不是直接中断连接。
        logger.error(f"流式聊天处理失败: {str(error)}")
        logger.info(
            "PERF session_id=%s stage=request_total status=error "
            "error_type=%s duration_ms=%.1f",
            session_id,
            type(error).__name__,
            (time.perf_counter() - request_started_at) * 1000,
        )
        yield _make_sse_event(
            "error",
            {
                "role": "error",
                "content": str(error),
            },
        )
