from fastapi import APIRouter, Body, UploadFile, File, HTTPException, Query, Security, status, Depends
import uuid
from schemas.chat import SessionResponse, ChatRequest
from fastapi.responses import StreamingResponse
import os
from dotenv import load_dotenv
from typing import List, Optional
from service.core.file_parse import execute_insert_process
from service.core.api.utils.file_utils import get_project_base_directory
from fastapi_jwt import JwtAuthorizationCredentials
from service.core.retrieval import retrieve_content
from service.core.chat import get_chat_completion
from service.auth import access_security
from utils import logger
from database.knowledgebase_operations import insert_knowledgebase, verify_user_knowledgebase
from sqlalchemy.orm import Session
from sqlalchemy import select
from models.message import KnowledgeBase
from utils.database import get_db
from service.quick_parse_service import quick_parse_service
from service.document_upload_service import DocumentUploadService
from schemas.document_upload import DocumentUploadResponse, SessionDocumentsResponse, SessionDocumentSummary

# 加载 .env 文件
load_dotenv()

# 配置日志
logger.info(f"ES_HOST: {os.getenv('ES_HOST')}")
logger.info(f"ELASTICSEARCH_URL: {os.getenv('ELASTICSEARCH_URL')}")

router = APIRouter()

##################################
# 创建一个新的对话 Session
##################################

@router.post("/create_session", response_model=SessionResponse)
async def create_session(
    credentials: JwtAuthorizationCredentials = Security(access_security),
):
    try:
        user_id = credentials.subject.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

        # 生成会话Id, 这里会生成一个随机字符串，用于区分不同聊天会话
        session_id = str(uuid.uuid4()).replace("-", "")[:16]

        # 这里没有立即把会话写入数据库，通常要等用户第一次真正提问时，chat.py才会创建会话记录
        return {
            "session_id": session_id,
            "status": "success",
            "message": "Session created successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

##################################
# 快速文档解析接口
##################################

@router.post("/quick_parse")
async def quick_parse_document(
    session_id: str = Query(..., description="会话ID"),
    file: UploadFile = File(..., description="要解析的文档"),
    credentials: JwtAuthorizationCredentials = Security(access_security),
    db: Session = Depends(get_db),
):
    """
    快速文档解析接口
    - 支持文档格式：docx, pdf, txt
    - 限制文档页数不超过4页
    - 每个session_id只能传递一个文档
    - 解析结果存储到Redis，保存时间为2小时
    """
    try:
        user_id = str(credentials.subject.get("user_id"))
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

        # 读取文件内容
        file_content = await file.read()
        
        # 获取文件信息
        file_size = len(file_content)
        file_extension = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        document_type = file_extension.replace(".", "") if file_extension else "unknown"
        
        # 调用服务层处理业务逻辑
        result = quick_parse_service.quick_parse_document(
            session_id=session_id,
            filename=file.filename,
            file_content=file_content
        )
        
        # 记录文档上传信息到数据库
        try:
            DocumentUploadService.create_upload_record(
                db=db,
                session_id=session_id,
                document_name=file.filename,
                document_type=document_type,
                file_size=file_size
            )
            logger.info(f"文档上传记录已保存: session_id={session_id}, document_name={file.filename}")
        except Exception as db_error:
            logger.error(f"保存文档上传记录失败: {str(db_error)}")
            # 数据库记录失败不影响主要功能，继续返回解析结果
        
        logger.info(f"用户 {user_id} 的文档解析完成，session_id: {session_id}")
        return result

    except HTTPException as e:
        logger.error(f"快速解析错误: {str(e)}")
        raise e
    except Exception as e:
        logger.exception(f"快速解析发生未知错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"内部服务器错误: {str(e)}"
        )

##################################
# 获取解析内容接口
##################################

@router.get("/get_parsed_content")
async def get_parsed_content(
    session_id: str = Query(..., description="会话ID"),
    credentials: JwtAuthorizationCredentials = Security(access_security),
):
    """
    获取已解析的文档内容
    """
    try:
        user_id = str(credentials.subject.get("user_id"))
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

        # 调用服务层获取内容
        result = quick_parse_service.get_parsed_content(session_id)
        
        logger.info(f"用户 {user_id} 获取解析内容，session_id: {session_id}")
        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

##################################
# 基于ragflow知识库对话
##################################

@router.post("/chat_on_docs")
async def chat_on_docs(
    session_id: str = Query(...),
    request: ChatRequest = Body(..., description="User message"),
    credentials: JwtAuthorizationCredentials = Security(access_security),
):
    try:
        user_id = str(credentials.subject.get("user_id"))
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        
        logger.info(f"开始处理用户 {user_id} 的请求")
        logger.info(f"问题内容: {request.message}")
        
        question = request.message
        
        # 尝试从知识库检索内容，如果没有知识库也不报错
        references = []
        try:
            logger.info("开始检索相关内容...")
            references = retrieve_content(user_id, question)
            logger.info(f"检索到 {len(references)} 条相关内容")
        except Exception as e:
            logger.info(f"用户 {user_id} 没有知识库或检索失败: {str(e)}，将不使用知识库内容")
            references = []

        logger.info("开始生成回答...")
        # 返回流式响应
        return StreamingResponse(
            get_chat_completion(session_id, question, references, user_id),
            media_type="text/event-stream"
        )
    
    except HTTPException as e:
        logger.error(f"HTTP错误: {str(e)}")
        raise e
    except Exception as e:
        logger.exception(f"发生未知错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# 上传知识库文件
XLSX_FILE_HEADER = b"PK"
XLS_FILE_HEADERS = (b"\xd0\xcf\x11\xe0", b"\x09\x08")


def _get_safe_upload_file_name(upload_file: UploadFile) -> str:
    """取得安全的文件名，防止文件被保存到会话目录之外。"""
    original_file_name = upload_file.filename
    if not original_file_name:
        raise HTTPException(status_code=400, detail="上传文件缺少文件名")

    # 文件名由客户端传入，理论上可能包含 ../ 或文件夹路径。
    # basename 只保留最后一段，例如 a/b.pdf 会得到 b.pdf。
    normalized_file_name = original_file_name.replace("\\", "/")
    safe_file_name = os.path.basename(normalized_file_name)
    if safe_file_name != normalized_file_name or safe_file_name in {"", ".", ".."}:
        raise HTTPException(
            status_code=400,
            detail=f"文件名不合法: {original_file_name}",
        )

    return safe_file_name


def _prepare_upload_directory(session_id: str) -> str:
    """创建当前会话的上传目录，并返回它的绝对路径。"""
    storage_dir = os.path.abspath(
        os.path.join(get_project_base_directory(), "storage/file")
    )
    session_dir = os.path.abspath(os.path.join(storage_dir, session_id))

    # commonpath 用来确认 session_dir 仍在 storage_dir 里面。
    # 这样恶意的 session_id（例如 ../../）就不能跳出上传目录。
    if os.path.commonpath([storage_dir, session_dir]) != storage_dir:
        raise HTTPException(status_code=400, detail="session_id 不合法")

    # exist_ok=True 表示目录已经存在时不报错。
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def _find_duplicate_file_names(
    db: Session,
    user_id: str,
    file_names: List[str],
) -> List[str]:
    """找出数据库中已有、或本次请求中重复出现的文件名。"""
    if not file_names:
        return []

    # 构造查询要求：从knowledgebases表中的file_name字段中查询当前用户上传的文件
    stmt = select(KnowledgeBase.file_name).where(
        KnowledgeBase.user_id == user_id,
        KnowledgeBase.file_name.in_(file_names),
    )
    #取出重复的文件的文件名
    existing_file_names = set(db.execute(stmt).scalars().all())

    # 检查本次请求中是否有重复的文件名
    duplicate_file_names = []
    seen_file_names = set()
    for file_name in file_names:
        already_seen = file_name in seen_file_names
        already_in_database = file_name in existing_file_names
        is_new_duplicate = file_name not in duplicate_file_names
        if (already_seen or already_in_database) and is_new_duplicate:
            duplicate_file_names.append(file_name)
        seen_file_names.add(file_name)

    # 返回重复的文件名列表（包括数据库中已有的和本次请求中重复的）
    return duplicate_file_names


def _validate_upload_content(file_name: str, file_content: bytes) -> Optional[str]:
    """检查文件内容；合法时返回 None，不合法时返回通俗的错误原因。"""
    if not file_content:
        return "文件内容为空"

    lower_file_name = file_name.lower()
    if lower_file_name.endswith(".xlsx"):
        # XLSX 本质上是 ZIP 压缩包，所以正常文件以 PK 开头。
        if not file_content.startswith(XLSX_FILE_HEADER):
            return "不是有效的 XLSX 文件格式，可能是 XLS 文件"

    if lower_file_name.endswith(".xls"):
        # 老版本 XLS 是二进制格式，通常以这两种文件头之一开头。
        if not file_content.startswith(XLS_FILE_HEADERS):
            return "不是有效的 XLS 文件格式"

    return None


def _save_upload_file(file_path: str, file_content: bytes) -> None:
    """把文件写入本地，并确认写入后的字节数没有变化。"""
    with open(file_path, "wb") as saved_file:
        saved_file.write(file_content)

    if os.path.getsize(file_path) != len(file_content):
        raise OSError("文件保存失败，大小不匹配")


def _remove_file_if_exists(file_path: str) -> None:
    """清理失败任务留下的本地文件；清理失败只记录日志。"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError as error:
        logger.error(f"清理上传文件失败 {file_path}: {str(error)}")


async def _process_single_upload(
    upload_file: UploadFile,
    file_name: str,
    session_dir: str,
    session_id: str,
    user_id: str,
) -> Optional[str]:
    """
    处理一个上传文件。

    成功时返回 None；失败时返回一段错误文字。调用者只需要根据返回值，
    把文件放入 successful_files 或 failed_files，不必了解内部细节。
    """
    file_path = os.path.join(session_dir, file_name)

    # UploadFile.read() 是异步方法，所以需要使用 await 等待文件读取完成。
    try:
        file_content = await upload_file.read()
    except Exception as error:
        logger.error(f"读取文件失败 {file_name}: {str(error)}")
        return f"{file_name}: 文件读取失败 - {str(error)}"

    # 验证文件内容是否合法，例如检查 XLSX 文件头，或检查文件是否为空。
    validation_error = _validate_upload_content(file_name, file_content)
    if validation_error:
        return f"{file_name}: {validation_error}"

    try:
        _save_upload_file(file_path, file_content)
    except Exception as error:
        logger.error(f"保存文件失败 {file_name}: {str(error)}")
        _remove_file_if_exists(file_path)
        return f"{file_name}: 文件保存失败 - {str(error)}"

    try:
        # execute_insert_process 会解析文档、生成向量，并把 chunk 写入 ES。
        logger.info(f"开始解析上传文件: {file_path}")
        execute_insert_process(file_path, file_name, session_id)
        logger.info(f"数据插入 ES 成功: {file_name}")

        # ES 保存文档内容；PostgreSQL 的 knowledgebases 表登记文件名和用户。
        insert_knowledgebase(user_id, file_name)
        logger.info(f"数据插入 PostgreSQL 成功: {file_name}")
        return None

    except Exception as error:
        logger.error(f"文件解析失败 {file_name}: {str(error)}")
        _remove_file_if_exists(file_path)
        return f"{file_name}: 文件解析失败 - {str(error)}"


def _build_upload_response(
    successful_files: List[str],
    failed_files: List[str],
    total_files: int,
) -> dict:
    """根据成功和失败的数量，构造统一的接口返回结果。"""
    if successful_files and not failed_files:
        return {
            "status": "success",
            "message": "所有文件解析成功",
            "successful_files": successful_files,
            "total_files": total_files,
        }

    if successful_files and failed_files:
        return {
            "status": "partial_success",
            "message": (
                f"部分文件解析成功，{len(successful_files)} 个成功，"
                f"{len(failed_files)} 个失败"
            ),
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_files": total_files,
        }

    raise HTTPException(
        status_code=400,
        detail={
            "status": "failed",
            "message": "所有文件解析失败",
            "failed_files": failed_files,
            "total_files": total_files,
        },
    )

##################################
# 上传文件并建立知识库
##################################

@router.post("/upload_files")
async def upload_files(
    session_id: Optional[str] = Query(None),
    files: List[UploadFile] = File(...),
    credentials: JwtAuthorizationCredentials = Security(access_security),
    db: Session = Depends(get_db),
):
    """
    上传文件并建立知识库。

    这个路由函数现在只负责安排执行顺序：
    1. 验证用户和文件名；2. 检查重名；3. 逐个处理文件；4. 汇总结果。
    具体的验证、保存和解析逻辑交给上方的小函数完成。
    """
    try:
        # 必须先判断原值，再转换成字符串。
        # 否则 str(None) 会得到字符串 "None"，它在 if 判断中反而算 True。
        raw_user_id = credentials.subject.get("user_id")
        if not raw_user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        user_id = str(raw_user_id)

        # 没有传 session_id 时，会使用 user_id 作为知识库索引名称。
        effective_session_id = session_id or user_id
        file_names = [_get_safe_upload_file_name(file) for file in files]

        # 检查数据库中是否已有同名文件，或本次请求中是否有重复文件名。
        duplicate_file_names = _find_duplicate_file_names(db, user_id, file_names)
        if duplicate_file_names:
            raise HTTPException(
                status_code=400,
                detail=(
                    "以下文件已存在，请勿重复上传: "
                    + ", ".join(duplicate_file_names)
                ),
            )

        session_dir = _prepare_upload_directory(effective_session_id)
        successful_files = []
        failed_files = []

        # zip 会把两个列表中同一位置的元素配成一对：
        # (UploadFile 对象, 已验证过的安全文件名)。
        for upload_file, file_name in zip(files, file_names):
            error_message = await _process_single_upload(
                upload_file=upload_file,
                file_name=file_name,
                session_dir=session_dir,
                session_id=effective_session_id,
                user_id=user_id,
            )
            if error_message:
                failed_files.append(error_message)
            else:
                successful_files.append(file_name)

        return _build_upload_response(
            successful_files=successful_files,
            failed_files=failed_files,
            total_files=len(files),
        )

    except HTTPException:
        # HTTPException 是有意返回给前端的 400/401 等错误，保持原状态码。
        raise
    except Exception as error:
        logger.exception(f"上传文件发生未知错误: {str(error)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        )

##################################
# 查询会话文档上传信息接口
##################################

@router.get("/sessions/{session_id}/documents", response_model=SessionDocumentsResponse)
async def get_session_documents(
    session_id: str,
    credentials: JwtAuthorizationCredentials = Security(access_security),
    db: Session = Depends(get_db),
):
    """
    获取指定会话的所有文档上传记录
    """
    try:
        user_id = str(credentials.subject.get("user_id"))
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

        # 获取会话的所有文档记录
        documents = DocumentUploadService.get_session_documents(db, session_id)
        has_documents = len(documents) > 0
        
        return SessionDocumentsResponse(
            session_id=session_id,
            has_documents=has_documents,
            documents=[DocumentUploadResponse.from_orm(doc) for doc in documents],
            total_count=len(documents)
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"获取会话文档信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/sessions/{session_id}/documents/summary", response_model=SessionDocumentSummary)
async def get_session_document_summary(
    session_id: str,
    credentials: JwtAuthorizationCredentials = Security(access_security),
    db: Session = Depends(get_db),
):
    """
    获取指定会话的文档上传摘要信息
    """
    try:
        user_id = str(credentials.subject.get("user_id"))
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

        # 检查是否有上传的文档
        has_documents = DocumentUploadService.has_uploaded_documents(db, session_id)
        
        # 获取最新的文档信息
        latest_document = DocumentUploadService.get_latest_document(db, session_id)
        
        # 获取总文档数量
        all_documents = DocumentUploadService.get_session_documents(db, session_id)
        total_documents = len(all_documents)
        
        return SessionDocumentSummary(
            session_id=session_id,
            has_documents=has_documents,
            latest_document_name=latest_document.document_name if latest_document else None,
            latest_document_type=latest_document.document_type if latest_document else None,
            latest_upload_time=latest_document.upload_time if latest_document else None,
            total_documents=total_documents
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"获取会话文档摘要失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
