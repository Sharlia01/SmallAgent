#!/usr/bin/env python3
"""Interactive terminal client for the AgentRAG FastAPI backend.

This module deliberately talks to the public HTTP API instead of importing the
RAG implementation.  The web UI and this CLI therefore share authentication,
session history, retrieval, citation, and refusal behaviour.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import getpass
import json
import mimetypes
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Iterable, Iterator, TextIO

import httpx


DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 300.0


class CliError(RuntimeError):
    """A user-facing API or command error."""


@dataclass(frozen=True)
class SseEvent:
    """One decoded server-sent event."""

    event: str
    data: str


def iter_sse_events(lines: Iterable[str | bytes]) -> Iterator[SseEvent]:
    """Decode an SSE line stream, including multiline ``data`` fields."""
    event_name = "message"
    data_lines: list[str] = []

    def build_event() -> SseEvent | None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = "message"
            return None
        event = SseEvent(event=event_name, data="\n".join(data_lines))
        event_name = "message"
        data_lines = []
        return event

    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.rstrip("\r\n")
        if not line:
            event = build_event()
            if event is not None:
                yield event
            continue
        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value or "message"
        elif field == "data":
            data_lines.append(value)

    event = build_event()
    if event is not None:
        yield event


def normalize_terminal_text(value: str) -> str:
    """Recover UTF-8 bytes represented as surrogates by a terminal decoder."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        source_encoding = getattr(sys.stdin, "encoding", None) or "utf-8"
        try:
            original_bytes = value.encode(
                source_encoding,
                errors="surrogateescape",
            )
        except UnicodeEncodeError:
            return "".join(
                "\ufffd" if 0xD800 <= ord(character) <= 0xDFFF else character
                for character in value
            )
        return original_bytes.decode("utf-8", errors="replace")
    return value


class RagApiClient:
    """Small typed wrapper around the FastAPI endpoints used by the CLI."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token: str | None = None
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            headers={"User-Agent": "AgentRAG-CLI/1.0"},
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "RagApiClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        if not self.token:
            raise CliError("尚未登录。")
        return {"Authorization": f"Bearer {self.token}"}

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return response.text.strip() or response.reason_phrase

        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        if isinstance(detail, str):
            return detail
        return json.dumps(detail, ensure_ascii=False)

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            response.read()
        except httpx.StreamConsumed:
            pass
        detail = self._error_detail(response)
        raise CliError(f"API 请求失败（HTTP {response.status_code}）：{detail}")

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.RequestError as error:
            raise CliError(f"无法连接后端 {self.base_url}：{error}") from error
        self._raise_for_status(response)
        try:
            return response.json()
        except json.JSONDecodeError as error:
            raise CliError("后端返回了无法解析的 JSON。") from error

    def register(self, username: str, password: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/register",
            json={"username": username, "password": password},
        )

    def login(self, username: str, password: str) -> None:
        payload = self._request(
            "POST",
            "/login",
            json={"username": username, "password": password},
        )
        token = payload.get("access_token") if isinstance(payload, dict) else None
        if not token:
            raise CliError("登录响应中缺少 access_token。")
        self.token = str(token)

    def create_session(self) -> str:
        payload = self._request(
            "POST",
            "/create_session",
            headers=self._headers(),
        )
        session_id = payload.get("session_id") if isinstance(payload, dict) else None
        if not session_id:
            raise CliError("创建会话的响应中缺少 session_id。")
        return str(session_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/get_sessions", headers=self._headers())
        sessions = payload.get("sessions", []) if isinstance(payload, dict) else []
        return sessions if isinstance(sessions, list) else []

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            "/get_messages",
            params={"session_id": session_id},
            headers=self._headers(),
        )
        return payload if isinstance(payload, list) else []

    def list_files(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/get_files", headers=self._headers())
        return payload if isinstance(payload, list) else []

    @staticmethod
    def _validate_files(paths: Iterable[str]) -> list[Path]:
        files: list[Path] = []
        for value in paths:
            path = Path(value).expanduser()
            if not path.exists():
                raise CliError(f"文件不存在：{path}")
            if not path.is_file():
                raise CliError(f"不是普通文件：{path}")
            files.append(path)
        if not files:
            raise CliError("请至少指定一个文件。")
        return files

    def upload_files(self, paths: Iterable[str]) -> dict[str, Any]:
        selected = self._validate_files(paths)
        try:
            with ExitStack() as stack:
                files = []
                for path in selected:
                    handle = stack.enter_context(path.open("rb"))
                    media_type = mimetypes.guess_type(path.name)[0]
                    files.append(
                        (
                            "files",
                            (
                                path.name,
                                handle,
                                media_type or "application/octet-stream",
                            ),
                        )
                    )
                return self._request(
                    "POST",
                    "/upload_files",
                    files=files,
                    headers=self._headers(),
                )
        except OSError as error:
            raise CliError(f"读取上传文件失败：{error}") from error

    def quick_parse(self, session_id: str, path_value: str) -> dict[str, Any]:
        path = self._validate_files([path_value])[0]
        media_type = mimetypes.guess_type(path.name)[0]
        try:
            with path.open("rb") as handle:
                return self._request(
                    "POST",
                    "/quick_parse",
                    params={"session_id": session_id},
                    files={
                        "file": (
                            path.name,
                            handle,
                            media_type or "application/octet-stream",
                        )
                    },
                    headers=self._headers(),
                )
        except OSError as error:
            raise CliError(f"读取会话文档失败：{error}") from error

    def stream_chat(
        self,
        session_id: str,
        question: str,
    ) -> Iterator[dict[str, Any]]:
        safe_question = normalize_terminal_text(question)
        try:
            stream = self._client.stream(
                "POST",
                "/chat_on_docs",
                params={
                    "session_id": session_id,
                    "include_recommended_questions": "false",
                },
                json={"message": safe_question},
                headers={**self._headers(), "Accept": "text/event-stream"},
            )
            with stream as response:
                self._raise_for_status(response)
                done_received = False
                for event in iter_sse_events(response.iter_lines()):
                    if event.data == "[DONE]":
                        # Keep draining the HTTP response.  The server persists
                        # the completed turn after emitting [DONE], so closing
                        # the socket here could interrupt that final work.
                        done_received = True
                        continue
                    if done_received:
                        continue
                    try:
                        payload = json.loads(event.data)
                    except json.JSONDecodeError as error:
                        raise CliError(
                            f"收到无法解析的 SSE 数据：{event.data[:120]}"
                        ) from error
                    if not isinstance(payload, dict):
                        raise CliError("后端 SSE 消息不是 JSON 对象。")
                    if event.event == "error" or payload.get("role") == "error":
                        raise CliError(str(payload.get("content") or "后端流式回答失败。"))
                    yield payload
        except httpx.RequestError as error:
            raise CliError(f"与后端的流式连接中断：{error}") from error
        except (TypeError, ValueError, UnicodeError) as error:
            raise CliError(
                f"无法构造流式请求（{type(error).__name__}）：{error}"
            ) from error


HELP_TEXT = """可用命令：
  /help                   显示帮助
  /new                    创建并切换到新会话
  /sessions               列出当前账号的会话
  /use <session_id>       切换会话
  /history                查看当前会话历史
  /upload <文件...>       上传文件到个人知识库（含空格时用引号）
  /quick <文件>           上传当前会话临时文档
  /files                  列出知识库文件
  /sources                再次显示上一轮引用来源
  /thinking on|off        开启或关闭思考过程显示
  /session                显示当前会话 ID
  /exit 或 /quit          退出

直接输入文字即可向当前会话提问。"""


def _page_label(document: dict[str, Any]) -> str:
    pages = document.get("page_num_int")
    if isinstance(pages, list) and pages:
        return ",".join(str(page) for page in pages)
    if pages not in (None, "", []):
        return str(pages)
    return ""


def format_source(document: dict[str, Any], number: int) -> str:
    """Format one response document without dumping its full chunk text."""
    source_type = str(document.get("source_type") or "knowledge_base")
    name = str(
        document.get("document_name")
        or document.get("title")
        or document.get("url")
        or "未知来源"
    )
    details = [source_type]
    pages = _page_label(document)
    if pages:
        details.append(f"页码 {pages}")
    chunk_id = document.get("chunk_id")
    if chunk_id:
        details.append(f"chunk {chunk_id}")
    url = document.get("url")
    suffix = f"\n      {url}" if url else ""
    return f"  [{number}] {name}（{'，'.join(details)}）{suffix}"


class InteractiveCli:
    """Terminal command loop and streaming renderer."""

    def __init__(
        self,
        api: RagApiClient,
        *,
        session_id: str | None = None,
        show_thinking: bool = False,
        output: TextIO = sys.stdout,
    ) -> None:
        self.api = api
        self.session_id = session_id
        self.show_thinking = show_thinking
        self.output = output
        self.last_sources: list[dict[str, Any]] = []

    def _print(self, value: str = "", *, end: str = "\n") -> None:
        print(value, end=end, file=self.output, flush=True)

    def ensure_session(self) -> str:
        if not self.session_id:
            self.session_id = self.api.create_session()
            self._print(f"已创建会话：{self.session_id}")
        return self.session_id

    def show_sources(self) -> None:
        if not self.last_sources:
            self._print("本轮没有返回引用来源。")
            return
        self._print("\n引用来源：")
        for number, document in enumerate(self.last_sources, start=1):
            self._print(format_source(document, number))

    def ask(self, question: str) -> None:
        if not question.strip():
            return
        session_id = self.ensure_session()
        self.last_sources = []
        recommendations: list[str] = []
        response_mode: str | None = None
        answer_started = False
        thinking_started = False

        self._print("\n助手：", end="")
        for payload in self.api.stream_chat(session_id, question.strip()):
            documents = payload.get("documents")
            if isinstance(documents, list):
                self.last_sources = [item for item in documents if isinstance(item, dict)]

            suggested = payload.get("recommended_questions")
            if isinstance(suggested, list):
                recommendations = [str(item) for item in suggested]

            content = payload.get("content")
            if not isinstance(content, str) or not content:
                continue
            if payload.get("response_mode"):
                response_mode = str(payload["response_mode"])

            if payload.get("thinking"):
                if self.show_thinking:
                    if not thinking_started:
                        self._print("\n[思考] ", end="")
                        thinking_started = True
                    self._print(content, end="")
                continue

            if thinking_started and not answer_started:
                self._print("\n[回答] ", end="")
            self._print(content, end="")
            answer_started = True

        if not answer_started:
            self._print("（后端未返回回答文本）", end="")
        self._print()
        if response_mode == "refuse":
            self._print("回答模式：拒答（refuse）")
        self.show_sources()
        if recommendations:
            self._print("\n推荐追问：")
            for index, item in enumerate(recommendations, start=1):
                self._print(f"  {index}. {item}")
        self._print()

    def _show_sessions(self) -> None:
        sessions = self.api.list_sessions()
        if not sessions:
            self._print("暂无会话。")
            return
        for session in sessions:
            session_id = str(session.get("session_id") or "")
            marker = "*" if session_id == self.session_id else " "
            name = session.get("session_name") or "未命名会话"
            updated = session.get("updated_at") or ""
            self._print(f"{marker} {session_id}  {name}  {updated}")

    def _show_history(self) -> None:
        messages = self.api.get_history(self.ensure_session())
        if not messages:
            self._print("当前会话还没有消息。")
            return
        for index, message in enumerate(messages, start=1):
            self._print(f"\n[{index}] 你：{message.get('user_question') or ''}")
            self._print(f"    助手：{message.get('model_answer') or ''}")

    def _show_files(self) -> None:
        files = self.api.list_files()
        if not files:
            self._print("知识库中还没有文件。")
            return
        for item in files:
            self._print(
                f"- {item.get('file_name') or '未知文件'}"
                f"  更新于 {item.get('updated_at') or ''}"
            )

    def handle_command(self, line: str) -> bool:
        """Handle one slash command. Return ``False`` to leave the loop."""
        try:
            parts = shlex.split(line)
        except ValueError as error:
            raise CliError(f"命令格式错误：{error}") from error
        if not parts:
            return True

        command, *arguments = parts
        if command in {"/exit", "/quit"}:
            return False
        if command in {"/help", "/?"}:
            self._print(HELP_TEXT)
        elif command == "/new":
            self.session_id = self.api.create_session()
            self.last_sources = []
            self._print(f"已切换到新会话：{self.session_id}")
        elif command == "/sessions":
            self._show_sessions()
        elif command == "/use":
            if len(arguments) != 1:
                raise CliError("用法：/use <session_id>")
            owned_ids = {
                str(item.get("session_id")) for item in self.api.list_sessions()
            }
            if arguments[0] not in owned_ids:
                raise CliError("该会话不存在，或不属于当前账号。")
            self.session_id = arguments[0]
            self.last_sources = []
            self._print(f"已切换会话：{self.session_id}")
        elif command == "/history":
            self._show_history()
        elif command == "/upload":
            if not arguments:
                raise CliError("用法：/upload <文件...>")
            result = self.api.upload_files(arguments)
            self._print(str(result.get("message") or "上传完成。"))
            for filename in result.get("successful_files", []):
                self._print(f"  成功：{filename}")
            for error in result.get("failed_files", []):
                self._print(f"  失败：{error}")
        elif command == "/quick":
            if len(arguments) != 1:
                raise CliError("用法：/quick <文件>")
            result = self.api.quick_parse(self.ensure_session(), arguments[0])
            self._print(str(result.get("message") or "会话文档解析完成。"))
        elif command == "/files":
            self._show_files()
        elif command == "/sources":
            self.show_sources()
        elif command == "/thinking":
            if len(arguments) != 1 or arguments[0].lower() not in {"on", "off"}:
                raise CliError("用法：/thinking on|off")
            self.show_thinking = arguments[0].lower() == "on"
            self._print(f"思考过程显示已{'开启' if self.show_thinking else '关闭'}。")
        elif command == "/session":
            self._print(f"当前会话：{self.ensure_session()}")
        else:
            raise CliError(f"未知命令：{command}。输入 /help 查看帮助。")
        return True

    def run(self) -> None:
        self.ensure_session()
        self._print("AgentRAG CLI 已就绪。输入 /help 查看命令。")
        while True:
            try:
                line = input("你：").strip()
                if not line:
                    continue
                if line.startswith("/"):
                    if not self.handle_command(line):
                        return
                else:
                    self.ask(line)
            except CliError as error:
                self._print(f"错误：{error}")
            except (EOFError, KeyboardInterrupt):
                self._print("\n已退出。")
                return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="通过终端使用 AgentRAG 后端（无需启动前端）。"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("RAG_API_URL", DEFAULT_API_URL),
        help="FastAPI 地址（默认：%(default)s，也可设置 RAG_API_URL）",
    )
    parser.add_argument("--username", help="登录用户名；省略时交互输入")
    parser.add_argument(
        "--register",
        action="store_true",
        help="先注册该用户名，再登录",
    )
    parser.add_argument("--session-id", help="使用已有会话；省略时自动创建")
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="显示模型返回的思考内容",
    )
    parser.add_argument(
        "--question",
        help="只提一个问题并退出；省略时进入交互模式",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="单次网络读写超时秒数（默认：%(default)s）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        print("错误：--timeout 必须大于 0。", file=sys.stderr)
        return 2

    try:
        username = args.username or input("用户名：").strip()
        if not username:
            print("错误：用户名不能为空。", file=sys.stderr)
            return 2
        password = os.getenv("RAG_CLI_PASSWORD") or getpass.getpass("密码：")
        if not password:
            print("错误：密码不能为空。", file=sys.stderr)
            return 2
    except (EOFError, KeyboardInterrupt):
        print("\n已退出。")
        return 130

    try:
        with RagApiClient(args.base_url, timeout=args.timeout) as api:
            if args.register:
                api.register(username, password)
                print("注册成功。", flush=True)
            api.login(username, password)
            print(f"登录成功：{username}", flush=True)

            cli = InteractiveCli(
                api,
                session_id=args.session_id,
                show_thinking=args.show_thinking,
            )
            if args.question:
                cli.ask(args.question)
            else:
                cli.run()
        return 0
    except CliError as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\n已退出。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
