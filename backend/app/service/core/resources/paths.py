"""集中定位源码资源；保留历史数据根路径，业务模块移动不会改变存储位置。"""

import os
from pathlib import Path

PROJECT_BASE = os.getenv("RAG_PROJECT_BASE") or os.getenv("RAG_DEPLOY_BASE")


def core_directory() -> Path:
    return Path(__file__).resolve().parents[1]


def config_file(name: str) -> Path:
    """内置配置跟随源码，不跟随查询、实体等调用方文件的位置。"""
    return core_directory() / "conf" / name


def get_project_base_directory(*args):
    global PROJECT_BASE
    if PROJECT_BASE is None:
        PROJECT_BASE = str(core_directory())

    if args:
        return os.path.join(PROJECT_BASE, *args)
    return PROJECT_BASE


def tokenizer_cache_directory() -> str:
    """兼容旧部署根目录中的词表缓存，新源码把内置词表放到 rag/res 下。"""
    base = Path(get_project_base_directory())
    cache_name = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
    if (base / cache_name).is_file():
        return str(base)
    packaged = base / "rag" / "res" / "tiktoken"
    if (packaged / cache_name).is_file():
        return str(packaged)
    return str(core_directory() / "rag" / "res" / "tiktoken")
