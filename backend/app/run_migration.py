#!/usr/bin/env python3
"""Run the application's Alembic migrations."""

import os
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from dotenv import load_dotenv


load_dotenv()

APP_DIRECTORY = Path(__file__).resolve().parent


def get_alembic_config() -> Config:
    return Config(str(APP_DIRECTORY / "alembic.ini"))


def run_migration() -> int:
    """Upgrade the configured database to the latest revision."""
    if not os.getenv("DATABASE_URL"):
        print("错误: 未找到 DATABASE_URL 环境变量")
        return 1

    alembic_config = get_alembic_config()

    try:
        print("检查当前数据库版本...")
        command.current(alembic_config, verbose=True)

        print("开始运行数据库迁移...")
        command.upgrade(alembic_config, "head")

        print("数据库迁移完成!")
        return 0
    except Exception as error:
        print(f"迁移失败: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(run_migration())
