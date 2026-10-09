-- PostgreSQL 仅在首次初始化 Docker 数据目录时执行此文件。
-- 应用表由 Alembic 管理；此初始化脚本仅启用
-- 消息表默认 UUID 所需的扩展。
CREATE EXTENSION IF NOT EXISTS pgcrypto;
