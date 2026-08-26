-- PostgreSQL executes this file only when the Docker data directory is first
-- initialized. Application tables are owned by Alembic; this bootstrap file
-- only enables the extension required by the messages UUID default.
CREATE EXTENSION IF NOT EXISTS pgcrypto;
