#!/bin/bash
set -euo pipefail

echo "=== 应用启动脚本 ==="

echo "等待数据库连接..."
python -c "
import os
import time

import psycopg2
from psycopg2 import OperationalError

max_retries = 30

for retry_count in range(1, max_retries + 1):
    try:
        connection = psycopg2.connect(os.environ['DATABASE_URL'])
        connection.close()
        print('数据库连接成功!')
        break
    except OperationalError:
        print(f'等待数据库... ({retry_count}/{max_retries})')
        time.sleep(2)
else:
    raise SystemExit('数据库连接失败!')
"

# Alembic owns the complete schema. A fresh database is upgraded from the
# initial revision; an existing database advances from its recorded revision.
# A migration failure stops startup instead of serving against a stale schema.
echo "运行数据库迁移..."
python run_migration.py

echo "启动应用服务..."
exec "$@"
