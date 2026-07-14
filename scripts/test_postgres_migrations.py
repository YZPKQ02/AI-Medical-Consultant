from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import uuid

import psycopg


ROOT = Path(__file__).resolve().parent.parent
ADMIN_URL = os.getenv("MIGRATION_TEST_ADMIN_URL", "").strip()


def run_alembic(database_url: str, *arguments: str) -> None:
    env = {**os.environ, "DATABASE_URL": database_url}
    subprocess.run(
        [sys.executable, "-m", "alembic", *arguments],
        cwd=ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    if not ADMIN_URL:
        raise SystemExit("MIGRATION_TEST_ADMIN_URL is required")

    database_name = "migration_test_" + uuid.uuid4().hex[:12]
    with psycopg.connect(ADMIN_URL, autocommit=True) as connection:
        connection.execute(f'CREATE DATABASE "{database_name}"')

    test_url = ADMIN_URL.rsplit("/", 1)[0] + "/" + database_name
    try:
        run_alembic(test_url, "upgrade", "head")
        with psycopg.connect(test_url) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                )
            }
            expected = {"users", "consultations", "messages", "agent_runs", "tool_calls", "alembic_version"}
            if not expected.issubset(tables):
                raise AssertionError(f"Missing tables: {sorted(expected - tables)}")

        run_alembic(test_url, "downgrade", "base")
        run_alembic(test_url, "upgrade", "head")
        print("PostgreSQL migration upgrade/downgrade/empty initialization passed")
    finally:
        with psycopg.connect(ADMIN_URL, autocommit=True) as connection:
            connection.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s",
                (database_name,),
            )
            connection.execute(f'DROP DATABASE IF EXISTS "{database_name}"')


if __name__ == "__main__":
    main()
